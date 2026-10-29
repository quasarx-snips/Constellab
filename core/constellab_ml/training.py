"""Day 38: the reusable PyTorch training framework.

A run is fully described by a declarative configuration (``TrainingConfig``,
loaded from JSON with validation): model family and dimensions, optimizer and
learning-rate schedule, loss weights, batch sizes, step budget, seed, early
stopping, checkpoint selection metric and export options. Nothing in the loop
branches on the model name — models come from the registry in
:mod:`constellab_ml.models` — and normalization is fitted on the training split
only and stored with the checkpoint.

Every checkpoint records the resolved training config, the dataset campaign id,
the observation/feature/parameter schema hashes, the code revision (best effort)
and the library versions, so a checkpoint can always be traced back to the data
and the code that produced it. Runs are resumable: ``resume_from`` restores the
model, the optimizer and the metric history, and steps continue from there.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import nn

from . import ENV_VERSION, OBSERVATION_VERSION, feature_schema_payload
from .batch import GraphSample, StepBatch, build_batch, samples_from_dataset
from .dataset import EpisodeDataset
from .features import Normalizer
from .models import MODEL_FAMILIES, PolicyOutput, build_model, model_parameter_count
from .policies import ModelPolicy, LearnedPolicyMetadata, evaluate_policy
from .expert import DEFAULT_EXPERT, expert_factory

TRAINING_VERSION = "training-1"
CHECKPOINT_VERSION = "checkpoint-1"


class TrainingError(RuntimeError):
    """A training configuration or run is invalid."""


def code_revision(root: Path | None = None) -> str:
    """Best-effort git revision of the working tree (never fatal)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # pragma: no cover - no git, or git unavailable
        return "unknown"
    return "unknown"


def library_versions() -> dict[str, str]:
    import sys

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


@dataclass
class TrainingConfig:
    """Declarative training configuration (JSON-serialisable)."""

    run_id: str
    dataset_root: str
    model: dict[str, Any] = field(default_factory=lambda: {"model": "gnn", "hidden": 32, "layers": 2})
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    schedule: str = "cosine"  # cosine | constant
    warmup_steps: int = 0
    label_smoothing: float = 0.0
    abstain_weight: float = 1.0
    graphs_per_batch: int = 8
    steps_per_chunk: int = 4
    batches_per_epoch: int = 16
    epochs: int = 4
    max_steps: int = 0
    seed: int = 0
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    selection_metric: str = "validation_loss"
    evaluate_every: int = 0
    evaluation_episodes: int = 4
    max_sequence_samples: int = 4000
    use_abstain_class: bool = True
    export: bool = True
    curriculum: dict[str, Any] | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            raise TrainingError("run_id must be non-empty")
        family = str(self.model.get("model", "gnn"))
        if family not in MODEL_FAMILIES:
            raise TrainingError(f"unknown model family {family!r}; known: {list(MODEL_FAMILIES)}")
        if self.learning_rate <= 0.0:
            raise TrainingError("learning_rate must be positive")
        if self.schedule not in ("cosine", "constant"):
            raise TrainingError(f"unknown schedule {self.schedule!r}")
        if self.graphs_per_batch <= 0 or self.steps_per_chunk <= 0 or self.batches_per_epoch <= 0:
            raise TrainingError("batch and epoch sizes must be positive")
        if self.epochs <= 0 and self.max_steps <= 0:
            raise TrainingError("a run needs a positive epochs or max_steps budget")
        if self.abstain_weight < 0.0 or self.label_smoothing < 0.0:
            raise TrainingError("loss weights must be non-negative")

    @property
    def temporal(self) -> bool:
        return str(self.model.get("model", "gnn")) in ("gnn_gru", "gnn_attention")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "labels": self.label,
            "dataset_root": self.dataset_root,
            "model": dict(self.model),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "schedule": self.schedule,
            "warmup_steps": self.warmup_steps,
            "label_smoothing": self.label_smoothing,
            "abstain_weight": self.abstain_weight,
            "graphs_per_batch": self.graphs_per_batch,
            "steps_per_chunk": self.steps_per_chunk,
            "batches_per_epoch": self.batches_per_epoch,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "early_stopping": {
                "patience": self.early_stopping_patience,
                "min_delta": self.early_stopping_min_delta,
                "metric": self.selection_metric,
            },
            "evaluate_every": self.evaluate_every,
            "evaluation_episodes": self.evaluation_episodes,
            "max_sequence_samples": self.max_sequence_samples,
            "use_abstain_class": self.use_abstain_class,
            "export": self.export,
            "curriculum": self.curriculum,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TrainingConfig":
        early = dict(payload.get("early_stopping", {}))
        return cls(
            run_id=str(payload["run_id"]),
            dataset_root=str(payload["dataset_root"]),
            model=dict(payload.get("model", {"model": "gnn"})),
            learning_rate=float(payload.get("learning_rate", 3e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            schedule=str(payload.get("schedule", "cosine")),
            warmup_steps=int(payload.get("warmup_steps", 0)),
            label_smoothing=float(payload.get("label_smoothing", 0.0)),
            abstain_weight=float(payload.get("abstain_weight", 1.0)),
            graphs_per_batch=int(payload.get("graphs_per_batch", 8)),
            steps_per_chunk=int(payload.get("steps_per_chunk", 4)),
            batches_per_epoch=int(payload.get("batches_per_epoch", 16)),
            epochs=int(payload.get("epochs", 4)),
            max_steps=int(payload.get("max_steps", 0)),
            seed=int(payload.get("seed", 0)),
            early_stopping_patience=int(early.get("patience", 3)),
            early_stopping_min_delta=float(early.get("min_delta", 1e-4)),
            selection_metric=str(early.get("metric", "validation_loss")),
            evaluate_every=int(payload.get("evaluate_every", 0)),
            evaluation_episodes=int(payload.get("evaluation_episodes", 4)),
            max_sequence_samples=int(payload.get("max_sequence_samples", 4000)),
            use_abstain_class=bool(payload.get("use_abstain_class", True)),
            export=bool(payload.get("export", True)),
            curriculum=payload.get("curriculum"),
            label=str(payload.get("labels", payload.get("label", ""))),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_payload(payload)


def imitation_loss(
    output: PolicyOutput,
    batch: StepBatch,
    *,
    label_smoothing: float = 0.0,
    abstain_weight: float = 1.0,
) -> dict[str, Any]:
    """Cross-entropy over each actionable satellite's masked action set.

    Classes are the satellite's masked candidate edges plus one explicit abstain
    class, so "wait" is a learned decision rather than an absence of output. The
    expert's greedily matched action is the target.
    """
    supervised = batch.supervised_satellites()
    if not supervised:
        zero = output.edge_logits.sum() * 0.0 + output.abstain_logits.sum() * 0.0
        return {"loss": zero, "count": 0, "correct": 0, "abstains": 0, "losses": []}
    total = torch.zeros((), dtype=output.edge_logits.dtype)
    correct = 0
    abstained = 0
    losses: list[tuple[str, float]] = []
    for node in supervised:
        candidate_edges = batch.candidates.get(node, [])
        scores = [output.edge_logits[edge] for edge in candidate_edges]
        scores.append(output.abstain_logits[node] * abstain_weight)
        logits = torch.stack(scores).unsqueeze(0)
        target_edge = batch.targets.get(node)
        target = candidate_edges.index(target_edge) if target_edge in candidate_edges else len(candidate_edges)
        loss = nn.functional.cross_entropy(
            logits,
            torch.tensor([target], dtype=torch.long),
            label_smoothing=label_smoothing,
        )
        weight = 1.0
        graph_index = batch.node_graph[node] if node < len(batch.node_graph) else 0
        if graph_index < len(batch.sample_weights):
            weight = max(0.0, float(batch.sample_weights[graph_index]))
        total = total + loss * weight
        prediction = int(logits.argmax(dim=-1).item())
        correct += int(prediction == target)
        abstained += int(target == len(candidate_edges))
        family = batch.families[graph_index] if graph_index < len(batch.families) else ""
        losses.append((family, float(loss.item())))
    count = len(supervised)
    return {
        "loss": total / count,
        "count": count,
        "correct": correct,
        "abstains": abstained,
        "losses": losses,
    }


def make_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def learning_rate_at(step: int, total_steps: int, config: TrainingConfig) -> float:
    if config.schedule == "constant" or total_steps <= 0:
        return config.learning_rate
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)))
    return 0.5 * config.learning_rate * (1.0 + math.cos(math.pi * progress))


def batch_plan(
    samples: Sequence[GraphSample],
    config: TrainingConfig,
) -> list[list[list[GraphSample]]]:
    """Epoch-sized plan: a list of batches, each a list of *temporal chunks*.

    Stateless families get one chunk per batch (several graphs in parallel).
    Recurrent families get chunks of consecutive steps of the same episode, so
    hidden state is carried across time inside a batch and reset between them.
    """
    by_episode: dict[str, list[GraphSample]] = {}
    for sample in samples:
        by_episode.setdefault(sample.episode_id, []).append(sample)
    for entries in by_episode.values():
        entries.sort(key=lambda item: item.step_index)
    episodes = sorted(by_episode)

    batches: list[list[list[GraphSample]]] = []
    if not config.temporal:
        flat: list[GraphSample] = []
        for episode in episodes:
            flat.extend(by_episode[episode])
        for start in range(0, len(flat), config.graphs_per_batch):
            chunk = flat[start : start + config.graphs_per_batch]
            if chunk:
                batches.append([list(chunk)])
    else:
        for episode in episodes:
            entries = by_episode[episode]
            for start in range(0, len(entries), config.steps_per_chunk):
                chunk = entries[start : start + config.steps_per_chunk]
                if chunk:
                    batches.append([[sample] for sample in chunk])
    return batches


@dataclass
class Trainer:
    """One training run: prepare, train, evaluate, checkpoint, export."""

    config: TrainingConfig
    output_dir: Path
    dataset: EpisodeDataset | None = None
    device: str = "cpu"
    # Optional curriculum hooks: which decision samples a stage uses, and how the
    # stage re-balances them so easy scenarios do not dominate rare ones.
    sample_filter: Any = None
    weight_fn: Any = None
    init_weights_from: Path | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.manifest_path = self.output_dir / "training-manifest.json"
        self.checkpoint_path = self.output_dir / "checkpoint-last.pt"
        self.best_path = self.output_dir / "checkpoint-best.pt"
        self.normalizer = Normalizer()
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.history: list[dict[str, Any]] = []
        self.global_step = 0
        self.epoch = 0
        self.best_metric: float | None = None

    # --- preparation -------------------------------------------------------------------
    def load_data(self) -> dict[str, Any]:
        self.dataset = self.dataset or EpisodeDataset(self.config.dataset_root)
        plan: dict[str, Any] = {}
        for split in ("train", "validation", "test"):
            samples = samples_from_dataset(self.dataset, split, limit=self.config.max_sequence_samples)
            plan[split] = samples
        if not plan["train"]:
            raise TrainingError(
                f"campaign {self.config.dataset_root!r} has no training decision samples"
            )
        selected = self._select(plan["train"])
        validation = self._select(plan["validation"])
        self.train_samples = selected
        self.validation_samples = validation
        self.test_samples = self._select(plan["test"])
        self.filtered_out = sum(len(plan[split]) for split in plan) - len(selected) - len(validation) - len(self.test_samples)
        return {
            "train": len(self.train_samples),
            "validation": len(self.validation_samples),
            "test": len(self.test_samples),
            "filtered_out": self.filtered_out,
        }

    def _select(self, samples: Sequence[GraphSample]) -> list[GraphSample]:
        kept = [sample for sample in samples if sample.candidate_pairs()]
        if self.sample_filter is not None:
            kept = [sample for sample in kept if self.sample_filter(sample)]
        if self.weight_fn is not None:
            kept = [
                GraphSample(**{**sample.__dict__, "weight": float(self.weight_fn(sample))})
                for sample in kept
            ]
        return kept

    def prepare(self) -> dict[str, Any]:
        """Fit normalization on training data and build model and optimizer."""
        counts = self.load_data()
        # Normalization is fitted on flat training graphs: temporal chunking is a
        # training-loop detail and must not change the stored statistics.
        train_batches = [
            build_batch(chunk)
            for chunk in flat_chunks(self.train_samples, self.config.graphs_per_batch)
        ]
        self.normalizer = Normalizer.fit(train_batches)
        torch.manual_seed(self.config.seed)
        self.model = build_model(self.config.model).to(self.device)
        self.optimizer = make_optimizer(self.model, self.config)
        return {
            **counts,
            "model": dict(self.config.model),
            "parameters": model_parameter_count(self.model),
            "normalization_rows": self.normalizer.counts,
        }

    # --- training ----------------------------------------------------------------------
    def train(self, *, resume_from: str | Path | None = None) -> dict[str, Any]:
        if self.model is None or self.optimizer is None:
            self.prepare()
        assert self.model is not None and self.optimizer is not None
        if self.init_weights_from is not None:
            # Curriculum transfer: weights come from the previous stage, while
            # the optimizer, epoch counter and history start fresh.
            payload = torch.load(Path(self.init_weights_from), weights_only=False)
            self.model.load_state_dict(payload["state_dict"])
            self.normalizer = Normalizer.from_payload(payload["normalizer"])
        if resume_from is not None:
            self._resume(Path(resume_from))
        total_steps = self.config.max_steps or self.config.epochs * self.config.batches_per_epoch
        while (self.config.max_steps and self.global_step < self.config.max_steps) or (
            not self.config.max_steps and self.epoch < self.config.epochs
        ):
            epoch_losses: list[float] = []
            correct = total = 0
            started = time.perf_counter()
            batches = batch_plan(self.train_samples, self.config)
            for batch_index in range(min(self.config.batches_per_epoch, max(1, len(batches)))):
                chunk_group = batches[(self.epoch * self.config.batches_per_epoch + batch_index) % len(batches)]
                step_metrics = self._train_batch(chunk_group)
                epoch_losses.append(step_metrics["loss"])
                correct += step_metrics["correct"]
                total += step_metrics["count"]
                self.global_step += 1
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break
            validation = self._validate()
            entry = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "train_loss": round(sum(epoch_losses) / max(1, len(epoch_losses)), 6),
                "train_accuracy": round(correct / total, 6) if total else None,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "wall_time_s": round(time.perf_counter() - started, 4),
            }
            if self.config.evaluate_every and (
                self.epoch % self.config.evaluate_every == 0 or self.epoch == self.config.epochs - 1
            ):
                entry["validation_reward"] = self._validation_reward()
            self.history.append(entry)
            self._log(entry)
            self._maybe_checkpoint(entry)
            self.epoch += 1
            if self._should_stop():
                entry["early_stopped"] = True
                break
        report = self.finish()
        return report

    def _train_batch(self, chunk_group: list[list[GraphSample]]) -> dict[str, Any]:
        """One optimizer step over a chunk group.

        A temporal chunk group is a short sequence of steps of the same episode,
        so its losses are summed and back-propagated **once**: that is truncated
        BPTT with the declared chunk length, and it is why hidden state may flow
        through the graph within a chunk but never between batches.
        """
        assert self.model is not None and self.optimizer is not None
        self.model.train()
        state: dict[str, Any] | None = None
        self.optimizer.zero_grad(set_to_none=True)
        aggregate = {"loss": 0.0, "correct": 0, "count": 0}
        chunks = chunk_group
        total_loss: torch.Tensor | None = None
        for step_index, chunk in enumerate(chunks):
            torch.manual_seed(self.config.seed + self.global_step * 7919 + step_index)
            batch = self.normalizer.apply(build_batch(chunk))
            output = self.model(batch, state=state)
            state = output.state or None
            result = imitation_loss(
                output,
                batch,
                label_smoothing=self.config.label_smoothing,
                abstain_weight=self.config.abstain_weight,
            )
            aggregate["loss"] += float(result["loss"].detach().item())
            aggregate["correct"] += int(result["correct"])
            aggregate["count"] += int(result["count"])
            total_loss = result["loss"] if total_loss is None else total_loss + result["loss"]
        if total_loss is not None:
            (total_loss / max(1, len(chunks))).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate_at(
                    self.global_step,
                    self.config.max_steps or self.config.epochs * self.config.batches_per_epoch,
                    self.config,
                )
        self.optimizer.step()
        aggregate["loss"] /= max(1, len(chunks))
        return aggregate

    @torch.no_grad()
    def _validate(self) -> dict[str, Any]:
        assert self.model is not None
        if not self.validation_samples:
            return {"loss": None, "accuracy": None, "count": 0}
        self.model.eval()
        total_loss = 0.0
        correct = total = 0
        batches = batch_plan(self.validation_samples, self.config)
        for chunk_group in batches[: self.config.batches_per_epoch]:
            state: dict[str, Any] | None = None
            for chunk in chunk_group:
                batch = self.normalizer.apply(build_batch(chunk))
                output = self.model(batch, state=state)
                state = output.state or None
                result = imitation_loss(output, batch, abstain_weight=self.config.abstain_weight)
                total_loss += float(result["loss"].item())
                correct += int(result["correct"])
                total += int(result["count"])
        return {
            "loss": round(total_loss / max(1, len(batches[: self.config.batches_per_epoch])), 6),
            "accuracy": round(correct / total, 6) if total else None,
            "count": total,
        }

    def _validation_reward(self) -> float | None:
        """Held-out environment reward (the metric that actually matters)."""
        if self.dataset is None:
            return None
        episodes = self.dataset.episodes("validation")[: self.config.evaluation_episodes]
        if not episodes:
            return None
        report = evaluate_policy(
            self.policy(), episodes, use_abstain_class=self.config.use_abstain_class
        )
        return report.aggregate["mean_coverage"]

    def _should_stop(self) -> bool:
        if not self.history or self.config.early_stopping_patience <= 0:
            return False
        metric = self.config.selection_metric
        values = [
            entry.get(metric if metric != "validation_loss" else "validation_loss")
            for entry in self.history
        ]
        if metric == "validation_accuracy":
            values = [entry.get("validation_accuracy") for entry in self.history]
            current = values[-1]
            if current is None:
                return False
            best = max(value for value in values if value is not None)
            return current < best - self.config.early_stopping_min_delta
        current = self.history[-1].get("validation_loss")
        if current is None:
            return False
        best = min(entry["validation_loss"] for entry in self.history if entry.get("validation_loss") is not None)
        if current > best + self.config.early_stopping_min_delta:
            patience = self.config.early_stopping_patience
            recent = self.history[-patience:]
            return all(
                (entry.get("validation_loss") or 0.0) > best + self.config.early_stopping_min_delta
                for entry in recent
            ) and len(self.history) > patience
        return False

    def _metric_value(self, entry: Mapping[str, Any]) -> float | None:
        metric = self.config.selection_metric
        if metric == "validation_accuracy":
            return entry.get("validation_accuracy")
        if metric == "validation_reward":
            return entry.get("validation_reward")
        return entry.get("validation_loss")

    def _maybe_checkpoint(self, entry: Mapping[str, Any]) -> None:
        self.save_checkpoint(self.checkpoint_path, entry=entry)
        value = self._metric_value(entry)
        if value is None:
            return
        better = (
            self.best_metric is None
            or (self.config.selection_metric == "validation_accuracy" and value > self.best_metric)
            or (self.config.selection_metric != "validation_accuracy" and value < self.best_metric)
        )
        if better:
            self.best_metric = float(value)
            self.save_checkpoint(self.best_path, entry=entry)

    # --- artefacts ---------------------------------------------------------------------
    def _log(self, entry: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(entry), sort_keys=True) + "\n")

    def checkpoint_payload(self, *, entry: Mapping[str, Any] | None = None) -> dict[str, Any]:
        assert self.model is not None
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "run_id": self.config.run_id,
            # Completed epochs, so resuming continues at the next one.
            "epoch": self.epoch + 1,
            "global_step": self.global_step,
            "policy_key": f"learned-{self.config.model.get('model', 'gnn')}",
            "model_config": dict(self.config.model),
            "state_dict": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "normalizer": self.normalizer.payload(),
            "feature_schema": feature_schema_payload(),
            "observation_version": OBSERVATION_VERSION,
            "env_version": ENV_VERSION,
            "parameter_schema_hash": self._parameter_schema_hash(),
            "dataset_campaign": self.dataset.manifest["campaign_id"] if self.dataset else "",
            "training_config": self.config.to_dict(),
            "history": list(self.history),
            "best_metric": self.best_metric,
            "code_revision": code_revision(),
            "versions": library_versions(),
            "entry": dict(entry or {}),
        }

    def _parameter_schema_hash(self) -> str:
        if self.dataset is not None:
            return str(self.dataset.manifest["schema"]["parameter_schema_hash"])
        return ""

    def save_checkpoint(self, path: str | Path, *, entry: Mapping[str, Any] | None = None) -> Path:
        payload = self.checkpoint_payload(entry=entry)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        return target

    def _resume(self, path: Path) -> None:
        payload = torch.load(path, weights_only=False)
        if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise TrainingError(
                f"checkpoint version {payload.get('checkpoint_version')!r} is not {CHECKPOINT_VERSION!r}"
            )
        assert self.model is not None and self.optimizer is not None
        self.model.load_state_dict(payload["state_dict"])
        if payload.get("optimizer_state"):
            self.optimizer.load_state_dict(payload["optimizer_state"])
        self.normalizer = Normalizer.from_payload(payload["normalizer"])
        self.history = list(payload.get("history", []))
        self.global_step = int(payload.get("global_step", 0))
        self.epoch = int(payload.get("epoch", 0))  # next epoch to run
        self.best_metric = payload.get("best_metric")

    def policy(self, *, use_abstain_class: bool | None = None) -> ModelPolicy:
        assert self.model is not None
        metadata = LearnedPolicyMetadata(
            key=f"learned-{self.config.model.get('model', 'gnn')}",
            model=dict(self.config.model),
            feature_schema=feature_schema_payload(),
        )
        return ModelPolicy(self.model, self.normalizer, metadata=metadata, device=self.device)

    def evaluate(self, split: str = "test", *, limit: int | None = None) -> dict[str, Any]:
        if self.dataset is None:
            raise TrainingError("no dataset loaded")
        records = self.dataset.episodes(split)[: limit if limit is not None else None]
        report = evaluate_policy(
            self.policy(), records, use_abstain_class=self.config.use_abstain_class
        )
        return report.to_dict()

    def finish(self) -> dict[str, Any]:
        manifest = self.manifest()
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        if self.config.export:
            export_path = export_checkpoint(self.best_path if self.best_path.exists() else self.checkpoint_path,
                                            self.output_dir / "export")
            manifest["export"] = str(export_path)
            self.manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
            )
        return manifest

    def manifest(self) -> dict[str, Any]:
        return {
            "training_version": TRAINING_VERSION,
            "run_id": self.config.run_id,
            "config": self.config.to_dict(),
            "resolved_config": dict(self.config.model),
            "dataset_campaign": self.dataset.manifest["campaign_id"] if self.dataset else "",
            "schema": {
                "observation_version": OBSERVATION_VERSION,
                "feature_schema": feature_schema_payload(),
                "parameter_schema_hash": self._parameter_schema_hash(),
            },
            "code_revision": code_revision(),
            "versions": library_versions(),
            "hyperparameters": {
                "parameters": model_parameter_count(self.model) if self.model else 0,
                "temporal": self.config.temporal,
            },
            "history": list(self.history),
            "best_metric": self.best_metric,
            "selection_metric": self.config.selection_metric,
            "checkpoints": {
                "last": str(self.checkpoint_path) if self.checkpoint_path.exists() else None,
                "best": str(self.best_path) if self.best_path.exists() else None,
            },
            "normalization": self.normalizer.describe(),
        }


def export_checkpoint(checkpoint_path: str | Path, directory: str | Path) -> Path:
    """Export a checkpoint as a portable policy (weights + config + normalizer)."""
    payload = torch.load(Path(checkpoint_path), weights_only=False)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    policy_payload = {
        key: value
        for key, value in payload.items()
        if key not in ("state_dict", "optimizer_state")
    }
    (target / "policy.json").write_text(
        json.dumps(policy_payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    torch.save(payload["state_dict"], target / "weights.pt")
    from .features import Normalizer

    (target / "normalizer.json").write_text(
        json.dumps(Normalizer.from_payload(payload["normalizer"]).payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_export(directory: str | Path) -> dict[str, Any]:
    """Read an exported policy back into the checkpoint shape policies expect."""
    target = Path(directory)
    policy = json.loads((target / "policy.json").read_text(encoding="utf-8"))
    state = torch.load(target / "weights.pt", weights_only=False)
    policy["state_dict"] = state
    policy["normalizer"] = json.loads((target / "normalizer.json").read_text(encoding="utf-8"))
    policy["path"] = str(target)
    return policy


def run_training(
    config: TrainingConfig,
    output_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Train one run and return its manifest (used by the CLI, sweeps, curriculum)."""
    trainer = Trainer(config, output_dir)
    return trainer.train(resume_from=resume_from)


def flat_chunks(samples: Sequence[GraphSample], size: int) -> Iterator[list[GraphSample]]:
    """Consecutive groups of decision samples (order preserved)."""
    size = max(1, size)
    buffer: list[GraphSample] = []
    for sample in samples:
        buffer.append(sample)
        if len(buffer) >= size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


__all__ = [
    "TRAINING_VERSION",
    "CHECKPOINT_VERSION",
    "TrainingError",
    "TrainingConfig",
    "flat_chunks",
    "Trainer",
    "imitation_loss",
    "batch_plan",
    "learning_rate_at",
    "run_training",
    "export_checkpoint",
    "load_export",
    "code_revision",
    "library_versions",
]
