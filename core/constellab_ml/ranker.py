"""Day 55: a neural candidate ranker for adaptive adversarial search.

The ranker sees a *set* of fault events plus episode/mutation context and
predicts whether the candidate will expose an invariant error. It is a ranking
aid, not an oracle: the invariant monitor remains authoritative, the baseline
search path can always be selected, and predictions are reported with ranking
and calibration diagnostics rather than accuracy alone.

Splits are by scenario family where more than one family is present. A dataset
built from a single family (the common smoke case) is split deterministically
and explicitly marked as not family-held-out in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from .adversarial import MUTATION_OPERATORS, Mutation
from .env import EpisodeSpec

RANKER_VERSION = "mutation-ranker-1"
RANKER_CHECKPOINT_VERSION = "mutation-ranker-checkpoint-1"

FAULT_KINDS: tuple[str, ...] = (
    "satellite-failure",
    "sensor-degrade",
    "link-cut",
    "message-delay",
)
FAMILIES: tuple[str, ...] = (
    "partitioned", "faulted", "degraded", "small-easy", "large-hard", "nominal",
)
FAULT_FEATURES: tuple[str, ...] = (
    "kind-satellite-failure", "kind-sensor-degrade", "kind-link-cut", "kind-message-delay",
    "start_fraction", "duration_fraction", "quality", "delay_fraction",
    "satellite_share", "has_peer",
)
CONTEXT_FEATURES: tuple[str, ...] = (
    *(f"operator:{name}" for name in MUTATION_OPERATORS),
    "fleet_norm", "targets_norm", "faults_norm", "horizon_norm", "step_norm",
    *(f"family:{name}" for name in FAMILIES),
)


class RankerError(RuntimeError):
    """A ranker dataset, configuration or checkpoint is invalid."""


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def encode_spec(spec: EpisodeSpec, *, operator: str = "combine-faults") -> tuple[list[float], list[list[float]], list[float]]:
    """Return context vector, fault token rows and their mask."""
    fault_rows = list((spec.fault_plan or {}).get("faults") or [])
    satellites = max(1, len(spec.config.get("satellites") or []))
    horizon = max(1.0, float(spec.horizon_s))
    context = [
        *(1.0 if operator == name else 0.0 for name in MUTATION_OPERATORS),
        min(1.0, satellites / 24.0),
        min(1.0, len(spec.targets) / 12.0),
        min(1.0, len(fault_rows) / 8.0),
        min(1.0, horizon / 7200.0),
        min(1.0, float(spec.step_s) / 300.0),
        *(1.0 if spec.family == name else 0.0 for name in FAMILIES),
    ]
    tokens: list[list[float]] = []
    mask: list[float] = []
    for row in fault_rows:
        kind = str(row.get("kind", ""))
        start = _num(row.get("start_s"))
        end = _num(row.get("end_s"), start + 1.0)
        duration = max(0.0, end - start)
        satellites_row = list(row.get("satellites") or [])
        peers = list(row.get("peers") or [])
        tokens.append([
            *(1.0 if kind == name else 0.0 for name in FAULT_KINDS),
            min(1.0, max(0.0, start / horizon)),
            min(1.0, duration / horizon),
            min(1.0, max(0.0, _num(row.get("quality"), 1.0))),
            min(1.0, max(0.0, _num(row.get("delay_s"))) / 300.0),
            min(1.0, len(satellites_row) / satellites),
            1.0 if peers else 0.0,
        ])
        mask.append(1.0)
    if not tokens:
        tokens = [[0.0] * len(FAULT_FEATURES)]
        mask = [0.0]
    return context, tokens, mask


class MutationRanker(nn.Module):
    """Masked set encoder over fault tokens plus mutation/episode context."""

    def __init__(
        self,
        *,
        hidden: int = 32,
        fault_features: int = len(FAULT_FEATURES),
        context_features: int = len(CONTEXT_FEATURES),
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.fault_encoder = nn.Sequential(
            nn.Linear(fault_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        context: torch.Tensor,
        fault_tokens: torch.Tensor,
        fault_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.fault_encoder(fault_tokens)  # [batch, faults, hidden]
        mask = fault_mask.unsqueeze(-1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean = (encoded * mask).sum(dim=1) / counts
        masked = encoded.masked_fill(mask <= 0.0, -1e9)
        maximum = masked.max(dim=1).values
        maximum = torch.where(counts > 0.0, maximum, torch.zeros_like(maximum))
        ctx = self.context_encoder(context)
        return self.head(torch.cat([mean, maximum, ctx], dim=-1)).squeeze(-1)


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source = Path(path)
    paths = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
    for item in paths:
        with item.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


@dataclass
class RankerConfig:
    dataset: str
    run_id: str = "mutation-ranker"
    hidden: int = 32
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    seed: int = 0
    validation_share: float = 0.25
    output: str = ""

    def __post_init__(self) -> None:
        if not self.dataset:
            raise RankerError("dataset path is required")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise RankerError("epochs and batch_size must be positive")
        if not 0.0 < self.validation_share < 1.0:
            raise RankerError("validation_share must be within (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "run_id": self.run_id,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "validation_share": self.validation_share,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RankerConfig":
        return cls(**{key: payload[key] for key in (
            "dataset", "run_id", "hidden", "epochs", "learning_rate", "weight_decay",
            "batch_size", "seed", "validation_share", "output",
        ) if key in payload})


@dataclass
class RankerDataset:
    contexts: list[list[float]] = field(default_factory=list)
    fault_tokens: list[list[list[float]]] = field(default_factory=list)
    fault_masks: list[list[float]] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(1, max((len(row) for row in self.fault_tokens), default=1))
        tokens = torch.zeros(len(self), width, len(FAULT_FEATURES))
        masks = torch.zeros(len(self), width)
        for index, rows in enumerate(self.fault_tokens):
            for offset, row in enumerate(rows):
                tokens[index, offset, : len(row)] = torch.tensor(row, dtype=torch.float32)
                masks[index, offset] = self.fault_masks[index][offset]
        contexts = torch.tensor(self.contexts, dtype=torch.float32)
        labels = torch.tensor(self.labels, dtype=torch.float32)
        return contexts, tokens, masks, labels


def build_dataset(records: Sequence[Mapping[str, Any]]) -> RankerDataset:
    dataset = RankerDataset()
    seen: set[str] = set()
    for record in records:
        spec_payload = (record.get("mutation") or {}).get("spec")
        if not spec_payload:
            continue
        spec = EpisodeSpec.from_payload(spec_payload)
        key = spec.spec_hash()
        if key in seen:
            continue
        seen.add(key)
        operator = str((record.get("mutation") or {}).get("operator", "combine-faults"))
        context, tokens, mask = encode_spec(spec, operator=operator)
        dataset.contexts.append(context)
        dataset.fault_tokens.append(tokens)
        dataset.fault_masks.append(mask)
        dataset.labels.append(1.0 if (record.get("result") or {}).get("failure") else 0.0)
        dataset.families.append(spec.family)
        dataset.hashes.append(key)
    return dataset


def _split(dataset: RankerDataset, *, share: float, seed: int) -> tuple[list[int], list[int], bool]:
    groups: dict[str, list[int]] = {}
    for index, family in enumerate(dataset.families):
        groups.setdefault(family, []).append(index)
    family_held_out = len(groups) > 1
    rng = random.Random(seed)
    if family_held_out:
        names = sorted(groups)
        rng.shuffle(names)
        validation_groups = set(names[: max(1, int(len(names) * share))])
        validation = [index for name in sorted(validation_groups) for index in groups[name]]
        train = [index for name in sorted(set(names) - validation_groups) for index in groups[name]]
    else:
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        cut = max(1, int(len(indices) * share))
        validation = sorted(indices[:cut])
        train = sorted(indices[cut:])
    if not train or not validation:
        raise RankerError("not enough distinct records for a train/validation split")
    return train, validation, family_held_out


def _auc(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label > 0.5]
    negatives = [score for label, score in zip(labels, scores) if label <= 0.5]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _average_precision(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    if sum(labels) == 0:
        return None
    order = sorted(range(len(labels)), key=lambda index: -scores[index])
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] > 0.5:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / sum(labels)


def _top_k(labels: Sequence[float], scores: Sequence[float], k: int) -> dict[str, Any]:
    order = sorted(range(len(labels)), key=lambda index: -scores[index])[: max(1, k)]
    found = sum(1 for index in order if labels[index] > 0.5)
    total = sum(1 for label in labels if label > 0.5)
    return {
        "k": len(order),
        "failures": found,
        "precision": round(found / len(order), 6) if order else 0.0,
        "recall": round(found / total, 6) if total else None,
    }


def _calibration(labels: Sequence[float], scores: Sequence[float], bins: int = 5) -> list[dict[str, Any]]:
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for label, score in zip(labels, scores):
        buckets[min(bins - 1, max(0, int(score * bins)))].append((label, score))
    rows = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        rows.append({
            "bin": index,
            "n": len(bucket),
            "mean_score": round(statistics.fmean(score for _, score in bucket), 6),
            "observed_failure_rate": round(statistics.fmean(label for label, _ in bucket), 6),
        })
    return rows


class RankerPredictor:
    """Checkpoint-backed callable used by the guided search runner."""

    def __init__(self, model: MutationRanker, config: RankerConfig, metadata: Mapping[str, Any] | None = None) -> None:
        self.model = model
        self.config = config
        self.metadata = dict(metadata or {})
        self.model.eval()

    def score_specs(self, specs: Sequence[EpisodeSpec], *, operators: Sequence[str] | None = None) -> list[float]:
        contexts, tokens, masks = [], [], []
        for index, spec in enumerate(specs):
            operator = operators[index] if operators and index < len(operators) else "combine-faults"
            context, rows, mask = encode_spec(spec, operator=operator)
            contexts.append(context)
            tokens.append(rows)
            masks.append(mask)
        dataset = RankerDataset(contexts=contexts, fault_tokens=tokens, fault_masks=masks,
                                labels=[0.0] * len(specs), families=[spec.family for spec in specs],
                                hashes=[spec.spec_hash() for spec in specs])
        context_t, tokens_t, masks_t, _ = dataset.tensors()
        with torch.no_grad():
            logits = self.model(context_t, tokens_t, masks_t)
        return [float(torch.sigmoid(value).item()) for value in logits]

    def __call__(self, mutations: Sequence[Mutation]) -> list[float]:
        return self.score_specs(
            [mutation.spec for mutation in mutations],
            operators=[mutation.operator for mutation in mutations],
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "checkpoint_version": RANKER_CHECKPOINT_VERSION,
            "model_state": self.model.state_dict(),
            "config": self.config.to_dict(),
            "context_features": list(CONTEXT_FEATURES),
            "fault_features": list(FAULT_FEATURES),
            "metadata": dict(self.metadata),
            "versions": {"torch": torch.__version__, "ranker": RANKER_VERSION},
        }, target)
        return target


def train_ranker(config: RankerConfig, output: str | Path | None = None) -> dict[str, Any]:
    root = Path(output or config.output or "runs/ranker")
    root.mkdir(parents=True, exist_ok=True)
    records = _load_records(config.dataset)
    if not records:
        raise RankerError(f"no records in {config.dataset!r}")
    dataset = build_dataset(records)
    train_idx, validation_idx, family_held_out = _split(
        dataset, share=config.validation_share, seed=config.seed
    )
    contexts, tokens, masks, labels = dataset.tensors()
    torch.manual_seed(config.seed)
    model = MutationRanker(hidden=config.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    positives = max(1.0, float(labels[train_idx].sum().item()))
    negatives = max(1.0, float(len(train_idx) - positives))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
    history = []
    best_loss = math.inf
    best_state = None
    for epoch in range(config.epochs):
        model.train()
        order = list(train_idx)
        random.Random(config.seed + epoch).shuffle(order)
        total = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(contexts[batch], tokens[batch], masks[batch])
            loss = criterion(logits, labels[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().item()) * len(batch)
        model.eval()
        with torch.no_grad():
            validation_logits = model(contexts[validation_idx], tokens[validation_idx], masks[validation_idx])
            validation_loss = float(criterion(validation_logits, labels[validation_idx]).item())
            scores = torch.sigmoid(validation_logits).tolist()
        entry = {
            "epoch": epoch,
            "train_loss": round(total / max(1, len(train_idx)), 6),
            "validation_loss": round(validation_loss, 6),
            "validation_auc": _auc(labels[validation_idx].tolist(), scores),
            "validation_average_precision": _average_precision(labels[validation_idx].tolist(), scores),
        }
        history.append(entry)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    predictor = RankerPredictor(model, config, metadata={
        "family_held_out": family_held_out,
        "train_records": len(train_idx),
        "validation_records": len(validation_idx),
    })
    checkpoint = predictor.save(root / "ranker.pt")
    model.eval()
    with torch.no_grad():
        validation_logits = model(contexts[validation_idx], tokens[validation_idx], masks[validation_idx])
        validation_scores = torch.sigmoid(validation_logits).tolist()
    labels_validation = labels[validation_idx].tolist()
    manifest = {
        "version": RANKER_VERSION,
        "config": config.to_dict(),
        "records": len(dataset),
        "train_records": len(train_idx),
        "validation_records": len(validation_idx),
        "family_held_out": family_held_out,
        "split_note": (
            "validation families are held out" if family_held_out
            else "single-family smoke dataset; split is deterministic, not family-held-out"
        ),
        "class_balance": {
            "failures": int(labels.sum().item()),
            "non_failures": int(len(labels) - labels.sum().item()),
        },
        "history": history,
        "metrics": {
            "validation_loss": round(best_loss, 6),
            "validation_auc": _auc(labels_validation, validation_scores),
            "validation_average_precision": _average_precision(labels_validation, validation_scores),
            "top_k": _top_k(labels_validation, validation_scores, max(1, len(validation_idx) // 3)),
            "calibration": _calibration(labels_validation, validation_scores),
        },
        "checkpoint": str(checkpoint),
        "features": {
            "context": list(CONTEXT_FEATURES),
            "fault": list(FAULT_FEATURES),
        },
        "selection_note": (
            "ranker metrics are advisory; the invariant monitor remains authoritative"
        ),
    }
    (root / "ranker-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_ranker(path: str | Path) -> RankerPredictor:
    payload = torch.load(Path(path), weights_only=False)
    if payload.get("checkpoint_version") != RANKER_CHECKPOINT_VERSION:
        raise RankerError(
            f"checkpoint version {payload.get('checkpoint_version')!r} is not "
            f"{RANKER_CHECKPOINT_VERSION!r}"
        )
    config = RankerConfig.from_payload(payload["config"])
    model = MutationRanker(hidden=int(payload["config"].get("hidden", 32)))
    model.load_state_dict(payload["model_state"])
    return RankerPredictor(model, config, metadata=payload.get("metadata"))


__all__ = [
    "RANKER_VERSION",
    "RANKER_CHECKPOINT_VERSION",
    "FAULT_KINDS",
    "FAMILIES",
    "FAULT_FEATURES",
    "CONTEXT_FEATURES",
    "RankerError",
    "encode_spec",
    "MutationRanker",
    "RankerConfig",
    "RankerDataset",
    "build_dataset",
    "RankerPredictor",
    "train_ranker",
    "load_ranker",
]
