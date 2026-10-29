"""Day 45: curriculum imitation training across randomized configurations.

The curriculum is a sequence of **data slices**, not a sequence of models: every
stage re-uses the same architecture and only changes which episodes are admitted
and how strongly they are weighted. That keeps the comparison honest — a stage
cannot "improve" by quietly becoming a different model — and it makes the
progression report a statement about coverage rather than about parameter counts.

Three rules keep the result interpretable:

* **Balance before size.** Later stages admit more satellites, tasks, faults and
  partitions, and every stage re-weights its samples by inverse frequency of
  ``(condition class, fleet bucket, scenario family)`` so easy, common missions
  do not drown the rare ones.
* **Selection is frozen before the test split is touched.** Each stage selects
  its checkpoint on validation data only; the curriculum picks its winner from
  those validation numbers, and only then evaluates on the held-out test split.
* **Stopping is a decision, not a failure.** A stage may early-stop on its own
  patience, and the whole curriculum stops when validation stops improving for
  ``stall_patience`` consecutive stages. Both outcomes are recorded with their
  reason and the best checkpoint is still evaluated.

The output is a machine-readable ``curriculum-manifest.json``: configuration,
per-stage coverage and loss curves, the frozen selection, seed-to-seed variation
and the held-out comparison against the rule expert and the wait baseline.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .batch import GraphSample, samples_from_dataset
from .dataset import EpisodeDataset
from .env import ACTION_WAIT, Action, EnvPolicy
from .expert import expert_factory
from .models import MODEL_FAMILIES
from .policies import build_model_policy_from_checkpoint, compare_policies
from .training import TrainingConfig, Trainer

CURRICULUM_VERSION = "curriculum-1"

# Fleet-size buckets used by the balanced sampler and the coverage report.
FLEET_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("small", 0, 4),
    ("medium", 5, 9),
    ("large", 10, 18),
    ("very-large", 19, 10_000),
)


class CurriculumError(RuntimeError):
    """The curriculum configuration or a stage is invalid."""


def fleet_bucket(size: Any) -> str:
    try:
        value = int(size)
    except (TypeError, ValueError):
        return "unknown"
    for name, low, high in FLEET_BUCKETS:
        if low <= value <= high:
            return name
    return "very-large"


# --- stages ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    """One rung of the ladder: which episodes are admitted, and for how long.

    Every limit is optional and inclusive. ``None`` means "no limit on this
    axis", so a final stage is simply a stage with no limits. Missing metadata
    is never treated as a violation: a sample is only rejected when the
    attribute exists and exceeds the declared bound.
    """

    name: str
    description: str = ""
    max_fleet_size: int | None = None
    max_targets: int | None = None
    max_faults: int | None = None
    max_partitions: int | None = None
    families: tuple[str, ...] | None = None
    epochs: int | None = None
    balance: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise CurriculumError("a stage needs a name")
        for label, value in (
            ("max_fleet_size", self.max_fleet_size),
            ("max_targets", self.max_targets),
            ("max_faults", self.max_faults),
            ("max_partitions", self.max_partitions),
        ):
            if value is not None and value < 0:
                raise CurriculumError(f"stage {self.name!r}: {label} must be >= 0")
        if self.epochs is not None and self.epochs <= 0:
            raise CurriculumError(f"stage {self.name!r}: epochs must be positive")
        if self.families is not None and not self.families:
            raise CurriculumError(f"stage {self.name!r}: families must not be empty")

    # --- gating -------------------------------------------------------------------
    def bounds(self) -> dict[str, int]:
        """The declared limits, keyed by the dataset attribute they apply to."""
        values: dict[str, int] = {}
        mapping = {
            "fleet.size": self.max_fleet_size,
            "tasks.target_count": self.max_targets,
            "faults.count": self.max_faults,
            "network.partitions": self.max_partitions,
        }
        for key, value in mapping.items():
            if value is not None:
                values[key] = int(value)
        return values

    def rejection_reason(self, sample: GraphSample) -> str | None:
        """Why ``sample`` is not admitted, or ``None`` when it is."""
        if self.families is not None and sample.family and sample.family not in self.families:
            return f"family {sample.family!r} is outside this stage"
        for key, limit in self.bounds().items():
            raw = sample.attributes.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > limit:
                return f"{key}={value:g} exceeds {limit}"
        return None

    def accepts(self, sample: GraphSample) -> bool:
        return self.rejection_reason(sample) is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "limits": self.bounds(),
            "families": list(self.families) if self.families is not None else None,
            "epochs": self.epochs,
            "balance": self.balance,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StageSpec":
        limits = dict(payload.get("limits") or {})
        families = payload.get("families")
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            max_fleet_size=limits.get("fleet.size"),
            max_targets=limits.get("tasks.target_count"),
            max_faults=limits.get("faults.count"),
            max_partitions=limits.get("network.partitions"),
            families=tuple(families) if families else None,
            epochs=payload.get("epochs"),
            balance=bool(payload.get("balance", True)),
        )


def default_stages() -> tuple[StageSpec, ...]:
    """Four rungs from small missions to the full distribution.

    The ladder widens *scale* (satellites and concurrent events) and leaves the
    hazard distribution intact at every rung. Conjoining every axis would make
    the smallest stage empty on most draws — a stage that admits no episode is
    transparent but useless — while hazard variety is exactly what balanced
    sampling is for. A design that wants fault-free warm-up stages can declare
    those limits explicitly.
    """
    return (
        StageSpec(
            "foundation",
            "small fleet with a light event load",
            max_fleet_size=6,
            max_targets=3,
        ),
        StageSpec(
            "growth",
            "medium fleet and a handful of concurrent events",
            max_fleet_size=10,
            max_targets=5,
        ),
        StageSpec(
            "stress",
            "large fleet with a heavy event load",
            max_fleet_size=14,
            max_targets=6,
        ),
        StageSpec(
            "full",
            "the complete training distribution",
        ),
    )


# --- balanced sampling -----------------------------------------------------------------


def sample_key(sample: GraphSample) -> tuple[str, str, str]:
    """The stratum a sample is balanced within."""
    condition = str(sample.attributes.get("condition_class") or "unlabelled")
    fleet = fleet_bucket(sample.attributes.get("fleet.size"))
    return (condition, fleet, sample.family or "unknown")


def balanced_weights(samples: Sequence[GraphSample], *, power: float = 1.0) -> dict[tuple[str, str, str], float]:
    """Inverse-frequency weights per stratum, normalised to mean 1.

    ``power`` < 1 softens the correction (full inverse frequency can make the
    smallest stratum extremely expensive); the default is complete correction.
    Weights are clipped to ``[0.2, 5.0]`` so one rare episode cannot dominate a
    batch on its own.
    """
    if not samples:
        return {}
    counts: dict[tuple[str, str, str], int] = {}
    for sample in samples:
        key = sample_key(sample)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    strata = len(counts)
    weights = {
        key: (total / (strata * count)) ** max(0.0, power)
        for key, count in counts.items()
    }
    mean = sum(weights.values()) / max(1, len(weights))
    if mean <= 0.0:
        return {}
    return {key: min(5.0, max(0.2, value / mean)) for key, value in weights.items()}


# --- configuration ---------------------------------------------------------------------


@dataclass
class CurriculumConfig:
    curriculum_id: str
    dataset_root: str
    model: dict[str, Any] = field(
        default_factory=lambda: {"model": "gnn_gru", "hidden": 32, "layers": 2}
    )
    stages: tuple[StageSpec, ...] = field(default_factory=default_stages)
    seeds: tuple[int, ...] = (0,)
    epochs_per_stage: int = 2
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    abstain_weight: float = 1.0
    graphs_per_batch: int = 8
    steps_per_chunk: int = 4
    batches_per_epoch: int = 8
    max_sequence_samples: int = 2000
    evaluation_episodes: int = 4
    early_stopping_patience: int = 2
    stall_patience: int = 2
    selection_metric: str = "validation_loss"
    balance_power: float = 1.0
    evaluate_test: bool = True

    def __post_init__(self) -> None:
        if not self.curriculum_id:
            raise CurriculumError("curriculum_id must be non-empty")
        if not self.stages:
            raise CurriculumError("a curriculum needs at least one stage")
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise CurriculumError(f"stage names must be unique, got {names}")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise CurriculumError("seeds must be a non-empty list of non-negative integers")
        family = str(self.model.get("model", "gnn_gru"))
        if family not in MODEL_FAMILIES:
            raise CurriculumError(f"unknown model family {family!r}; known: {list(MODEL_FAMILIES)}")
        if self.epochs_per_stage <= 0:
            raise CurriculumError("epochs_per_stage must be positive")
        if self.selection_metric not in ("validation_loss", "validation_accuracy", "validation_reward"):
            raise CurriculumError(
                "selection_metric must be validation_loss, validation_accuracy or validation_reward"
            )
        if self.stall_patience < 0:
            raise CurriculumError("stall_patience must be >= 0 (0 disables the stall stop)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum_id": self.curriculum_id,
            "dataset_root": self.dataset_root,
            "model": dict(self.model),
            "stages": [stage.to_dict() for stage in self.stages],
            "seeds": list(self.seeds),
            "epochs_per_stage": self.epochs_per_stage,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "abstain_weight": self.abstain_weight,
            "graphs_per_batch": self.graphs_per_batch,
            "steps_per_chunk": self.steps_per_chunk,
            "batches_per_epoch": self.batches_per_epoch,
            "max_sequence_samples": self.max_sequence_samples,
            "evaluation_episodes": self.evaluation_episodes,
            "early_stopping_patience": self.early_stopping_patience,
            "stall_patience": self.stall_patience,
            "selection_metric": self.selection_metric,
            "balance_power": self.balance_power,
            "evaluate_test": self.evaluate_test,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CurriculumConfig":
        stages = tuple(StageSpec.from_payload(item) for item in payload.get("stages", []))
        return cls(
            curriculum_id=str(payload["curriculum_id"]),
            dataset_root=str(payload["dataset_root"]),
            model=dict(payload.get("model", {"model": "gnn_gru"})),
            stages=stages or default_stages(),
            seeds=tuple(int(seed) for seed in payload.get("seeds", (0,))),
            epochs_per_stage=int(payload.get("epochs_per_stage", 2)),
            learning_rate=float(payload.get("learning_rate", 3e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            abstain_weight=float(payload.get("abstain_weight", 1.0)),
            graphs_per_batch=int(payload.get("graphs_per_batch", 8)),
            steps_per_chunk=int(payload.get("steps_per_chunk", 4)),
            batches_per_epoch=int(payload.get("batches_per_epoch", 8)),
            max_sequence_samples=int(payload.get("max_sequence_samples", 2000)),
            evaluation_episodes=int(payload.get("evaluation_episodes", 4)),
            early_stopping_patience=int(payload.get("early_stopping_patience", 2)),
            stall_patience=int(payload.get("stall_patience", 2)),
            selection_metric=str(payload.get("selection_metric", "validation_loss")),
            balance_power=float(payload.get("balance_power", 1.0)),
            evaluate_test=bool(payload.get("evaluate_test", True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CurriculumConfig":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


# --- baselines -------------------------------------------------------------------------


class WaitPolicy(EnvPolicy):
    """Never tasks anything: the lower bound every learned policy must beat."""

    label = "baseline-wait"
    version = "1"

    def act(self, observation: Mapping[str, Any]) -> list[Action]:
        return [Action(kind=ACTION_WAIT)]

    def metadata(self) -> dict[str, Any]:
        return {"key": self.label, "kind": "baseline", "deployable": True, "version": self.version}


class FirstCandidatePolicy(EnvPolicy):
    """Picks the first masked candidate pair; a non-neural, deterministic baseline."""

    label = "baseline-first-feasible"
    version = "1"

    def act(self, observation: Mapping[str, Any]) -> list[Action]:
        pairs = list(observation["mask"]["assign"])
        if not pairs:
            return [Action(kind=ACTION_WAIT)]
        satellite, task = pairs[0]
        return [Action(
            kind="assign",
            satellite_id=str(observation["satellite_ids"][int(satellite)]),
            task_id=str(observation["task_ids"][int(task)]),
            score=0.0,
        )]

    def metadata(self) -> dict[str, Any]:
        return {"key": self.label, "kind": "baseline", "deployable": True, "version": self.version}


def baseline_policies() -> dict[str, EnvPolicy]:
    """The non-neural comparison set (deployable expert + two trivial bounds)."""
    return {
        "expert-rule": expert_factory("expert-rule"),
        "first-feasible": FirstCandidatePolicy(),
        "wait": WaitPolicy(),
    }


# --- runner ----------------------------------------------------------------------------


def _metric_direction(metric: str) -> str:
    return "higher" if metric in ("validation_accuracy", "validation_reward") else "lower"


def _best_of(history: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    values = [entry.get(metric) for entry in history if entry.get(metric) is not None]
    if not values:
        return None
    best = max(values) if _metric_direction(metric) == "higher" else min(values)
    return round(float(best), 6)


def _summarise_samples(samples: Sequence[GraphSample]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0, "episodes": 0, "families": {}, "fleet_buckets": {},
                "condition_classes": {}, "fleet_size": None, "targets": None,
                "faults": None, "partitions": None}

    def numbers(key: str) -> list[float]:
        values: list[float] = []
        for sample in samples:
            raw = sample.attributes.get(key)
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        return values

    def distribution(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"min": None, "median": None, "max": None}
        ordered = sorted(values)
        return {
            "min": round(ordered[0], 3),
            "median": round(statistics.median(ordered), 3),
            "max": round(ordered[-1], 3),
        }

    families: dict[str, int] = {}
    conditions: dict[str, int] = {}
    buckets: dict[str, int] = {}
    for sample in samples:
        families[sample.family] = families.get(sample.family, 0) + 1
        condition = str(sample.attributes.get("condition_class") or "unlabelled")
        conditions[condition] = conditions.get(condition, 0) + 1
        bucket = fleet_bucket(sample.attributes.get("fleet.size"))
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return {
        "samples": len(samples),
        "episodes": len({sample.episode_id for sample in samples}),
        "families": dict(sorted(families.items())),
        "fleet_buckets": dict(sorted(buckets.items())),
        "condition_classes": dict(sorted(conditions.items())),
        "fleet_size": distribution(numbers("fleet.size")),
        "targets": distribution(numbers("tasks.target_count")),
        "faults": distribution(numbers("faults.count")),
        "partitions": distribution(numbers("network.partitions")),
    }


def _coverage_growth(stage_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How the admitted distribution widened from stage to stage."""
    previous: set[str] = set()
    growth: list[dict[str, Any]] = []
    for stage in stage_summaries:
        families = set((stage.get("coverage") or {}).get("families", {}))
        if stage.get("samples"):
            growth.append({
                "stage": stage["stage"],
                "samples": stage["samples"],
                "new_families": sorted(families - previous),
                "cumulative_families": sorted(previous | families),
                "fleet_max": ((stage.get("coverage") or {}).get("fleet_size") or {}).get("max"),
                "targets_max": ((stage.get("coverage") or {}).get("targets") or {}).get("max"),
            })
        previous |= families
    return {"stages": growth, "families_seen": sorted(previous)}


def _checkpoint_policy(path: Path) -> Any:
    import torch

    payload = torch.load(path, weights_only=False)
    payload["path"] = str(path)
    return build_model_policy_from_checkpoint(payload)


@dataclass
class CurriculumRunner:
    """Runs every seed's ladder of stages and freezes one checkpoint."""

    config: CurriculumConfig
    output_dir: Path
    dataset: EpisodeDataset | None = None
    stages_trained: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "curriculum-manifest.json"

    # --- data -----------------------------------------------------------------------
    def load(self) -> EpisodeDataset:
        self.dataset = self.dataset or EpisodeDataset(self.config.dataset_root)
        self.train_samples = samples_from_dataset(
            self.dataset, "train", limit=self.config.max_sequence_samples
        )
        if not self.train_samples:
            raise CurriculumError(
                f"campaign {self.config.dataset_root!r} has no training decision samples"
            )
        return self.dataset

    def stage_samples(self, stage: StageSpec) -> list[GraphSample]:
        return [sample for sample in self.train_samples if stage.accepts(sample)]

    def weight_function(self, stage: StageSpec) -> Callable[[GraphSample], float] | None:
        if not stage.balance:
            return None
        weights = balanced_weights(self.stage_samples(stage), power=self.config.balance_power)
        if not weights:
            return None
        return lambda sample: weights.get(sample_key(sample), 1.0)

    # --- one seed -------------------------------------------------------------------
    def run_seed(self, seed: int) -> dict[str, Any]:
        assert self.dataset is not None
        seed_dir = self.output_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        initialized_from: Path | None = None
        reports: list[dict[str, Any]] = []
        best_value: float | None = None
        stalled = 0
        stop_reason = "completed"
        for index, stage in enumerate(self.config.stages):
            admitted = self.stage_samples(stage)
            directory = seed_dir / f"{index:02d}-{stage.name}"
            stage_started = time.perf_counter()
            history: list[dict[str, Any]] = []
            reward: dict[str, Any] | None = None
            status = "completed"
            error: dict[str, Any] | None = None
            if not admitted:
                status = "skipped: no samples admitted"
                report = {
                    "stage": stage.name,
                    "index": index,
                    "spec": stage.to_dict(),
                    "status": status,
                    "samples": 0,
                    "coverage": _summarise_samples([]),
                    "history": [],
                    "best": {},
                    "reward": None,
                    "initialized_from": str(initialized_from) if initialized_from else None,
                    "checkpoint": None,
                }
                reports.append(report)
                continue
            config = TrainingConfig(
                run_id=f"{self.config.curriculum_id}-s{seed}-{stage.name}",
                dataset_root=self.config.dataset_root,
                model=dict(self.config.model),
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                abstain_weight=self.config.abstain_weight,
                graphs_per_batch=self.config.graphs_per_batch,
                steps_per_chunk=self.config.steps_per_chunk,
                batches_per_epoch=self.config.batches_per_epoch,
                epochs=stage.epochs or self.config.epochs_per_stage,
                seed=seed * 101 + index,
                early_stopping_patience=self.config.early_stopping_patience,
                selection_metric=self.config.selection_metric,
                evaluation_episodes=self.config.evaluation_episodes,
                max_sequence_samples=self.config.max_sequence_samples,
                export=False,
                curriculum={
                    "curriculum_id": self.config.curriculum_id,
                    "stage": stage.name,
                    "index": index,
                    "of": len(self.config.stages),
                    "seed": seed,
                },
                label=f"{self.config.curriculum_id}/{stage.name}",
            )
            transfer_from = initialized_from
            trainer = Trainer(
                config,
                directory,
                dataset=self.dataset,
                sample_filter=stage.accepts,
                weight_fn=self.weight_function(stage),
                init_weights_from=transfer_from,
            )
            try:
                manifest = trainer.train()
                history = list(manifest.get("history", []))
                checkpoint = trainer.best_path if trainer.best_path.exists() else trainer.checkpoint_path
                if self.config.evaluation_episodes:
                    evaluation = trainer.evaluate(
                        "validation", limit=self.config.evaluation_episodes
                    )
                    reward = {
                        "mean_coverage": evaluation["aggregate"]["mean_coverage"],
                        "mean_reward": evaluation["aggregate"]["mean_reward"],
                        "families": {
                            name: value.get("mean_coverage")
                            for name, value in evaluation["families"].items()
                        },
                    }
                initialized_from = checkpoint
            except Exception as exc:  # a broken stage is recorded, never hidden
                status = "failed"
                error = {"type": type(exc).__name__, "message": str(exc)}
                checkpoint = None
            best = {
                metric: _best_of(history, metric)
                for metric in ("train_loss", "validation_loss", "validation_accuracy")
            }
            report = {
                "stage": stage.name,
                "index": index,
                "spec": stage.to_dict(),
                "status": status,
                "samples": len(admitted),
                "coverage": _summarise_samples(admitted),
                "history": history,
                "best": best,
                "reward": reward,
                "initialized_from": str(transfer_from) if transfer_from else None,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "wall_time_s": round(time.perf_counter() - stage_started, 4),
                "error": error,
            }
            reports.append(report)
            if status == "failed":
                stop_reason = f"stage {stage.name!r} failed"
                break
            # Stall detection uses the same validation metric the checkpoint
            # selection uses; it can stop the ladder, never rescue it.
            value = best.get(self.config.selection_metric)
            if value is None:
                continue
            if best_value is None or (
                (value > best_value) if _metric_direction(self.config.selection_metric) == "higher"
                else (value < best_value)
            ):
                best_value = float(value)
                stalled = 0
            else:
                stalled += 1
                if self.config.stall_patience and stalled >= self.config.stall_patience:
                    stop_reason = (
                        f"validation {self.config.selection_metric} did not improve for "
                        f"{stalled} consecutive stages"
                    )
                    break
        selection = self._select(reports)
        evaluation: dict[str, Any] = {}
        if selection is None:
            stop_reason = stop_reason if stop_reason != "completed" else "no stage produced a checkpoint"
        else:
            evaluation = self._evaluate(selection["checkpoint"])
        report = {
            "seed": seed,
            "stages": reports,
            "selection": selection,
            "evaluation": evaluation,
            "coverage_growth": _coverage_growth(reports),
            "stop_reason": stop_reason,
            "stages_completed": sum(1 for item in reports if item["status"] == "completed"),
            "wall_time_s": round(time.perf_counter() - started, 4),
        }
        (seed_dir / "curriculum-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8"
        )
        return report

    def _select(self, reports: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        """Best stage checkpoint by validation only — the test split is not read."""
        metric = self.config.selection_metric
        candidates = [
            report for report in reports
            if report.get("checkpoint") and (report.get("best") or {}).get(metric) is not None
        ]
        if not candidates:
            return None
        higher = _metric_direction(metric) == "higher"
        ordered = sorted(
            candidates,
            key=lambda report: (
                -float(report["best"][metric]) if higher else float(report["best"][metric]),
                int(report["index"]),
            ),
        )
        winner = ordered[0]
        return {
            "stage": winner["stage"],
            "index": winner["index"],
            "checkpoint": winner["checkpoint"],
            "metric": metric,
            "value": winner["best"][metric],
            "direction": "higher-is-better" if higher else "lower-is-better",
            "stages_considered": [report["stage"] for report in candidates],
            "frozen_before_test": True,
        }

    def _evaluate(self, checkpoint: str) -> dict[str, Any]:
        assert self.dataset is not None
        learned = _checkpoint_policy(Path(checkpoint))
        baselines = baseline_policies()
        comparison: dict[str, Any] = {}
        for split in ("validation", "test") if self.config.evaluate_test else ("validation",):
            records = self.dataset.episodes(split)
            if not records:
                continue
            policies: dict[str, Any] = {"learned": learned, **baselines}
            comparison[split] = compare_policies(policies, records)
            comparison[split]["episodes_used"] = len(records)
        return comparison

    # --- whole curriculum -----------------------------------------------------------
    def run(self) -> dict[str, Any]:
        dataset = self.load()
        started = time.perf_counter()
        seed_reports = [self.run_seed(seed) for seed in self.config.seeds]
        best_seed = self._best_seed(seed_reports)
        manifest = {
            "curriculum_version": CURRICULUM_VERSION,
            "curriculum_id": self.config.curriculum_id,
            "created_at_s": round(time.time(), 3),
            "dataset": dataset.summary(),
            "schema": {
                "observation_version": dataset.manifest["schema"]["observation_version"],
                "parameter_schema_hash": dataset.manifest["schema"]["parameter_schema_hash"],
                "campaign_id": dataset.manifest["campaign_id"],
            },
            "config": self.config.to_dict(),
            "stages": [stage.to_dict() for stage in self.config.stages],
            "seeds": seed_reports,
            "selected_seed": best_seed,
            "selection_rule": (
                "checkpoints are ranked by "
                f"{self.config.selection_metric} on validation data only; "
                "the held-out test split is evaluated after the selection is frozen"
            ),
            "seed_variation": self._seed_variation(seed_reports),
            "progression": self._progression(seed_reports),
            "wall_time_s": round(time.perf_counter() - started, 4),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=False, default=str) + "\n",
            encoding="utf-8",
        )
        return manifest

    def _best_seed(self, seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        scored = [
            report for report in seed_reports
            if report.get("evaluation", {}).get("validation", {}).get("policies", {}).get("learned")
        ]
        if not scored:
            return None
        metric = self.config.selection_metric
        higher = _metric_direction(metric) == "higher"

        def score(report: Mapping[str, Any]) -> float:
            learned = report["evaluation"]["validation"]["policies"]["learned"]["aggregate"]
            if metric == "validation_reward":
                return float(learned.get("mean_reward") or 0.0)
            if metric == "validation_accuracy":
                return float(learned.get("coverage_by_tasks") or 0.0)
            # No held-out loss is recomputed here; the frozen stage selection value
            # is the validation loss the checkpoint was chosen with.
            return float((report.get("selection") or {}).get("value") or math.inf)

        ordered = sorted(scored, key=score, reverse=higher)
        winner = ordered[0]
        return {
            "seed": winner["seed"],
            "stage": (winner.get("selection") or {}).get("stage"),
            "checkpoint": (winner.get("selection") or {}).get("checkpoint"),
            "selection_value": (winner.get("selection") or {}).get("value"),
            "metric": metric,
        }

    def _seed_variation(self, seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if len(seed_reports) < 2:
            return {"seeds": len(seed_reports), "note": "a single seed cannot estimate seed variation"}
        values: dict[str, list[float]] = {}
        for report in seed_reports:
            learned = (report.get("evaluation", {}).get("test", {})
                       .get("policies", {}).get("learned", {}).get("aggregate", {}))
            if learned:
                values.setdefault("test_mean_coverage", []).append(float(learned["mean_coverage"]))
                values.setdefault("test_mean_reward", []).append(float(learned["mean_reward"]))
            selection = (report.get("selection") or {}).get("value")
            if selection is not None:
                values.setdefault("selection_value", []).append(float(selection))
            accuracy = [
                entry.get("train_accuracy") for entry in report.get("stages", [])[-1].get("history", [])
                if entry.get("train_accuracy") is not None
            ]
            if accuracy:
                values.setdefault("final_train_accuracy", []).append(float(accuracy[-1]))
        summary = {}
        for key, entries in values.items():
            summary[key] = {
                "count": len(entries),
                "mean": round(statistics.fmean(entries), 6),
                "std": round(statistics.pstdev(entries), 6) if len(entries) > 1 else 0.0,
                "min": round(min(entries), 6),
                "max": round(max(entries), 6),
            }
        return {"seeds": len(seed_reports), "metrics": summary}

    def _progression(self, seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Stage-by-stage ladder for every seed, in the order training ran."""
        stages: list[dict[str, Any]] = []
        for report in seed_reports:
            for stage in report["stages"]:
                history = stage.get("history") or []
                stages.append({
                    "seed": report["seed"],
                    "stage": stage["stage"],
                    "status": stage["status"],
                    "samples": stage["samples"],
                    "fleet_max": ((stage.get("coverage") or {}).get("fleet_size") or {}).get("max"),
                    "targets_max": ((stage.get("coverage") or {}).get("targets") or {}).get("max"),
                    "train_loss_start": history[0].get("train_loss") if history else None,
                    "train_loss_end": history[-1].get("train_loss") if history else None,
                    "validation_loss": (stage.get("best") or {}).get("validation_loss"),
                    "validation_accuracy": (stage.get("best") or {}).get("validation_accuracy"),
                    "validation_reward": (stage.get("reward") or {}).get("mean_coverage"),
                    "initialized_from_previous": bool(stage.get("initialized_from")),
                    "wall_time_s": stage.get("wall_time_s"),
                })
        return {
            "stages": stages,
            "note": (
                "supervised loss and environment reward are recorded separately; "
                "more training is not reported as better unless validation improves"
            ),
        }


def run_curriculum(config: CurriculumConfig, output_dir: str | Path) -> dict[str, Any]:
    return CurriculumRunner(config, output_dir).run()


def curriculum_report(output_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(output_dir) / "curriculum-manifest.json").read_text(encoding="utf-8"))


__all__ = [
    "CURRICULUM_VERSION",
    "FLEET_BUCKETS",
    "CurriculumError",
    "StageSpec",
    "default_stages",
    "fleet_bucket",
    "sample_key",
    "balanced_weights",
    "CurriculumConfig",
    "CurriculumRunner",
    "WaitPolicy",
    "FirstCandidatePolicy",
    "baseline_policies",
    "run_curriculum",
    "curriculum_report",
]
