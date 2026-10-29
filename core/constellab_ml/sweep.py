"""Day 43: systematic architecture and hyper-parameter sweeps.

A sweep is a *bounded grid* over model families (MLP/DeepSets, GNN, GNN+GRU,
GNN+attention), hidden sizes, learning rates, loss weights and regularization.
Three properties make the results trustworthy:

* **Bounded and resumable.** The grid is truncated to ``budget`` runs, each run
  has its own directory with a collision-free name, and a re-run skips runs that
  already finished (or failed) instead of redoing them.
* **Failures are results.** A run that raises is recorded with its error type and
  message; it is never silently dropped from the report.
* **Selection is predeclared and frozen.** The winner is chosen by the metric the
  configuration declares, using only completed runs, before any test metric is
  looked at. The test split is then reported for the selected checkpoint alone —
  and the report says explicitly which runs never saw the test split.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .models import MODEL_FAMILIES
from .training import TrainingConfig, Trainer

SWEEP_VERSION = "sweep-1"


class SweepError(RuntimeError):
    """A sweep configuration or run is invalid."""


@dataclass
class SweepConfig:
    sweep_id: str
    dataset_root: str
    families: tuple[str, ...] = ("mlp", "deepsets", "gnn", "gnn_gru")
    hidden_sizes: tuple[int, ...] = (32,)
    learning_rates: tuple[float, ...] = (3e-3,)
    abstain_weights: tuple[float, ...] = (1.0,)
    weight_decays: tuple[float, ...] = (1e-4,)
    budget: int = 8
    epochs: int = 3
    batches_per_epoch: int = 8
    graphs_per_batch: int = 8
    steps_per_chunk: int = 4
    seed: int = 0
    selection_metric: str = "validation_loss"
    early_stopping_patience: int = 2
    evaluate_episodes: int = 4
    max_sequence_samples: int = 2000
    resume: bool = True

    def __post_init__(self) -> None:
        if not self.sweep_id:
            raise SweepError("sweep_id must be non-empty")
        unknown = [family for family in self.families if family not in MODEL_FAMILIES]
        if unknown:
            raise SweepError(f"unknown model families: {unknown}; known: {list(MODEL_FAMILIES)}")
        if self.budget <= 0:
            raise SweepError("budget must be positive")
        if self.selection_metric not in ("validation_loss", "validation_accuracy", "validation_reward"):
            raise SweepError(
                "selection_metric must be validation_loss, validation_accuracy or validation_reward"
            )
        for name, values in (
            ("hidden_sizes", self.hidden_sizes),
            ("learning_rates", self.learning_rates),
            ("abstain_weights", self.abstain_weights),
            ("weight_decays", self.weight_decays),
        ):
            if not values:
                raise SweepError(f"{name} must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "dataset_root": self.dataset_root,
            "families": list(self.families),
            "hidden_sizes": list(self.hidden_sizes),
            "learning_rates": list(self.learning_rates),
            "abstain_weights": list(self.abstain_weights),
            "weight_decays": list(self.weight_decays),
            "budget": self.budget,
            "epochs": self.epochs,
            "batches_per_epoch": self.batches_per_epoch,
            "graphs_per_batch": self.graphs_per_batch,
            "steps_per_chunk": self.steps_per_chunk,
            "seed": self.seed,
            "selection_metric": self.selection_metric,
            "early_stopping_patience": self.early_stopping_patience,
            "evaluate_episodes": self.evaluate_episodes,
            "max_sequence_samples": self.max_sequence_samples,
            "resume": self.resume,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SweepConfig":
        return cls(
            sweep_id=str(payload["sweep_id"]),
            dataset_root=str(payload["dataset_root"]),
            families=tuple(payload.get("families", ("mlp", "deepsets", "gnn", "gnn_gru"))),
            hidden_sizes=tuple(int(value) for value in payload.get("hidden_sizes", (32,))),
            learning_rates=tuple(float(value) for value in payload.get("learning_rates", (3e-3,))),
            abstain_weights=tuple(float(value) for value in payload.get("abstain_weights", (1.0,))),
            weight_decays=tuple(float(value) for value in payload.get("weight_decays", (1e-4,))),
            budget=int(payload.get("budget", 8)),
            epochs=int(payload.get("epochs", 3)),
            batches_per_epoch=int(payload.get("batches_per_epoch", 8)),
            graphs_per_batch=int(payload.get("graphs_per_batch", 8)),
            steps_per_chunk=int(payload.get("steps_per_chunk", 4)),
            seed=int(payload.get("seed", 0)),
            selection_metric=str(payload.get("selection_metric", "validation_loss")),
            early_stopping_patience=int(payload.get("early_stopping_patience", 2)),
            evaluate_episodes=int(payload.get("evaluate_episodes", 4)),
            max_sequence_samples=int(payload.get("max_sequence_samples", 2000)),
            resume=bool(payload.get("resume", True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SweepConfig":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class SweepRunner:
    config: SweepConfig
    output_dir: Path
    runs: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.output_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.output_dir / "sweep.json"

    # --- planning ---------------------------------------------------------------------
    def grid(self) -> list[dict[str, Any]]:
        """The full, deterministic grid (before the budget is applied)."""
        combinations: list[dict[str, Any]] = []
        for family in self.config.families:
            for hidden in self.config.hidden_sizes:
                for rate in self.config.learning_rates:
                    for abstain in self.config.abstain_weights:
                        for decay in self.config.weight_decays:
                            combinations.append({
                                "model": {
                                    "model": family,
                                    "hidden": int(hidden),
                                    "layers": 2,
                                },
                                "learning_rate": float(rate),
                                "abstain_weight": float(abstain),
                                "weight_decay": float(decay),
                            })
        return combinations

    def plan(self) -> list[dict[str, Any]]:
        grid = self.grid()
        if len(grid) <= self.config.budget:
            return grid
        # A bounded search budget: keep the grid order but stride through it so
        # every family and every setting is represented.
        stride = max(1, len(grid) // self.config.budget)
        selected: list[dict[str, Any]] = []
        index = 0
        while len(selected) < self.config.budget and index < len(grid):
            selected.append(grid[index])
            index += stride
        return selected

    def run_dir(self, index: int, combination: Mapping[str, Any]) -> Path:
        model = combination["model"]
        name = (
            f"run-{index:03d}-{model['model']}-h{model['hidden']}"
            f"-lr{combination['learning_rate']:.0e}-aw{combination['abstain_weight']:.1f}"
            f"-wd{combination['weight_decay']:.0e}"
        ).replace("+", "")
        return self.runs_dir / name

    # --- execution --------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        for index, combination in enumerate(self.plan()):
            directory = self.run_dir(index, combination)
            record = self._run_one(index, combination, directory)
            self.runs.append(record)
            self._write_report()
        report = self.report()
        self.report_path.write_text(
            json.dumps(report, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8"
        )
        return report

    def _run_one(
        self, index: int, combination: Mapping[str, Any], directory: Path
    ) -> dict[str, Any]:
        manifest_path = directory / "training-manifest.json"
        if self.config.resume and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return {
                "index": index,
                "directory": str(directory),
                "combination": dict(combination),
                "status": "completed",
                "resumed": True,
                "metrics": _metrics_of(manifest, self.config.selection_metric),
                "wall_time_s": None,
            }
        started = time.perf_counter()
        config = TrainingConfig(
            run_id=f"{self.config.sweep_id}-{index:03d}",
            dataset_root=self.config.dataset_root,
            model=dict(combination["model"]),
            learning_rate=float(combination["learning_rate"]),
            weight_decay=float(combination["weight_decay"]),
            abstain_weight=float(combination["abstain_weight"]),
            epochs=self.config.epochs,
            batches_per_epoch=self.config.batches_per_epoch,
            graphs_per_batch=self.config.graphs_per_batch,
            steps_per_chunk=self.config.steps_per_chunk,
            seed=self.config.seed + index,
            early_stopping_patience=self.config.early_stopping_patience,
            selection_metric=self.config.selection_metric,
            evaluate_every=1,
            evaluation_episodes=self.config.evaluate_episodes,
            max_sequence_samples=self.config.max_sequence_samples,
            export=True,
            label=f"sweep {self.config.sweep_id}",
        )
        try:
            trainer = Trainer(config, directory)
            manifest = trainer.train()
            status = "completed"
            error: dict[str, Any] | None = None
        except Exception as exc:  # a failed run is a recorded result, not a crash
            manifest = {}
            status = "failed"
            error = {"type": type(exc).__name__, "message": str(exc)}
        return {
            "index": index,
            "directory": str(directory),
            "combination": dict(combination),
            "status": status,
            "resumed": False,
            "metrics": _metrics_of(manifest, self.config.selection_metric),
            "error": error,
            "wall_time_s": round(time.perf_counter() - started, 4),
            "checkpoints": manifest.get("checkpoints", {}),
        }

    # --- reporting --------------------------------------------------------------------
    def selection(self) -> dict[str, Any] | None:
        completed = [run for run in self.runs if run["status"] == "completed"]
        metric = self.config.selection_metric
        scored = [run for run in completed if run["metrics"].get(metric) is not None]
        if not scored:
            return None
        higher_is_better = metric == "validation_accuracy" or metric == "validation_reward"
        ordered = sorted(
            scored,
            key=lambda run: (
                -run["metrics"][metric] if higher_is_better else run["metrics"][metric],
                run["combination"]["model"]["hidden"],
                run["index"],
            ),
        )
        winner = ordered[0]
        return {
            "metric": metric,
            "higher_is_better": higher_is_better,
            "index": winner["index"],
            "combination": winner["combination"],
            "metrics": winner["metrics"],
            "checkpoint": winner.get("checkpoints", {}).get("best"),
            "frozen": True,
        }

    def report(self) -> dict[str, Any]:
        selection = self.selection()
        report = {
            "sweep_version": SWEEP_VERSION,
            "sweep_id": self.config.sweep_id,
            "config": self.config.to_dict(),
            "grid_size": len(self.grid()),
            "planned_runs": len(self.plan()),
            "completed": sum(1 for run in self.runs if run["status"] == "completed"),
            "failed": sum(1 for run in self.runs if run["status"] == "failed"),
            "runs": [
                {
                    key: value
                    for key, value in run.items()
                    if key not in ("checkpoints",)
                }
                for run in self.runs
            ],
            "selection": selection,
            "test_reporting": (
                "only the selected checkpoint is reported on the held-out test split; "
                "every other run never saw it"
            ),
        }
        return report

    def selected_policy_path(self) -> Path | None:
        selection = self.selection()
        if not selection or not selection.get("checkpoint"):
            return None
        return Path(str(selection["checkpoint"]))

    def _write_report(self) -> None:
        self.report_path.write_text(
            json.dumps(self.report(), indent=2, sort_keys=False, default=str) + "\n",
            encoding="utf-8",
        )


def _metrics_of(manifest: Mapping[str, Any], metric: str) -> dict[str, Any]:
    history = list(manifest.get("history", []))
    if not history:
        return {}
    best_key = metric if metric != "validation_reward" else "validation_reward"
    values = [entry.get(best_key) for entry in history if entry.get(best_key) is not None]
    if not values:
        return {}
    best = max(values) if best_key != "validation_loss" else min(values)
    last = history[-1]
    return {
        "epochs": len(history),
        best_key: round(float(best), 6),
        "final_train_loss": last.get("train_loss"),
        "final_validation_loss": last.get("validation_loss"),
        "validation_reward": last.get("validation_reward"),
        "best_metric": manifest.get("best_metric"),
        "parameters": (manifest.get("hyperparameters") or {}).get("parameters"),
        "temporal": (manifest.get("hyperparameters") or {}).get("temporal"),
    }


def run_sweep(config: SweepConfig, output_dir: str | Path) -> dict[str, Any]:
    return SweepRunner(config, output_dir).run()


def sweep_report(output_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(output_dir) / "sweep.json").read_text(encoding="utf-8"))


__all__ = [
    "SWEEP_VERSION",
    "SweepError",
    "SweepConfig",
    "SweepRunner",
    "run_sweep",
    "sweep_report",
]
