"""Day 49: a resumable, multi-seed PPO curriculum over swarm episodes.

The ladder deliberately changes *episode difficulty*, not the reward definition:
small fleets and few targets first, then larger fleets, more targets, faults and
partitions. Each stage is initialized from the previous stage's selected
checkpoint; test episodes are never used for selection. Every seed is recorded
separately so a lucky run can not be presented as the curriculum's result.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import EpisodeDataset
from .env import EpisodeSpec, SwarmEnvironment
from .ppo import PPOConfig, PPOPolicy, train_ppo

RL_CURRICULUM_VERSION = "rl-curriculum-1"


class RLCurriculumError(RuntimeError):
    """A curriculum configuration or stage is invalid."""


@dataclass
class RLStage:
    name: str
    max_fleet: int
    max_targets: int
    episodes: int = 2
    updates: int = 1
    max_steps: int = 8
    require_fault: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise RLCurriculumError("a stage needs a name")
        if self.max_fleet < 1 or self.max_targets < 1:
            raise RLCurriculumError("stage limits must be positive")
        if self.episodes < 1 or self.updates < 1 or self.max_steps < 1:
            raise RLCurriculumError("stage budgets must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_fleet": self.max_fleet,
            "max_targets": self.max_targets,
            "episodes": self.episodes,
            "updates": self.updates,
            "max_steps": self.max_steps,
            "require_fault": self.require_fault,
            "description": self.description,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RLStage":
        return cls(
            name=str(payload["name"]),
            max_fleet=int(payload.get("max_fleet", 4)),
            max_targets=int(payload.get("max_targets", 3)),
            episodes=int(payload.get("episodes", 2)),
            updates=int(payload.get("updates", 1)),
            max_steps=int(payload.get("max_steps", 8)),
            require_fault=bool(payload.get("require_fault", False)),
            description=str(payload.get("description", "")),
        )


def default_stages() -> tuple[RLStage, ...]:
    return (
        RLStage("small", 4, 3, 2, 1, 8, False, "small, well-connected, no faults"),
        RLStage("medium", 8, 5, 2, 1, 10, False, "medium fleet and task load"),
        RLStage("faulted", 8, 5, 2, 1, 10, True, "faults and degraded links"),
        RLStage("large", 12, 8, 2, 1, 12, False, "large fleet and dense task load"),
    )


@dataclass
class RLCurriculumConfig:
    run_id: str
    dataset_root: str
    stages: list[RLStage] = field(default_factory=lambda: list(default_stages()))
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    model: dict[str, Any] = field(default_factory=lambda: {"model": "gnn", "hidden": 24, "layers": 1})
    learning_rate: float = 3e-4
    gamma: float = 0.99
    max_train_episodes: int = 24
    validation_episodes: int = 4
    resume: bool = True

    def __post_init__(self) -> None:
        if not self.run_id:
            raise RLCurriculumError("run_id must be non-empty")
        if not self.stages:
            raise RLCurriculumError("a curriculum needs at least one stage")
        if not self.seeds:
            raise RLCurriculumError("a curriculum needs at least one seed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_root": self.dataset_root,
            "stages": [stage.to_dict() for stage in self.stages],
            "seeds": list(self.seeds),
            "model": dict(self.model),
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "max_train_episodes": self.max_train_episodes,
            "validation_episodes": self.validation_episodes,
            "resume": self.resume,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RLCurriculumConfig":
        stages = [RLStage.from_payload(item) for item in payload.get("stages") or []]
        return cls(
            run_id=str(payload.get("run_id", "rl-curriculum")),
            dataset_root=str(payload["dataset_root"]),
            stages=stages or list(default_stages()),
            seeds=[int(value) for value in (payload.get("seeds") or [0, 1])],
            model=dict(payload.get("model") or {"model": "gnn", "hidden": 24, "layers": 1}),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            gamma=float(payload.get("gamma", 0.99)),
            max_train_episodes=int(payload.get("max_train_episodes", 24)),
            validation_episodes=int(payload.get("validation_episodes", 4)),
            resume=bool(payload.get("resume", True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RLCurriculumConfig":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def _episode_fleet(spec: EpisodeSpec) -> int:
    return len(spec.config.get("satellites") or [])


def _episode_targets(spec: EpisodeSpec) -> int:
    return len(spec.targets)


def _episode_faults(spec: EpisodeSpec) -> int:
    return len((spec.fault_plan or {}).get("faults") or [])


def _select(
    specs: Sequence[EpisodeSpec],
    stage: RLStage,
    *,
    limit: int,
) -> list[EpisodeSpec]:
    selected = [
        spec for spec in specs
        if _episode_fleet(spec) <= stage.max_fleet
        and _episode_targets(spec) <= stage.max_targets
        and (not stage.require_fault or _episode_faults(spec) > 0)
    ]
    if not selected:
        selected = list(specs)
    return selected[:limit]


def _evaluate(policy: PPOPolicy, specs: Sequence[EpisodeSpec]) -> dict[str, Any]:
    rows = []
    for spec in specs:
        env = SwarmEnvironment(spec)
        outcome = env.rollout(policy)
        rows.append({
            "episode_id": spec.episode_id,
            "coverage": float(outcome.metrics.get("coverage", 0.0)),
            "observed": int(outcome.metrics.get("observed", 0)),
            "tasks_total": int(outcome.metrics.get("tasks_total", 0)),
            "mean_reward": float(outcome.metrics.get("mean_reward", 0.0)),
            "conflicts": int(outcome.metrics.get("conflicts", 0)),
        })
    if not rows:
        return {"episodes": 0, "mean_coverage": None, "mean_reward": None}
    return {
        "episodes": len(rows),
        "mean_coverage": round(statistics.fmean(row["coverage"] for row in rows), 6),
        "mean_reward": round(statistics.fmean(row["mean_reward"] for row in rows), 6),
        "observed": sum(row["observed"] for row in rows),
        "tasks_total": sum(row["tasks_total"] for row in rows),
        "conflicts": sum(row["conflicts"] for row in rows),
        "episodes_detail": rows,
    }


@dataclass
class RLCurriculumRunner:
    config: RLCurriculumConfig
    output_dir: Path
    dataset: EpisodeDataset | None = None
    reports: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "rl-curriculum-manifest.json"

    def load(self) -> tuple[list[EpisodeSpec], list[EpisodeSpec]]:
        self.dataset = self.dataset or EpisodeDataset(self.config.dataset_root)
        train = [
            EpisodeSpec.from_payload(row["spec"])
            for row in self.dataset.episodes("train")[: self.config.max_train_episodes]
        ]
        validation = [
            EpisodeSpec.from_payload(row["spec"])
            for row in self.dataset.episodes("validation")[: self.config.validation_episodes]
        ]
        if not train:
            raise RLCurriculumError("the campaign has no training episodes")
        return train, validation or train[: self.config.validation_episodes]

    def _previous_manifest(self) -> dict[str, Any]:
        if not self.config.resume or not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def run(self) -> dict[str, Any]:
        train, validation = self.load()
        previous = self._previous_manifest()
        completed = {
            (int(report["seed"]), str(stage["stage"])): stage
            for report in previous.get("seed_reports", [])
            for stage in report.get("stages", [])
            if stage.get("status") == "completed"
        }
        seed_reports: list[dict[str, Any]] = []
        started = time.perf_counter()
        for seed in self.config.seeds:
            stage_reports: list[dict[str, Any]] = []
            checkpoint: Path | None = None
            for stage in self.config.stages:
                key = (seed, stage.name)
                if key in completed:
                    entry = dict(completed[key])
                    stage_reports.append(entry)
                    path = entry.get("checkpoint")
                    if path and Path(path).exists():
                        checkpoint = Path(path)
                    continue
                train_specs = _select(train, stage, limit=self.config.max_train_episodes)
                validation_specs = _select(validation, stage, limit=self.config.validation_episodes)
                ppo_config = PPOConfig(
                    run_id=f"{self.config.run_id}-seed{seed}-{stage.name}",
                    dataset_root=self.config.dataset_root,
                    model=dict(self.config.model),
                    learning_rate=self.config.learning_rate,
                    gamma=self.config.gamma,
                    updates=stage.updates,
                    episodes_per_update=max(1, min(stage.episodes, len(train_specs))),
                    max_episode_steps=stage.max_steps,
                    validation_episodes=len(validation_specs),
                    evaluate_every=1,
                    seed=seed,
                    max_train_episodes=len(train_specs),
                )
                stage_dir = self.output_dir / f"seed-{seed}" / stage.name
                initialized_from_previous = checkpoint is not None
                manifest = train_ppo(
                    ppo_config,
                    stage_dir,
                    train_specs=train_specs,
                    validation_specs=validation_specs,
                    initial_checkpoint=checkpoint,
                )
                best = manifest.get("checkpoints", {}).get("best") or manifest.get("checkpoints", {}).get("last")
                if best:
                    checkpoint = Path(best)
                entry = {
                    "stage": stage.name,
                    "seed": seed,
                    "status": "completed",
                    "fleet_limit": stage.max_fleet,
                    "target_limit": stage.max_targets,
                    "train_episodes": [_episode_fleet(spec) for spec in train_specs],
                    "validation_episodes": [_episode_fleet(spec) for spec in validation_specs],
                    "checkpoint": str(checkpoint) if checkpoint else None,
                    "initialized_from_previous": initialized_from_previous,
                    "manifest": str(stage_dir / "ppo-manifest.json"),
                    "history": manifest.get("history", []),
                    "selection": {
                        "metric": "validation_mean_coverage",
                        "value": manifest.get("best_validation_coverage"),
                    },
                }
                stage_reports.append(entry)
                self.reports.append(entry)
                self._write(seed_reports + [{"seed": seed, "stages": stage_reports}])
            seed_reports.append({"seed": seed, "stages": stage_reports})
        payload = self._manifest(seed_reports, started)
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        return payload

    def _write(self, seed_reports: Sequence[Mapping[str, Any]]) -> None:
        self.manifest_path.write_text(
            json.dumps(self._manifest(seed_reports, time.perf_counter()), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    def _manifest(self, seed_reports: Sequence[Mapping[str, Any]], started: float) -> dict[str, Any]:
        scores = [
            float(stage["selection"]["value"])
            for report in seed_reports
            for stage in report.get("stages", [])
            if (stage.get("selection") or {}).get("value") is not None
        ]
        winner = None
        if seed_reports:
            for report in seed_reports:
                for stage in report.get("stages", []):
                    value = (stage.get("selection") or {}).get("value")
                    if value is not None and (winner is None or float(value) > float(winner["value"])):
                        winner = {"seed": report["seed"], "stage": stage["stage"], "value": float(value),
                                  "checkpoint": stage.get("checkpoint")}
        return {
            "version": RL_CURRICULUM_VERSION,
            "run_id": self.config.run_id,
            "config": self.config.to_dict(),
            "seed_reports": list(seed_reports),
            "summary": {
                "seeds": len(seed_reports),
                "stages_completed": sum(len(report.get("stages", [])) for report in seed_reports),
                "validation_coverage": {
                    "count": len(scores),
                    "mean": round(statistics.fmean(scores), 6) if scores else None,
                    "std": round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0,
                    "min": round(min(scores), 6) if scores else None,
                    "max": round(max(scores), 6) if scores else None,
                },
                "winner": winner,
            },
            "selection_note": (
                "the ladder selects on validation coverage; the winner was frozen "
                "before any held-out test report and seed variation is reported"
            ),
        }


def run_rl_curriculum(config: RLCurriculumConfig, output_dir: str | Path) -> dict[str, Any]:
    return RLCurriculumRunner(config, output_dir).run()


def rl_curriculum_report(output_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(output_dir) / "rl-curriculum-manifest.json").read_text(encoding="utf-8"))


__all__ = [
    "RL_CURRICULUM_VERSION",
    "RLCurriculumError",
    "RLStage",
    "default_stages",
    "RLCurriculumConfig",
    "RLCurriculumRunner",
    "run_rl_curriculum",
    "rl_curriculum_report",
]
