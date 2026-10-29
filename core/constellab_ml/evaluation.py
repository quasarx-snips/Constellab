"""Days 46/52: frozen evaluation suites, per-slice reports and policy comparison.

The evaluation contract is intentionally boring: every policy runs on the same
stored episodes, every metric is computed from the environment's own counters,
and every aggregate carries a bootstrap interval over *episodes* when the sample
is large enough. The test split is never used for selection; it is only read
here, after a checkpoint has been frozen.

Day 46 adds slices that are not a single global average: fleet-size buckets,
exact unseen fleet sizes, scenario family, condition class, held-out/OOD status
and parameter-compatibility failures. Day 52 uses the same runner to compare
policy families and shield variants on identical episodes.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import feature_schema_payload
from .batch import GraphSample, build_batch
from .compatibility import TrainingEnvelope
from .curriculum import FirstCandidatePolicy, WaitPolicy, baseline_policies
from .env import EpisodeSpec, SwarmEnvironment
from .policies import ModelPolicy, build_model_policy_from_checkpoint
from .scenarios import ScenarioError, ScenarioSampler
from .shield import SafetyShield, ShieldConfig, ShieldedPolicy

EVALUATION_VERSION = "policy-evaluation-1"
GENERALIZATION_VERSION = "generalization-suite-1"
BENCHMARK_VERSION = "policy-benchmark-1"


class EvaluationError(RuntimeError):
    """A suite, policy or report is outside its contract."""


@dataclass
class TimedPolicy:
    """Transparent timing wrapper; delegation preserves the policy contract."""

    policy: Any
    times_s: list[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        return getattr(self.policy, "label", "policy")

    def reset(self) -> None:
        self.times_s = []
        self.policy.reset()

    def act(self, observation: Mapping[str, Any]) -> Any:
        started = time.perf_counter()
        actions = self.policy.act(observation)
        self.times_s.append(time.perf_counter() - started)
        return actions

    def metadata(self) -> dict[str, Any]:
        data = self.policy.metadata() if hasattr(self.policy, "metadata") else {"key": self.label}
        return {**dict(data), "inference_ms": self.latency()["mean_ms"]}

    def latency(self) -> dict[str, Any]:
        if not self.times_s:
            return {"samples": 0, "mean_ms": None, "p95_ms": None}
        ordered = sorted(self.times_s)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
        return {
            "samples": len(ordered),
            "mean_ms": round(statistics.fmean(ordered) * 1000.0, 4),
            "p95_ms": round(p95 * 1000.0, 4),
        }


def load_policy_checkpoint(path: str | Path) -> Any:
    """Load either a supervised policy or a PPO actor-critic checkpoint."""
    import torch

    from .ppo import PPO_CHECKPOINT_VERSION, PPOPolicy

    payload = torch.load(Path(path), weights_only=False)
    if payload.get("checkpoint_version") == PPO_CHECKPOINT_VERSION:
        return PPOPolicy.load(path)
    payload["path"] = str(path)
    return build_model_policy_from_checkpoint(payload)


def baseline_set() -> dict[str, Any]:
    return {
        "expert-rule": baseline_policies()["expert-rule"],
        "first-feasible": FirstCandidatePolicy(),
        "wait": WaitPolicy(),
    }


def _record_values(record: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(record.get("values") or {})
    spec = record.get("spec") or {}
    values.setdefault("fleet.size", len(spec.get("config", {}).get("satellites", [])))
    values.setdefault("tasks.target_count", len(spec.get("targets", [])))
    values.setdefault("faults.count", len((spec.get("fault_plan") or {}).get("faults", [])))
    return values


def _fleet_bucket(size: float) -> str:
    if size <= 4:
        return "small"
    if size <= 9:
        return "medium"
    return "large"


def build_generalization_suite(
    dataset_root: str | Path,
    *,
    split: str = "test",
    limit: int | None = None,
    extra_episodes: int = 0,
    seed: int = 46,
    held_out_extra: bool = True,
) -> dict[str, Any]:
    """Assemble the frozen evaluation episodes and their slice metadata."""
    from .dataset import EpisodeDataset

    dataset = EpisodeDataset(dataset_root)
    records = list(dataset.episodes(split))
    if limit is not None:
        records = records[:limit]
    if not records:
        raise EvaluationError(f"campaign {dataset_root!r} has no {split} episodes")
    training = list(dataset.episodes("train"))
    train_fleets = {int(_record_values(row)["fleet.size"]) for row in training}
    train_values = [_record_values(row) for row in training]
    train_fleet_range = (
        (min(float(v["fleet.size"]) for v in train_values),
         max(float(v["fleet.size"]) for v in train_values))
        if train_values else (None, None)
    )
    train_task_range = (
        (min(float(v["tasks.target_count"]) for v in train_values),
         max(float(v["tasks.target_count"]) for v in train_values))
        if train_values else (None, None)
    )

    extra: list[dict[str, Any]] = []
    if extra_episodes > 0:
        sampler = ScenarioSampler(seed=seed, salt="generalization-suite")
        base = 100_000 + seed * 1_000
        for offset in range(extra_episodes * 4):
            if len(extra) >= extra_episodes:
                break
            try:
                sample = sampler.sample(
                    base + offset, split="test", held_out=held_out_extra,
                )
            except Exception:
                continue
            extra.append({
                "episode_id": sample.spec.episode_id,
                "split": "test",
                "family": sample.family,
                "condition_class": sample.condition_class,
                "values": dict(sample.values),
                "spec": sample.spec.payload(),
                "source": "sampled-held-out",
            })
    all_records = records + extra
    inventory = {
        "training_episodes": len(training),
        "training_fleets": sorted(train_fleets),
        "training_fleet_range": list(train_fleet_range),
        "training_task_range": list(train_task_range),
        "test_episodes": len(records),
        "sampled_extra": len(extra),
        "source_campaign": dataset.manifest.get("campaign_id"),
    }
    envelope = TrainingEnvelope.from_dataset(dataset_root)

    for record in all_records:
        values = _record_values(record)
        fleet = float(values.get("fleet.size") or 0.0)
        tasks = float(values.get("tasks.target_count") or 0.0)
        exact = int(fleet)
        ood = exact not in train_fleets
        if train_fleet_range[0] is not None and not (
            train_fleet_range[0] <= fleet <= train_fleet_range[1]
        ):
            ood = True
        if train_task_range[0] is not None and not (
            train_task_range[0] <= tasks <= train_task_range[1]
        ):
            ood = True
        record["_evaluation"] = {
            "fleet_bucket": _fleet_bucket(fleet),
            "fleet_size": exact,
            "task_count": int(tasks),
            "unseen_fleet": exact not in train_fleets,
            "ood": ood,
            "compatible": bool(
                (record.get("spec", {}).get("parameters", {}) or {})
                .get("compatibility", {})
                .get("compatible", True)
            ),
        }
        slices = [
            "all",
            f"family:{record.get('family') or 'unknown'}",
            f"condition:{record.get('condition_class') or 'unknown'}",
            f"fleet:{record['_evaluation']['fleet_bucket']}",
            "fleet:unseen" if record["_evaluation"]["unseen_fleet"] else "fleet:seen",
            "ood" if record["_evaluation"]["ood"] else "in-distribution",
            "parameter-compatible" if record["_evaluation"]["compatible"] else "parameter-incompatible",
        ]
        if split == "test":
            slices.append("held-out")
        record["_slices"] = slices
    slice_counts: dict[str, int] = {}
    for record in all_records:
        for name in record["_slices"]:
            slice_counts[name] = slice_counts.get(name, 0) + 1
    return {
        "version": GENERALIZATION_VERSION,
        "split": split,
        "seed": seed,
        "records": all_records,
        "inventory": inventory,
        "envelope": envelope.to_dict(),
        "slice_counts": dict(sorted(slice_counts.items())),
        "note": (
            "test episodes are read only after selection; sampled extras use the "
            "held-out ranges and a separate seed"
        ),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(fraction * (len(ordered) - 1))))]


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int = 0,
    resamples: int = 500,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap interval over episode-level values (deterministic seed)."""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "low": None, "high": None, "episodes": 0}
    if len(clean) == 1:
        return {"mean": round(clean[0], 6), "low": None, "high": None, "episodes": 1}
    import random

    rng = random.Random(seed)
    means = []
    for _ in range(max(50, resamples)):
        means.append(statistics.fmean(rng.choice(clean) for _ in range(len(clean))))
    low = _percentile(means, alpha / 2.0)
    high = _percentile(means, 1.0 - alpha / 2.0)
    return {
        "mean": round(statistics.fmean(clean), 6),
        "low": round(low, 6),
        "high": round(high, 6),
        "episodes": len(clean),
        "resamples": max(50, resamples),
    }


def _aggregate(episodes: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    coverages = [float(row["coverage"]) for row in episodes]
    rewards = [float(row["mean_reward"]) for row in episodes]
    latencies = [float(row["latency_ms"]) for row in episodes if row.get("latency_ms") is not None]
    observed = sum(int(row["observed"]) for row in episodes)
    tasks = sum(int(row["tasks_total"]) for row in episodes)
    return {
        "episodes": len(episodes),
        "coverage": bootstrap_ci(coverages, seed=seed),
        "reward": bootstrap_ci(rewards, seed=seed + 1),
        "coverage_by_tasks": round(observed / tasks, 6) if tasks else 0.0,
        "observed": observed,
        "tasks_total": tasks,
        "expired": sum(int(row["expired"]) for row in episodes),
        "conflicts": sum(int(row["conflicts"]) for row in episodes),
        "invalid_actions": sum(int(row["invalid_actions"]) for row in episodes),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 4) if latencies else None,
            "p95": round(_percentile(latencies, 0.95), 4) if latencies else None,
            "samples": len(latencies),
        },
    }


def evaluate_policy_records(
    policy: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    use_shield: bool = False,
    shield_config: Any = None,
) -> dict[str, Any]:
    """Run one policy over records, returning per-episode metrics and slices."""
    base = policy
    if use_shield:
        config = (
            ShieldConfig.from_payload(shield_config)
            if isinstance(shield_config, Mapping)
            else shield_config
        )
        base = ShieldedPolicy(policy, SafetyShield(config))
    timed = TimedPolicy(base)
    episodes: list[dict[str, Any]] = []
    for record in records:
        spec = EpisodeSpec.from_payload(record["spec"])
        environment = SwarmEnvironment(spec)
        timed.reset()
        outcome = environment.rollout(timed)
        metrics = dict(outcome.metrics)
        latency = timed.latency()
        evaluation = dict(record.get("_evaluation") or {})
        episodes.append({
            "episode_id": record.get("episode_id"),
            "family": record.get("family"),
            "condition_class": record.get("condition_class"),
            "fleet_size": evaluation.get("fleet_size"),
            "task_count": evaluation.get("task_count"),
            "ood": bool(evaluation.get("ood", False)),
            "unseen_fleet": bool(evaluation.get("unseen_fleet", False)),
            "coverage": float(metrics.get("coverage", 0.0)),
            "observed": int(metrics.get("observed", 0)),
            "tasks_total": int(metrics.get("tasks_total", 0)),
            "expired": int(metrics.get("expired", 0)),
            "conflicts": int(metrics.get("conflicts", 0)),
            "invalid_actions": int(metrics.get("invalid_actions", 0)),
            "mean_reward": float(metrics.get("mean_reward", 0.0)),
            "mean_latency_s": metrics.get("mean_latency_s"),
            "latency_ms": latency.get("mean_ms"),
            "latency_p95_ms": latency.get("p95_ms"),
            "slices": list(record.get("_slices") or ["all"]),
        })
    by_slice: dict[str, list[dict[str, Any]]] = {}
    for row in episodes:
        for name in row["slices"]:
            by_slice.setdefault(name, []).append(row)
    slices = {
        name: _aggregate(rows, seed=seed + index)
        for index, (name, rows) in enumerate(sorted(by_slice.items()))
    }
    metadata = base.metadata() if hasattr(base, "metadata") else {"key": getattr(base, "label", "policy")}
    return {
        "policy": metadata,
        "episodes": episodes,
        "slices": slices,
        "aggregate": _aggregate(episodes, seed=seed + 999),
        "shield": (
            base.summary() if isinstance(base, ShieldedPolicy)
            else None
        ),
    }


def _policies_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    baselines = baseline_set()
    requested = config.get("policies") or {}
    if isinstance(requested, Mapping):
        for name, value in requested.items():
            if isinstance(value, str):
                policies[name] = load_policy_checkpoint(value)
            elif isinstance(value, Mapping):
                kind = str(value.get("type", "checkpoint"))
                if kind == "baseline":
                    baseline_name = str(value.get("key", name))
                    if baseline_name not in baselines:
                        raise EvaluationError(f"unknown baseline {baseline_name!r}")
                    policies[name] = baselines[baseline_name]
                else:
                    policies[name] = load_policy_checkpoint(value["checkpoint"])
    return policies


def run_generalization(
    config: Mapping[str, Any],
    output: str | Path | None = None,
    *,
    policies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Day 46: evaluate frozen policies on held-out and unseen slices."""
    suite = build_generalization_suite(
        config["dataset_root"],
        split=str(config.get("split", "test")),
        limit=config.get("limit"),
        extra_episodes=int(config.get("extra_episodes", 0)),
        seed=int(config.get("seed", 46)),
    )
    selected = dict(policies or _policies_from_config(config))
    if not selected:
        selected = baseline_set()
    reports = {}
    for index, (name, policy) in enumerate(selected.items()):
        reports[name] = evaluate_policy_records(
            policy,
            suite["records"],
            seed=int(config.get("seed", 46)) + index * 101,
            use_shield=bool(config.get("shield", False)),
        )
    payload = {
        "version": GENERALIZATION_VERSION,
        "generated_at_s": round(time.time(), 3),
        "suite": {
            "split": suite["split"],
            "seed": suite["seed"],
            "inventory": suite["inventory"],
            "slice_counts": suite["slice_counts"],
            "envelope": suite["envelope"],
        },
        "policies": reports,
        "selection_note": (
            "all policies use the same episodes; test slices are never used for "
            "checkpoint selection, and OOD slices are labelled from the training envelope"
        ),
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return payload


def run_policy_benchmark(
    config: Mapping[str, Any],
    output: str | Path | None = None,
    *,
    policies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Day 52: compare policy families and shield variants on one frozen suite."""
    suite = build_generalization_suite(
        config["dataset_root"],
        split=str(config.get("split", "test")),
        limit=config.get("limit"),
        extra_episodes=int(config.get("extra_episodes", 0)),
        seed=int(config.get("seed", 52)),
    )
    selected = dict(policies or _policies_from_config(config))
    if not selected:
        selected = baseline_set()
    reports = {}
    for index, (name, policy) in enumerate(selected.items()):
        use_shield = bool(config.get("shield", False)) or name.endswith("+shield")
        shield_config = None
        entry = (config.get("policies") or {}).get(name)
        if isinstance(entry, Mapping):
            shield_config = entry.get("shield_config")
        reports[name] = evaluate_policy_records(
            policy,
            suite["records"],
            seed=int(config.get("seed", 52)) + index * 101,
            use_shield=use_shield,
            shield_config=None if shield_config is None else shield_config,
        )
    payload = {
        "version": BENCHMARK_VERSION,
        "generated_at_s": round(time.time(), 3),
        "suite": {
            "split": suite["split"],
            "seed": suite["seed"],
            "episodes": len(suite["records"]),
            "inventory": suite["inventory"],
            "slice_counts": suite["slice_counts"],
        },
        "policies": reports,
        "selection_note": (
            "paired episodes; report raw per-episode rows, intervals and inference "
            "cost together; a policy is not promoted on a single aggregate number"
        ),
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "EVALUATION_VERSION",
    "GENERALIZATION_VERSION",
    "BENCHMARK_VERSION",
    "EvaluationError",
    "TimedPolicy",
    "load_policy_checkpoint",
    "baseline_set",
    "build_generalization_suite",
    "bootstrap_ci",
    "evaluate_policy_records",
    "run_generalization",
    "run_policy_benchmark",
]
