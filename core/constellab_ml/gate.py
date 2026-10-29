"""Day 56: the G3 trained-swarm-policy gate.

G3 is deliberately not "training finished". It is a promotion decision made
from saved artefacts, with the acceptance criteria declared *before* the numbers
are read:

* the checkpoint must match its manifest (schema, versions, weight hash);
* the policy must beat (or tie) the strongest non-learning baseline on the
  frozen held-out suite, without a catastrophic OOD slice;
* the safety shield suite must pass every boundary case, including a hostile
  policy that tries to switch it off;
* a compatibility suite must reproduce its declared fallbacks;
* the benchmark, ablation and multi-seed RL campaign evidence must exist and be
  internally consistent;
* two clean re-executions of the same seeds must produce the same trace hash.

Every criterion carries the measured value beside its threshold. A missing or
malformed artefact is reported as ``blocked`` rather than silently passing, and
the strongest simpler baseline is named as the fallback candidate.

The bundle is data, not code::

    {
      "gate_id": "g3-demo",
      "candidate": "learned",
      "checkpoint": "runs/.../checkpoint-best.pt",
      "manifest": "runs/.../model-manifest.json",
      "campaign": "data/ml/demo",
      "evidence": {
        "generalization": "benchmarks/generalization.json",
        "benchmark": "benchmarks/policy-comparison.json",
        "shield": "benchmarks/shield-suite.json",
        "compatibility": "benchmarks/compatibility.json",
        "ablation": "benchmarks/ablations.json",
        "rl_curriculum": "runs/rl/rl-curriculum-manifest.json",
        "ranker": "runs/ranker/ranker-manifest.json"
      },
      "criteria": {"max_coverage_drop": 0.02}
    }
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

G3_VERSION = "g3-gate-1"


class GateError(RuntimeError):
    """The gate bundle or an evidence file violates its contract."""


@dataclass(frozen=True)
class GateCriteria:
    """Declared thresholds. Change them in the bundle, never after the fact."""

    max_coverage_drop: float = 0.02
    min_ood_coverage: float = 0.20
    max_episode_conflicts: int = 0
    max_invalid_actions: int = 0
    max_p95_latency_ms: float = 250.0
    max_shield_rejection_rate: float = 0.35
    min_generalization_episodes: int = 4
    min_benchmark_policies: int = 3
    min_ablation_variants: int = 4
    min_rl_seeds: int = 2
    min_shield_cases: int = 10
    replay_episodes: int = 2
    require_ablation: bool = True
    require_rl: bool = True
    require_ranker: bool = False

    def __post_init__(self) -> None:
        if self.max_coverage_drop < 0.0:
            raise GateError("max_coverage_drop must be non-negative")
        if not 0.0 <= self.min_ood_coverage <= 1.0:
            raise GateError("min_ood_coverage must be within [0, 1]")
        if self.replay_episodes < 1:
            raise GateError("replay_episodes must be at least one")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "GateCriteria":
        return cls(**{key: payload[key] for key in payload or {} if key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise GateError(f"{label} evidence is missing: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"{label} evidence is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateError(f"{label} evidence must be a JSON object")
    return payload


def _criterion(
    key: str,
    title: str,
    *,
    required: bool = True,
    passed: bool = False,
    reasons: Sequence[str] = (),
    measured: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "required": required,
        "passed": bool(passed),
        "reasons": [str(reason) for reason in reasons],
        "measured": dict(measured or {}),
        "thresholds": dict(thresholds or {}),
    }


def _mean(value: Any) -> float | None:
    if isinstance(value, Mapping):
        candidate = value.get("mean")
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _coverage(entry: Mapping[str, Any]) -> float | None:
    return _mean((entry.get("aggregate") or {}).get("coverage"))


def _slice_coverage(entry: Mapping[str, Any], name: str) -> float | None:
    slices = entry.get("slices") or {}
    if name not in slices:
        return None
    return _mean((slices[name] or {}).get("coverage"))


def _baseline_ranking(policies: Mapping[str, Any], candidate: str) -> list[tuple[str, float]]:
    ranked = []
    for name, entry in policies.items():
        if name == candidate or name.endswith("+shield"):
            continue
        coverage = _coverage(entry)
        if coverage is not None:
            ranked.append((name, coverage))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _check_artifact(
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    checkpoint_payload: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    thresholds = {"manifest": "required", "checkpoint_hash": "must match"}
    reasons: list[str] = []
    path = Path(str(bundle.get("checkpoint", "")))
    if not str(bundle.get("checkpoint", "")):
        reasons.append("bundle does not name a checkpoint")
    elif not path.exists():
        reasons.append(f"checkpoint is missing: {path}")
    if manifest is None:
        reasons.append("model manifest is missing or unreadable")
    if checkpoint_payload is None:
        reasons.append("checkpoint could not be loaded")
    if manifest is not None and checkpoint_payload is not None:
        from .compatibility import ModelManifest

        entry = ModelManifest.from_dict(manifest)
        reasons.extend(entry.verify_checkpoint(checkpoint_payload))
        if not entry.envelope.ranges:
            reasons.append("manifest has an empty accepted envelope")
        if not entry.feature_schema_hash:
            reasons.append("manifest has no feature schema hash")
    measured = {
        "checkpoint": str(path) if str(path) else None,
        "checkpoint_sha256": _sha256(path) if path.is_file() else None,
        "manifest_checkpoint_hash": (manifest or {}).get("checkpoint_hash"),
        "model_family": (manifest or {}).get("model_family"),
        "trained_campaign": (manifest or {}).get("trained_campaign"),
    }
    return _criterion(
        "artifact", "Checkpoint matches its promotion manifest",
        passed=not reasons, reasons=reasons, measured=measured, thresholds=thresholds,
    ), (dict(manifest) if manifest is not None else None)


def _check_generalization(
    report: Mapping[str, Any] | None,
    candidate: str,
    criteria: GateCriteria,
) -> dict[str, Any]:
    reasons: list[str] = []
    if report is None:
        return _criterion(
            "generalization", "Held-out and OOD generalization",
            passed=False, reasons=["generalization evidence is missing"],
            thresholds={"max_coverage_drop": criteria.max_coverage_drop},
        )
    policies = report.get("policies") or {}
    suite = report.get("suite") or {}
    inventory = suite.get("inventory") or {}
    episodes = int(
        suite.get("episodes")
        or (int(inventory.get("test_episodes") or 0) + int(inventory.get("sampled_extra") or 0))
    )
    if candidate not in policies:
        reasons.append(f"candidate {candidate!r} is absent from the report")
    if episodes < criteria.min_generalization_episodes:
        reasons.append(f"{episodes} episodes is below the {criteria.min_generalization_episodes} minimum")
    entry = policies.get(candidate) or {}
    coverage = _coverage(entry)
    baselines = _baseline_ranking(policies, candidate)
    best_name, best_coverage = (baselines[0] if baselines else (None, None))
    delta = coverage - best_coverage if coverage is not None and best_coverage is not None else None
    if coverage is None:
        reasons.append("candidate coverage is missing")
    if best_coverage is None:
        reasons.append("no baseline coverage to compare against")
    elif delta is not None and delta < -criteria.max_coverage_drop:
        reasons.append(
            f"candidate coverage {coverage:.3f} trails {best_name} {best_coverage:.3f} "
            f"by more than {criteria.max_coverage_drop:.3f}"
        )
    ood = _slice_coverage(entry, "ood")
    if ood is None:
        reasons.append("no out-of-distribution slice was reported")
    elif ood < criteria.min_ood_coverage:
        reasons.append(f"OOD coverage {ood:.3f} is below {criteria.min_ood_coverage:.3f}")
    aggregate = entry.get("aggregate") or {}
    conflicts = int(aggregate.get("conflicts") or 0)
    invalid = int(aggregate.get("invalid_actions") or 0)
    if conflicts > criteria.max_episode_conflicts:
        reasons.append(f"{conflicts} duplicate assignments exceeds {criteria.max_episode_conflicts}")
    if invalid > criteria.max_invalid_actions:
        reasons.append(f"{invalid} invalid actions exceeds {criteria.max_invalid_actions}")
    latency = aggregate.get("latency_ms") or {}
    p95_ms = latency.get("p95") if isinstance(latency, Mapping) else None
    if isinstance(p95_ms, (int, float)) and p95_ms > criteria.max_p95_latency_ms:
        reasons.append(f"p95 inference {p95_ms:.1f} ms exceeds {criteria.max_p95_latency_ms:.1f} ms")
    measured = {
        "candidate_coverage": coverage,
        "strongest_baseline": best_name,
        "baseline_coverage": best_coverage,
        "coverage_delta": round(delta, 6) if delta is not None else None,
        "ood_coverage": ood,
        "episodes": episodes,
        "conflicts": conflicts,
        "invalid_actions": invalid,
        "p95_latency_ms": p95_ms,
        "policy_count": len(policies),
    }
    return _criterion(
        "generalization", "Held-out and OOD generalization",
        passed=not reasons, reasons=reasons, measured=measured,
        thresholds={
            "max_coverage_drop": criteria.max_coverage_drop,
            "min_ood_coverage": criteria.min_ood_coverage,
            "min_episodes": criteria.min_generalization_episodes,
            "max_conflicts": criteria.max_episode_conflicts,
            "max_invalid_actions": criteria.max_invalid_actions,
            "max_p95_latency_ms": criteria.max_p95_latency_ms,
        },
    )


def _check_benchmark(
    report: Mapping[str, Any] | None,
    candidate: str,
    criteria: GateCriteria,
) -> dict[str, Any]:
    reasons: list[str] = []
    if report is None:
        return _criterion(
            "benchmark", "Paired policy comparison",
            passed=False, reasons=["benchmark evidence is missing"],
        )
    policies = report.get("policies") or {}
    episodes = int((report.get("suite") or {}).get("episodes") or 0)
    entry_name = candidate if candidate in policies else f"{candidate}+shield"
    if entry_name not in policies:
        reasons.append(f"candidate {candidate!r} is absent from the comparison")
    if len(policies) < criteria.min_benchmark_policies:
        reasons.append(f"{len(policies)} policies is below the {criteria.min_benchmark_policies} minimum")
    entry = policies.get(entry_name) or {}
    coverage = _coverage(entry)
    baselines = _baseline_ranking(policies, candidate)
    best_name, best_coverage = (baselines[0] if baselines else (None, None))
    delta = coverage - best_coverage if coverage is not None and best_coverage is not None else None
    if delta is not None and delta < -criteria.max_coverage_drop:
        reasons.append(f"benchmark coverage trails {best_name} by {-delta:.3f}")
    shield = entry.get("shield") or {}
    proposed = int(shield.get("proposed") or 0)
    rejected = int(shield.get("rejected") or 0)
    rejection_rate = rejected / proposed if proposed else None
    if shield and rejection_rate is not None and rejection_rate > criteria.max_shield_rejection_rate:
        reasons.append(
            f"shield rejection {rejection_rate:.2%} exceeds "
            f"{criteria.max_shield_rejection_rate:.0%}"
        )
    if not shield:
        reasons.append("candidate was not evaluated behind the shield")
    measured = {
        "episodes": episodes,
        "policies": len(policies),
        "entry": entry_name,
        "coverage": coverage,
        "strongest_baseline": best_name,
        "baseline_coverage": best_coverage,
        "shield_rejection_rate": round(rejection_rate, 6) if rejection_rate is not None else None,
    }
    return _criterion(
        "benchmark", "Paired policy comparison",
        passed=not reasons, reasons=reasons, measured=measured,
        thresholds={
            "min_policies": criteria.min_benchmark_policies,
            "max_shield_rejection_rate": criteria.max_shield_rejection_rate,
        },
    )


def _check_shield(report: Mapping[str, Any] | None, criteria: GateCriteria) -> dict[str, Any]:
    reasons: list[str] = []
    if report is None:
        return _criterion(
            "shield", "Independent action shield boundary suite",
            passed=False, reasons=["shield suite evidence is missing"],
        )
    summary = report.get("summary") or {}
    cases = int(summary.get("cases") or len(report.get("cases") or []))
    failed = list(summary.get("failed") or [])
    if not report.get("passed", False):
        reasons.append(f"shield cases failed: {', '.join(failed) or 'unnamed'}")
    if cases < criteria.min_shield_cases:
        reasons.append(f"{cases} cases is below the {criteria.min_shield_cases} minimum")
    if summary.get("disable_via_policy_output") is not False:
        reasons.append("shield does not prove it is independent of policy output")
    if not summary.get("fallback"):
        reasons.append("shield documents no fallback")
    measured = {
        "cases": cases,
        "failed": failed,
        "reason_codes_covered": len(summary.get("reason_codes_covered") or []),
        "reason_codes_missing": list(summary.get("reason_codes_missing") or []),
        "fallback": summary.get("fallback"),
    }
    return _criterion(
        "shield", "Independent action shield boundary suite",
        passed=not reasons, reasons=reasons, measured=measured,
        thresholds={"min_cases": criteria.min_shield_cases, "fallback": "declared"},
    )


def _check_compatibility(report: Mapping[str, Any] | None, manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    if report is None:
        return _criterion(
            "compatibility", "Schema compatibility and safe fallback",
            passed=False, reasons=["compatibility suite evidence is missing"],
        )
    if not report.get("passed", False):
        reasons.append(f"compatibility cases failed: {', '.join(report.get('failed') or [])}")
    counts = report.get("counts") or {}
    if not counts.get("incompatible"):
        reasons.append("no incompatible case was exercised")
    if not counts.get("out-of-distribution"):
        reasons.append("no out-of-distribution case was exercised")
    envelope = report.get("accepted_envelope") or {}
    if not envelope.get("ranges"):
        reasons.append("accepted envelope is empty")
    suite_manifest = report.get("manifest") or {}
    if manifest and suite_manifest.get("feature_schema_hash") != manifest.get("feature_schema_hash"):
        reasons.append("compatibility suite used a different feature schema than the gate manifest")
    if suite_manifest.get("parameter_schema_hash") != (manifest or {}).get("parameter_schema_hash"):
        reasons.append("compatibility suite used a different parameter schema than the gate manifest")
    cases = report.get("cases") or []
    fallbacks = {str((case.get("report") or {}).get("fallback")) for case in cases}
    if not {"baseline"} <= fallbacks:
        reasons.append("suite did not demonstrate the baseline fallback")
    measured = {
        "cases": len(cases),
        "counts": counts,
        "envelope_ranges": sorted(envelope.get("ranges") or {}),
        "fallbacks": sorted(fallbacks),
    }
    return _criterion(
        "compatibility", "Schema compatibility and safe fallback",
        passed=not reasons, reasons=reasons, measured=measured,
        thresholds={"incompatible_and_ood_cases": "required", "fallback": "baseline"},
    )


def _check_ablation(
    report: Mapping[str, Any] | None,
    criteria: GateCriteria,
    *,
    required: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if report is None:
        return _criterion(
            "ablation", "Reward/feature/algorithm ablation table",
            required=required, passed=not required,
            reasons=["ablation evidence is missing"],
        )
    variants = report.get("variants") or {}
    if len(variants) < criteria.min_ablation_variants:
        reasons.append(f"{len(variants)} variants is below the {criteria.min_ablation_variants} minimum")
    full = variants.get("full")
    if full is None:
        reasons.append("no full variant is reported")
    deployable = [
        name for name, row in variants.items()
        if not bool((row.get("variant") or {}).get("diagnostic", True))
    ]
    if deployable != ["full"]:
        reasons.append(f"deployable variants must be exactly ['full'], found {deployable}")
    if full is not None and not (full.get("variant") or {}).get("shield"):
        reasons.append("the full variant was measured without the shield")
    if "no-shield" in variants and not bool((variants["no-shield"].get("variant") or {}).get("diagnostic")):
        reasons.append("the no-shield diagnostic is presented as deployable")
    measured = {
        "variants": sorted(variants),
        "deployable": deployable,
        "coverage": {
            name: (row.get("metrics") or {}).get("mean_coverage")
            for name, row in sorted(variants.items())
        },
    }
    return _criterion(
        "ablation", "Reward/feature/algorithm ablation table",
        required=required, passed=not reasons, reasons=reasons, measured=measured,
        thresholds={"min_variants": criteria.min_ablation_variants, "deployable": ["full"]},
    )


def _check_rl(
    report: Mapping[str, Any] | None,
    criteria: GateCriteria,
    *,
    required: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if report is None:
        return _criterion(
            "rl_curriculum", "Multi-seed decentralized RL campaign",
            required=required, passed=not required,
            reasons=["RL curriculum evidence is missing"],
        )
    seeds = list(report.get("seed_reports") or [])
    if len(seeds) < criteria.min_rl_seeds:
        reasons.append(f"{len(seeds)} seeds is below the {criteria.min_rl_seeds} minimum")
    metrics = [
        str((stage.get("selection") or {}).get("metric", ""))
        for seed_report in seeds
        for stage in (seed_report.get("stages") or [])
    ]
    metric = metrics[0] if metrics else ""
    if not metrics or any("validation" not in item for item in metrics):
        reasons.append(f"selection metrics {sorted(set(metrics))} are not validation-based")
    for seed_report in seeds:
        stages = list(seed_report.get("stages") or [])
        if not stages:
            reasons.append(f"seed {seed_report.get('seed')} has no completed stages")
            continue
        if any(stage.get("status") != "completed" for stage in stages):
            reasons.append(f"seed {seed_report.get('seed')} has an incomplete stage")
        if not any(stage.get("checkpoint") for stage in stages):
            reasons.append(f"seed {seed_report.get('seed')} saved no checkpoint")
    winner = (report.get("summary") or {}).get("winner")
    if winner is None:
        reasons.append("no winner was frozen by validation coverage")
    measured = {
        "seeds": len(seeds),
        "stages": sum(len(seed.get("stages") or []) for seed in seeds),
        "selection_metric": metric,
        "winner": winner,
    }
    return _criterion(
        "rl_curriculum", "Multi-seed decentralized RL campaign",
        required=required, passed=not reasons, reasons=reasons, measured=measured,
        thresholds={"min_seeds": criteria.min_rl_seeds, "selection": "validation"},
    )


def _check_ranker(report: Mapping[str, Any] | None, *, required: bool) -> dict[str, Any]:
    if report is None:
        return _criterion(
            "ranker", "Adversarial ranking evidence (adopted, not required for G3)",
            required=required, passed=not required,
            reasons=["no ranker evidence supplied"],
        )
    metrics = report.get("metrics") or {}
    auc = metrics.get("validation_auc")
    reasons: list[str] = []
    if auc is None:
        reasons.append("held-out ranking AUC is missing")
    if int(report.get("records") or 0) <= 0:
        reasons.append("ranker dataset is empty")
    if not report.get("split_note"):
        reasons.append("ranker split note is missing")
    return _criterion(
        "ranker", "Adversarial ranking evidence (adopted, not required for G3)",
        required=required, passed=not reasons, reasons=reasons,
        measured={
            "records": report.get("records"),
            "validation_auc": auc,
            "split_note": report.get("split_note"),
        },
        thresholds={"held_out_auc": "reported"},
    )


def _check_replay(
    bundle: Mapping[str, Any],
    criteria: GateCriteria,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Re-execute the frozen policy twice and compare deterministic trace hashes."""
    reasons: list[str] = []
    if not enabled:
        return _criterion(
            "replay", "Deterministic re-execution of saved seeds",
            passed=False, reasons=["replay was disabled for this run"],
        )
    campaign = str(bundle.get("campaign", ""))
    checkpoint = str(bundle.get("checkpoint", ""))
    if not campaign or not Path(campaign).exists():
        return _criterion(
            "replay", "Deterministic re-execution of saved seeds",
            passed=False, reasons=[f"campaign is missing: {campaign!r}"],
        )
    from .dataset import EpisodeDataset
    from .env import EpisodeSpec, SwarmEnvironment
    from .evaluation import load_policy_checkpoint

    records = list(EpisodeDataset(campaign).episodes("test"))[: criteria.replay_episodes]
    if not records:
        return _criterion(
            "replay", "Deterministic re-execution of saved seeds",
            passed=False, reasons=["campaign has no test episodes"],
        )
    runs: list[dict[str, Any]] = []
    for record in records:
        spec = EpisodeSpec.from_payload(record["spec"])
        hashes: list[str] = []
        actions: list[list[dict[str, Any]]] = []
        for _ in range(2):
            environment = SwarmEnvironment(spec)
            policy = load_policy_checkpoint(checkpoint)
            outcome = environment.rollout(policy)
            hashes.append(outcome.trace_hash)
            actions.append([
                {"actions": list(step.get("actions") or [])}
                for step in (outcome.records or [])
            ])
        match = len(set(hashes)) == 1 and actions[0] == actions[1]
        if not match:
            reasons.append(f"episode {record.get('episode_id')} did not replay deterministically")
        runs.append({
            "episode_id": record.get("episode_id"),
            "trace_hashes": hashes,
            "match": match,
        })
    return _criterion(
        "replay", "Deterministic re-execution of saved seeds",
        passed=not reasons, reasons=reasons,
        measured={"episodes": runs},
        thresholds={"replay_episodes": criteria.replay_episodes},
    )


def run_gate(
    bundle: Mapping[str, Any],
    *,
    replay: bool = True,
) -> dict[str, Any]:
    """Evaluate a gate bundle and return a promotion decision plus evidence."""
    if not isinstance(bundle, Mapping):
        raise GateError("the gate bundle must be an object")
    candidate = str(bundle.get("candidate") or "learned")
    criteria = GateCriteria.from_payload(bundle.get("criteria"))
    evidence_paths = dict(bundle.get("evidence") or {})
    loaded: dict[str, dict[str, Any] | None] = {}
    evidence_hashes: dict[str, Any] = {}
    for key, path in evidence_paths.items():
        try:
            loaded[key] = _read_json(path, key)
        except GateError:
            loaded[key] = None
        target = Path(str(path))
        evidence_hashes[key] = {
            "path": str(path),
            "sha256": _sha256(target) if target.is_file() else None,
        }

    manifest = loaded.get("manifest")
    manifest_path = str(bundle.get("manifest", ""))
    if manifest is None and manifest_path:
        try:
            manifest = _read_json(manifest_path, "manifest")
        except GateError:
            manifest = None
    checkpoint_payload = None
    checkpoint_path = str(bundle.get("checkpoint", ""))
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            import torch

            checkpoint_payload = torch.load(Path(checkpoint_path), weights_only=False)
        except Exception:
            checkpoint_payload = None

    artifact, manifest = _check_artifact(bundle, manifest, checkpoint_payload)
    checks = [
        artifact,
        _check_generalization(loaded.get("generalization"), candidate, criteria),
        _check_benchmark(loaded.get("benchmark"), candidate, criteria),
        _check_shield(loaded.get("shield"), criteria),
        _check_compatibility(loaded.get("compatibility"), manifest),
        _check_ablation(loaded.get("ablation"), criteria, required=criteria.require_ablation),
        _check_rl(loaded.get("rl_curriculum"), criteria, required=criteria.require_rl),
        _check_ranker(loaded.get("ranker"), required=criteria.require_ranker),
    ]
    if replay:
        checks.append(_check_replay(bundle, criteria, enabled=True))
    else:
        checks.append(_criterion(
            "replay", "Deterministic re-execution of saved seeds",
            passed=False, reasons=["replay was disabled for this run"],
        ))
    required_failed = [check for check in checks if check["required"] and not check["passed"]]
    status = "blocked" if not artifact["passed"] else ("pass" if not required_failed else "fail")

    benchmark = loaded.get("benchmark") or {}
    policies = benchmark.get("policies") or (loaded.get("generalization") or {}).get("policies") or {}
    ranked = _baseline_ranking(policies, candidate)
    declared_fallback = str(bundle.get("fallback_policy") or "")
    fallback = (
        declared_fallback if declared_fallback and declared_fallback in policies
        else (ranked[0][0] if ranked else None)
    )
    retained = [candidate] if status == "pass" else []
    rejected = [] if status == "pass" else [candidate]
    envelope = (manifest or {}).get("envelope") or {}
    return {
        "version": G3_VERSION,
        "gate_id": str(bundle.get("gate_id") or "g3"),
        "generated_at_s": round(time.time(), 3),
        "status": status,
        "promoted": retained[0] if retained else None,
        "fallback_policy": fallback,
        "candidate": candidate,
        "checkpoint": checkpoint_path or None,
        "accepted_envelope": envelope,
        "criteria": checks,
        "required_failures": [check["key"] for check in required_failed],
        "retained": retained,
        "rejected": rejected,
        "evidence": evidence_hashes,
        "selection_note": (
            "G3 promotes only on declared thresholds; the strongest non-learning "
            "baseline remains the fallback and failed candidates are retained for analysis"
        ),
    }


def run_gate_file(path: str | Path, *, replay: bool = True) -> dict[str, Any]:
    return run_gate(_read_json(path, "gate bundle"), replay=replay)


def render_gate_report(report: Mapping[str, Any]) -> str:
    """Human-readable Markdown table; the JSON report remains authoritative."""
    lines = [
        f"# G3 gate: {report.get('gate_id')}",
        "",
        f"**Status:** `{report.get('status')}`  ",
        f"**Promoted:** `{report.get('promoted')}`  ",
        f"**Fallback:** `{report.get('fallback_policy')}`",
        "",
        "| Criterion | Required | Result | Measured |",
        "| --- | --- | --- | --- |",
    ]
    for check in report.get("criteria") or []:
        measured = json.dumps(check.get("measured") or {}, sort_keys=True, default=str)
        if len(measured) > 180:
            measured = measured[:177] + "..."
        lines.append(
            f"| {check.get('title')} | {check.get('required')} | "
            f"{'pass' if check.get('passed') else 'fail'} | `{measured}` |"
        )
    failures = [
        f"- **{check.get('key')}**: {'; '.join(check.get('reasons') or [])}"
        for check in report.get("criteria") or []
        if check.get("required") and not check.get("passed")
    ]
    if failures:
        lines.extend(["", "## Blocking reasons", *failures])
    return "\n".join(lines) + "\n"


__all__ = [
    "G3_VERSION",
    "GateError",
    "GateCriteria",
    "run_gate",
    "run_gate_file",
    "render_gate_report",
]
