"""Read-only view of the Day 46–56 learned-policy evidence for the UI.

The renderer must never parse a raw checkpoint or recompute a metric. This
module reads the checked-in reports, trims them to the fields the ML Lab
displays, and preserves the report values verbatim in the JSON response. A
missing or malformed artefact is surfaced as ``available: false`` with the
reason, not replaced by a fabricated table.

There is deliberately no arbitrary path parameter: the endpoint can only read
the fixed evidence files below, so the UI cannot turn into a filesystem browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ML_REPORT_VERSION = "ml-report-1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS: dict[str, str] = {
    "gate": "benchmarks/g3-gate.json",
    "generalization": "benchmarks/generalization.json",
    "benchmark": "benchmarks/policy-comparison.json",
    "shield": "benchmarks/shield-suite.json",
    "compatibility": "benchmarks/compatibility.json",
    "ablations": "benchmarks/ablations.json",
    "search": "benchmarks/adversarial-search.json",
    "rl": "runs/rl-demo/rl-curriculum-manifest.json",
    "ranker": "runs/ranker/ranker-manifest.json",
    "baselines": "benchmarks/ml-baselines.json",
    "curriculum": "runs/curriculum-demo/curriculum-manifest.json",
    "manifest": "runs/g3/model-manifest.json",
}


def _read(root: Path, key: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    relative = ARTIFACTS[key]
    path = root / relative
    meta: dict[str, Any] = {"key": key, "path": relative, "available": False}
    if not path.is_file():
        meta["error"] = "report not found"
        return None, meta
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return None, meta
    if not isinstance(payload, dict):
        meta["error"] = "report is not a JSON object"
        return None, meta
    meta["available"] = True
    meta["bytes"] = path.stat().st_size
    if isinstance(payload.get("generated_at_s"), (int, float)):
        meta["generated_at_s"] = payload["generated_at_s"]
    return payload, meta


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _interval(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"mean": None, "low": None, "high": None}
    return {
        "mean": _number(value.get("mean")),
        "low": _number(value.get("low")),
        "high": _number(value.get("high")),
    }


def _policy_rows(policies: Any) -> list[dict[str, Any]]:
    if not isinstance(policies, dict):
        return []
    rows: list[dict[str, Any]] = []
    for name, entry in policies.items():
        if not isinstance(entry, dict):
            continue
        aggregate = entry.get("aggregate") or {}
        slices = entry.get("slices") or {}
        compact_slices = {}
        for slice_name, slice_entry in slices.items():
            if not isinstance(slice_entry, dict):
                continue
            compact_slices[str(slice_name)] = {
                "coverage": _number((slice_entry.get("coverage") or {}).get("mean")),
                "episodes": int(slice_entry.get("episodes") or 0),
                "expired": int(slice_entry.get("expired") or 0),
                "conflicts": int(slice_entry.get("conflicts") or 0),
            }
        latency = aggregate.get("latency_ms") or {}
        rows.append({
            "name": str(name),
            "coverage": _interval(aggregate.get("coverage")),
            "reward": _interval(aggregate.get("reward")),
            "observed": int(aggregate.get("observed") or 0),
            "tasks_total": int(aggregate.get("tasks_total") or 0),
            "expired": int(aggregate.get("expired") or 0),
            "conflicts": int(aggregate.get("conflicts") or 0),
            "invalid_actions": int(aggregate.get("invalid_actions") or 0),
            "latency_ms": {
                "mean": _number(latency.get("mean")),
                "p95": _number(latency.get("p95")),
            },
            "shield": entry.get("shield"),
            "slices": compact_slices,
        })
    return rows


def _suite(report: dict[str, Any]) -> dict[str, Any]:
    suite = report.get("suite") or {}
    return {
        "split": suite.get("split"),
        "seed": suite.get("seed"),
        "episodes": suite.get("episodes"),
        "inventory": suite.get("inventory") or {},
        "slice_counts": suite.get("slice_counts") or {},
    }


def _gate(report: dict[str, Any]) -> dict[str, Any]:
    criteria = []
    for item in report.get("criteria") or []:
        if not isinstance(item, dict):
            continue
        criteria.append({
            "key": str(item.get("key") or ""),
            "title": str(item.get("title") or ""),
            "required": bool(item.get("required", True)),
            "passed": bool(item.get("passed", False)),
            "reasons": [str(reason) for reason in item.get("reasons") or []],
            "measured": item.get("measured") or {},
            "thresholds": item.get("thresholds") or {},
        })
    return {
        "status": report.get("status"),
        "promoted": report.get("promoted"),
        "fallback": report.get("fallback_policy"),
        "candidate": report.get("candidate"),
        "checkpoint": report.get("checkpoint"),
        "accepted_envelope": report.get("accepted_envelope") or {},
        "criteria": criteria,
        "required_failures": report.get("required_failures") or [],
        "retained": report.get("retained") or [],
        "rejected": report.get("rejected") or [],
        "selection_note": report.get("selection_note"),
    }


def _shield(report: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for item in report.get("cases") or []:
        if not isinstance(item, dict):
            continue
        observed = item.get("observed") or {}
        cases.append({
            "name": str(item.get("name") or ""),
            "description": str(item.get("description") or ""),
            "expected": item.get("expected") or {},
            "observed": {
                "accepted": int(observed.get("accepted") or 0),
                "reason_counts": observed.get("reason_counts") or {},
                "fallback": observed.get("fallback") or [],
            },
            "passed": bool(item.get("passed", False)),
        })
    return {
        "summary": report.get("summary") or {},
        "config": report.get("config") or {},
        "source": report.get("source"),
        "passed": bool(report.get("passed", False)),
        "cases": cases,
    }


def _compatibility(report: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for item in report.get("cases") or []:
        if not isinstance(item, dict):
            continue
        detail = item.get("report") or {}
        cases.append({
            "name": str(item.get("name") or ""),
            "kind": str(item.get("kind") or ""),
            "expected": item.get("expected") or {},
            "passed": bool(item.get("passed", False)),
            "status": detail.get("status"),
            "fallback": detail.get("fallback"),
            "ood_score": _number(detail.get("ood_score")),
            "reasons": detail.get("reasons") or [],
            "warnings": detail.get("warnings") or [],
        })
    return {
        "passed": bool(report.get("passed", False)),
        "counts": report.get("counts") or {},
        "failed": report.get("failed") or [],
        "accepted_envelope": report.get("accepted_envelope") or {},
        "fallback_contract": report.get("fallback_contract"),
        "manifest": {
            "checkpoint": (report.get("manifest") or {}).get("checkpoint"),
            "model_family": (report.get("manifest") or {}).get("model_family"),
            "checkpoint_hash": (report.get("manifest") or {}).get("checkpoint_hash"),
            "feature_schema_hash": (report.get("manifest") or {}).get("feature_schema_hash"),
        },
        "cases": cases,
    }


def _ablations(report: dict[str, Any]) -> dict[str, Any]:
    variants = []
    rows = report.get("variants") or {}
    if isinstance(rows, dict):
        for name, entry in rows.items():
            if not isinstance(entry, dict):
                continue
            variant = entry.get("variant") or {}
            metrics = entry.get("metrics") or {}
            variants.append({
                "name": str(name),
                "kind": variant.get("kind"),
                "transform": variant.get("transform"),
                "shield": bool(variant.get("shield", False)),
                "diagnostic": bool(variant.get("diagnostic", False)),
                "note": variant.get("note"),
                "checkpoint": variant.get("checkpoint"),
                "coverage": _interval(entry.get("aggregate", {}).get("coverage")),
                "metrics": {
                    "mean_coverage": _number(metrics.get("mean_coverage")),
                    "mean_reward": _number(metrics.get("mean_reward")),
                    "conflicts": int(metrics.get("conflicts") or 0),
                    "invalid_actions": int(metrics.get("invalid_actions") or 0),
                },
            })
    return {
        "suite": report.get("suite") or {},
        "selection_note": report.get("selection_note"),
        "variants": variants,
    }


def _baselines(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    rows = []
    policies = report.get("policies") or {}
    if isinstance(policies, dict):
        for name, entry in policies.items():
            if not isinstance(entry, dict):
                continue
            aggregate = entry.get("aggregate") or {}
            rows.append({
                "name": str(name),
                "coverage": _number(aggregate.get("mean_coverage")),
                "reward": _number(aggregate.get("mean_reward")),
                "episodes": int(aggregate.get("episodes") or 0),
                "observed": int(aggregate.get("observed") or 0),
                "expired": int(aggregate.get("expired") or 0),
                "conflicts": int(aggregate.get("conflicts") or 0),
                "invalid_actions": int(aggregate.get("invalid_actions") or 0),
                "tasks_total": int(aggregate.get("tasks_total") or 0),
            })
    return {
        "campaign": report.get("campaign"),
        "split": report.get("split"),
        "episodes": report.get("episodes"),
        "selection_note": report.get("selection_note"),
        "policies": rows,
    }


def _curriculum(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    selected = report.get("selected_seed") or {}
    variation = report.get("seed_variation") or {}
    variation_metrics = variation.get("metrics") or {}
    seeds = []
    for seed_report in report.get("seeds") or []:
        if not isinstance(seed_report, dict):
            continue
        stages = []
        for stage in seed_report.get("stages") or []:
            if not isinstance(stage, dict):
                continue
            best = stage.get("best") or {}
            reward = stage.get("reward") or {}
            stages.append({
                "stage": stage.get("stage"),
                "status": stage.get("status"),
                "samples": int(stage.get("samples") or 0),
                "validation_loss": _number(best.get("validation_loss")),
                "validation_accuracy": _number(best.get("validation_accuracy")),
                "mean_coverage": _number(reward.get("mean_coverage")),
                "initialized_from_previous": bool(stage.get("initialized_from")),
                "wall_time_s": _number(stage.get("wall_time_s")),
            })
        seeds.append({"seed": seed_report.get("seed"), "stages": stages})
    return {
        "curriculum_id": report.get("curriculum_id"),
        "curriculum_version": report.get("curriculum_version"),
        "selected_seed": {
            "seed": selected.get("seed"),
            "stage": selected.get("stage"),
            "metric": selected.get("metric"),
            "value": _number(selected.get("selection_value")),
        },
        "selection_rule": report.get("selection_rule"),
        "seed_variation": {
            "seeds": variation.get("seeds"),
            "metrics": {
                name: {
                    "mean": _number(row.get("mean")),
                    "std": _number(row.get("std")),
                    "min": _number(row.get("min")),
                    "max": _number(row.get("max")),
                }
                for name, row in variation_metrics.items()
                if isinstance(row, dict)
            },
        },
        "seeds": seeds,
    }


def _model_manifest(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    training = report.get("training_config") or {}
    model = training.get("model") or {}
    envelope = report.get("envelope") or {}
    return {
        "version": report.get("version"),
        "checkpoint": report.get("checkpoint"),
        "checkpoint_hash": report.get("checkpoint_hash"),
        "model_family": report.get("model_family"),
        "observation_version": report.get("observation_version"),
        "env_version": report.get("env_version"),
        "feature_schema_hash": report.get("feature_schema_hash"),
        "parameter_schema_hash": report.get("parameter_schema_hash"),
        "trained_campaign": report.get("trained_campaign"),
        "model": {
            "name": model.get("model"),
            "hidden": model.get("hidden"),
            "layers": model.get("layers"),
        },
        "training": {
            "learning_rate": _number(training.get("learning_rate")),
            "epochs": training.get("epochs"),
            "batches_per_epoch": training.get("batches_per_epoch"),
            "graphs_per_batch": training.get("graphs_per_batch"),
            "max_sequence_samples": training.get("max_sequence_samples"),
            "seed": training.get("seed"),
            "curriculum": training.get("curriculum") or {},
        },
        "envelope": {
            "ranges": envelope.get("ranges") or {},
            "families": envelope.get("families") or [],
        },
    }


def _training(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    summary = report.get("summary") or {}
    seeds = []
    for seed_report in report.get("seed_reports") or []:
        if not isinstance(seed_report, dict):
            continue
        stages = []
        for stage in seed_report.get("stages") or []:
            if not isinstance(stage, dict):
                continue
            stages.append({
                "stage": stage.get("stage"),
                "status": stage.get("status"),
                "selection": stage.get("selection") or {},
                "checkpoint": stage.get("checkpoint"),
                "fleet_limit": stage.get("fleet_limit"),
                "target_limit": stage.get("target_limit"),
                "initialized_from_previous": bool(stage.get("initialized_from_previous", False)),
            })
        seeds.append({"seed": seed_report.get("seed"), "stages": stages})
    return {
        "run_id": report.get("run_id"),
        "config": report.get("config") or {},
        "summary": summary,
        "seeds": seeds,
        "selection_note": report.get("selection_note"),
    }


def _ranker(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    metrics = report.get("metrics") or {}
    return {
        "version": report.get("version"),
        "records": report.get("records"),
        "train_records": report.get("train_records"),
        "validation_records": report.get("validation_records"),
        "family_held_out": bool(report.get("family_held_out", False)),
        "split_note": report.get("split_note"),
        "class_balance": report.get("class_balance") or {},
        "metrics": {
            "validation_loss": _number(metrics.get("validation_loss")),
            "validation_auc": _number(metrics.get("validation_auc")),
            "validation_average_precision": _number(metrics.get("validation_average_precision")),
            "top_k": metrics.get("top_k") or {},
            "calibration": metrics.get("calibration") or [],
        },
        "checkpoint": report.get("checkpoint"),
        "selection_note": report.get("selection_note"),
    }


def ml_report(root: str | Path | None = None) -> dict[str, Any]:
    """Build the compact ML Lab payload from fixed evidence paths."""
    base = Path(root) if root is not None else PROJECT_ROOT
    loaded: dict[str, dict[str, Any] | None] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for key in ARTIFACTS:
        loaded[key], artifacts[key] = _read(base, key)

    gate = _gate(loaded["gate"]) if loaded["gate"] else None
    generalization = loaded["generalization"]
    benchmark = loaded["benchmark"]
    shield = loaded["shield"]
    compatibility = loaded["compatibility"]
    ablations = loaded["ablations"]
    search = loaded["search"]
    rl = loaded["rl"]
    ranker = loaded["ranker"]
    baselines = loaded["baselines"]
    curriculum = loaded["curriculum"]
    manifest = loaded["manifest"]
    return {
        "version": ML_REPORT_VERSION,
        "artifacts": artifacts,
        "gate": gate,
        "generalization": {
            "suite": _suite(generalization) if generalization else {},
            "policies": _policy_rows((generalization or {}).get("policies")),
            "selection_note": (generalization or {}).get("selection_note"),
        },
        "benchmark": {
            "suite": _suite(benchmark) if benchmark else {},
            "policies": _policy_rows((benchmark or {}).get("policies")),
            "selection_note": (benchmark or {}).get("selection_note"),
        },
        "shield": _shield(shield) if shield else None,
        "compatibility": _compatibility(compatibility) if compatibility else None,
        "ablations": _ablations(ablations) if ablations else None,
        "training": {
            "rl": _training(rl),
            "ranker": _ranker(ranker),
        },
        "search": {
            "totals": (search or {}).get("totals") or {},
            "campaigns": (search or {}).get("campaigns") or [],
            "budget_per_campaign": (search or {}).get("budget_per_campaign"),
            "note": (search or {}).get("note"),
        } if search else None,
        "pipeline": {
            "baselines": _baselines(baselines),
            "curriculum": _curriculum(curriculum),
            "manifest": _model_manifest(manifest),
        },
    }


__all__ = ["ML_REPORT_VERSION", "ARTIFACTS", "ml_report"]
