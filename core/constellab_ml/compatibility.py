"""Day 51: checkpoint compatibility, calibration and safe fallback.

A parsed client configuration is not the same thing as a configuration the
learned policy understands. This module keeps that distinction explicit:

* a **model manifest** records the schema versions, feature/parameter hashes,
  checkpoint hash and the training envelope of a promoted policy;
* a **compatibility report** compares a concrete episode/config against that
  manifest and returns ``compatible``, ``out-of-distribution`` or
  ``incompatible`` with named reasons;
* an **OOD score** is a transparent distance from the trained envelope, never a
  magical confidence number, and it only drives a fallback warning;
* a **fallback decision** chooses the learned policy inside the envelope and a
  named baseline outside it, preserving the reason.

The checks are deliberately split: hash/version mismatch is not a numeric
extrapolation, and an unknown component semantic is not something a slightly
larger fleet can excuse.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from constellab_core import trace as core_trace

from . import ENV_VERSION, OBSERVATION_VERSION, feature_schema_payload
from .env import EpisodeSpec

MANIFEST_VERSION = "model-manifest-1"
COMPATIBILITY_VERSION = "model-compatibility-1"
STATUS_COMPATIBLE = "compatible"
STATUS_OOD = "out-of-distribution"
STATUS_INCOMPATIBLE = "incompatible"

# Parameters used to describe the training envelope. They are ordinary runtime
# measurements, not hidden simulator state.
ENVELOPE_KEYS: tuple[str, ...] = (
    "fleet.size",
    "tasks.target_count",
    "faults.count",
    "network.partitions",
    "energy.initial_charge_ratio",
    "sensor.duty_cycle_ratio",
)


class CompatibilityError(RuntimeError):
    """A manifest or checkpoint is malformed."""


def feature_schema_hash() -> str:
    return core_trace.stable_hash(feature_schema_payload())


def checkpoint_hash(payload: Mapping[str, Any]) -> str:
    """Stable hash of the weights and the policy-shaping metadata."""
    digest = hashlib.sha256()
    state = payload.get("state_dict") or {}
    for key in sorted(state):
        tensor = state[key]
        try:
            value = tensor.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        except AttributeError:  # pragma: no cover - already a plain value
            digest.update(key.encode("utf-8"))
            digest.update(repr(tensor).encode("utf-8"))
    for key in ("model_config", "feature_schema", "observation_version", "env_version",
                "parameter_schema_hash"):
        digest.update(key.encode("utf-8"))
        digest.update(json.dumps(payload.get(key), sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class Range:
    low: float | None = None
    high: float | None = None

    def contains(self, value: float) -> bool:
        if self.low is not None and value < self.low:
            return False
        if self.high is not None and value > self.high:
            return False
        return True

    def distance(self, value: float) -> float:
        span = max(1e-9, (self.high if self.high is not None else value)
                   - (self.low if self.low is not None else value))
        if self.low is not None and value < self.low:
            return (self.low - value) / span
        if self.high is not None and value > self.high:
            return (value - self.high) / span
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"min": self.low, "max": self.high}


@dataclass
class TrainingEnvelope:
    """Observed training ranges plus the categorical slices that were present."""

    ranges: dict[str, Range] = field(default_factory=dict)
    families: list[str] = field(default_factory=list)
    condition_classes: list[str] = field(default_factory=list)
    held_out_keys: list[str] = field(default_factory=list)

    @classmethod
    def from_dataset_manifest(cls, manifest: Mapping[str, Any]) -> "TrainingEnvelope":
        coverage = dict(manifest.get("coverage") or {})
        keys = dict(coverage.get("keys") or {})
        ranges: dict[str, Range] = {}
        for key in ENVELOPE_KEYS:
            entry = keys.get(key) or {}
            if entry.get("min") is None and entry.get("max") is None:
                continue
            ranges[key] = Range(entry.get("min"), entry.get("max"))
        return cls(
            ranges=ranges,
            families=sorted((coverage.get("scenario_families") or {}).keys()),
            condition_classes=sorted((coverage.get("condition_classes") or {}).keys()),
            held_out_keys=list(coverage.get("held_out_keys") or []),
        )

    @classmethod
    def from_dataset(cls, dataset_root: str | Path) -> "TrainingEnvelope":
        path = Path(dataset_root) / "manifest.json"
        if not path.exists():
            raise CompatibilityError(f"{dataset_root!r} has no dataset manifest")
        return cls.from_dataset_manifest(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranges": {key: value.to_dict() for key, value in sorted(self.ranges.items())},
            "families": list(self.families),
            "condition_classes": list(self.condition_classes),
            "held_out_keys": list(self.held_out_keys),
        }


@dataclass
class ModelManifest:
    """The promotion record a checkpoint must carry before it can be used."""

    checkpoint: str = ""
    checkpoint_hash: str = ""
    model_family: str = ""
    observation_version: str = OBSERVATION_VERSION
    env_version: str = ENV_VERSION
    feature_schema_hash: str = ""
    parameter_schema_hash: str = ""
    trained_campaign: str = ""
    training_config: dict[str, Any] = field(default_factory=dict)
    envelope: TrainingEnvelope = field(default_factory=TrainingEnvelope)
    versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "checkpoint": self.checkpoint,
            "checkpoint_hash": self.checkpoint_hash,
            "model_family": self.model_family,
            "observation_version": self.observation_version,
            "env_version": self.env_version,
            "feature_schema_hash": self.feature_schema_hash,
            "parameter_schema_hash": self.parameter_schema_hash,
            "trained_campaign": self.trained_campaign,
            "training_config": dict(self.training_config),
            "envelope": self.envelope.to_dict(),
            "versions": dict(self.versions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelManifest":
        envelope = payload.get("envelope") or {}
        ranges = {
            key: Range(value.get("min"), value.get("max"))
            for key, value in dict(envelope.get("ranges") or {}).items()
        }
        return cls(
            checkpoint=str(payload.get("checkpoint", "")),
            checkpoint_hash=str(payload.get("checkpoint_hash", "")),
            model_family=str(payload.get("model_family", "")),
            observation_version=str(payload.get("observation_version", OBSERVATION_VERSION)),
            env_version=str(payload.get("env_version", ENV_VERSION)),
            feature_schema_hash=str(payload.get("feature_schema_hash", "")),
            parameter_schema_hash=str(payload.get("parameter_schema_hash", "")),
            trained_campaign=str(payload.get("trained_campaign", "")),
            training_config=dict(payload.get("training_config") or {}),
            envelope=TrainingEnvelope(
                ranges=ranges,
                families=list(envelope.get("families") or []),
                condition_classes=list(envelope.get("condition_classes") or []),
                held_out_keys=list(envelope.get("held_out_keys") or []),
            ),
            versions=dict(payload.get("versions") or {}),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | Mapping[str, Any],
        *,
        dataset_root: str | Path | None = None,
    ) -> "ModelManifest":
        if isinstance(checkpoint, Mapping):
            payload = dict(checkpoint)
            path = str(payload.get("path", ""))
        else:
            import torch

            path = str(checkpoint)
            payload = torch.load(Path(checkpoint), weights_only=False)
            payload["path"] = path
        root = dataset_root
        if root is None:
            root = (payload.get("training_config") or {}).get("dataset_root")
        envelope = TrainingEnvelope()
        if root:
            try:
                envelope = TrainingEnvelope.from_dataset(root)
            except CompatibilityError:
                envelope = TrainingEnvelope()
        return cls(
            checkpoint=path,
            checkpoint_hash=checkpoint_hash(payload),
            model_family=str((payload.get("model_config") or {}).get("model", "")),
            observation_version=str(payload.get("observation_version", OBSERVATION_VERSION)),
            env_version=str(payload.get("env_version", ENV_VERSION)),
            feature_schema_hash=core_trace.stable_hash(
                payload.get("feature_schema") or feature_schema_payload()
            ),
            parameter_schema_hash=str(payload.get("parameter_schema_hash", "")),
            trained_campaign=str(payload.get("dataset_campaign", "")),
            training_config=dict(payload.get("training_config") or {}),
            envelope=envelope,
            versions=dict(payload.get("versions") or {}),
        )

    def verify_checkpoint(self, payload: Mapping[str, Any]) -> list[str]:
        """Reasons this checkpoint does not match its manifest, empty when valid."""
        reasons: list[str] = []
        if self.checkpoint_hash and checkpoint_hash(payload) != self.checkpoint_hash:
            reasons.append("checkpoint-hash-mismatch")
        if self.observation_version != str(payload.get("observation_version", OBSERVATION_VERSION)):
            reasons.append("observation-version-mismatch")
        if self.env_version != str(payload.get("env_version", ENV_VERSION)):
            reasons.append("env-version-mismatch")
        expected = core_trace.stable_hash(
            payload.get("feature_schema") or feature_schema_payload()
        )
        if self.feature_schema_hash and expected != self.feature_schema_hash:
            reasons.append("feature-schema-mismatch")
        trained = str((payload.get("model_config") or {}).get("model", ""))
        if self.model_family and trained != self.model_family:
            reasons.append("model-family-mismatch")
        return reasons


@dataclass
class CompatibilityReport:
    status: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ood_score: float = 0.0
    envelope: dict[str, Any] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)
    fallback: str = "model"

    @property
    def compatible(self) -> bool:
        return self.status == STATUS_COMPATIBLE

    @property
    def usable(self) -> bool:
        return self.status != STATUS_INCOMPATIBLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": COMPATIBILITY_VERSION,
            "status": self.status,
            "compatible": self.compatible,
            "usable": self.usable,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "ood_score": round(float(self.ood_score), 6),
            "envelope": dict(self.envelope),
            "schema": dict(self.schema),
            "fallback": self.fallback,
        }


def _episode_values(spec: EpisodeSpec) -> dict[str, float]:
    faults = (spec.fault_plan or {}).get("faults") or []
    values = {
        "fleet.size": float(spec.fleet_size),
        "tasks.target_count": float(len(spec.targets)),
        "faults.count": float(len(faults)),
    }
    tokens = ((spec.parameters or {}).get("tokens") or [])
    for token in tokens:
        field = str(token.get("field", ""))
        if field == "network.partitions":
            values[field] = float(token.get("value") or 0.0)
        elif field == "energy.initial_charge_ratio":
            values[field] = float(token.get("value") or 1.0)
        elif field == "sensor.duty_cycle_ratio":
            values[field] = float(token.get("value") or 1.0)
    return values


def compatibility_report(
    manifest: ModelManifest | Mapping[str, Any] | None,
    spec: EpisodeSpec,
    *,
    ood_threshold: float = 1.0,
    observation: Mapping[str, Any] | None = None,
) -> CompatibilityReport:
    """Compare one episode against a promoted policy's accepted envelope."""
    if manifest is None:
        return CompatibilityReport(
            status=STATUS_COMPATIBLE,
            warnings=["no manifest supplied; policy is treated as unverified"],
            fallback="model",
        )
    entry = manifest if isinstance(manifest, ModelManifest) else ModelManifest.from_dict(manifest)
    reasons: list[str] = []
    warnings: list[str] = []
    schema = {
        "observation_version": OBSERVATION_VERSION,
        "expected_observation_version": entry.observation_version,
        "env_version": ENV_VERSION,
        "expected_env_version": entry.env_version,
        "feature_schema_hash": feature_schema_hash(),
        "expected_feature_schema_hash": entry.feature_schema_hash,
        "parameter_schema_hash": spec.parameter_registry,
        "expected_parameter_schema_hash": entry.parameter_schema_hash,
    }
    if entry.observation_version and entry.observation_version != OBSERVATION_VERSION:
        reasons.append("observation-version-mismatch")
    if entry.env_version and entry.env_version != ENV_VERSION:
        reasons.append("env-version-mismatch")
    if entry.feature_schema_hash and entry.feature_schema_hash != feature_schema_hash():
        reasons.append("feature-schema-mismatch")
    if entry.parameter_schema_hash and spec.parameter_registry and (
        entry.parameter_schema_hash != spec.parameter_registry
    ):
        reasons.append("parameter-schema-mismatch")
    parameter_report = dict((spec.parameters or {}).get("compatibility") or {})
    if parameter_report and not parameter_report.get("compatible", True):
        reasons.append("unsupported-parameter-semantics")
        warnings.extend(str(item) for item in parameter_report.get("reasons", []))

    values = _episode_values(spec)
    distance = 0.0
    outside: list[str] = []
    for key, bound in entry.envelope.ranges.items():
        if key not in values:
            continue
        gap = bound.distance(values[key])
        if gap > 0.0:
            outside.append(key)
            distance += gap
    ood_score = min(1.0, distance)
    if outside:
        warnings.append(f"outside training range: {', '.join(sorted(outside))}")
    if outside and ood_score >= ood_threshold:
        status = STATUS_OOD
        fallback = "baseline"
    elif reasons:
        status = STATUS_INCOMPATIBLE
        fallback = "baseline"
    else:
        status = STATUS_COMPATIBLE
        fallback = "model"
    if observation is not None:
        schema["observed_satellites"] = len(observation.get("satellite_ids") or [])
        schema["observed_tasks"] = len(observation.get("task_ids") or [])
    return CompatibilityReport(
        status=status,
        reasons=reasons,
        warnings=warnings,
        ood_score=ood_score,
        envelope=entry.envelope.to_dict(),
        schema=schema,
        fallback=fallback,
    )


def compatibility_suite(
    manifest: ModelManifest | Mapping[str, Any],
    dataset_root: str | Path,
    *,
    split: str = "test",
    limit: int = 3,
    ood_threshold: float = 1.0,
) -> dict[str, Any]:
    """Day 51/56: a deterministic case suite over a real campaign.

    Covers an in-envelope episode, a hash/schema mismatch, an unknown parameter
    semantic and a numeric extrapolation. Each case declares the status and
    fallback it must produce; the suite reports the actual report beside it so
    a mismatch can be inspected rather than hidden behind a single boolean.
    """
    from .dataset import EpisodeDataset

    entry = manifest if isinstance(manifest, ModelManifest) else ModelManifest.from_dict(manifest)
    records = list(EpisodeDataset(dataset_root).episodes(split))[: max(1, limit)]
    if not records:
        raise CompatibilityError(f"campaign {dataset_root!r} has no {split} episodes")
    cases: list[dict[str, Any]] = []
    for record in records:
        spec = EpisodeSpec.from_payload(record["spec"])
        report = compatibility_report(entry, spec, ood_threshold=ood_threshold)
        cases.append({
            "name": f"episode:{record.get('episode_id')}",
            "kind": "dataset-episode",
            "expected": {"usable": True, "fallback": ["model", "baseline"]},
            "report": report.to_dict(),
            "passed": report.usable and report.fallback in ("model", "baseline"),
        })

    base_spec = EpisodeSpec.from_payload(records[0]["spec"])
    tampered = ModelManifest.from_dict({**entry.to_dict(), "feature_schema_hash": "0" * 16})
    report = compatibility_report(tampered, base_spec, ood_threshold=ood_threshold)
    cases.append({
        "name": "schema-mismatch",
        "kind": "tampered-manifest",
        "expected": {"status": STATUS_INCOMPATIBLE, "fallback": "baseline"},
        "report": report.to_dict(),
        "passed": report.status == STATUS_INCOMPATIBLE and report.fallback == "baseline",
    })

    payload = base_spec.payload()
    parameters = dict(payload.get("parameters") or {})
    parameters["compatibility"] = {
        "compatible": False,
        "reasons": ["synthetic test: unsupported component semantics"],
    }
    payload["parameters"] = parameters
    payload["episode_id"] = f"{base_spec.episode_id}-unsupported"
    unsupported = EpisodeSpec.from_payload(payload)
    report = compatibility_report(entry, unsupported, ood_threshold=ood_threshold)
    cases.append({
        "name": "unsupported-semantics",
        "kind": "synthetic",
        "expected": {"status": STATUS_INCOMPATIBLE, "fallback": "baseline"},
        "report": report.to_dict(),
        "passed": report.status == STATUS_INCOMPATIBLE and report.fallback == "baseline",
    })

    payload = base_spec.payload()
    parameters = dict(payload.get("parameters") or {})
    tokens = [dict(token) for token in parameters.get("tokens") or []]
    tokens.append({"field": "network.partitions", "value": 99.0})
    parameters["tokens"] = tokens
    payload["parameters"] = parameters
    payload["episode_id"] = f"{base_spec.episode_id}-extrapolated"
    extrapolated = EpisodeSpec.from_payload(payload)
    report = compatibility_report(entry, extrapolated, ood_threshold=ood_threshold)
    cases.append({
        "name": "numeric-extrapolation",
        "kind": "synthetic",
        "expected": {"status": STATUS_OOD, "fallback": "baseline"},
        "report": report.to_dict(),
        "passed": report.status == STATUS_OOD and report.fallback == "baseline",
    })

    counts: dict[str, int] = {}
    for case in cases:
        status = str(case["report"]["status"])
        counts[status] = counts.get(status, 0) + 1
    failed = [case["name"] for case in cases if not case["passed"]]
    return {
        "version": COMPATIBILITY_VERSION,
        "dataset": str(dataset_root),
        "split": split,
        "manifest": entry.to_dict(),
        "cases": cases,
        "counts": dict(sorted(counts.items())),
        "accepted_envelope": entry.envelope.to_dict(),
        "fallback_contract": "baseline for incompatible and out-of-distribution configurations",
        "passed": not failed,
        "failed": failed,
    }


def choose_policy(
    models: Mapping[str, Any],
    baselines: Mapping[str, Any],
    manifest: ModelManifest | Mapping[str, Any] | None,
    spec: EpisodeSpec,
    *,
    model_key: str = "learned",
    fallback_key: str = "expert-rule",
    ood_threshold: float = 1.0,
) -> tuple[str, Any, CompatibilityReport]:
    """Select the learned policy inside the envelope, else a named baseline."""
    report = compatibility_report(manifest, spec, ood_threshold=ood_threshold)
    if report.fallback == "model" and model_key in models:
        return model_key, models[model_key], report
    if fallback_key not in baselines:
        raise CompatibilityError(
            f"fallback {fallback_key!r} is not in {sorted(baselines)}"
        )
    return fallback_key, baselines[fallback_key], report


__all__ = [
    "MANIFEST_VERSION",
    "COMPATIBILITY_VERSION",
    "STATUS_COMPATIBLE",
    "STATUS_OOD",
    "STATUS_INCOMPATIBLE",
    "ENVELOPE_KEYS",
    "CompatibilityError",
    "Range",
    "TrainingEnvelope",
    "ModelManifest",
    "CompatibilityReport",
    "feature_schema_hash",
    "checkpoint_hash",
    "compatibility_report",
    "compatibility_suite",
    "choose_policy",
]
