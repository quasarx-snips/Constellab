"""Day 36: large-scale episode generation, storage and loading.

One *campaign* is a directory of gzipped JSONL episode records plus a manifest:

* ``episodes-<split>.jsonl.gz`` — one episode per line, each with the episode
  spec (configuration, event schedule, injection plan, link model), the expert's
  decision trace, the per-step observation/action records and the outcome
  metrics,
* ``manifest.json`` — campaign id, seeds, versions, counts, hashes, coverage,
  the expert catalogue and the leakage audit,
* ``index.json`` — the resumable progress ledger (which campaign indices are
  already recorded, keyed by split).

Generation is **resumable**: an interrupted batch continues at the first index
the campaign has not recorded yet, and a completed campaign records the split of
every row, so rows generated after a model was tuned can never be silently mixed
into a held-out split (``load`` pins a campaign id and refuses to merge).

Splits come from the scenario sampler (seed-based), never from row position, so
neighbouring rows do not leak across train/validation/test. Every stored
observation is exactly the Day 35 contract, and a leakage audit runs over a
sample record before the manifest is written.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from constellab_core import trace as core_trace

from . import OBSERVATION_VERSION, feature_schema_payload
from .env import EpisodeOutcome, EpisodeSpec, SwarmEnvironment
from .expert import (
    DEFAULT_EXPERT,
    EXPERT_VERSION,
    expert_catalog,
    expert_factory,
    leakage_scan,
)
from .randomization import DistributionSet
from .scenarios import SCENARIO_VERSION, SPLIT_NAMES, ScenarioSample, ScenarioSampler

DATASET_VERSION = "episode-dataset-1"
MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.json"


class DatasetError(RuntimeError):
    """The dataset is out of contract (missing manifest, bad campaign, ...)."""


@dataclass
class DatasetConfig:
    """Declarative campaign configuration."""

    campaign_id: str
    seed: int = 0
    expert_key: str = DEFAULT_EXPERT
    salt: str = ""
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15)
    include_observations: bool = True
    allow_diagnostic_expert: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if not str(self.campaign_id).strip():
            raise DatasetError("campaign_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "seed": self.seed,
            "expert_key": self.expert_key,
            "salt": self.salt,
            "splits": list(self.splits),
            "include_observations": self.include_observations,
            "allow_diagnostic_expert": self.allow_diagnostic_expert,
            "label": self.label,
        }


def compact_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """The observation without the parameter context (stored once per episode)."""
    return {
        key: value
        for key, value in observation.items()
        if key not in ("context", "notes")
    }


def episode_record(
    sample: ScenarioSample,
    environment: SwarmEnvironment,
    expert: Any,
    outcome: EpisodeOutcome,
    *,
    include_observations: bool = True,
    wall_time_s: float | None = None,
) -> dict[str, Any]:
    """One stored episode: spec, seed, schedule, decisions, outcomes."""
    steps: list[dict[str, Any]] = []
    observations = list(getattr(environment, "captured_observations", []))
    for index, step in enumerate(outcome.steps):
        decision = expert.decisions[index] if index < len(expert.decisions) else None
        record = outcome.records[index] if index < len(outcome.records) else {}
        entry: dict[str, Any] = {
            "time_s": step.time_s,
            "actions": [dict(item) for item in record.get("actions", [])],
            "targets": [list(pair) for pair in record.get("targets", [])],
            "reward": round(step.reward, 6),
            "terms": {key: round(value, 6) for key, value in step.terms.items()},
            "observed": list(step.observed),
            "expired": list(step.expired),
            "rejected": list(step.rejected),
            "waits": step.waits,
        }
        if decision is not None:
            entry["expert"] = {
                "candidates": decision.candidate_count,
                "abstained": decision.abstained,
                "abstain_reason": decision.abstain_reason,
                "chosen": decision.chosen,
            }
        if include_observations and index < len(observations):
            entry["observation"] = observations[index]
        steps.append(entry)

    return {
        "dataset_version": DATASET_VERSION,
        "episode_id": sample.spec.episode_id,
        "split": sample.spec.split,
        "family": sample.family,
        "condition_class": sample.condition_class,
        "spec_hash": sample.spec.spec_hash(),
        "seed": sample.spec.seed,
        "dropped_targets": sample.dropped_targets,
        "warnings": list(sample.warnings),
        "values": dict(sample.values),
        "spec": sample.spec.payload(),
        "parameter_context": dict(sample.spec.parameters or {}),
        "metrics": dict(outcome.metrics),
        "trace_hash": outcome.trace_hash,
        "expected_trace_hash": outcome.manifest.get("expected_trace_hash"),
        "expert": expert.metadata(),
        "wall_time_s": round(wall_time_s, 4) if wall_time_s is not None else None,
        "steps": steps,
    }


class CampaignWriter:
    """Writes one campaign directory, resumably."""

    def __init__(
        self,
        root: str | Path,
        config: DatasetConfig,
        *,
        distribution_set: DistributionSet | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.sampler = ScenarioSampler(
            seed=config.seed,
            distribution_set=distribution_set,
            splits=config.splits,
            salt=config.salt,
        )
        self.expert_key = config.expert_key
        if not config.allow_diagnostic_expert:
            entry = {item["key"]: item for item in expert_catalog()}.get(config.expert_key)
            if entry is None:
                raise DatasetError(
                    f"unknown expert {config.expert_key!r}; known: "
                    f"{', '.join(item['key'] for item in expert_catalog())}"
                )
            if not entry["deployable"]:
                raise DatasetError(
                    f"expert {config.expert_key!r} is diagnostic-only; imitation training "
                    "must not be initialised from a policy that sees the future "
                    "(pass allow_diagnostic_expert=True only for a labelled diagnostic set)"
                )
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self._read_index()
        self.samples: dict[int, ScenarioSample] = {}
        self.records: dict[str, int] = {name: 0 for name in SPLIT_NAMES}

    # --- ledger -----------------------------------------------------------------------
    def _read_index(self) -> dict[str, Any]:
        path = self.root / INDEX_NAME
        if not path.exists():
            return {"version": DATASET_VERSION, "campaign_id": self.config.campaign_id, "indices": {}}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("campaign_id") != self.config.campaign_id:
            raise DatasetError(
                f"campaign {payload.get('campaign_id')!r} != {self.config.campaign_id!r}: "
                "refusing to mix two campaigns in one directory"
            )
        return payload

    def _write_index(self) -> None:
        (self.root / INDEX_NAME).write_text(
            json.dumps(self.index, indent=2, sort_keys=True), encoding="utf-8"
        )

    @property
    def recorded(self) -> dict[str, int]:
        return {split: int(self.index["indices"].get(split, {}).__len__()) for split in SPLIT_NAMES}

    def _path(self, split: str) -> Path:
        return self.root / f"episodes-{split}.jsonl.gz"

    # --- generation -------------------------------------------------------------------
    def generate(
        self,
        count: int,
        *,
        start: int = 0,
        resume: bool = True,
        progress: Any = None,
    ) -> dict[str, Any]:
        """Generate ``count`` campaign indices from ``start`` (resumable)."""
        if count < 0:
            raise DatasetError("count must be non-negative")
        generated = 0
        skipped = 0
        indices = list(range(start, start + count))
        for index in indices:
            split = self.sampler.split_for(index)
            known = self.index["indices"].setdefault(split, {})
            episode_id = f"{split}-{index:05d}"
            if resume and episode_id in known:
                skipped += 1
                continue
            sample = self.sampler.sample(index, split=split)
            environment = SwarmEnvironment(sample.spec)
            expert = expert_factory(self.expert_key, environment=environment)
            started = time.perf_counter()
            outcome = environment.rollout(expert, capture_observations=True)
            elapsed = time.perf_counter() - started
            record = episode_record(
                sample,
                environment,
                expert,
                outcome,
                include_observations=self.config.include_observations,
                wall_time_s=elapsed,
            )
            self._append(split, record)
            known[episode_id] = {
                "spec_hash": sample.spec.spec_hash(),
                "family": sample.family,
                "condition_class": sample.condition_class,
                "trace_hash": outcome.trace_hash,
            }
            self.samples[index] = sample
            self.records[split] = self.records.get(split, 0) + 1
            generated += 1
            if progress is not None:
                progress(index, sample, outcome)
        self._write_index()
        return {
            "generated": generated,
            "skipped": skipped,
            "recorded": self.recorded,
            "indices": indices[0] if indices else None,
        }

    def _append(self, split: str, record: Mapping[str, Any]) -> None:
        with gzip.open(self._path(split), "at", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=False) + "\n")

    # --- manifest ---------------------------------------------------------------------
    def manifest(self, *, samples: Sequence[ScenarioSample] | None = None) -> dict[str, Any]:
        collected = list(samples) if samples is not None else list(self.samples.values())
        coverage = self.sampler.coverage(collected) if collected else {}
        audit = self.audit()
        counts = {
            split: len(self.index["indices"].get(split, {})) for split in SPLIT_NAMES
        }
        payload = {
            "dataset_version": DATASET_VERSION,
            "campaign_id": self.config.campaign_id,
            "label": self.config.label or self.config.campaign_id,
            "created_at_s": round(time.time(), 3),
            "config": self.config.to_dict(),
            "scenario": self.sampler.metadata(),
            "expert": {
                "key": self.expert_key,
                "version": EXPERT_VERSION,
                "catalog": expert_catalog(),
            },
            "schema": {
                "observation_version": OBSERVATION_VERSION,
                "feature_schema": feature_schema_payload(),
                "parameter_schema_hash": self.sampler.registry.schema_hash(),
            },
            "counts": counts,
            "total_episodes": sum(counts.values()),
            "index_hash": core_trace.stable_hash(
                {split: sorted(self.index["indices"].get(split, {})) for split in SPLIT_NAMES}
            ),
            "coverage": coverage,
            "leakage": audit,
        }
        return payload

    def write_manifest(self, *, samples: Sequence[ScenarioSample] | None = None) -> Path:
        payload = self.manifest(samples=samples)
        path = self.root / MANIFEST_NAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        return path

    def audit(self, *, limit: int = 1) -> dict[str, Any]:
        """Scan stored records for hidden fields a student model must never see."""
        findings: dict[str, list[str]] = {}
        checked = 0
        for split in SPLIT_NAMES:
            for record in self.iter_records(split, limit=limit):
                checked += 1
                steps = record.get("steps", [])
                student_view = {
                    "steps": [
                        {
                            "observation": step.get("observation"),
                            "actions": step.get("actions"),
                            "targets": step.get("targets"),
                        }
                        for step in steps
                    ]
                }
                leaked = leakage_scan(student_view)
                if leaked:
                    findings[split] = leaked
        return {
            "checked_episodes": checked,
            "leaked_fields": findings,
            "clean": not findings,
            "note": (
                "student inputs are the stored observations only; the spec, decisions "
                "and outcomes stay on the dataset side"
            ),
        }

    def iter_records(self, split: str, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
        path = self._path(split)
        if not path.exists():
            return
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for position, line in enumerate(handle):
                if limit is not None and position >= limit:
                    return
                line = line.strip()
                if line:
                    yield json.loads(line)


def generate_campaign(
    root: str | Path,
    config: DatasetConfig,
    count: int,
    *,
    start: int = 0,
    distribution_set: DistributionSet | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper: write a campaign and its manifest in one call."""
    writer = CampaignWriter(root, config, distribution_set=distribution_set)
    report = writer.generate(count, start=start, resume=resume)
    writer.write_manifest()
    report["manifest"] = writer.manifest()
    return report


class EpisodeDataset:
    """Read access to a campaign: records, coverage, decision samples."""

    def __init__(self, root: str | Path, *, campaign_id: str | None = None) -> None:
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.exists():
            raise DatasetError(f"{self.root} has no {MANIFEST_NAME}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if campaign_id is not None and campaign_id != self.manifest["campaign_id"]:
            raise DatasetError(
                f"campaign mismatch: dataset is {self.manifest['campaign_id']!r}, "
                f"caller asked for {campaign_id!r}"
            )
        if self.manifest["dataset_version"] != DATASET_VERSION:
            raise DatasetError(
                f"dataset version {self.manifest['dataset_version']!r} is not {DATASET_VERSION!r}"
            )
        self.feature_schema = self.manifest["schema"]["feature_schema"]
        self._cache: dict[str, list[dict[str, Any]]] = {}

    # --- loading ----------------------------------------------------------------------
    def episodes(self, split: str) -> list[dict[str, Any]]:
        if split not in SPLIT_NAMES:
            raise DatasetError(f"unknown split {split!r}; known: {list(SPLIT_NAMES)}")
        if split not in self._cache:
            self._cache[split] = list(_iter_split(self.root, split))
        return self._cache[split]

    def summary(self) -> dict[str, Any]:
        return {
            "campaign_id": self.manifest["campaign_id"],
            "counts": self.manifest["counts"],
            "total_episodes": self.manifest["total_episodes"],
            "families": _family_counts(self.manifest["coverage"]),
            "expert": self.manifest["expert"]["key"],
            "leakage_clean": self.manifest["leakage"]["clean"],
        }

    def decision_samples(self, split: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Flatten episodes into supervised decision samples (torch-free)."""
        samples: list[dict[str, Any]] = []
        for episode in self.episodes(split):
            context = episode.get("parameter_context") or {}
            for step in episode.get("steps", []):
                observation = step.get("observation")
                if observation is None:
                    continue
                samples.append({
                    "episode_id": episode["episode_id"],
                    "family": episode.get("family"),
                    "condition_class": episode.get("condition_class"),
                    "time_s": step["time_s"],
                    "observation": observation,
                    "targets": [list(pair) for pair in step.get("targets", [])],
                    "reward": step.get("reward", 0.0),
                    "terms": dict(step.get("terms", {})),
                    "context": context,
                })
                if limit is not None and len(samples) >= limit:
                    return samples
        return samples


def _iter_split(root: Path, split: str, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    path = root / f"episodes-{split}.jsonl.gz"
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if limit is not None and position >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def _family_counts(coverage: Mapping[str, Any]) -> dict[str, int]:
    return dict(coverage.get("scenario_families", {})) if coverage else {}


def load_dataset(root: str | Path, **kwargs: Any) -> EpisodeDataset:
    return EpisodeDataset(root, **kwargs)


__all__ = [
    "DATASET_VERSION",
    "MANIFEST_NAME",
    "INDEX_NAME",
    "DatasetError",
    "DatasetConfig",
    "CampaignWriter",
    "EpisodeDataset",
    "generate_campaign",
    "load_dataset",
    "episode_record",
    "compact_observation",
]
