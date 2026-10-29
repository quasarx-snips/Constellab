"""Days 53/54: adversarial scenario mutations, objective and labeled search data.

The mutation grammar edits a **validated** episode specification, never a running
simulator. Every child has a deterministic id and a parent id, is revalidated
through the same config/target/fault contracts, and is evaluated by the shared
core policy harness plus the invariant engine. A failure means an invariant
produced an ``error`` finding, not that a metric "looked bad".

The search objective keeps its components visible: new failure, distinct
failure family, severity, executions-to-find and computational cost. The
dataset records failures and non-failures with provenance, so a ranker does not
learn only from positive examples.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from constellab_core import comms, faults as core_faults, policy as core_policy, trace as core_trace
from constellab_core.config import load_config
from constellab_core.invariants import InvariantEngine
from constellab_core.targets import EventTarget

from .dataset import EpisodeDataset
from .env import EpisodeSpec

MUTATION_VERSION = "scenario-mutations-1"
SEARCH_VERSION = "adversarial-search-1"
OBJECTIVE_VERSION = "search-objective-1"

MUTATION_OPERATORS: tuple[str, ...] = (
    "shift-fault",
    "extend-fault",
    "change-target",
    "combine-faults",
    "delay-delivery",
    "trigger-at-transition",
)


class MutationError(ValueError):
    """A mutation could not be built or did not validate."""


@dataclass
class Mutation:
    mutation_id: str
    parent_id: str
    operator: str
    spec: EpisodeSpec
    seed: int = 0
    description: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_spec: bool = True) -> dict[str, Any]:
        return {
            "version": MUTATION_VERSION,
            "mutation_id": self.mutation_id,
            "parent_id": self.parent_id,
            "operator": self.operator,
            "seed": self.seed,
            "description": self.description,
            "payload": dict(self.payload),
            "spec": self.spec.payload() if include_spec else None,
        }


def _faults(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in (payload.get("fault_plan") or {}).get("faults", [])]


def _new_fault(
    spec: EpisodeSpec,
    *,
    fault_id: str,
    kind: str,
    start_s: float,
    duration_s: float,
    target: str,
    peer: str | None = None,
    reason: str = "",
) -> dict[str, Any]:
    payload = {
        "id": fault_id,
        "kind": kind,
        "start_s": round(max(0.0, start_s), 6),
        "end_s": round(max(start_s + 1.0, start_s + duration_s), 6),
        "satellites": [target],
        "peers": [] if peer is None else [peer],
        "quality": 0.0 if kind == "sensor-degrade" else 1.0,
        "delay_s": 60.0 if kind == "message-delay" else 0.0,
        "reason": reason,
    }
    return payload


def _validate_spec(spec: EpisodeSpec) -> None:
    load_config(spec.config)
    for target in spec.targets:
        EventTarget.from_dict(target)
    if spec.fault_plan:
        core_faults.FaultPlan.from_dict(spec.fault_plan)
    if spec.horizon_s <= 0.0 or spec.step_s <= 0.0:
        raise MutationError("horizon_s and step_s must be positive")


def _normalise_plan(spec: EpisodeSpec, mutations: list[dict[str, Any]]) -> dict[str, Any]:
    plan = dict(spec.fault_plan or {"strategy": "manual", "seed": spec.seed, "generator": {}})
    ids = [str(item["id"]) for item in mutations]
    if len(set(ids)) != len(ids):
        raise MutationError("mutation produced duplicate fault ids")
    plan["faults"] = sorted(mutations, key=lambda item: (float(item["start_s"]), str(item["id"])))
    plan.setdefault("version", core_faults.FAULT_VERSION)
    plan["strategy"] = "manual"
    plan["seed"] = spec.seed
    plan.setdefault("generator", {"mutated": True})
    plan["generator"] = {**dict(plan.get("generator") or {}), "mutated": True}
    return plan


class MutationEngine:
    """Apply one registered mutation operator to a validated episode spec."""

    version = MUTATION_VERSION

    def __init__(self, operators: Sequence[str] = MUTATION_OPERATORS) -> None:
        unknown = [operator for operator in operators if operator not in MUTATION_OPERATORS]
        if unknown:
            raise MutationError(f"unknown mutation operators: {unknown}")
        self.operators = tuple(operators)

    def mutate(
        self,
        spec: EpisodeSpec,
        *,
        seed: int,
        operator: str | None = None,
    ) -> Mutation:
        rng = random.Random(seed)
        chosen = operator or rng.choice(self.operators)
        if chosen not in self.operators:
            raise MutationError(f"operator {chosen!r} is not enabled")
        payload = spec.payload()
        faults = _faults(payload)
        satellites = [str(item.get("id")) for item in payload["config"].get("satellites", [])]
        if len(satellites) < 2:
            raise MutationError("a swarm mutation needs at least two satellites")
        horizon = float(payload["horizon_s"])
        if not faults:
            start = min(horizon * 0.2, max(0.0, horizon - 60.0))
            target = rng.choice(satellites)
            peer = rng.choice([item for item in satellites if item != target])
            faults = [_new_fault(
                spec, fault_id="fault-00", kind="link-cut", start_s=start,
                duration_s=max(30.0, horizon * 0.2), target=target, peer=peer,
                reason="seed fault for a mutation",
            )]
        changed = copy.deepcopy(faults)
        description = ""
        mutation_payload: dict[str, Any] = {}
        if chosen == "shift-fault":
            index = rng.randrange(len(changed))
            fault = changed[index]
            duration = float(fault["end_s"]) - float(fault["start_s"])
            shift = rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.25) * horizon
            start = max(0.0, min(horizon - 1.0, float(fault["start_s"]) + shift))
            fault["start_s"] = round(start, 6)
            fault["end_s"] = round(min(horizon, start + duration), 6)
            description = f"shift {fault['id']} by {shift:+.1f}s"
            mutation_payload = {"fault_id": fault["id"], "shift_s": round(shift, 6)}
        elif chosen == "extend-fault":
            index = rng.randrange(len(changed))
            fault = changed[index]
            extension = rng.uniform(0.05, 0.3) * horizon
            fault["end_s"] = round(min(horizon, float(fault["end_s"]) + extension), 6)
            description = f"extend {fault['id']} by {extension:.1f}s"
            mutation_payload = {"fault_id": fault["id"], "extension_s": round(extension, 6)}
        elif chosen == "change-target":
            index = rng.randrange(len(changed))
            fault = changed[index]
            candidates = [item for item in satellites if item not in fault.get("satellites", [])]
            if not candidates:
                candidates = satellites
            target = rng.choice(candidates)
            old = list(fault.get("satellites") or [])
            fault["satellites"] = [target]
            if fault["kind"] in ("link-cut", "message-delay"):
                peers = [item for item in satellites if item != target]
                fault["peers"] = [rng.choice(peers)]
            description = f"retarget {fault['id']} from {old} to {target}"
            mutation_payload = {"fault_id": fault["id"], "target": target}
        elif chosen == "combine-faults":
            index = rng.randrange(len(changed))
            source = changed[index]
            new_id = f"fault-{len(changed):02d}"
            target = rng.choice(satellites)
            peer = rng.choice([item for item in satellites if item != target])
            kind = rng.choice(["link-cut", "sensor-degrade", "satellite-failure", "message-delay"])
            start = max(0.0, min(horizon - 2.0, float(source["start_s"]) + rng.uniform(0.0, 0.2) * horizon))
            changed.append(_new_fault(
                spec, fault_id=new_id, kind=kind, start_s=start,
                duration_s=max(30.0, min(horizon - start, horizon * 0.25)),
                target=target, peer=peer if kind in ("link-cut", "message-delay") else None,
                reason=f"combined with {source['id']}",
            ))
            description = f"combine {source['id']} with {kind} {new_id}"
            mutation_payload = {"source": source["id"], "new_fault": new_id, "kind": kind}
        elif chosen == "delay-delivery":
            index = rng.randrange(len(changed))
            source = changed[index]
            target = rng.choice(satellites)
            peer = rng.choice([item for item in satellites if item != target])
            new_id = f"fault-{len(changed):02d}"
            start = max(0.0, min(horizon - 2.0, float(source["start_s"])))
            changed.append(_new_fault(
                spec, fault_id=new_id, kind="message-delay", start_s=start,
                duration_s=max(30.0, min(horizon - start, horizon * 0.2)),
                target=target, peer=peer, reason="mutated delivery delay",
            ))
            description = f"add delivery delay {new_id}"
            mutation_payload = {"new_fault": new_id, "target": target, "peer": peer}
        elif chosen == "trigger-at-transition":
            target_row = rng.choice(payload.get("targets", []))
            transition = float(target_row.get("start_s", 0.0))
            index = rng.randrange(len(changed))
            fault = changed[index]
            duration = max(30.0, float(fault["end_s"]) - float(fault["start_s"]))
            fault["start_s"] = round(max(0.0, min(horizon - 1.0, transition)), 6)
            fault["end_s"] = round(min(horizon, float(fault["start_s"]) + duration), 6)
            description = f"move {fault['id']} to target transition at {transition:.1f}s"
            mutation_payload = {"fault_id": fault["id"], "transition_s": round(transition, 6)}
        else:  # pragma: no cover - guarded by the constructor
            raise MutationError(f"unhandled operator {chosen!r}")

        payload["fault_plan"] = _normalise_plan(spec, changed)
        payload["episode_id"] = f"{spec.episode_id}~m{seed}"
        payload["family"] = spec.family
        payload["label"] = f"{spec.label} mutation {chosen}"
        child = EpisodeSpec.from_payload(payload)
        _validate_spec(child)
        return Mutation(
            mutation_id=f"{spec.episode_id}~m{seed}",
            parent_id=spec.episode_id,
            operator=chosen,
            spec=child,
            seed=seed,
            description=description,
            payload=mutation_payload,
        )

    def offspring(
        self, spec: EpisodeSpec, *, count: int, seed: int
    ) -> list[Mutation]:
        mutations: list[Mutation] = []
        for index in range(count):
            try:
                mutations.append(self.mutate(spec, seed=seed + index * 7919))
            except Exception:
                continue
        return mutations


def evaluate_scenario(
    spec: EpisodeSpec,
    *,
    policy_key: str = "distributed-greedy",
    trace: bool = True,
) -> dict[str, Any]:
    """Run a spec through the core harness and return metrics + invariant findings."""
    config = load_config(spec.config)
    targets = [EventTarget.from_dict(item) for item in spec.targets]
    plan = core_faults.FaultPlan.from_dict(spec.fault_plan) if spec.fault_plan else core_faults.empty_plan()
    scene = core_policy.Scene.build(
        config, targets,
        horizon_s=spec.horizon_s, step_s=spec.step_s, seed=spec.seed,
        window_step_s=spec.window_step_s,
    )
    registry: dict[str, Any] = {
        "local-greedy": core_policy.LocalGreedyPolicy,
        "fixed-priority": core_policy.FixedPriorityPolicy,
        "leases-only": core_policy.LeasePolicy,
        "distributed-greedy": core_policy.DistributedGreedyPolicy,
    }
    if policy_key not in registry:
        raise MutationError(f"unknown policy {policy_key!r}; known: {sorted(registry)}")
    selected = registry[policy_key]()
    satellites = [str(satellite.id) for satellite in config.satellites]
    bus = comms.ClaimBus(satellites, faults=plan)
    recorder = core_trace.TraceRecorder(label=f"search:{spec.episode_id}") if trace else None
    result = core_policy.run_episode(
        scene, selected, faults=plan, comms=bus, trace=recorder,
    )
    findings = InvariantEngine().evaluate(result, faults=plan)
    errors = [finding for finding in findings if finding.severity == "error"]
    signature = "|".join(sorted({finding.rule for finding in errors})) or "none"
    return {
        "version": SEARCH_VERSION,
        "spec_hash": spec.spec_hash(),
        "policy": selected.metadata.key,
        "metrics": dict(result.metrics),
        "findings": {
            "counts": _count_rules(findings),
            "errors": len(errors),
            "violations": len(findings),
            "signature": signature,
            "items": [finding.to_dict() for finding in findings],
        },
        "failure": bool(errors),
        "trace_hash": recorder.hash() if recorder is not None else None,
    }


def _count_rules(findings: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.rule] = counts.get(finding.rule, 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class SearchObjective:
    """Visible components; no hidden scalar weights live in the search loop."""

    new_failure: float = 5.0
    distinct_failure: float = 2.0
    severity: float = 1.0
    quickness: float = 0.5
    cost: float = 0.05
    version: str = OBJECTIVE_VERSION

    def score(
        self,
        *,
        failure: bool,
        signature: str,
        new_signature: bool,
        errors: int,
        executions: int,
    ) -> tuple[float, dict[str, float]]:
        components = {
            "new_failure": self.new_failure if failure else 0.0,
            "distinct_failure": self.distinct_failure if failure and new_signature else 0.0,
            "severity": self.severity * min(5, errors),
            "quickness": self.quickness / max(1, executions),
            "cost": -self.cost * max(0, executions),
        }
        return round(sum(components.values()), 6), {key: round(value, 6) for key, value in components.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "new_failure": self.new_failure,
            "distinct_failure": self.distinct_failure,
            "severity": self.severity,
            "quickness": self.quickness,
            "cost": self.cost,
        }


@dataclass
class SearchRecord:
    mutation: Mutation
    valid: bool
    result: dict[str, Any] | None = None
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    error: str = ""
    execution: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SEARCH_VERSION,
            "mutation": self.mutation.to_dict(),
            "valid": self.valid,
            "error": self.error,
            "execution": self.execution,
            "score": self.score,
            "components": dict(self.components),
            "result": dict(self.result or {}),
        }


class SearchDataset:
    """Append-only labeled dataset with canonical deduplication."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.records: list[SearchRecord] = []
        self.seen: set[str] = set()
        self.duplicates = 0

    def add(self, record: SearchRecord) -> bool:
        key = record.mutation.spec.spec_hash()
        if key in self.seen:
            self.duplicates += 1
            return False
        self.seen.add(key)
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), sort_keys=False) + "\n")
        return True

    def summary(self) -> dict[str, Any]:
        failures = [record for record in self.records if record.result and record.result.get("failure")]
        signatures = {
            str(record.result.get("findings", {}).get("signature"))
            for record in failures
        }
        operators: dict[str, dict[str, int]] = {}
        for record in self.records:
            bucket = operators.setdefault(record.mutation.operator, {"evaluations": 0, "failures": 0})
            bucket["evaluations"] += 1
            if record.result and record.result.get("failure"):
                bucket["failures"] += 1
        return {
            "version": SEARCH_VERSION,
            "records": len(self.records),
            "valid": sum(1 for record in self.records if record.valid),
            "failures": len(failures),
            "failure_rate": round(len(failures) / len(self.records), 6) if self.records else 0.0,
            "distinct_signatures": sorted(signatures),
            "operators": dict(sorted(operators.items())),
            "duplicates_skipped": self.duplicates,
            "class_balance": {
                "failures": len(failures),
                "non_failures": len(self.records) - len(failures),
            },
        }


def run_search(
    config: Mapping[str, Any],
    output: str | Path | None = None,
    *,
    ranker: Callable[[Sequence[Mutation]], Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Run random or guided mutations against one base episode's family."""
    root = config.get("dataset_root")
    if not root:
        raise MutationError("search needs dataset_root")
    dataset = EpisodeDataset(root)
    split = str(config.get("split", "train"))
    index = int(config.get("base_index", 0))
    records = dataset.episodes(split)
    if not records:
        raise MutationError(f"campaign has no {split} episodes")
    base = EpisodeSpec.from_payload(records[index % len(records)]["spec"])
    budget = int(config.get("budget", 8))
    seed = int(config.get("seed", 53))
    mode = str(config.get("mode", "random"))
    engine = MutationEngine()
    objective = SearchObjective()
    dataset_out = SearchDataset(config.get("dataset_out"))
    pool: list[Mutation] = []
    if ranker is not None and mode == "guided":
        pool = engine.offspring(base, count=max(budget * 4, budget), seed=seed + 90_000)
        scores = list(ranker(pool))
        ranked = [
            mutation for _, mutation in sorted(
                zip(scores, pool), key=lambda item: (-item[0], item[1].mutation_id)
            )
        ]
        exploration = max(0, min(len(ranked), int(config.get("exploration", 2))))
        selected = ranked[: max(0, budget - exploration)] + ranked[-exploration:] if exploration else ranked[:budget]
    else:
        selected = engine.offspring(base, count=budget, seed=seed)
    seen_signatures: set[str] = set()
    for execution, mutation in enumerate(selected, start=1):
        try:
            result = evaluate_scenario(
                mutation.spec,
                policy_key=str(config.get("policy", "distributed-greedy")),
            )
            failure = bool(result["failure"])
            signature = str(result["findings"]["signature"])
            is_new = failure and signature not in seen_signatures
            score, components = objective.score(
                failure=failure,
                signature=signature,
                new_signature=is_new,
                errors=int(result["findings"]["errors"]),
                executions=execution,
            )
            if failure:
                seen_signatures.add(signature)
            record = SearchRecord(
                mutation=mutation, valid=True, result=result,
                score=score, components=components, execution=execution,
            )
        except Exception as exc:
            record = SearchRecord(
                mutation=mutation, valid=False, error=str(exc), execution=execution,
            )
        dataset_out.add(record)
    payload = {
        "version": SEARCH_VERSION,
        "mode": mode,
        "base_episode": base.episode_id,
        "base_family": base.family,
        "seed": seed,
        "budget": budget,
        "objective": objective.to_dict(),
        "records": [record.to_dict() for record in dataset_out.records],
        "summary": dataset_out.summary(),
        "selection_note": (
            "non-failures are retained; guided selection must use the same "
            "mutation space and evaluation budget as the random baseline"
        ),
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "MUTATION_VERSION",
    "SEARCH_VERSION",
    "OBJECTIVE_VERSION",
    "MUTATION_OPERATORS",
    "MutationError",
    "Mutation",
    "MutationEngine",
    "evaluate_scenario",
    "SearchObjective",
    "SearchRecord",
    "SearchDataset",
    "run_search",
]
