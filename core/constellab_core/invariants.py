"""Day 33: declarative invariant monitoring and allocation comparisons.

An invariant is **configuration**, not a code change: a :class:`InvariantRule`
names a known evaluator, its threshold parameters, a severity and the unit those
parameters are expressed in. Findings are structured records (rule key, time,
entity ids, evidence) rather than free-form strings, so the UI, the CLI report
and tests can all consume the same facts.

Shipped rules:

* ``duplicate-task-claim`` — two satellites proposed the same task at the same
  decision (the G2 failure; requires the claim visibility bug to fire);
* ``assignment-expired`` — a task expired while still assigned to a satellite;
* ``missed-deadline`` — an urgent task reached its deadline without completion;
* ``invalid-policy-action`` — the harness rejected a proposal (unknown task,
  busy satellite, illegal transition …); each rejection is attributable;
* ``failed-satellite-held-task`` — a task was assigned across a satellite
  failure window, i.e. the fault actually cost work;
* ``resource-budget`` — summed observation energy on one satellite exceeded its
  configured budget.

``compare_allocations`` runs the transparent baselines through the *same*
harness, fault plan and rule set, and reports metrics plus finding counts. It is
the Day 33 baseline comparison, not a learned-policy evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .policy import (
    DistributedGreedyPolicy,
    EpisodeResult,
    FixedPriorityPolicy,
    LeasePolicy,
    LocalGreedyPolicy,
    Scene,
    run_episode,
)

INVARIANT_VERSION = "invariants-1"


class InvariantError(ValueError):
    """An invalid rule definition (unknown evaluator, bad parameters)."""


_KNOWN_EVALUATORS: frozenset[str] = frozenset({
    "duplicate_task_claim",
    "assignment_expired",
    "missed_deadline",
    "invalid_policy_action",
    "failed_satellite_held_task",
    "resource_budget",
})


@dataclass(frozen=True)
class InvariantRule:
    key: str
    evaluator: str
    description: str
    severity: str = "warning"
    unit: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or not self.evaluator:
            raise InvariantError("a rule needs a key and an evaluator")
        if self.severity not in ("info", "warning", "error"):
            raise InvariantError("severity must be info, warning or error")
        if self.evaluator not in _KNOWN_EVALUATORS:
            raise InvariantError(
                f"unknown invariant evaluator {self.evaluator!r}; "
                f"known: {', '.join(sorted(_KNOWN_EVALUATORS))}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "evaluator": self.evaluator,
            "description": self.description,
            "severity": self.severity,
            "unit": self.unit,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    time_s: float
    message: str
    entities: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "time_s": round(float(self.time_s), 6),
            "message": self.message,
            "entities": list(self.entities),
            "evidence": dict(self.evidence),
        }


DEFAULT_RULES: tuple[InvariantRule, ...] = (
    InvariantRule(
        key="duplicate-task-claim",
        evaluator="duplicate_task_claim",
        description="Two or more satellites proposed the same task at the same decision.",
        severity="error",
        unit="satellites",
        params={"min_satellites": 2},
    ),
    InvariantRule(
        key="assignment-expired",
        evaluator="assignment_expired",
        description="A task expired while a satellite still owned it.",
        severity="error",
    ),
    InvariantRule(
        key="missed-deadline",
        evaluator="missed_deadline",
        description="An urgent task reached its deadline without a completed observation.",
        severity="warning",
        unit="urgency",
        params={"min_urgency": 4},
    ),
    InvariantRule(
        key="invalid-policy-action",
        evaluator="invalid_policy_action",
        description="The harness rejected a proposed action.",
        severity="info",
    ),
    InvariantRule(
        key="failed-satellite-held-task",
        evaluator="failed_satellite_held_task",
        description="A task was assigned to a satellite whose failure window covered the assignment.",
        severity="error",
    ),
    InvariantRule(
        key="resource-budget",
        evaluator="resource_budget",
        description="Observation energy on one satellite exceeded its configured budget.",
        severity="warning",
        unit="J",
        params={"energy_budget_j": 5.0e5},
    ),
)


def default_rules() -> tuple[InvariantRule, ...]:
    return DEFAULT_RULES


def rules_payload(rules: Sequence[InvariantRule] | None = None) -> dict[str, Any]:
    selected = DEFAULT_RULES if rules is None else rules
    return {
        "version": INVARIANT_VERSION,
        "rules": [rule.to_dict() for rule in selected],
        "evaluators": sorted(_KNOWN_EVALUATORS),
    }


# --- evaluators ---------------------------------------------------------------------

def _duplicate_task_claim(result: EpisodeResult, rule: InvariantRule, faults: Any) -> list[Finding]:
    minimum = int(rule.params.get("min_satellites", 2))
    grouped: dict[tuple[float, str], set[str]] = {}
    for decision in result.decisions:
        for proposal in decision.proposals:
            if proposal.get("action") != "assign":
                continue
            key = (decision.time_s, proposal["task_id"])
            grouped.setdefault(key, set()).add(proposal["satellite_id"])
    findings: list[Finding] = []
    for (time_s, task_id), satellites in sorted(grouped.items()):
        if len(satellites) < minimum:
            continue
        findings.append(Finding(
            rule=rule.key,
            severity=rule.severity,
            time_s=time_s,
            message=(
                f"{len(satellites)} satellites proposed {task_id} at t={time_s:.0f} s; "
                "their claims never crossed the partition"
            ),
            entities=tuple(sorted(satellites)) + (task_id,),
            evidence={"task_id": task_id, "satellites": sorted(satellites)},
        ))
    return findings


def _assignment_expired(result: EpisodeResult, rule: InvariantRule, faults: Any) -> list[Finding]:
    findings: list[Finding] = []
    for transition in result.transitions:
        if transition.get("action") != "expire" or not transition.get("accepted"):
            continue
        if transition.get("from") not in ("assigned", "in-progress"):
            continue
        if not transition.get("satellite_id"):
            continue
        findings.append(Finding(
            rule=rule.key,
            severity=rule.severity,
            time_s=float(transition["at_s"]),
            message=(
                f"{transition['task_id']} expired while assigned to "
                f"{transition['satellite_id']}"
            ),
            entities=(transition["task_id"], transition["satellite_id"]),
            evidence=dict(transition),
        ))
    return findings


def _missed_deadline(result: EpisodeResult, rule: InvariantRule, faults: Any) -> list[Finding]:
    minimum = int(rule.params.get("min_urgency", 4))
    findings: list[Finding] = []
    for task in result.tasks:
        if task.get("state") != "expired":
            continue
        if int(task.get("urgency", 0)) < minimum:
            continue
        findings.append(Finding(
            rule=rule.key,
            severity=rule.severity,
            time_s=float(task["deadline_s"]),
            message=(
                f"{task['id']} (urgency {task['urgency']}) missed its deadline "
                f"at t={task['deadline_s']:.0f} s"
            ),
            entities=(task["id"], task["target_id"]),
            evidence=dict(task),
        ))
    return findings


def _invalid_policy_action(result: EpisodeResult, rule: InvariantRule, faults: Any) -> list[Finding]:
    findings: list[Finding] = []
    for decision in result.decisions:
        for rejected in decision.rejected:
            findings.append(Finding(
                rule=rule.key,
                severity=rule.severity,
                time_s=decision.time_s,
                message=f"rejected {rejected.get('action')} for {rejected.get('task_id')}: {rejected.get('reason')}",
                entities=tuple(filter(None, (
                    rejected.get("satellite_id"), rejected.get("task_id"),
                ))),
                evidence=dict(rejected),
            ))
    return findings


def _failed_satellite_held_task(result: EpisodeResult, rule: InvariantRule, faults: Any) -> list[Finding]:
    if faults is None:
        return []
    # Assignment time comes from the first accepted assign transition; the end
    # is the closing transition (complete/expire/cancel) or the episode horizon.
    assign_at: dict[str, tuple[float, str]] = {}
    close_at: dict[str, float] = {}
    for transition in result.transitions:
        task_id = transition["task_id"]
        if transition.get("action") == "assign" and transition.get("accepted"):
            assign_at.setdefault(task_id, (float(transition["at_s"]), transition["satellite_id"] or ""))
        if transition.get("action") in ("complete", "expire", "cancel") and transition.get("accepted"):
            close_at.setdefault(task_id, float(transition["at_s"]))
    findings: list[Finding] = []
    for task in result.tasks:
        task_id = task["id"]
        if task_id not in assign_at:
            continue
        start, satellite = assign_at[task_id]
        end = close_at.get(task_id, float(task["deadline_s"]))
        for fault in faults.satellite_faults(satellite):
            if fault.kind != "satellite-failure":
                continue
            if fault.end_s <= start or fault.start_s >= end:
                continue
            findings.append(Finding(
                rule=rule.key,
                severity=rule.severity,
                time_s=max(start, fault.start_s),
                message=(
                    f"{satellite} lost {task_id} during a failure that began at "
                    f"t={fault.start_s:.0f} s"
                ),
                entities=(satellite, task_id, fault.id),
                evidence={
                    "fault": fault.to_dict(),
                    "assigned_s": start,
                    "closed_s": end,
                },
            ))
    return findings


def _resource_budget(result: EpisodeResult, rule: InvariantRule, faults: Any) -> list[Finding]:
    budget = float(rule.params.get("energy_budget_j", 5.0e5))
    energy: dict[str, float] = {}
    for outcome in result.outcomes:
        if not outcome.get("accepted"):
            continue
        satellite = outcome["satellite_id"]
        energy[satellite] = energy.get(satellite, 0.0) + float(outcome.get("energy_j", 0.0))
    findings: list[Finding] = []
    for satellite, used in sorted(energy.items()):
        if used <= budget:
            continue
        findings.append(Finding(
            rule=rule.key,
            severity=rule.severity,
            time_s=0.0,
            message=f"{satellite} spent {used:.0f} J of observation energy over a {budget:.0f} J budget",
            entities=(satellite,),
            evidence={"energy_j": round(used, 6), "budget_j": budget},
        ))
    return findings


_EVALUATORS: dict[str, Callable[[EpisodeResult, InvariantRule, Any], list[Finding]]] = {
    "duplicate_task_claim": _duplicate_task_claim,
    "assignment_expired": _assignment_expired,
    "missed_deadline": _missed_deadline,
    "invalid_policy_action": _invalid_policy_action,
    "failed_satellite_held_task": _failed_satellite_held_task,
    "resource_budget": _resource_budget,
}


class InvariantEngine:
    """Runs a rule set over one episode result and returns ordered findings."""

    def __init__(self, rules: Sequence[InvariantRule] | None = None) -> None:
        self.rules = tuple(DEFAULT_RULES if rules is None else rules)
        keys = [rule.key for rule in self.rules]
        if len(set(keys)) != len(keys):
            raise InvariantError("invariant rule keys must be unique")

    def evaluate(self, result: EpisodeResult, *, faults: Any = None) -> list[Finding]:
        findings: list[Finding] = []
        for rule in self.rules:
            findings.extend(_EVALUATORS[rule.evaluator](result, rule, faults))
        return sorted(findings, key=lambda finding: (finding.time_s, finding.rule, finding.entities))

    def summary(self, result: EpisodeResult, *, faults: Any = None) -> dict[str, Any]:
        findings = self.evaluate(result, faults=faults)
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding.rule] = counts.get(finding.rule, 0) + 1
        return {
            "version": INVARIANT_VERSION,
            "findings": [finding.to_dict() for finding in findings],
            "counts": dict(sorted(counts.items())),
            "violations": len(findings),
            "errors": sum(1 for finding in findings if finding.severity == "error"),
        }


def evaluate_invariants(
    result: EpisodeResult,
    *,
    faults: Any = None,
    rules: Sequence[InvariantRule] | None = None,
) -> dict[str, Any]:
    return InvariantEngine(rules).summary(result, faults=faults)


# --- comparisons --------------------------------------------------------------------

def compare_allocations(
    scene: Scene,
    *,
    faults: Any = None,
    policies: Sequence[Any] | None = None,
    rules: Sequence[InvariantRule] | None = None,
) -> dict[str, Any]:
    """Run greedy, leases-only and deterministic priority through one harness."""
    from .comms import ClaimBus
    from .trace import TraceRecorder, stable_hash

    selected = list(policies or (
        LocalGreedyPolicy(), LeasePolicy(), FixedPriorityPolicy(),
    ))
    engine = InvariantEngine(rules)
    entries: dict[str, Any] = {}
    for policy in selected:
        bus = ClaimBus(
            [satellite.id for satellite in scene.config.satellites], faults=faults,
        ) if hasattr(policy, "act_agent") else None
        recorder = TraceRecorder(label=f"allocation:{policy.metadata.key}")
        result = run_episode(scene, policy, faults=faults, comms=bus, trace=recorder)
        summary = engine.summary(result, faults=faults)
        entries[policy.metadata.key] = {
            "policy": result.policy,
            "metrics": result.metrics,
            "invariants": summary,
            "trace_hash": recorder.hash(),
            "trace_digest": stable_hash(result.to_dict()),
        }
    return {
        "version": INVARIANT_VERSION,
        "episode": {
            "horizon_s": scene.horizon_s,
            "step_s": scene.step_s,
            "seed": scene.seed,
        },
        "faults": None if faults is None else faults.to_dict(),
        "policies": entries,
    }
