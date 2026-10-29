"""Day 37: transparent expert policies that produce supervised demonstrations.

Two references are provided, and the difference between them is the point:

* :class:`RuleExpert` — a **deployable** scoring rule over the observation
  contract. It only reads features the policy is allowed to see (candidate edges
  from the Day 35 observation), assigns greedily by score with explicit
  tie-breaking, and abstains when nothing is feasible. It is the non-neural
  baseline the learned policies are compared against.
* :class:`WindowOrderExpert` — a **diagnostic** reference that is allowed to
  look at the episode's full access plan (the core's ground truth about the
  future). It exists to bound small scenarios and to show what an optimizer
  would do; it is marked ``deployable=False`` and imitation training refuses it
  unless a caller explicitly opts in.

Both record their decision process: for every decision round the candidate list
(every masked pair with its score), which candidate was chosen, how far the
choice fell from the best candidate, and why the rest were dropped. That record
is what a dataset stores next to the observation, so a student model can be
trained on *actions* while still being audited for leakage.

Student inputs are exactly the Day 35 observations; nothing here can add a field
to them, and a test asserts that the demonstration archive contains no future
event, ground-truth payload or reward field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from . import (
    EDGE_FEATURES,
    SATELLITE_FEATURES,
    TASK_FEATURES,
    feature_index,
)
from .env import ACTION_ASSIGN, ACTION_WAIT, Action, EnvPolicy

EXPERT_VERSION = "expert-1"
EXPERT_KIND_RULE = "rule"
EXPERT_KIND_DIAGNOSTIC = "diagnostic"

SAT_AVAILABLE = feature_index("satellite", "available")
SAT_BUSY = feature_index("satellite", "busy")
SAT_ENERGY = feature_index("satellite", "energy_fraction")
TASK_URGENCY = feature_index("task", "urgency_norm")
TASK_DEADLINE = feature_index("task", "deadline_norm")
TASK_ACTIVE = feature_index("task", "active")
EDGE_QUALITY = feature_index("edge", "peak_quality")
EDGE_TIME_TO_OPEN = feature_index("edge", "time_to_open_norm")
EDGE_DURATION = feature_index("edge", "duration_norm")
EDGE_RESOLUTION = feature_index("edge", "resolution_norm")


@dataclass(frozen=True)
class ScoreWeights:
    """Declarative scoring for the rule expert (no per-target special cases)."""

    quality: float = 1.0
    urgency: float = 0.4
    deadline: float = 0.6
    window_soon: float = 0.5
    duration: float = 0.2
    contention: float = 0.35
    resource: float = 0.05

    def to_dict(self) -> dict[str, float]:
        return {
            "quality": self.quality,
            "urgency": self.urgency,
            "deadline": self.deadline,
            "window_soon": self.window_soon,
            "duration": self.duration,
            "contention": self.contention,
            "resource": self.resource,
        }


@dataclass
class Candidate:
    """One scored, masked action candidate (all fields observable)."""

    satellite_index: int
    task_index: int
    satellite_id: str
    task_id: str
    score: float
    terms: dict[str, float] = field(default_factory=dict)
    chosen: bool = False
    dropped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "satellite_index": self.satellite_index,
            "task_index": self.task_index,
            "satellite_id": self.satellite_id,
            "task_id": self.task_id,
            "score": round(float(self.score), 6),
            "terms": {key: round(float(value), 6) for key, value in self.terms.items()},
            "chosen": bool(self.chosen),
            "dropped_reason": self.dropped_reason,
        }


def score_candidates(
    observation: Mapping[str, Any], weights: ScoreWeights | None = None
) -> list[Candidate]:
    """Score every masked pair from observation features alone."""
    active_weights = weights or ScoreWeights()
    satellites = observation["satellites"]
    tasks = observation["tasks"]
    edges = {
        (int(edge[0]), int(edge[1])): edge[2 : 2 + len(EDGE_FEATURES)]
        for edge in observation["edges"]["assign"]
    }
    # How many satellites could take each candidate task: a crowded task is
    # worth less, which is how the rule avoids pile-ups without a global plan.
    contention: dict[int, int] = {}
    for _, task_index in observation["mask"]["assign"]:
        contention[int(task_index)] = contention.get(int(task_index), 0) + 1

    candidates: list[Candidate] = []
    for pair in observation["mask"]["assign"]:
        satellite_index, task_index = int(pair[0]), int(pair[1])
        feature_row = edges.get((satellite_index, task_index))
        if feature_row is None:
            continue
        quality, time_to_open, duration, resolution = (
            float(feature_row[EDGE_QUALITY]),
            float(feature_row[EDGE_TIME_TO_OPEN]),
            float(feature_row[EDGE_DURATION]),
            float(feature_row[EDGE_RESOLUTION]),
        )
        satellite = satellites[satellite_index]["features"]
        task = tasks[task_index]["features"]
        terms = {
            "quality": active_weights.quality * quality,
            "resolution": -active_weights.quality * 0.05 * resolution,
            "urgency": active_weights.urgency * float(task[TASK_URGENCY]),
            "deadline": active_weights.deadline * (1.0 - float(task[TASK_DEADLINE])),
            "window_soon": active_weights.window_soon * (1.0 - time_to_open),
            "duration": active_weights.duration * duration,
            "active": 0.1 * float(task[TASK_ACTIVE]),
            "contention": -active_weights.contention * (contention.get(task_index, 1) - 1),
            "resource": active_weights.resource * (1.0 - float(satellite[SAT_ENERGY])),
        }
        candidates.append(
            Candidate(
                satellite_index=satellite_index,
                task_index=task_index,
                satellite_id=str(observation["satellite_ids"][satellite_index]),
                task_id=str(observation["task_ids"][task_index]),
                score=sum(terms.values()),
                terms=terms,
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.satellite_id, item.task_id))
    return candidates


def greedy_assignment(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Highest score first: one task per satellite, one satellite per task."""
    used_satellites: set[int] = set()
    used_tasks: set[int] = set()
    chosen: list[Candidate] = []
    for candidate in candidates:
        if candidate.satellite_index in used_satellites:
            candidate.dropped_reason = "satellite already assigned in this round"
            continue
        if candidate.task_index in used_tasks:
            candidate.dropped_reason = "task already claimed in this round"
            continue
        candidate.chosen = True
        used_satellites.add(candidate.satellite_index)
        used_tasks.add(candidate.task_index)
        chosen.append(candidate)
    return chosen


@dataclass
class DecisionRecord:
    """The expert's full decision trace for one step (dataset-side provenance)."""

    time_s: float
    candidate_count: int
    chosen: list[dict[str, Any]]
    abstained: bool
    abstain_reason: str | None
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": round(float(self.time_s), 6),
            "candidate_count": self.candidate_count,
            "chosen": list(self.chosen),
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "candidates": list(self.candidates),
        }


class _ExpertBase(EnvPolicy):
    kind = EXPERT_KIND_RULE
    deployable = True

    def __init__(self, weights: ScoreWeights | None = None) -> None:
        self.weights = weights or ScoreWeights()
        self.decisions: list[DecisionRecord] = []
        self.last_candidates: list[Candidate] = []

    def reset(self) -> None:
        self.decisions = []
        self.last_candidates = []

    # --- decision plumbing ------------------------------------------------------------
    def _plan(self, observation: Mapping[str, Any]) -> list[Candidate]:
        raise NotImplementedError

    def act(self, observation: Mapping[str, Any]) -> list[Action]:
        candidates = self._plan(observation)
        self.last_candidates = candidates
        chosen = greedy_assignment(candidates)
        if chosen:
            record = DecisionRecord(
                time_s=float(observation["time_s"]),
                candidate_count=len(candidates),
                chosen=[candidate.to_dict() for candidate in chosen],
                abstained=False,
                abstain_reason=None,
                candidates=[candidate.to_dict() for candidate in candidates],
            )
        else:
            reason = (
                "no masked candidate remains"
                if candidates
                else (
                    "action mask is empty"
                    if not observation["mask"]["assign"]
                    else "no scored candidate"
                )
            )
            record = DecisionRecord(
                time_s=float(observation["time_s"]),
                candidate_count=len(candidates),
                chosen=[],
                abstained=True,
                abstain_reason=reason,
                candidates=[candidate.to_dict() for candidate in candidates],
            )
        self.decisions.append(record)
        return [
            Action(
                kind=ACTION_ASSIGN,
                satellite_id=candidate.satellite_id,
                task_id=candidate.task_id,
                score=candidate.score,
            )
            for candidate in chosen
        ] or [Action(kind=ACTION_WAIT, score=0.0)]

    # --- reporting --------------------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        abstains = sum(1 for record in self.decisions if record.abstained)
        return {
            "key": self.label,
            "kind": self.kind,
            "deployable": self.deployable,
            "version": EXPERT_VERSION,
            "weights": self.weights.to_dict(),
            "decisions": len(self.decisions),
            "abstains": abstains,
            "candidates": sum(record.candidate_count for record in self.decisions),
        }


class RuleExpert(_ExpertBase):
    """Deployable scoring rule: observation in, scored masked actions out."""

    label = "expert-rule"
    version = EXPERT_VERSION
    kind = EXPERT_KIND_RULE
    deployable = True
    assumptions = (
        "reads only the Day 35 observation: masked candidate edges, task and satellite features",
        "greedy matching by score with explicit id tie-breaking; no global plan and no future knowledge",
        "abstains when the action mask is empty instead of inventing an action",
    )

    def _plan(self, observation: Mapping[str, Any]) -> list[Candidate]:
        return score_candidates(observation, self.weights)


class WindowOrderExpert(_ExpertBase):
    """Diagnostic reference that is allowed to know the episode's access plan."""

    label = "expert-window-order"
    version = EXPERT_VERSION
    kind = EXPERT_KIND_DIAGNOSTIC
    deployable = False
    assumptions = (
        "uses the core's full access plan (future windows) and must never be presented as deployable",
        "picks the task whose next usable window opens earliest, then breaks ties by score",
        "exists to bound small scenarios and to contrast against the deployable rule",
    )

    def __init__(self, environment: Any, weights: ScoreWeights | None = None) -> None:
        super().__init__(weights)
        self.environment = environment

    def _plan(self, observation: Mapping[str, Any]) -> list[Candidate]:
        candidates = score_candidates(observation, self.weights)
        scene = self.environment.scene
        time_s = float(observation["time_s"])
        order: dict[tuple[int, int], float] = {}
        for candidate in candidates:
            windows = scene.usable_windows(
                time_s=time_s,
                satellite_id=candidate.satellite_id,
                target_id=self.environment.board.get(candidate.task_id).target_id,
            )
            order[(candidate.satellite_index, candidate.task_index)] = (
                min(window.start_s for window in windows) if windows else float("inf")
            )
        candidates.sort(
            key=lambda item: (
                order.get((item.satellite_index, item.task_index), float("inf")),
                -item.score,
                item.satellite_id,
                item.task_id,
            )
        )
        return candidates


EXPERT_REGISTRY: dict[str, tuple[type, bool]] = {
    RuleExpert.label: (RuleExpert, True),
    WindowOrderExpert.label: (WindowOrderExpert, False),
}
EXPERT_KEYS: tuple[str, ...] = tuple(sorted(EXPERT_REGISTRY))
# Backwards/explicit alias used by the sweep and curriculum defaults.
DEFAULT_EXPERT = RuleExpert.label


def expert_factory(key: str, *, environment: Any = None, weights: ScoreWeights | None = None) -> EnvPolicy:
    """Build an expert by key (callers never branch on the key themselves)."""
    entry = EXPERT_REGISTRY.get(key)
    if entry is None:
        raise KeyError(f"unknown expert {key!r}; known: {', '.join(EXPERT_KEYS)}")
    policy_type, deployable = entry
    if policy_type is WindowOrderExpert:
        if environment is None:
            raise ValueError(f"expert {key!r} needs an environment to read the access plan")
        return WindowOrderExpert(environment, weights)
    del deployable
    return policy_type(weights)


def expert_catalog() -> list[dict[str, Any]]:
    """Describe every expert for reports (kind and deployability included)."""
    catalog: list[dict[str, Any]] = []
    for key in EXPERT_KEYS:
        policy_type, deployable = EXPERT_REGISTRY[key]
        catalog.append({
            "key": key,
            "kind": policy_type.kind,
            "deployable": deployable,
            "version": EXPERT_VERSION,
            "assumptions": list(getattr(policy_type, "assumptions", ())),
        })
    return catalog


def demonstration_targets(
    observation: Mapping[str, Any], actions: Sequence[Action]
) -> list[list[int]]:
    """Convert chosen actions into ``[satellite_index, task_index]`` targets."""
    satellite_index = {
        satellite_id: index for index, satellite_id in enumerate(observation["satellite_ids"])
    }
    task_index = {task_id: index for index, task_id in enumerate(observation["task_ids"])}
    targets: list[list[int]] = []
    for action in actions:
        if action.kind != ACTION_ASSIGN or action.satellite_id is None or action.task_id is None:
            continue
        if action.satellite_id not in satellite_index or action.task_id not in task_index:
            continue
        targets.append([satellite_index[action.satellite_id], task_index[action.task_id]])
    return targets


def leakage_scan(payload: Any) -> list[str]:
    """Names of hidden fields found anywhere in a demonstration payload.

    The student model must never receive the episode's ground truth, the
    simulator's task reward, or a future event schedule: those are exactly the
    keys this scan searches for.
    """
    forbidden = {"truth", "reward", "fault_plan", "expected_trace_hash", "future", "oracle"}
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key) in forbidden:
                    found.add(str(key))
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(payload)
    return sorted(found)


__all__ = [
    "EXPERT_VERSION",
    "EXPERT_KIND_RULE",
    "EXPERT_KIND_DIAGNOSTIC",
    "EXPERT_KEYS",
    "DEFAULT_EXPERT",
    "ScoreWeights",
    "Candidate",
    "DecisionRecord",
    "RuleExpert",
    "WindowOrderExpert",
    "expert_factory",
    "expert_catalog",
    "score_candidates",
    "greedy_assignment",
    "demonstration_targets",
    "leakage_scan",
]
