"""Day 48: an independent runtime safety shield for learned swarm policies.

The learned policy only *proposes*; this module decides whether a proposal may
reach the simulator. It is deliberately separate from the model: it reads only
the observation contract and a small declarative policy, never model logits or
simulator ground truth. A model can not disable it, and a rejected action is
recorded with a stable reason code together with the fallback that was used.

The shield is an *operational constraint* mechanism, not a physical flight
safety proof. It checks what the observation can certify: the mask, satellite
availability, task state/deadline, duplicate assignments and resource floors.
When a value is absent the verdict is ``blocked`` rather than a fabricated
approval, matching the project rule that missing data is never zero.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import SATELLITE_FEATURES, TASK_FEATURES, feature_index
from .env import ACTION_ASSIGN, ACTION_WAIT, Action, EnvPolicy

SHIELD_VERSION = "safety-shield-1"
REASON_CODES: tuple[str, ...] = (
    "unknown-satellite",
    "unknown-task",
    "unknown-kind",
    "not-in-mask",
    "satellite-unavailable",
    "satellite-busy",
    "task-not-available",
    "deadline-too-close",
    "resource-unknown",
    "energy-floor",
    "storage-floor",
    "duplicate-assignment",
    "max-assignments",
)

SAT_AVAILABLE = feature_index("satellite", "available")
SAT_BUSY = feature_index("satellite", "busy")
SAT_ENERGY = feature_index("satellite", "energy_fraction")
SAT_STORAGE = feature_index("satellite", "storage_fraction")
TASK_DEADLINE = feature_index("task", "deadline_norm")
TASK_OPEN = feature_index("task", "is_open")


class ShieldError(RuntimeError):
    """The shield configuration or an observation is outside its contract."""


@dataclass(frozen=True)
class ShieldConfig:
    """Declarative constraints the shield enforces on every proposal."""

    min_energy_fraction: float = 0.10
    min_storage_fraction: float = 0.0
    deadline_margin_s: float = 0.0
    max_assignments_per_step: int = 8
    require_energy_certified: bool = True
    fallback: str = "wait"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_energy_fraction <= 1.0:
            raise ShieldError("min_energy_fraction must be within [0, 1]")
        if not 0.0 <= self.min_storage_fraction <= 1.0:
            raise ShieldError("min_storage_fraction must be within [0, 1]")
        if self.deadline_margin_s < 0.0:
            raise ShieldError("deadline_margin_s must be non-negative")
        if self.max_assignments_per_step < 0:
            raise ShieldError("max_assignments_per_step must be non-negative")
        if self.fallback not in ("wait", "none"):
            raise ShieldError("fallback must be 'wait' or 'none'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SHIELD_VERSION,
            "min_energy_fraction": self.min_energy_fraction,
            "min_storage_fraction": self.min_storage_fraction,
            "deadline_margin_s": self.deadline_margin_s,
            "max_assignments_per_step": self.max_assignments_per_step,
            "require_energy_certified": self.require_energy_certified,
            "fallback": self.fallback,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ShieldConfig":
        return cls(
            min_energy_fraction=float(payload.get("min_energy_fraction", 0.10)),
            min_storage_fraction=float(payload.get("min_storage_fraction", 0.0)),
            deadline_margin_s=float(payload.get("deadline_margin_s", 0.0)),
            max_assignments_per_step=int(payload.get("max_assignments_per_step", 8)),
            require_energy_certified=bool(payload.get("require_energy_certified", True)),
            fallback=str(payload.get("fallback", "wait")),
        )


@dataclass
class ShieldDecision:
    """What the shield did with one policy proposal."""

    time_s: float = 0.0
    proposed: list[dict[str, Any]] = field(default_factory=list)
    accepted: list[Action] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    fallback: list[dict[str, Any]] = field(default_factory=list)
    reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SHIELD_VERSION,
            "time_s": round(float(self.time_s), 6),
            "proposed": list(self.proposed),
            "accepted": [action.to_dict() for action in self.accepted],
            "rejected": list(self.rejected),
            "fallback": list(self.fallback),
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


class SafetyShield:
    """Screen proposed actions against the observation's own masked contract."""

    version = SHIELD_VERSION

    def __init__(self, config: ShieldConfig | None = None) -> None:
        self.config = config or ShieldConfig()
        self.decisions: list[ShieldDecision] = []

    def reset(self) -> None:
        self.decisions = []

    def _reject(self, decision: ShieldDecision, action: Action, code: str, message: str) -> None:
        decision.rejected.append({
            "action": action.to_dict(),
            "code": code,
            "message": message,
        })
        decision.reason_counts[code] = decision.reason_counts.get(code, 0) + 1

    def screen(
        self,
        observation: Mapping[str, Any],
        actions: Sequence[Action | Mapping[str, Any]],
    ) -> ShieldDecision:
        """Return the subset of ``actions`` that may reach the simulator."""
        satellites = list(observation.get("satellite_ids") or [])
        tasks = list(observation.get("task_ids") or [])
        satellite_index = {identifier: index for index, identifier in enumerate(satellites)}
        task_index = {identifier: index for index, identifier in enumerate(tasks)}
        try:
            mask = {
                (int(pair[0]), int(pair[1]))
                for pair in observation["mask"]["assign"]
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ShieldError("observation has no legal action mask") from exc
        satellite_rows = list(observation.get("satellites") or [])
        task_rows = list(observation.get("tasks") or [])
        time_s = float(observation.get("time_s", 0.0))
        decision = ShieldDecision(time_s=time_s)
        used_satellites: set[int] = set()
        used_tasks: set[int] = set()

        for raw in actions:
            action = raw if isinstance(raw, Action) else Action.from_dict(raw)
            decision.proposed.append(action.to_dict())
            if action.kind == ACTION_WAIT:
                continue
            if action.kind != ACTION_ASSIGN:
                self._reject(decision, action, "unknown-kind",
                             f"action kind {action.kind!r} is not permitted")
                continue
            if not action.satellite_id or not action.task_id:
                self._reject(decision, action, "unknown-satellite",
                             "an assign action needs a satellite and a task")
                continue
            if action.satellite_id not in satellite_index:
                self._reject(decision, action, "unknown-satellite",
                             f"satellite {action.satellite_id!r} is not in this observation")
                continue
            if action.task_id not in task_index:
                self._reject(decision, action, "unknown-task",
                             f"task {action.task_id!r} is not in this observation")
                continue
            satellite_position = satellite_index[action.satellite_id]
            task_position = task_index[action.task_id]
            if (satellite_position, task_position) not in mask:
                self._reject(decision, action, "not-in-mask",
                             "the pair is not in the current legal action mask")
                continue
            if task_position >= len(task_rows) or satellite_position >= len(satellite_rows):
                self._reject(decision, action, "not-in-mask",
                             "the observation rows do not match its id lists")
                continue
            satellite = satellite_rows[satellite_position]
            task = task_rows[task_position]
            if not bool(satellite.get("is_available", False)):
                self._reject(decision, action, "satellite-unavailable",
                             "the satellite is not available")
                continue
            if bool(satellite.get("busy", False)):
                self._reject(decision, action, "satellite-busy",
                             "the satellite already holds an assignment")
                continue
            if str(task.get("state")) != "available" or not bool(task.get("open", False)):
                self._reject(decision, action, "task-not-available",
                             f"task state is {task.get('state')!r}")
                continue
            deadline_s = task.get("deadline_s")
            if not isinstance(deadline_s, (int, float)) or isinstance(deadline_s, bool):
                self._reject(decision, action, "task-not-available",
                             "task deadline is missing")
                continue
            if float(deadline_s) - time_s <= self.config.deadline_margin_s:
                self._reject(decision, action, "deadline-too-close",
                             "the task deadline is inside the safety margin")
                continue
            energy = self._feature(satellite, SAT_ENERGY)
            storage = self._feature(satellite, SAT_STORAGE)
            if energy is None:
                if self.config.require_energy_certified:
                    self._reject(decision, action, "resource-unknown",
                                 "energy state is unavailable and can not be certified")
                    continue
            elif energy < self.config.min_energy_fraction:
                self._reject(decision, action, "energy-floor",
                             f"energy {energy:.3f} is below the {self.config.min_energy_fraction:.3f} floor")
                continue
            if storage is not None and storage < self.config.min_storage_fraction:
                self._reject(decision, action, "storage-floor",
                             f"storage {storage:.3f} is below the {self.config.min_storage_fraction:.3f} floor")
                continue
            if satellite_position in used_satellites or task_position in used_tasks:
                self._reject(decision, action, "duplicate-assignment",
                             "satellite or task is already used in this proposal")
                continue
            if len(decision.accepted) >= self.config.max_assignments_per_step:
                self._reject(decision, action, "max-assignments",
                             "the per-step assignment cap is reached")
                continue
            used_satellites.add(satellite_position)
            used_tasks.add(task_position)
            decision.accepted.append(action)

        if not decision.accepted and self.config.fallback == "wait":
            decision.fallback = [Action(kind=ACTION_WAIT).to_dict()]
        self.decisions.append(decision)
        return decision

    @staticmethod
    def _feature(row: Mapping[str, Any], index: int) -> float | None:
        available = row.get("available") or []
        values = row.get("features") or []
        if index >= len(values):
            return None
        if index < len(available) and float(available[index]) <= 0.0:
            return None
        try:
            value = float(values[index])
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, value))

    def summary(self) -> dict[str, Any]:
        proposed = sum(len(decision.proposed) for decision in self.decisions)
        accepted = sum(len(decision.accepted) for decision in self.decisions)
        rejected = sum(len(decision.rejected) for decision in self.decisions)
        counts: dict[str, int] = {}
        for decision in self.decisions:
            for code, count in decision.reason_counts.items():
                counts[code] = counts.get(code, 0) + count
        return {
            "version": SHIELD_VERSION,
            "config": self.config.to_dict(),
            "steps": len(self.decisions),
            "proposed": proposed,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": round(accepted / proposed, 6) if proposed else None,
            "reason_counts": dict(sorted(counts.items())),
        }


def _suite_pair(observation: Mapping[str, Any]) -> tuple[dict[str, Any], int, int]:
    payload = copy.deepcopy(dict(observation))
    pairs = list(((payload.get("mask") or {}).get("assign") or []))
    if not pairs:
        raise ShieldError("the shield suite needs an observation with a legal assign pair")
    return payload, int(pairs[0][0]), int(pairs[0][1])


def _case(
    name: str,
    description: str,
    decision: ShieldDecision,
    *,
    expect_accepted: int,
    expect_code: str | None = None,
    expect_fallback: bool = False,
) -> dict[str, Any]:
    codes = sorted(decision.reason_counts)
    fallback = [item.get("kind") for item in decision.fallback]
    checks = {
        "accepted": len(decision.accepted) == expect_accepted,
        "reason": expect_code is None or (expect_code in codes),
        "fallback": (bool(fallback) == expect_fallback),
    }
    return {
        "name": name,
        "description": description,
        "expected": {
            "accepted": expect_accepted,
            "reason": expect_code,
            "fallback": expect_fallback,
        },
        "observed": {
            "accepted": len(decision.accepted),
            "reason_counts": dict(sorted(decision.reason_counts.items())),
            "fallback": fallback,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def shield_suite(
    observation: Mapping[str, Any],
    *,
    config: ShieldConfig | Mapping[str, Any] | None = None,
    source: str = "supplied observation",
) -> dict[str, Any]:
    """Day 48/56: reproducible boundary suite over one real observation.

    Every case is a deliberate boundary crossing: a masked pair is accepted,
    and each rejection path (mask, availability, deadline, resource floor,
    duplicate, unknown kind, assignment cap) must produce its documented code
    and fallback. The final case wraps a hostile policy that tries to switch the
    shield off; the shield must still reject and record the proposal.
    """
    base = ShieldConfig.from_payload(config) if isinstance(config, Mapping) else (config or ShieldConfig())
    original, satellite_position, task_position = _suite_pair(observation)
    satellite_id = str(original["satellite_ids"][satellite_position])
    task_id = str(original["task_ids"][task_position])
    legal = Action(kind=ACTION_ASSIGN, satellite_id=satellite_id, task_id=task_id)
    cases: list[dict[str, Any]] = []

    def screen(payload: Mapping[str, Any] | None = None, *, actions=None, override=None) -> ShieldDecision:
        shield = SafetyShield(override or base)
        return shield.screen(payload or original, actions if actions is not None else [legal])

    cases.append(_case(
        "legal-assign",
        "a pair inside the mask with certified resources is accepted",
        screen(), expect_accepted=1,
    ))

    masked = copy.deepcopy(original)
    masked["mask"]["assign"] = []
    cases.append(_case(
        "not-in-mask", "a proposal outside the legal mask is rejected",
        screen(masked), expect_accepted=0, expect_code="not-in-mask", expect_fallback=True,
    ))

    busy = copy.deepcopy(original)
    busy["satellites"][satellite_position]["busy"] = True
    cases.append(_case(
        "satellite-busy", "a busy satellite may not take a second assignment",
        screen(busy), expect_accepted=0, expect_code="satellite-busy", expect_fallback=True,
    ))

    unavailable = copy.deepcopy(original)
    unavailable["satellites"][satellite_position]["is_available"] = False
    cases.append(_case(
        "satellite-unavailable", "an unavailable satellite is rejected",
        screen(unavailable), expect_accepted=0, expect_code="satellite-unavailable", expect_fallback=True,
    ))

    closed = copy.deepcopy(original)
    closed["tasks"][task_position]["state"] = "assigned"
    cases.append(_case(
        "task-not-available", "a task that is no longer open is rejected",
        screen(closed), expect_accepted=0, expect_code="task-not-available", expect_fallback=True,
    ))

    urgency = copy.deepcopy(original)
    urgency["tasks"][task_position]["deadline_s"] = float(original["time_s"]) + 1.0
    cases.append(_case(
        "deadline-too-close", "a deadline inside the safety margin is rejected",
        screen(urgency, override=ShieldConfig(deadline_margin_s=30.0)),
        expect_accepted=0, expect_code="deadline-too-close", expect_fallback=True,
    ))

    energy_index = SATELLITE_FEATURES.index("energy_fraction")
    low_energy = copy.deepcopy(original)
    low_energy["satellites"][satellite_position]["features"][energy_index] = 0.0
    cases.append(_case(
        "energy-floor", "an available but depleted satellite is rejected",
        screen(low_energy), expect_accepted=0, expect_code="energy-floor", expect_fallback=True,
    ))

    unknown_energy = copy.deepcopy(original)
    unknown_energy["satellites"][satellite_position]["available"][energy_index] = 0.0
    cases.append(_case(
        "resource-unknown", "uncertifiable energy is blocked, never approved",
        screen(unknown_energy), expect_accepted=0, expect_code="resource-unknown", expect_fallback=True,
    ))

    duplicate = Action(kind=ACTION_ASSIGN, satellite_id=satellite_id, task_id=task_id)
    cases.append(_case(
        "duplicate-assignment", "the same pair proposed twice is accepted once",
        screen(actions=[legal, duplicate]), expect_accepted=1, expect_code="duplicate-assignment",
    ))

    # Action construction validates the kind before the shield is reached; keep
    # that as an explicit contract case instead of fabricating an unreachable
    # shield path. A forged mapping must fail loudly, never pass silently.
    try:
        Action.from_dict({"kind": "disable-shield"})
    except Exception:
        invalid_kind_blocked = True
    else:  # pragma: no cover - the action contract is tested elsewhere
        invalid_kind_blocked = False
    cases.append({
        "name": "unknown-kind-contract",
        "description": "an invented action kind is rejected by the action contract",
        "expected": {"accepted": 0, "reason": "unknown-kind", "fallback": False},
        "observed": {"accepted": 0, "reason_counts": {}, "fallback": []},
        "checks": {"rejected_at_contract": invalid_kind_blocked},
        "passed": invalid_kind_blocked,
    })

    bogus = Action(kind=ACTION_ASSIGN, satellite_id="satellite-does-not-exist", task_id=task_id)
    cases.append(_case(
        "unknown-satellite", "an identifier outside the observation is rejected",
        screen(actions=[bogus]), expect_accepted=0, expect_code="unknown-satellite", expect_fallback=True,
    ))

    bogus_task = Action(kind=ACTION_ASSIGN, satellite_id=satellite_id, task_id="task-does-not-exist")
    cases.append(_case(
        "unknown-task", "an unknown task identifier is rejected",
        screen(actions=[bogus_task]), expect_accepted=0, expect_code="unknown-task", expect_fallback=True,
    ))

    storage_index = SATELLITE_FEATURES.index("storage_fraction")
    low_storage = copy.deepcopy(original)
    low_storage["satellites"][satellite_position]["features"][storage_index] = 0.0
    cases.append(_case(
        "storage-floor", "a storage floor blocks an otherwise legal pair",
        screen(low_storage, override=ShieldConfig(min_storage_fraction=0.5)),
        expect_accepted=0, expect_code="storage-floor", expect_fallback=True,
    ))

    cases.append(_case(
        "max-assignments", "a zero assignment cap rejects even a legal pair",
        screen(override=ShieldConfig(max_assignments_per_step=0)),
        expect_accepted=0, expect_code="max-assignments", expect_fallback=True,
    ))

    cases.append(_case(
        "no-fallback", "fallback 'none' suppresses the wait action but not the rejection",
        screen(masked, override=ShieldConfig(fallback="none")),
        expect_accepted=0, expect_code="not-in-mask", expect_fallback=False,
    ))

    class HostilePolicy(EnvPolicy):
        label = "hostile-policy"

        def act(self, observation: Mapping[str, Any]):
            return [Action(kind=ACTION_ASSIGN, satellite_id=satellite_id, task_id=task_id)]

        def metadata(self) -> dict[str, Any]:
            return {"key": self.label}

    wrapped = ShieldedPolicy(HostilePolicy(), SafetyShield(base))
    hostile_actions = wrapped.act(masked)
    hostile_summary = wrapped.shield.summary()
    hostile_passed = (
        [action.kind for action in hostile_actions] == [ACTION_WAIT]
        and hostile_summary["rejected"] == 1
        and hostile_summary["reason_counts"].get("not-in-mask") == 1
    )
    cases.append({
        "name": "hostile-policy",
        "description": "a policy can not disable the shield or return its own fallback",
        "expected": {"accepted": 0, "reason": "not-in-mask", "fallback": True},
        "observed": {
            "accepted": 0,
            "reason_counts": dict(hostile_summary["reason_counts"]),
            "fallback": [action.kind for action in hostile_actions],
        },
        "checks": {
            "accepted": not any(action.kind != ACTION_WAIT for action in hostile_actions),
            "reason": hostile_summary["reason_counts"].get("not-in-mask") == 1,
            "fallback": [action.kind for action in hostile_actions] == [ACTION_WAIT],
        },
        "passed": hostile_passed,
    })

    covered = sorted({
        code for case in cases for code in case["observed"]["reason_counts"]
    })
    failed = [case["name"] for case in cases if not case["passed"]]
    return {
        "version": SHIELD_VERSION,
        "source": source,
        "config": base.to_dict(),
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "passed": len(cases) - len(failed),
            "failed": failed,
            "reason_codes_covered": covered,
            "reason_codes_missing": [code for code in REASON_CODES if code not in covered],
            "fallback": base.fallback,
            "disable_via_policy_output": False,
            "unknown_kind_blocked_at_contract": invalid_kind_blocked,
        },
        "passed": not failed,
    }


class ShieldedPolicy(EnvPolicy):
    """Wrap any policy so every proposal passes through a :class:`SafetyShield`."""

    version = SHIELD_VERSION

    def __init__(self, policy: EnvPolicy, shield: SafetyShield | None = None) -> None:
        self.policy = policy
        self.shield = shield or SafetyShield()
        self.label = f"{getattr(policy, 'label', 'policy')}+shield"
        self._lifetime: dict[str, Any] = {
            "steps": 0, "proposed": 0, "accepted": 0, "rejected": 0,
            "reason_counts": {},
        }

    def _absorb(self) -> None:
        """Fold the finished episode into the lifetime counters."""
        current = self.shield.summary()
        self._lifetime["steps"] += int(current["steps"])
        self._lifetime["proposed"] += int(current["proposed"])
        self._lifetime["accepted"] += int(current["accepted"])
        self._lifetime["rejected"] += int(current["rejected"])
        counts = self._lifetime["reason_counts"]
        for code, count in current["reason_counts"].items():
            counts[code] = counts.get(code, 0) + int(count)

    def reset(self) -> None:
        self._absorb()
        self.policy.reset()
        self.shield.reset()

    def summary(self) -> dict[str, Any]:
        """Lifetime screening statistics across every episode this policy ran."""
        current = self.shield.summary()
        proposed = self._lifetime["proposed"] + int(current["proposed"])
        accepted = self._lifetime["accepted"] + int(current["accepted"])
        rejected = self._lifetime["rejected"] + int(current["rejected"])
        counts = dict(self._lifetime["reason_counts"])
        for code, count in current["reason_counts"].items():
            counts[code] = counts.get(code, 0) + int(count)
        return {
            "version": SHIELD_VERSION,
            "config": self.shield.config.to_dict(),
            "steps": self._lifetime["steps"] + int(current["steps"]),
            "proposed": proposed,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": round(accepted / proposed, 6) if proposed else None,
            "reason_counts": dict(sorted(counts.items())),
        }

    def act(self, observation: Mapping[str, Any]) -> list[Action]:
        proposed = self.policy.act(observation)
        decision = self.shield.screen(observation, proposed)
        if decision.accepted:
            return list(decision.accepted)
        if self.shield.config.fallback == "wait":
            return [Action(kind=ACTION_WAIT)]
        return []

    def metadata(self) -> dict[str, Any]:
        inner = self.policy.metadata() if hasattr(self.policy, "metadata") else {}
        return {
            **dict(inner),
            "key": self.label,
            "shield": self.summary(),
        }


__all__ = [
    "SHIELD_VERSION",
    "REASON_CODES",
    "ShieldError",
    "ShieldConfig",
    "ShieldDecision",
    "SafetyShield",
    "ShieldedPolicy",
    "shield_suite",
]
