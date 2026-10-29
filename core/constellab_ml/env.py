"""Day 35: the scalable episode environment and its observation/action contract.

The environment wraps the simulation core with ``reset``/``step`` semantics.
It owns no physics of its own: orbits come from ``constellab_core.dynamics``,
access windows from ``sensors``, lifecycle from ``tasks``, component state from
``components``, faults from ``faults`` and claims from ``comms``. What this
module adds is the *decision interface* an ML policy needs:

* a structured observation with **stable ordering** (satellites and tasks are
  always sorted by id, so index ``i`` means the same entity on every step),
* **masked, variable-size entity collections** (the fleet size, the task count
  and the contact graph change per episode and per step),
* a bipartite ``assign`` relation plus a ``contact`` relation between
  satellites, both emitted as edge lists (never padded graphs),
* a masked action set: an action is ``assign(satellite, task)`` or ``wait``,
  and only pairs present in ``action_mask`` are legal,
* rewards **decomposed into named terms** before they are collapsed, and
* episode metadata: termination reason, counts, trace hash, replay manifest.

Ground-truth leakage is a tested invariant: ``EventTarget.truth``, task reward
payloads, the fault plan and any future event never appear in an observation.
Missing data is carried by ``*_available`` masks instead of being reported as a
fabricated zero.

The environment is Gymnasium-*shaped* (``reset``/``step`` returning the
standard tuples) without depending on Gymnasium: ``observation_space`` and
``action_space`` describe the contract as data.

The contact relation comes from the spec's declared link model (slant range
plus a per-pair duty cycle); it is a link-availability model, not a radio or
protocol simulation, and it is applied by the core ``ClaimBus`` exactly like a
fault-driven partition.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from constellab_core import comms, components, dynamics, faults, sensors, tasks, trace
from constellab_core.config import load_config
from constellab_core.targets import EventTarget

from . import (
    EDGE_FEATURES,
    ENV_VERSION,
    GLOBAL_FEATURES,
    OBSERVATION_VERSION,
    SATELLITE_FEATURES,
    TASK_FEATURES,
    feature_schema_payload,
)

EARTH_RADIUS_KM = 6378.137

# Reward term weights. Every term is logged separately so a training run can
# reweight them without re-generating data.
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "coverage": 1.0,
    "latency": 0.5,
    "duplicate_work": 0.5,
    "resource_use": 0.02,
    "task_loss": 1.0,
    "constraint_penalty": 0.25,
}
REWARD_TERMS: tuple[str, ...] = tuple(DEFAULT_REWARD_WEIGHTS)

# Action kinds.
ACTION_ASSIGN = "assign"
ACTION_WAIT = "wait"
ACTION_KINDS: tuple[str, ...] = (ACTION_ASSIGN, ACTION_WAIT)


class EnvError(RuntimeError):
    """The environment was used outside its contract."""


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class Action:
    """One proposed decision: task this satellite on this task, or wait."""

    kind: str = ACTION_WAIT
    satellite_id: str | None = None
    task_id: str | None = None
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise EnvError(f"action kind {self.kind!r} must be one of {list(ACTION_KINDS)}")
        if self.kind == ACTION_ASSIGN and (not self.satellite_id or not self.task_id):
            raise EnvError("an assign action needs both satellite_id and task_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "satellite_id": self.satellite_id,
            "task_id": self.task_id,
            "score": round(float(self.score), 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Action":
        return cls(
            kind=str(payload.get("kind", ACTION_WAIT)),
            satellite_id=payload.get("satellite_id"),
            task_id=payload.get("task_id"),
            score=_finite(payload.get("score", 0.0)),
        )


@dataclass
class EpisodeSpec:
    """Everything needed to rebuild one episode byte-for-byte."""

    episode_id: str
    config: dict[str, Any]
    targets: list[dict[str, Any]]
    horizon_s: float = 3_600.0
    step_s: float = 60.0
    seed: int = 0
    window_step_s: float = 30.0
    family: str = "nominal"
    label: str = ""
    fault_plan: dict[str, Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    parameter_registry: str = ""
    contact: dict[str, Any] | None = None
    versions: dict[str, str] = field(default_factory=dict)
    split: str = "train"

    def payload(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "family": self.family,
            "label": self.label,
            "split": self.split,
            "seed": self.seed,
            "horizon_s": self.horizon_s,
            "step_s": self.step_s,
            "window_step_s": self.window_step_s,
            "config": self.config,
            "targets": self.targets,
            "fault_plan": self.fault_plan,
            "contact": self.contact,
            "parameters": self.parameters,
            "parameter_registry": self.parameter_registry,
            "versions": dict(self.versions),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeSpec":
        return cls(
            episode_id=str(payload["episode_id"]),
            config=dict(payload["config"]),
            targets=[dict(item) for item in payload.get("targets", [])],
            horizon_s=_finite(payload.get("horizon_s", 3_600.0)),
            step_s=_finite(payload.get("step_s", 60.0)),
            seed=int(payload.get("seed", 0)),
            window_step_s=_finite(payload.get("window_step_s", 30.0)),
            family=str(payload.get("family", "nominal")),
            label=str(payload.get("label", "")),
            fault_plan=payload.get("fault_plan"),
            parameters=dict(payload.get("parameters", {})),
            parameter_registry=str(payload.get("parameter_registry", "")),
            contact=payload.get("contact"),
            versions=dict(payload.get("versions", {})),
            split=str(payload.get("split", "train")),
        )

    def spec_hash(self) -> str:
        return trace.stable_hash(self.payload())

    @property
    def fleet_size(self) -> int:
        return len(self.config.get("satellites", []))


@dataclass
class StepStats:
    """What one step did, decomposed into reward terms and counters."""

    time_s: float
    terms: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in REWARD_TERMS})
    observed: list[dict[str, Any]] = field(default_factory=list)
    expired: list[str] = field(default_factory=list)
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    waits: int = 0
    reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": round(self.time_s, 6),
            "terms": {key: round(value, 6) for key, value in self.terms.items()},
            "reward": round(self.reward, 6),
            "observed": list(self.observed),
            "expired": list(self.expired),
            "accepted": list(self.accepted),
            "rejected": list(self.rejected),
            "waits": self.waits,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StepStats":
        return cls(
            time_s=_finite(payload.get("time_s", 0.0)),
            terms={str(key): _finite(value) for key, value in dict(payload.get("terms", {})).items()},
            observed=[dict(item) for item in payload.get("observed", [])],
            expired=[str(item) for item in payload.get("expired", [])],
            accepted=[dict(item) for item in payload.get("accepted", [])],
            rejected=[dict(item) for item in payload.get("rejected", [])],
            waits=int(payload.get("waits", 0)),
            reward=_finite(payload.get("reward", 0.0)),
        )


@dataclass
class EpisodeOutcome:
    episode_id: str
    spec_hash: str
    trace_hash: str
    metrics: dict[str, Any]
    steps: list[StepStats]
    records: list[dict[str, Any]]
    manifest: dict[str, Any]

    def to_dict(self, *, include_records: bool = True) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "spec_hash": self.spec_hash,
            "trace_hash": self.trace_hash,
            "metrics": self.metrics,
            "steps": [step.to_dict() for step in self.steps],
            "records": [dict(record) for record in self.records] if include_records else [],
            "manifest": dict(self.manifest),
        }


class SwarmEnvironment:
    """Single-agent (centralized) episode environment over the swarm core."""

    def __init__(
        self,
        spec: EpisodeSpec,
        *,
        reward_weights: Mapping[str, float] | None = None,
        lookahead_s: float | None = None,
    ) -> None:
        unknown = sorted(set(reward_weights or {}) - set(REWARD_TERMS))
        if unknown:
            raise EnvError(f"unknown reward terms: {unknown}; known: {list(REWARD_TERMS)}")
        self.spec = spec
        weights = dict(DEFAULT_REWARD_WEIGHTS)
        weights.update({key: _finite(value) for key, value in (reward_weights or {}).items()})
        self.reward_weights = weights
        # How far ahead a policy may reason about access: an explicit, bounded
        # planning horizon, never "the future".
        self.lookahead_s = _finite(lookahead_s if lookahead_s is not None else spec.horizon_s)
        self.config = load_config(spec.config)
        self.targets: list[EventTarget] = [EventTarget.from_dict(item) for item in spec.targets]
        self.target_by_id = {target.id: target for target in self.targets}
        self.satellite_ids: list[str] = sorted(sat.id for sat in self.config.satellites)
        self.orbit_by_id = {sat.id: sat.orbit for sat in self.config.satellites}
        self.fault_plan = (
            faults.FaultPlan.from_dict(spec.fault_plan) if spec.fault_plan else faults.empty_plan()
        )
        self._reset_state()

    # --- construction helpers ---------------------------------------------------------
    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], **kwargs: Any) -> "SwarmEnvironment":
        return cls(EpisodeSpec.from_payload(payload), **kwargs)

    def _reset_state(self) -> None:
        self.scene = scenes = _scene_for(self.spec, self.config, self.targets)
        self.runtime = components.initial_runtime(self.config)
        self.bus = comms.ClaimBus(
            self.satellite_ids,
            faults=self.fault_plan,
            link=contact_predicate(self.spec.contact, self.orbit_by_id),
        )
        self.board = tasks.TaskBoard(scenes.fresh_tasks())
        self.recorder = trace.TraceRecorder(label=f"env:{self.spec.episode_id}")
        self.step_index = 0
        self.time_s = 0.0
        self._busy: set[str] = set()
        self._assignments: dict[str, str] = {}  # satellite -> task
        self._counts = {
            "observed": 0,
            "completed": 0,
            "expired": 0,
            "conflicts": 0,
            "invalid_actions": 0,
            "latency_s": 0.0,
        }
        self.step_records: list[dict[str, Any]] = []
        self.captured_observations: list[dict[str, Any]] = []
        self._done = False
        self._termination_reason: str | None = None

    # --- spaces (declared as data; no Gymnasium dependency) ---------------------------
    @property
    def observation_space(self) -> dict[str, Any]:
        return {
            "version": OBSERVATION_VERSION,
            "satellite_features": list(SATELLITE_FEATURES),
            "task_features": list(TASK_FEATURES),
            "edge_features": list(EDGE_FEATURES),
            "global_features": list(GLOBAL_FEATURES),
            "variable_size": ["satellites", "tasks", "edges"],
        }

    @property
    def action_space(self) -> dict[str, Any]:
        return {
            "kind": "masked-bipartite-assignment",
            "actions": ["assign(satellite, task)", "wait"],
            "mask": "observation['mask']['assign'] lists the legal pairs",
        }

    # --- episode lifecycle ------------------------------------------------------------
    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.spec = dataclasses.replace(self.spec, seed=int(seed))
        self._reset_state()
        self.recorder.record(
            "episode.start",
            0.0,
            {
                "episode_id": self.spec.episode_id,
                "family": self.spec.family,
                "seed": self.spec.seed,
                "satellites": len(self.satellite_ids),
                "tasks": len(self.scene.task_templates),
                "horizon_s": self.spec.horizon_s,
                "step_s": self.spec.step_s,
                "faults_hash": self.fault_plan.hash(),
            },
            cause="reset",
        )
        return self.observation(), self.metadata()

    def metadata(self) -> dict[str, Any]:
        return {
            "episode_id": self.spec.episode_id,
            "family": self.spec.family,
            "split": self.spec.split,
            "seed": self.spec.seed,
            "step_index": self.step_index,
            "time_s": round(self.time_s, 6),
            "horizon_s": self.spec.horizon_s,
            "step_s": self.spec.step_s,
            "fleet_size": len(self.satellite_ids),
            "task_count": len(self.board.tasks()),
            "lookahead_s": self.lookahead_s,
            "spec_hash": self.spec.spec_hash(),
            "termination_reason": self._termination_reason,
            "done": self._done,
        }

    @property
    def done(self) -> bool:
        return self._done

    def step(self, actions: Sequence[Action] | Sequence[Mapping[str, Any]]) -> tuple[
        dict[str, Any], float, bool, bool, dict[str, Any]
    ]:
        """Apply one decision round, advance the clock, return the next step."""
        if self._done:
            raise EnvError("episode already finished; call reset() before step()")
        parsed: list[Action] = []
        for item in actions:
            parsed.append(item if isinstance(item, Action) else Action.from_dict(item))
        stats = StepStats(time_s=round(self.time_s, 6))
        mask = self._action_mask()
        mask_pairs = {(sat, task) for sat, task in mask}
        legal_satellites = set(self.satellite_ids)

        for action in parsed:
            record = action.to_dict()
            if action.kind == ACTION_WAIT:
                stats.waits += 1
                continue
            assert action.satellite_id and action.task_id  # validated by Action
            if action.satellite_id not in legal_satellites:
                stats.rejected.append({**record, "reason": "unknown satellite"})
                stats.terms["constraint_penalty"] -= 1.0
                self._counts["invalid_actions"] += 1
                continue
            if (action.satellite_id, action.task_id) not in mask_pairs:
                reason = self._mask_reason(action.satellite_id, action.task_id)
                stats.rejected.append({**record, "reason": reason})
                if reason in ("task already assigned", "task is 'in-progress'", "task is 'completed'"):
                    stats.terms["duplicate_work"] -= 1.0
                    self._counts["conflicts"] += 1
                else:
                    stats.terms["constraint_penalty"] -= 1.0
                    self._counts["invalid_actions"] += 1
                continue
            self._assign(action, stats)

        # Advance the core: component pipeline, expiry, scheduled observations.
        self._advance(stats)

        reward = sum(
            self.reward_weights[name] * stats.terms.get(name, 0.0) for name in REWARD_TERMS
        )
        stats.reward = reward
        self.recorder.record("env.step", stats.time_s, stats.to_dict(), cause="decision")

        self.step_index += 1
        terminated = all(not task.open for task in self.board.tasks())
        truncated = self.step_index > int(self.spec.horizon_s // self.spec.step_s)
        if terminated:
            self._termination_reason = "all-tasks-terminal"
        elif truncated:
            self._termination_reason = "horizon"
        self._done = terminated or truncated
        info = {
            **self.metadata(),
            "terms": dict(stats.terms),
            "step": stats.to_dict(),
            "metrics": self.metrics(),
        }
        observation = self.observation()
        self.step_records.append({
            "time_s": stats.time_s,
            "step_index": self.step_index,
            "actions": [action.to_dict() for action in parsed],
            "terms": {key: round(value, 6) for key, value in stats.terms.items()},
            "reward": round(reward, 6),
            "metrics": self.metrics(),
        })
        return observation, reward, terminated, truncated, info

    # --- reward bookkeeping -----------------------------------------------------------
    def _assign(self, action: Action, stats: StepStats) -> None:
        assert action.satellite_id and action.task_id
        satellite_id, task_id = action.satellite_id, action.task_id
        task = self.board.get(task_id)
        assigned = self.board.apply(task_id, "assign", at_s=self.time_s, satellite_id=satellite_id)
        started = self.board.apply(task_id, "start", at_s=self.time_s)
        if not assigned.accepted or not started.accepted:
            reason = assigned.reason or started.reason
            stats.rejected.append({**action.to_dict(), "reason": reason})
            if "already assigned" in reason:
                # A duplicate claim inside one decision round is exactly the
                # failure mode the invariant engine watches for.
                stats.terms["duplicate_work"] -= 1.0
                self._counts["conflicts"] += 1
            else:
                stats.terms["constraint_penalty"] -= 1.0
                self._counts["invalid_actions"] += 1
            return
        self._busy.add(satellite_id)
        self._assignments[satellite_id] = task_id
        self.bus.record_claim(task_id, satellite_id, time_s=self.time_s, target_id=task.target_id)
        stats.accepted.append({**action.to_dict(), "task_id": task_id})
        self.recorder.record(
            "action.assign",
            self.time_s,
            {"satellite_id": satellite_id, "task_id": task_id, "target_id": task.target_id},
            cause="policy",
        )
        window = self._window_for(satellite_id, task.target_id)
        if window is not None:
            self._record_observation(task_id, window, satellite_id, stats)

    def _record_observation(self, task_id: str, window: Any, satellite_id: str, stats: StepStats) -> None:
        task = self.board.get(task_id)
        outcome = self.board.record_observation(
            task_id, window, satellite_id=satellite_id, time_s=max(self.time_s, window.start_s)
        )
        payload = outcome.to_dict()
        stats.terms["resource_use"] -= _clamp(_finite(outcome.cost))
        if not outcome.accepted:
            # An unsuccessful look (cloud, revisit gap, deadline) still costs the
            # attempt; it is not coverage and the task stays claimed for a retry.
            stats.rejected.append({
                "kind": ACTION_ASSIGN,
                "satellite_id": satellite_id,
                "task_id": task_id,
                "reason": "; ".join(outcome.reasons) or "look rejected",
            })
            self.recorder.record(
                "task.look-rejected", self.time_s, payload, cause="quality"
            )
            self._refresh_busy()
            return
        latency = max(0.0, outcome.time_s - task.created_s)
        stats.observed.append({
            "task_id": task_id,
            "target_id": task.target_id,
            "satellite_id": satellite_id,
            "confidence": round(_finite(outcome.confidence), 6),
            "latency_s": round(latency, 6),
        })
        stats.terms["coverage"] += _finite(outcome.confidence)
        stats.terms["latency"] -= latency / max(1.0, self.spec.horizon_s)
        self._counts["observed"] += 1
        self._counts["latency_s"] += latency
        self.recorder.record("task.observed", self.time_s, payload, cause="access-window")
        self._refresh_busy()

    def _refresh_busy(self) -> None:
        """Busy satellites are exactly those holding a live assignment."""
        self._busy = {
            task.assigned_to
            for task in self.board.tasks()
            if task.assigned_to and task.state in (tasks.TaskState.ASSIGNED, tasks.TaskState.IN_PROGRESS)
        }
        self._assignments = {
            task.assigned_to: task.id
            for task in self.board.tasks()
            if task.assigned_to and task.state in (tasks.TaskState.ASSIGNED, tasks.TaskState.IN_PROGRESS)
        }

    def _advance(self, stats: StepStats) -> None:
        # Windows that open inside the step just decided are serviced first: the
        # clock then moves and the next observation is taken at the new time.
        self._service_assignments(stats)
        step = self.spec.step_s
        self.time_s = round(self.time_s + step, 6)
        components.advance_runtime(self.config, self.runtime, time_s=self.time_s, dt_s=step)
        for transition in self.board.advance(self.time_s):
            stats.expired.append(transition.task_id)
            stats.terms["task_loss"] -= 1.0
            self._counts["expired"] += 1
            self.recorder.record("task.expired", self.time_s, transition.to_dict(), cause="deadline")
        self._refresh_busy()
        stats.terms["resource_use"] -= self._resource_load()

    def _service_assignments(self, stats: StepStats) -> None:
        """Look after every claimed task whose access window opens in this step."""
        for task in self.board.tasks():
            if task.state not in (tasks.TaskState.ASSIGNED, tasks.TaskState.IN_PROGRESS):
                continue
            satellite_id = task.assigned_to
            if satellite_id is None or not self._available(satellite_id):
                continue
            window = self._window_for(satellite_id, task.target_id)
            if window is None:
                continue
            if task.state is tasks.TaskState.ASSIGNED:
                started = self.board.apply(task.id, "start", at_s=self.time_s)
                if not started.accepted:
                    continue
            self._record_observation(task.id, window, satellite_id, stats)

    def _resource_load(self) -> float:
        """Normalised power load of the fleet over the last step (0 when unknown)."""
        total = 0.0
        counted = 0
        for satellite_id in self.satellite_ids:
            entry = self.runtime.get(satellite_id)
            if entry is None:
                continue
            power = entry.state.get("power.solar-battery")
            if power:
                base = _finite(entry.params["power.solar-battery"].get("base_load_w"), 0.0)
                load = _finite(power.get("load_w"), base)
                total += 0.0 if base <= 0.0 else min(4.0, load / base)
                counted += 1
            else:
                supply = entry.state.get("power.constant")
                if supply:
                    counted += 1
        return 0.0 if counted == 0 else total / counted

    # --- access helpers ---------------------------------------------------------------
    def _window_for(self, satellite_id: str, target_id: str) -> Any | None:
        """The usable window that opens inside the current step, if any."""
        step = self.spec.step_s
        due = [
            window
            for window in self.scene.usable_windows(
                time_s=self.time_s, lookahead_s=step, satellite_id=satellite_id, target_id=target_id
            )
            if window.start_s >= self.time_s - 1e-9 and window.start_s < self.time_s + step - 1e-9
        ]
        if not due:
            return None
        return min(due, key=lambda window: (window.start_s, window.end_s))

    def _feasible_window(self, satellite_id: str, target_id: str) -> Any | None:
        """The next usable window inside the declared planning lookahead."""
        windows = self.scene.usable_windows(
            time_s=self.time_s, lookahead_s=self.lookahead_s,
            satellite_id=satellite_id, target_id=target_id,
        )
        if not windows:
            return None
        return min(windows, key=lambda window: (window.start_s, window.end_s))

    def _available(self, satellite_id: str) -> bool:
        if not self.fault_plan.satellite_available(satellite_id, self.time_s):
            return False
        sensor = self.scene.sensors[satellite_id]
        return sensors.sensor_model(sensor.model).is_available(sensor, self.time_s)

    def _quality_factor(self, satellite_id: str) -> float:
        return _clamp(self.fault_plan.sensor_quality_factor(satellite_id, self.time_s))

    def _visible_tasks(self, satellite_id: str | None = None) -> list[Any]:
        visible: list[Any] = []
        for task in self.board.tasks():
            if task.created_s > self.time_s + 1e-9:
                continue  # a future event is not knowable
            payload = task
            if satellite_id is not None and task.state in (tasks.TaskState.ASSIGNED, tasks.TaskState.IN_PROGRESS):
                if task.assigned_to != satellite_id:
                    # Only a claim that actually arrived makes a peer's
                    # assignment visible to this agent.
                    claim = self.bus.claim_for(task.id, satellite_id, self.time_s)
                    if claim is None:
                        payload = dataclasses.replace(
                            task, state=tasks.TaskState.AVAILABLE, assigned_to=None
                        )
            visible.append(payload)
        return visible

    def _task_list(self, satellite_id: str | None = None) -> list[Any]:
        return sorted(self._visible_tasks(satellite_id), key=lambda task: task.id)

    def _action_mask(self, satellite_id: str | None = None) -> list[list[Any]]:
        """The legal ``(satellite_id, task_id)`` pairs right now."""
        pairs: list[list[Any]] = []
        for identifier in self.satellite_ids:
            if satellite_id is not None and identifier != satellite_id:
                continue
            if not self._available(identifier) or identifier in self._busy:
                continue
            for task in self._task_list():
                if task.state is not tasks.TaskState.AVAILABLE:
                    continue
                if task.deadline_s <= self.time_s:
                    continue
                window = self._feasible_window(identifier, task.target_id)
                if window is None:
                    continue
                pairs.append([identifier, task.id])
        return pairs

    def _mask_reason(self, satellite_id: str, task_id: str) -> str:
        try:
            task = self.board.get(task_id)
        except tasks.TaskError as exc:
            return str(exc)
        if not self.fault_plan.satellite_available(satellite_id, self.time_s):
            return "satellite failed"
        if not self._available(satellite_id):
            return "sensor unavailable"
        if satellite_id in self._busy:
            return "satellite busy"
        if task.state is not tasks.TaskState.AVAILABLE:
            return f"task is {task.state.value!r}"
        if task.deadline_s <= self.time_s:
            return "deadline passed"
        if self._feasible_window(satellite_id, task.target_id) is None:
            return "no usable window inside the lookahead"
        return "not a legal pair"

    # --- observation ------------------------------------------------------------------
    def observation(self, *, satellite_id: str | None = None) -> dict[str, Any]:
        """The structured observation (globally, or as one agent's local view)."""
        satellites = sorted(self.satellite_ids)
        index_of = {identifier: index for index, identifier in enumerate(satellites)}
        fleet = max(1, len(satellites) - 1)
        task_list = self._task_list(satellite_id)
        task_index = {task.id: index for index, task in enumerate(task_list)}

        satellite_rows: list[dict[str, Any]] = []
        for identifier in satellites:
            satellite_rows.append(self._satellite_row(identifier, fleet))

        target_active = {
            target.id for target in self.targets if target.start_s <= self.time_s < target.end_s
        }
        task_rows: list[dict[str, Any]] = []
        for task in task_list:
            task_rows.append(self._task_row(task, target_active))

        assign_edges: list[list[Any]] = []
        for identifier in satellites:
            for task in task_list:
                if task.state is not tasks.TaskState.AVAILABLE or task.deadline_s <= self.time_s:
                    continue
                if not self._available(identifier):
                    continue
                window = self._feasible_window(identifier, task.target_id)
                if window is None:
                    continue
                assign_edges.append([
                    index_of[identifier],
                    task_index[task.id],
                    *self._edge_features(window),
                ])

        contact_edges = [
            [index_of[a], index_of[b]]
            for position, a in enumerate(satellites)
            for b in satellites[position + 1:]
            if self.bus.connected(a, b, self.time_s)
        ]

        mask = [
            [index_of[identifier], task_index[task_id]]
            for identifier, task_id in self._action_mask(satellite_id)
            if task_id in task_index
        ]

        return {
            "version": OBSERVATION_VERSION,
            "episode_id": self.spec.episode_id,
            "family": self.spec.family,
            "time_s": round(self.time_s, 6),
            "step_index": self.step_index,
            "horizon_s": self.spec.horizon_s,
            "step_s": self.spec.step_s,
            "satellite_ids": satellites,
            "task_ids": [task.id for task in task_list],
            "satellites": satellite_rows,
            "tasks": task_rows,
            "edges": {"assign": assign_edges, "contact": contact_edges},
            "mask": {"assign": mask},
            "global": self._global_features(task_list, target_active, contact_edges),
            "context": self._parameter_context(),
            "counts": {
                "satellites": len(satellites),
                "tasks": len(task_list),
                "assign_edges": len(assign_edges),
                "contact_edges": len(contact_edges),
                "actions": len(mask),
            },
            "notes": [
                "features are ordered by the module-level schemas; missing values carry available=0",
                "injected outages, EventTarget.truth and task reward payloads never appear here",
            ],
        }

    def _satellite_row(self, satellite_id: str, fleet: int) -> dict[str, Any]:
        state_position, velocity = dynamics.dynamics_model().state_at(
            self.orbit_by_id[satellite_id], self.time_s
        )
        position = state_position
        radius = sum(component * component for component in position) ** 0.5
        speed = sum(component * component for component in velocity) ** 0.5
        entry = self.runtime.get(satellite_id)
        visible_tasks = self._task_list(satellite_id)
        window_times = [
            window.start_s - self.time_s
            for window in self.scene.usable_windows(time_s=self.time_s, satellite_id=satellite_id)
            if window.start_s >= self.time_s - 1e-9
        ]
        sensor = self.scene.sensors[satellite_id]
        sensor_state = (entry.state.get("sensor.conical") if entry else None) or {}
        power_state = (entry.state.get("power.solar-battery") if entry else None) or {}
        storage_state = (entry.state.get("storage.buffer") if entry else None) or {}
        compute_state = (entry.state.get("compute.capacity") if entry else None) or {}
        degree = sum(
            1 for peer in self.satellite_ids
            if peer != satellite_id and self.bus.connected(satellite_id, peer, self.time_s)
        )
        values = {
            "altitude_km_norm": _clamp((radius - EARTH_RADIUS_KM) / 2_000.0),
            "speed_km_s_norm": _clamp(speed / 8.0),
            "available": 1.0 if self._available(satellite_id) else 0.0,
            "busy": 1.0 if satellite_id in self._busy else 0.0,
            "energy_fraction": _clamp(_finite(power_state.get("state_of_charge_fraction"), 0.0)),
            "storage_fraction": _clamp(_finite(storage_state.get("fill_fraction"), 0.0)),
            "compute_duty": _clamp(_finite(compute_state.get("utilization_fraction"), 0.0)),
            "footprint_km_norm": _clamp(_finite(sensor_state.get("footprint_radius_km"), 0.0) / 2_000.0),
            "visible_target_norm": _clamp(len({
                target.id for target in self.targets if target.start_s <= self.time_s < target.end_s
            }) / 10.0),
            "visible_task_norm": _clamp(len(visible_tasks) / 10.0),
            "time_to_window_norm": _clamp(min(window_times) / max(1.0, self.spec.horizon_s)) if window_times else 1.0,
            "contact_degree_norm": _clamp(degree / fleet),
            "claims_seen_norm": _clamp(len(self.bus.visible_claims(satellite_id, self.time_s)) / 10.0),
            "max_urgency_norm": _clamp(
                (max((task.urgency for task in visible_tasks if task.open), default=0)) / 5.0
            ),
        }
        available = {
            "energy_fraction": 1.0 if power_state else 0.0,
            "storage_fraction": 1.0 if storage_state else 0.0,
            "compute_duty": 1.0 if compute_state else 0.0,
            "footprint_km_norm": 1.0 if sensor_state else 0.0,
        }
        row = {
            "id": satellite_id,
            "sensor": sensor.model,
            "features": [round(values[name], 6) for name in SATELLITE_FEATURES],
            "available": [round(available.get(name, 1.0), 6) for name in SATELLITE_FEATURES],
            "is_available": bool(values["available"]),
            "busy": bool(values["busy"]),
        }
        return row

    def _task_row(self, task: Any, target_active: set[str]) -> dict[str, Any]:
        target = self.target_by_id.get(task.target_id)
        horizon = max(1.0, self.spec.horizon_s)
        values = {
            "active": 1.0 if task.target_id in target_active else 0.0,
            "urgency_norm": _clamp(task.urgency / 5.0),
            "deadline_norm": _clamp((task.deadline_s - self.time_s) / horizon),
            "age_norm": _clamp(max(0.0, self.time_s - task.created_s) / horizon),
            "min_confidence": _clamp(_finite((target.requirements or {}).get("min_confidence"), 0.0)),
            "attempts_norm": _clamp(task.attempts / 3.0),
            "is_assigned": 1.0 if task.state is tasks.TaskState.ASSIGNED else 0.0,
            "is_in_progress": 1.0 if task.state is tasks.TaskState.IN_PROGRESS else 0.0,
            "is_open": 1.0 if task.open else 0.0,
            "target_radius_km_norm": _clamp(_finite(getattr(target, "radius_km", 0.0)) / 2_000.0),
            "is_change_detection": 1.0 if task.type == "task.change-detection" else 0.0,
        }
        return {
            "id": task.id,
            "type": task.type,
            "target_id": task.target_id,
            "state": task.state.value,
            "assigned_to": task.assigned_to,
            "open": bool(task.open),
            "deadline_s": task.deadline_s,
            "kind": getattr(target, "kind", None),
            "features": [round(values[name], 6) for name in TASK_FEATURES],
        }

    def _edge_features(self, window: Any) -> list[float]:
        horizon = max(1.0, self.spec.horizon_s)
        factor = self._quality_factor(window.satellite_id)
        return [
            round(_clamp(_finite(window.peak_quality) * factor), 6),
            round(_clamp(max(0.0, window.start_s - self.time_s) / horizon), 6),
            round(_clamp((window.end_s - window.start_s) / horizon), 6),
            round(_clamp(_finite(window.best_resolution_m) / 1_000.0), 6),
        ]

    def _global_features(
        self, task_list: Sequence[Any], target_active: set[str], contact_edges: Sequence[Sequence[int]]
    ) -> list[float]:
        total = max(1, len(self.scene.task_templates))
        open_tasks = [task for task in task_list if task.open]
        fleet = max(1, len(self.satellite_ids))
        possible_contacts = max(1, fleet * (fleet - 1) // 2)
        horizon = max(1.0, self.spec.horizon_s)
        deadline_pressure = sum(
            _clamp(1.0 - (task.deadline_s - self.time_s) / horizon) for task in open_tasks
        ) / max(1, len(open_tasks))
        active_faults = self.fault_plan.active_at(self.time_s)
        values = {
            "time_fraction": _clamp(self.time_s / horizon),
            "open_task_fraction": _clamp(len(open_tasks) / total),
            "fleet_available_fraction": _clamp(
                sum(1 for identifier in self.satellite_ids if self._available(identifier)) / fleet
            ),
            "contact_density": _clamp(len(contact_edges) / possible_contacts),
            "fault_intensity": _clamp(
                sum(1.0 if fault.kind == "satellite-failure" else 0.5 for fault in active_faults) / 4.0
            ),
            "deadline_pressure": _clamp(deadline_pressure),
        }
        _ = target_active
        return [round(values[name], 6) for name in GLOBAL_FEATURES]

    def _parameter_context(self) -> dict[str, Any]:
        """Parameter tokens are attached by the scenario pipeline (Day 40)."""
        return dict(self.spec.parameters or {})

    # --- metrics, trace, replay -------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        board_metrics = self.board.metrics()
        total = max(1, len(self.board.tasks()))
        observed = self._counts["observed"]
        return {
            "episode_id": self.spec.episode_id,
            "family": self.spec.family,
            "steps": self.step_index,
            "time_s": round(self.time_s, 6),
            "fleet_size": len(self.satellite_ids),
            "tasks_total": len(self.board.tasks()),
            "observed": observed,
            "completed": board_metrics.get("completed", 0),
            "expired": board_metrics.get("expired", 0),
            "coverage": round(observed / total, 6),
            "mean_latency_s": round(self._counts["latency_s"] / observed, 6) if observed else None,
            "conflicts": self._counts["conflicts"],
            "invalid_actions": self._counts["invalid_actions"],
            "faults_active": len(self.fault_plan.active_at(self.time_s)),
            "termination_reason": self._termination_reason,
            "board": board_metrics,
        }

    def trace_hash(self) -> str:
        return self.recorder.hash()

    def manifest(self) -> dict[str, Any]:
        return {
            "version": ENV_VERSION,
            "episode_id": self.spec.episode_id,
            "spec_hash": self.spec.spec_hash(),
            "spec": self.spec.payload(),
            "expected_trace_hash": self.trace_hash(),
            "feature_schema": feature_schema_payload(),
            "metrics": self.metrics(),
        }

    # --- rollouts ---------------------------------------------------------------------
    def rollout(
        self,
        policy: "EnvPolicy",
        *,
        reset: bool = True,
        capture_observations: bool = False,
    ) -> EpisodeOutcome:
        """Run one full episode with ``policy`` and return the recorded outcome.

        ``capture_observations`` keeps the observation the policy actually acted
        on for every step (what a supervised dataset stores); it is off by
        default so batch evaluations stay cheap.
        """
        if reset:
            observation, _ = self.reset()
            policy.reset()
        else:
            observation = self.observation()
        self.captured_observations = []
        steps: list[StepStats] = []
        records: list[dict[str, Any]] = []
        terminated = truncated = False
        while not (terminated or truncated):
            if capture_observations:
                self.captured_observations.append(
                    {key: value for key, value in observation.items() if key not in ("context", "notes")}
                )
            actions = policy.act(observation)
            observation, reward, terminated, truncated, info = self.step(actions)
            stats = StepStats.from_dict(info["step"])
            steps.append(stats)
            records.append({
                "time_s": stats.time_s,
                "actions": [action.to_dict() for action in actions],
                "targets": action_targets(observation, actions),
                "terms": dict(stats.terms),
                "reward": round(reward, 6),
                "metrics": self.metrics(),
            })
        metrics = self.metrics()
        metrics["mean_reward"] = round(
            sum(step.reward for step in steps) / max(1, len(steps)), 6
        )
        metrics["reward_terms"] = {
            name: round(sum(step.terms.get(name, 0.0) for step in steps), 6) for name in REWARD_TERMS
        }
        return EpisodeOutcome(
            episode_id=self.spec.episode_id,
            spec_hash=self.spec.spec_hash(),
            trace_hash=self.trace_hash(),
            metrics=metrics,
            steps=steps,
            records=records,
            manifest=self.manifest(),
        )

    def replay(self, actions_per_step: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        """Re-run a recorded action sequence through the same core and compare."""
        self.reset()
        for actions in actions_per_step:
            if self.done:
                break
            self.step([Action.from_dict(item) for item in actions])
        return {
            "replayed_trace_hash": self.trace_hash(),
            "metrics": self.metrics(),
        }


def replay_episode(manifest: Mapping[str, Any], actions_per_step: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Rebuild an episode from its manifest and replay a recorded action sequence."""
    spec = EpisodeSpec.from_payload(manifest["spec"])
    environment = SwarmEnvironment(spec)
    environment.reset()
    for actions in actions_per_step:
        if environment.done:
            break
        environment.step([Action.from_dict(item) for item in actions])
    replayed = environment.trace_hash()
    expected = manifest.get("expected_trace_hash")
    return {
        "episode_id": spec.episode_id,
        "expected_trace_hash": expected,
        "replayed_trace_hash": replayed,
        "matches": replayed == expected,
        "metrics": environment.metrics(),
    }


class EnvPolicy:
    """Interface every policy follows inside the environment.

    ``label`` names the policy in artefacts; ``act`` receives one observation and
    returns proposed actions (the environment validates them against the mask).
    Recurrent policies keep their own state and drop it in ``reset``.
    """

    label = "policy"
    version = "1"

    def reset(self) -> None:
        """Drop episode-local state (called once per episode)."""

    def act(self, observation: Mapping[str, Any]) -> list[Action]:
        raise NotImplementedError


def multi_agent_split(observation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Split a joint observation into per-satellite masked local views.

    Each agent keeps the full entity collections (so a decentralized model can
    still see the graph) but only its own rows are marked actionable: the mask
    holds the pairs whose first index is that agent.
    """
    views: dict[str, dict[str, Any]] = {}
    for index, satellite_id in enumerate(observation["satellite_ids"]):
        mask = [pair for pair in observation["mask"]["assign"] if pair[0] == index]
        views[satellite_id] = {
            **{key: value for key, value in observation.items() if key not in ("mask", "satellites", "tasks")},
            "satellites": observation["satellites"],
            "tasks": observation["tasks"],
            "mask": {"assign": mask},
            "agent_index": index,
            "agent_id": satellite_id,
        }
    return views


def action_targets(
    observation: Mapping[str, Any], actions: Iterable[Action]
) -> list[list[int]]:
    """The chosen actions as ``[satellite_index, task_index]`` pairs.

    Targets are expressed in *observation index space*, which is the only stable
    identity a learned policy has: index ``i`` always means
    ``observation['satellite_ids'][i]``.
    """
    satellite_index = {
        satellite_id: index for index, satellite_id in enumerate(observation["satellite_ids"])
    }
    task_index = {task_id: index for index, task_id in enumerate(observation["task_ids"])}
    pairs: list[list[int]] = []
    for action in actions:
        if action.kind != ACTION_ASSIGN or action.satellite_id is None or action.task_id is None:
            continue
        if action.satellite_id not in satellite_index or action.task_id not in task_index:
            continue
        pairs.append([satellite_index[action.satellite_id], task_index[action.task_id]])
    return pairs


def flatten_actions(actions: Iterable[Action]) -> list[dict[str, Any]]:
    return [action.to_dict() for action in actions]


def observation_hash(observation: Mapping[str, Any]) -> str:
    """Stable hash of an observation (used by leakage and determinism tests)."""
    return hashlib.sha256(
        trace.canonical_json(dict(observation)).encode("utf-8")
    ).hexdigest()[:16]


def contact_predicate(contact: Mapping[str, Any] | None, orbit_by_id: Mapping[str, Any]) -> Any:
    """A declared link-availability model: slant range plus a duty cycle.

    This is a *link availability* model, not a radio or protocol simulation: a
    pair is connected while the two satellites are within ``range_km`` and the
    pair's deterministic duty-cycle phase is inside ``duty_cycle``. The phase
    offset comes from a hash of the two ids, so links are not synchronised and
    the same configuration always produces the same contact pattern.
    """
    if not contact:
        return None
    range_km = float(contact.get("range_km", 0.0))
    if range_km <= 0.0:
        return None
    duty_cycle = min(1.0, max(0.0, float(contact.get("duty_cycle", 1.0))))
    period_s = max(1.0, float(contact.get("period_s", 900.0)))
    model = dynamics.dynamics_model()
    cache: dict[float, dict[str, Any]] = {}

    def positions_at(time_s: float) -> dict[str, Any]:
        key = round(float(time_s), 6)
        entry = cache.get(key)
        if entry is None:
            cache.clear()
            entry = {satellite_id: model.state_at(orbit, key)[0] for satellite_id, orbit in orbit_by_id.items()}
            cache[key] = entry
        return entry

    def predicate(a: str, b: str, time_s: float) -> bool:
        positions = positions_at(time_s)
        first, second = positions.get(a), positions.get(b)
        if first is None or second is None:
            return False
        distance = sum((left - right) ** 2 for left, right in zip(first, second)) ** 0.5
        if distance > range_km:
            return False
        if duty_cycle >= 1.0:
            return True
        offset = int(hashlib.sha256(f"{a}|{b}".encode("utf-8")).hexdigest()[:8], 16) % 997 / 997.0
        phase = (float(time_s) / period_s + offset) % 1.0
        return phase < duty_cycle

    return predicate


def _scene_for(spec: EpisodeSpec, config: Any, targets: Sequence[EventTarget]) -> Any:
    from constellab_core.policy import Scene

    return Scene.build(
        config,
        list(targets),
        horizon_s=spec.horizon_s,
        step_s=spec.step_s,
        seed=spec.seed,
        window_step_s=spec.window_step_s,
    )
