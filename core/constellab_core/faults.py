"""Day 31: configurable fault injection and reproducible swarm failures.

A fault plan is **data**: typed, time-bounded events with an explicit scope,
target selection rule, duration and provenance. The episode harness applies the
plan in exactly four ways:

* ``satellite-failure`` — a satellite is unavailable (it cannot task or observe)
  while the event is active;
* ``sensor-degrade`` — an available satellite's access quality is scaled by
  ``quality`` (0.0 means no usable access);
* ``link-cut`` — every pair between the two target groups is treated as
  disconnected while the event is active (a partition);
* ``message-delay`` — claim traffic between the two target groups is delayed by
  ``delay_s``.

The fault generator is separate from normal simulation logic: faults never
change orbit dynamics, task rules or policy code. They only alter what the
harness makes visible/possible at a given simulated time, and every event is
recorded in the trace with a cause. Plans are deterministic for a seed, and two
strategies are shipped:

* ``manual`` — an explicit plan (used by the G2 demo and tests);
* ``random`` — seeded uniform selection of scope, time and duration;
* ``worst-moment`` — cuts the first pair of satellites that can both see the
  same target during an overlapping window, i.e. the moment a claim conflict is
  possible. This is a transparent heuristic, not an optimizer.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

FAULT_VERSION = "fault-injection-1"

FAULT_KINDS: tuple[str, ...] = (
    "satellite-failure",
    "sensor-degrade",
    "link-cut",
    "message-delay",
)

FAULT_STRATEGIES: tuple[str, ...] = ("manual", "random", "worst-moment")


class FaultError(ValueError):
    """An invalid fault definition or plan; no episode state is touched."""


def _finite(name: str, value: Any) -> float:
    import math

    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise FaultError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Fault:
    """One scoped, time-bounded fault event."""

    id: str
    kind: str
    start_s: float
    end_s: float
    satellites: tuple[str, ...] = ()
    peers: tuple[str, ...] = ()
    quality: float = 1.0
    delay_s: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise FaultError("a fault needs a non-empty id")
        if self.kind not in FAULT_KINDS:
            raise FaultError(f"fault kind {self.kind!r} must be one of {FAULT_KINDS}")
        object.__setattr__(self, "start_s", _finite("start_s", self.start_s))
        object.__setattr__(self, "end_s", _finite("end_s", self.end_s))
        if self.start_s < 0.0:
            raise FaultError("fault start_s must be non-negative")
        if self.end_s <= self.start_s:
            raise FaultError("a fault must have a positive duration")
        if not self.satellites:
            raise FaultError(f"fault {self.id!r} needs at least one target satellite")
        if len(set(self.satellites)) != len(self.satellites):
            raise FaultError(f"fault {self.id!r} lists a satellite twice")
        if self.kind in ("link-cut", "message-delay") and not self.peers:
            raise FaultError(f"{self.kind} fault {self.id!r} needs peer targets")
        if set(self.satellites) & set(self.peers):
            raise FaultError(f"fault {self.id!r} targets a satellite on both sides")
        object.__setattr__(self, "quality", _finite("quality", self.quality))
        if not 0.0 <= self.quality <= 1.0:
            raise FaultError("sensor quality factor must be within [0, 1]")
        object.__setattr__(self, "delay_s", _finite("delay_s", self.delay_s))
        if self.delay_s < 0.0:
            raise FaultError("message delay must be non-negative")

    def active(self, time_s: float) -> bool:
        return self.start_s <= time_s < self.end_s

    def covers_satellite(self, satellite_id: str, time_s: float) -> bool:
        return self.active(time_s) and satellite_id in self.satellites

    def covers_link(self, a: str, b: str, time_s: float) -> bool:
        if not self.active(time_s):
            return False
        return (a in self.satellites and b in self.peers) or (a in self.peers and b in self.satellites)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "satellites": list(self.satellites),
            "peers": list(self.peers),
            "quality": self.quality,
            "delay_s": self.delay_s,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FaultPlan:
    """An ordered set of faults plus how the plan was generated."""

    faults: tuple[Fault, ...]
    strategy: str = "manual"
    seed: int = 0
    generator: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in FAULT_STRATEGIES:
            raise FaultError(f"unknown strategy {self.strategy!r}")
        ordered = sorted(self.faults, key=lambda item: (item.start_s, item.id))
        object.__setattr__(self, "faults", tuple(ordered))
        ids = [fault.id for fault in ordered]
        if len(set(ids)) != len(ids):
            raise FaultError("fault plan ids must be unique")

    # --- queries used by the harness ------------------------------------------------
    def satellite_available(self, satellite_id: str, time_s: float) -> bool:
        return not any(
            fault.kind == "satellite-failure" and fault.covers_satellite(satellite_id, time_s)
            for fault in self.faults
        )

    def sensor_quality_factor(self, satellite_id: str, time_s: float) -> float:
        factor = 1.0
        for fault in self.faults:
            if fault.kind == "sensor-degrade" and fault.covers_satellite(satellite_id, time_s):
                factor = min(factor, fault.quality)
        return factor

    def link_blocked(self, a: str, b: str, time_s: float) -> bool:
        return any(fault.kind == "link-cut" and fault.covers_link(a, b, time_s) for fault in self.faults)

    def message_delay(self, source: str, destination: str, time_s: float) -> float:
        return max(
            (
                fault.delay_s
                for fault in self.faults
                if fault.kind == "message-delay" and fault.covers_link(source, destination, time_s)
            ),
            default=0.0,
        )

    def active_at(self, time_s: float) -> list[Fault]:
        return [fault for fault in self.faults if fault.active(time_s)]

    def satellite_faults(self, satellite_id: str) -> list[Fault]:
        return [fault for fault in self.faults if satellite_id in fault.satellites]

    def impact_times(self) -> tuple[float, ...]:
        """Every start/end boundary, for traces and UI markers."""
        return tuple(sorted({bound for fault in self.faults for bound in (fault.start_s, fault.end_s)}))

    # --- serialization --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FAULT_VERSION,
            "strategy": self.strategy,
            "seed": self.seed,
            "generator": dict(self.generator),
            "faults": [fault.to_dict() for fault in self.faults],
        }

    def canonical(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    @classmethod
    def manual(cls, faults: Sequence[Fault], *, label: str = "manual") -> "FaultPlan":
        return cls(tuple(faults), strategy="manual", seed=0, generator={"label": label})

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FaultPlan":
        faults = tuple(
            Fault(
                id=str(item["id"]),
                kind=str(item["kind"]),
                start_s=float(item["start_s"]),
                end_s=float(item["end_s"]),
                satellites=tuple(str(value) for value in item.get("satellites", ())),
                peers=tuple(str(value) for value in item.get("peers", ())),
                quality=float(item.get("quality", 1.0)),
                delay_s=float(item.get("delay_s", 0.0)),
                reason=str(item.get("reason", "")),
            )
            for item in payload.get("faults", ())
        )
        return cls(
            faults,
            strategy=str(payload.get("strategy", "manual")),
            seed=int(payload.get("seed", 0)),
            generator=dict(payload.get("generator", {})),
        )


def empty_plan() -> FaultPlan:
    return FaultPlan((), strategy="manual", generator={"label": "no-fault control"})


# --- generators ---------------------------------------------------------------------

def _rng(seed: int, purpose: str) -> random.Random:
    # Hash the purpose into the seed: adding a draw in one plan type never
    # shifts another plan generated from the same master seed.
    digest = hashlib.sha256(f"{seed}|{purpose}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def random_plan(
    satellites: Sequence[str],
    *,
    horizon_s: float,
    seed: int,
    count: int = 2,
    kinds: Sequence[str] = ("satellite-failure", "link-cut"),
    min_duration_s: float = 300.0,
    max_duration_s: float = 1_200.0,
) -> FaultPlan:
    """Seeded faults with explicit scope, window, targets and duration."""
    ids = sorted(set(satellites))
    if not ids:
        raise FaultError("a random fault plan needs at least one satellite")
    if horizon_s <= 0:
        raise FaultError("horizon_s must be positive")
    if count < 0:
        raise FaultError("count must be non-negative")
    unknown = [kind for kind in kinds if kind not in FAULT_KINDS]
    if unknown:
        raise FaultError(f"unknown fault kinds: {unknown}")
    rng = _rng(seed, "faults.random")
    faults: list[Fault] = []
    for index in range(count):
        kind = kinds[index % len(kinds)]
        duration = min(
            horizon_s,
            rng.uniform(min_duration_s, max(max_duration_s, min_duration_s)),
        )
        start = rng.uniform(0.0, max(0.0, horizon_s - duration))
        target = rng.choice(ids)
        if kind in ("link-cut", "message-delay"):
            peers = [item for item in ids if item != target]
            if not peers:
                continue
            peer = rng.choice(peers)
            faults.append(Fault(
                id=f"fault-{index:02d}",
                kind=kind,
                start_s=start,
                end_s=min(horizon_s, start + duration),
                satellites=(target,),
                peers=(peer,),
                delay_s=60.0 if kind == "message-delay" else 0.0,
                reason=f"seeded {kind} between {target} and {peer}",
            ))
        else:
            faults.append(Fault(
                id=f"fault-{index:02d}",
                kind=kind,
                start_s=start,
                end_s=min(horizon_s, start + duration),
                satellites=(target,),
                quality=0.0 if kind == "sensor-degrade" else 1.0,
                reason=f"seeded {kind} on {target}",
            ))
    return FaultPlan(
        tuple(faults),
        strategy="random",
        seed=seed,
        generator={
            "satellites": ids,
            "horizon_s": horizon_s,
            "count": count,
            "kinds": list(kinds),
            "min_duration_s": min_duration_s,
            "max_duration_s": max_duration_s,
        },
    )


def worst_moment_plan(
    scene: Any,
    *,
    horizon_s: float | None = None,
    duration_s: float = 600.0,
    seed: int = 0,
) -> FaultPlan:
    """Partition the first pair of satellites that can see the same target.

    Overlapping windows are the moments where two independent agents can both
    decide the same task is theirs. The plan cuts the pair at that overlap, so
    the failure is attached to a concrete, explainable moment rather than a
    random time. Falls back to a seeded plan when no overlap exists.
    """
    horizon = float(horizon_s if horizon_s is not None else scene.horizon_s)
    by_target: dict[str, list[Any]] = {}
    for window in scene.usable_windows(time_s=0.0, lookahead_s=horizon):
        by_target.setdefault(window.target_id, []).append(window)

    best: tuple[float, str, Any, Any] | None = None
    for target_id in sorted(by_target):
        windows = sorted(by_target[target_id], key=lambda window: (window.start_s, window.satellite_id))
        for first in windows:
            for second in windows:
                if first.satellite_id >= second.satellite_id:
                    continue
                start = max(first.start_s, second.start_s)
                end = min(first.end_s, second.end_s)
                if end <= start:
                    continue
                candidate = (start, target_id, first.satellite_id, second.satellite_id)
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        return random_plan(
            [satellite.id for satellite in scene.config.satellites],
            horizon_s=horizon, seed=seed, count=1, kinds=("link-cut",),
        )

    start, target_id, a, b = best
    faults = (
        Fault(
            id="fault-worst-00",
            kind="link-cut",
            start_s=max(0.0, start),
            end_s=min(horizon, start + duration_s),
            satellites=(a,), peers=(b,),
            reason=f"overlapping access to {target_id}; claims can not cross",
        ),
    )
    return FaultPlan(
        faults,
        strategy="worst-moment",
        seed=seed,
        generator={
            "target_id": target_id,
            "satellites": [a, b],
            "horizon_s": horizon,
            "duration_s": duration_s,
        },
    )
