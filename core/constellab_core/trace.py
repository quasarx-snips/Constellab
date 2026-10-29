"""Day 32: deterministic event traces, seed streams and replay manifests.

Every state-changing event is recorded with a simulated timestamp, a stable
kind, its entity ids and a cause. The recorder never reads the wall clock, and
canonical serialization sorts by ``(time_s, sequence)`` and then by JSON keys,
so two runs that make the same decisions hash identically regardless of object
identity or dict insertion order.

Randomness is derived per subsystem and per entity from a master seed
(:class:`SeedBank`). Adding one draw inside the fault generator therefore does
not shift the claim-bus delay, the transport loss, or any other stream. Seed
derivation is a hash, not a shared PRNG: ``seed = sha256(master|stream)[:8]``.

A :class:`ReplayManifest` records everything needed to reproduce an episode
through the same simulation core: scenario payload, policy key, seed, fault plan
and contract/schema versions. :func:`core.g2.replay` re-runs it and compares the
canonical trace hash; it never replays by printing stored final state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

TRACE_VERSION = "event-trace-1"
SEED_VERSION = "seed-streams-1"


def canonical_json(payload: Any) -> str:
    """Stable JSON: sorted keys, no whitespace; NaN/Inf are rejected."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class SeedBank:
    """Independent, derived random seeds for every subsystem/entity."""

    def __init__(self, master_seed: int) -> None:
        if not isinstance(master_seed, int) or isinstance(master_seed, bool):
            raise ValueError("master seed must be an integer")
        if master_seed < 0:
            raise ValueError("master seed must be non-negative")
        self.master_seed = master_seed
        self._derived: dict[str, int] = {}

    def derive(self, stream: str) -> int:
        if not stream:
            raise ValueError("a seed stream needs a name")
        if stream not in self._derived:
            digest = hashlib.sha256(f"{SEED_VERSION}|{self.master_seed}|{stream}".encode("utf-8"))
            self._derived[stream] = int(digest.hexdigest()[:16], 16)
        return self._derived[stream]

    def entity(self, stream: str, entity_id: str) -> int:
        return self.derive(f"{stream}|{entity_id}")

    def payload(self) -> dict[str, Any]:
        return {
            "version": SEED_VERSION,
            "master_seed": self.master_seed,
            "streams": {name: value for name, value in sorted(self._derived.items())},
        }


@dataclass(frozen=True)
class TraceEvent:
    """One deterministic event, in a stable global order."""

    sequence: int
    time_s: float
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    cause: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "time_s": round(self.time_s, 6),
            "kind": self.kind,
            "cause": self.cause,
            "payload": dict(self.payload),
        }


class TraceRecorder:
    """Append-only event recorder with canonical ordering and hashing."""

    def __init__(self, *, label: str = "") -> None:
        self.label = label
        self._events: list[TraceEvent] = []
        self._sequence = 0

    def record(
        self,
        kind: str,
        time_s: float,
        payload: Mapping[str, Any] | None = None,
        *,
        cause: str = "",
    ) -> TraceEvent:
        if not kind:
            raise ValueError("a trace event needs a kind")
        if time_s < 0:
            raise ValueError("a trace event needs a non-negative time")
        self._sequence += 1
        event = TraceEvent(
            sequence=self._sequence,
            time_s=round(float(time_s), 6),
            kind=kind,
            payload=dict(payload or {}),
            cause=cause,
        )
        self._events.append(event)
        return event

    def events(self) -> list[TraceEvent]:
        return sorted(self._events, key=lambda event: (event.time_s, event.sequence))

    def kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self._events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        return dict(sorted(counts.items()))

    def canonical(self) -> str:
        # The label is display metadata: two runs of the same episode must hash
        # identically whether they are labelled "g2:fault" or "replay:...".
        return canonical_json({
            "version": TRACE_VERSION,
            "events": [event.to_dict() for event in self.events()],
        })

    def hash(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def payload(self, *, limit: int | None = None) -> dict[str, Any]:
        events = self.events()
        if limit is not None:
            events = events[:limit]
        return {
            "version": TRACE_VERSION,
            "label": self.label,
            "count": len(self._events),
            "kinds": self.kinds(),
            "events": [event.to_dict() for event in events],
        }

    def __len__(self) -> int:
        return len(self._events)


def diff_traces(left: Iterable[Mapping[str, Any]], right: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """First event where two serialized traces differ, or None if identical.

    The comparison is over canonical event dictionaries, so it reports the
    attributable starting point (time/kind) instead of only a whole-run hash.
    """
    left_events = list(left)
    right_events = list(right)
    for index in range(max(len(left_events), len(right_events))):
        a = left_events[index] if index < len(left_events) else None
        b = right_events[index] if index < len(right_events) else None
        if canonical_json(a) != canonical_json(b):
            return {
                "index": index,
                "left": a,
                "right": b,
                "time_s": (a or b or {}).get("time_s"),
                "kind": (a or b or {}).get("kind"),
            }
    return None


@dataclass(frozen=True)
class ReplayManifest:
    """Everything needed to reproduce an episode through the simulation core."""

    scenario: Mapping[str, Any]
    policy_key: str
    seed: int
    faults: Mapping[str, Any]
    expected_trace_hash: str
    horizon_s: float
    step_s: float
    versions: Mapping[str, str] = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": TRACE_VERSION,
            "label": self.label,
            "scenario": dict(self.scenario),
            "policy_key": self.policy_key,
            "seed": self.seed,
            "faults": dict(self.faults),
            "expected_trace_hash": self.expected_trace_hash,
            "horizon_s": self.horizon_s,
            "step_s": self.step_s,
            "versions": dict(self.versions),
        }

    def canonical(self) -> str:
        return canonical_json(self.to_dict())

    def hash(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayManifest":
        return cls(
            scenario=dict(payload["scenario"]),
            policy_key=str(payload["policy_key"]),
            seed=int(payload["seed"]),
            faults=dict(payload["faults"]),
            expected_trace_hash=str(payload["expected_trace_hash"]),
            horizon_s=float(payload["horizon_s"]),
            step_s=float(payload["step_s"]),
            versions=dict(payload.get("versions", {})),
            label=str(payload.get("label", "")),
        )
