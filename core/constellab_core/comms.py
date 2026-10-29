"""Day 31/33: a partition-aware claim bus for decentralized baselines.

The simulator is centralized, but a deployed swarm is not: each satellite sees
its own task state and only learns what its peers claimed when a link carries
that claim. This module models that **claim visibility** explicitly:

* every accepted assignment publishes a claim onto the bus;
* a viewer sees claims from peers reachable through the current link graph,
  after the configured delivery delay (plus any active ``message-delay`` fault);
* claims expire after ``claim_ttl_s`` so a stale broadcast can not suppress work
  forever;
* a failed satellite's claims stop being visible.

The link graph is all-to-all by default so a healthy fleet behaves like a shared
plan. A ``link-cut`` fault removes every edge between its two target groups,
which is what lets two disconnected groups both believe a task is unclaimed.
The renderer may draw the same graph, but delivery/visibility is decided here,
not in the UI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

COMMS_VERSION = "claim-bus-1"

LinkPredicate = Callable[[str, str, float], bool]


@dataclass(frozen=True)
class Claim:
    task_id: str
    satellite_id: str
    time_s: float
    sequence: int
    target_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "satellite_id": self.satellite_id,
            "target_id": self.target_id,
            "time_s": round(self.time_s, 6),
            "sequence": self.sequence,
        }


class ClaimBus:
    """Stores published claims and resolves which ones a satellite can see."""

    def __init__(
        self,
        satellites: Iterable[str],
        *,
        faults: Any = None,
        link: LinkPredicate | None = None,
        claim_ttl_s: float = 1_800.0,
        base_delay_s: float = 0.0,
    ) -> None:
        self.satellites = tuple(sorted(set(satellites)))
        if not self.satellites:
            raise ValueError("a claim bus needs at least one satellite")
        if claim_ttl_s <= 0.0:
            raise ValueError("claim_ttl_s must be positive")
        if base_delay_s < 0.0:
            raise ValueError("base_delay_s must be non-negative")
        self.faults = faults
        self._link = link
        self.claim_ttl_s = claim_ttl_s
        self.base_delay_s = base_delay_s
        self._claims: list[Claim] = []
        self._sequence = 0

    # --- geometry of visibility -----------------------------------------------------
    def connected(self, a: str, b: str, time_s: float) -> bool:
        if a == b:
            return True
        if a not in self.satellites or b not in self.satellites:
            return False
        if self._link is not None and not self._link(a, b, time_s):
            return False
        if self.faults is not None and self.faults.link_blocked(a, b, time_s):
            return False
        return True

    def component(self, viewer: str, time_s: float) -> list[str]:
        """Every satellite reachable from ``viewer`` (including itself)."""
        if viewer not in self.satellites:
            return []
        reachable = {viewer}
        frontier = [viewer]
        while frontier:
            node = frontier.pop()
            for peer in self.satellites:
                if peer in reachable:
                    continue
                if self.connected(node, peer, time_s):
                    reachable.add(peer)
                    frontier.append(peer)
        return sorted(reachable)

    def topology(self, time_s: float) -> dict[str, Any]:
        groups: list[list[str]] = []
        remaining = set(self.satellites)
        while remaining:
            seed = min(remaining)
            group = set(self.component(seed, time_s))
            groups.append(sorted(group))
            remaining -= group
        edges = [
            [a, b]
            for index, a in enumerate(self.satellites)
            for b in self.satellites[index + 1:]
            if self.connected(a, b, time_s)
        ]
        return {"time_s": round(time_s, 6), "groups": groups, "edges": edges}

    # --- claims ---------------------------------------------------------------------
    def record_claim(
        self,
        task_id: str,
        satellite_id: str,
        *,
        time_s: float,
        target_id: str = "",
    ) -> Claim:
        if satellite_id not in self.satellites:
            raise ValueError(f"unknown satellite {satellite_id!r}")
        self._sequence += 1
        claim = Claim(
            task_id=task_id,
            satellite_id=satellite_id,
            time_s=round(float(time_s), 6),
            sequence=self._sequence,
            target_id=target_id,
        )
        self._claims.append(claim)
        return claim

    def delay_to(self, source: str, viewer: str, time_s: float) -> float:
        delay = self.base_delay_s
        if self.faults is not None:
            delay += self.faults.message_delay(source, viewer, time_s)
        return delay

    def visible_claims(
        self,
        viewer: str,
        time_s: float,
        *,
        include_own: bool = True,
    ) -> list[dict[str, Any]]:
        """Claims whose source is reachable and whose delivery time has passed."""
        if viewer not in self.satellites:
            return []
        visible: list[dict[str, Any]] = []
        component = set(self.component(viewer, time_s))
        for claim in self._claims:
            if claim.satellite_id not in component:
                continue
            if not include_own and claim.satellite_id == viewer:
                continue
            if self.faults is not None and not self.faults.satellite_available(claim.satellite_id, time_s):
                continue
            age = time_s - claim.time_s
            if age > self.claim_ttl_s:
                continue
            if age + 1e-9 < self.delay_to(claim.satellite_id, viewer, time_s):
                continue
            visible.append(claim.to_dict())
        visible.sort(key=lambda item: (item["time_s"], item["sequence"]))
        return visible

    def claim_for(self, task_id: str, viewer: str, time_s: float) -> dict[str, Any] | None:
        for claim in self.visible_claims(viewer, time_s):
            if claim["task_id"] == task_id and claim["satellite_id"] != viewer:
                return claim
        return None

    # --- reporting ------------------------------------------------------------------
    def payload(self, time_s: float | None = None) -> dict[str, Any]:
        return {
            "version": COMMS_VERSION,
            "satellites": list(self.satellites),
            "claim_ttl_s": self.claim_ttl_s,
            "base_delay_s": self.base_delay_s,
            "claims": [claim.to_dict() for claim in self._claims],
            "topology": self.topology(time_s or 0.0),
        }

    def hash(self) -> str:
        import json

        canonical = json.dumps(
            {
                "satellites": self.satellites,
                "claims": [claim.to_dict() for claim in self._claims],
            },
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
