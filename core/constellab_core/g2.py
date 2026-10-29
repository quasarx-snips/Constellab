"""Day 34: the G2 gate — one reproducible swarm failure with a visible cause.

The demo is deliberately small and transparent:

* four satellites share one near-identical orbit (two per side), so all four can
  see the same hotspot and every one of them can claim the same task;
* a :class:`~constellab_core.faults.Fault` partitions the fleet into
  ``g2-alpha``/``g2-bravo`` versus ``g2-charlie``/``g2-delta`` for the whole run;
* each satellite runs ``distributed-greedy``, which knows only the claims the
  link graph delivered;
* the simulator's task board still rejects the second claim, so the run *fails*
  as a duplicate-tasking violation instead of quietly double-assigning.

The control case removes the partition: claims cross, so only one satellite
claims the task and no duplicate-claim finding is produced. Both cases are run
twice with the same seed and compared by canonical trace hash, and the report
carries a replay manifest so :func:`replay` can reproduce the failure through
the same harness (never by re-printing stored state).

Everything the UI needs is computed here: per-satellite positions come from the
contract snapshot (the renderer never integrates orbits), link topology comes
from the claim bus, and the first invariant violation is a structured finding
from the Day 33 rule engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import comms, dynamics, faults, frames, invariants, policy, trace
from .config import load_config, load_config_file
from .contract import SimulationClock
from .orbit import snapshot_at_time
from .targets import EventTarget

G2_VERSION = "g2-failure-1"
G2_SCENARIO_ID = "g2-partition"
G2_SEED = 7
G2_HORIZON_S = 1_800.0
G2_STEP_S = 30.0
G2_TARGET_TIME_S = 600.0

G2_GROUPS: dict[str, tuple[str, ...]] = {
    "A": ("g2-alpha", "g2-bravo"),
    "B": ("g2-charlie", "g2-delta"),
}
G2_SATELLITES: tuple[str, ...] = G2_GROUPS["A"] + G2_GROUPS["B"]


class G2Error(RuntimeError):
    """The G2 demo could not be built or replayed."""


def scenario_payload() -> dict[str, Any]:
    """A validated four-satellite scenario, two satellites per partition side."""
    base = load_config_file("visual-orbit-demo").to_dict()
    template = dict(base["satellites"][0]["orbit"])
    satellites = []
    for index, satellite_id in enumerate(G2_SATELLITES):
        orbit = dict(template)
        orbit["true_anomaly_deg"] = round(
            (float(template["true_anomaly_deg"]) + index * 0.2) % 360.0, 6
        )
        satellites.append({
            "id": satellite_id,
            "name": satellite_id.upper(),
            "orbit": orbit,
        })
    return {
        **base,
        "scenario_id": G2_SCENARIO_ID,
        "name": "G2 partition failure (4 satellites, 2 groups)",
        "seed": G2_SEED,
        "time_scale": 60.0,
        "satellites": satellites,
    }


def build_config() -> Any:
    return load_config(scenario_payload())


def target_for(config: Any) -> EventTarget:
    """Hotspot under the shared orbit, computed from core dynamics."""
    position = dynamics.dynamics_model().state_at(
        config.satellites[0].orbit, G2_TARGET_TIME_S
    )[0]
    point = frames.geodetic_from_eci(position, G2_TARGET_TIME_S)
    return EventTarget(
        id="hotspot.g2-eq",
        kind="hotspot",
        lat_deg=point.latitude_deg,
        lon_deg=point.longitude_deg,
        start_s=0.0,
        end_s=G2_HORIZON_S,
        radius_km=120.0,
        urgency=5,
        requirements={"min_confidence": 0.4},
        truth={"intensity_mw": 12.0, "cause": "G2 demo hotspot"},
    )


def partition_plan() -> faults.FaultPlan:
    return faults.FaultPlan.manual((
        faults.Fault(
            id="g2-partition-00",
            kind="link-cut",
            start_s=0.0,
            end_s=G2_HORIZON_S,
            satellites=G2_GROUPS["A"],
            peers=G2_GROUPS["B"],
            reason="G2 demo: the two halves of the fleet never exchange claims",
        ),
    ), label="G2 partition")


def build_scene() -> tuple[Any, policy.Scene]:
    config = build_config()
    scene = policy.Scene.build(
        config, [target_for(config)],
        horizon_s=G2_HORIZON_S, step_s=G2_STEP_S, seed=G2_SEED,
    )
    return config, scene


def policy_for_key(key: str, scene: policy.Scene) -> Any:
    """Resolve a manifest's policy key without a hard-coded branch in callers."""
    registry: dict[str, Any] = {
        "local-greedy": lambda: policy.LocalGreedyPolicy(),
        "fixed-priority": lambda: policy.FixedPriorityPolicy(),
        "leases-only": lambda: policy.LeasePolicy(),
        "distributed-greedy": lambda: policy.DistributedGreedyPolicy(),
        "oracle-upper-bound": lambda: policy.OracleUpperBoundPolicy(scene),
    }
    factory = registry.get(key)
    if factory is None:
        raise G2Error(f"unknown policy key {key!r}; known: {', '.join(sorted(registry))}")
    return factory()


def run_case(
    *,
    faults_on: bool,
    policy_key: str = "distributed-greedy",
    seed: int = G2_SEED,
) -> dict[str, Any]:
    """Run one G2 case with the shared harness, bus, recorder and rule engine."""
    config, scene = build_scene()
    plan = partition_plan() if faults_on else faults.empty_plan()
    selected_policy = policy_for_key(policy_key, scene)
    bus = comms.ClaimBus(list(G2_SATELLITES), faults=plan)
    recorder = trace.TraceRecorder(label=f"g2:{'fault' if faults_on else 'control'}")
    result = policy.run_episode(
        scene, selected_policy, faults=plan, comms=bus, trace=recorder,
    )
    findings = invariants.InvariantEngine().evaluate(result, faults=plan)
    manifest = trace.ReplayManifest(
        scenario=scenario_payload(),
        policy_key=policy_key,
        seed=seed,
        faults=plan.to_dict(),
        expected_trace_hash=recorder.hash(),
        horizon_s=G2_HORIZON_S,
        step_s=G2_STEP_S,
        versions={"g2": G2_VERSION, "policy": policy.POLICY_VERSION, "faults": faults.FAULT_VERSION},
        label=f"g2-{'fault-on' if faults_on else 'fault-off'}",
    )
    return {
        "config": config,
        "scene": scene,
        "faults": plan,
        "bus": bus,
        "recorder": recorder,
        "result": result,
        "findings": findings,
        "manifest": manifest,
    }


def replay(manifest: trace.ReplayManifest) -> dict[str, Any]:
    """Re-run a manifest through the same core and compare canonical hashes."""
    config = load_config(manifest.scenario)
    plan = faults.FaultPlan.from_dict(manifest.faults)
    scene = policy.Scene.build(
        config, [target_for(config)],
        horizon_s=manifest.horizon_s, step_s=manifest.step_s, seed=manifest.seed,
    )
    selected_policy = policy_for_key(manifest.policy_key, scene)
    bus = comms.ClaimBus(list(G2_SATELLITES), faults=plan)
    recorder = trace.TraceRecorder(label=f"replay:{manifest.label}")
    result = policy.run_episode(
        scene, selected_policy, faults=plan, comms=bus, trace=recorder,
    )
    findings = invariants.InvariantEngine().evaluate(result, faults=plan)
    replayed_hash = recorder.hash()
    return {
        "manifest": manifest.to_dict(),
        "trace_hash": replayed_hash,
        "matches": replayed_hash == manifest.expected_trace_hash,
        "metrics": result.metrics,
        "invariants": invariants.InvariantEngine().summary(result, faults=plan),
        "findings": [finding.to_dict() for finding in findings],
        "events": recorder.payload()["events"],
    }


def _positions_at(config: Any, time_s: float) -> dict[str, list[float]]:
    clock = SimulationClock(
        epoch_utc=config.epoch_utc,
        simulation_time_s=time_s,
        time_scale=1.0,
        paused=True,
    )
    snapshot = snapshot_at_time(config, time_s, clock)
    return {
        satellite.id: [round(value, 6) for value in satellite.position_eci_km]
        for satellite in snapshot.satellites
    }


def _target_eci(config: Any, target: EventTarget, time_s: float) -> list[float]:
    """Target direction in ECI at ``time_s`` so the globe overlay stays aligned."""
    ecef = frames.ecef_from_geodetic(target.lat_deg, target.lon_deg, 0.0)
    eci = frames.ecef_to_eci(ecef, None, time_s)
    return [round(value, 6) for value in eci]


def _timeline(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Recorded frames: positions from snapshots, links from the claim bus."""
    config = case["config"]
    bus = case["bus"]
    scene = case["scene"]
    target = target_for(config)
    decisions = case["result"].decisions
    times = sorted({0.0, scene.horizon_s} | {decision.time_s for decision in decisions})
    frames_out: list[dict[str, Any]] = []
    for time_s in times:
        topology = bus.topology(time_s)
        active = case["faults"].active_at(time_s)
        frames_out.append({
            "time_s": round(time_s, 6),
            "positions_km": _positions_at(config, time_s),
            "target_eci_km": _target_eci(config, target, time_s),
            "links": topology["edges"],
            "groups": topology["groups"],
            "active_faults": [fault.id for fault in active],
            "faulted_satellites": sorted({
                satellite for fault in active for satellite in fault.satellites
            }),
        })
    return frames_out


def _case_summary(case: dict[str, Any], *, include_events: bool = True) -> dict[str, Any]:
    findings = case["findings"]
    summary = invariants.InvariantEngine().summary(case["result"], faults=case["faults"])
    payload = {
        "policy": case["result"].policy,
        "metrics": case["result"].metrics,
        "trace_hash": case["recorder"].hash(),
        "events": case["recorder"].payload()["events"] if include_events else [],
        "invariants": summary,
        "manifest": case["manifest"].to_dict(),
    }
    payload["first_violation"] = findings[0].to_dict() if findings else None
    return payload


def run_demo(*, include_events: bool = True) -> dict[str, Any]:
    """Run the fault case twice, the control twice, and build the gate report."""
    config = build_config()
    fault_first = run_case(faults_on=True)
    fault_second = run_case(faults_on=True)
    control_first = run_case(faults_on=False)
    control_second = run_case(faults_on=False)
    replay_check = replay(fault_first["manifest"])

    fault_hash = fault_first["recorder"].hash()
    control_hash = control_first["recorder"].hash()
    fault_rules = fault_first["findings"]
    control_rules = control_first["findings"]
    duplicate = [finding for finding in fault_rules if finding.rule == "duplicate-task-claim"]
    control_duplicate = [finding for finding in control_rules if finding.rule == "duplicate-task-claim"]
    first_violation = duplicate[0] if duplicate else (fault_rules[0] if fault_rules else None)

    checks = {
        "fault_reproduces_duplicate_claim": bool(duplicate),
        "control_has_no_duplicate_claim": not control_duplicate,
        "same_seed_identical_hashes": (
            fault_hash == fault_second["recorder"].hash()
            and control_hash == control_second["recorder"].hash()
        ),
        "fault_and_control_hashes_differ": fault_hash != control_hash,
        "replay_reproduces_hash": replay_check["matches"],
        "violation_names_entities": bool(
            first_violation and len(first_violation.entities) >= 3
        ),
    }
    checks["g2_pass"] = all(checks.values())

    return {
        "version": G2_VERSION,
        "scenario_id": G2_SCENARIO_ID,
        "seed": G2_SEED,
        "horizon_s": G2_HORIZON_S,
        "step_s": G2_STEP_S,
        "groups": {key: list(value) for key, value in G2_GROUPS.items()},
        "target": target_for(config).observable_view(0.0),
        "checks": checks,
        "fault": _case_summary(fault_first, include_events=include_events),
        "control": _case_summary(control_first, include_events=include_events),
        "replay": {
            "trace_hash": replay_check["trace_hash"],
            "matches": replay_check["matches"],
        },
        "timeline": _timeline(fault_first),
        "first_violation": first_violation.to_dict() if first_violation else None,
        "limitations": [
            "claim delivery is a link-graph visibility model, not a radio or protocol simulation",
            "the local task view hides peer assignments that no delivered claim covered",
            "terminal task states remain globally visible so completed work is not re-proposed",
        ],
    }


_CACHE: dict[str, Any] = {}


def cached_demo() -> dict[str, Any]:
    """Build the report once per process; the UI polls it repeatedly."""
    if "report" not in _CACHE:
        _CACHE["report"] = run_demo()
    return _CACHE["report"]


def write_report(path: Path | None = None) -> Path:
    """Generate and persist the G2 acceptance report (used by CLI and tests)."""
    target = path or Path(__file__).resolve().parents[2] / "benchmarks" / "g2-acceptance.json"
    report = run_demo()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target
