"""Day 36: the randomized scenario pipeline that feeds episode generation.

The sampler turns one draw of the Day 44 distribution set into a *validated*
:class:`~constellab_ml.env.EpisodeSpec`: fleet geometry, per-satellite component
budgets, an event schedule, injected outages and a link model — all declared in
the spec so an episode can be rebuilt byte-for-byte and replayed.

Two design rules matter for training data:

* **Event schedules are anchored to real access opportunities.** A target is
  placed under the ground track of a chosen satellite and its active interval is
  widened to cover the pass the core actually predicts, so every sampled episode
  contains reachable work. The anchor choice and the offsets are seeded, so
  schedule diversity is real but never accidental.
* **Everything is validated before it is simulated.** Configs go through
  ``load_config`` (contract, physics, component registries) and the parameter
  tokens through the Day 40 compatibility report; an unsupported configuration
  is reported, never silently simulated.

Splits are assigned per *episode seed*, not per row position, so neighbouring
episodes do not leak from train into validation or test, and the test split
draws from the held-out parameter ranges the distribution set reserved for it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from constellab_core import dynamics, faults, frames, sensors
from constellab_core.config import load_config
from constellab_core.parameters import builtin_registry
from constellab_core.targets import EventTarget
from constellab_core import trace as core_trace

from . import param_encoder
from .env import EpisodeSpec
from .randomization import (
    CONDITION_CLASSES,
    RANDOMIZATION_VERSION,
    DistributionSet,
    RandomizationError,
    condition_classes_for,
    invalid_combinations,
)

SCENARIO_VERSION = "scenario-sampler-1"

# Named slices used by evaluation reports (first match wins, so the order is
# the priority order).
SCENARIO_FAMILIES: tuple[str, ...] = (
    "partitioned",
    "faulted",
    "degraded",
    "small-easy",
    "large-hard",
    "nominal",
)

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
DEFAULT_SPLITS: tuple[float, float, float] = (0.7, 0.15, 0.15)

COMPONENT_BASELINE: dict[str, dict[str, float]] = {
    "payload.imaging": {"data_rate_bit_s": 3.2e8, "resolution_m": 3.0, "swath_km": 40.0, "draw_w": 18.0},
    "power.solar-battery": {
        "efficiency_ratio": 0.28,
        "charge_efficiency_ratio": 0.9,
        "sun_tracking": True,  # type: ignore[dict-item]
    },
    "compute.capacity": {"draw_w": 25.0},
    "sensor.conical": {"draw_w": 12.0},
    "storage.buffer": {"initial_fill_ratio": 0.0},
}


class ScenarioError(RuntimeError):
    """A scenario could not be sampled, validated or anchored."""


@dataclass
class ScenarioSample:
    """One sampled episode plus the provenance a dataset needs to store."""

    spec: EpisodeSpec
    values: dict[str, Any]
    condition_class: str
    family: str
    dropped_targets: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.payload(),
            "values": dict(self.values),
            "condition_class": self.condition_class,
            "family": self.family,
            "dropped_targets": self.dropped_targets,
            "warnings": list(self.warnings),
        }


def scenario_family(values: Mapping[str, Any]) -> str:
    """The reporting slice a draw belongs to (deterministic, priority ordered)."""
    fleet = int(values.get("fleet.size", 0))
    targets = int(values.get("tasks.target_count", 0))
    partitions = int(values.get("network.partitions", 0))
    link_duty = float(values.get("network.duty_cycle", 1.0))
    fault_count = int(values.get("faults.count", 0))
    degraded = (
        float(values.get("energy.initial_charge_ratio", 1.0)) < 0.7
        or float(values.get("sensor.duty_cycle_ratio", 1.0)) < 0.4
        or float(values.get("compute.duty_cycle_ratio", 1.0)) < 0.35
    )
    if partitions > 0 or link_duty < 0.35:
        return "partitioned"
    if fault_count > 0:
        return "faulted"
    if degraded:
        return "degraded"
    if fleet <= 4 and targets <= 3:
        return "small-easy"
    if fleet >= 10 and targets >= 5:
        return "large-hard"
    return "nominal"


class ScenarioSampler:
    """Deterministic, resumable generator of randomized episodes."""

    def __init__(
        self,
        *,
        seed: int = 0,
        distribution_set: DistributionSet | None = None,
        splits: Sequence[float] | None = None,
        salt: str = "",
    ) -> None:
        self.seed = int(seed)
        self.distribution_set = distribution_set or DistributionSet.default()
        self.salt = salt
        self.splits = _validate_splits(splits or DEFAULT_SPLITS)
        self.registry = builtin_registry()

    # --- metadata ---------------------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        distributions = self.distribution_set.payload()
        return {
            "scenario_version": SCENARIO_VERSION,
            "randomization_version": RANDOMIZATION_VERSION,
            "seed": self.seed,
            "salt": self.salt,
            "splits": list(self.splits),
            "families": list(SCENARIO_FAMILIES),
            "condition_classes": list(CONDITION_CLASSES),
            "distribution_set": distributions,
            "distribution_hash": distributions["hash"],
            "parameter_schema_hash": self.registry.schema_hash(),
            "param_token_version": param_encoder.PARAM_TOKEN_VERSION,
        }

    MAX_SEED = 2**31 - 1

    def seed_for(self, index: int) -> int:
        """Per-episode seed (32-bit so the config payload stays JSON-exact)."""
        bank = core_trace.SeedBank(self.seed + hash_salt(self.salt))
        return bank.derive(f"episode-{index}") % self.MAX_SEED

    def split_for(self, index: int, *, split: str | None = None) -> str:
        if split is not None:
            if split not in SPLIT_NAMES:
                raise ScenarioError(f"split {split!r} must be one of {list(SPLIT_NAMES)}")
            return split
        seed = self.seed_for(index)
        bucket = (seed % 10_000) / 10_000.0
        running = 0.0
        for name, share in zip(SPLIT_NAMES, self.splits):
            running += share
            if bucket < running:
                return name
        return SPLIT_NAMES[-1]

    def condition_class_for(self, index: int) -> str:
        return condition_classes_for(index + self.seed)

    # --- sampling ---------------------------------------------------------------------
    def sample(
        self,
        index: int,
        *,
        split: str | None = None,
        condition_class: str | None = None,
        held_out: bool | None = None,
    ) -> ScenarioSample:
        split_name = self.split_for(index, split=split)
        condition = condition_class or self.condition_class_for(index)
        held = (split_name == "test") if held_out is None else held_out
        seed = self.seed_for(index)
        rng = random.Random(seed)
        try:
            values = self.distribution_set.sample(
                rng, condition_class=condition, held_out=held
            )
        except RandomizationError as exc:
            raise ScenarioError(f"episode {index} could not be drawn: {exc}") from exc

        horizon_s = float(values["time.horizon_s"])
        step_s = float(values["time.step_s"])
        if step_s * 4.0 > horizon_s:  # constraint, defended here too
            step_s = max(10.0, horizon_s / 8.0)

        config = self.build_config(values, index=index, seed=seed)
        typed = load_config(config)
        context = param_encoder.build_parameter_context(config, registry=self.registry)
        warnings: list[str] = []
        if not context["compatibility"]["compatible"]:
            warnings.extend(context["compatibility"]["reasons"])

        targets, dropped = self.build_targets(
            typed, values, index=index, seed=seed, horizon_s=horizon_s
        )
        if not targets:
            raise ScenarioError(f"episode {index}: no reachable event could be scheduled")

        plan = self.build_faults(values, typed, index=index, seed=seed, horizon_s=horizon_s)
        contact = {
            "range_km": float(values["network.link_range_km"]),
            "duty_cycle": float(values["network.duty_cycle"]),
            "period_s": 900.0,
            "model": "range-and-duty-cycle",
        }
        spec = EpisodeSpec(
            episode_id=f"{split_name}-{index:05d}",
            config=config,
            targets=[target.to_dict() for target in targets],
            horizon_s=horizon_s,
            step_s=step_s,
            seed=seed,
            window_step_s=min(30.0, step_s),
            family=scenario_family(values),
            label=f"{condition} {scenario_family(values)}",
            fault_plan=plan.to_dict(),
            parameters=context,
            parameter_registry=self.registry.schema_hash(),
            contact=contact,
            versions={
                "scenario": SCENARIO_VERSION,
                "randomization": RANDOMIZATION_VERSION,
                "param_tokens": param_encoder.PARAM_TOKEN_VERSION,
                "env": "swarm-env-1",
            },
            split=split_name,
        )
        return ScenarioSample(
            spec=spec,
            values=values,
            condition_class=condition,
            family=spec.family,
            dropped_targets=dropped,
            warnings=warnings,
        )

    # --- pieces -----------------------------------------------------------------------
    def build_config(self, values: Mapping[str, Any], *, index: int, seed: int) -> dict[str, Any]:
        size = int(values["fleet.size"])
        planes = max(1, min(int(values["orbit.planes"]), size))
        semi_major_axis = 6378.137 + float(values["orbit.altitude_km"])
        inclination = float(values["orbit.inclination_deg"])
        eccentricity = float(values["orbit.eccentricity"])
        components = self.build_components(values)
        rng = random.Random(seed ^ 0x5EED)
        satellites = []
        for position in range(size):
            plane = position % planes
            slot = position // planes
            raan = (plane * 360.0 / planes) % 360.0
            anomaly = (
                slot * 360.0 / max(1, size // planes) + plane * 360.0 / size
            ) % 360.0
            satellites.append({
                "id": f"ml-{position:03d}",
                "name": f"ML-{plane + 1:02d}/{slot + 1:03d}",
                "orbit": {
                    "semi_major_axis_km": round(semi_major_axis, 6),
                    "eccentricity": round(eccentricity, 6),
                    "inclination_deg": round(inclination, 6),
                    "raan_deg": round(raan, 6),
                    "argument_of_perigee_deg": 0.0,
                    "true_anomaly_deg": round(anomaly, 6),
                },
                "components": components,
            })
        return {
            "schema_version": _schema_version(),
            "scenario_id": f"ml-scenario-{index:05d}",
            "name": f"ML scenario {index:05d}",
            "seed": seed,
            "epoch_utc": "2026-10-12T00:00:00Z",
            "time_scale": 60.0,
            "dynamics_model": "keplerian-two-body",
            "satellites": satellites,
        }

    def build_components(self, values: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Declared component budget for one satellite (registry keys only)."""
        parameters = {
            "sensor.conical": {
                "half_angle_deg": round(float(values["sensor.half_angle_deg"]), 6),
                "max_range_km": round(float(values["sensor.max_range_km"]), 6),
                "integration_time_s": round(float(values["sensor.integration_time_s"]), 6),
                "duty_cycle_ratio": round(float(values["sensor.duty_cycle_ratio"]), 6),
                "draw_w": COMPONENT_BASELINE["sensor.conical"]["draw_w"],
            },
            "power.solar-battery": {
                "array_area_m2": round(float(values["energy.array_area_m2"]), 6),
                "battery_capacity_j": round(float(values["energy.battery_capacity_j"]), 6),
                "base_load_w": round(float(values["energy.base_load_w"]), 6),
                "initial_charge_ratio": round(float(values["energy.initial_charge_ratio"]), 6),
                **COMPONENT_BASELINE["power.solar-battery"],
            },
            "storage.buffer": {
                "capacity_bit": round(float(values["storage.capacity_bit"]), 6),
                "write_rate_bit_s": round(float(values["storage.write_rate_bit_s"]), 6),
                "read_rate_bit_s": round(float(values["storage.read_rate_bit_s"]), 6),
                **COMPONENT_BASELINE["storage.buffer"],
            },
            "compute.capacity": {
                "throughput_hz": round(float(values["compute.throughput_hz"]), 6),
                "duty_cycle_ratio": round(float(values["compute.duty_cycle_ratio"]), 6),
                **COMPONENT_BASELINE["compute.capacity"],
            },
            "payload.imaging": dict(COMPONENT_BASELINE["payload.imaging"]),
        }
        return [
            {"type": owner, "parameters": dict(params)}
            for owner, params in sorted(parameters.items())
        ]

    def build_targets(
        self,
        config: Any,
        values: Mapping[str, Any],
        *,
        index: int,
        seed: int,
        horizon_s: float,
    ) -> tuple[list[EventTarget], int]:
        """Schedule events under real access opportunities, then widen the window."""
        rng = random.Random(seed ^ 0x7A26E7)
        count = int(values["tasks.target_count"])
        deadline_scale = float(values["tasks.deadline_scale"])
        kinds = _kind_sequence(str(values["tasks.kind_mix"]), count)
        urgency_plan = _urgency_plan(str(values["tasks.urgency_mix"]), count)
        satellites = sorted(config.satellites, key=lambda sat: sat.id)
        model = dynamics.dynamics_model()
        targets: list[EventTarget] = []
        dropped = 0
        for position in range(count):
            kind = kinds[position]
            anchor_candidates = [
                satellites[(position + offset) % len(satellites)]
                for offset in range(min(len(satellites), 3))
            ]
            opportunity = None
            for anchor in anchor_candidates:
                opportunity = self._anchor_opportunity(
                    config, anchor, kind=kind, rng=rng, horizon_s=horizon_s, seed=seed
                )
                if opportunity is not None:
                    break
            if opportunity is None:
                dropped += 1
                continue
            latitude, longitude, window, anchor = opportunity
            slack = max(240.0, deadline_scale * horizon_s / (2.0 * max(1, count)))
            start = max(0.0, min(window.start_s - slack, horizon_s - 2.0))
            end = min(horizon_s, max(window.end_s + slack, start + 2.0))
            targets.append(
                EventTarget(
                    id=f"{_id_prefix(kind)}-{position:02d}",
                    kind=kind,
                    lat_deg=latitude,
                    lon_deg=longitude,
                    start_s=round(start, 3),
                    end_s=round(end, 3),
                    radius_km=_radius_for(kind, rng),
                    urgency=urgency_plan[position],
                    confidence=round(rng.uniform(0.2, 0.95), 4),
                    requirements=_requirements_for(kind),
                    truth={
                        "scheduled": True,
                        "anchor": anchor.id,
                        "anchor_window_s": [round(window.start_s, 3), round(window.end_s, 3)],
                    },
                )
            )
        return targets, dropped

    def _anchor_opportunity(
        self,
        config: Any,
        satellite: Any,
        *,
        kind: str,
        rng: random.Random,
        horizon_s: float,
        seed: int,
    ) -> tuple[float, float, Any, Any] | None:
        """Find a point under the ground track that this satellite really can see.

        An overflight is not enough: the sensor has a duty cycle and a maximum
        slant range, so a candidate time is accepted only when the core predicts
        a *usable* access window inside a short scan around it. Candidates are
        seeded, which keeps the schedule diverse without making it accidental.
        """
        model = dynamics.dynamics_model()
        sensor_config = sensors.sensor_config_for(config, satellite, seed=seed)
        for _ in range(10):
            # Keep away from the very edges, where no decision could be made.
            anchor_time = round(horizon_s * (0.15 + 0.7 * rng.random()), 3)
            state_position = model.state_at(satellite.orbit, anchor_time)[0]
            point = frames.geodetic_from_eci(state_position, anchor_time)
            latitude = _wrap_latitude(point.latitude_deg + rng.uniform(-3.0, 3.0))
            longitude = _wrap_longitude(point.longitude_deg + rng.uniform(-3.0, 3.0))
            probe = EventTarget(
                id="anchor-probe",
                kind=kind,
                lat_deg=latitude,
                lon_deg=longitude,
                start_s=0.0,
                end_s=max(1.0, horizon_s),
                radius_km=1.0,
                urgency=3,
                requirements=_requirements_for(kind),
                truth={},
            )
            window_s = (
                max(0.0, anchor_time - 420.0),
                min(horizon_s, anchor_time + 420.0),
            )
            windows = sensors.access_windows(
                satellite.id,
                satellite.orbit,
                probe,
                config=sensor_config,
                horizon_s=horizon_s,
                step_s=30.0,
                seed=seed,
                window_s=window_s,
            )
            usable = [window for window in windows if window.usable]
            if usable:
                return latitude, longitude, min(usable, key=lambda w: (w.start_s, w.end_s)), satellite
        return None

    def build_faults(
        self, values: Mapping[str, Any], config: Any, *, index: int, seed: int, horizon_s: float
    ) -> faults.FaultPlan:
        rng = random.Random(seed ^ 0x1A57)
        identifiers = [sat.id for sat in config.satellites]
        count = int(values["faults.count"])
        duration = max(60.0, float(values["faults.duration_scale"]) * horizon_s)
        kinds = _fault_kinds(str(values["faults.kind_mix"]))
        plans: list[faults.Fault] = []
        if count > 0:
            plan = faults.random_plan(
                identifiers,
                horizon_s=horizon_s,
                seed=rng.randrange(1, 2**31 - 1),
                count=count,
                kinds=kinds,
                min_duration_s=duration,
                max_duration_s=duration,
            )
            plans.extend(plan.faults)
        partitions = int(values["network.partitions"])
        for position in range(partitions):
            group = rng.randint(1, max(1, len(identifiers) - 1))
            cut = sorted(rng.sample(identifiers, group))
            peers = [item for item in identifiers if item not in cut]
            if not peers:
                continue
            start = round(rng.uniform(0.0, 0.3) * horizon_s, 3)
            plans.append(faults.Fault(
                id=f"partition-{position:02d}",
                kind="link-cut",
                start_s=start,
                end_s=horizon_s,
                satellites=tuple(cut),
                peers=tuple(peers),
                reason="scenario partition: no claim crosses between the groups",
            ))
        # The core owns the plan vocabulary: seeded outages are a 'random' plan,
        # while an outage set that only partitions the fleet is 'manual'. The
        # sampler's own provenance travels in ``generator``.
        return faults.FaultPlan(
            tuple(sorted(plans, key=lambda fault: (fault.start_s, fault.id))),
            strategy="random" if count > 0 else "manual",
            seed=seed,
            generator={
                "count": count,
                "partitions": partitions,
                "kinds": list(kinds),
                "duration_s": duration,
                "horizon_s": horizon_s,
                "randomization_version": RANDOMIZATION_VERSION,
            },
        )

    # --- coverage ---------------------------------------------------------------------
    def coverage(self, samples: Sequence[ScenarioSample]) -> dict[str, Any]:
        report = self.distribution_set.summarize([sample.values for sample in samples])
        families: dict[str, int] = {}
        conditions: dict[str, int] = {}
        splits: dict[str, int] = {}
        dropped = 0
        for sample in samples:
            families[sample.family] = families.get(sample.family, 0) + 1
            conditions[sample.condition_class] = conditions.get(sample.condition_class, 0) + 1
            splits[sample.spec.split] = splits.get(sample.spec.split, 0) + 1
            dropped += sample.dropped_targets
        report["scenario_families"] = {name: families.get(name, 0) for name in SCENARIO_FAMILIES}
        report["condition_classes"] = {name: conditions.get(name, 0) for name in CONDITION_CLASSES}
        report["splits"] = {name: splits.get(name, 0) for name in SPLIT_NAMES}
        report["dropped_unreachable_targets"] = dropped
        report["scenario_version"] = SCENARIO_VERSION
        # Explicit, probed rejections: every entry names a real combination the
        # sampler refuses to simulate, not merely a rule declared in a file.
        report["invalid_combinations"] = invalid_combinations(self.distribution_set)
        return report


def _validate_splits(splits: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in splits)
    if len(values) != len(SPLIT_NAMES):
        raise ScenarioError(f"splits must hold {len(SPLIT_NAMES)} shares")
    if any(value <= 0.0 for value in values):
        raise ScenarioError("every split share must be positive")
    total = sum(values)
    if abs(total - 1.0) > 1e-6:
        raise ScenarioError(f"split shares must sum to 1.0, got {total}")
    return values  # type: ignore[return-value]


def _kind_sequence(mix: str, count: int) -> list[str]:
    if mix == "hotspot":
        return ["hotspot"] * count
    if mix == "change-detection":
        return ["change-detection"] * count
    cycle = ["hotspot", "flood-region", "change-detection"]
    return [cycle[position % len(cycle)] for position in range(count)]


def _urgency_plan(mix: str, count: int) -> list[int]:
    if mix == "low":
        return [2] * count
    if mix == "high":
        return [5] * count
    return [2 + (position % 4) for position in range(count)]


def _fault_kinds(mix: str) -> tuple[str, ...]:
    if mix == "link-cut":
        return ("link-cut",)
    if mix == "satellite-failure":
        return ("satellite-failure", "sensor-degrade")
    return ("link-cut", "satellite-failure", "sensor-degrade", "message-delay")


def _id_prefix(kind: str) -> str:
    return {"hotspot": "fire", "flood-region": "flood", "change-detection": "change"}[kind]


def _radius_for(kind: str, rng: random.Random) -> float:
    if kind == "flood-region":
        return round(rng.uniform(20.0, 120.0), 3)
    if kind == "change-detection":
        return round(rng.uniform(1.0, 15.0), 3)
    return round(rng.uniform(0.5, 8.0), 3)


def _requirements_for(kind: str) -> dict[str, float]:
    if kind == "flood-region":
        return {"min_resolution_m": 30.0}
    if kind == "change-detection":
        return {"min_confidence": 0.4, "min_obs_duration_s": 30.0}
    return {"min_confidence": 0.45}


def _wrap_latitude(value: float) -> float:
    return max(-89.0, min(89.0, round(value, 5)))


def _wrap_longitude(value: float) -> float:
    wrapped = math.fmod(value + 180.0, 360.0)
    if wrapped < 0.0:
        wrapped += 360.0
    return round(wrapped - 180.0, 5)


def _schema_version() -> str:
    from constellab_core import SCHEMA_VERSION

    return SCHEMA_VERSION


def hash_salt(salt: str) -> int:
    """A stable integer for a salt string (so campaigns differ by name only)."""
    if not salt:
        return 0
    import hashlib

    return int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)


def sample_many(sampler: ScenarioSampler, indices: Iterable[int], **kwargs: Any) -> list[ScenarioSample]:
    return [sampler.sample(index, **kwargs) for index in indices]


__all__ = [
    "SCENARIO_VERSION",
    "SCENARIO_FAMILIES",
    "SPLIT_NAMES",
    "DEFAULT_SPLITS",
    "ScenarioError",
    "ScenarioSample",
    "ScenarioSampler",
    "scenario_family",
    "sample_many",
    "hash_salt",
]
