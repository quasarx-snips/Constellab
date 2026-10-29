"""Day 44: versioned domain randomization with correlated, conditioned sampling.

The sampler draws *structured* scenario parameters — orbit, fleet, sensor,
energy, storage, compute, task density/deadlines, network capability and fault
intensity — from a declarative, versioned distribution set. Three properties
matter and are all tested:

* **Correlation.** Related parameters share a latent factor (capability,
  budget, harshness) instead of being drawn independently, so the campaign does
  not fill up with combinations that are physically incoherent ("cheap sensor,
  huge array, 40 satellites").
* **Conditional constraints.** Combinations that are invalid by construction
  are *rejected by name*, and sampling retries with a different draw; a draw
  that never validates is reported as such, never silently accepted as valid.
* **Coverage.** A campaign reports how much of each parameter family, and of
  each family-pair slice, was actually sampled, and which regions stayed
  unexplored.
* **Held-out ranges.** Distributions may declare a sub-range reserved for
  robustness evaluation; training draws never use it.

Sampling is seeded and deterministic: the same seed and distribution version
reproduce the same values, which is what makes a campaign resumable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

RANDOMIZATION_VERSION = "randomization-1"

# Condition classes a scenario family can be sampled in.
CONDITION_CLASSES: tuple[str, ...] = ("nominal", "boundary", "noisy", "degraded", "rare")

# Latent correlation groups.
GROUPS: tuple[str, ...] = ("capability", "budget", "harshness")

_FAMILY_LABELS = {
    "orbit": "Orbit geometry",
    "fleet": "Fleet size",
    "sensor": "Sensor budget",
    "energy": "Energy budget",
    "storage": "Storage budget",
    "compute": "Compute budget",
    "tasks": "Target density and deadlines",
    "network": "Communication capability",
    "faults": "Fault intensity",
    "time": "Episode length",
}
FAMILIES: tuple[str, ...] = tuple(_FAMILY_LABELS)


class RandomizationError(ValueError):
    """A distribution set or a draw is not usable."""


@dataclass(frozen=True)
class Distribution:
    """One sampled parameter (or small choice set) with its declared ranges."""

    key: str
    family: str
    kind: str = "float"  # float | int | choice
    low: float = 0.0
    high: float = 1.0
    choices: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()
    unit: str = ""
    group: str | None = None
    group_weight: float = 0.0
    noise: float = 0.25
    held_out_span: tuple[float, float] | None = None
    scale: str = "linear"  # linear | log
    note: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise RandomizationError(f"unknown family {self.family!r}; known: {list(FAMILIES)}")
        if self.kind not in ("float", "int", "choice"):
            raise RandomizationError(f"distribution kind {self.kind!r} must be float, int or choice")
        if self.group is not None and self.group not in GROUPS:
            raise RandomizationError(f"unknown correlation group {self.group!r}")
        if self.kind == "choice":
            if not self.choices:
                raise RandomizationError(f"choice distribution {self.key!r} needs choices")
            if self.weights and len(self.weights) != len(self.choices):
                raise RandomizationError(f"{self.key!r}: weights must match choices")
        else:
            if not self.low < self.high:
                raise RandomizationError(f"{self.key!r}: low must be below high")
            if self.scale not in ("linear", "log"):
                raise RandomizationError(f"{self.key!r}: scale must be linear or log")
            if self.scale == "log" and self.low <= 0.0:
                raise RandomizationError(f"{self.key!r}: a log range needs a positive low bound")
        if self.held_out_span is not None:
            low, high = self.held_out_span
            if not low < high:
                raise RandomizationError(f"{self.key!r}: held_out_span must be increasing")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "family": self.family,
            "kind": self.kind,
            "unit": self.unit,
            "group": self.group,
            "group_weight": self.group_weight,
            "noise": self.noise,
            "scale": self.scale,
            "held_out_span": list(self.held_out_span) if self.held_out_span else None,
            "note": self.note,
        }
        if self.kind == "choice":
            payload["choices"] = list(self.choices)
            payload["weights"] = list(self.weights) if self.weights else None
        else:
            payload["low"] = self.low
            payload["high"] = self.high
        return payload

    # --- sampling ---------------------------------------------------------------------
    def _quantile(self, u: float, *, held_out: bool) -> float:
        low, high = self.low, self.high
        if held_out and self.held_out_span is not None:
            low, high = self.held_out_span
        if self.scale == "log":
            return math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
        return low + u * (high - low)

    def value(self, u: float, *, held_out: bool = False) -> Any:
        value = self._quantile(min(0.999, max(0.001, u)), held_out=held_out)
        if self.kind == "int":
            return int(round(value))
        return round(float(value), 6)

    def draw_choice(self, rng: random.Random) -> str:
        if self.weights:
            return rng.choices(list(self.choices), weights=list(self.weights), k=1)[0]
        return rng.choice(list(self.choices))


def constraint_identifier(key: str) -> str:
    """``fleet.size`` -> ``fleet__size``: a rule is written in dotted keys."""
    return key.replace(".", "__")


@dataclass(frozen=True)
class Constraint:
    """A named rule that rejects a combination before it is ever simulated."""

    name: str
    applies: tuple[str, ...]
    rule: str
    description: str

    def __post_init__(self) -> None:
        for key in self.applies:
            if key not in self.rule:
                raise RandomizationError(
                    f"constraint {self.name!r} applies to {key!r} but its rule never mentions it"
                )

    def compiled(self) -> str:
        """The rule with every dotted key replaced by a legal identifier."""
        compiled = self.rule
        for key in sorted(self.applies, key=len, reverse=True):
            compiled = compiled.replace(key, constraint_identifier(key))
        return compiled

    def violated(self, values: Mapping[str, Any]) -> bool:
        """True when the draw breaks this rule (the rule states the requirement)."""
        if not all(key in values for key in self.applies):
            return False
        namespace: dict[str, Any] = {
            constraint_identifier(key): values[key] for key in self.applies
        }
        try:
            satisfied = bool(eval(self.compiled(), {"__builtins__": {}}, namespace))  # noqa: S307
        except Exception as exc:  # pragma: no cover - a broken rule is a bug, not a draw
            raise RandomizationError(f"constraint {self.name!r} could not be evaluated: {exc}") from exc
        return not satisfied

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "applies_to": list(self.applies),
            "requirement": self.rule,
            "description": self.description,
        }


def default_constraints() -> tuple[Constraint, ...]:
    """The conditional constraints the default set ships with."""
    return (
        Constraint(
            name="fleet-at-least-two",
            applies=("fleet.size",),
            rule="fleet.size >= 2",
            description="a swarm episode needs at least two satellites",
        ),
        Constraint(
            name="partitions-fit-the-fleet",
            applies=("network.partitions", "fleet.size"),
            rule="network.partitions <= fleet.size - 1",
            description="a partition plan can not split a fleet into more groups than it has satellites",
        ),
        Constraint(
            name="faults-fit-the-fleet",
            applies=("faults.count", "fleet.size"),
            rule="faults.count == 0 or faults.count <= fleet.size",
            description="at most one injected fault per satellite",
        ),
        Constraint(
            name="fault-window-fits-the-episode",
            applies=("faults.count", "faults.duration_scale"),
            rule="faults.count == 0 or faults.duration_scale > 0.0",
            description="an injected fault needs a positive duration",
        ),
        Constraint(
            name="buffer-rate-fits-capacity",
            applies=("storage.capacity_bit", "storage.write_rate_bit_s"),
            rule="storage.write_rate_bit_s * 600.0 <= storage.capacity_bit",
            description="the buffer write rate must be able to fill its capacity within ten minutes",
        ),
        Constraint(
            name="downlink-rate-fits-write-rate",
            applies=("storage.write_rate_bit_s", "storage.read_rate_bit_s"),
            rule="storage.read_rate_bit_s <= storage.write_rate_bit_s * 4.0",
            description="a drain rate far above the write rate would model a free buffer",
        ),
        Constraint(
            name="targets-need-a-deadline",
            applies=("tasks.deadline_scale",),
            rule="tasks.deadline_scale >= 0.2",
            description="a deadline scale below 0.2 s per event second leaves no window to observe",
        ),
        Constraint(
            name="sensor-range-covers-the-orbit",
            applies=("sensor.max_range_km", "orbit.altitude_km"),
            rule="sensor.max_range_km >= orbit.altitude_km",
            description="a slant range below the orbital altitude can never see the ground",
        ),
        Constraint(
            name="step-fits-the-horizon",
            applies=("time.horizon_s", "time.step_s"),
            rule="time.step_s * 4.0 <= time.horizon_s",
            description="an episode needs at least a few decision steps",
        ),
    )


def default_distributions() -> tuple[Distribution, ...]:
    """The shipped distribution set: ten families, three latent factors."""
    return (
        # --- orbit ---------------------------------------------------------------------
        Distribution("orbit.altitude_km", "orbit", low=450.0, high=1_100.0, unit="km",
                     group="capability", group_weight=0.4, noise=0.2,
                     held_out_span=(1_100.0, 1_400.0), note="circular altitude"),
        Distribution("orbit.inclination_deg", "orbit", low=30.0, high=98.0, unit="deg", noise=0.3,
                     held_out_span=(98.0, 110.0)),
        Distribution("orbit.eccentricity", "orbit", low=0.0, high=0.02, noise=0.2),
        Distribution("orbit.planes", "orbit", kind="int", low=1, high=6, noise=0.1,
                     note="Walker-delta plane count"),
        # --- fleet ---------------------------------------------------------------------
        Distribution("fleet.size", "fleet", kind="int", low=2, high=16, noise=0.5,
                     held_out_span=(17.0, 24.0), note="satellites in the episode"),
        # --- sensor --------------------------------------------------------------------
        Distribution("sensor.half_angle_deg", "sensor", low=8.0, high=55.0, unit="deg",
                     group="capability", group_weight=0.6, noise=0.25,
                     held_out_span=(4.0, 8.0), note="field of regard"),
        Distribution("sensor.max_range_km", "sensor", low=700.0, high=6_000.0, unit="km",
                     scale="log", group="capability", group_weight=0.45, noise=0.25,
                     note="slant range the instrument can still look at"),
        Distribution("sensor.integration_time_s", "sensor", low=0.1, high=2.0, unit="s",
                     scale="log", noise=0.3, note="per-look integration time"),
        Distribution("sensor.duty_cycle_ratio", "sensor", low=0.25, high=0.95,
                     group="capability", group_weight=0.3, noise=0.2,
                     note="share of each availability period the sensor is on"),
        # --- energy --------------------------------------------------------------------
        Distribution("energy.battery_capacity_j", "energy", low=5.0e4, high=5.0e6, unit="J",
                     scale="log", group="budget", group_weight=0.6, noise=0.2),
        Distribution("energy.array_area_m2", "energy", low=0.2, high=3.0, unit="m^2",
                     group="budget", group_weight=0.5, noise=0.2),
        Distribution("energy.base_load_w", "energy", low=10.0, high=80.0, unit="W", noise=0.3),
        Distribution("energy.initial_charge_ratio", "energy", low=0.5, high=1.0, noise=0.3),
        # --- storage -------------------------------------------------------------------
        Distribution("storage.capacity_bit", "storage", low=1.0e8, high=1.0e10, unit="bit",
                     scale="log", group="budget", group_weight=0.4, noise=0.25),
        Distribution("storage.write_rate_bit_s", "storage", low=1.0e5, high=5.0e6, unit="bit/s",
                     scale="log", group="budget", group_weight=0.3, noise=0.25),
        Distribution("storage.read_rate_bit_s", "storage", low=1.0e5, high=2.0e7, unit="bit/s",
                     scale="log", noise=0.3),
        # --- compute -------------------------------------------------------------------
        Distribution("compute.throughput_hz", "compute", low=1.0e8, high=1.0e10, unit="Hz",
                     scale="log", group="capability", group_weight=0.4, noise=0.25),
        Distribution("compute.duty_cycle_ratio", "compute", low=0.2, high=1.0,
                     group="budget", group_weight=0.3, noise=0.25),
        # --- tasks ---------------------------------------------------------------------
        Distribution("tasks.target_count", "tasks", kind="int", low=1, high=8,
                     group="harshness", group_weight=0.5, noise=0.3,
                     held_out_span=(9.0, 14.0), note="events in the episode"),
        Distribution("tasks.deadline_scale", "tasks", low=0.4, high=1.6,
                     group="harshness", group_weight=-0.5, noise=0.25,
                     note="fraction of the horizon a target stays active"),
        Distribution("tasks.kind_mix", "tasks", kind="choice",
                     choices=("hotspot", "mixed", "change-detection"),
                     weights=(0.5, 0.35, 0.15), note="event kind mixture"),
        Distribution("tasks.urgency_mix", "tasks", kind="choice",
                     choices=("low", "mixed", "high"), weights=(0.2, 0.6, 0.2)),
        # --- network -------------------------------------------------------------------
        Distribution("network.link_range_km", "network", low=1_000.0, high=20_000.0, unit="km",
                     group="capability", group_weight=-0.7, noise=0.2,
                     held_out_span=(200.0, 1_000.0), note="contact range; keeps the fleet connected"),
        Distribution("network.duty_cycle", "network", low=0.2, high=1.0,
                     group="capability", group_weight=-0.35, noise=0.25,
                     note="share of time a link is available (intermittent contacts)"),
        Distribution("network.partitions", "network", kind="int", low=0, high=2,
                     group="harshness", group_weight=0.5, noise=0.2),
        # --- faults --------------------------------------------------------------------
        Distribution("faults.count", "faults", kind="int", low=0, high=4,
                     group="harshness", group_weight=0.7, noise=0.3,
                     held_out_span=(5.0, 6.0), note="injected outages"),
        Distribution("faults.duration_scale", "faults", low=0.1, high=0.6,
                     group="harshness", group_weight=0.5, noise=0.25,
                     note="share of the horizon an outage lasts"),
        Distribution("faults.kind_mix", "faults", kind="choice",
                     choices=("link-cut", "mixed", "satellite-failure"),
                     weights=(0.45, 0.4, 0.15)),
        # --- time ----------------------------------------------------------------------
        Distribution("time.horizon_s", "time", low=1_800.0, high=7_200.0, unit="s", noise=0.4,
                     held_out_span=(8_000.0, 10_800.0)),
        Distribution("time.step_s", "time", low=30.0, high=120.0, unit="s", noise=0.3),
    )


@dataclass
class DistributionSet:
    """A versioned, hashed collection of distributions plus its constraints."""

    distributions: tuple[Distribution, ...]
    constraints: tuple[Constraint, ...]
    version: str = RANDOMIZATION_VERSION
    label: str = "default"

    def __post_init__(self) -> None:
        keys = [item.key for item in self.distributions]
        if len(set(keys)) != len(keys):
            raise RandomizationError("duplicate distribution keys in the set")
        for item in self.distributions:
            if item.group is not None and item.group_weight == 0.0:
                raise RandomizationError(
                    f"{item.key!r} declares group {item.group!r} with a zero weight"
                )

    @classmethod
    def default(cls) -> "DistributionSet":
        return cls(distributions=default_distributions(), constraints=default_constraints())

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.distributions)

    def by_key(self, key: str) -> Distribution:
        for item in self.distributions:
            if item.key == key:
                return item
        raise RandomizationError(f"unknown parameter key {key!r}")

    def families(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {family: [] for family in FAMILIES}
        for item in self.distributions:
            grouped[item.family].append(item.key)
        return {family: keys for family, keys in grouped.items() if keys}

    def payload(self) -> dict[str, Any]:
        import hashlib
        import json

        body = {
            "version": self.version,
            "label": self.label,
            "families": _FAMILY_LABELS,
            "groups": list(GROUPS),
            "condition_classes": list(CONDITION_CLASSES),
            "distributions": [item.to_dict() for item in self.distributions],
            "constraints": [item.to_dict() for item in self.constraints],
        }
        body["hash"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return body

    def held_out_keys(self) -> list[str]:
        return [item.key for item in self.distributions if item.held_out_span is not None]

    # --- sampling ---------------------------------------------------------------------
    def sample(
        self,
        rng: random.Random,
        *,
        condition_class: str = "nominal",
        held_out: bool = False,
        max_attempts: int = 64,
    ) -> dict[str, Any]:
        """Draw one valid value set; invalid combinations are rejected by name."""
        if condition_class not in CONDITION_CLASSES:
            raise RandomizationError(
                f"condition class {condition_class!r} must be one of {list(CONDITION_CLASSES)}"
            )
        latents = {group: _latent(rng, condition_class, group) for group in GROUPS}
        last_rejections: list[str] = []
        for _ in range(max(1, max_attempts)):
            values: dict[str, Any] = {}
            for item in self.distributions:
                if item.kind == "choice":
                    values[item.key] = item.draw_choice(rng)
                    continue
                if item.group is None:
                    u = _uniform(rng, condition_class)
                else:
                    z = latents[item.group]
                    swing = item.group_weight
                    if condition_class == "degraded" and swing < 0.0:
                        z = abs(z) * 1.2  # degraded pushes the *bad* direction of every group
                    noise = item.noise * _noise_scale(condition_class)
                    u = min(0.999, max(0.001, 0.5 + 0.5 * swing * z + noise * (rng.random() - 0.5) * 2.0))
                values[item.key] = item.value(u, held_out=held_out)
            rejections = self.violations(values)
            if not rejections:
                return values
            last_rejections = rejections
        raise RandomizationError(
            "no valid draw after "
            f"{max_attempts} attempts; last rejected combinations: {last_rejections}"
        )

    def violations(self, values: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Named constraint violations for one draw (empty when the draw is valid)."""
        found: list[dict[str, Any]] = []
        for constraint in self.constraints:
            if constraint.violated(values):
                found.append({
                    "name": constraint.name,
                    "applies_to": list(constraint.applies),
                    "description": constraint.description,
                    "values": {key: values[key] for key in constraint.applies if key in values},
                })
        return found

    def summarize(self, draws: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Coverage report: per key, per family and per family-pair slice."""
        keys: dict[str, Any] = {}
        for item in self.distributions:
            observed = [draw[item.key] for draw in draws if item.key in draw]
            entry: dict[str, Any] = {
                "family": item.family,
                "samples": len(observed),
                "held_out_reserved": list(item.held_out_span) if item.held_out_span else None,
            }
            if item.kind == "choice":
                counts = {choice: observed.count(choice) for choice in item.choices}
                entry["choice_counts"] = counts
                entry["unexplored"] = [choice for choice, count in counts.items() if count == 0]
            else:
                entry["min"] = min(observed) if observed else None
                entry["max"] = max(observed) if observed else None
                entry["mean"] = round(sum(observed) / len(observed), 6) if observed else None
                thresholds = [item.low + (item.high - item.low) * (index + 1) / 4 for index in range(3)]
                buckets = [0, 0, 0, 0]
                if item.scale == "log" and item.low > 0.0:
                    log_low, log_high = math.log(item.low), math.log(item.high)
                    thresholds = [math.exp(log_low + (log_high - log_low) * (index + 1) / 4) for index in range(3)]
                for value in observed:
                    if value <= thresholds[0]:
                        buckets[0] += 1
                    elif value <= thresholds[1]:
                        buckets[1] += 1
                    elif value <= thresholds[2]:
                        buckets[2] += 1
                    else:
                        buckets[3] += 1
                entry["buckets"] = buckets
                entry["unexplored_buckets"] = [index for index, count in enumerate(buckets) if count == 0]
            keys[item.key] = entry

        families: dict[str, Any] = {}
        for family, family_keys in self.families().items():
            filled = sum(
                1
                for key in family_keys
                for bucket in keys[key].get("buckets", [])
                if bucket > 0
            )
            total = sum(4 for key in family_keys if keys[key].get("buckets") is not None)
            choices_covered = sum(
                1
                for key in family_keys
                if "unexplored" in keys[key]
                for choice, count in keys[key].get("choice_counts", {}).items()
                if count > 0
            )
            families[family] = {
                "label": _FAMILY_LABELS[family],
                "keys": list(family_keys),
                "buckets_filled": filled,
                "buckets_total": total,
                "choice_values_seen": choices_covered,
                "keys_never_sampled": [
                    key for key in family_keys if keys[key]["samples"] == 0
                ],
            }

        joint: dict[str, Any] = {}
        family_names = sorted(self.families())
        for left_position, left in enumerate(family_names):
            for right in family_names[left_position + 1:]:
                left_key = self.families()[left][0]
                right_key = self.families()[right][0]
                matrix = [[0, 0], [0, 0]]
                left_mid = _midpoint(self.by_key(left_key))
                right_mid = _midpoint(self.by_key(right_key))
                for draw in draws:
                    if left_key not in draw or right_key not in draw:
                        continue
                    if not isinstance(draw[left_key], (int, float)) or not isinstance(draw[right_key], (int, float)):
                        continue
                    row = 0 if draw[left_key] <= left_mid else 1
                    column = 0 if draw[right_key] <= right_mid else 1
                    matrix[row][column] += 1
                joint[f"{left_key}|{right_key}"] = {
                    "matrix": matrix,
                    "unexplored": [
                        [row, column]
                        for row in (0, 1)
                        for column in (0, 1)
                        if matrix[row][column] == 0
                    ],
                }
        return {
            "version": self.version,
            "label": self.label,
            "set_hash": self.payload()["hash"],
            "draws": len(draws),
            "held_out_keys": self.held_out_keys(),
            "keys": keys,
            "families": families,
            "joint": joint,
        }


def _midpoint(item: Distribution) -> float:
    if item.scale == "log" and item.low > 0.0:
        return math.sqrt(item.low * item.high)
    return (item.low + item.high) / 2.0


def _latent(rng: random.Random, condition_class: str, group: str) -> float:
    """One latent factor, conditioned (never a per-parameter independent draw)."""
    shift = 0.0
    scale = 1.0
    if condition_class == "degraded":
        shift = -1.0 if group in ("capability", "budget") else 0.6
    elif condition_class == "rare":
        shift = -1.4 if group in ("capability", "budget") else 1.4
        scale = 1.3
    elif condition_class == "boundary":
        scale = 1.8
    elif condition_class == "noisy":
        scale = 1.1
    return shift + scale * _standard_normal(rng)


def _standard_normal(rng: random.Random) -> float:
    return rng.gauss(0.0, 1.0)


def _uniform(rng: random.Random, condition_class: str) -> float:
    u = rng.random()
    if condition_class == "boundary":
        # Push the draw towards the edges of the declared range.
        return u * 0.15 if u < 0.5 else 1.0 - (1.0 - u) * 0.15
    if condition_class == "rare":
        return u ** 3
    return u


def _noise_scale(condition_class: str) -> float:
    return {"nominal": 1.0, "boundary": 1.2, "noisy": 2.5, "degraded": 1.2, "rare": 1.5}[condition_class]


def condition_classes_for(index: int, *, family: str = "") -> str:
    """Deterministic condition class for a campaign index (balance, not chance)."""
    del family
    cycle = CONDITION_CLASSES
    weights = (0.5, 0.2, 0.15, 0.1, 0.05)
    # A deterministic weighted cycle keeps rare conditions present in every
    # campaign instead of depending on luck.
    position = index % sum(int(weight * 20) for weight in weights)
    running = 0.0
    for name, weight in zip(cycle, weights):
        running += weight * 20
        if position < running:
            return name
    return "nominal"


def summarize_draws(
    distribution_set: DistributionSet,
    draws: Sequence[Mapping[str, Any]],
    *,
    condition_classes: Iterable[str] = (),
) -> dict[str, Any]:
    """Coverage plus a condition-class breakdown for one campaign slice."""
    report = distribution_set.summarize(draws)
    classes = list(condition_classes)
    if classes:
        counts: dict[str, int] = {}
        for name in classes:
            counts[name] = counts.get(name, 0) + 1
        report["condition_classes"] = counts
    return report


def invalid_combinations(distribution_set: DistributionSet) -> list[dict[str, Any]]:
    """The explicit list of combinations the sampler refuses to simulate.

    Every constraint is probed with a deliberately violating draw so the report
    names a real rejection, not a rule that merely exists in a file.
    """
    found: list[dict[str, Any]] = []
    probes: dict[str, dict[str, Any]] = {
        "fleet-at-least-two": {"fleet.size": 1},
        "partitions-fit-the-fleet": {"network.partitions": 3, "fleet.size": 2},
        "faults-fit-the-fleet": {"faults.count": 4, "fleet.size": 2},
        "fault-window-fits-the-episode": {"faults.count": 2, "faults.duration_scale": 0.0},
        "buffer-rate-fits-capacity": {"storage.capacity_bit": 1.0e6, "storage.write_rate_bit_s": 5.0e6},
        "downlink-rate-fits-write-rate": {"storage.write_rate_bit_s": 1.0e5, "storage.read_rate_bit_s": 1.0e7},
        "targets-need-a-deadline": {"tasks.deadline_scale": 0.1},
        "sensor-range-covers-the-orbit": {"sensor.max_range_km": 500.0, "orbit.altitude_km": 900.0},
        "step-fits-the-horizon": {"time.horizon_s": 100.0, "time.step_s": 60.0},
    }
    for constraint in distribution_set.constraints:
        values = probes.get(constraint.name)
        if values is not None and constraint.violated(values):
            found.append({**constraint.to_dict(), "probe": values, "rejected": True})
    return found


__all__ = [
    "RANDOMIZATION_VERSION",
    "CONDITION_CLASSES",
    "GROUPS",
    "FAMILIES",
    "RandomizationError",
    "Distribution",
    "Constraint",
    "DistributionSet",
    "default_distributions",
    "default_constraints",
    "condition_classes_for",
    "summarize_draws",
    "invalid_combinations",
]
