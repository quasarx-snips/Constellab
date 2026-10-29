"""Day 36 acceptance: the randomized scenario pipeline and its provenance."""

from __future__ import annotations

import dataclasses
import json
import random
import unittest

from constellab_core import faults, sensors
from constellab_core.config import load_config
from constellab_core.targets import EventTarget

from constellab_ml.env import EpisodeSpec, SwarmEnvironment
from constellab_ml.randomization import DistributionSet
from constellab_ml.scenarios import (
    SCENARIO_FAMILIES,
    SPLIT_NAMES,
    ScenarioError,
    ScenarioSampler,
    scenario_family,
)

MINUTE = 60.0


def fast_distribution_set() -> DistributionSet:
    """A small, quick set for tests (tiny fleets, short horizons, no faults)."""
    overrides = {
        "fleet.size": dict(low=2, high=4),
        "orbit.altitude_km": dict(low=520.0, high=620.0, held_out_span=(700.0, 760.0)),
        "orbit.planes": dict(low=1, high=2),
        "orbit.inclination_deg": dict(low=45.0, high=70.0),
        "sensor.half_angle_deg": dict(low=25.0, high=45.0),
        "sensor.max_range_km": dict(low=2_000.0, high=4_000.0),
        "sensor.duty_cycle_ratio": dict(low=0.8, high=1.0),
        "sensor.integration_time_s": dict(low=0.4, high=0.8),
        "tasks.target_count": dict(low=1, high=2, held_out_span=(3.0, 3.4)),
        "tasks.deadline_scale": dict(low=0.8, high=1.2),
        "network.partitions": dict(low=0, high=0.5),
        "network.duty_cycle": dict(low=0.9, high=1.0),
        "network.link_range_km": dict(low=8_000.0, high=20_000.0),
        "faults.count": dict(low=0, high=0.5, held_out_span=(1.0, 1.4)),
        "faults.duration_scale": dict(low=0.2, high=0.4),
        "time.horizon_s": dict(low=1_500.0, high=1_800.0),
        "time.step_s": dict(low=45.0, high=75.0),
        "energy.battery_capacity_j": dict(low=1.0e5, high=5.0e5),
    }
    distributions = tuple(
        dataclasses.replace(item, **overrides[item.key]) if item.key in overrides else item
        for item in DistributionSet.default().distributions
    )
    return DistributionSet(
        distributions=distributions,
        constraints=DistributionSet.default().constraints,
        label="unit-fast",
    )


def fast_sampler(**kwargs) -> ScenarioSampler:
    return ScenarioSampler(seed=3, distribution_set=fast_distribution_set(), salt="unit", **kwargs)


class TestScenarioSampler(unittest.TestCase):
    def test_episode_is_deterministic_and_resumable(self) -> None:
        sampler = fast_sampler()
        first = sampler.sample(4)
        second = sampler.sample(4)
        self.assertEqual(first.spec.payload(), second.spec.payload())
        self.assertEqual(first.spec.spec_hash(), second.spec.spec_hash())
        self.assertNotEqual(first.spec.spec_hash(), sampler.sample(5).spec.spec_hash())

    def test_spec_validates_and_builds_an_environment(self) -> None:
        sampler = fast_sampler()
        sample = sampler.sample(0)
        typed = load_config(sample.spec.config)
        self.assertEqual(len(typed.satellites), sample.spec.fleet_size)
        environment = SwarmEnvironment(sample.spec)
        observation, info = environment.reset()
        self.assertEqual(info["fleet_size"], sample.spec.fleet_size)
        self.assertEqual(info["task_count"], len(sample.spec.targets))
        self.assertTrue(observation["mask"]["assign"], "a sampled episode must contain reachable work")

    def test_every_episode_has_a_usable_window(self) -> None:
        sampler = fast_sampler()
        for index in range(4):
            sample = sampler.sample(index)
            typed = load_config(sample.spec.config)
            windows, _ = sensors.opportunity_windows(
                typed,
                [EventTarget.from_dict(item) for item in sample.spec.targets],
                horizon_s=sample.spec.horizon_s,
                step_s=30.0,
                seed=sample.spec.seed,
            )
            self.assertTrue(any(window.usable for window in windows), f"episode {index}")

    def test_payload_round_trips(self) -> None:
        sampler = fast_sampler()
        sample = sampler.sample(1)
        payload = sample.spec.payload()
        restored = EpisodeSpec.from_payload(json.loads(json.dumps(payload)))
        self.assertEqual(restored.spec_hash(), sample.spec.spec_hash())
        self.assertEqual(restored.fault_plan, sample.spec.fault_plan)

    def test_metadata_is_versioned_and_hashed(self) -> None:
        metadata = fast_sampler().metadata()
        self.assertEqual(metadata["scenario_version"], "scenario-sampler-1")
        self.assertEqual(metadata["randomization_version"], "randomization-1")
        self.assertEqual(len(metadata["distribution_hash"]), 16)
        self.assertTrue(metadata["parameter_schema_hash"])
        self.assertEqual(metadata["splits"], [0.7, 0.15, 0.15])

    def test_parameter_tokens_are_attached_and_compatible(self) -> None:
        sample = fast_sampler().sample(0)
        context = sample.spec.parameters
        self.assertEqual(context["version"], "parameter-tokens-1")
        self.assertTrue(context["compatibility"]["compatible"], context["compatibility"])
        self.assertGreater(len(context["tokens"]), 10)
        fields = {token["field"] for token in context["tokens"]}
        self.assertIn("sensor.conical.half_angle_deg", fields)
        self.assertIn("storage.buffer.capacity_bit", fields)
        self.assertTrue(sample.spec.parameter_registry)


class TestSplits(unittest.TestCase):
    def test_splits_are_seed_based_and_reproducible(self) -> None:
        sampler = fast_sampler()
        first = [sampler.split_for(index) for index in range(200)]
        second = [sampler.split_for(index) for index in range(200)]
        self.assertEqual(first, second)
        seen = set(first)
        self.assertTrue(seen.issubset(set(SPLIT_NAMES)))
        self.assertGreater(len(seen), 1)

    def test_explicit_split_is_respected(self) -> None:
        sampler = fast_sampler()
        for name in SPLIT_NAMES:
            sample = sampler.sample(0, split=name)
            self.assertEqual(sample.spec.split, name)
            self.assertTrue(sample.spec.episode_id.startswith(name))

    def test_unknown_split_is_refused(self) -> None:
        with self.assertRaises(ScenarioError):
            fast_sampler().sample(0, split="holdout")

    def test_test_split_uses_held_out_ranges(self) -> None:
        sampler = fast_sampler()
        sample = sampler.sample(0, split="test")
        self.assertGreaterEqual(sample.values["fleet.size"], 2)
        self.assertGreaterEqual(sample.values["orbit.altitude_km"], 700.0)
        self.assertLessEqual(sample.values["orbit.altitude_km"], 760.0)
        self.assertEqual(sample.values["tasks.target_count"], 3)

    def test_train_draws_avoid_the_held_out_spans(self) -> None:
        sampler = fast_sampler()
        for index in range(20):
            sample = sampler.sample(index, split="train")
            self.assertLessEqual(sample.values["orbit.altitude_km"], 620.0)
            self.assertLessEqual(sample.values["tasks.target_count"], 2)


class TestFaultsAndLinkModel(unittest.TestCase):
    def test_partition_plan_cuts_claims_between_groups(self) -> None:
        sampler = fast_sampler()
        values = sampler.distribution_set.sample(random.Random(11), condition_class="nominal")
        values["network.partitions"] = 1
        typed = load_config(sampler.build_config(values, index=0, seed=1))
        plan = sampler.build_faults(values, typed, index=0, seed=1, horizon_s=1_800.0)
        self.assertTrue(plan.faults)
        cut = [fault for fault in plan.faults if fault.kind == "link-cut"]
        self.assertTrue(cut)
        first = cut[0]
        self.assertTrue(first.satellites)
        self.assertTrue(first.peers)
        self.assertEqual(set(first.satellites) & set(first.peers), set())

    def test_injected_fault_plan_is_a_core_plan(self) -> None:
        sampler = fast_sampler()
        values = sampler.distribution_set.sample(random.Random(5), condition_class="nominal")
        values["faults.count"] = 2
        values["network.partitions"] = 0
        typed = load_config(sampler.build_config(values, index=0, seed=5))
        plan = sampler.build_faults(values, typed, index=0, seed=5, horizon_s=1_800.0)
        self.assertEqual(plan.strategy, "random")
        self.assertEqual(len(plan.faults), 2)
        restored = faults.FaultPlan.from_dict(plan.to_dict())
        self.assertEqual(restored.hash(), plan.hash())
        for fault in plan.faults:
            self.assertGreater(fault.end_s, fault.start_s)

    def test_declared_link_model_changes_the_contact_graph(self) -> None:
        sampler = fast_sampler()
        sample = sampler.sample(2)
        connected = SwarmEnvironment(sample.spec)
        connected_observation, _ = connected.reset()
        far_apart = dataclasses.replace(
            sample.spec,
            contact={**sample.spec.contact, "range_km": 1.0},
        )
        parted = SwarmEnvironment(far_apart)
        parted_observation, _ = parted.reset()
        self.assertGreater(
            connected_observation["counts"]["contact_edges"],
            parted_observation["counts"]["contact_edges"],
        )
        self.assertEqual(parted_observation["counts"]["contact_edges"], 0)

    def test_intermittent_links_are_deterministic(self) -> None:
        sampler = fast_sampler()
        sample = sampler.sample(2)
        spec = dataclasses.replace(
            sample.spec,
            contact={**sample.spec.contact, "duty_cycle": 0.5, "period_s": 300.0},
        )
        first = SwarmEnvironment(spec).observation()
        second = SwarmEnvironment(spec).observation()
        self.assertEqual(first["edges"]["contact"], second["edges"]["contact"])


class TestCoverageReport(unittest.TestCase):
    def test_coverage_names_families_conditions_and_splits(self) -> None:
        sampler = fast_sampler()
        samples = [sampler.sample(index) for index in range(12)]
        report = sampler.coverage(samples)
        self.assertEqual(report["scenario_version"], "scenario-sampler-1")
        self.assertEqual(set(report["scenario_families"]), set(SCENARIO_FAMILIES))
        self.assertEqual(sum(report["splits"].values()), len(samples))
        self.assertEqual(sum(report["condition_classes"].values()), len(samples))
        self.assertIn("keys", report)
        self.assertIn("joint", report)

    def test_scenario_family_labels_follow_priority(self) -> None:
        base = {
            "fleet.size": 3,
            "tasks.target_count": 1,
            "network.partitions": 0,
            "network.duty_cycle": 1.0,
            "faults.count": 0,
            "energy.initial_charge_ratio": 1.0,
            "sensor.duty_cycle_ratio": 1.0,
            "compute.duty_cycle_ratio": 1.0,
        }
        self.assertEqual(scenario_family(base), "small-easy")
        self.assertEqual(scenario_family({**base, "fleet.size": 12, "tasks.target_count": 6}), "large-hard")
        self.assertEqual(scenario_family({**base, "sensor.duty_cycle_ratio": 0.2}), "degraded")
        self.assertEqual(scenario_family({**base, "faults.count": 1}), "faulted")
        self.assertEqual(scenario_family({**base, "network.partitions": 1}), "partitioned")
        self.assertEqual(scenario_family({**base, "network.duty_cycle": 0.3}), "partitioned")


class TestSamplerGuards(unittest.TestCase):
    def test_split_shares_must_sum_to_one(self) -> None:
        with self.assertRaises(ScenarioError):
            ScenarioSampler(seed=1, splits=(0.5, 0.2, 0.2))

    def test_zero_share_is_refused(self) -> None:
        with self.assertRaises(ScenarioError):
            ScenarioSampler(seed=1, splits=(1.0, 0.0, 0.0))

    def test_wrong_number_of_shares_is_refused(self) -> None:
        with self.assertRaises(ScenarioError):
            ScenarioSampler(seed=1, splits=(0.5, 0.5))

    def test_unreachable_scheduling_is_reported(self) -> None:
        # A sensor that can never look at the ground produces no schedule at
        # all; the sampler must say so instead of emitting an empty episode.
        distributions = tuple(
            dataclasses.replace(item, low=0.1, high=0.2)
            if item.key == "sensor.half_angle_deg" else item
            for item in fast_distribution_set().distributions
        )
        custom = DistributionSet(
            distributions=distributions,
            constraints=fast_distribution_set().constraints,
            label="unit-blind",
        )
        sampler = ScenarioSampler(seed=9, distribution_set=custom)
        with self.assertRaises(ScenarioError):
            sampler.sample(0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
