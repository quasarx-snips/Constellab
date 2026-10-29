"""Day 44 acceptance: versioned, correlated, conditioned domain randomization."""

from __future__ import annotations

import random
import statistics
import unittest

from constellab_ml.randomization import (
    CONDITION_CLASSES,
    FAMILIES,
    Distribution,
    DistributionSet,
    RandomizationError,
    condition_classes_for,
    invalid_combinations,
)


def pearson(left, right) -> float:
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = (
        sum((a - mean_left) ** 2 for a in left) ** 0.5
        * sum((b - mean_right) ** 2 for b in right) ** 0.5
    )
    return numerator / denominator if denominator else 0.0


class TestDistributionSet(unittest.TestCase):
    def setUp(self) -> None:
        self.distributions = DistributionSet.default()

    def test_payload_is_versioned_hashed_and_json_safe(self) -> None:
        import json

        payload = self.distributions.payload()
        self.assertEqual(payload["version"], "randomization-1")
        self.assertEqual(len(payload["hash"]), 16)
        self.assertTrue(json.dumps(payload))
        self.assertEqual(set(payload["families"]), set(FAMILIES))
        self.assertEqual(payload["condition_classes"], list(CONDITION_CLASSES))

    def test_every_declared_family_is_sampled(self) -> None:
        drawn = set()
        for item in self.distributions.distributions:
            drawn.add(item.family)
        self.assertEqual(drawn, set(FAMILIES))

    def test_unknown_group_is_rejected(self) -> None:
        with self.assertRaises(RandomizationError):
            Distribution(key="orbit.altitude_km", family="orbit", group="magic", group_weight=0.5)

    def test_group_with_zero_weight_is_rejected(self) -> None:
        with self.assertRaises(RandomizationError):
            DistributionSet(
                distributions=(Distribution("orbit.altitude_km", "orbit", group="capability", group_weight=0.0),),
                constraints=(),
            )

    def test_choice_weights_must_match_choices(self) -> None:
        with self.assertRaises(RandomizationError):
            Distribution(
                key="tasks.kind_mix", family="tasks", kind="choice",
                choices=("a", "b"), weights=(1.0,),
            )


class TestSamplingDeterminism(unittest.TestCase):
    def test_same_seed_same_draw(self) -> None:
        distributions = DistributionSet.default()
        first = distributions.sample(random.Random(7), condition_class="nominal")
        second = distributions.sample(random.Random(7), condition_class="nominal")
        self.assertEqual(first, second)

    def test_different_seed_different_draw(self) -> None:
        distributions = DistributionSet.default()
        first = distributions.sample(random.Random(7), condition_class="nominal")
        second = distributions.sample(random.Random(8), condition_class="nominal")
        self.assertNotEqual(first, second)

    def test_draws_are_exact_json_values(self) -> None:
        # 32-bit-safe integers and finite floats: a campaign file must reproduce
        # the same draw when it is read back.
        import json

        distributions = DistributionSet.default()
        draw = distributions.sample(random.Random(3), condition_class="rare")
        self.assertEqual(draw, json.loads(json.dumps(draw)))
        for value in draw.values():
            if isinstance(value, int) and not isinstance(value, bool):
                self.assertLess(abs(value), 2**31)

    def test_condition_classes_are_all_used_by_a_campaign(self) -> None:
        names = {condition_classes_for(index) for index in range(200)}
        self.assertEqual(names, set(CONDITION_CLASSES))


class TestCorrelation(unittest.TestCase):
    def setUp(self) -> None:
        self.distributions = DistributionSet.default()
        self.draws = [
            self.distributions.sample(random.Random(seed), condition_class="nominal")
            for seed in range(120)
        ]

    def test_positive_weights_move_together(self) -> None:
        half_angle = [draw["sensor.half_angle_deg"] for draw in self.draws]
        max_range = [draw["sensor.max_range_km"] for draw in self.draws]
        self.assertGreater(pearson(half_angle, max_range), 0.15)

    def test_negative_weight_moves_against_the_group(self) -> None:
        half_angle = [draw["sensor.half_angle_deg"] for draw in self.draws]
        link_range = [draw["network.link_range_km"] for draw in self.draws]
        self.assertLess(pearson(half_angle, link_range), -0.15)

    def test_groups_are_not_independent_column_noise(self) -> None:
        # A capability-rich fleet is also a well-budgeted one: the two latent
        # groups are still drawn independently, so the *within-group* ordering
        # must be much stronger than the across-group one.
        half_angle = [draw["sensor.half_angle_deg"] for draw in self.draws]
        battery = [draw["energy.battery_capacity_j"] for draw in self.draws]
        self.assertLess(abs(pearson(half_angle, battery)), 0.35)


class TestConditionClasses(unittest.TestCase):
    def setUp(self) -> None:
        self.distributions = DistributionSet.default()

    def _draws(self, condition: str, count: int = 90) -> list[dict]:
        return [
            self.distributions.sample(random.Random(1000 + seed), condition_class=condition)
            for seed in range(count)
        ]

    def test_degraded_is_worse_than_nominal(self) -> None:
        nominal = self._draws("nominal")
        degraded = self._draws("degraded")
        self.assertLess(
            statistics.fmean(draw["energy.battery_capacity_j"] for draw in degraded),
            statistics.fmean(draw["energy.battery_capacity_j"] for draw in nominal),
        )
        self.assertLess(
            statistics.fmean(draw["sensor.half_angle_deg"] for draw in degraded),
            statistics.fmean(draw["sensor.half_angle_deg"] for draw in nominal),
        )

    def test_boundary_prefers_the_edges(self) -> None:
        item = self.distributions.by_key("orbit.inclination_deg")
        midpoint = (item.low + item.high) / 2.0
        span = item.high - item.low

        def edge_distance(condition: str) -> float:
            draws = self._draws(condition, 60)
            return statistics.fmean(
                abs(draw["orbit.inclination_deg"] - midpoint) / span for draw in draws
            )

        self.assertGreater(edge_distance("boundary"), edge_distance("nominal"))

    def test_rare_pushes_the_harsh_group(self) -> None:
        nominal = self._draws("nominal")
        rare = self._draws("rare")
        self.assertGreater(
            statistics.fmean(draw["faults.count"] for draw in rare),
            statistics.fmean(draw["faults.count"] for draw in nominal),
        )

    def test_every_condition_class_yields_valid_draws(self) -> None:
        for condition in CONDITION_CLASSES:
            draws = self._draws(condition, 30)
            for draw in draws:
                self.assertEqual(self.distributions.violations(draw), [])

    def test_unknown_condition_class_is_refused(self) -> None:
        with self.assertRaises(RandomizationError):
            self.distributions.sample(random.Random(1), condition_class="impossible")


class TestHeldOutRanges(unittest.TestCase):
    def setUp(self) -> None:
        self.distributions = DistributionSet.default()

    def test_training_draws_stay_inside_the_training_range(self) -> None:
        draws = [
            self.distributions.sample(random.Random(seed), condition_class="nominal", held_out=False)
            for seed in range(40)
        ]
        for key in self.distributions.held_out_keys():
            item = self.distributions.by_key(key)
            low, high = item.held_out_span
            for draw in draws:
                self.assertTrue(item.low <= draw[key] <= item.high, key)
                self.assertFalse(low <= draw[key] <= high and item.low <= draw[key] <= item.high and item.high < low, key)

    def test_held_out_draws_use_the_reserved_span(self) -> None:
        draws = [
            self.distributions.sample(random.Random(seed), condition_class="nominal", held_out=True)
            for seed in range(40)
        ]
        for key in self.distributions.held_out_keys():
            item = self.distributions.by_key(key)
            low, high = item.held_out_span
            for draw in draws:
                self.assertTrue(low <= draw[key] <= high, f"{key}: {draw[key]} not in ({low}, {high})")

    def test_held_out_span_must_be_increasing(self) -> None:
        with self.assertRaises(RandomizationError):
            Distribution("orbit.altitude_km", "orbit", held_out_span=(900.0, 500.0))


class TestConstraints(unittest.TestCase):
    def setUp(self) -> None:
        self.distributions = DistributionSet.default()

    def test_every_constraint_has_a_named_rejection(self) -> None:
        rejected = invalid_combinations(self.distributions)
        names = {entry["name"] for entry in rejected}
        declared = {constraint.name for constraint in self.distributions.constraints}
        self.assertEqual(names, declared)

    def test_named_violation_reports_the_offending_values(self) -> None:
        found = self.distributions.violations({
            "fleet.size": 2,
            "network.partitions": 3,
            "faults.count": 0,
            "faults.duration_scale": 0.3,
            "storage.capacity_bit": 1.0e9,
            "storage.write_rate_bit_s": 1.0e6,
            "storage.read_rate_bit_s": 1.0e6,
            "tasks.deadline_scale": 1.0,
            "time.horizon_s": 3_600.0,
            "time.step_s": 60.0,
            "sensor.max_range_km": 2_000.0,
            "orbit.altitude_km": 600.0,
        })
        self.assertEqual([entry["name"] for entry in found], ["partitions-fit-the-fleet"])
        self.assertEqual(found[0]["values"], {"network.partitions": 3, "fleet.size": 2})

    def test_invalid_draw_is_retried_not_accepted(self) -> None:
        # A tiny custom set whose first attempt is forced to violate and whose
        # retry succeeds: the sampler must not return the invalid combination.
        defs = (
            Distribution("fleet.size", "fleet", kind="int", low=1, high=3, noise=0.0),
            Distribution("network.partitions", "network", kind="int", low=0, high=5, noise=0.0),
        )
        constraints = (
            *[c for c in self.distributions.constraints if c.name == "partitions-fit-the-fleet"],
        )
        custom = DistributionSet(distributions=defs, constraints=constraints)
        for seed in range(40):
            draw = custom.sample(random.Random(seed), max_attempts=20)
            self.assertEqual(custom.violations(draw), [])
            self.assertLessEqual(draw["network.partitions"], draw["fleet.size"] - 1)

    def test_impossible_set_reports_rather_than_accepts(self) -> None:
        from constellab_ml.randomization import Constraint

        defs = (Distribution("fleet.size", "fleet", kind="int", low=1, high=3, noise=0.0),)
        constraints = (
            Constraint(
                name="unit-needs-five-satellites",
                applies=("fleet.size",),
                rule="fleet.size >= 5",
                description="a deliberately unsatisfiable unit-test requirement",
            ),
        )
        custom = DistributionSet(distributions=defs, constraints=constraints)
        with self.assertRaises(RandomizationError) as caught:
            custom.sample(random.Random(1), max_attempts=5)
        self.assertIn("unit-needs-five-satellites", str(caught.exception))

    def test_constraint_rule_must_mention_its_keys(self) -> None:
        from constellab_ml.randomization import Constraint

        with self.assertRaises(RandomizationError):
            Constraint(name="broken", applies=("fleet.size",), rule="network.partitions >= 0", description="x")


class TestCoverage(unittest.TestCase):
    def test_report_covers_keys_families_and_joint_slices(self) -> None:
        distributions = DistributionSet.default()
        draws = [
            distributions.sample(random.Random(seed), condition_class="nominal")
            for seed in range(30)
        ]
        report = distributions.summarize(draws)
        self.assertEqual(report["draws"], 30)
        self.assertEqual(set(report["keys"]), set(distributions.keys))
        for family, entry in report["families"].items():
            self.assertIn(family, FAMILIES)
            self.assertTrue(entry["keys"])
        self.assertTrue(report["joint"])
        for entry in report["joint"].values():
            self.assertEqual(len(entry["matrix"]), 2)
        self.assertEqual(report["held_out_keys"], distributions.held_out_keys())

    def test_thin_campaign_reports_unexplored_regions(self) -> None:
        distributions = DistributionSet.default()
        draws = [distributions.sample(random.Random(1), condition_class="nominal")]
        report = distributions.summarize(draws)
        unexplored = [
            key for key, entry in report["keys"].items() if entry.get("unexplored_buckets")
        ]
        self.assertTrue(unexplored)
        self.assertTrue(any(entry["unexplored"] for entry in report["joint"].values()))

    def test_choice_coverage_lists_unseen_values(self) -> None:
        distributions = DistributionSet.default()
        draws = [
            distributions.sample(random.Random(seed), condition_class="nominal")
            for seed in range(4)
        ]
        report = distributions.summarize(draws)
        entry = report["keys"]["tasks.kind_mix"]
        self.assertEqual(set(entry["choice_counts"]), set(distributions.by_key("tasks.kind_mix").choices))
        self.assertIn("unexplored", entry)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
