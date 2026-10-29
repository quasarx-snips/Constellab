"""Day 44 acceptance: large randomized campaigns with explicit coverage reports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from constellab_ml.dataset import DatasetConfig, EpisodeDataset, generate_campaign
from constellab_ml.randomization import default_constraints, invalid_combinations
from constellab_ml.scenarios import ScenarioSampler

from tests.test_ml_scenarios import fast_distribution_set


class CampaignCoverageTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        cls.root = cls.workspace / "campaign"
        cls.config = DatasetConfig(campaign_id="coverage-campaign", seed=11, salt="coverage")
        generate_campaign(cls.root, cls.config, 24, distribution_set=fast_distribution_set())
        cls.dataset = EpisodeDataset(cls.root)
        cls.manifest = cls.dataset.manifest

    def test_every_parameter_family_and_joint_slice_is_reported(self) -> None:
        coverage = self.manifest["coverage"]
        expected = set(fast_distribution_set().families())
        self.assertEqual(set(coverage["families"]), expected)
        for family, entry in coverage["families"].items():
            self.assertEqual(entry["keys"], list(fast_distribution_set().families()[family]))
            self.assertIn("keys_never_sampled", entry)
            self.assertGreaterEqual(entry["buckets_total"], 0)
        joint = coverage["joint"]
        self.assertTrue(joint)
        numeric_families = sorted(fast_distribution_set().families())
        for left_position, left in enumerate(numeric_families):
            for right in numeric_families[left_position + 1:]:
                key = f"{fast_distribution_set().families()[left][0]}|{fast_distribution_set().families()[right][0]}"
                self.assertIn(key, joint)
                matrix = joint[key]["matrix"]
                self.assertEqual(len(matrix), 2)
                self.assertTrue(all(len(row) == 2 for row in matrix))
                self.assertEqual(sum(sum(row) for row in matrix), coverage["draws"])

    def test_unexplored_regions_are_named(self) -> None:
        coverage = self.manifest["coverage"]
        unexplored = {
            key: entry["unexplored_buckets"]
            for key, entry in coverage["keys"].items()
            if entry.get("unexplored_buckets")
        }
        self.assertIsInstance(unexplored, dict)
        for key, entry in coverage["keys"].items():
            self.assertEqual(entry["samples"], self.dataset.manifest["total_episodes"])
            if "buckets" in entry:
                self.assertEqual(len(entry["buckets"]), 4)
                self.assertEqual(sum(entry["buckets"]), entry["samples"])
            if entry.get("unexplored_buckets"):
                self.assertIn(key, coverage["families"][entry["family"]]["keys"])
            if "choice_counts" in entry:
                self.assertEqual(sum(entry["choice_counts"].values()), entry["samples"])

    def test_invalid_combinations_are_explicit_probed_rejections(self) -> None:
        combinations = self.manifest["coverage"]["invalid_combinations"]
        self.assertTrue(combinations)
        names = [entry["name"] for entry in combinations]
        self.assertEqual(len(names), len(set(names)))
        for entry in combinations:
            self.assertTrue(entry["rejected"])
            self.assertTrue(entry["probe"])
            self.assertTrue(entry["requirement"])
        # Every declared constraint is either probed-and-rejected or does not
        # apply to the shipped default set.
        self.assertLessEqual(
            len(combinations),
            len(fast_distribution_set().constraints),
        )

    def test_sampled_episodes_never_violate_a_constraint(self) -> None:
        constraints = default_constraints()
        sampler = ScenarioSampler(
            seed=self.config.seed,
            distribution_set=fast_distribution_set(),
            salt=self.config.salt,
        )
        for index in range(12):
            values = sampler.sample(index).values
            violations = [
                constraint.name
                for constraint in constraints
                if constraint.violated(values)
            ]
            self.assertEqual(violations, [], f"episode {index} violated {violations}")
        # The sampler's own report is the same rule set, not a second copy.
        self.assertEqual(
            [entry["name"] for entry in invalid_combinations(fast_distribution_set())],
            [entry["name"] for entry in self.manifest["coverage"]["invalid_combinations"]],
        )

    def test_dataset_slices_are_labelled_by_scenario_family_and_split(self) -> None:
        coverage = self.manifest["coverage"]
        self.assertEqual(set(coverage["scenario_families"]), set(coverage["scenario_families"]))
        self.assertEqual(sum(coverage["scenario_families"].values()), self.manifest["total_episodes"])
        self.assertEqual(sum(coverage["splits"].values()), self.manifest["total_episodes"])
        self.assertEqual(sum(coverage["splits"].values()), sum(self.manifest["counts"].values()))
        self.assertEqual(sum(coverage["condition_classes"].values()), self.manifest["total_episodes"])
        self.assertIn("train", coverage["splits"])
        self.assertIn("validation", coverage["splits"])
        self.assertIn("test", coverage["splits"])


if __name__ == "__main__":
    unittest.main()
