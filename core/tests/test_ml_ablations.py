"""Day 50: ablation variants, observation transforms and comparison report."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from constellab_ml.ablations import (
    AblationVariant,
    ObservationTransformPolicy,
    default_variants,
    run_ablations,
)
from constellab_ml.curriculum import FirstCandidatePolicy
from constellab_ml.dataset import DatasetConfig, generate_campaign

from tests.test_ml_scenarios import fast_distribution_set


class AblationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp())
        cls.campaign = cls.root / "campaign"
        generate_campaign(
            cls.campaign,
            DatasetConfig(campaign_id="ablations", seed=50),
            16,
            distribution_set=fast_distribution_set(),
        )

    def test_no_communication_and_no_context_transforms(self) -> None:
        observation = {
            "satellites": [{"features": [0.0] * 14}],
            "tasks": [{"features": [0.0] * 11}],
            "edges": {"assign": [], "contact": [[0, 0]]},
            "context": {"tokens": [{"field": "x"}]},
        }
        communication = ObservationTransformPolicy(FirstCandidatePolicy(), "no-communication")
        transformed = communication.transform_observation(observation)
        self.assertEqual(transformed["edges"]["contact"], [])
        self.assertEqual(transformed["satellites"][0]["features"][11], 0.0)
        context = ObservationTransformPolicy(FirstCandidatePolicy(), "no-parameter-context")
        self.assertEqual(context.transform_observation(observation)["context"], {})
        self.assertEqual(observation["edges"]["contact"], [[0, 0]])

    def test_default_variants_keep_shield_and_diagnostic_labels(self) -> None:
        variants = default_variants("ppo.pt", imitation="imitation.pt")
        names = {variant.name for variant in variants}
        self.assertIn("full", names)
        self.assertIn("no-shield", names)
        self.assertIn("imitation-only", names)
        self.assertTrue(next(variant for variant in variants if variant.name == "full").shield)
        self.assertTrue(next(variant for variant in variants if variant.name == "no-shield").diagnostic)

    def test_invalid_variant_and_transform_are_refused(self) -> None:
        with self.assertRaises(Exception):
            AblationVariant("bad", kind="other")
        with self.assertRaises(Exception):
            AblationVariant("bad", transform="unknown")

    def test_run_ablations_writes_a_table(self) -> None:
        target = self.root / "ablations.json"
        payload = run_ablations(
            {
                "dataset_root": str(self.campaign),
                "limit": 2,
                "seed": 50,
                "variants": [
                    {"name": "full", "checkpoint": "", "shield": True},
                    {"name": "no-shield", "checkpoint": "", "shield": False, "diagnostic": True},
                ],
            },
            target,
        )
        self.assertTrue(target.exists())
        self.assertEqual(payload["version"], "policy-ablations-1")
        self.assertEqual(set(payload["variants"]), {"full", "no-shield"})
        for row in payload["variants"].values():
            self.assertIn("metrics", row)
            self.assertIn("mean_coverage", row["metrics"])
            self.assertIn("slices", row)


if __name__ == "__main__":
    unittest.main()
