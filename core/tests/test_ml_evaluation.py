"""Days 46/52: suite construction, slice metrics, intervals and benchmark JSON."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from constellab_ml.curriculum import FirstCandidatePolicy, WaitPolicy
from constellab_ml.dataset import DatasetConfig, generate_campaign
from constellab_ml.evaluation import (
    bootstrap_ci,
    build_generalization_suite,
    evaluate_policy_records,
    run_generalization,
    run_policy_benchmark,
)

from tests.test_ml_scenarios import fast_distribution_set


class EvaluationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp())
        cls.campaign = cls.root / "campaign"
        generate_campaign(
            cls.campaign,
            DatasetConfig(campaign_id="evaluation", seed=9),
            24,
            distribution_set=fast_distribution_set(),
        )

    def test_suite_labels_test_episodes_and_unseen_fleets(self) -> None:
        suite = build_generalization_suite(self.campaign, limit=4)
        self.assertEqual(suite["version"], "generalization-suite-1")
        self.assertTrue(suite["records"])
        self.assertIn("all", suite["slice_counts"])
        for record in suite["records"]:
            self.assertIn("_slices", record)
            self.assertTrue(
                "fleet:seen" in record["_slices"] or "fleet:unseen" in record["_slices"]
            )
        self.assertIn("training_fleets", suite["inventory"])

    def test_wait_policy_has_zero_coverage_and_finite_latency(self) -> None:
        suite = build_generalization_suite(self.campaign, limit=2)
        report = evaluate_policy_records(WaitPolicy(), suite["records"])
        self.assertEqual(report["aggregate"]["coverage"]["mean"], 0.0)
        self.assertEqual(report["aggregate"]["observed"], 0)
        self.assertIsNotNone(report["aggregate"]["latency_ms"]["mean"])
        self.assertEqual(report["episodes"][0]["slices"][0], "all")

    def test_first_feasible_is_measured_on_the_same_episodes(self) -> None:
        suite = build_generalization_suite(self.campaign, limit=3)
        wait = evaluate_policy_records(WaitPolicy(), suite["records"])
        first = evaluate_policy_records(FirstCandidatePolicy(), suite["records"])
        self.assertEqual(wait["aggregate"]["episodes"], first["aggregate"]["episodes"])
        self.assertGreaterEqual(first["aggregate"]["coverage"]["mean"], 0.0)
        self.assertIn("family", " ".join(first["slices"]))

    def test_bootstrap_interval_is_deterministic(self) -> None:
        first = bootstrap_ci([0.1, 0.5, 0.9], seed=4)
        second = bootstrap_ci([0.1, 0.5, 0.9], seed=4)
        self.assertEqual(first, second)
        self.assertLessEqual(first["low"], first["mean"])
        self.assertGreaterEqual(first["high"], first["mean"])

    def test_shielded_evaluation_records_the_shield_summary(self) -> None:
        suite = build_generalization_suite(self.campaign, limit=2)
        report = evaluate_policy_records(
            FirstCandidatePolicy(), suite["records"], use_shield=True,
        )
        self.assertIsNotNone(report["shield"])
        self.assertEqual(report["shield"]["version"], "safety-shield-1")
        self.assertIn("accepted", report["shield"])

    def test_generalization_report_is_written(self) -> None:
        target = self.root / "generalization.json"
        payload = run_generalization(
            {"dataset_root": str(self.campaign), "limit": 2, "seed": 46},
            target,
            policies={"wait": WaitPolicy(), "first": FirstCandidatePolicy()},
        )
        self.assertTrue(target.exists())
        loaded = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(loaded["version"], "generalization-suite-1")
        self.assertEqual(set(loaded["policies"]), {"wait", "first"})
        self.assertIn("selection_note", payload)

    def test_policy_benchmark_compares_pairs(self) -> None:
        target = self.root / "benchmark.json"
        payload = run_policy_benchmark(
            {"dataset_root": str(self.campaign), "limit": 2, "seed": 52},
            target,
            policies={"wait": WaitPolicy(), "first": FirstCandidatePolicy()},
        )
        self.assertTrue(target.exists())
        self.assertEqual(payload["version"], "policy-benchmark-1")
        self.assertEqual(payload["suite"]["episodes"], 2)
        for report in payload["policies"].values():
            self.assertIn("aggregate", report)
            self.assertIn("episodes", report)


if __name__ == "__main__":
    unittest.main()
