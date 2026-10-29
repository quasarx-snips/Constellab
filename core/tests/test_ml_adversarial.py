"""Days 53/54: mutations, invariant failures, objective and labeled search data."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from constellab_ml.adversarial import (
    MUTATION_OPERATORS,
    MutationEngine,
    SearchDataset,
    SearchObjective,
    SearchRecord,
    evaluate_scenario,
    run_search,
)
from constellab_ml.dataset import DatasetConfig, generate_campaign

from tests.test_ml_scenarios import fast_distribution_set


class AdversarialTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp())
        cls.campaign = cls.root / "campaign"
        generate_campaign(
            cls.campaign,
            DatasetConfig(campaign_id="adversarial", seed=53),
            12,
            distribution_set=fast_distribution_set(),
        )
        from constellab_ml.dataset import EpisodeDataset

        cls.base = EpisodeDataset(cls.campaign).episodes("train")[0]
        from constellab_ml.env import EpisodeSpec

        cls.spec = EpisodeSpec.from_payload(cls.base["spec"])

    def test_every_operator_produces_a_valid_child(self) -> None:
        engine = MutationEngine()
        for index, operator in enumerate(MUTATION_OPERATORS):
            mutation = engine.mutate(self.spec, seed=100 + index, operator=operator)
            self.assertEqual(mutation.parent_id, self.spec.episode_id)
            self.assertEqual(mutation.operator, operator)
            self.assertNotEqual(mutation.spec.spec_hash(), self.spec.spec_hash())
            self.assertIn("faults", mutation.spec.fault_plan)

    def test_mutation_of_a_single_satellite_fleet_is_refused(self) -> None:
        from constellab_ml.env import EpisodeSpec

        single = EpisodeSpec.from_payload({
            **self.spec.payload(),
            "config": {
                **self.spec.payload()["config"],
                "satellites": self.spec.payload()["config"]["satellites"][:1],
            },
        })
        with self.assertRaises(Exception):
            MutationEngine().mutate(single, seed=1, operator="shift-fault")

    def test_scenario_evaluation_reports_invariants_and_metrics(self) -> None:
        result = evaluate_scenario(self.spec)
        self.assertIn("metrics", result)
        self.assertIn("findings", result)
        self.assertIn("failure", result)
        self.assertIn("trace_hash", result)
        self.assertIn("signature", result["findings"])

    def test_objective_components_are_visible(self) -> None:
        objective = SearchObjective()
        score, components = objective.score(
            failure=True, signature="duplicate-task-claim", new_signature=True,
            errors=2, executions=3,
        )
        self.assertGreater(score, 0)
        self.assertEqual(components["new_failure"], 5.0)
        self.assertEqual(components["distinct_failure"], 2.0)
        self.assertEqual(components["severity"], 2.0)

    def test_dataset_deduplicates_by_spec_hash(self) -> None:
        mutation = MutationEngine().mutate(self.spec, seed=5, operator="extend-fault")
        record = SearchRecord(mutation=mutation, valid=True, result={"failure": False})
        dataset = SearchDataset()
        self.assertTrue(dataset.add(record))
        self.assertFalse(dataset.add(record))
        self.assertEqual(dataset.duplicates, 1)
        self.assertEqual(dataset.summary()["records"], 1)

    def test_random_campaign_keeps_failures_and_non_failures(self) -> None:
        target = self.root / "search.json"
        payload = run_search(
            {
                "dataset_root": str(self.campaign),
                "base_index": 0,
                "budget": 3,
                "seed": 53,
                "mode": "random",
            },
            target,
        )
        self.assertTrue(target.exists())
        self.assertEqual(payload["version"], "adversarial-search-1")
        self.assertLessEqual(payload["summary"]["records"], 3)
        self.assertIn("class_balance", payload["summary"])
        self.assertIn("objective", payload)

    def test_guided_campaign_uses_ranker_scores(self) -> None:
        calls = {"count": 0}

        def ranker(mutations):
            calls["count"] = len(mutations)
            return list(range(len(mutations)))

        payload = run_search(
            {
                "dataset_root": str(self.campaign),
                "base_index": 0,
                "budget": 2,
                "seed": 53,
                "mode": "guided",
                "exploration": 1,
            },
            self.root / "guided.json",
            ranker=ranker,
        )
        self.assertEqual(payload["mode"], "guided")
        self.assertGreater(calls["count"], 0)
        self.assertLessEqual(payload["summary"]["records"], 2)


if __name__ == "__main__":
    unittest.main()
