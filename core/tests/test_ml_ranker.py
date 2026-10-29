"""Day 55: set features, family split, ranker training/calibration and inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from constellab_ml.adversarial import MutationEngine, run_search
from constellab_ml.dataset import DatasetConfig, generate_campaign, EpisodeDataset
from constellab_ml.ranker import (
    CONTEXT_FEATURES,
    FAULT_FEATURES,
    RankerConfig,
    build_dataset,
    encode_spec,
    load_ranker,
    train_ranker,
)

from tests.test_ml_scenarios import fast_distribution_set


class RankerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp())
        cls.campaign = cls.root / "campaign"
        generate_campaign(
            cls.campaign,
            DatasetConfig(campaign_id="ranker", seed=55),
            12,
            distribution_set=fast_distribution_set(),
        )
        cls.base = EpisodeDataset(cls.campaign).episodes("train")[0]
        from constellab_ml.env import EpisodeSpec

        cls.spec = EpisodeSpec.from_payload(cls.base["spec"])
        cls.dataset_out = cls.root / "search" / "records.jsonl"
        run_search(
            {
                "dataset_root": str(cls.campaign),
                "budget": 6,
                "seed": 55,
                "mode": "random",
                "dataset_out": str(cls.dataset_out),
            },
            cls.root / "search" / "search.json",
        )

    def test_feature_dimensions_and_mask(self) -> None:
        context, tokens, mask = encode_spec(self.spec, operator="shift-fault")
        self.assertEqual(len(context), len(CONTEXT_FEATURES))
        self.assertTrue(tokens)
        self.assertEqual(len(tokens[0]), len(FAULT_FEATURES))
        self.assertEqual(len(mask), len(tokens))

    def test_dataset_builds_aligned_tensors(self) -> None:
        records = []
        import json

        for line in self.dataset_out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        dataset = build_dataset(records)
        self.assertGreater(len(dataset), 0)
        contexts, tokens, masks, labels = dataset.tensors()
        self.assertEqual(contexts.shape[0], len(dataset))
        self.assertEqual(tokens.shape[0], len(dataset))
        self.assertEqual(masks.shape[0], len(dataset))
        self.assertEqual(labels.shape[0], len(dataset))

    def test_training_writes_manifest_and_checkpoint(self) -> None:
        target = self.root / "ranker-run"
        manifest = train_ranker(
            RankerConfig(
                dataset=str(self.dataset_out),
                hidden=8,
                epochs=3,
                batch_size=4,
                seed=2,
            ),
            target,
        )
        self.assertTrue((target / "ranker.pt").exists())
        self.assertTrue((target / "ranker-manifest.json").exists())
        self.assertIn("validation_auc", manifest["metrics"])
        self.assertIn("calibration", manifest["metrics"])
        self.assertIn("split_note", manifest)
        predictor = load_ranker(target / "ranker.pt")
        engine = MutationEngine()
        mutations = [engine.mutate(self.spec, seed=900 + index) for index in range(2)]
        scores = predictor(mutations)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

    def test_guided_search_can_use_a_ranker_checkpoint(self) -> None:
        ranker_run = self.root / "ranker-guided"
        train_ranker(
            RankerConfig(dataset=str(self.dataset_out), hidden=8, epochs=2, batch_size=4, seed=3),
            ranker_run,
        )
        predictor = load_ranker(ranker_run / "ranker.pt")
        payload = run_search(
            {
                "dataset_root": str(self.campaign),
                "budget": 2,
                "seed": 56,
                "mode": "guided",
                "exploration": 1,
            },
            self.root / "guided.json",
            ranker=predictor,
        )
        self.assertEqual(payload["mode"], "guided")
        self.assertLessEqual(payload["summary"]["records"], 2)


if __name__ == "__main__":
    unittest.main()
