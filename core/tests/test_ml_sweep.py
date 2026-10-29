"""Day 43 acceptance: bounded, resumable sweeps with frozen selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from constellab_ml.dataset import DatasetConfig, EpisodeDataset, generate_campaign
from constellab_ml.sweep import (
    SweepConfig,
    SweepError,
    SweepRunner,
    run_sweep,
    sweep_report,
)
from constellab_ml.training import Trainer

from tests.test_ml_scenarios import fast_distribution_set


def campaign(root: Path, *, count: int = 12, seed: int = 3) -> EpisodeDataset:
    generate_campaign(
        root,
        DatasetConfig(campaign_id="sweep-campaign", seed=seed),
        count,
        distribution_set=fast_distribution_set(),
    )
    return EpisodeDataset(root)


class SweepTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        cls.dataset = campaign(cls.workspace / "campaign")

    def config(self, **overrides) -> SweepConfig:
        base = dict(
            sweep_id="unit-sweep",
            dataset_root=str(self.workspace / "campaign"),
            families=("mlp", "gnn"),
            hidden_sizes=(8,),
            learning_rates=(3e-3,),
            abstain_weights=(1.0,),
            weight_decays=(1e-4,),
            budget=2,
            epochs=1,
            batches_per_epoch=2,
            graphs_per_batch=4,
            steps_per_chunk=2,
            evaluate_episodes=1,
            max_sequence_samples=60,
            early_stopping_patience=0,
        )
        base.update(overrides)
        return SweepConfig(**base)

    def test_plan_is_bounded_and_covers_every_family(self) -> None:
        config = self.config(families=("mlp", "deepsets", "gnn", "gnn_gru", "gnn_attention"), budget=4)
        runner = SweepRunner(config, self.workspace / "plan")
        grid = runner.grid()
        plan = runner.plan()
        self.assertEqual(len(grid), 5)
        self.assertEqual(len(plan), 4)
        self.assertEqual(
            {entry["model"]["model"] for entry in plan},
            {"mlp", "deepsets", "gnn", "gnn_gru"},
        )

    def test_sweep_records_runs_and_freezes_a_selection(self) -> None:
        report = run_sweep(self.config(), self.workspace / "sweep-a")
        path = self.workspace / "sweep-a" / "sweep.json"
        self.assertTrue(path.exists())
        self.assertEqual(sweep_report(self.workspace / "sweep-a")["sweep_id"], "unit-sweep")
        self.assertEqual(report["planned_runs"], 2)
        self.assertEqual(report["completed"] + report["failed"], 2)
        self.assertEqual(len(report["runs"]), 2)
        self.assertIn("test_reporting", report)
        self.assertIn("never saw it", report["test_reporting"])
        selection = report["selection"]
        self.assertIsNotNone(selection)
        self.assertTrue(selection["frozen"])
        self.assertEqual(selection["metric"], "validation_loss")
        self.assertTrue(Path(selection["checkpoint"]).exists())
        # Each run maps back to its full training configuration.
        manifest = json.loads(
            (Path(report["runs"][0]["directory"]) / "training-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["run_id"], "unit-sweep-000")
        self.assertEqual(
            manifest["config"]["model"]["model"], report["runs"][0]["combination"]["model"]["model"]
        )
        self.assertEqual(
            manifest["config"]["learning_rate"], report["runs"][0]["combination"]["learning_rate"]
        )

    def test_rerun_resumes_finished_runs(self) -> None:
        first = run_sweep(self.config(), self.workspace / "sweep-b")
        again = run_sweep(self.config(), self.workspace / "sweep-b")
        self.assertEqual(first["completed"], again["completed"])
        self.assertTrue(all(run["resumed"] for run in again["runs"]))
        self.assertTrue(all(run["wall_time_s"] is None for run in again["runs"]))
        self.assertEqual(again["selection"]["index"], first["selection"]["index"])

    def test_failed_runs_are_results_not_crashes(self) -> None:
        with mock.patch.object(Trainer, "train", side_effect=RuntimeError("synthetic failure")):
            report = run_sweep(self.config(budget=1), self.workspace / "sweep-c")
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["completed"], 0)
        self.assertIsNone(report["selection"])
        failure = report["runs"][0]
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["error"]["type"], "RuntimeError")
        self.assertIn("synthetic failure", failure["error"]["message"])

    def test_invalid_sweep_config_is_refused(self) -> None:
        with self.assertRaises(SweepError):
            self.config(families=("not-a-family",))
        with self.assertRaises(SweepError):
            self.config(budget=0)
        with self.assertRaises(SweepError):
            self.config(selection_metric="train_loss")
        with self.assertRaises(SweepError):
            self.config(hidden_sizes=())

    def test_config_round_trips_through_json(self) -> None:
        config = self.config(families=("gnn", "gnn_gru"), hidden_sizes=(8, 16), budget=3)
        restored = SweepConfig.from_payload(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())


if __name__ == "__main__":
    unittest.main()
