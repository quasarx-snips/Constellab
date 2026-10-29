"""CLI acceptance: every pipeline command runs end to end and reports JSON."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from constellab_ml.__main__ import main


def run_cli(*argv: str) -> tuple[int, dict]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
        code = main(list(argv))
    text = buffer.getvalue().strip()
    return code, json.loads(text) if text else {}


class CliTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        cls.campaign = cls.workspace / "campaign"
        code, payload = run_cli(
            "campaign", str(cls.campaign), "--id", "cli-campaign", "--episodes", "16", "--seed", "4"
        )
        assert code == 0, payload
        cls.campaign_payload = payload
        cls.train_config = cls.workspace / "train.json"
        cls.train_config.write_text(json.dumps({
            "run_id": "cli-train",
            "dataset_root": str(cls.campaign),
            "model": {"model": "gnn", "hidden": 8, "layers": 1},
            "epochs": 1,
            "batches_per_epoch": 1,
            "graphs_per_batch": 2,
            "steps_per_chunk": 2,
            "max_sequence_samples": 20,
            "export": False,
        }), encoding="utf-8")

    def test_campaign_command_writes_a_clean_manifest(self) -> None:
        self.assertEqual(self.campaign_payload["campaign_id"], "cli-campaign")
        self.assertEqual(sum(self.campaign_payload["counts"].values()), 16)
        self.assertTrue(self.campaign_payload["leakage_clean"])
        self.assertTrue((self.campaign / "manifest.json").exists())

    def test_train_and_evaluate_round_trip(self) -> None:
        output = self.workspace / "train-run"
        code, payload = run_cli("train", str(self.train_config), "--out", str(output))
        self.assertEqual(code, 0, payload)
        self.assertEqual(payload["run_id"], "cli-train")
        checkpoint = payload["checkpoints"]["last"]
        self.assertTrue(Path(checkpoint).exists())
        code, report = run_cli(
            "evaluate", str(self.campaign), "--checkpoint", checkpoint, "--split", "test", "--limit", "1"
        )
        self.assertEqual(code, 0, report)
        self.assertEqual(report["campaign"], "cli-campaign")
        self.assertEqual(set(report["policies"]), {"learned", "expert-rule", "first-feasible", "wait"})
        self.assertIn("mean_coverage", report["policies"]["learned"]["aggregate"])
        output = self.workspace / "baselines.json"
        code, comparison = run_cli(
            "compare", str(self.campaign), "--checkpoint", f"gnn={checkpoint}",
            "--split", "test", "--limit", "1", "--out", str(output),
        )
        self.assertEqual(code, 0, comparison)
        self.assertTrue(output.exists())
        self.assertEqual(comparison["policies"]["gnn"]["aggregate"]["tasks_total"],
                         comparison["policies"]["expert-rule"]["aggregate"]["tasks_total"])
        self.assertIn("selected on validation", comparison["selection_note"])

    def test_sweep_command_selects_a_run(self) -> None:
        config = self.workspace / "sweep.json"
        config.write_text(json.dumps({
            "sweep_id": "cli-sweep",
            "dataset_root": str(self.campaign),
            "families": ["mlp"],
            "hidden_sizes": [8],
            "budget": 1,
            "epochs": 1,
            "batches_per_epoch": 1,
            "graphs_per_batch": 2,
            "evaluate_episodes": 1,
            "max_sequence_samples": 20,
            "early_stopping_patience": 0,
        }), encoding="utf-8")
        code, payload = run_cli("sweep", str(config), "--out", str(self.workspace / "sweep"))
        self.assertEqual(code, 0, payload)
        self.assertEqual(payload["planned_runs"], 1)
        self.assertEqual(payload["completed"], 1)
        self.assertEqual(payload["selection"]["combination"]["model"]["model"], "mlp")

    def test_curriculum_command_freezes_a_checkpoint(self) -> None:
        config = self.workspace / "curriculum.json"
        config.write_text(json.dumps({
            "curriculum_id": "cli-curriculum",
            "dataset_root": str(self.campaign),
            "model": {"model": "gnn_gru", "hidden": 8, "layers": 1},
            "stages": [{"name": "full"}],
            "seeds": [0],
            "epochs_per_stage": 1,
            "batches_per_epoch": 1,
            "graphs_per_batch": 2,
            "max_sequence_samples": 20,
            "evaluation_episodes": 1,
            "early_stopping_patience": 0,
            "stall_patience": 0,
            "evaluate_test": False,
        }), encoding="utf-8")
        code, payload = run_cli("curriculum", str(config), "--out", str(self.workspace / "curriculum"))
        self.assertEqual(code, 0, payload)
        self.assertEqual(payload["curriculum_id"], "cli-curriculum")
        self.assertEqual(payload["selected_seed"]["seed"], 0)
        seed = payload["seed_reports"][0]
        self.assertEqual(seed["stages_completed"], 1)
        self.assertEqual(seed["stop_reason"], "completed")
        self.assertTrue(Path(seed["selection"]["checkpoint"]).exists())

    def test_checked_in_configs_parse(self) -> None:
        from constellab_ml.curriculum import CurriculumConfig
        from constellab_ml.sweep import SweepConfig
        from constellab_ml.training import TrainingConfig

        root = Path(__file__).resolve().parents[2] / "configs"
        self.assertEqual(
            TrainingConfig.load(root / "train.json").model["model"], "gnn_gru"
        )
        self.assertGreaterEqual(len(SweepConfig.load(root / "sweep.json").families), 2)
        curriculum = CurriculumConfig.load(root / "curriculum.json")
        self.assertEqual([stage.name for stage in curriculum.stages],
                         ["foundation", "growth", "stress", "full"])
        self.assertGreaterEqual(len(curriculum.seeds), 1)

    def test_bad_configuration_is_an_error_exit(self) -> None:
        code, _ = run_cli("train", str(self.workspace / "missing.json"), "--out", str(self.workspace / "nope"))
        self.assertEqual(code, 2)
        code, _ = run_cli("campaign", str(self.workspace / "bad"), "--id", "", "--episodes", "1")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
