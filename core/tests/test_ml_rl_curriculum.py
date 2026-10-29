"""Day 49: PPO curriculum stages, seed report, transfer and resume."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from constellab_ml.dataset import DatasetConfig, generate_campaign
from constellab_ml.rl_curriculum import (
    RLCurriculumConfig,
    RLStage,
    run_rl_curriculum,
)

from tests.test_ml_scenarios import fast_distribution_set


class RLCurriculumTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp())
        cls.campaign = cls.root / "campaign"
        generate_campaign(
            cls.campaign,
            DatasetConfig(campaign_id="rl-curriculum", seed=49),
            16,
            distribution_set=fast_distribution_set(),
        )

    def config(self, *, resume: bool = True) -> RLCurriculumConfig:
        return RLCurriculumConfig(
            run_id="rl-smoke",
            dataset_root=str(self.campaign),
            stages=[
                RLStage("small", 4, 3, episodes=1, updates=1, max_steps=1),
                RLStage("larger", 12, 8, episodes=1, updates=1, max_steps=1),
            ],
            seeds=[3],
            model={"model": "gnn", "hidden": 8, "layers": 1},
            max_train_episodes=4,
            validation_episodes=2,
            resume=resume,
        )

    def test_curriculum_writes_stage_checkpoints_and_seed_report(self) -> None:
        target = self.root / "run"
        payload = run_rl_curriculum(self.config(), target)
        self.assertEqual(payload["version"], "rl-curriculum-1")
        self.assertEqual(len(payload["seed_reports"]), 1)
        stages = payload["seed_reports"][0]["stages"]
        self.assertEqual([stage["stage"] for stage in stages], ["small", "larger"])
        self.assertTrue(all(stage["status"] == "completed" for stage in stages))
        self.assertFalse(stages[0]["initialized_from_previous"])
        self.assertTrue(stages[1]["initialized_from_previous"])
        self.assertTrue((target / "rl-curriculum-manifest.json").exists())
        self.assertTrue((target / "seed-3" / "small" / "ppo-manifest.json").exists())

    def test_resume_skips_completed_stages(self) -> None:
        target = self.root / "resume-run"
        first = run_rl_curriculum(self.config(resume=True), target)
        second = run_rl_curriculum(self.config(resume=True), target)
        self.assertEqual(
            len(first["seed_reports"][0]["stages"]),
            len(second["seed_reports"][0]["stages"]),
        )
        self.assertEqual(
            [stage["status"] for stage in second["seed_reports"][0]["stages"]],
            ["completed", "completed"],
        )
        saved = json.loads((target / "rl-curriculum-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["summary"]["seeds"], 1)


if __name__ == "__main__":
    unittest.main()
