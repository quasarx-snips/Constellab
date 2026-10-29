"""Day 47: PPO rollout/update, checkpoint reload and legal deployment actions."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from constellab_ml.env import SwarmEnvironment
from constellab_ml.ppo import (
    PPOConfig,
    PPOPolicy,
    PPOError,
    Transition,
    compute_gae,
    train_ppo,
)
from constellab_ml.scenarios import ScenarioSampler

from tests.test_ml_scenarios import fast_distribution_set


def sample_spec() -> object:
    sampler = ScenarioSampler(seed=47, distribution_set=fast_distribution_set())
    for index in range(15):
        try:
            spec = sampler.sample(index, split="train").spec
            if SwarmEnvironment(spec).reset()[0]["mask"]["assign"]:
                return spec
        except Exception:
            continue
    raise AssertionError("no usable PPO sample")


class PPOTestCase(unittest.TestCase):
    def test_gae_is_finite_and_terminal_masks_bootstrap(self) -> None:
        transitions = [
            Transition({}, 0, -0.1, 0.2, 1.0, False, 0),
            Transition({}, 0, -0.2, 0.5, 1.0, True, 0),
        ]
        advantages, returns = compute_gae(transitions, gamma=0.9, gae_lambda=0.9)
        self.assertEqual(len(advantages), 2)
        self.assertTrue(all(math.isfinite(value) for value in advantages + returns))
        self.assertAlmostEqual(returns[1], 1.0, places=6)

    def test_recurrent_model_is_refused(self) -> None:
        with self.assertRaises(PPOError):
            PPOConfig(model={"model": "gnn_gru"})

    def test_ppo_training_writes_a_reloadable_checkpoint(self) -> None:
        root = Path(tempfile.mkdtemp())
        spec = sample_spec()
        config = PPOConfig(
            run_id="ppo-smoke",
            model={"model": "gnn", "hidden": 8, "layers": 1},
            updates=2,
            epochs=1,
            episodes_per_update=1,
            max_episode_steps=2,
            validation_episodes=1,
            evaluate_every=1,
            seed=4,
        )
        manifest = train_ppo(
            config, root / "run", train_specs=[spec], validation_specs=[spec]
        )
        self.assertTrue((root / "run" / "ppo-manifest.json").exists())
        self.assertTrue((root / "run" / "checkpoint-last.pt").exists())
        self.assertEqual(len(manifest["history"]), 2)
        for entry in manifest["history"]:
            self.assertTrue(math.isfinite(entry["policy_loss"]))
            self.assertTrue(math.isfinite(entry["value_loss"]))
        policy = PPOPolicy.load(root / "run" / "checkpoint-last.pt")
        observation, _ = SwarmEnvironment(spec).reset()
        actions = policy.act(observation)
        self.assertTrue(actions)
        self.assertIn(actions[0].kind, ("assign", "wait"))
        self.assertEqual(policy.metadata()["version"], "ppo-1")

    def test_training_is_reproducible_with_the_same_seed(self) -> None:
        root = Path(tempfile.mkdtemp())
        spec = sample_spec()
        config = PPOConfig(
            run_id="repro",
            model={"model": "gnn", "hidden": 8, "layers": 1},
            updates=1,
            epochs=1,
            episodes_per_update=1,
            max_episode_steps=1,
            evaluate_every=0,
            seed=8,
        )
        first = train_ppo(config, root / "a", train_specs=[spec])
        second = train_ppo(config, root / "b", train_specs=[spec])
        self.assertEqual(first["history"][0]["policy_loss"], second["history"][0]["policy_loss"])
        payload = json.loads((root / "a" / "ppo-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], "ppo-1")


if __name__ == "__main__":
    unittest.main()
