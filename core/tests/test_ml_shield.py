"""Day 48: independent shield — mask, resource, deadline, duplicate and fallback."""

from __future__ import annotations

import copy
import unittest

from constellab_ml import SATELLITE_FEATURES
from constellab_ml.env import ACTION_ASSIGN, ACTION_WAIT, Action, SwarmEnvironment
from constellab_ml.env import EnvPolicy
from constellab_ml.scenarios import ScenarioSampler
from constellab_ml.shield import (
    REASON_CODES,
    SafetyShield,
    ShieldConfig,
    ShieldedPolicy,
)

from tests.test_ml_scenarios import fast_distribution_set


def build_environment() -> SwarmEnvironment:
    sampler = ScenarioSampler(seed=11, distribution_set=fast_distribution_set())
    for index in range(20):
        try:
            sample = sampler.sample(index, split="train")
            environment = SwarmEnvironment(sample.spec)
            observation, _ = environment.reset()
            if observation["mask"]["assign"]:
                return environment
        except Exception:
            continue
    raise AssertionError("no sample with a legal action was generated")


def first_pair(observation: dict) -> tuple[str, str]:
    satellite, task = observation["mask"]["assign"][0]
    return (
        str(observation["satellite_ids"][int(satellite)]),
        str(observation["task_ids"][int(task)]),
    )


class ShieldTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = build_environment()
        self.observation, _ = self.environment.reset()
        self.satellite_id, self.task_id = first_pair(self.observation)
        self.action = Action(
            kind=ACTION_ASSIGN, satellite_id=self.satellite_id, task_id=self.task_id
        )

    def test_a_masked_pair_is_accepted(self) -> None:
        decision = SafetyShield().screen(self.observation, [self.action])
        self.assertEqual(len(decision.accepted), 1)
        self.assertFalse(decision.rejected)
        self.assertEqual(decision.accepted[0].satellite_id, self.satellite_id)

    def test_a_pair_outside_the_mask_is_rejected(self) -> None:
        observation = copy.deepcopy(self.observation)
        observation["mask"]["assign"] = []
        decision = SafetyShield().screen(observation, [self.action])
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.rejected[0]["code"], "not-in-mask")
        self.assertEqual(decision.fallback[0]["kind"], ACTION_WAIT)

    def test_energy_floor_and_unknown_resource_are_distinct(self) -> None:
        index = SATELLITE_FEATURES.index("energy_fraction")
        satellite_position = self.observation["satellite_ids"].index(self.satellite_id)
        low = copy.deepcopy(self.observation)
        low["satellites"][satellite_position]["features"][index] = 0.01
        decision = SafetyShield().screen(low, [self.action])
        self.assertEqual(decision.rejected[0]["code"], "energy-floor")

        unknown = copy.deepcopy(self.observation)
        unknown["satellites"][satellite_position]["available"][index] = 0.0
        decision = SafetyShield().screen(unknown, [self.action])
        self.assertEqual(decision.rejected[0]["code"], "resource-unknown")

    def test_deadline_margin_is_enforced(self) -> None:
        task_position = self.observation["task_ids"].index(self.task_id)
        observation = copy.deepcopy(self.observation)
        observation["tasks"][task_position]["deadline_s"] = observation["time_s"] + 5.0
        decision = SafetyShield(ShieldConfig(deadline_margin_s=30.0)).screen(
            observation, [self.action]
        )
        self.assertEqual(decision.rejected[0]["code"], "deadline-too-close")

    def test_duplicate_assignment_in_one_proposal_is_rejected(self) -> None:
        second = Action(
            kind=ACTION_ASSIGN, satellite_id=self.satellite_id, task_id=self.task_id
        )
        decision = SafetyShield().screen(self.observation, [self.action, second])
        self.assertEqual(len(decision.accepted), 1)
        self.assertEqual(decision.rejected[0]["code"], "duplicate-assignment")

    def test_shielded_policy_records_rejections_and_falls_back(self) -> None:
        class ForcedPolicy(EnvPolicy):
            label = "forced-invalid"

            def __init__(self, satellite_id, task_id):
                self.satellite_id = satellite_id
                self.task_id = task_id

            def act(self, observation):
                return [Action(
                    kind=ACTION_ASSIGN,
                    satellite_id=self.satellite_id,
                    task_id=self.task_id,
                )]

            def metadata(self):
                return {"key": self.label}

        policy = ShieldedPolicy(
            ForcedPolicy(self.satellite_id, self.task_id), SafetyShield()
        )
        observation = copy.deepcopy(self.observation)
        observation["mask"]["assign"] = []
        actions = policy.act(observation)
        self.assertEqual([action.kind for action in actions], [ACTION_WAIT])
        summary = policy.shield.summary()
        self.assertEqual(summary["rejected"], 1)
        self.assertIn("not-in-mask", summary["reason_counts"])
        self.assertTrue(set(summary["reason_counts"]).issubset(set(REASON_CODES)))

    def test_report_round_trip_is_json_safe(self) -> None:
        decision = SafetyShield().screen(self.observation, [self.action])
        payload = decision.to_dict()
        self.assertEqual(payload["accepted"][0]["kind"], ACTION_ASSIGN)
        self.assertIn("reason_counts", payload)


if __name__ == "__main__":
    unittest.main()
