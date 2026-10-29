"""Day 33 acceptance tests: every invariant rule fires and stays quiet when valid."""

import unittest

from constellab_core import faults, g2, invariants, policy
from constellab_core.invariants import (
    DEFAULT_RULES,
    Finding,
    InvariantEngine,
    InvariantRule,
    compare_allocations,
    evaluate_invariants,
)


def engine() -> InvariantEngine:
    return InvariantEngine()


class RuleRegistryTests(unittest.TestCase):
    def test_shipped_rules_are_valid_and_unique(self) -> None:
        keys = [rule.key for rule in DEFAULT_RULES]
        self.assertEqual(len(keys), len(set(keys)))
        payload = invariants.rules_payload()
        self.assertEqual(len(payload["rules"]), len(DEFAULT_RULES))
        self.assertIn("duplicate_task_claim", payload["evaluators"])

    def test_unknown_evaluator_is_rejected(self) -> None:
        with self.assertRaises(invariants.InvariantError):
            InvariantRule(key="k", evaluator="wishful_thinking", description="?")
        with self.assertRaises(invariants.InvariantError):
            InvariantRule(key="k", evaluator="missed_deadline", description="?", severity="fatal")


class InvariantGateTests(unittest.TestCase):
    """The G2 fault case must fire the duplicate rule; the control must not."""

    def setUp(self) -> None:
        self.faulty = g2.run_case(faults_on=True)
        self.control = g2.run_case(faults_on=False)

    def test_duplicate_claim_fires_only_under_the_partition(self) -> None:
        faulty = engine().summary(self.faulty["result"], faults=self.faulty["faults"])
        control = engine().summary(self.control["result"], faults=self.control["faults"])
        self.assertGreaterEqual(faulty["counts"].get("duplicate-task-claim", 0), 1)
        self.assertEqual(control["counts"].get("duplicate-task-claim", 0), 0)

    def test_the_finding_names_the_satellites_and_the_task(self) -> None:
        findings = [
            finding for finding in self.faulty["findings"]
            if finding.rule == "duplicate-task-claim"
        ]
        self.assertTrue(findings)
        entities = findings[0].entities
        self.assertIn("task.hotspot.g2-eq", entities)
        self.assertGreaterEqual(len(entities), 3)
        self.assertTrue(any(entity in g2.G2_GROUPS["A"] for entity in entities))
        self.assertTrue(any(entity in g2.G2_GROUPS["B"] for entity in entities))

    def test_every_rejected_proposal_is_attributable(self) -> None:
        summary = engine().summary(self.faulty["result"], faults=self.faulty["faults"])
        rejected = summary["counts"].get("invalid-policy-action", 0)
        self.assertEqual(rejected, self.faulty["result"].metrics["conflicts"])

    def test_a_clean_episode_has_no_error_findings(self) -> None:
        summary = engine().summary(self.control["result"], faults=self.control["faults"])
        self.assertEqual(summary["errors"], 0)

    def test_a_gameable_low_urgency_task_no_longer_fires_missed_deadline(self) -> None:
        # The rule only covers urgent tasks (urgency >= 4), so a quiet task is not
        # reported as a failure. Direct check against the control result.
        summary = engine().summary(self.control["result"], faults=self.control["faults"])
        self.assertNotIn("missed-deadline", summary["counts"])


class SyntheticRuleTests(unittest.TestCase):
    """Each rule has a positive case and a near-boundary negative case."""

    def _episode(self, **overrides):
        base = dict(
            policy={"key": "test"},
            metrics={"conflicts": 0, "usable_windows": 1},
            decisions=[],
            transitions=[],
            outcomes=[],
            tasks=[],
        )
        base.update(overrides)
        return policy.EpisodeResult(**base)

    def test_assignment_expired_fires_only_when_a_satellite_held_it(self) -> None:
        held = self._episode(transitions=[{
            "task_id": "task.1", "action": "expire", "at_s": 100.0, "accepted": True,
            "from": "assigned", "to": "expired", "satellite_id": "sat-a",
        }])
        unheld = self._episode(transitions=[{
            "task_id": "task.2", "action": "expire", "at_s": 100.0, "accepted": True,
            "from": "available", "to": "expired", "satellite_id": None,
        }])
        rule = [item for item in DEFAULT_RULES if item.key == "assignment-expired"][0]
        self.assertEqual(len(InvariantEngine([rule]).evaluate(held)), 1)
        self.assertEqual(InvariantEngine([rule]).evaluate(unheld), [])

    def test_missed_deadline_respects_the_urgency_threshold(self) -> None:
        rule = [item for item in DEFAULT_RULES if item.key == "missed-deadline"][0]
        urgent = self._episode(tasks=[{
            "id": "task.hot", "target_id": "hotspot.1", "state": "expired",
            "urgency": 5, "deadline_s": 900.0,
        }])
        calm = self._episode(tasks=[{
            "id": "task.calm", "target_id": "hotspot.2", "state": "expired",
            "urgency": 2, "deadline_s": 900.0,
        }])
        self.assertEqual(len(InvariantEngine([rule]).evaluate(urgent)), 1)
        self.assertEqual(InvariantEngine([rule]).evaluate(calm), [])

    def test_resource_budget_stays_quiet_just_below_the_floor(self) -> None:
        rule = [item for item in DEFAULT_RULES if item.key == "resource-budget"][0]
        under = self._episode(outcomes=[{
            "satellite_id": "sat-a", "accepted": True, "energy_j": 400_000.0,
        }])
        over = self._episode(outcomes=[{
            "satellite_id": "sat-a", "accepted": True, "energy_j": 600_000.0,
        }])
        self.assertEqual(InvariantEngine([rule]).evaluate(under), [])
        self.assertEqual(len(InvariantEngine([rule]).evaluate(over)), 1)

    def test_failed_satellite_held_task_needs_a_failure_window(self) -> None:
        plan = faults.FaultPlan.manual((
            faults.Fault(id="f", kind="satellite-failure", start_s=50.0, end_s=150.0, satellites=("sat-a",)),
        ))
        rule = [item for item in DEFAULT_RULES if item.key == "failed-satellite-held-task"][0]
        episode = self._episode(
            transitions=[{
                "task_id": "task.1", "action": "assign", "at_s": 0.0, "accepted": True,
                "from": "available", "to": "assigned", "satellite_id": "sat-a",
            }],
            tasks=[{"id": "task.1", "deadline_s": 900.0}],
        )
        self.assertEqual(len(InvariantEngine([rule]).evaluate(episode, faults=plan)), 1)
        early = self._episode(
            transitions=[{
                "task_id": "task.1", "action": "assign", "at_s": 0.0, "accepted": True,
                "from": "available", "to": "assigned", "satellite_id": "sat-a",
            }, {
                "task_id": "task.1", "action": "complete", "at_s": 40.0, "accepted": True,
                "from": "in-progress", "to": "completed", "satellite_id": "sat-a",
            }],
            tasks=[{"id": "task.1", "deadline_s": 900.0}],
        )
        self.assertEqual(InvariantEngine([rule]).evaluate(early, faults=plan), [])


class AllocationComparisonTests(unittest.TestCase):
    def test_comparison_runs_the_same_harness_for_each_baseline(self) -> None:
        _config, scene = g2.build_scene()
        report = compare_allocations(scene, faults=g2.partition_plan())
        self.assertEqual(
            set(report["policies"]), {"local-greedy", "leases-only", "fixed-priority"},
        )
        for entry in report["policies"].values():
            self.assertIn("metrics", entry)
            self.assertIn("invariants", entry)
            self.assertEqual(len(entry["trace_hash"]), 64)
        # Centralized baselines share the global board, so they never produce the
        # decentralized duplicate-claim finding on this scenario.
        for entry in report["policies"].values():
            self.assertEqual(entry["invariants"]["counts"].get("duplicate-task-claim", 0), 0)


if __name__ == "__main__":
    unittest.main()
