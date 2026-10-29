"""Day 28 acceptance tests: policy interface, baselines and the episode harness."""

import json
import unittest

from constellab_core import dynamics, frames, policy
from constellab_core.config import load_config_file
from constellab_core.policy import (
    FixedPriorityPolicy,
    LocalGreedyPolicy,
    OracleUpperBoundPolicy,
    PolicyError,
    PolicyMetadata,
    PolicyObservation,
    Proposal,
    Scene,
    compare_baselines,
    run_episode,
)
from constellab_core.targets import EventTarget


def build_scene(*, horizon_s: float = 3_600.0, step_s: float = 60.0) -> Scene:
    """Four targets, each under a different satellite at some point in time."""
    config = load_config_file("visual-orbit-demo")
    targets = []
    for satellite, time_s in zip(config.satellites, (600.0, 1200.0, 1800.0, 2400.0)):
        position = dynamics.dynamics_model().state_at(satellite.orbit, time_s)[0]
        point = frames.geodetic_from_eci(position, time_s)
        kind = "hotspot" if satellite.id.endswith("alpha") else "change-detection"
        targets.append(
            EventTarget(
                id=f"{kind}.{satellite.id}",
                kind=kind,
                lat_deg=point.latitude_deg,
                lon_deg=point.longitude_deg,
                start_s=0.0,
                end_s=horizon_s,
                urgency=4,
                truth={"intensity_mw": 10.0 + len(targets)},
            )
        )
    return Scene.build(config, targets, horizon_s=horizon_s, step_s=step_s, seed=2)


def synthetic_observation(**overrides) -> PolicyObservation:
    payload = {
        "time_s": 0.0,
        "horizon_s": 600.0,
        "satellites": (
            {"id": "sat-alpha", "sensor": "sensor.conical", "available": True},
        ),
        "targets": (
            {"id": "hotspot.a", "kind": "hotspot", "urgency": 5},
            {"id": "change-detection.b", "kind": "change-detection", "urgency": 2},
        ),
        "tasks": (
            {"id": "task.change-detection.b", "kind": "change-detection", "target_id": "change-detection.b",
             "state": "available", "urgency": 2, "created_s": 0.0, "deadline_s": 900.0},
            {"id": "task.hotspot.a", "kind": "hotspot", "target_id": "hotspot.a",
             "state": "available", "urgency": 5, "created_s": 0.0, "deadline_s": 900.0},
        ),
        "access": (
            {"satellite_id": "sat-alpha", "target_id": "hotspot.a", "start_s": 60.0,
             "end_s": 120.0, "peak_quality": 0.7, "best_resolution_m": 12.0},
            {"satellite_id": "sat-alpha", "target_id": "change-detection.b", "start_s": 60.0,
             "end_s": 120.0, "peak_quality": 0.7, "best_resolution_m": 12.0},
        ),
    }
    payload.update(overrides)
    return PolicyObservation(**payload)


class MetadataTests(unittest.TestCase):
    def test_shipped_metadata_is_valid(self) -> None:
        for metadata in policy.registered_policy_metadata():
            self.assertIn(metadata["kind"], ("baseline", "learned", "oracle"))
            if metadata["kind"] == "oracle":
                self.assertFalse(metadata["deployable"])
            else:
                self.assertTrue(metadata["deployable"])
            self.assertTrue(metadata["key"])
        keys = {item["key"] for item in policy.registered_policy_metadata()}
        self.assertEqual(keys, {
            "local-greedy", "fixed-priority", "leases-only",
            "distributed-greedy", "oracle-upper-bound",
        })

    def test_oracle_can_never_be_marked_deployable(self) -> None:
        with self.assertRaises(PolicyError):
            PolicyMetadata(key="o", label="O", kind="oracle", deployable=True)

    def test_unknown_kind_is_rejected(self) -> None:
        with self.assertRaises(PolicyError):
            PolicyMetadata(key="x", label="X", kind="wizard", deployable=True)


class ObservationSchemaTests(unittest.TestCase):
    def test_observation_contains_no_ground_truth(self) -> None:
        scene = build_scene()
        observation = scene.observe(0.0)
        payload = observation.canonical()
        self.assertNotIn("truth", payload)
        self.assertNotIn("intensity_mw", payload)
        for target in observation.targets:
            self.assertNotIn("truth", target)

    def test_observation_hides_events_that_have_not_started(self) -> None:
        scene = build_scene()
        later = scene.targets[0]
        later.start_s = later.end_s - 60.0
        early = scene.observe(0.0)
        self.assertNotIn(later.id, [target["id"] for target in early.targets])
        late = scene.observe(later.start_s + 1.0)
        self.assertIn(later.id, [target["id"] for target in late.targets])

    def test_observation_is_deterministic(self) -> None:
        scene = build_scene()
        self.assertEqual(scene.observe(0.0).canonical(), scene.observe(0.0).canonical())

    def test_leaked_truth_is_rejected_at_construction(self) -> None:
        with self.assertRaises(PolicyError):
            PolicyObservation(
                time_s=0.0, horizon_s=1.0, satellites=(), tasks=(), access=(),
                targets=({"id": "a", "truth": {"intensity_mw": 1.0}},),
            )


class BaselineTests(unittest.TestCase):
    def test_greedy_and_fixed_priority_run_in_the_same_harness(self) -> None:
        scene = build_scene()
        results = {
            "local-greedy": run_episode(scene, LocalGreedyPolicy()),
            "fixed-priority": run_episode(scene, FixedPriorityPolicy()),
        }
        for result in results.values():
            payload = result.to_dict()
            self.assertIn("metrics", payload)
            self.assertIn("decisions", payload)
            self.assertIn("transitions", payload)
            for key in ("tasks", "completed", "reward", "proposals", "conflicts"):
                self.assertIn(key, result.metrics)
        self.assertEqual(
            set(results["local-greedy"].metrics),
            set(results["fixed-priority"].metrics),
        )

    def test_episodes_are_deterministic_for_the_same_seed(self) -> None:
        scene = build_scene()
        first = run_episode(scene, LocalGreedyPolicy()).trace()
        second = run_episode(build_scene(), LocalGreedyPolicy()).trace()
        self.assertEqual(first, second)

    def test_greedy_claims_a_task_it_can_actually_see(self) -> None:
        scene = build_scene()
        result = run_episode(scene, LocalGreedyPolicy())
        self.assertGreater(result.metrics["proposals"], 0)
        self.assertGreaterEqual(result.metrics["completed"], 1)
        assignments = [
            executed for decision in result.decisions for executed in decision.executed
            if executed.get("result") == "assigned"
        ]
        self.assertTrue(assignments)
        for assignment in assignments:
            self.assertTrue(assignment["task_id"].startswith("task."))

    def test_fixed_priority_claims_the_configured_kind_first(self) -> None:
        observation = synthetic_observation()
        proposals = FixedPriorityPolicy().act(observation)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].task_id, "task.hotspot.a")

    def test_greedy_tie_break_is_stable_by_task_id(self) -> None:
        observation = synthetic_observation(
            tasks=(
                {"id": "task.b", "kind": "hotspot", "target_id": "hotspot.a",
                 "state": "available", "urgency": 3, "created_s": 0.0, "deadline_s": 900.0},
                {"id": "task.a", "kind": "hotspot", "target_id": "hotspot.a",
                 "state": "available", "urgency": 3, "created_s": 0.0, "deadline_s": 900.0},
            ),
            targets=({"id": "hotspot.a", "kind": "hotspot", "urgency": 3},),
            access=(
                {"satellite_id": "sat-alpha", "target_id": "hotspot.a", "start_s": 60.0,
                 "end_s": 120.0, "peak_quality": 0.6, "best_resolution_m": 10.0},
            ),
        )
        proposals = LocalGreedyPolicy().act(observation)
        self.assertEqual(proposals[0].task_id, "task.a")

    def test_a_satellite_already_assigned_is_not_claimed_again(self) -> None:
        observation = synthetic_observation(
            tasks=(
                {"id": "task.a", "kind": "hotspot", "target_id": "hotspot.a",
                 "state": "assigned", "urgency": 5, "created_s": 0.0, "deadline_s": 900.0},
                {"id": "task.b", "kind": "hotspot", "target_id": "hotspot.b",
                 "state": "available", "urgency": 1, "created_s": 0.0, "deadline_s": 900.0},
            ),
            targets=(
                {"id": "hotspot.a", "kind": "hotspot", "urgency": 5},
                {"id": "hotspot.b", "kind": "hotspot", "urgency": 1},
            ),
            access=(
                {"satellite_id": "sat-alpha", "target_id": "hotspot.b", "start_s": 60.0,
                 "end_s": 120.0, "peak_quality": 0.5, "best_resolution_m": 10.0},
            ),
        )
        proposals = LocalGreedyPolicy().act(observation)
        self.assertEqual([p.task_id for p in proposals], ["task.b"])


class HarnessValidationTests(unittest.TestCase):
    def test_illegal_proposals_are_rejected_and_recorded(self) -> None:
        class RoguePolicy:
            metadata = PolicyMetadata(
                key="rogue-test", label="Rogue", kind="baseline", deployable=True,
            )

            def reset(self) -> None:
                pass

            def act(self, observation: PolicyObservation) -> list[Proposal]:
                return [
                    Proposal(action="assign", task_id="task.does-not-exist",
                             satellite_id="sat-alpha", at_s=observation.time_s),
                    Proposal(action="explode", task_id="task.hotspot.sat-alpha",
                             satellite_id="sat-alpha", at_s=observation.time_s),
                ]

        scene = build_scene()
        result = run_episode(scene, RoguePolicy())
        self.assertGreaterEqual(result.metrics["proposals"], 2)
        self.assertGreaterEqual(result.metrics["conflicts"], 2)
        reasons = [
            rejected["reason"] for decision in result.decisions for rejected in decision.rejected
        ]
        self.assertTrue(any("unknown task" in reason for reason in reasons))
        self.assertTrue(any("unknown action" in reason for reason in reasons))

    def test_busy_satellite_proposal_is_rejected(self) -> None:
        class DoubleClaim:
            metadata = PolicyMetadata(
                key="double-claim", label="Double claim", kind="baseline", deployable=True,
            )

            def reset(self) -> None:
                pass

            def act(self, observation: PolicyObservation) -> list[Proposal]:
                return [
                    Proposal(action="assign", task_id="task.hotspot.sat-alpha",
                              satellite_id="sat-alpha", at_s=observation.time_s),
                    Proposal(action="assign", task_id="task.change-detection.sat-bravo",
                              satellite_id="sat-alpha", at_s=observation.time_s),
                ]

        scene = build_scene(step_s=30.0)
        result = run_episode(scene, DoubleClaim())
        busy_rejections = [
            rejected for decision in result.decisions for rejected in decision.rejected
            if "already has an assignment" in rejected["reason"]
        ]
        self.assertTrue(busy_rejections)


class OracleTests(unittest.TestCase):
    def test_oracle_is_refused_unless_explicitly_allowed(self) -> None:
        scene = build_scene()
        with self.assertRaises(PolicyError):
            run_episode(scene, OracleUpperBoundPolicy(scene))

    def test_oracle_runs_as_a_labelled_diagnostic(self) -> None:
        scene = build_scene()
        result = run_episode(scene, OracleUpperBoundPolicy(scene), allow_oracle=True)
        self.assertEqual(result.policy["kind"], "oracle")
        self.assertFalse(result.policy["deployable"])

    def test_comparison_uses_one_harness_for_every_policy(self) -> None:
        report = compare_baselines(build_scene(), allow_oracle=True)
        self.assertEqual(
            set(report["policies"]),
            {"local-greedy", "fixed-priority", "oracle-upper-bound"},
        )
        for entry in report["policies"].values():
            self.assertIn("metrics", entry)
            self.assertEqual(len(entry["trace_hash"]), 16)
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
