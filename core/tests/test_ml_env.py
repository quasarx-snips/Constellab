"""Day 35 acceptance: the episode environment, its contract and its leakage rules."""

from __future__ import annotations

import json
import unittest

from constellab_core import dynamics, faults, frames
from constellab_core.config import load_config, load_config_file
from constellab_core.targets import EventTarget

from constellab_ml import EDGE_FEATURES, SATELLITE_FEATURES, TASK_FEATURES
from constellab_ml.env import (
    Action,
    EnvError,
    EnvPolicy,
    EpisodeSpec,
    SwarmEnvironment,
    multi_agent_split,
    observation_hash,
    replay_episode,
)


def satellites(count: int, *, spread: float = 5.0, components=None) -> list[dict]:
    template = dict(load_config_file("visual-orbit-demo").to_dict()["satellites"][0]["orbit"])
    rows = []
    for index in range(count):
        orbit = dict(template)
        orbit["true_anomaly_deg"] = (float(template["true_anomaly_deg"]) + index * spread) % 360.0
        row = {"id": f"sat-{index:02d}", "name": f"SAT-{index:02d}", "orbit": orbit}
        if components is not None:
            row["components"] = components
        rows.append(row)
    return rows


def hotspot(config, *, target_id: str, time_s: float, horizon_s: float, lon_offset: float = 0.0) -> EventTarget:
    position = dynamics.dynamics_model().state_at(config.satellites[0].orbit, time_s)[0]
    point = frames.geodetic_from_eci(position, time_s)
    return EventTarget(
        id=target_id,
        kind="hotspot",
        lat_deg=point.latitude_deg,
        lon_deg=point.longitude_deg + lon_offset,
        start_s=0.0,
        end_s=horizon_s,
        radius_km=150.0,
        urgency=5,
        requirements={"min_confidence": 0.3},
        truth={"intensity_mw": 9.0, "cause": "test fixture"},
    )


def build_spec(
    *,
    satellite_count: int = 4,
    target_count: int = 2,
    horizon_s: float = 2_400.0,
    step_s: float = 60.0,
    seed: int = 11,
    fault_plan: dict | None = None,
    components=None,
    episode_id: str = "ml-env-test",
    spread: float = 5.0,
) -> EpisodeSpec:
    base = load_config_file("visual-orbit-demo").to_dict()
    config = {
        **base,
        "scenario_id": episode_id,
        "name": episode_id,
        "seed": seed,
        "satellites": satellites(satellite_count, spread=spread, components=components),
    }
    typed = load_config(config)
    targets = [
        hotspot(
            typed,
            target_id=f"hotspot-{index}",
            time_s=horizon_s * (index + 1) / (target_count + 1),
            horizon_s=horizon_s,
            lon_offset=index * 12.0,
        ).to_dict()
        for index in range(target_count)
    ]
    return EpisodeSpec(
        episode_id=episode_id,
        config=config,
        targets=targets,
        horizon_s=horizon_s,
        step_s=step_s,
        seed=seed,
        family="unit",
        fault_plan=fault_plan,
    )


class GreedyRule(EnvPolicy):
    """A deterministic rule policy: one satellite per distinct task, in mask order."""

    label = "greedy-rule"

    def act(self, observation):
        used_satellites: set[int] = set()
        used_tasks: set[int] = set()
        actions = []
        for satellite_index, task_index in observation["mask"]["assign"]:
            if satellite_index in used_satellites or task_index in used_tasks:
                continue
            used_satellites.add(satellite_index)
            used_tasks.add(task_index)
            actions.append(Action(
                kind="assign",
                satellite_id=observation["satellite_ids"][satellite_index],
                task_id=observation["task_ids"][task_index],
            ))
        return actions


class AlwaysWait(EnvPolicy):
    label = "always-wait"

    def act(self, observation):
        del observation
        return [Action(kind="wait")]


class TestObservationContract(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = SwarmEnvironment(build_spec())
        self.observation, self.info = self.environment.reset()

    def test_feature_rows_match_the_declared_schema(self) -> None:
        self.assertEqual(self.observation["version"], self.environment.observation_space["version"])
        for row in self.observation["satellites"]:
            self.assertEqual(len(row["features"]), len(SATELLITE_FEATURES))
            self.assertEqual(len(row["available"]), len(SATELLITE_FEATURES))
        for row in self.observation["tasks"]:
            self.assertEqual(len(row["features"]), len(TASK_FEATURES))
        for edge in self.observation["edges"]["assign"]:
            self.assertEqual(len(edge), 2 + len(EDGE_FEATURES))

    def test_ordering_is_stable_and_identity_is_an_id(self) -> None:
        self.assertEqual(self.observation["satellite_ids"], sorted(self.observation["satellite_ids"]))
        self.assertEqual(self.observation["task_ids"], sorted(self.observation["task_ids"]))
        self.assertEqual(
            [row["id"] for row in self.observation["satellites"]], self.observation["satellite_ids"]
        )

    def test_observation_is_json_safe_and_hash_stable(self) -> None:
        payload = json.dumps(self.observation)
        self.assertIn("satellites", payload)
        self.assertEqual(
            observation_hash(self.observation), observation_hash(self.environment.observation())
        )

    def test_mask_lists_only_legal_pairs(self) -> None:
        legal = {(edge[0], edge[1]) for edge in self.observation["edges"]["assign"]}
        for pair in self.observation["mask"]["assign"]:
            self.assertIn(tuple(pair), legal)
        self.assertTrue(self.observation["mask"]["assign"])

    def test_ground_truth_and_hidden_fields_never_leak(self) -> None:
        payload = json.dumps(self.observation)
        self.assertNotIn('"truth"', payload)
        self.assertNotIn("intensity_mw", payload)
        self.assertNotIn('"reward"', payload)
        for row in self.observation["tasks"]:
            self.assertNotIn("reward", row)
            self.assertNotIn("confidence", row)  # the graded truth stays hidden
        # The injection plan is not exported either — only its observable effects.
        self.assertNotIn("fault_plan", payload)
        self.assertNotIn("injection", payload)

    def test_actions_space_describes_the_mask(self) -> None:
        space = self.environment.action_space
        self.assertEqual(space["kind"], "masked-bipartite-assignment")
        self.assertIn("assign", space["actions"][0])


class TestVariableSizes(unittest.TestCase):
    def test_fleet_and_graph_sizes_vary(self) -> None:
        shapes = {}
        for count in (2, 6, 9):
            environment = SwarmEnvironment(build_spec(satellite_count=count, target_count=1))
            observation, _ = environment.reset()
            shapes[count] = observation["counts"]
            self.assertEqual(observation["counts"]["satellites"], count)
            # Every satellite is order-independent: index i is the i-th id.
            self.assertEqual(observation["satellite_ids"][0], "sat-00")
        self.assertLess(shapes[2]["satellites"], shapes[9]["satellites"])
        self.assertLess(shapes[2]["contact_edges"], shapes[9]["contact_edges"])

    def test_contact_edges_follow_the_link_graph(self) -> None:
        environment = SwarmEnvironment(build_spec(satellite_count=5, target_count=1))
        observation, _ = environment.reset()
        pairs = {tuple(edge) for edge in observation["edges"]["contact"]}
        self.assertEqual(len(pairs), 5 * 4 // 2)  # nothing separates the fleet
        plan = faults.FaultPlan.manual((
            faults.Fault(
                id="partition",
                kind="link-cut",
                start_s=0.0,
                end_s=10_000.0,
                satellites=("sat-00", "sat-01"),
                peers=("sat-02", "sat-03", "sat-04"),
            ),
        ), label="unit partition")
        parted = SwarmEnvironment(build_spec(satellite_count=5, target_count=1, fault_plan=plan.to_dict()))
        observation, _ = parted.reset()
        self.assertEqual(len(observation["edges"]["contact"]), 1 + 3)  # two groups
        self.assertLess(observation["global"][4], 1.0)  # fault_intensity is observable

    def test_missing_component_state_is_unavailable_not_zero(self) -> None:
        components = [{"type": "compute.capacity", "parameters": {"throughput_hz": 1.0e9}}]
        environment = SwarmEnvironment(
            build_spec(satellite_count=2, target_count=1, components=components)
        )
        observation, _ = environment.reset()
        row = observation["satellites"][0]
        index = SATELLITE_FEATURES.index("energy_fraction")
        self.assertEqual(row["available"][index], 0.0)  # no power component declared
        storage = SATELLITE_FEATURES.index("storage_fraction")
        self.assertEqual(row["available"][storage], 0.0)
        compute = SATELLITE_FEATURES.index("compute_duty")
        self.assertEqual(row["available"][compute], 1.0)


class TestStepContract(unittest.TestCase):
    def test_reset_step_and_termination(self) -> None:
        environment = SwarmEnvironment(build_spec(satellite_count=3, target_count=1, horizon_s=1_200.0))
        observation, info = environment.reset()
        self.assertEqual(info["step_index"], 0)
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = environment.step([Action(kind="wait")])
            self.assertIsInstance(reward, float)
        self.assertTrue(terminated or truncated)
        self.assertIn(info["termination_reason"], ("all-tasks-terminal", "horizon"))
        with self.assertRaises(EnvError):
            environment.step([Action(kind="wait")])
        environment.reset()
        self.assertFalse(environment.done)

    def test_horizon_truncation_reason(self) -> None:
        environment = SwarmEnvironment(build_spec(satellite_count=2, target_count=1, horizon_s=600.0))
        environment.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = environment.step([Action(kind="wait")])
        self.assertTrue(truncated or terminated)
        self.assertIn(info["termination_reason"], ("all-tasks-terminal", "horizon"))

    def test_illegal_action_is_rejected_without_state_change(self) -> None:
        environment = SwarmEnvironment(build_spec(satellite_count=3, target_count=1))
        observation, _ = environment.reset()
        illegal = Action(kind="assign", satellite_id="sat-00", task_id="task.hotspot.unknown")
        _, _, _, _, info = environment.step([illegal])
        self.assertEqual(info["metrics"]["conflicts"], 0)
        self.assertEqual(info["metrics"]["invalid_actions"], 1)
        self.assertLess(info["terms"]["constraint_penalty"], 0.0)
        self.assertEqual(info["metrics"]["observed"], 0)

    def test_duplicate_claim_inside_one_round_is_penalised(self) -> None:
        environment = SwarmEnvironment(build_spec(satellite_count=3, target_count=1))
        observation, _ = environment.reset()
        satellite_index = observation["mask"]["assign"][0][0]
        task_index = observation["mask"]["assign"][0][1]
        actions = [
            Action(kind="assign", satellite_id=observation["satellite_ids"][satellite_index],
                   task_id=observation["task_ids"][task_index]),
            Action(kind="assign", satellite_id=observation["satellite_ids"][satellite_index + 1],
                   task_id=observation["task_ids"][task_index]),
        ]
        _, _, _, _, info = environment.step(actions)
        self.assertEqual(info["metrics"]["conflicts"], 1)
        self.assertLess(info["terms"]["duplicate_work"], 0.0)

    def test_reward_terms_are_separately_logged(self) -> None:
        environment = SwarmEnvironment(build_spec(satellite_count=3, target_count=2))
        environment.reset()
        outcome = environment.rollout(GreedyRule())
        terms = outcome.metrics["reward_terms"]
        self.assertEqual(set(terms), {"coverage", "latency", "duplicate_work", "resource_use", "task_loss", "constraint_penalty"})
        self.assertGreater(terms["coverage"], 0.0)
        self.assertLessEqual(terms["latency"], 0.0)

    def test_reward_weights_can_be_reconfigured(self) -> None:
        spec = build_spec(satellite_count=3, target_count=1)
        illegal = [Action(kind="assign", satellite_id="sat-00", task_id="task.hotspot.missing")]
        weighted = SwarmEnvironment(spec)
        weighted.reset()
        _, weighted_reward, _, _, weighted_info = weighted.step(illegal)
        neutral = SwarmEnvironment(spec, reward_weights={"constraint_penalty": 0.0})
        neutral.reset()
        _, neutral_reward, _, _, neutral_info = neutral.step(illegal)
        # The raw term is always logged; the weight decides its influence.
        self.assertEqual(neutral_info["terms"]["constraint_penalty"], weighted_info["terms"]["constraint_penalty"])
        self.assertGreater(neutral_reward, weighted_reward)
        self.assertAlmostEqual(
            neutral_reward - weighted_reward,
            0.25 * abs(weighted_info["terms"]["constraint_penalty"]),
            places=6,
        )

    def test_unknown_reward_term_is_refused(self) -> None:
        with self.assertRaises(EnvError):
            SwarmEnvironment(build_spec(), reward_weights={"not_a_term": 1.0})

    def test_always_wait_never_covers_and_loses_every_task(self) -> None:
        environment = SwarmEnvironment(build_spec(satellite_count=2, target_count=2))
        environment.reset()
        outcome = environment.rollout(AlwaysWait())
        self.assertEqual(outcome.metrics["observed"], 0)
        self.assertEqual(outcome.metrics["coverage"], 0.0)
        self.assertGreater(outcome.metrics["expired"], 0)
        self.assertLess(outcome.metrics["reward_terms"]["task_loss"], 0.0)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_reproduces_trace_and_observations(self) -> None:
        spec = build_spec(satellite_count=4, target_count=2)
        first = SwarmEnvironment(spec)
        first_observation, _ = first.reset()
        first_outcome = first.rollout(GreedyRule(), reset=False)
        second = SwarmEnvironment(spec)
        second_observation, _ = second.reset()
        second_outcome = second.rollout(GreedyRule(), reset=False)
        self.assertEqual(observation_hash(first_observation), observation_hash(second_observation))
        self.assertEqual(first_outcome.trace_hash, second_outcome.trace_hash)
        self.assertEqual(first_outcome.metrics["coverage"], second_outcome.metrics["coverage"])

    def test_different_seed_changes_the_episode(self) -> None:
        first = SwarmEnvironment(build_spec(satellite_count=4, target_count=2, seed=1))
        second = SwarmEnvironment(build_spec(satellite_count=4, target_count=2, seed=2))
        first_outcome = first.rollout(GreedyRule())
        second_outcome = second.rollout(GreedyRule())
        self.assertNotEqual(first_outcome.trace_hash, second_outcome.trace_hash)

    def test_replay_reproduces_the_trace(self) -> None:
        environment = SwarmEnvironment(build_spec(satellite_count=4, target_count=2))
        environment.reset()
        outcome = environment.rollout(GreedyRule(), reset=False)
        report = replay_episode(outcome.manifest, [record["actions"] for record in outcome.records])
        self.assertTrue(report["matches"], report)
        self.assertEqual(report["metrics"]["coverage"], outcome.metrics["coverage"])
        self.assertEqual(report["metrics"]["observed"], outcome.metrics["observed"])


class TestFaultInteraction(unittest.TestCase):
    def test_failed_satellite_leaves_the_action_mask(self) -> None:
        plan = faults.FaultPlan.manual((
            faults.Fault(
                id="kill-00",
                kind="satellite-failure",
                start_s=0.0,
                end_s=100_000.0,
                satellites=("sat-00",),
            ),
        ), label="unit failure")
        healthy = SwarmEnvironment(build_spec(satellite_count=4, target_count=1))
        failed = SwarmEnvironment(build_spec(satellite_count=4, target_count=1, fault_plan=plan.to_dict()))
        healthy_observation, _ = healthy.reset()
        failed_observation, _ = failed.reset()
        self.assertTrue(any(pair[0] == 0 for pair in healthy_observation["mask"]["assign"]))
        self.assertFalse(any(pair[0] == 0 for pair in failed_observation["mask"]["assign"]))
        row = failed_observation["satellites"][0]
        self.assertFalse(row["is_available"])
        self.assertEqual(row["features"][SATELLITE_FEATURES.index("available")], 0.0)

    def test_sensor_degrade_changes_observable_quality(self) -> None:
        plan = faults.FaultPlan.manual((
            faults.Fault(
                id="degrade-00",
                kind="sensor-degrade",
                start_s=0.0,
                end_s=100_000.0,
                satellites=("sat-00",),
                quality=0.25,
            ),
        ), label="unit degrade")
        clean = SwarmEnvironment(build_spec(satellite_count=3, target_count=1))
        degraded = SwarmEnvironment(build_spec(satellite_count=3, target_count=1, fault_plan=plan.to_dict()))
        clean_observation, _ = clean.reset()
        degraded_observation, _ = degraded.reset()

        def first_quality(observation):
            return min(edge[2] for edge in observation["edges"]["assign"] if edge[0] == 0)

        self.assertLess(first_quality(degraded_observation), first_quality(clean_observation))


class TestMultiAgentView(unittest.TestCase):
    def test_each_agent_only_may_act_on_its_own_row(self) -> None:
        environment = SwarmEnvironment(build_spec(satellite_count=4, target_count=2))
        observation, _ = environment.reset()
        views = multi_agent_split(observation)
        self.assertEqual(sorted(views), observation["satellite_ids"])
        for index, satellite_id in enumerate(observation["satellite_ids"]):
            view = views[satellite_id]
            self.assertEqual(view["agent_index"], index)
            self.assertTrue(all(pair[0] == index for pair in view["mask"]["assign"]))
            self.assertEqual(view["satellites"], observation["satellites"])

    def test_claim_delivery_decides_what_a_peer_sees(self) -> None:
        # No partition: the claim crosses, so every agent sees the assignment.
        open_fleet = SwarmEnvironment(build_spec(satellite_count=4, target_count=1))
        observation, _ = open_fleet.reset()
        pair = observation["mask"]["assign"][0]
        owner = observation["satellite_ids"][pair[0]]
        task_id = observation["task_ids"][pair[1]]
        open_fleet.step([Action(kind="assign", satellite_id=owner, task_id=task_id)])
        peers = [item for item in observation["satellite_ids"] if item != owner]
        delivered = open_fleet.observation(satellite_id=peers[0])
        self.assertEqual(
            next(row for row in delivered["tasks"] if row["id"] == task_id)["state"], "in-progress"
        )

        # Partition the owner from the rest: the peer's local view goes stale and
        # still reports the task as available, which is the failure mechanism.
        plan = faults.FaultPlan.manual((
            faults.Fault(
                id="unit-partition",
                kind="link-cut",
                start_s=0.0,
                end_s=100_000.0,
                satellites=(owner,),
                peers=tuple(peers),
            ),
        ), label="unit partition")
        parted = SwarmEnvironment(
            build_spec(satellite_count=4, target_count=1, fault_plan=plan.to_dict())
        )
        parted_observation, _ = parted.reset()
        parted.step([Action(kind="assign", satellite_id=owner, task_id=task_id)])
        global_row = next(
            row for row in parted.observation()["tasks"] if row["id"] == task_id
        )
        self.assertEqual(global_row["state"], "in-progress")
        local = parted.observation(satellite_id=peers[0])
        local_row = next(row for row in local["tasks"] if row["id"] == task_id)
        self.assertEqual(local_row["state"], "available")  # no claim reached this agent
        owner_view = parted.observation(satellite_id=owner)
        owner_row = next(row for row in owner_view["tasks"] if row["id"] == task_id)
        self.assertEqual(owner_row["state"], "in-progress")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
