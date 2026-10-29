"""Days 36-37 acceptance: expert demonstrations and the episode dataset pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from constellab_ml.dataset import (
    DATASET_VERSION,
    DatasetConfig,
    DatasetError,
    EpisodeDataset,
    compact_observation,
    generate_campaign,
)
from constellab_ml.env import EnvPolicy, SwarmEnvironment
from constellab_ml.expert import (
    DEFAULT_EXPERT,
    EXPERT_KEYS,
    RuleExpert,
    ScoreWeights,
    WindowOrderExpert,
    demonstration_targets,
    expert_catalog,
    expert_factory,
    greedy_assignment,
    leakage_scan,
    score_candidates,
)
from constellab_ml.scenarios import SPLIT_NAMES, ScenarioSampler

from tests.test_ml_scenarios import fast_distribution_set


def fast_sampler(**kwargs) -> ScenarioSampler:
    return ScenarioSampler(seed=3, distribution_set=fast_distribution_set(), salt="unit", **kwargs)


class TestCandidateScoring(unittest.TestCase):
    def setUp(self) -> None:
        sample = fast_sampler().sample(0, split="train")
        self.environment = SwarmEnvironment(sample.spec)
        self.observation, _ = self.environment.reset()

    def test_candidates_are_a_subset_of_the_mask(self) -> None:
        candidates = score_candidates(self.observation)
        mask = {(int(pair[0]), int(pair[1])) for pair in self.observation["mask"]["assign"]}
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertIn((candidate.satellite_index, candidate.task_index), mask)
        self.assertEqual(len(candidates), len(mask))

    def test_candidates_are_sorted_by_score(self) -> None:
        candidates = score_candidates(self.observation)
        scores = [candidate.score for candidate in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_are_explainable_terms(self) -> None:
        candidate = score_candidates(self.observation)[0]
        self.assertAlmostEqual(candidate.score, sum(candidate.terms.values()), places=6)
        self.assertIn("contention", candidate.terms)

    def test_weights_change_the_ranking(self) -> None:
        quality_only = ScoreWeights(quality=5.0, urgency=0.0, deadline=0.0, window_soon=0.0,
                                    duration=0.0, contention=0.0, resource=0.0)
        ranked = score_candidates(self.observation, quality_only)
        best_quality = max(float(edge[2]) for edge in self.observation["edges"]["assign"])
        self.assertAlmostEqual(ranked[0].terms["quality"], 5.0 * best_quality, places=5)

    def test_greedy_assignment_is_a_matching(self) -> None:
        candidates = score_candidates(self.observation)
        chosen = greedy_assignment(candidates)
        satellites = [item.satellite_index for item in chosen]
        tasks = [item.task_index for item in chosen]
        self.assertEqual(len(satellites), len(set(satellites)))
        self.assertEqual(len(tasks), len(set(tasks)))
        for candidate in candidates:
            if not candidate.chosen:
                self.assertTrue(candidate.dropped_reason)


class TestRuleExpert(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = fast_sampler().sample(0, split="train")
        self.environment = SwarmEnvironment(self.sample.spec)

    def test_rollout_covers_work_and_records_decisions(self) -> None:
        expert = RuleExpert()
        outcome = self.environment.rollout(expert)
        self.assertGreaterEqual(outcome.metrics["observed"], 1)
        self.assertEqual(len(expert.decisions), len(outcome.steps))
        self.assertTrue(all(record.candidate_count >= 0 for record in expert.decisions))

    def test_metadata_is_explicit_about_deployability(self) -> None:
        expert = RuleExpert()
        self.environment.rollout(expert)
        metadata = expert.metadata()
        self.assertEqual(metadata["kind"], "rule")
        self.assertTrue(metadata["deployable"])
        self.assertEqual(metadata["version"], "expert-1")
        self.assertEqual(metadata["decisions"], len(expert.decisions))

    def test_expert_abstains_when_the_mask_is_empty(self) -> None:
        observation, _ = self.environment.reset()
        observation = json.loads(json.dumps(observation))
        observation["mask"]["assign"] = []
        expert = RuleExpert()
        actions = expert.act(observation)
        self.assertEqual([action.kind for action in actions], ["wait"])
        self.assertTrue(expert.decisions[-1].abstained)
        self.assertEqual(expert.decisions[-1].abstain_reason, "action mask is empty")

    def test_demonstration_targets_use_observation_indices(self) -> None:
        observation, _ = self.environment.reset()
        expert = RuleExpert()
        actions = expert.act(observation)
        targets = demonstration_targets(observation, actions)
        self.assertEqual(len(targets), len([a for a in actions if a.kind == "assign"]))
        for satellite_index, task_index in targets:
            self.assertLess(satellite_index, len(observation["satellite_ids"]))
            self.assertLess(task_index, len(observation["task_ids"]))

    def test_reset_clears_decision_history(self) -> None:
        expert = RuleExpert()
        expert.act(self.environment.reset()[0])
        self.assertTrue(expert.decisions)
        expert.reset()
        self.assertEqual(expert.decisions, [])


class TestDiagnosticExpert(unittest.TestCase):
    def test_diagnostic_expert_is_labelled_not_deployable(self) -> None:
        sample = fast_sampler().sample(1, split="train")
        environment = SwarmEnvironment(sample.spec)
        expert = WindowOrderExpert(environment)
        self.assertFalse(expert.deployable)
        self.assertEqual(expert.kind, "diagnostic")
        environment.rollout(expert)
        self.assertFalse(expert.metadata()["deployable"])

    def test_registry_lists_every_expert_with_its_kind(self) -> None:
        catalog = expert_catalog()
        self.assertEqual({entry["key"] for entry in catalog}, set(EXPERT_KEYS))
        for entry in catalog:
            self.assertIn(entry["kind"], ("rule", "diagnostic"))
            self.assertEqual(entry["deployable"], entry["kind"] == "rule")

    def test_factory_refuses_unknown_keys(self) -> None:
        with self.assertRaises(KeyError):
            expert_factory("expert-imaginary")

    def test_diagnostic_expert_needs_an_environment(self) -> None:
        with self.assertRaises(ValueError):
            expert_factory("expert-window-order")

    def test_default_expert_is_the_rule(self) -> None:
        self.assertEqual(DEFAULT_EXPERT, RuleExpert.label)
        self.assertIsInstance(expert_factory(DEFAULT_EXPERT), RuleExpert)


class TestLeakageScan(unittest.TestCase):
    def test_scan_finds_hidden_fields(self) -> None:
        found = leakage_scan({"observation": {"tasks": [{"truth": {"x": 1}}]}})
        self.assertEqual(found, ["truth"])

    def test_scan_accepts_a_clean_student_view(self) -> None:
        self.assertEqual(leakage_scan({"observation": {"tasks": [{"id": "t", "features": [0.1]}]}}), [])

    def test_compact_observation_drops_the_context(self) -> None:
        sample = fast_sampler().sample(0, split="train")
        observation, _ = SwarmEnvironment(sample.spec).reset()
        compact = compact_observation(observation)
        self.assertNotIn("context", compact)
        self.assertIn("satellites", compact)


class TestCampaignPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp()) / "campaign"
        self.config = DatasetConfig(
            campaign_id="unit-campaign",
            seed=1,
            expert_key=DEFAULT_EXPERT,
            salt="unit",
        )

    def test_campaign_writes_splits_manifest_and_index(self) -> None:
        report = generate_campaign(
            self.root, self.config, 8, distribution_set=fast_distribution_set()
        )
        self.assertEqual(report["generated"], 8)
        manifest = report["manifest"]
        self.assertEqual(manifest["dataset_version"], DATASET_VERSION)
        self.assertEqual(sum(manifest["counts"].values()), 8)
        self.assertTrue(manifest["coverage"])
        self.assertTrue(manifest["leakage"]["clean"])
        self.assertEqual(manifest["expert"]["key"], DEFAULT_EXPERT)
        self.assertIn("distribution_hash", manifest["scenario"])
        self.assertTrue((self.root / "manifest.json").exists())
        self.assertTrue((self.root / "index.json").exists())
        self.assertTrue(list(self.root.glob("episodes-*.jsonl.gz")))

    def test_resume_skips_recorded_indices(self) -> None:
        first = generate_campaign(self.root, self.config, 5, distribution_set=fast_distribution_set())
        second = generate_campaign(self.root, self.config, 5, distribution_set=fast_distribution_set())
        self.assertEqual(first["generated"], 5)
        self.assertEqual(second["generated"], 0)
        self.assertEqual(second["skipped"], 5)
        third = generate_campaign(
            self.root, self.config, 3, start=5, distribution_set=fast_distribution_set()
        )
        self.assertEqual(third["generated"], 3)
        self.assertEqual(sum(third["manifest"]["counts"].values()), 8)

    def test_regeneration_is_deterministic(self) -> None:
        generate_campaign(self.root, self.config, 4, distribution_set=fast_distribution_set())
        other = Path(tempfile.mkdtemp()) / "campaign"
        generate_campaign(other, self.config, 4, distribution_set=fast_distribution_set())
        first = EpisodeDataset(self.root)
        second = EpisodeDataset(other)
        for split in SPLIT_NAMES:
            left = [(episode["spec_hash"], episode["trace_hash"]) for episode in first.episodes(split)]
            right = [(episode["spec_hash"], episode["trace_hash"]) for episode in second.episodes(split)]
            self.assertEqual(left, right, split)

    def test_dataset_loads_samples_with_targets(self) -> None:
        generate_campaign(self.root, self.config, 8, distribution_set=fast_distribution_set())
        dataset = EpisodeDataset(self.root, campaign_id="unit-campaign")
        summary = dataset.summary()
        self.assertEqual(summary["campaign_id"], "unit-campaign")
        self.assertTrue(summary["leakage_clean"])
        samples = dataset.decision_samples("train")
        self.assertTrue(samples)
        self.assertTrue(any(sample["targets"] for sample in samples))
        for sample in samples:
            self.assertNotIn("truth", json.dumps(sample["observation"]))
            self.assertNotIn("reward", json.dumps(sample["observation"]))

    def test_split_assignment_comes_from_the_sampler(self) -> None:
        generate_campaign(self.root, self.config, 40, distribution_set=fast_distribution_set())
        dataset = EpisodeDataset(self.root)
        sampler = ScenarioSampler(
            seed=self.config.seed, distribution_set=fast_distribution_set(), salt=self.config.salt
        )
        for split in SPLIT_NAMES:
            for episode in dataset.episodes(split):
                index = int(episode["episode_id"].rsplit("-", 1)[-1])
                self.assertEqual(episode["split"], split)
                self.assertEqual(sampler.split_for(index), split)

    def test_campaign_mismatch_is_refused(self) -> None:
        generate_campaign(self.root, self.config, 2, distribution_set=fast_distribution_set())
        with self.assertRaises(DatasetError):
            EpisodeDataset(self.root, campaign_id="another-campaign")

    def test_missing_manifest_is_reported(self) -> None:
        with self.assertRaises(DatasetError):
            EpisodeDataset(Path(tempfile.mkdtemp()))

    def test_diagnostic_expert_is_refused_for_imitation_data(self) -> None:
        config = DatasetConfig(
            campaign_id="diagnostic",
            seed=2,
            expert_key=WindowOrderExpert.label,
        )
        with self.assertRaises(DatasetError):
            generate_campaign(
                Path(tempfile.mkdtemp()) / "diagnostic",
                config,
                2,
                distribution_set=fast_distribution_set(),
            )

    def test_unknown_expert_is_refused(self) -> None:
        config = DatasetConfig(campaign_id="x", seed=1, expert_key="expert-nope")
        with self.assertRaises(DatasetError):
            generate_campaign(
                Path(tempfile.mkdtemp()) / "x", config, 1, distribution_set=fast_distribution_set()
            )

    def test_observations_can_be_omitted(self) -> None:
        config = DatasetConfig(
            campaign_id="no-obs",
            seed=3,
            include_observations=False,
        )
        root = Path(tempfile.mkdtemp()) / "no-obs"
        generate_campaign(root, config, 4, distribution_set=fast_distribution_set())
        dataset = EpisodeDataset(root)
        for split in SPLIT_NAMES:
            for episode in dataset.episodes(split):
                self.assertTrue(all("observation" not in step for step in episode["steps"]))
        self.assertEqual(dataset.decision_samples("train"), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
