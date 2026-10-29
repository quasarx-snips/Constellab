"""Day 45 acceptance: staged curriculum training and held-out evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from constellab_ml.batch import GraphSample
from constellab_ml.curriculum import (
    CURRICULUM_VERSION,
    CurriculumConfig,
    CurriculumError,
    CurriculumRunner,
    FirstCandidatePolicy,
    StageSpec,
    WaitPolicy,
    balanced_weights,
    baseline_policies,
    curriculum_report,
    default_stages,
    fleet_bucket,
    run_curriculum,
    sample_key,
)
from constellab_ml.dataset import DatasetConfig, EpisodeDataset, generate_campaign

from tests.test_ml_scenarios import fast_distribution_set


def campaign(root: Path, *, count: int = 18, seed: int = 5) -> EpisodeDataset:
    generate_campaign(
        root,
        DatasetConfig(campaign_id="curriculum-campaign", seed=seed),
        count,
        distribution_set=fast_distribution_set(),
    )
    return EpisodeDataset(root)


def sample(**attributes) -> GraphSample:
    return GraphSample(observation={}, episode_id="ep", family="balanced", attributes=dict(attributes))


class StageTestCase(unittest.TestCase):
    def test_default_ladder_widens_and_ends_unconstrained(self) -> None:
        stages = default_stages()
        self.assertEqual([stage.name for stage in stages], ["foundation", "growth", "stress", "full"])
        self.assertEqual(stages[-1].bounds(), {})
        previous = 0
        for stage in stages[:-1]:
            fleet = stage.bounds().get("fleet.size", 0)
            self.assertGreater(fleet, previous)
            previous = fleet

    def test_stage_rejections_name_the_axis(self) -> None:
        stage = StageSpec(
            "small",
            max_fleet_size=4,
            max_targets=2,
            max_faults=1,
            max_partitions=0,
            families=("balanced",),
        )
        self.assertIsNone(stage.rejection_reason(sample(**{"fleet.size": 3})))
        self.assertIn("fleet.size", stage.rejection_reason(sample(**{"fleet.size": 9})) or "")
        self.assertIn("tasks.target_count", stage.rejection_reason(sample(**{"tasks.target_count": 5})) or "")
        self.assertIn("faults.count", stage.rejection_reason(sample(**{"faults.count": 2})) or "")
        self.assertIn("network.partitions", stage.rejection_reason(sample(**{"network.partitions": 1})) or "")
        self.assertIn("family", stage.rejection_reason(
            GraphSample(observation={}, family="sparse", attributes={})
        ) or "")

    def test_missing_metadata_does_not_reject_an_episode(self) -> None:
        stage = StageSpec("small", max_fleet_size=2)
        self.assertTrue(stage.accepts(GraphSample(observation={}, episode_id="missing")))

    def test_fleet_buckets_are_contiguous(self) -> None:
        self.assertEqual(fleet_bucket(1), "small")
        self.assertEqual(fleet_bucket(5), "medium")
        self.assertEqual(fleet_bucket(12), "large")
        self.assertEqual(fleet_bucket(24), "very-large")
        self.assertEqual(fleet_bucket(None), "unknown")

    def test_stage_payload_round_trips(self) -> None:
        stage = StageSpec("slice", max_fleet_size=6, families=("balanced",), epochs=2, balance=False)
        restored = StageSpec.from_payload(stage.to_dict())
        self.assertEqual(restored.to_dict(), stage.to_dict())


class BalancedSamplingTestCase(unittest.TestCase):
    def test_minority_strata_get_larger_weights(self) -> None:
        samples = [
            sample(**{"fleet.size": 2, "condition_class": "nominal"})
            for _ in range(6)
        ] + [
            sample(**{"fleet.size": 12, "condition_class": "rare-event"})
            for _ in range(2)
        ]
        weights = balanced_weights(samples)
        self.assertEqual(set(weights), {sample_key(item) for item in samples})
        majority = weights[sample_key(samples[0])]
        minority = weights[sample_key(samples[-1])]
        self.assertGreater(minority, majority)
        self.assertLessEqual(max(weights.values()), 5.0)
        self.assertGreaterEqual(min(weights.values()), 0.2)
        self.assertAlmostEqual(sum(weights.values()) / len(weights), 1.0, places=6)

    def test_weights_are_empty_without_samples(self) -> None:
        self.assertEqual(balanced_weights([]), {})


class CurriculumConfigTestCase(unittest.TestCase):
    def base(self, **overrides) -> CurriculumConfig:
        values = dict(
            curriculum_id="unit-curriculum",
            dataset_root="/tmp/does-not-matter",
            stages=(StageSpec("one"),),
            seeds=(0,),
            epochs_per_stage=1,
        )
        values.update(overrides)
        return CurriculumConfig(**values)

    def test_invalid_configs_are_refused(self) -> None:
        with self.assertRaises(CurriculumError):
            self.base(stages=(StageSpec("one"), StageSpec("one")))
        with self.assertRaises(CurriculumError):
            self.base(seeds=())
        with self.assertRaises(CurriculumError):
            self.base(model={"model": "unknown"})
        with self.assertRaises(CurriculumError):
            self.base(epochs_per_stage=0)
        with self.assertRaises(CurriculumError):
            self.base(selection_metric="train_loss")
        with self.assertRaises(CurriculumError):
            self.base(stall_patience=-1)
        with self.assertRaises(CurriculumError):
            StageSpec("bad", max_fleet_size=-1)

    def test_config_round_trips_through_json(self) -> None:
        config = self.base(
            stages=(StageSpec("small", max_fleet_size=3, balance=False), StageSpec("full")),
            seeds=(0, 1),
            model={"model": "gnn_gru", "hidden": 8, "layers": 1},
        )
        restored = CurriculumConfig.from_payload(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())


class CurriculumRunTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        cls.dataset = campaign(cls.workspace / "campaign")
        cls.config = CurriculumConfig(
            curriculum_id="unit-curriculum",
            dataset_root=str(cls.workspace / "campaign"),
            model={"model": "gnn_gru", "hidden": 8, "layers": 1},
            stages=(
                StageSpec("tiny", max_fleet_size=2, max_targets=1, max_faults=0, max_partitions=0),
                StageSpec("full"),
            ),
            seeds=(0,),
            epochs_per_stage=1,
            batches_per_epoch=1,
            graphs_per_batch=2,
            steps_per_chunk=2,
            max_sequence_samples=40,
            evaluation_episodes=1,
            early_stopping_patience=0,
            stall_patience=0,
        )
        cls.output = cls.workspace / "curriculum"
        cls.manifest = run_curriculum(cls.config, cls.output)

    def test_manifest_is_written_and_versioned(self) -> None:
        path = self.output / "curriculum-manifest.json"
        self.assertTrue(path.exists())
        payload = curriculum_report(self.output)
        self.assertEqual(payload["curriculum_version"], CURRICULUM_VERSION)
        self.assertEqual(payload["curriculum_id"], "unit-curriculum")
        self.assertEqual(payload["config"]["curriculum_id"], "unit-curriculum")
        self.assertEqual(payload["dataset"]["campaign_id"], "curriculum-campaign")
        self.assertTrue(payload["schema"]["parameter_schema_hash"])
        self.assertEqual(payload["selection_rule"].count("frozen"), 1)

    def test_every_stage_records_its_own_slice(self) -> None:
        seed = self.manifest["seeds"][0]
        self.assertEqual(seed["seed"], 0)
        self.assertEqual([stage["stage"] for stage in seed["stages"]], ["tiny", "full"])
        for stage in seed["stages"]:
            self.assertIn(stage["status"], ("completed", "skipped: no samples admitted"))
            self.assertIn("coverage", stage)
            self.assertIn("history", stage)
        completed = [stage for stage in seed["stages"] if stage["status"] == "completed"]
        self.assertTrue(completed)
        if len(completed) > 1:
            self.assertTrue(completed[-1]["initialized_from"])

    def test_selection_is_frozen_on_validation_and_checkpoint_exists(self) -> None:
        seed = self.manifest["seeds"][0]
        selection = seed["selection"]
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertTrue(selection["frozen_before_test"])
        self.assertEqual(selection["metric"], "validation_loss")
        self.assertTrue(Path(selection["checkpoint"]).exists())
        self.assertEqual(seed["stop_reason"], "completed")
        self.assertEqual(seed["stages_completed"], len(
            [stage for stage in seed["stages"] if stage["status"] == "completed"]
        ))

    def test_held_out_evaluation_compares_expert_and_baselines(self) -> None:
        seed = self.manifest["seeds"][0]
        test = seed["evaluation"].get("test")
        self.assertIsNotNone(test)
        assert test is not None
        self.assertIn("learned", test["policies"])
        self.assertIn("expert-rule", test["policies"])
        self.assertIn("wait", test["policies"])
        self.assertIn("first-feasible", test["policies"])
        for entry in test["policies"].values():
            self.assertIn("mean_coverage", entry["aggregate"])
            self.assertIn("mean_reward", entry["aggregate"])
            for family in entry["families"].values():
                self.assertGreater(family["tasks_total"], 0)
                self.assertGreaterEqual(family["coverage_by_tasks"], 0.0)
        # The wait baseline never observes anything; the expert must cover work.
        self.assertEqual(test["policies"]["wait"]["aggregate"]["observed"], 0)
        self.assertGreater(test["policies"]["expert-rule"]["aggregate"]["observed"], 0)

    def test_progression_separates_loss_from_reward_and_shows_coverage_growth(self) -> None:
        progression = self.manifest["progression"]["stages"]
        self.assertTrue(progression)
        stages = self.manifest["seeds"][0]["stages"]
        for entry, stage in zip(progression, stages):
            self.assertEqual(entry["stage"], stage["stage"])
            self.assertIn("train_loss_start", entry)
            self.assertIn("validation_loss", entry)
            self.assertIn("validation_reward", entry)
        growth = self.manifest["seeds"][0]["coverage_growth"]
        self.assertTrue(growth["families_seen"])
        self.assertIn("new_families", growth["stages"][0])
        self.assertIn("supervised loss", self.manifest["progression"]["note"])

    def test_single_seed_has_no_variation_estimate(self) -> None:
        variation = self.manifest["seed_variation"]
        self.assertEqual(variation["seeds"], 1)
        self.assertIn("single seed", variation["note"])

    def test_report_is_machine_readable_and_reproducible(self) -> None:
        path = self.output / "seed-0" / "curriculum-report.json"
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(
            self.manifest["selected_seed"]["checkpoint"],
            self.manifest["seeds"][0]["selection"]["checkpoint"],
        )


class TransferTestCase(unittest.TestCase):
    def test_later_stages_start_from_the_previous_checkpoint(self) -> None:
        workspace = Path(tempfile.mkdtemp())
        campaign(workspace / "campaign", count=10)
        config = CurriculumConfig(
            curriculum_id="transfer",
            dataset_root=str(workspace / "campaign"),
            model={"model": "gnn", "hidden": 8, "layers": 1},
            stages=(StageSpec("first", max_fleet_size=3), StageSpec("second", max_fleet_size=4)),
            seeds=(0,),
            epochs_per_stage=1,
            batches_per_epoch=1,
            graphs_per_batch=2,
            max_sequence_samples=24,
            evaluation_episodes=0,
            evaluate_test=False,
            early_stopping_patience=0,
            stall_patience=0,
        )
        seed = CurriculumRunner(config, workspace / "out").run()["seeds"][0]
        completed = [stage for stage in seed["stages"] if stage["status"] == "completed"]
        self.assertGreaterEqual(len(completed), 1)
        if len(completed) > 1:
            self.assertIsNone(completed[0]["initialized_from"])
            self.assertEqual(completed[1]["initialized_from"], completed[0]["checkpoint"])
            self.assertNotEqual(completed[0]["checkpoint"], completed[1]["checkpoint"])


class SeedVariationTestCase(unittest.TestCase):
    def test_two_seeds_report_mean_and_spread(self) -> None:
        workspace = Path(tempfile.mkdtemp())
        campaign(workspace / "campaign", count=12)
        config = CurriculumConfig(
            curriculum_id="variation",
            dataset_root=str(workspace / "campaign"),
            model={"model": "gnn", "hidden": 8, "layers": 1},
            stages=(StageSpec("full"),),
            seeds=(0, 1),
            epochs_per_stage=1,
            batches_per_epoch=1,
            graphs_per_batch=2,
            max_sequence_samples=24,
            evaluation_episodes=1,
            early_stopping_patience=0,
            stall_patience=0,
        )
        manifest = CurriculumRunner(config, workspace / "out").run()
        variation = manifest["seed_variation"]
        self.assertEqual(variation["seeds"], 2)
        self.assertEqual(variation["metrics"]["selection_value"]["count"], 2)
        for metric in variation["metrics"].values():
            self.assertGreaterEqual(metric["max"], metric["mean"])
            self.assertLessEqual(metric["min"], metric["mean"])
            self.assertGreaterEqual(metric["std"], 0.0)


class BaselineTestCase(unittest.TestCase):
    def test_baselines_are_non_neural_and_explicit(self) -> None:
        policies = baseline_policies()
        self.assertEqual(set(policies), {"expert-rule", "first-feasible", "wait"})
        self.assertIsInstance(policies["wait"], WaitPolicy)
        self.assertIsInstance(policies["first-feasible"], FirstCandidatePolicy)
        self.assertEqual(policies["wait"].metadata()["kind"], "baseline")


if __name__ == "__main__":
    unittest.main()
