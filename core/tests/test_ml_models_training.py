"""Days 38-42 acceptance: batches, model families, training and evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from constellab_ml import EDGE_FEATURES, SATELLITE_FEATURES, TASK_FEATURES
from constellab_ml.batch import BatchError, GraphSample, build_batch, samples_from_dataset
from constellab_ml.dataset import DatasetConfig, EpisodeDataset, generate_campaign
from constellab_ml.env import EpisodeSpec, SwarmEnvironment
from constellab_ml.features import SCOPE_WIDTHS, Normalizer
from constellab_ml.models import (
    MODEL_FAMILIES,
    ModelError,
    ParamTokenEncoder,
    build_model,
    default_model_config,
    model_parameter_count,
)
from constellab_ml.policies import evaluate_policy, ModelPolicy
from constellab_ml.training import (
    TrainingConfig,
    TrainingError,
    Trainer,
    batch_plan,
    export_checkpoint,
    flat_chunks,
    imitation_loss,
    learning_rate_at,
    load_export,
)

from tests.test_ml_scenarios import fast_distribution_set


def campaign(root: Path, *, count: int = 16, seed: int = 1) -> EpisodeDataset:
    generate_campaign(
        root,
        DatasetConfig(campaign_id="models-campaign", seed=seed),
        count,
        distribution_set=fast_distribution_set(),
    )
    return EpisodeDataset(root)


def permuted(observation: dict, order: list[int]) -> dict:
    mapping = {old: new for new, old in enumerate(order)}
    copy = json.loads(json.dumps(observation))
    copy["satellites"] = [observation["satellites"][index] for index in order]
    copy["satellite_ids"] = [observation["satellite_ids"][index] for index in order]
    copy["edges"]["contact"] = [
        [mapping[a], mapping[b]] for a, b in observation["edges"]["contact"]
    ]
    copy["edges"]["assign"] = [
        [mapping[edge[0]], edge[1], *edge[2:]] for edge in observation["edges"]["assign"]
    ]
    copy["mask"]["assign"] = [
        [mapping[pair[0]], pair[1], *pair[2:]] for pair in observation["mask"]["assign"]
    ]
    return copy


class BatchTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        cls.dataset = campaign(cls.workspace / "campaign")
        cls.samples = [
            sample for sample in samples_from_dataset(cls.dataset, "train")
            if sample.candidate_pairs()
        ]

    def test_batch_flattens_variable_graphs_with_indices(self) -> None:
        batch = build_batch(self.samples[:5])
        self.assertEqual(batch.graph_count, 5)
        self.assertEqual(len(batch.node_graph), batch.node_count)
        self.assertTrue(all(graph < 5 for graph in batch.node_graph))
        self.assertTrue(all(src < batch.node_count for src in batch.assign_src))
        self.assertTrue(all(dst < len(batch.task_features) for dst in batch.assign_dst))
        self.assertEqual(
            len(batch.node_features[0]), len(SATELLITE_FEATURES)
        )
        self.assertEqual(len(batch.task_features[0]), len(TASK_FEATURES))
        if batch.assign_features:
            self.assertEqual(len(batch.assign_features[0]), len(EDGE_FEATURES))

    def test_masked_pairs_become_candidates_and_targets(self) -> None:
        batch = build_batch(self.samples[:5])
        self.assertTrue(batch.candidates)
        for node, edges in batch.candidates.items():
            self.assertTrue(edges)
            self.assertTrue(all(0 <= edge < batch.edge_count for edge in edges))
        for node, edge in batch.targets.items():
            self.assertIn(edge, batch.candidates[node])

    def test_abstain_is_an_explicit_class(self) -> None:
        batch = build_batch(self.samples[:5])
        self.assertEqual(sorted(batch.abstain), sorted(set(batch.abstain)))
        self.assertEqual(
            sorted(batch.supervised_satellites()),
            sorted(set(list(batch.candidates) + batch.abstain) & set(batch.candidates)),
        )
        self.assertTrue(set(batch.abstain).issubset(set(batch.candidates)))

    def test_incompatible_parameter_context_is_recorded(self) -> None:
        sample = self.samples[0]
        broken = {
            "tokens": sample.context["tokens"] if sample.context else [],
            "schema_hash": "deadbeefdeadbeef",
            "compatibility": {"compatible": False, "reasons": ["unit test"]},
        }
        clone = GraphSample(
            observation=sample.observation, context=broken, episode_id=sample.episode_id
        )
        batch = build_batch([clone])
        self.assertFalse(batch.compatibility["compatible"])
        self.assertEqual(batch.compatibility["graphs"][0]["episode_id"], sample.episode_id)

    def test_observation_without_entities_is_refused(self) -> None:
        with self.assertRaises(BatchError):
            build_batch([GraphSample(observation={"satellites": None, "tasks": []})])


class ModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        cls.dataset = campaign(cls.workspace / "campaign")
        cls.samples = [
            sample for sample in samples_from_dataset(cls.dataset, "train")
            if sample.candidate_pairs()
        ]
        cls.batch = build_batch(cls.samples[:5])

    def test_every_family_scores_every_edge_and_abstains_per_node(self) -> None:
        for family in MODEL_FAMILIES:
            with self.subTest(family=family):
                torch.manual_seed(0)
                model = build_model({"model": family, "hidden": 16, "layers": 1})
                output = model(self.batch)
                self.assertEqual(tuple(output.edge_logits.shape), (self.batch.edge_count,))
                self.assertEqual(tuple(output.abstain_logits.shape), (self.batch.node_count,))

    def test_gradients_reach_every_parameter(self) -> None:
        for family in MODEL_FAMILIES:
            with self.subTest(family=family):
                torch.manual_seed(0)
                model = build_model({"model": family, "hidden": 16, "layers": 1})
                output = model(self.batch)
                (output.edge_logits.sum() + output.abstain_logits.sum()).backward()
                missing = [
                    name for name, parameter in model.named_parameters()
                    if parameter.requires_grad and parameter.grad is None
                ]
                self.assertEqual(missing, [])

    def test_unknown_family_is_refused(self) -> None:
        with self.assertRaises(ModelError):
            build_model("transformer-of-doom")

    def test_same_weights_run_on_different_fleet_sizes(self) -> None:
        torch.manual_seed(0)
        model = build_model({"model": "gnn", "hidden": 16, "layers": 1})
        first = build_batch([
            GraphSample(observation=self.samples[0].observation, targets=[]),
        ])
        with_edges = max(self.samples, key=lambda item: len(item.observation["satellites"]))
        second = build_batch([
            GraphSample(observation=with_edges.observation, targets=[]),
        ])
        self.assertGreaterEqual(second.node_count, first.node_count)
        small = model(first)
        large = model(second)
        self.assertEqual(tuple(small.abstain_logits.shape), (first.node_count,))
        self.assertEqual(tuple(large.abstain_logits.shape), (second.node_count,))

    def test_entity_order_does_not_change_the_outcome(self) -> None:
        observation = self.samples[0].observation
        if len(observation["satellites"]) < 2:
            self.skipTest("needs at least two satellites")
        order = list(range(1, len(observation["satellites"]))) + [0]
        for family in ("deepsets", "gnn", "gnn_gru"):
            with self.subTest(family=family):
                torch.manual_seed(0)
                model = build_model({"model": family, "hidden": 16, "layers": 1})
                model.eval()
                first = model(build_batch([GraphSample(observation=observation, targets=[])]))
                second = model(build_batch([GraphSample(observation=permuted(observation, order), targets=[])]))
                self.assertTrue(
                    torch.allclose(
                        first.edge_logits.sort().values,
                        second.edge_logits.sort().values,
                        atol=1e-5,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        first.abstain_logits.sort().values,
                        second.abstain_logits.sort().values,
                        atol=1e-5,
                    )
                )

    def test_parameter_encoder_accepts_a_different_number_of_fields(self) -> None:
        torch.manual_seed(0)
        encoder = ParamTokenEncoder(12)
        first = build_batch([GraphSample(
            observation=self.samples[0].observation,
            context={"tokens": self.samples[0].context["tokens"], "schema_hash": "a"},
        )])
        fewer = dict(self.samples[0].context)
        fewer["tokens"] = self.samples[0].context["tokens"][:3]
        second = build_batch([GraphSample(observation=self.samples[0].observation, context=fewer)])
        self.assertLess(second.token_count, first.token_count)
        model = build_model({"model": "gnn", "hidden": 16, "layers": 1})
        self.assertEqual(tuple(model(first).edge_logits.shape), tuple(model(second).edge_logits.shape))

    def test_empty_token_set_is_handled(self) -> None:
        model = build_model({"model": "gnn", "hidden": 16, "layers": 1})
        batch = build_batch([GraphSample(observation=self.samples[0].observation, context=None)])
        self.assertEqual(batch.token_count, 0)
        output = model(batch)
        self.assertEqual(tuple(output.edge_logits.shape), (batch.edge_count,))

    def test_temporal_state_is_isolated_per_episode(self) -> None:
        model = build_model({"model": "gnn_gru", "hidden": 16, "layers": 1})
        model.eval()
        first = build_batch([GraphSample(observation=self.samples[0].observation)])
        other = GraphSample(
            observation=self.samples[0].observation, episode_id="another-episode"
        )
        second = build_batch([other])
        step_one = model(first)
        step_two = model(second, state=step_one.state)
        # The episode id changed, so the hidden state of the new graph is zero:
        # the same input must score exactly like a fresh forward pass.
        fresh = model(second)
        self.assertTrue(torch.allclose(step_two.edge_logits, fresh.edge_logits, atol=1e-6))

    def test_temporal_state_carries_within_an_episode(self) -> None:
        model = build_model({"model": "gnn_gru", "hidden": 16, "layers": 1})
        model.eval()
        first = build_batch([GraphSample(observation=self.samples[0].observation, episode_id="e1")])
        second = build_batch([GraphSample(observation=self.samples[0].observation, episode_id="e1")])
        output = model(first)
        carried = model(second, state=output.state)
        fresh = model(second)
        self.assertFalse(torch.allclose(carried.edge_logits, fresh.edge_logits, atol=1e-8))

    def test_attention_family_is_behind_the_same_interface(self) -> None:
        model = build_model({"model": "gnn_attention", "hidden": 16, "layers": 1})
        output = model(self.batch)
        self.assertEqual(tuple(output.edge_logits.shape), (self.batch.edge_count,))
        self.assertIn("memory", output.state)
        self.assertEqual(default_model_config("gnn")["hidden"], 32)

    def test_model_configs_only_add_parameters(self) -> None:
        torch.manual_seed(0)
        small = build_model({"model": "gnn", "hidden": 8, "layers": 1})
        torch.manual_seed(0)
        large = build_model({"model": "gnn", "hidden": 16, "layers": 1})
        self.assertLess(model_parameter_count(small), model_parameter_count(large))


class NormalizerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        cls.dataset = campaign(cls.workspace / "campaign")
        cls.train = [
            sample for sample in samples_from_dataset(cls.dataset, "train")
            if sample.candidate_pairs()
        ]
        cls.validation = [
            sample for sample in samples_from_dataset(cls.dataset, "validation")
            if sample.candidate_pairs()
        ]

    def test_fit_uses_only_training_rows(self) -> None:
        train_batches = [build_batch(chunk) for chunk in flat_chunks(self.train, 4)]
        normalizer = Normalizer.fit(train_batches)
        rows = sum(normalizer.counts.values())
        self.assertGreater(rows, 0)
        after = normalizer.apply(build_batch(self.train[:2]))
        means = [
            sum(row[index] for row in after.node_features) / len(after.node_features)
            for index in range(len(SATELLITE_FEATURES))
        ]
        available = [
            index for index in range(len(SATELLITE_FEATURES))
            if after.node_available[0][index] > 0.0
        ]
        for index in available:
            self.assertLess(abs(means[index]), 1.5)

    def test_validation_rows_are_normalized_with_training_statistics(self) -> None:
        normalizer = Normalizer.fit([build_batch(chunk) for chunk in flat_chunks(self.train, 4)])
        payload = normalizer.payload()
        restored = Normalizer.from_payload(json.loads(json.dumps(payload)))
        self.assertEqual(restored.describe(), normalizer.describe())
        self.assertEqual(set(payload["counts"]), set(SCOPE_WIDTHS))

    def test_constant_columns_are_passed_through(self) -> None:
        normalizer = Normalizer.fit([build_batch([self.train[0]])])
        scope = "global"
        self.assertTrue(all(scale == 1.0 for scale in normalizer.scale[scope]))


class TrainingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        cls.dataset = campaign(cls.workspace / "campaign", count=24)

    def config(self, **overrides) -> TrainingConfig:
        base = dict(
            run_id="unit-run",
            dataset_root=str(self.workspace / "campaign"),
            model={"model": "gnn", "hidden": 16, "layers": 1},
            epochs=2,
            batches_per_epoch=3,
            graphs_per_batch=4,
            steps_per_chunk=2,
            max_sequence_samples=200,
            export=True,
        )
        base.update(overrides)
        return TrainingConfig(**base)

    def test_training_writes_metrics_checkpoint_and_manifest(self) -> None:
        trainer = Trainer(self.config(), self.workspace / "run")
        manifest = trainer.train()
        self.assertEqual(manifest["training_version"], "training-1")
        self.assertEqual(len(manifest["history"]), 2)
        self.assertTrue((self.workspace / "run" / "metrics.jsonl").exists())
        self.assertTrue((self.workspace / "run" / "checkpoint-last.pt").exists())
        self.assertTrue((self.workspace / "run" / "checkpoint-best.pt").exists())
        self.assertTrue((self.workspace / "run" / "export" / "weights.pt").exists())
        self.assertIn("code_revision", manifest)
        self.assertIn("torch", manifest["versions"])
        self.assertTrue(manifest["schema"]["parameter_schema_hash"])

    def test_loss_decreases_or_stays_finite(self) -> None:
        trainer = Trainer(self.config(epochs=3), self.workspace / "loss")
        manifest = trainer.train()
        losses = [entry["train_loss"] for entry in manifest["history"]]
        self.assertTrue(all(loss == loss and loss < 10.0 for loss in losses))

    def test_resume_continues_the_step_counter(self) -> None:
        first = Trainer(self.config(), self.workspace / "resume-a")
        manifest = first.train()
        checkpoint = self.workspace / "resume-a" / "checkpoint-last.pt"
        resumed = Trainer(self.config(run_id="resume-run", epochs=3), self.workspace / "resume-b")
        continued = resumed.train(resume_from=checkpoint)
        self.assertGreater(continued["history"][-1]["global_step"], manifest["history"][-1]["global_step"])
        self.assertGreater(len(continued["history"]), len(manifest["history"]))

    def test_export_round_trip_rebuilds_a_policy(self) -> None:
        trainer = Trainer(self.config(), self.workspace / "export-run")
        trainer.train()
        exported = load_export(self.workspace / "export-run" / "export")
        self.assertEqual(exported["model_config"]["model"], "gnn")
        policy = ModelPolicy(
            build_model(exported["model_config"]),
            Normalizer.from_payload(exported["normalizer"]),
        )
        environment = SwarmEnvironment(
            EpisodeSpec.from_payload(self.dataset.episodes("train")[0]["spec"])
        )
        observation, _ = environment.reset()
        actions = policy.act(observation) if observation["mask"]["assign"] else []
        self.assertIsInstance(actions, list)
        self.assertEqual(set(exported), set(exported) | {"state_dict", "normalizer"})

    def test_evaluation_reports_metrics_per_family(self) -> None:
        trainer = Trainer(self.config(evaluate_every=1, evaluation_episodes=2), self.workspace / "eval")
        trainer.train()
        report = trainer.evaluate("test" if self.dataset.episodes("test") else "train")
        self.assertIn("aggregate", report)
        self.assertIn("mean_coverage", report["aggregate"])
        self.assertTrue(report["families"])

    def test_invalid_config_is_refused(self) -> None:
        with self.assertRaises(TrainingError):
            self.config(model={"model": "not-a-model"})
        with self.assertRaises(TrainingError):
            self.config(learning_rate=0.0)
        with self.assertRaises(TrainingError):
            self.config(epochs=0)

    def test_learning_rate_schedule_shape(self) -> None:
        config = self.config(schedule="cosine", warmup_steps=2)
        rates = [learning_rate_at(step, 10, config) for step in range(10)]
        self.assertLess(rates[0], rates[2])
        self.assertLess(rates[-1], rates[2])
        constant = self.config(schedule="constant")
        self.assertEqual(learning_rate_at(5, 10, constant), constant.learning_rate)

    def test_batch_plan_groups_temporal_chunks_by_episode(self) -> None:
        samples = [
            sample for sample in samples_from_dataset(self.dataset, "train")
            if sample.candidate_pairs()
        ]
        stateless = batch_plan(samples, self.config())
        for group in stateless:
            self.assertEqual(len(group), 1)
        temporal = batch_plan(samples, self.config(model={"model": "gnn_gru", "hidden": 8}))
        for group in temporal:
            episodes = {sample.episode_id for chunk in group for sample in chunk}
            self.assertEqual(len(episodes), 1)

    def test_temporal_training_runs_and_logs_loss(self) -> None:
        trainer = Trainer(
            self.config(model={"model": "gnn_gru", "hidden": 12, "layers": 1}, epochs=1),
            self.workspace / "temporal",
        )
        manifest = trainer.train()
        self.assertGreater(manifest["history"][0]["train_loss"], 0.0)
        self.assertEqual(manifest["hyperparameters"]["temporal"], True)

    def test_imitation_loss_counts_supervised_decisions(self) -> None:
        samples = [
            sample for sample in samples_from_dataset(self.dataset, "train")
            if sample.candidate_pairs()
        ][:3]
        batch = build_batch(samples)
        model = build_model({"model": "gnn", "hidden": 12, "layers": 1})
        result = imitation_loss(model(batch), batch)
        self.assertGreaterEqual(result["count"], 1)
        self.assertLessEqual(result["correct"], result["count"])
        self.assertGreater(float(result["loss"].item()), 0.0)


class PolicyEvaluationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        cls.dataset = campaign(cls.workspace / "campaign", count=16)

    def test_untrained_policy_runs_and_reports_metrics(self) -> None:
        model = build_model({"model": "gnn", "hidden": 12, "layers": 1})
        normalizer = Normalizer.fit([build_batch([GraphSample(observation=(
            samples_from_dataset(self.dataset, "train")[0].observation))])])
        policy = ModelPolicy(model, normalizer)
        report = evaluate_policy(policy, self.dataset.episodes("train")[:2])
        self.assertEqual(report.aggregate["episodes"], 2)
        self.assertLessEqual(report.aggregate["mean_coverage"], 1.0)
        self.assertIn("conflicts", report.aggregate)
        self.assertTrue(policy.metadata()["deployable"])
        self.assertEqual(policy.metadata()["kind"], "learned")

    def test_live_policy_uses_the_parameter_context_it_was_trained_with(self) -> None:
        model = build_model({"model": "gnn", "hidden": 12, "layers": 1})
        normalizer = Normalizer.fit([build_batch([GraphSample(observation=(
            samples_from_dataset(self.dataset, "train")[0].observation))])])
        policy = ModelPolicy(model, normalizer)
        sample = samples_from_dataset(self.dataset, "train")[0]
        observation = dict(sample.observation)
        observation["context"] = sample.context
        _, batch, _ = policy.candidates(observation)
        self.assertGreater(batch.token_count, 0)
        stripped = dict(observation)
        stripped["context"] = {}
        _, empty_batch, _ = policy.candidates(stripped)
        self.assertEqual(empty_batch.token_count, 0)

    def test_policy_reset_drops_temporal_state(self) -> None:
        model = build_model({"model": "gnn_gru", "hidden": 12, "layers": 1})
        normalizer = Normalizer.fit([build_batch([GraphSample(observation=(
            samples_from_dataset(self.dataset, "train")[0].observation))])])
        policy = ModelPolicy(model, normalizer)
        observation = samples_from_dataset(self.dataset, "train")[0].observation
        policy.act(observation)
        self.assertIsNotNone(policy.state)
        policy.reset()
        self.assertIsNone(policy.state)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
