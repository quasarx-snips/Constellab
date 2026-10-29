"""Day 51: manifests, schema/hash checks, OOD envelope and safe fallback."""

from __future__ import annotations

import unittest

from constellab_ml import feature_schema_payload
from constellab_ml.compatibility import (
    ModelManifest,
    Range,
    TrainingEnvelope,
    checkpoint_hash,
    choose_policy,
    compatibility_report,
    feature_schema_hash,
)
from constellab_ml.env import EpisodeSpec


def spec(fleet: int = 4, targets: int = 2, compatible: bool = True) -> EpisodeSpec:
    satellites = [{"id": f"sat-{index}"} for index in range(fleet)]
    return EpisodeSpec(
        episode_id="episode",
        config={"satellites": satellites},
        targets=[{"id": f"target-{index}"} for index in range(targets)],
        parameters={
            "compatibility": {"compatible": compatible, "reasons": [] if compatible else ["unknown owner"]},
            "tokens": [
                {"field": "network.partitions", "value": 0},
                {"field": "energy.initial_charge_ratio", "value": 0.9},
            ],
        },
        parameter_registry="schema-1",
    )


def manifest() -> ModelManifest:
    return ModelManifest(
        checkpoint="checkpoint.pt",
        checkpoint_hash="abc",
        model_family="gnn",
        feature_schema_hash=feature_schema_hash(),
        parameter_schema_hash="schema-1",
        envelope=TrainingEnvelope(
            ranges={
                "fleet.size": Range(2, 8),
                "tasks.target_count": Range(1, 5),
            },
            families=["nominal", "faulted"],
        ),
    )


class CompatibilityTestCase(unittest.TestCase):
    def test_no_manifest_is_usable_but_unverified(self) -> None:
        report = compatibility_report(None, spec())
        self.assertTrue(report.usable)
        self.assertIn("unverified", report.warnings[0])

    def test_inside_envelope_is_compatible(self) -> None:
        report = compatibility_report(manifest(), spec())
        self.assertEqual(report.status, "compatible")
        self.assertEqual(report.fallback, "model")
        self.assertEqual(report.ood_score, 0.0)

    def test_fleet_outside_envelope_is_ood_and_falls_back(self) -> None:
        report = compatibility_report(manifest(), spec(fleet=20, targets=2))
        self.assertEqual(report.status, "out-of-distribution")
        self.assertEqual(report.fallback, "baseline")
        self.assertGreater(report.ood_score, 0.0)

    def test_unknown_parameter_semantics_are_incompatible(self) -> None:
        report = compatibility_report(manifest(), spec(compatible=False))
        self.assertEqual(report.status, "incompatible")
        self.assertIn("unsupported-parameter-semantics", report.reasons)

    def test_schema_hash_mismatch_is_incompatible(self) -> None:
        entry = manifest()
        entry.feature_schema_hash = "different"
        report = compatibility_report(entry, spec())
        self.assertIn("feature-schema-mismatch", report.reasons)

    def test_checkpoint_hash_is_stable_and_manifest_verifies(self) -> None:
        payload = {
            "state_dict": {"weight": [1.0, 2.0]},
            "model_config": {"model": "gnn"},
            "feature_schema": feature_schema_payload(),
            "observation_version": "swarm-observation-1",
            "env_version": "swarm-env-1",
            "parameter_schema_hash": "schema-1",
        }
        digest = checkpoint_hash(payload)
        self.assertEqual(digest, checkpoint_hash(payload))
        entry = ModelManifest.from_checkpoint(payload)
        self.assertEqual(entry.verify_checkpoint(payload), [])
        payload["state_dict"]["weight"] = [9.0]
        self.assertIn("checkpoint-hash-mismatch", entry.verify_checkpoint(payload))

    def test_choose_policy_uses_expert_outside_the_envelope(self) -> None:
        models = {"learned": object()}
        baselines = {"expert-rule": object()}
        key, selected, report = choose_policy(models, baselines, manifest(), spec(fleet=30))
        self.assertEqual(key, "expert-rule")
        self.assertIs(selected, baselines["expert-rule"])
        self.assertEqual(report.fallback, "baseline")

    def test_choose_policy_uses_model_inside_the_envelope(self) -> None:
        models = {"learned": object()}
        baselines = {"expert-rule": object()}
        key, selected, _ = choose_policy(models, baselines, manifest(), spec())
        self.assertEqual(key, "learned")
        self.assertIs(selected, models["learned"])

    def test_report_and_manifest_are_json_safe(self) -> None:
        self.assertIn("envelope", compatibility_report(manifest(), spec()).to_dict())
        payload = manifest().to_dict()
        self.assertEqual(payload["version"], "model-manifest-1")
        self.assertIn("ranges", payload["envelope"])


if __name__ == "__main__":
    unittest.main()
