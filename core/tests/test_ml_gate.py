"""Days 48/51/56: shield suite, compatibility suite and the G3 promotion gate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from constellab_ml import feature_schema_payload
from constellab_ml.compatibility import (
    ModelManifest,
    Range,
    TrainingEnvelope,
    compatibility_suite,
    feature_schema_hash,
)
from constellab_ml.dataset import DatasetConfig, EpisodeDataset, generate_campaign
from constellab_ml.gate import GateCriteria, GateError, render_gate_report, run_gate
from constellab_ml.shield import shield_suite

from tests.test_ml_scenarios import fast_distribution_set
from tests.test_ml_shield import build_environment


def _checkpoint_payload() -> dict:
    import torch

    return {
        "state_dict": {"weight": torch.tensor([1.0, 2.0, 3.0])},
        "model_config": {"model": "gnn", "hidden": 8, "layers": 1},
        "feature_schema": feature_schema_payload(),
        "observation_version": "swarm-observation-1",
        "env_version": "swarm-env-1",
        "parameter_schema_hash": "schema-test",
    }


class ShieldSuiteTestCase(unittest.TestCase):
    def test_boundary_suite_passes_on_a_real_observation(self) -> None:
        environment = build_environment()
        observation, _ = environment.reset()
        report = shield_suite(observation, source="unit")
        self.assertTrue(report["passed"], report["summary"]["failed"])
        self.assertGreaterEqual(report["summary"]["cases"], 10)
        self.assertFalse(report["summary"]["disable_via_policy_output"])
        names = {case["name"] for case in report["cases"]}
        for required in ("legal-assign", "not-in-mask", "energy-floor", "hostile-policy"):
            self.assertIn(required, names)
        self.assertEqual(report["summary"]["fallback"], "wait")


class CompatibilitySuiteTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        cls.root = cls.workspace / "campaign"
        generate_campaign(
            cls.root,
            DatasetConfig(campaign_id="compat-campaign", seed=17, salt="compat"),
            12,
            distribution_set=fast_distribution_set(),
        )
        cls.dataset = EpisodeDataset(cls.root)
        cls.record = cls.dataset.episodes("test")[0]
        entry = ModelManifest.from_checkpoint(_checkpoint_payload())
        entry.envelope = TrainingEnvelope.from_dataset(cls.root)
        entry.parameter_schema_hash = str(
            (cls.record["spec"].get("parameter_registry") or "")
        )
        cls.manifest = entry

    def test_suite_covers_compatible_incompatible_and_ood(self) -> None:
        report = compatibility_suite(self.manifest, self.root, limit=2)
        self.assertTrue(report["passed"], report["failed"])
        self.assertGreaterEqual(report["counts"].get("incompatible", 0), 2)
        self.assertGreaterEqual(report["counts"].get("out-of-distribution", 0), 1)
        self.assertTrue(report["accepted_envelope"]["ranges"])
        names = {case["name"] for case in report["cases"]}
        self.assertIn("schema-mismatch", names)
        self.assertIn("unsupported-semantics", names)
        self.assertIn("numeric-extrapolation", names)


class GateTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp())
        import torch

        cls.checkpoint = cls.workspace / "checkpoint.pt"
        torch.save(_checkpoint_payload(), cls.checkpoint)
        entry = ModelManifest.from_checkpoint(cls.checkpoint)
        entry.parameter_schema_hash = "schema-test"
        entry.envelope = TrainingEnvelope(
            ranges={"fleet.size": Range(3, 8), "tasks.target_count": Range(1, 6)}
        )
        cls.manifest_path = cls.workspace / "model-manifest.json"
        cls.manifest_path.write_text(json.dumps(entry.to_dict()), encoding="utf-8")
        cls.manifest = entry.to_dict()

    def _evidence(self) -> dict:
        policies = {
            "learned": {
                "aggregate": {
                    "coverage": {"mean": 0.66, "low": 0.5, "high": 0.8},
                    "conflicts": 0,
                    "invalid_actions": 0,
                    "latency_ms": {"mean": 3.0, "p95": 9.0},
                },
                "slices": {"ood": {"coverage": {"mean": 0.42}}},
                "shield": {"proposed": 100, "rejected": 8, "accepted": 92},
            },
            "expert-rule": {"aggregate": {"coverage": {"mean": 0.52}}},
            "first-feasible": {"aggregate": {"coverage": {"mean": 0.44}}},
            "wait": {"aggregate": {"coverage": {"mean": 0.0}}},
        }
        return {
            "generalization": {
                "suite": {"inventory": {"test_episodes": 9}},
                "policies": policies,
            },
            "benchmark": {
                "suite": {"episodes": 9},
                "policies": policies,
            },
            "shield": {
                "passed": True,
                "summary": {
                    "cases": 14,
                    "failed": [],
                    "fallback": "wait",
                    "disable_via_policy_output": False,
                    "reason_codes_covered": ["not-in-mask"],
                    "reason_codes_missing": ["storage-floor"],
                },
            },
            "compatibility": {
                "passed": True,
                "counts": {"compatible": 2, "incompatible": 2, "out-of-distribution": 1},
                "failed": [],
                "accepted_envelope": {"ranges": {"fleet.size": {"min": 3, "max": 8}}},
                "manifest": {
                    "feature_schema_hash": feature_schema_hash(),
                    "parameter_schema_hash": "schema-test",
                },
                "cases": [{"report": {"fallback": "baseline"}}, {"report": {"fallback": "model"}}],
            },
            "ablation": {
                "variants": {
                    "full": {"variant": {"shield": True, "diagnostic": False}, "metrics": {"mean_coverage": 0.6}},
                    "no-shield": {"variant": {"shield": False, "diagnostic": True}},
                    "no-communication": {"variant": {"shield": True, "diagnostic": True}},
                    "no-parameter-context": {"variant": {"shield": True, "diagnostic": True}},
                }
            },
            "rl_curriculum": {
                "seed_reports": [
                    {
                        "seed": 0,
                        "stages": [
                            {"status": "completed", "checkpoint": "a.pt",
                             "selection": {"metric": "validation_mean_coverage", "value": 0.6}}
                        ],
                    },
                    {
                        "seed": 1,
                        "stages": [
                            {"status": "completed", "checkpoint": "b.pt",
                             "selection": {"metric": "validation_mean_coverage", "value": 0.55}}
                        ],
                    },
                ],
                "summary": {"winner": {"seed": 0, "value": 0.6}},
            },
        }

    def _write_bundle(self, evidence: dict, *, mutate=None) -> Path:
        directory = self.workspace / f"bundle-{len(list(self.workspace.glob('bundle-*')))}"
        directory.mkdir()
        paths = {}
        for key, payload in evidence.items():
            path = directory / f"{key}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths[key] = str(path)
        bundle = {
            "gate_id": "unit-g3",
            "candidate": "learned",
            "checkpoint": str(self.checkpoint),
            "manifest": str(self.manifest_path),
            "campaign": "",
            "evidence": paths,
            "criteria": {"min_benchmark_policies": 3},
        }
        if mutate:
            mutate(bundle)
        path = directory / "bundle.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_complete_evidence_passes_everything_except_disabled_replay(self) -> None:
        bundle = json.loads(self._write_bundle(self._evidence()).read_text())
        report = run_gate(bundle, replay=False)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["required_failures"], ["replay"])
        keys = {check["key"]: check for check in report["criteria"]}
        for key in ("artifact", "generalization", "benchmark", "shield", "compatibility",
                    "ablation", "rl_curriculum"):
            self.assertTrue(keys[key]["passed"], (key, keys[key]["reasons"]))
        self.assertEqual(report["fallback_policy"], "expert-rule")

    def test_weak_candidate_is_rejected_and_baseline_named(self) -> None:
        evidence = self._evidence()
        evidence["generalization"]["policies"]["learned"]["aggregate"]["coverage"]["mean"] = 0.1
        bundle = json.loads(self._write_bundle(evidence).read_text())
        report = run_gate(bundle, replay=False)
        keys = {check["key"]: check for check in report["criteria"]}
        self.assertFalse(keys["generalization"]["passed"])
        self.assertEqual(report["fallback_policy"], "expert-rule")
        self.assertIn("learned", report["rejected"])

    def test_tampered_manifest_blocks_the_gate(self) -> None:
        tampered = dict(self.manifest)
        tampered["checkpoint_hash"] = "0" * 64
        path = self.workspace / "bad-manifest.json"
        path.write_text(json.dumps(tampered), encoding="utf-8")
        bundle = json.loads(self._write_bundle(self._evidence()).read_text())
        bundle["manifest"] = str(path)
        report = run_gate(bundle, replay=False)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("checkpoint-hash-mismatch", report["criteria"][0]["reasons"])

    def test_markdown_report_lists_criteria(self) -> None:
        bundle = json.loads(self._write_bundle(self._evidence()).read_text())
        text = render_gate_report(run_gate(bundle, replay=False))
        self.assertIn("G3 gate", text)
        self.assertIn("Held-out and OOD generalization", text)
        self.assertIn("Blocking reasons", text)

    def test_criteria_reject_invalid_values(self) -> None:
        with self.assertRaises(GateError):
            GateCriteria.from_payload({"min_ood_coverage": 1.5})


if __name__ == "__main__":
    unittest.main()
