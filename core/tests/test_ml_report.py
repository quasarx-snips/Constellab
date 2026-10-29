"""Days 46–56: the read-only ML Lab evidence payload."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from constellab_core.ml import ARTIFACTS, ml_report


class MlReportTestCase(unittest.TestCase):
    def test_missing_artefacts_are_unavailable_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            report = ml_report(workspace)
        self.assertEqual(report["version"], "ml-report-1")
        self.assertIsNone(report["gate"])
        self.assertIsNone(report["shield"])
        self.assertIsNone(report["pipeline"]["manifest"])
        self.assertIsNone(report["pipeline"]["curriculum"])
        self.assertIsNone(report["pipeline"]["baselines"])
        for key in ARTIFACTS:
            self.assertFalse(report["artifacts"][key]["available"])
            self.assertIn("error", report["artifacts"][key])

    def test_checked_in_reports_are_compacted_for_the_ui(self) -> None:
        report = ml_report()
        self.assertTrue(report["artifacts"]["gate"]["available"])
        gate = report["gate"]
        self.assertIn(gate["status"], {"pass", "fail", "blocked"})
        self.assertTrue(gate["criteria"])
        self.assertTrue(gate["accepted_envelope"].get("ranges"))
        self.assertTrue(report["generalization"]["policies"])
        self.assertIn("coverage", report["generalization"]["policies"][0])
        self.assertNotIn("episodes", report["generalization"])
        self.assertTrue(report["benchmark"]["policies"])
        self.assertTrue(report["shield"]["cases"])
        self.assertTrue(report["compatibility"]["cases"])
        self.assertTrue(report["ablations"]["variants"])
        self.assertTrue(report["training"]["rl"]["seeds"])
        self.assertIn("validation_auc", report["training"]["ranker"]["metrics"])
        pipeline = report["pipeline"]
        self.assertIsNotNone(pipeline["manifest"])
        self.assertEqual(pipeline["manifest"]["model_family"], "gnn_gru")
        self.assertIsNotNone(pipeline["curriculum"])
        self.assertTrue(pipeline["curriculum"]["seeds"])
        self.assertIsNotNone(pipeline["baselines"])
        self.assertTrue(pipeline["baselines"]["policies"])

    def test_malformed_json_is_reported_as_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            for key, relative in ARTIFACTS.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{not json", encoding="utf-8")
            report = ml_report(root)
        self.assertFalse(report["artifacts"]["gate"]["available"])
        self.assertIn("error", report["artifacts"]["gate"])
        self.assertIsNone(report["gate"])


if __name__ == "__main__":
    unittest.main()
