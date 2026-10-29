"""Day 34 acceptance tests: the G2 gate passes and its report is reproducible."""

import json
import tempfile
import unittest
from pathlib import Path

from constellab_core import g2, trace


class G2GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = g2.run_demo()

    def test_every_gate_check_passes(self) -> None:
        for name, value in self.report["checks"].items():
            self.assertTrue(value, f"G2 check {name!r} did not pass")

    def test_fault_case_reports_a_duplicate_claim_with_ids(self) -> None:
        violation = self.report["first_violation"]
        assert violation is not None
        self.assertEqual(violation["rule"], "duplicate-task-claim")
        self.assertIn("task.hotspot.g2-eq", violation["entities"])
        self.assertGreaterEqual(len(violation["entities"]), 3)

    def test_control_differs_from_fault_and_is_clean(self) -> None:
        self.assertNotEqual(
            self.report["fault"]["trace_hash"], self.report["control"]["trace_hash"],
        )
        self.assertEqual(self.report["control"]["invariants"]["counts"].get("duplicate-task-claim", 0), 0)
        self.assertGreater(self.report["fault"]["metrics"]["conflicts"], 0)
        self.assertEqual(self.report["control"]["metrics"]["conflicts"], 0)

    def test_replay_reproduces_the_trace_through_the_core(self) -> None:
        manifest = trace.ReplayManifest.from_dict(self.report["fault"]["manifest"])
        replay = g2.replay(manifest)
        self.assertTrue(replay["matches"])
        self.assertEqual(replay["trace_hash"], self.report["fault"]["trace_hash"])
        self.assertGreater(len(replay["events"]), 0)

    def test_timeline_consumes_core_positions_and_link_state(self) -> None:
        timeline = self.report["timeline"]
        self.assertGreater(len(timeline), 2)
        for frame in timeline:
            self.assertEqual(set(frame["positions_km"]), set(g2.G2_SATELLITES))
            for position in frame["positions_km"].values():
                self.assertEqual(len(position), 3)
        first = timeline[0]
        self.assertEqual(first["groups"], [list(g2.G2_GROUPS["A"]), list(g2.G2_GROUPS["B"])])
        self.assertEqual(first["active_faults"], ["g2-partition-00"])

    def test_report_is_json_serializable_and_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = g2.write_report(Path(directory) / "g2-acceptance.json")
            payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertTrue(payload["checks"]["g2_pass"])
        self.assertEqual(payload["scenario_id"], g2.G2_SCENARIO_ID)


if __name__ == "__main__":
    unittest.main()
