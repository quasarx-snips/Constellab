"""Day 32 acceptance tests: seed streams, canonical traces and manifests."""

import unittest

from constellab_core import trace
from constellab_core.trace import ReplayManifest, SeedBank, TraceRecorder


class SeedBankTests(unittest.TestCase):
    def test_derivation_is_deterministic_and_independent(self) -> None:
        first = SeedBank(42)
        second = SeedBank(42)
        self.assertEqual(first.derive("faults"), second.derive("faults"))
        self.assertEqual(first.derive("faults"), first.derive("faults"))
        self.assertNotEqual(first.derive("faults"), first.derive("comms"))
        self.assertNotEqual(first.derive("faults"), SeedBank(43).derive("faults"))

    def test_entity_streams_are_scoped_by_id(self) -> None:
        bank = SeedBank(7)
        self.assertNotEqual(bank.entity("comms", "sat-a"), bank.entity("comms", "sat-b"))
        self.assertNotEqual(bank.entity("comms", "sat-a"), bank.entity("faults", "sat-a"))

    def test_adding_a_stream_does_not_change_existing_streams(self) -> None:
        bank = SeedBank(3)
        before = bank.derive("dynamics")
        bank.derive("brand-new-subsystem")
        self.assertEqual(bank.derive("dynamics"), before)

    def test_payload_reports_the_master_seed(self) -> None:
        payload = SeedBank(5).payload()
        self.assertEqual(payload["master_seed"], 5)
        self.assertEqual(payload["version"], trace.SEED_VERSION)


class TraceRecorderTests(unittest.TestCase):
    def test_events_are_ordered_by_time_then_sequence(self) -> None:
        recorder = TraceRecorder()
        recorder.record("b", 10.0)
        recorder.record("a", 5.0)
        recorder.record("c", 10.0)
        self.assertEqual(
            [(event.kind, event.sequence) for event in recorder.events()],
            [("a", 2), ("b", 1), ("c", 3)],
        )

    def test_hash_is_stable_across_labels_and_instances(self) -> None:
        first = TraceRecorder(label="g2:fault")
        second = TraceRecorder(label="replay:something-else")
        for recorder in (first, second):
            recorder.record("task.assigned", 0.0, {"task_id": "task.1", "satellite_id": "a"})
            recorder.record("task.observed", 30.0, {"accepted": True})
        self.assertEqual(first.hash(), second.hash())
        self.assertEqual(len(first.hash()), 64)

    def test_hash_changes_when_a_meaningful_event_changes(self) -> None:
        first = TraceRecorder()
        second = TraceRecorder()
        first.record("task.assigned", 0.0, {"satellite_id": "a"})
        second.record("task.assigned", 0.0, {"satellite_id": "b"})
        self.assertNotEqual(first.hash(), second.hash())

    def test_kinds_and_payload_counts(self) -> None:
        recorder = TraceRecorder()
        recorder.record("x", 0.0)
        recorder.record("x", 1.0)
        recorder.record("y", 2.0)
        self.assertEqual(recorder.kinds(), {"x": 2, "y": 1})
        self.assertEqual(recorder.payload()["count"], 3)

    def test_diff_reports_the_first_divergent_event(self) -> None:
        left = TraceRecorder()
        right = TraceRecorder()
        for recorder in (left, right):
            recorder.record("same", 0.0, {"value": 1})
        left.record("only-left", 10.0, {})
        right.record("only-right", 10.0, {})
        difference = trace.diff_traces(
            [event.to_dict() for event in left.events()],
            [event.to_dict() for event in right.events()],
        )
        assert difference is not None
        self.assertEqual(difference["index"], 1)
        self.assertEqual(difference["time_s"], 10.0)
        self.assertIsNone(trace.diff_traces([], []))


class ReplayManifestTests(unittest.TestCase):
    def test_manifest_round_trips_and_hashes_stably(self) -> None:
        manifest = ReplayManifest(
            scenario={"scenario_id": "s", "satellites": []},
            policy_key="distributed-greedy",
            seed=7,
            faults={"faults": []},
            expected_trace_hash="abc",
            horizon_s=1_800.0,
            step_s=30.0,
            versions={"g2": "1"},
            label="g2-fault-on",
        )
        restored = ReplayManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored.to_dict(), manifest.to_dict())
        self.assertEqual(restored.hash(), manifest.hash())


if __name__ == "__main__":
    unittest.main()
