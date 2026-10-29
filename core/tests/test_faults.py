"""Day 31 acceptance tests: fault definitions, scope and generation."""

import unittest

from constellab_core import faults
from constellab_core.faults import Fault, FaultError, FaultPlan


class FaultValidationTests(unittest.TestCase):
    def test_a_fault_needs_a_positive_window_and_a_target(self) -> None:
        with self.assertRaises(FaultError):
            Fault(id="f", kind="satellite-failure", start_s=10.0, end_s=5.0, satellites=("a",))
        with self.assertRaises(FaultError):
            Fault(id="f", kind="satellite-failure", start_s=0.0, end_s=5.0)
        with self.assertRaises(FaultError):
            Fault(id="f", kind="warp-drive", start_s=0.0, end_s=5.0, satellites=("a",))

    def test_link_faults_need_peers_on_the_other_side(self) -> None:
        with self.assertRaises(FaultError):
            Fault(id="f", kind="link-cut", start_s=0.0, end_s=5.0, satellites=("a",))
        with self.assertRaises(FaultError):
            Fault(
                id="f", kind="link-cut", start_s=0.0, end_s=5.0,
                satellites=("a",), peers=("a",),
            )

    def test_plan_rejects_duplicate_ids(self) -> None:
        one = Fault(id="dup", kind="satellite-failure", start_s=0.0, end_s=5.0, satellites=("a",))
        with self.assertRaises(FaultError):
            FaultPlan((one, one))


class FaultQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = FaultPlan.manual((
            Fault(id="fail-a", kind="satellite-failure", start_s=100.0, end_s=400.0, satellites=("a",)),
            Fault(id="blind-b", kind="sensor-degrade", start_s=50.0, end_s=150.0, satellites=("b",), quality=0.25),
            Fault(id="cut", kind="link-cut", start_s=0.0, end_s=1_000.0, satellites=("a",), peers=("c", "d")),
            Fault(id="slow", kind="message-delay", start_s=0.0, end_s=1_000.0, satellites=("c",), peers=("d",), delay_s=45.0),
        ))

    def test_windows_are_half_open_and_scoped(self) -> None:
        self.assertTrue(self.plan.satellite_available("a", 99.0))
        self.assertFalse(self.plan.satellite_available("a", 100.0))
        self.assertFalse(self.plan.satellite_available("a", 399.999))
        self.assertTrue(self.plan.satellite_available("a", 400.0))
        self.assertTrue(self.plan.satellite_available("b", 200.0))

    def test_sensor_quality_is_the_lowest_active_factor(self) -> None:
        self.assertEqual(self.plan.sensor_quality_factor("b", 75.0), 0.25)
        self.assertEqual(self.plan.sensor_quality_factor("b", 200.0), 1.0)

    def test_links_are_blocked_only_between_the_two_sides(self) -> None:
        self.assertTrue(self.plan.link_blocked("a", "c", 10.0))
        self.assertTrue(self.plan.link_blocked("d", "a", 10.0))
        self.assertFalse(self.plan.link_blocked("c", "d", 10.0))
        self.assertFalse(self.plan.link_blocked("a", "b", 10.0))
        self.assertFalse(self.plan.link_blocked("a", "c", 1_000.0))

    def test_message_delay_applies_to_the_configured_pair_only(self) -> None:
        self.assertEqual(self.plan.message_delay("c", "d", 10.0), 45.0)
        self.assertEqual(self.plan.message_delay("d", "c", 10.0), 45.0)
        self.assertEqual(self.plan.message_delay("a", "b", 10.0), 0.0)

    def test_active_faults_and_impact_times_are_ordered(self) -> None:
        self.assertEqual(
            [item.id for item in self.plan.active_at(120.0)],
            ["cut", "slow", "blind-b", "fail-a"],
        )
        self.assertEqual(self.plan.impact_times(), (0.0, 50.0, 100.0, 150.0, 400.0, 1_000.0))

    def test_plan_round_trips_through_dict(self) -> None:
        restored = FaultPlan.from_dict(self.plan.to_dict())
        self.assertEqual(restored.canonical(), self.plan.canonical())
        self.assertEqual(restored.hash(), self.plan.hash())

    def test_no_fault_plan_is_an_explicit_control(self) -> None:
        control = faults.empty_plan()
        self.assertEqual(control.faults, ())
        self.assertTrue(control.satellite_available("a", 0.0))
        self.assertFalse(control.link_blocked("a", "b", 0.0))


class FaultGeneratorTests(unittest.TestCase):
    def test_random_plan_is_deterministic_per_seed(self) -> None:
        ids = [f"sat-{index}" for index in range(6)]
        first = faults.random_plan(ids, horizon_s=3_600.0, seed=11, count=4)
        second = faults.random_plan(ids, horizon_s=3_600.0, seed=11, count=4)
        other = faults.random_plan(ids, horizon_s=3_600.0, seed=12, count=4)
        self.assertEqual(first.canonical(), second.canonical())
        self.assertNotEqual(first.canonical(), other.canonical())
        for fault in first.faults:
            self.assertGreaterEqual(fault.start_s, 0.0)
            self.assertLessEqual(fault.end_s, 3_600.0)
            self.assertTrue(set(fault.satellites) <= set(ids))

    def test_worst_moment_partitions_the_first_overlapping_pair(self) -> None:
        from constellab_core import g2

        _config, scene = g2.build_scene()
        plan = faults.worst_moment_plan(scene, duration_s=300.0, seed=5)
        self.assertEqual(plan.strategy, "worst-moment")
        fault = plan.faults[0]
        self.assertEqual(fault.kind, "link-cut")
        self.assertEqual(len(fault.satellites), 1)
        self.assertEqual(len(fault.peers), 1)
        self.assertNotEqual(fault.satellites, fault.peers)
        self.assertIn(fault.satellites[0], g2.G2_SATELLITES)
        self.assertIn(fault.peers[0], g2.G2_SATELLITES)
        self.assertIn("target_id", plan.generator)


if __name__ == "__main__":
    unittest.main()
