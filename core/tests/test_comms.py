"""Day 31 acceptance tests: partition-aware claim visibility."""

import unittest

from constellab_core import faults
from constellab_core.comms import ClaimBus


def partition(*, start_s: float = 0.0, end_s: float = 1_000.0) -> faults.FaultPlan:
    return faults.FaultPlan.manual((
        faults.Fault(
            id="cut", kind="link-cut", start_s=start_s, end_s=end_s,
            satellites=("a", "b"), peers=("c", "d"),
        ),
    ))


class ClaimVisibilityTests(unittest.TestCase):
    def test_a_healthy_bus_delivers_every_claim(self) -> None:
        bus = ClaimBus(["a", "b", "c"])
        bus.record_claim("task.1", "a", time_s=0.0, target_id="hotspot.1")
        for viewer in ("b", "c", "a"):
            visible = bus.visible_claims(viewer, 0.0)
            self.assertEqual([claim["task_id"] for claim in visible], ["task.1"])
        self.assertEqual(bus.topology(0.0)["groups"], [["a", "b", "c"]])

    def test_a_partition_hides_claims_from_the_other_side(self) -> None:
        bus = ClaimBus(["a", "b", "c", "d"], faults=partition())
        bus.record_claim("task.1", "a", time_s=0.0, target_id="hotspot.1")
        self.assertEqual(len(bus.visible_claims("b", 0.0)), 1)
        self.assertEqual(bus.visible_claims("c", 0.0), [])
        self.assertEqual(bus.topology(0.0)["groups"], [["a", "b"], ["c", "d"]])

    def test_claim_delivery_waits_for_the_configured_delay(self) -> None:
        plan = faults.FaultPlan.manual((
            faults.Fault(
                id="slow", kind="message-delay", start_s=0.0, end_s=1_000.0,
                satellites=("a",), peers=("b",), delay_s=60.0,
            ),
        ))
        bus = ClaimBus(["a", "b"], faults=plan)
        bus.record_claim("task.1", "a", time_s=0.0)
        self.assertEqual(bus.visible_claims("b", 59.9), [])
        self.assertEqual(len(bus.visible_claims("b", 60.0)), 1)

    def test_claims_from_a_failed_satellite_are_not_visible(self) -> None:
        plan = faults.FaultPlan.manual((
            faults.Fault(
                id="fail", kind="satellite-failure", start_s=100.0, end_s=200.0,
                satellites=("a",),
            ),
        ))
        bus = ClaimBus(["a", "b"], faults=plan)
        bus.record_claim("task.1", "a", time_s=0.0)
        self.assertEqual(len(bus.visible_claims("b", 50.0)), 1)
        self.assertEqual(bus.visible_claims("b", 150.0), [])

    def test_a_claim_expires_after_the_ttl(self) -> None:
        bus = ClaimBus(["a", "b"], claim_ttl_s=120.0)
        bus.record_claim("task.1", "a", time_s=0.0)
        self.assertEqual(len(bus.visible_claims("b", 119.0)), 1)
        self.assertEqual(bus.visible_claims("b", 121.0), [])

    def test_unknown_viewers_and_sources_are_rejected_cleanly(self) -> None:
        bus = ClaimBus(["a", "b"])
        self.assertEqual(bus.visible_claims("ghost", 0.0), [])
        with self.assertRaises(ValueError):
            bus.record_claim("task.1", "ghost", time_s=0.0)

    def test_bus_hash_is_order_independent_for_the_same_claims(self) -> None:
        first = ClaimBus(["a", "b"])
        second = ClaimBus(["a", "b"])
        for bus in (first, second):
            bus.record_claim("task.1", "a", time_s=0.0)
            bus.record_claim("task.2", "b", time_s=10.0)
        self.assertEqual(first.hash(), second.hash())


if __name__ == "__main__":
    unittest.main()
