/** Day 34: the overlay derives drawable geometry, never simulation facts. */
import test from "node:test";
import assert from "node:assert/strict";

import { SCENE_CONFIG } from "../config";
import { faultFrameGeometry, frameIndexAt, parseG2Report } from "../faultOverlay";

const frame = {
  time_s: 0,
  positions_km: {
    "g2-bravo": [0, 6378.137, 0],
    "g2-alpha": [6378.137, 0, 0],
  } as Record<string, [number, number, number]>,
  target_eci_km: [6378.137, 0, 0] as [number, number, number],
  links: [["g2-alpha", "g2-bravo"], ["g2-alpha", "ghost"]] as Array<[string, string]>,
  groups: [["g2-alpha"], ["g2-bravo"]],
  active_faults: ["g2-partition-00"],
  faulted_satellites: ["g2-alpha"],
};

function reportPayload(overrides: Record<string, unknown> = {}) {
  return {
    version: "g2-failure-1",
    scenario_id: "g2-partition",
    seed: 7,
    horizon_s: 1800,
    step_s: 30,
    groups: { A: ["g2-alpha"], B: ["g2-bravo"] },
    target: { id: "hotspot.g2-eq", kind: "hotspot", lat_deg: 1, lon_deg: 2 },
    checks: { g2_pass: true },
    fault: { metrics: {}, trace_hash: "a".repeat(64), invariants: { counts: {}, violations: 1, errors: 1 }, first_violation: null, events: [], manifest: {} },
    control: { metrics: {}, trace_hash: "b".repeat(64), invariants: { counts: {}, violations: 0, errors: 0 }, first_violation: null, events: [], manifest: {} },
    replay: { trace_hash: "a".repeat(64), matches: true },
    timeline: [frame],
    first_violation: null,
    limitations: ["synthetic"],
    ...overrides,
  };
}

test("parseG2Report accepts a well-formed report", () => {
  const report = parseG2Report(reportPayload());
  assert.equal(report.scenario_id, "g2-partition");
  assert.equal(report.timeline.length, 1);
  assert.deepEqual(report.groups.A, ["g2-alpha"]);
});

test("parseG2Report rejects payloads without a timeline", () => {
  assert.throws(() => parseG2Report({ version: "g2-failure-1" }));
  assert.throws(() => parseG2Report(reportPayload({ timeline: [] })));
  assert.throws(() => parseG2Report(reportPayload({ timeline: [{ positions_km: {} }] })));
  assert.throws(() => parseG2Report(null));
});

test("faultFrameGeometry converts km to scene units and keeps ids sorted", () => {
  const report = parseG2Report(reportPayload());
  const geometry = faultFrameGeometry(report, report.timeline[0]);
  const scale = 1 / SCENE_CONFIG.units.kmPerSceneUnit;
  assert.deepEqual(geometry.satellites.map((satellite) => satellite.id), ["g2-alpha", "g2-bravo"]);
  assert.deepEqual(geometry.satellites[0].position, [6378.137 * scale, 0, 0]);
  assert.equal(geometry.satellites[0].group, "A");
  assert.equal(geometry.satellites[0].faulted, true);
  assert.equal(geometry.satellites[1].faulted, false);
  assert.deepEqual(geometry.activeFaults, ["g2-partition-00"]);
});

test("links with unknown satellites are skipped, never invented", () => {
  const report = parseG2Report(reportPayload());
  const geometry = faultFrameGeometry(report, report.timeline[0]);
  assert.equal(geometry.segments.length, 6, "exactly one link pair is drawn");
  assert.deepEqual(geometry.segments.slice(0, 3), geometry.satellites[0].position);
  assert.deepEqual(geometry.segments.slice(3, 6), geometry.satellites[1].position);
});

test("frameIndexAt clamps to the recorded frames", () => {
  const report = parseG2Report(reportPayload({
    timeline: [
      { ...frame, time_s: 0 },
      { ...frame, time_s: 600 },
      { ...frame, time_s: 1200 },
    ],
  }));
  assert.equal(frameIndexAt(report, -5), 0);
  assert.equal(frameIndexAt(report, 0), 0);
  assert.equal(frameIndexAt(report, 700), 1);
  assert.equal(frameIndexAt(report, 5000), 2);
});
