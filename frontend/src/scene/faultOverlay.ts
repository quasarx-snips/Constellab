/**
 * Day 34: pure derivation for the G2 failure-lab overlay.
 *
 * The core already decided every fact in this report (positions, links, active
 * faults, the first invariant violation). This module only validates the
 * payload and turns one recorded frame into numbers the Three.js layer can
 * draw — it never recomputes orbit state, link geometry or correctness.
 */
import { SCENE_CONFIG } from "./config";

export interface G2Violation {
  rule: string;
  severity: string;
  time_s: number;
  message: string;
  entities: string[];
  evidence: Record<string, unknown>;
}

export interface G2CaseSummary {
  metrics: Record<string, number>;
  trace_hash: string;
  invariants: { counts: Record<string, number>; violations: number; errors: number };
  first_violation: G2Violation | null;
  events: Array<Record<string, unknown>>;
  manifest: Record<string, unknown>;
}

export interface G2Frame {
  time_s: number;
  positions_km: Record<string, [number, number, number]>;
  target_eci_km: [number, number, number];
  links: Array<[string, string]>;
  groups: string[][];
  active_faults: string[];
  faulted_satellites: string[];
}

export interface G2Report {
  version: string;
  scenario_id: string;
  seed: number;
  horizon_s: number;
  step_s: number;
  groups: Record<string, string[]>;
  target: { id: string; kind: string; lat_deg: number; lon_deg: number };
  checks: Record<string, boolean>;
  fault: G2CaseSummary;
  control: G2CaseSummary;
  replay: { trace_hash: string; matches: boolean };
  timeline: G2Frame[];
  first_violation: G2Violation | null;
  limitations: string[];
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function requireString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`invalid G2 report: ${path} must be a non-empty string`);
  }
  return value;
}

/** Validate the fields the overlay depends on; unknown extras are ignored. */
export function parseG2Report(payload: unknown): G2Report {
  if (!isRecord(payload)) throw new Error("invalid G2 report: expected an object");
  const timeline = payload.timeline;
  if (!Array.isArray(timeline) || timeline.length === 0) {
    throw new Error("invalid G2 report: timeline must be a non-empty array");
  }
  const frames: G2Frame[] = timeline.map((frame, index) => {
    if (!isRecord(frame)) throw new Error(`invalid G2 report: timeline[${index}] must be an object`);
    const positions = frame.positions_km;
    if (!isRecord(positions)) {
      throw new Error(`invalid G2 report: timeline[${index}].positions_km must be an object`);
    }
    const target = frame.target_eci_km;
    if (!Array.isArray(target) || target.length !== 3) {
      throw new Error(`invalid G2 report: timeline[${index}].target_eci_km must be [x, y, z]`);
    }
    return {
      time_s: Number(frame.time_s ?? 0),
      positions_km: positions as Record<string, [number, number, number]>,
      target_eci_km: target.map(Number) as [number, number, number],
      links: Array.isArray(frame.links) ? (frame.links as Array<[string, string]>) : [],
      groups: Array.isArray(frame.groups) ? (frame.groups as string[][]) : [],
      active_faults: Array.isArray(frame.active_faults) ? (frame.active_faults as string[]) : [],
      faulted_satellites: Array.isArray(frame.faulted_satellites)
        ? (frame.faulted_satellites as string[])
        : [],
    };
  });
  return {
    version: requireString(payload.version, "version"),
    scenario_id: requireString(payload.scenario_id, "scenario_id"),
    seed: Number(payload.seed ?? 0),
    horizon_s: Number(payload.horizon_s ?? 0),
    step_s: Number(payload.step_s ?? 0),
    groups: (isRecord(payload.groups) ? payload.groups : {}) as Record<string, string[]>,
    target: (isRecord(payload.target) ? payload.target : {}) as G2Report["target"],
    checks: (isRecord(payload.checks) ? payload.checks : {}) as Record<string, boolean>,
    fault: (isRecord(payload.fault) ? payload.fault : {}) as unknown as G2CaseSummary,
    control: (isRecord(payload.control) ? payload.control : {}) as unknown as G2CaseSummary,
    replay: (isRecord(payload.replay) ? payload.replay : { trace_hash: "", matches: false }) as G2Report["replay"],
    timeline: frames,
    first_violation: (payload.first_violation as G2Violation | null) ?? null,
    limitations: Array.isArray(payload.limitations) ? (payload.limitations as string[]) : [],
  };
}

export interface OverlaySatellite {
  id: string;
  group: string;
  position: [number, number, number];
  faulted: boolean;
}

export interface FaultFrameGeometry {
  satellites: OverlaySatellite[];
  /** Flat [x1,y1,z1,x2,y2,z2, …] line pairs, already in scene units. */
  segments: number[];
  target: [number, number, number];
  activeFaults: string[];
}

function toScene(position: [number, number, number]): [number, number, number] {
  const scale = 1 / SCENE_CONFIG.units.kmPerSceneUnit;
  return [position[0] * scale, position[1] * scale, position[2] * scale];
}

/**
 * Turn one recorded frame into drawable geometry.
 *
 * Links between satellites missing from the frame are skipped (a partial
 * snapshot must never invent a line), and every satellite id appears once, in
 * sorted order, so the instanced marker layer has a stable mapping.
 */
export function faultFrameGeometry(
  report: Pick<G2Report, "groups">,
  frame: G2Frame,
): FaultFrameGeometry {
  const groupOf = new Map<string, string>();
  for (const [group, members] of Object.entries(report.groups)) {
    for (const id of members) groupOf.set(id, group);
  }
  const faulted = new Set(frame.faulted_satellites);
  const satellites: OverlaySatellite[] = Object.keys(frame.positions_km)
    .sort()
    .map((id) => ({
      id,
      group: groupOf.get(id) ?? "?",
      position: toScene(frame.positions_km[id]),
      faulted: faulted.has(id),
    }));
  const sceneById = new Map(satellites.map((satellite) => [satellite.id, satellite.position]));
  const segments: number[] = [];
  for (const [a, b] of frame.links) {
    const from = sceneById.get(a);
    const to = sceneById.get(b);
    if (!from || !to) continue;
    segments.push(from[0], from[1], from[2], to[0], to[1], to[2]);
  }
  return {
    satellites,
    segments,
    target: toScene(frame.target_eci_km),
    activeFaults: [...frame.active_faults],
  };
}

/** Nearest recorded frame at or before ``timeS`` (clamped at both ends). */
export function frameIndexAt(report: Pick<G2Report, "timeline">, timeS: number): number {
  const frames = report.timeline;
  let best = 0;
  for (let index = 0; index < frames.length; index++) {
    if (frames[index].time_s <= timeS + 1e-9) best = index;
    else break;
  }
  return best;
}
