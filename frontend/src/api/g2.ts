/**
 * Day 34: read the core's precomputed G2 failure-gate report.
 *
 * The report is generated and cached by the core (`/api/g2`); the renderer only
 * validates and displays it, so a UI bug can never invent a passing gate.
 */
import { CORE_URL } from "../scenario/api";
import { parseG2Report, type G2Report } from "../scene/faultOverlay";

export async function fetchG2Report(): Promise<G2Report> {
  const res = await fetch(`${CORE_URL}/api/g2`);
  if (!res.ok) throw new Error(`core returned ${res.status} for /api/g2`);
  return parseG2Report(await res.json());
}
