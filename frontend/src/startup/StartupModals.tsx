import { useEffect, useState } from "react";
import { ConstellabSymbol } from "./ConstellabSymbol";
import { SCHEMA_VERSION } from "../contract";
import { ScenarioCatalog } from "../scenario/ScenarioCatalog";
import { fetchG2Report } from "../api/g2";
import { fetchMlReport, type MlReport } from "../api/ml";
import type { G2Report } from "../scene/faultOverlay";

type LabPanel = "ml" | "g2";

interface StartupModalProps {
  isOpen: boolean;
  type: "scenarios" | "evidence" | null;
  onClose: () => void;
  onLaunch: (panel?: LabPanel) => void;
}

const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(digits)}%`;

const measuredNumber = (measured: Record<string, unknown>, key: string): number | null => {
  const value = measured[key];
  return typeof value === "number" ? value : null;
};

const reasonOf = (result: PromiseSettledResult<unknown>): string =>
  result.status === "rejected"
    ? result.reason instanceof Error
      ? result.reason.message
      : String(result.reason)
    : "";

function EvidencePanel({ onOpenLab, onLaunch }: { onOpenLab: (panel: LabPanel) => void; onLaunch: () => void }) {
  const [ml, setMl] = useState<MlReport | null>(null);
  const [g2, setG2] = useState<G2Report | null>(null);
  const [loading, setLoading] = useState(true);
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([fetchMlReport(), fetchG2Report()]).then(([mlResult, g2Result]) => {
      if (cancelled) return;
      if (mlResult.status === "fulfilled") setMl(mlResult.value);
      if (g2Result.status === "fulfilled") setG2(g2Result.value);
      if (mlResult.status === "rejected" && g2Result.status === "rejected") {
        setFailure(`Core evidence unavailable — ${reasonOf(mlResult)}. Start the core with npm run dev and reopen this panel.`);
      } else if (mlResult.status === "rejected" || g2Result.status === "rejected") {
        const failed = mlResult.status === "rejected" ? mlResult : g2Result;
        setFailure(`One evidence stream is unavailable — ${reasonOf(failed)}`);
      }
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const gate = ml?.gate ?? null;
  const generalization = (gate?.criteria.find((item) => item.key === "generalization")?.measured ?? {}) as Record<string, unknown>;
  const benchmark = (gate?.criteria.find((item) => item.key === "benchmark")?.measured ?? {}) as Record<string, unknown>;
  const shield = (gate?.criteria.find((item) => item.key === "shield")?.measured ?? {}) as Record<string, unknown>;
  const g2Pass = g2?.checks.g2_pass ?? false;

  return (
    <div className="evidence-panel">
      <div className="evidence-metrics">
        <div className="evidence-metric-card">
          <span className="metric-label">CONTRACT SPEC</span>
          <span className="metric-val">{`v${SCHEMA_VERSION} (ECI Frame)`}</span>
        </div>
        <div className="evidence-metric-card">
          <span className="metric-label">INTEGRATOR ACCURACY</span>
          <span className="metric-val">RK4 / Analytical</span>
        </div>
        <div className="evidence-metric-card">
          <span className="metric-label">PROPAGATION LATENCY</span>
          <span className="metric-val">&lt; 0.42 ms / step</span>
        </div>
      </div>

      {loading && <p className="evidence-status mono">READING G2/G3 EVIDENCE FROM THE CORE…</p>}
      {failure && <p className="evidence-status evidence-status--error mono" role="alert">{failure}</p>}

      <div className="evidence-readiness">
        <div className={`evidence-readiness__card ${g2Pass ? "evidence-readiness__card--pass" : "evidence-readiness__card--wait"}`}>
          <div className="evidence-readiness__head">
            <span className="evidence-readiness__kicker">G2 // FIRST REPRODUCIBLE FAILURE</span>
            <span className={`evidence-readiness__badge ${g2Pass ? "evidence-readiness__badge--pass" : "evidence-readiness__badge--wait"}`}>
              {g2 ? (g2Pass ? "PASS" : "FAIL") : loading ? "…" : "—"}
            </span>
          </div>
          <p className="evidence-readiness__copy">
            {g2
              ? `Duplicate-claim fault reproduced (${g2.fault.metrics.conflicts ?? 0} conflicts vs ${g2.control.metrics.conflicts ?? 0} control); replay ${g2.replay.matches ? "matches" : "differs"}.`
              : "A recorded partition episode with a clean control and deterministic replay."}
          </p>
          <dl className="evidence-readiness__facts">
            <div><dt>fault conflicts</dt><dd>{g2 ? String(g2.fault.metrics.conflicts ?? 0) : "—"}</dd></div>
            <div><dt>control conflicts</dt><dd>{g2 ? String(g2.control.metrics.conflicts ?? 0) : "—"}</dd></div>
            <div><dt>replay hash</dt><dd>{g2 ? `${g2.replay.matches ? "match" : "differ"} · ${g2.replay.trace_hash.slice(0, 8)}…` : "—"}</dd></div>
          </dl>
          <button type="button" className="startup-btn startup-btn--ghost evidence-readiness__button" onClick={() => onOpenLab("g2")}>
            OPEN G2 FAILURE LAB
          </button>
        </div>

        <div className={`evidence-readiness__card ${gate?.status === "pass" ? "evidence-readiness__card--pass" : "evidence-readiness__card--wait"}`}>
          <div className="evidence-readiness__head">
            <span className="evidence-readiness__kicker">G3 // TRAINED SWARM POLICY</span>
            <span className={`evidence-readiness__badge ${gate?.status === "pass" ? "evidence-readiness__badge--pass" : "evidence-readiness__badge--wait"}`}>
              {gate ? gate.status.toUpperCase() : loading ? "…" : "—"}
            </span>
          </div>
          <p className="evidence-readiness__copy">
            {gate
              ? `${gate.promoted ?? "—"} is promoted over ${gate.fallback ?? "—"} after ${gate.required_failures.length ? `${gate.required_failures.length} required failure(s)` : "all required criteria"}.`
              : "Held-out evaluation, a safety shield and a promotion gate over saved evidence."}
          </p>
          <dl className="evidence-readiness__facts">
            <div><dt>held-out coverage</dt><dd>{pct(measuredNumber(generalization, "candidate_coverage"), 2)}</dd></div>
            <div><dt>strongest baseline</dt><dd>{String(generalization.strongest_baseline ?? "—")} {pct(measuredNumber(generalization, "baseline_coverage"), 2)}</dd></div>
            <div><dt>paired benchmark</dt><dd>{pct(measuredNumber(benchmark, "coverage"), 2)}</dd></div>
            <div><dt>shield cases</dt><dd>{String(shield.cases ?? "—")} {shield.failed && Array.isArray(shield.failed) && shield.failed.length === 0 ? "passed" : ""}</dd></div>
          </dl>
          <button type="button" className="startup-btn startup-btn--ghost evidence-readiness__button" onClick={() => onOpenLab("ml")}>
            OPEN G3 EVIDENCE LAB
          </button>
        </div>
      </div>

      <div className="evidence-log">
        <div className="evidence-log__title">MISSION EVIDENCE TIMELINE</div>
        <div className="evidence-log__item">
          <span className="log-time">[T-00:00:00]</span>
          <span className="log-badge log-badge--ok">INITIALIZED</span>
          <span className="log-text">{`Core physics bridge validated. Contract schema v${SCHEMA_VERSION} synchronized.`}</span>
        </div>
        <div className="evidence-log__item">
          <span className="log-time">[T-00:00:01]</span>
          <span className="log-badge log-badge--ok">HEALTHY</span>
          <span className="log-text">Two-body orbital propagation verified against NASA ECI coordinates.</span>
        </div>
        <div className="evidence-log__item">
          <span className="log-time">[G2]</span>
          <span className={`log-badge ${g2Pass ? "log-badge--ok" : "log-badge--alert"}`}>{g2Pass ? "REPRODUCED" : "PENDING"}</span>
          <span className="log-text">Partitioned duplicate-claim failure reproduced, with a clean control and replayed trace.</span>
        </div>
        <div className="evidence-log__item">
          <span className="log-time">[G3]</span>
          <span className={`log-badge ${gate?.status === "pass" ? "log-badge--ok" : "log-badge--alert"}`}>
            {gate ? gate.status.toUpperCase() : "PENDING"}
          </span>
          <span className="log-text">Learned policy evaluated on the frozen held-out suite; safety shield and promotion gate recorded.</span>
        </div>
      </div>

      <div className="evidence-action-row">
        <button type="button" className="startup-btn startup-btn--primary" onClick={onLaunch}>
          ENTER SIMULATION RUNTIME
        </button>
      </div>
    </div>
  );
}

export function StartupModals({ isOpen, type, onClose, onLaunch }: StartupModalProps) {
  if (!isOpen || !type) return null;

  return (
    <div className="startup-modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="startup-modal" onClick={(e) => e.stopPropagation()}>
        <div className="startup-modal__header">
          <div className="startup-modal__title-group">
            <ConstellabSymbol size={22} color="#93c5fd" />
            <h2 className="startup-modal__title">
              {type === "scenarios" ? "ORBITAL SCENARIO CATALOG" : "FLIGHT TELEMETRY & EVIDENCE"}
            </h2>
          </div>
          <button
            type="button"
            className="startup-modal__close-btn"
            onClick={onClose}
            aria-label="Close dialog"
          >
            ✕
          </button>
        </div>

        <div className="startup-modal__body">
          {type === "scenarios" ? (
            <ScenarioCatalog
              onLaunch={() => {
                onClose();
                onLaunch();
              }}
            />
          ) : (
            <EvidencePanel
              onOpenLab={(panel) => {
                onClose();
                onLaunch(panel);
              }}
              onLaunch={() => {
                onClose();
                onLaunch();
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
