/**
 * Day 34: the G2 failure lab.
 *
 * Everything shown here comes from the core's `/api/g2` report: gate checks,
 * trace hashes, invariant findings, link topology and recorded positions. The
 * slider scrubs the recorded episode; the parent passes the selected frame to
 * the 3D scene, so the globe draws stored state rather than re-deriving it.
 */
import type { G2Report, G2Violation } from "./faultOverlay";

interface G2FaultLabProps {
  report: G2Report | null;
  error: string | null;
  loading: boolean;
  frameIndex: number;
  onFrameIndex: (index: number) => void;
  onRefresh: () => void;
  onClose: () => void;
}

const shortHash = (hash: string) => (hash ? `${hash.slice(0, 12)}…` : "—");
const num = (value: unknown) => (typeof value === "number" ? value : 0);

function GateList({ checks }: { checks: Record<string, boolean> }) {
  const entries = Object.entries(checks);
  return (
    <ul className="g2-lab__checks" data-testid="g2-gate-checks">
      {entries.map(([name, passed]) => (
        <li key={name} className={passed ? "g2-lab__check g2-lab__check--pass" : "g2-lab__check g2-lab__check--fail"}>
          <span aria-hidden="true">{passed ? "✓" : "✗"}</span>
          <span className="mono">{name}</span>
        </li>
      ))}
    </ul>
  );
}

function ViolationBlock({ violation, label }: { violation: G2Violation | null; label: string }) {
  if (!violation) {
    return <p className="g2-lab__quiet">{label}: no invariant findings</p>;
  }
  return (
    <div className="g2-lab__violation" data-testid="g2-first-violation">
      <div className="g2-lab__violation-head">
        <span className="g2-lab__badge">{violation.severity}</span>
        <span className="mono">{violation.rule}</span>
        <span className="mono g2-lab__muted">t={violation.time_s.toFixed(0)} s</span>
      </div>
      <p className="g2-lab__message">{violation.message}</p>
      <div className="g2-lab__entities">
        {violation.entities.map((entity) => (
          <span key={entity} className="g2-lab__chip mono">{entity}</span>
        ))}
      </div>
    </div>
  );
}

export function G2FaultLab({
  report,
  error,
  loading,
  frameIndex,
  onFrameIndex,
  onRefresh,
  onClose,
}: G2FaultLabProps) {
  const timeline = report?.timeline ?? [];
  const index = Math.max(0, Math.min(frameIndex, Math.max(timeline.length - 1, 0)));
  const frame = timeline[index];
  const selectedTimeS = frame?.time_s ?? 0;
  const events = (report?.fault.events ?? [])
    .filter((event) => typeof event.time_s === "number" && (event.time_s as number) <= selectedTimeS + 1e-9)
    .slice(-9)
    .reverse();

  return (
    <aside className="g2-lab" data-testid="g2-fault-lab" aria-label="G2 failure lab">
      <header className="g2-lab__header">
        <div>
          <h2 className="g2-lab__title">G2 FAILURE LAB</h2>
          <p className="g2-lab__subtitle mono">
            {report ? `${report.scenario_id} · seed ${report.seed} · ${shortHash(report.fault.trace_hash)}` : "loading core report…"}
          </p>
        </div>
        <div className="g2-lab__header-actions">
          <button type="button" className="g2-lab__icon-button" onClick={onRefresh} title="Regenerate report" aria-label="Refresh G2 report">
            ⟳
          </button>
          <button type="button" className="g2-lab__icon-button" onClick={onClose} title="Close failure lab" aria-label="Close failure lab">
            ✕
          </button>
        </div>
      </header>

      {error && <p className="g2-lab__error" role="alert">{error}</p>}
      {loading && !report && !error && <p className="g2-lab__quiet">Running the gate on the core…</p>}

      {report && (
        <>
          <section className="g2-lab__section">
            <h3 className="g2-lab__section-title">GATE</h3>
            <GateList checks={report.checks} />
            <p
              className={report.checks.g2_pass ? "g2-lab__verdict g2-lab__verdict--pass" : "g2-lab__verdict g2-lab__verdict--fail"}
              data-testid="g2-verdict"
            >
              {report.checks.g2_pass ? "G2 PASS — fault reproduced, control clean, replay matches" : "G2 FAIL — see checks above"}
            </p>
          </section>

          <section className="g2-lab__section">
            <h3 className="g2-lab__section-title">FIRST VIOLATION</h3>
            <ViolationBlock violation={report.first_violation} label="Fault case" />
          </section>

          <section className="g2-lab__section">
            <h3 className="g2-lab__section-title">RECORDED TIMELINE</h3>
            <input
              className="g2-lab__slider"
              type="range"
              min={0}
              max={Math.max(timeline.length - 1, 0)}
              value={index}
              onChange={(event) => onFrameIndex(Number(event.target.value))}
              aria-label="Recorded timeline position"
              data-testid="g2-timeline-slider"
            />
            <div className="g2-lab__frame mono" data-testid="g2-frame-readout">
              <span>t={selectedTimeS.toFixed(0)} s</span>
              <span>links {frame?.links.length ?? 0}</span>
              <span>faults {(frame?.active_faults.length ?? 0) > 0 ? frame?.active_faults.join(", ") : "none"}</span>
            </div>
            <div className="g2-lab__groups">
              {Object.entries(report.groups).map(([group, members], groupIndex) => (
                <div key={group} className="g2-lab__group">
                  <span className={`g2-lab__dot g2-lab__dot--${groupIndex === 0 ? "a" : "b"}`} aria-hidden="true" />
                  <span className="mono">{group}</span>
                  <span className="mono g2-lab__muted">{members.join(" · ")}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="g2-lab__section">
            <h3 className="g2-lab__section-title">EVENTS AT t ≤ {selectedTimeS.toFixed(0)} s</h3>
            <ul className="g2-lab__events" data-testid="g2-events">
              {events.length === 0 && <li className="g2-lab__quiet">no events yet</li>}
              {events.map((event, eventIndex) => (
                <li key={`${String(event.kind)}-${String(event.time_s)}-${eventIndex}`} className="mono">
                  <span className="g2-lab__muted">t={num(event.time_s).toFixed(0)}</span>{" "}
                  <span>{String(event.kind)}</span>
                </li>
              ))}
            </ul>
          </section>

          <section className="g2-lab__section">
            <h3 className="g2-lab__section-title">FAULT VS CONTROL</h3>
            <table className="g2-lab__table mono">
              <thead>
                <tr><th>metric</th><th>fault</th><th>control</th></tr>
              </thead>
              <tbody>
                {["completed", "conflicts", "proposals", "reward"].map((metric) => (
                  <tr key={metric}>
                    <td>{metric}</td>
                    <td>{num(report.fault.metrics[metric]).toFixed(metric === "reward" ? 2 : 0)}</td>
                    <td>{num(report.control.metrics[metric]).toFixed(metric === "reward" ? 2 : 0)}</td>
                  </tr>
                ))}
                <tr>
                  <td>invariant findings</td>
                  <td>{num(report.fault.invariants?.violations)}</td>
                  <td>{num(report.control.invariants?.violations)}</td>
                </tr>
              </tbody>
            </table>
            <p className="g2-lab__muted mono">
              replay {report.replay.matches ? "matches" : "differs"} · control {shortHash(report.control.trace_hash)}
            </p>
          </section>

          <section className="g2-lab__section">
            <h3 className="g2-lab__section-title">LIMITATIONS</h3>
            <ul className="g2-lab__limitations">
              {report.limitations.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </section>
        </>
      )}
    </aside>
  );
}
