import { useEffect, useId, useMemo, useRef, useState } from "react";
import { publicAssetUrl } from "../assetUrl";
import type { SnapshotState } from "../App";
import type { ConnectionInfo } from "../api/connection";
import { connectionLabel, connectionMessage } from "../api/connection";
import type { SatelliteState, SimulationClock } from "../contract";
import type { SatelliteProjection } from "./EarthScene";
import { EarthScene } from "./EarthScene";
import { GroundTrackMap } from "./GroundTrackMap";
import { InspectorPanel } from "./InspectorPanel";
import { SceneDiagnostics } from "./SceneDiagnostics";
import { DEFAULT_DIAGNOSTIC_TOGGLES, type DiagnosticToggles, snapshotAgeMs } from "./diagnostics";
import { ScenarioCatalog } from "../scenario/ScenarioCatalog";
import { ParameterEditor } from "../scenario/ParameterEditor";
import { RegistryEditor } from "../scenario/RegistryEditor";
import { getSatelliteColor } from "./orbit";
import { SCENE_CONFIG } from "./config";

const CORE_URL = "http://127.0.0.1:8765";

interface CoreStatus {
  name: string;
  version: string;
  schema_version: string;
  status: string;
}

interface SceneShellProps {
  snapshot: SnapshotState;
  status?: CoreStatus | null;
  statusError?: string | null;
  connection?: ConnectionInfo | null;
  snapshotReceivedAtMs?: number | null;
  onExit?: () => void;
}

/** Link state banner; ticks its own countdown while the core is unreachable. */
function ConnectionBanner({ connection }: { connection: ConnectionInfo }) {
  const live = connection.phase === "live";
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    if (live) return;
    const timer = window.setInterval(() => setNowMs(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, [live]);
  const message = connectionMessage(connection, nowMs);
  return (
    <div
      className={`scene-shell__connection scene-shell__connection--${connection.phase}`}
      role="status"
      data-testid="connection-banner"
      data-connection-phase={connection.phase}
    >
      <span className={`topbar-status ${live ? "topbar-status--ok" : "topbar-status--down"}`}>
        {connectionLabel(connection)}
      </span>
      <span className="scene-shell__connection-text mono">{message}</span>
    </div>
  );
}

async function clockPost(action: string, body?: object): Promise<SimulationClock | null> {
  try {
    const res = await fetch(`${CORE_URL}/api/clock/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) return null;
    return (await res.json()) as SimulationClock;
  } catch {
    return null;
  }
}

const km = (v: number) => v.toFixed(1);
const kms = (v: number) => v.toFixed(3);

const TIME_SCALE_OPTIONS = [1, 10, 60, 300, 1000, 10000, 36000];
const formatScale = (value: number) => `${value.toLocaleString("en-US")}×`;

function formatSimulatedTime(epochUtc?: string, simTimeS: number = 0): string {
  const epoch = epochUtc ? new Date(epochUtc).getTime() : Date.now();
  const current = new Date(epoch + simTimeS * 1000);
  return current.toISOString().replace(/\.\d{3}Z$/, "");
}

function SatelliteRow({
  sat,
  color,
  isSelected,
  onSelect,
}: {
  sat: SatelliteState;
  color: number;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const [x, y, z] = sat.position_eci_km;
  const [vx, vy, vz] = sat.velocity_eci_km_s;
  const colorHex = `#${color.toString(16).padStart(6, "0")}`;
  // NOTE: the "anomaly" label is a Day 4 visual badge carried over from the reference
  // design. The contract has no health field, so this is presentation only; Day 9+ replaces
  // it with the real parameter model.
  const isAnomaly = sat.id.toLowerCase().includes("anomaly");

  return (
    <tr
      data-satellite-id={sat.id}
      className={`tactical-table__row ${isSelected ? "tactical-table__row--selected" : ""}`}
      style={{ color: colorHex }}
      onClick={onSelect}
      role="button"
      tabIndex={0}
    >
      <td className="mono tactical-table__id">
        <span className="tactical-status-dot" style={{ backgroundColor: colorHex }} />
        {sat.id}
      </td>
      <td className="mono">{`${km(x)}, ${km(y)}, ${km(z)}`}</td>
      <td className="mono">{`${kms(vx)}, ${kms(vy)}, ${kms(vz)}`}</td>
      <td className="mono">{sat.orbit ? `${km(sat.orbit.semi_major_axis_km)} km / ${sat.orbit.inclination_deg}°` : "—"}</td>
      <td>
        <span className={`tactical-badge ${isAnomaly ? "tactical-badge--anomaly" : "tactical-badge--nominal"}`}>
          {isAnomaly ? "TRUE ANOMALY" : "NOMINAL"}
        </span>
      </td>
    </tr>
  );
}

export function SceneShell({ snapshot, status, statusError, connection, snapshotReceivedAtMs, onExit }: SceneShellProps) {
  const viewport = useRef<HTMLDivElement>(null);
  const scene = useRef<EarthScene | null>(null);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [timeWarp, setTimeWarp] = useState<number>(SCENE_CONFIG.timeWarp.default);
  const timeWarpId = useId();
  const [clockState, setClockState] = useState<SimulationClock | null>(null);

  // Tactical UI State
  const [viewMode, setViewMode] = useState<"3d" | "2d">("3d");
  const [projections, setProjections] = useState<SatelliteProjection[]>([]);
  const [selectedSatId, setSelectedSatId] = useState<string | null>(null);
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  // Day 22: the rail settings button opens either the contract-scaled schema
  // editor or the registry-scale editor; both act on the same running scenario.
  const [settingsMode, setSettingsMode] = useState<"schema" | "registry">("registry");
  const [isScenarioOpen, setIsScenarioOpen] = useState(false);
  const [autoRotate] = useState(true);
  const [showOverlays, setShowOverlays] = useState(true);
  const [showOrbits, setShowOrbits] = useState(true);
  const [showCallouts] = useState(true);
  // Day 12: diagnostics are display-only layers with independent toggles.
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [diagnosticToggles, setDiagnosticToggles] = useState<DiagnosticToggles>(DEFAULT_DIAGNOSTIC_TOGGLES);
  const [renderFps, setRenderFps] = useState<number | null>(null);

  // Mount Three.js viewport
  useEffect(() => {
    if (!viewport.current) return;
    try {
      scene.current = new EarthScene(viewport.current);
      scene.current.setOnRender((proj) => setProjections(proj));
      scene.current.setOnSelect((id) => {
        if (!id) return;
        setSelectedSatId(id);
        setIsInspectorOpen(true);
      });
      scene.current.setAutoRotate(autoRotate);
      setSceneError(null);
    } catch (err) {
      setSceneError(err instanceof Error ? err.message : String(err));
    }
    return () => {
      scene.current?.dispose();
      scene.current = null;
    };
  }, []);

  // Update Three.js with simulation snapshots
  useEffect(() => {
    scene.current?.setSnapshot(snapshot.kind === "ready" ? snapshot.snapshot : null);
  }, [snapshot]);

  // Toggle tactical globe overlays (coordinate rings, LEO boundary)
  useEffect(() => {
    scene.current?.setOverlaysVisible(showOverlays);
  }, [showOverlays]);

  // Toggle orbit paths; Day 4 requires every path to be switchable.
  useEffect(() => {
    scene.current?.setOrbitsVisible(showOrbits);
  }, [showOrbits]);

  // Toggle auto-rotation
  useEffect(() => {
    scene.current?.setAutoRotate(autoRotate);
  }, [autoRotate]);

  // Mirror the selection into the 3D scene (highlight only, never simulation state)
  useEffect(() => {
    scene.current?.setSelection(selectedSatId);
  }, [selectedSatId]);

  // Diagnostic plane layer follows its own toggle.
  useEffect(() => {
    scene.current?.setOrbitPlaneVisible(diagnosticToggles.orbitPlane);
  }, [diagnosticToggles.orbitPlane]);

  // Renderer health readout; read from the scene rather than measuring in React.
  useEffect(() => {
    const timer = window.setInterval(() => {
      const stats = scene.current?.renderStats();
      setRenderFps(stats && stats.frames > 0 ? stats.fps : null);
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  const satellites = snapshot.kind === "ready" ? snapshot.snapshot.satellites : [];
  const activeScenarioId = snapshot.kind === "ready" ? snapshot.snapshot.scenario_id : null;

  // A scenario switch replaces the world; a selection from the old world must not
  // survive into the new one.
  useEffect(() => {
    setSelectedSatId(null);
  }, [activeScenarioId]);

  const handleSelectSatellite = (id: string) => {
    setSelectedSatId(id);
    setIsInspectorOpen(true);
  };

  const handleFocusSatellite = (id: string) => {
    if (viewMode === "2d") {
      setViewMode("3d");
      // Give the viewport one frame to become visible before moving the camera.
      window.setTimeout(() => scene.current?.focusSatellite(id), 60);
      return;
    }
    scene.current?.focusSatellite(id);
  };

  // Clock Actions
  const handlePause = async () => {
    const newClock = await clockPost("pause");
    if (newClock) setClockState(newClock);
  };

  const handleResume = async () => {
    const newClock = await clockPost("resume");
    if (newClock) setClockState(newClock);
  };

  const handleReset = async () => {
    const newClock = await clockPost("reset");
    if (newClock) setClockState(newClock);
  };

  const handleStep = async () => {
    const newClock = await clockPost("step");
    if (newClock) setClockState(newClock);
  };

  const handleTimeScaleChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const v = Number(e.target.value);
    setTimeWarp(v);
    const newClock = await clockPost("time_scale", { time_scale: v });
    if (newClock) setClockState(newClock);
  };

  const currentClock = snapshot.kind === "ready" ? snapshot.snapshot.clock : null;
  const isPaused = currentClock?.paused ?? clockState?.paused ?? true;
  // The snapshot clock is canonical (scenario loads carry their own time_scale);
  // local action state is only a fallback while a request is in flight.
  const displayTimeWarp = currentClock?.time_scale ?? clockState?.time_scale ?? timeWarp;
  const simSeconds = currentClock?.simulation_time_s ?? 0;
  const epochUtc = currentClock?.epoch_utc;
  const formattedTimestamp = useMemo(() => formatSimulatedTime(epochUtc, simSeconds), [epochUtc, simSeconds]);
  const snapshotAge = snapshotAgeMs(snapshotReceivedAtMs ?? null, Date.now());

  return (
    <section
      className="scene-shell"
      data-testid="scene-shell"
      aria-label="Orbital viewport"
      data-connection-phase={connection?.phase ?? "connecting"}
      data-snapshot-age-ms={snapshotAge === null ? "" : String(Math.round(snapshotAge))}
    >
      {/* 3D WebGL Canvas Viewport (preserved in DOM for contract & scene tests) */}
      <div
        className="scene-shell__viewport"
        ref={viewport}
        data-testid="earth-viewport"
        style={{ visibility: viewMode === "3d" ? "visible" : "hidden" }}
      />

      {/* 2D Equirectangular Ground Track Projection Map */}
      {viewMode === "2d" && (
        <GroundTrackMap
          snapshot={snapshot}
          selectedSatId={selectedSatId}
          onSelectSat={setSelectedSatId}
          showCallouts={showCallouts}
        />
      )}

      {/* 1. Left Tactical Navigation Rail */}
      <aside className="scene-shell__rail" aria-label="Tactical Navigation">
        {/* Brand Icon (Sleek Constellab Blue Diamond identity) */}
        <button
          type="button"
          className="rail-item rail-item--logo"
          onClick={onExit}
          title="Return to Startup Flight Deck"
          aria-label="Constellab Mission Hub"
        >
          <img src={publicAssetUrl("constellab_symbol_white.svg")} alt="Constellab" className="rail-logo-img" />
        </button>

        {/* Tactical Modes */}
        <div className="rail-modes" role="tablist">
          {/* 3D Orbital Globe View */}
          <button
            type="button"
            className={`rail-item ${viewMode === "3d" ? "rail-item--active" : ""}`}
            onClick={() => setViewMode("3d")}
            title="3D Orbital Globe View"
            aria-label="3D Globe View"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <circle cx="12" cy="12" r="9" />
              <ellipse cx="12" cy="12" rx="9" ry="3.5" transform="rotate(-28 12 12)" />
              <circle cx="12" cy="12" r="1.5" fill="currentColor" />
            </svg>
          </button>

          {/* 2D Ground Track Map Projection */}
          <button
            type="button"
            className={`rail-item ${viewMode === "2d" ? "rail-item--active" : ""}`}
            onClick={() => setViewMode("2d")}
            title="2D Ground Track Map Projection"
            aria-label="2D Ground Track"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6" />
              <line x1="8" y1="2" x2="8" y2="18" />
              <line x1="16" y1="6" x2="16" y2="22" />
            </svg>
          </button>

          <button
            type="button"
            className={`rail-item ${showOverlays ? "rail-item--active" : ""}`}
            aria-pressed={showOverlays}
            onClick={() => setShowOverlays((v) => !v)}
            title="Toggle graticule + polar reticle overlays"
            aria-label="Toggle coordinate overlays"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <circle cx="12" cy="12" r="2.5" fill="currentColor" />
              <path d="M4.93 4.93a10 10 0 0 1 14.14 0M7.76 7.76a6 6 0 0 1 8.48 0M10.59 10.59a2 2 0 0 1 2.82 0" />
            </svg>
          </button>

          <button
            type="button"
            className={`rail-item ${isInspectorOpen ? "rail-item--active" : ""}`}
            aria-pressed={isInspectorOpen}
            onClick={() => {
              setIsInspectorOpen((v) => !v);
              setIsSettingsOpen(false);
              setIsDrawerOpen(false);
            }}
            title="Satellite Inspector (select by ID, orbit elements, ECI state)"
            aria-label="Satellite Inspector"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
              <circle cx="12" cy="12" r="8.5" strokeDasharray="3 3" />
            </svg>
          </button>

          <button
            type="button"
            className={`rail-item ${showDiagnostics ? "rail-item--active" : ""}`}
            aria-pressed={showDiagnostics}
            onClick={() => setShowDiagnostics((v) => !v)}
            title="Visual diagnostics (ids, orbit plane, time, selection, snapshot age)"
            aria-label="Toggle diagnostics"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <polyline points="2 13 6 13 9 5 13 19 16 10 19 13 22 13" />
            </svg>
          </button>

          <button
            type="button"
            className={`rail-item ${isDrawerOpen ? "rail-item--active" : ""}`}
            aria-pressed={isDrawerOpen}
            onClick={() => {
              setIsDrawerOpen((v) => !v);
              setIsInspectorOpen(false);
              setIsSettingsOpen(false);
            }}
            title="Swarm telemetry table"
            aria-label="Swarm telemetry table"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <line x1="3" y1="9" x2="21" y2="9" />
              <line x1="9" y1="21" x2="9" y2="9" />
            </svg>
          </button>

          {/* Functional Settings Panel */}
          <button
            type="button"
            className={`rail-item ${isSettingsOpen ? "rail-item--active" : ""}`}
            aria-pressed={isSettingsOpen}
            onClick={() => {
              setIsSettingsOpen((v) => !v);
              setIsInspectorOpen(false);
              setIsDrawerOpen(false);
            }}
            title="Mission parameters (schema-driven editor)"
            aria-label="Mission Parameters"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </button>
        </div>

        {/* Bottom Exit Button */}
        {onExit && (
          <button
            type="button"
            className="rail-item rail-item--exit"
            onClick={onExit}
            title="Return to Startup HUD"
            aria-label="Return to Startup HUD"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
          </button>
        )}
      </aside>

      {/* 2. Top Classification Header */}
      <header className="scene-shell__topbar">
        <div className="topbar-classification">CLASSIFICATION // UNCLASSIFIED</div>
        <div className="topbar-sub">
          <span className="topbar-title">CONSTELLAB TACTICAL TELEMETRY</span>
          <span className="topbar-sep">·</span>
          <span
            className={`topbar-status ${status ? "topbar-status--ok" : "topbar-status--down"}`}
            data-testid="core-status"
          >
            {status
              ? `core ${status.name} v${status.version} (${status.schema_version}) — ${status.status}`
              : statusError
                ? `core unreachable: ${statusError}`
                : "connecting to core…"}
          </span>
          <span className="topbar-sep">·</span>
          <button
            type="button"
            className="topbar-scenario mono"
            onClick={() => setIsScenarioOpen(true)}
            title="Open scenario catalog"
            aria-haspopup="dialog"
          >
            {snapshot.kind === "ready" ? `SCENARIO: ${snapshot.snapshot.scenario_id.toUpperCase()}` : "LOADING"}
            <span className="topbar-scenario__chevron" aria-hidden="true">▾</span>
          </button>
        </div>
      </header>

      {/* Day 12: link state. A stale globe must never look live. */}
      {connection && connection.phase !== "live" && <ConnectionBanner connection={connection} />}

      {/* Day 12: display-only diagnostics with independent layer toggles. */}
      {showDiagnostics && (
        <SceneDiagnostics
          snapshot={snapshot.kind === "ready" ? snapshot.snapshot : null}
          projections={projections}
          toggles={diagnosticToggles}
          selectedId={selectedSatId}
          receivedAtMs={snapshotReceivedAtMs ?? null}
          renderFps={renderFps}
          onToggle={(key) => setDiagnosticToggles((current) => ({ ...current, [key]: !current[key] }))}
          onClose={() => setShowDiagnostics(false)}
        />
      )}

      {/* 3. 2D Satellite Tactical Callout Overlay Layer */}
      <div className="scene-shell__callout-layer" style={{ pointerEvents: "none" }}>
        <svg className="scene-shell__callout-svg" width="100%" height="100%">
          {projections
            .filter((p) => !p.isOccluded)
            .map((p) => {
              // Only the selected satellite gets a leader line: a large fleet would
              // otherwise stack dozens of overlapping badges over the globe.
              if (selectedSatId !== p.id) return null;

              const dx = p.screenX > 500 ? 55 : -55;
              const dy = -42;
              const lineEnd = dx > 0 ? dx + 70 : dx - 70;

              return (
                <g key={`leader-${p.id}`} className="callout-leader-group">
                  <path
                    d={`M ${p.screenX} ${p.screenY} L ${p.screenX + dx} ${p.screenY + dy} L ${p.screenX + lineEnd} ${p.screenY + dy}`}
                    stroke={p.colorHex}
                    strokeWidth="1.25"
                    strokeDasharray="4 2"
                    fill="none"
                    opacity="0.85"
                  />
                  <circle cx={p.screenX} cy={p.screenY} r="3" fill={p.colorHex} />
                  <circle cx={p.screenX + lineEnd} cy={p.screenY + dy} r="2" fill={p.colorHex} />
                </g>
              );
            })}
        </svg>

        {/* HTML Tactical Callout Badges */}
        {projections
          .filter((p) => !p.isOccluded && p.id === selectedSatId)
          .map((p) => {
            const isSelected = true;
            const isAnomaly = p.id.toLowerCase().includes("anomaly");
            const dx = p.screenX > 500 ? 60 : -190;
            const dy = -62;

            return (
              <div
                key={`badge-${p.id}`}
                className={`tactical-callout ${isSelected ? "tactical-callout--selected" : ""} ${isAnomaly ? "tactical-callout--anomaly" : ""}`}
                style={{
                  left: `${p.screenX + dx}px`,
                  top: `${p.screenY + dy}px`,
                  borderColor: p.colorHex,
                }}
                onClick={() => handleSelectSatellite(p.id)}
              >
                <div className="callout-distance mono">Distance {p.distanceKm.toFixed(2)} km</div>
                <div className="callout-header">
                  <span className="callout-name mono" style={{ color: p.colorHex }}>
                    {isAnomaly ? "Jackal-II" : p.id.toUpperCase()}
                  </span>
                  <span className={`callout-tag ${isAnomaly ? "callout-tag--anomaly" : "callout-tag--nominal"}`}>
                    {isAnomaly ? "TRUE ANOMALY" : "LEO"}
                  </span>
                </div>
                <div className="callout-metrics mono">
                  Alt: {p.altitudeKm.toFixed(1)} km · Vel: {p.velocityKms.toFixed(2)} km/s
                </div>
              </div>
            );
          })}
      </div>

      {/* 4a. Day 7: Satellite Inspector (typed, selection-driven) */}
      {isInspectorOpen && (
        <InspectorPanel
          snapshot={snapshot.kind === "ready" ? snapshot.snapshot : null}
          satellites={satellites}
          selectedSatId={selectedSatId}
          onSelect={handleSelectSatellite}
          onFocus={handleFocusSatellite}
          onClearSelection={() => setSelectedSatId(null)}
          onClose={() => setIsInspectorOpen(false)}
          focusAvailable={viewMode === "3d" && !sceneError}
        />
      )}

      {/* Day 9 + Day 22: schema editor and registry-scale editor, one rail item */}
      {isSettingsOpen && (
        <div className="settings-panels" data-testid="settings-panels">
          <div className="settings-panels__mode" role="tablist" aria-label="Parameter editor mode">
            <button
              type="button"
              role="tab"
              aria-selected={settingsMode === "registry"}
              className={settingsMode === "registry" ? "settings-panels__tab settings-panels__tab--active" : "settings-panels__tab"}
              onClick={() => setSettingsMode("registry")}
            >
              REGISTRY
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={settingsMode === "schema"}
              className={settingsMode === "schema" ? "settings-panels__tab settings-panels__tab--active" : "settings-panels__tab"}
              onClick={() => setSettingsMode("schema")}
            >
              CONTRACT
            </button>
          </div>
          {settingsMode === "registry" ? (
            <RegistryEditor
              activeScenarioId={activeScenarioId}
              onClose={() => setIsSettingsOpen(false)}
              onApplied={() => setSelectedSatId(null)}
            />
          ) : (
            <ParameterEditor
              activeScenarioId={activeScenarioId}
              onClose={() => setIsSettingsOpen(false)}
              onApplied={() => setSelectedSatId(null)}
            />
          )}
        </div>
      )}

      {/* 4b. Telemetry Drawer (Right Slideout Inspector) */}
      <aside
        className={`scene-shell__drawer ${isDrawerOpen ? "scene-shell__drawer--open" : ""}`}
        aria-label="Telemetry Data Grid"
      >
        <div className="drawer-header">
          <div className="drawer-title">
            <span className="drawer-dot" />
            SWARM TELEMETRY INSPECTOR
          </div>
          <button
            type="button"
            className="drawer-close"
            onClick={() => setIsDrawerOpen(false)}
            aria-label="Close Inspector"
          >
            ✕
          </button>
        </div>

        {snapshot.kind === "loading" && (
          <div className="drawer-note">
            <span className="loading-screen__spinner" /> waiting for snapshot…
          </div>
        )}

        {snapshot.kind === "rejected" && (
          <div className="scene-shell__error drawer-error" data-testid="snapshot-error">
            <div>{snapshot.reason}</div>
            <ul>
              {snapshot.errors.map((e) => (
                <li key={e.path} className="mono">{`${e.path}: ${e.message}`}</li>
              ))}
            </ul>
          </div>
        )}

        {snapshot.kind === "ready" && (
          <div className="snapshot" data-testid="snapshot-view">
            <div className="snapshot__clock mono">
              {`scenario ${snapshot.snapshot.scenario_id} · epoch ${snapshot.snapshot.clock.epoch_utc} · t = ${snapshot.snapshot.clock.simulation_time_s.toFixed(1)} s · ×${snapshot.snapshot.clock.time_scale} · ${snapshot.snapshot.clock.paused ? "paused" : "running"} · ${snapshot.snapshot.satellites.length} satellites`}
            </div>

            {snapshot.snapshot.satellites.length > SCENE_CONFIG.fleet.telemetryRowLimit && (
              <div className="drawer-note" data-testid="telemetry-window">
                {`showing ${SCENE_CONFIG.fleet.telemetryRowLimit} of ${snapshot.snapshot.satellites.length} satellites · use the inspector to select any ID`}
              </div>
            )}

            <div className="snapshot__scroll">
              <table className="snapshot__table tactical-table">
                <thead>
                  <tr>
                    <th>IDENT</th>
                    <th>POSITION ECI (km)</th>
                    <th>VELOCITY ECI (km/s)</th>
                    <th>SMA / INC</th>
                    <th>STATE</th>
                  </tr>
                </thead>
                <tbody>
                  {/* Day 11: a 1,000-satellite fleet must not mount 1,000 table
                      rows. The camera, selection and 3D markers still address all
                      of them; the table is a bounded window with a count. */}
                  {snapshot.snapshot.satellites
                    .slice(0, SCENE_CONFIG.fleet.telemetryRowLimit)
                    .map((sat, i) => (
                      <SatelliteRow
                        key={sat.id}
                        sat={sat}
                        color={getSatelliteColor(i)}
                        isSelected={selectedSatId === sat.id}
                        onSelect={() => handleSelectSatellite(sat.id)}
                      />
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </aside>

      {/* 5. Bottom Aerospace Timeline Scrubber Rail */}
      <footer className="scene-shell__scrubber-bar" role="toolbar" aria-label="Simulation Controls">
        {/* Left Playback & Clock Controls */}
        <div className="scrubber-controls" role="group" aria-label="Playback controls">
          <button
            type="button"
            className="scrubber-btn"
            onClick={handleReset}
            disabled={!!sceneError}
            title="Reset to Epoch (t=0)"
            aria-label="Reset to Epoch"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
            </svg>
          </button>

          <button
            type="button"
            className={`scrubber-btn scrubber-btn--primary ${isPaused ? "" : "scrubber-btn--active"}`}
            onClick={isPaused ? handleResume : handlePause}
            disabled={!!sceneError}
            aria-label={isPaused ? "Play" : "Pause"}
          >
            {isPaused ? (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
            ) : (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="4" width="4" height="16" />
                <rect x="14" y="4" width="4" height="16" />
              </svg>
            )}
          </button>

          <button
            type="button"
            className="scrubber-btn"
            onClick={handleStep}
            disabled={!!sceneError || !isPaused}
            title="Single Step (1s)"
            aria-label="Step 1 second"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 4 15 12 5 20 5 4" />
              <line x1="19" y1="5" x2="19" y2="19" stroke="currentColor" strokeWidth="2" />
            </svg>
          </button>

          <div className="scrubber-timewarp">
            <select
              id={timeWarpId}
              value={displayTimeWarp}
              onChange={handleTimeScaleChange}
              disabled={!!sceneError}
              className="scrubber-select mono"
              title="Simulation speed multiplier"
            >
              {/* Include the live scale even when the parameter editor sets a
                  value outside the preset list, so the control never lies. */}
              {(TIME_SCALE_OPTIONS.includes(displayTimeWarp)
                ? TIME_SCALE_OPTIONS
                : [...TIME_SCALE_OPTIONS, displayTimeWarp].sort((a, b) => a - b)
              ).map((scale) => (
                <option key={scale} value={scale}>{formatScale(scale)}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Center Timeline Track with Live UTC readout */}
        <div className="scrubber-timeline">
          <span className="scrubber-endpoint mono">-24h</span>
          <div className="timeline-track">
            <div
              className="timeline-progress"
              style={{
                width: `${Math.min(100, Math.max(5, (simSeconds % 86400) / 864))}%`,
              }}
            />
            <div
              className="timeline-thumb"
              style={{
                left: `${Math.min(100, Math.max(5, (simSeconds % 86400) / 864))}%`,
              }}
            />
          </div>
          <span className="timeline-timestamp mono">{formattedTimestamp}</span>
          <span className="scrubber-endpoint mono">+24h</span>
        </div>

        {/* Right Camera & Projection Tools */}
        <div className="scrubber-tools" role="group" aria-label="Camera and projection tools">
          <button
            type="button"
            className="tool-btn"
            onClick={() => scene.current?.reset()}
            disabled={!!sceneError}
            title="Reset Camera View"
            aria-label="Reset View"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>RESET</span>
          </button>

          <button
            type="button"
            className="tool-btn tool-btn--icon"
            onClick={() => scene.current?.zoom("in")}
            disabled={!!sceneError}
            title="Zoom In"
            aria-label="Zoom in"
          >
            +
          </button>

          <button
            type="button"
            className="tool-btn tool-btn--icon"
            onClick={() => scene.current?.zoom("out")}
            disabled={!!sceneError}
            title="Zoom Out"
            aria-label="Zoom out"
          >
            −
          </button>

          <button
            type="button"
            className={`tool-btn ${showOrbits ? "tool-btn--active" : ""}`}
            onClick={() => setShowOrbits((v) => !v)}
            title="Toggle Orbit Paths"
            aria-label="Toggle Orbits"
            aria-pressed={showOrbits}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <ellipse cx="12" cy="12" rx="10" ry="4.5" transform="rotate(-28 12 12)" />
              <circle cx="19" cy="6.5" r="1.8" />
            </svg>
          </button>

          <button
            type="button"
            className={`tool-btn ${showOverlays ? "tool-btn--active" : ""}`}
            onClick={() => setShowOverlays((v) => !v)}
            title="Toggle Tactical Coordinate Grid & Polar Reticle"
            aria-label="Toggle Grid"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="2" y1="12" x2="22" y2="12" />
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          </button>

          {/* Tactical 3D Orientation Gizmo (X, Y, Z axes matching reference image) */}
          <div className="scrubber-gizmo" title="ECI Coordinate Frame (+Z North)">
            <svg width="34" height="34" viewBox="0 0 40 40" className="gizmo-svg">
              <line x1="20" y1="20" x2="36" y2="28" stroke="#FF765F" strokeWidth="2" />
              <text x="37" y="32" fill="#FF765F" fontSize="8" fontFamily="monospace">X</text>
              <line x1="20" y1="20" x2="6" y2="28" stroke="#6F9B97" strokeWidth="2" />
              <text x="1" y="32" fill="#6F9B97" fontSize="8" fontFamily="monospace">Y</text>
              <line x1="20" y1="20" x2="20" y2="4" stroke="#3D68B2" strokeWidth="2" />
              <text x="23" y="8" fill="#3D68B2" fontSize="8" fontFamily="monospace">Z</text>
              <circle cx="20" cy="20" r="2.5" fill="#E7E9ED" />
            </svg>
          </div>
        </div>
      </footer>

      {/* Day 8: in-app scenario catalog. The core validates and loads; the renderer
          just resumes polling the snapshot stream from the new scenario. */}
      {isScenarioOpen && (
        <div
          className="startup-modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label="Scenario catalog"
          onClick={() => setIsScenarioOpen(false)}
        >
          <div className="startup-modal" onClick={(event) => event.stopPropagation()}>
            <div className="startup-modal__header">
              <div className="startup-modal__title-group">
                <span className="drawer-dot" />
                <h2 className="startup-modal__title">SCENARIO CATALOG</h2>
              </div>
              <button
                type="button"
                className="startup-modal__close-btn"
                onClick={() => setIsScenarioOpen(false)}
                aria-label="Close scenario catalog"
              >
                ✕
              </button>
            </div>
            <div className="startup-modal__body">
              <ScenarioCatalog
                activeScenarioId={activeScenarioId}
                onLoaded={() => {
                  setSelectedSatId(null);
                  setIsScenarioOpen(false);
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* WebGL Unavailable Fallback */}
      {sceneError && (
        <div className="scene-shell__unavailable" role="alert">
          <h2>3D Viewport Unavailable</h2>
          <p>WebGL is required. Enable hardware acceleration or use a WebGL-capable browser.</p>
          <p className="mono">{sceneError}</p>
        </div>
      )}
    </section>
  );
}