import { useEffect, useState } from "react";
import { publicAssetUrl } from "./assetUrl";
import { HudReticles } from "./startup/HudReticles";
import { StartupHeader, HudNavMode } from "./startup/StartupHeader";
import { StartupHero } from "./startup/StartupHero";
import { StartupFooter } from "./startup/StartupFooter";
import { StartupBackground } from "./startup/StartupBackground";
import { StartupModals } from "./startup/StartupModals";

const SPLASH_MS = 1600;

interface StartupScreenProps {
  onLaunch: (panel?: "ml" | "g2") => void;
}

/**
 * Startup Experience:
 * 1. Initial white splash with the app icon and the "constellab" wordmark (stage = "splash")
 * 2. High-fidelity Constellab HUD screen (stage = "enter"):
 *    - Corner HUD reticle brackets
 *    - Header: Constellab icon, title, separator, green status, "SWARM TELEMETRY", modes
 *    - Center Hero: Constellab vector symbol, highlighted "constellab" title, subtitle,
 *      and action pill buttons (ENTER SIMULATOR, EXPLORE SCENARIOS, VIEW EVIDENCE)
 *    - Footer: Bottom-left telemetry motto, bottom-right online status badge
 *    - Interactive space backdrop with starfield, Earth horizon, and orbital swarm paths
 *    - Pressing Enter launches the simulator directly
 */
export function StartupScreen({ onLaunch }: StartupScreenProps) {
  const [stage, setStage] = useState<"splash" | "enter">("splash");
  const [modalType, setModalType] = useState<"scenarios" | "evidence" | null>(null);

  // Transition from initial splash to HUD screen
  useEffect(() => {
    const timer = setTimeout(() => setStage("enter"), SPLASH_MS);
    return () => clearTimeout(timer);
  }, []);

  // Keyboard shortcut: pressing Enter launches the simulator (when no modal is open)
  // Pressing Escape closes any open modal
  useEffect(() => {
    if (stage !== "enter") return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Enter" && !modalType) {
        event.preventDefault();
        onLaunch();
      } else if (event.key === "Escape" && modalType) {
        event.preventDefault();
        setModalType(null);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [stage, modalType, onLaunch]);

  return (
    <div
      className="loading-screen startup-hud-screen"
      data-testid="startup-screen"
      data-stage={stage}
    >
      {/* Stage 2: Constellab HUD Telemetry Experience */}
      {stage === "enter" && (
        <div className="startup-hud-container">
          {/* 4 Corner HUD Reticles */}
          <HudReticles />

          {/* Background visuals: space artwork + dynamic canvas + atmosphere glow */}
          <StartupBackground />

          {/* Top Header Navbar */}
          <StartupHeader
            onModeChange={(mode: HudNavMode) => {
              if (mode === "ANALYZE" || mode === "UNDERSTAND") {
                setModalType("evidence");
              } else if (mode === "OBSERVE") {
                setModalType("scenarios");
              }
            }}
          />

          {/* Center Hero Section */}
          <main className="startup-hud-main">
            <StartupHero
              onEnterSimulator={onLaunch}
              onExploreScenarios={() => setModalType("scenarios")}
              onViewEvidence={() => setModalType("evidence")}
            />
          </main>

          {/* Bottom HUD Footer */}
          <StartupFooter />

          {/* Interactive Modal Dialogs (Scenarios & Evidence) */}
          <StartupModals
            isOpen={modalType !== null}
            type={modalType}
            onClose={() => setModalType(null)}
            onLaunch={onLaunch}
          />
        </div>
      )}

      {/* Stage 1: Initial White Splash */}
      <div
        className={`loading-screen__splash${stage === "splash" ? "" : " loading-screen__splash--hidden"}`}
        data-testid="splash"
        aria-hidden={stage !== "splash"}
      >
        <div className="splash__brand">
          <img className="splash__symbol" src={publicAssetUrl("constellab_symbol.svg")} alt="constellab" />
          <h1 className="splash__title">constellab</h1>
        </div>
      </div>
    </div>
  );
}
