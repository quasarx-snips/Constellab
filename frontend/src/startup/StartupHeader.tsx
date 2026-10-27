import { useState } from "react";
import { publicAssetUrl } from "../assetUrl";

export type HudNavMode = "OBSERVE" | "ANALYZE" | "UNDERSTAND";

interface StartupHeaderProps {
  activeMode?: HudNavMode;
  onModeChange?: (mode: HudNavMode) => void;
}

export function StartupHeader({
  activeMode = "UNDERSTAND",
  onModeChange,
}: StartupHeaderProps) {
  const [currentMode, setCurrentMode] = useState<HudNavMode>(activeMode);

  const handleModeClick = (mode: HudNavMode) => {
    setCurrentMode(mode);
    onModeChange?.(mode);
  };

  return (
    <header className="startup-header" role="banner">
      {/* Left Brand & Telemetry Status */}
      <div className="startup-header__left">
        <div className="startup-header__brand">
          <img
            src={publicAssetUrl("constellab_symbol.svg")}
            alt="constellab"
            className="startup-header__icon"
          />
          <span className="startup-header__title">constellab</span>
        </div>

        <span className="startup-header__separator" aria-hidden="true">|</span>

        <div className="startup-header__telemetry">
          <span className="startup-telemetry-dot" aria-hidden="true" />
          <span className="startup-telemetry-label">SWARM TELEMETRY</span>
        </div>
      </div>

      {/* Right Navigation / System Phase Modes */}
      <nav className="startup-header__nav" aria-label="HUD Modes">
        <button
          type="button"
          className={`startup-nav-item ${currentMode === "OBSERVE" ? "startup-nav-item--active" : ""}`}
          onClick={() => handleModeClick("OBSERVE")}
        >
          OBSERVE
          {currentMode === "OBSERVE" && <span className="startup-nav-indicator" />}
        </button>

        <span className="startup-nav-slash" aria-hidden="true">/</span>

        <button
          type="button"
          className={`startup-nav-item ${currentMode === "ANALYZE" ? "startup-nav-item--active" : ""}`}
          onClick={() => handleModeClick("ANALYZE")}
        >
          ANALYZE
          {currentMode === "ANALYZE" && <span className="startup-nav-indicator" />}
        </button>

        <span className="startup-nav-slash" aria-hidden="true">/</span>

        <button
          type="button"
          className={`startup-nav-item ${currentMode === "UNDERSTAND" ? "startup-nav-item--active" : ""}`}
          onClick={() => handleModeClick("UNDERSTAND")}
        >
          UNDERSTAND
          {currentMode === "UNDERSTAND" && <span className="startup-nav-indicator" />}
        </button>
      </nav>
    </header>
  );
}
