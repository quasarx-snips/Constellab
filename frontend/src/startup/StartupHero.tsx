import { publicAssetUrl } from "../assetUrl";

interface StartupHeroProps {
  onEnterSimulator: () => void;
  onExploreScenarios?: () => void;
  onViewEvidence?: () => void;
}

export function StartupHero({
  onEnterSimulator,
  onExploreScenarios,
  onViewEvidence,
}: StartupHeroProps) {
  return (
    <div className="startup-hero">
      {/* Brand Lockup: Symbol on top of title, center aligned */}
      <div className="startup-hero__brand">
        <img
          src={publicAssetUrl("constellab_symbol.svg")}
          alt="constellab"
          className="startup-hero__symbol"
        />
        <h1 className="startup-hero__title" aria-label="constellab">
          constellab
        </h1>
      </div>

      {/* Action Buttons */}
      <div className="startup-hero__actions" role="group" aria-label="Startup actions">
        <button
          type="button"
          className="startup-btn startup-btn--primary"
          onClick={onEnterSimulator}
          data-testid="launch-button"
          autoFocus
        >
          ENTER SIMULATOR
        </button>

        <button
          type="button"
          className="startup-btn startup-btn--ghost"
          onClick={onExploreScenarios}
        >
          EXPLORE SCENARIOS
        </button>

        <button
          type="button"
          className="startup-btn startup-btn--ghost"
          onClick={onViewEvidence}
        >
          VIEW EVIDENCE
        </button>
      </div>
    </div>
  );
}
