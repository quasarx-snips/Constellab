/**
 * Days 46–56: the G3 / ML evidence lab.
 *
 * Every value here comes from `/api/ml`, which is a compact view of the
 * checked-in reports. The panel never recomputes a gate result, never loads a
 * checkpoint, and never hides a failed criterion: missing artefacts and failed
 * criteria are rendered with their reason.
 */
import { useMemo, useState, type ReactNode } from "react";
import type {
  MlAblations,
  MlCompatibility,
  MlCriterion,
  MlGate,
  MlPipeline,
  MlPolicy,
  MlRL,
  MlRanker,
  MlReport,
  MlSearch,
  MlShield,
  MlSuite,
} from "../api/ml";

interface MLLabProps {
  report: MlReport | null;
  error: string | null;
  loading: boolean;
  onRefresh: () => void;
  onClose: () => void;
}

type TabKey = "gate" | "generalization" | "benchmark" | "pipeline" | "safety" | "ablations" | "training" | "search";

const TABS: { key: TabKey; label: string }[] = [
  { key: "gate", label: "G3 GATE" },
  { key: "generalization", label: "GENERALIZATION" },
  { key: "benchmark", label: "BENCHMARK" },
  { key: "pipeline", label: "PIPELINE" },
  { key: "safety", label: "SAFETY" },
  { key: "ablations", label: "ABLATIONS" },
  { key: "training", label: "TRAINING" },
  { key: "search", label: "SEARCH" },
];

const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(digits)}%`;

const fixed = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);

const ms = (value: number | null | undefined) =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : `${value.toFixed(2)} ms`;

const short = (value: string | null | undefined, length = 16) =>
  !value ? "—" : value.length > length ? `${value.slice(0, length)}…` : value;

const basename = (value: string | null | undefined) => {
  if (!value) return "—";
  const parts = value.split(/[\\/]/);
  return parts[parts.length - 1] || value;
};

const artifactLabel: Record<string, string> = {
  gate: "G3",
  generalization: "GEN",
  benchmark: "BENCH",
  shield: "SHIELD",
  compatibility: "COMPAT",
  ablations: "ABLATE",
  search: "SEARCH",
  rl: "RL",
  ranker: "RANKER",
  baselines: "DAY 39",
  curriculum: "LADDER",
  manifest: "MANIFEST",
};

function valueText(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) return value.length ? value.map(valueText).join(", ") : "none";
  if (typeof value === "object") {
    const text = JSON.stringify(value);
    return text.length > 72 ? `${text.slice(0, 69)}…` : text;
  }
  return String(value);
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: "pass" | "fail" | "warn" }) {
  return (
    <div className={`ml-lab__metric ${tone ? `ml-lab__metric--${tone}` : ""}`}>
      <span className="ml-lab__metric-label">{label}</span>
      <span className="ml-lab__metric-value mono">{value}</span>
    </div>
  );
}

function Pill({ children, tone }: { children: ReactNode; tone: "pass" | "fail" | "warn" | "muted" }) {
  return <span className={`ml-lab__pill ml-lab__pill--${tone}`}>{children}</span>;
}

function Empty({ children }: { children: ReactNode }) {
  return <p className="ml-lab__empty">{children}</p>;
}

function SuiteFacts({ suite }: { suite: MlSuite }) {
  const inventory = suite.inventory || {};
  const testEpisodes = Number(inventory.test_episodes ?? suite.episodes ?? 0);
  const sampled = Number(inventory.sampled_extra ?? 0);
  return (
    <div className="ml-lab__facts">
      <Metric label="split" value={suite.split ?? "—"} />
      <Metric label="seed" value={String(suite.seed ?? "—")} />
      <Metric label="stored test" value={String(testEpisodes)} />
      <Metric label="sampled held-out" value={String(sampled)} />
      <Metric label="total" value={String(testEpisodes + sampled)} />
    </div>
  );
}

function SliceChips({ counts }: { counts?: Record<string, number> }) {
  if (!counts) return null;
  return (
    <div className="ml-lab__chips">
      {Object.entries(counts)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([name, count]) => (
          <span key={name} className="ml-lab__chip mono">
            {name} <b>{count}</b>
          </span>
        ))}
    </div>
  );
}

function PolicyTable({
  policies,
  promoted,
  showShield = false,
}: {
  policies: MlPolicy[];
  promoted: string | null;
  showShield?: boolean;
}) {
  if (policies.length === 0) return <Empty>No policy rows in this report.</Empty>;
  return (
    <div className="ml-lab__table-wrap">
      <table className="ml-lab__table mono">
        <thead>
          <tr>
            <th>policy</th>
            <th>coverage</th>
            <th>95% CI</th>
            <th>reward</th>
            <th>conf</th>
            <th>invalid</th>
            {showShield && <th>shield rej</th>}
            <th>p95</th>
          </tr>
        </thead>
        <tbody>
          {policies.map((policy) => {
            const shield = policy.shield;
            const rate = shield?.proposed ? `${shield.rejected}/${shield.proposed}` : "—";
            return (
              <tr key={policy.name} className={policy.name === promoted ? "ml-lab__row--promoted" : ""}>
                <td>
                  <span className="ml-lab__policy-name">{policy.name}</span>
                  {policy.name === promoted && <span className="ml-lab__promoted-tag">PROMOTED</span>}
                </td>
                <td>{pct(policy.coverage.mean, 2)}</td>
                <td className="ml-lab__muted">{`${pct(policy.coverage.low, 1)}–${pct(policy.coverage.high, 1)}`}</td>
                <td>{fixed(policy.reward.mean, 4)}</td>
                <td>{policy.conflicts}</td>
                <td>{policy.invalid_actions}</td>
                {showShield && (
                  <td title={Object.entries(shield?.reason_counts || {}).map(([k, v]) => `${k}: ${v}`).join(", ")}>
                    {rate}
                  </td>
                )}
                <td>{ms(policy.latency_ms.p95)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function GateView({ gate, report }: { gate: MlGate; report: MlReport }) {
  const [showAll, setShowAll] = useState(false);
  const criteria = showAll ? gate.criteria : gate.criteria.filter((item) => item.required || !item.passed);
  const ranges = Object.entries(gate.accepted_envelope.ranges || {});
  const statusTone = gate.status === "pass" ? "pass" : gate.status === "fail" ? "fail" : "warn";
  return (
    <section className="ml-lab__section" data-testid="ml-gate-view">
      <div className={`ml-lab__verdict ml-lab__verdict--${statusTone}`}>
        <div>
          <span className="ml-lab__verdict-kicker">G3 TRAINED SWARM POLICY GATE</span>
          <h2 className="ml-lab__verdict-title">{gate.status.toUpperCase()}</h2>
        </div>
        <div className="ml-lab__verdict-side mono">
          <span>{gate.required_failures.length ? `${gate.required_failures.length} required failure(s)` : "all required criteria pass"}</span>
        </div>
      </div>
      <div className="ml-lab__facts">
        <Metric label="promoted" value={gate.promoted ?? "—"} tone={gate.status === "pass" ? "pass" : "fail"} />
        <Metric label="fallback" value={gate.fallback ?? "—"} />
        <Metric label="candidate" value={gate.candidate ?? "—"} />
        <Metric label="checkpoint" value={basename(gate.checkpoint)} />
      </div>

      <h3 className="ml-lab__section-title">ACCEPTANCE CRITERIA</h3>
      <ul className="ml-lab__criteria">
        {criteria.map((criterion) => (
          <CriterionRow key={criterion.key} criterion={criterion} />
        ))}
      </ul>
      {gate.criteria.some((item) => !item.required) && (
        <button type="button" className="ml-lab__link-button" onClick={() => setShowAll((value) => !value)}>
          {showAll ? "hide optional criteria" : "show optional evidence (ranker)"}
        </button>
      )}

      <h3 className="ml-lab__section-title">ACCEPTED PARAMETER ENVELOPE</h3>
      {ranges.length === 0 ? (
        <Empty>The manifest carries no numerical envelope.</Empty>
      ) : (
        <div className="ml-lab__envelope">
          {ranges.map(([field, range]) => (
            <div key={field} className="ml-lab__envelope-row mono">
              <span>{field}</span>
              <span className="ml-lab__muted">{`${valueText(range.min)} → ${valueText(range.max)}`}</span>
            </div>
          ))}
        </div>
      )}
      {gate.accepted_envelope.families && gate.accepted_envelope.families.length > 0 && (
        <p className="ml-lab__muted mono">families: {gate.accepted_envelope.families.join(" · ")}</p>
      )}

      <h3 className="ml-lab__section-title">ARTEFACT INTEGRITY</h3>
      <div className="ml-lab__artifact-table">
        {Object.values(report.artifacts).map((artifact) => (
          <div key={artifact.key} className="ml-lab__artifact-row mono">
            <span className={artifact.available ? "ml-lab__dot ml-lab__dot--pass" : "ml-lab__dot ml-lab__dot--fail"} />
            <span>{artifactLabel[artifact.key] || artifact.key}</span>
            <span className="ml-lab__muted">{artifact.available ? artifact.path : artifact.error || "missing"}</span>
            <span className="ml-lab__muted">{artifact.available ? `${Math.round((artifact.bytes || 0) / 1024)} KB` : "—"}</span>
          </div>
        ))}
      </div>
      {gate.selection_note && <p className="ml-lab__note">{gate.selection_note}</p>}
    </section>
  );
}

function CriterionRow({ criterion }: { criterion: MlCriterion }) {
  const measured = Object.entries(criterion.measured || {}).slice(0, 5);
  return (
    <li className={`ml-lab__criterion ${criterion.passed ? "ml-lab__criterion--pass" : "ml-lab__criterion--fail"}`}>
      <div className="ml-lab__criterion-head">
        <span className="ml-lab__criterion-mark" aria-hidden="true">{criterion.passed ? "✓" : "✗"}</span>
        <span className="ml-lab__criterion-title">{criterion.title}</span>
        {criterion.required ? <Pill tone="muted">required</Pill> : <Pill tone="warn">optional</Pill>}
        <Pill tone={criterion.passed ? "pass" : "fail"}>{criterion.passed ? "PASS" : "FAIL"}</Pill>
      </div>
      {criterion.reasons.length > 0 && (
        <p className="ml-lab__error-text">{criterion.reasons.join(" · ")}</p>
      )}
      <div className="ml-lab__measured">
        {measured.map(([key, value]) => (
          <span key={key} className="ml-lab__chip mono">
            {key} <b>{valueText(value)}</b>
          </span>
        ))}
      </div>
      <details className="ml-lab__details">
        <summary>thresholds + measurements</summary>
        <pre className="ml-lab__pre mono">{JSON.stringify({ thresholds: criterion.thresholds, measured: criterion.measured }, null, 2)}</pre>
      </details>
    </li>
  );
}

function GeneralizationView({ report, promoted }: { report: MlReport; promoted: string | null }) {
  const { suite, policies } = report.generalization;
  const [selected, setSelected] = useState(() => promoted && policies.some((p) => p.name === promoted) ? promoted : policies[0]?.name || "");
  const policy = policies.find((item) => item.name === selected) || policies[0];
  const slices = useMemo(
    () => Object.entries(policy?.slices || {}).sort(([a], [b]) => a.localeCompare(b)),
    [policy],
  );
  return (
    <section className="ml-lab__section" data-testid="ml-generalization-view">
      <h3 className="ml-lab__section-title">FROZEN HELD-OUT SUITE</h3>
      <SuiteFacts suite={suite} />
      <SliceChips counts={suite.slice_counts} />
      <p className="ml-lab__note">All policies run on the same stored episodes. The test split is read only after selection.</p>

      <h3 className="ml-lab__section-title">POLICY COVERAGE</h3>
      <PolicyTable policies={policies} promoted={promoted} />
      <p className="ml-lab__note">
        Coverage is the fraction of tasks observed. CI is a 500-resample bootstrap over episodes; a wide interval means the sample is small.
      </p>

      <h3 className="ml-lab__section-title">SLICES</h3>
      {!policy ? (
        <Empty>No policy slices in this report.</Empty>
      ) : (
        <>
          <label className="ml-lab__select-label mono">
            policy
            <select className="ml-lab__select" value={policy.name} onChange={(event) => setSelected(event.target.value)}>
              {policies.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}
            </select>
          </label>
          <div className="ml-lab__table-wrap">
            <table className="ml-lab__table mono">
              <thead><tr><th>slice</th><th>coverage</th><th>episodes</th><th>expired</th><th>conflicts</th></tr></thead>
              <tbody>
                {slices.map(([name, row]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{pct(row.coverage, 2)}</td>
                    <td>{row.episodes}</td>
                    <td>{row.expired}</td>
                    <td>{row.conflicts}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {report.generalization.selection_note && <p className="ml-lab__note">{report.generalization.selection_note}</p>}
    </section>
  );
}

function PipelineView({ pipeline }: { pipeline: MlPipeline }) {
  const { baselines, curriculum, manifest } = pipeline;
  const variation = curriculum?.seed_variation.metrics || {};
  const testCoverage = variation.test_mean_coverage;
  return (
    <section className="ml-lab__section" data-testid="ml-pipeline-view">
      <h3 className="ml-lab__section-title">TRAINED MODEL MANIFEST // DAY 51</h3>
      {!manifest ? (
        <Empty>Model manifest unavailable. Run `npm run ml -- manifest`.</Empty>
      ) : (
        <>
          <div className="ml-lab__facts">
            <Metric label="family" value={manifest.model_family ?? "—"} tone="pass" />
            <Metric label="hidden / layers" value={`${manifest.model.hidden ?? "—"} / ${manifest.model.layers ?? "—"}`} />
            <Metric label="campaign" value={manifest.trained_campaign ?? "—"} />
            <Metric label="env / obs" value={`${manifest.env_version ?? "—"} / ${manifest.observation_version ?? "—"}`} />
            <Metric label="checkpoint hash" value={short(manifest.checkpoint_hash, 12)} />
          </div>
          <div className="ml-lab__envelope">
            {Object.entries(manifest.envelope.ranges).map(([field, range]) => (
              <div key={field} className="ml-lab__envelope-row mono">
                <span>{field}</span>
                <span className="ml-lab__muted">{`${valueText(range.min)} → ${valueText(range.max)}`}</span>
              </div>
            ))}
          </div>
          {manifest.envelope.families.length > 0 && (
            <p className="ml-lab__muted mono">families: {manifest.envelope.families.join(" · ")}</p>
          )}
          <p className="ml-lab__note">
            {manifest.training.epochs ?? "—"} epochs × {manifest.training.batches_per_epoch ?? "—"} batches · lr {manifest.training.learning_rate ?? "—"} · seed {manifest.training.seed ?? "—"}
            {manifest.training.max_sequence_samples ? ` · ${manifest.training.max_sequence_samples} sequence samples` : ""}
          </p>
          <p className="ml-lab__note mono">
            feature schema {short(manifest.feature_schema_hash, 12)} · parameter schema {short(manifest.parameter_schema_hash, 12)} · checkpoint {basename(manifest.checkpoint)}
          </p>
        </>
      )}

      <h3 className="ml-lab__section-title">CURRICULUM LADDER // DAY 45</h3>
      {!curriculum ? (
        <Empty>Curriculum manifest unavailable. Run `npm run ml -- curriculum`.</Empty>
      ) : (
        <>
          <div className="ml-lab__facts">
            <Metric label="curriculum" value={curriculum.curriculum_id ?? "—"} />
            <Metric label="seeds" value={String(curriculum.seed_variation.seeds ?? curriculum.seeds.length)} />
            <Metric label="selected seed" value={String(curriculum.selected_seed.seed ?? "—")} />
            <Metric label="stage" value={curriculum.selected_seed.stage ?? "—"} />
            <Metric label="selection" value={`${curriculum.selected_seed.metric ?? "—"} ${fixed(curriculum.selected_seed.value ?? null, 4)}`} />
            <Metric label="test coverage" value={testCoverage ? `${pct(testCoverage.mean, 2)} ± ${fixed(testCoverage.std, 3)}` : "—"} />
          </div>
          <div className="ml-lab__table-wrap">
            <table className="ml-lab__table mono">
              <thead><tr><th>seed</th><th>stage</th><th>status</th><th>val loss</th><th>val acc</th><th>coverage</th><th>transfer</th></tr></thead>
              <tbody>
                {curriculum.seeds.flatMap((seed) => seed.stages.map((stage) => (
                  <tr key={`${seed.seed}-${stage.stage}`} className={stage.status === "failed" ? "ml-lab__row--fail" : ""}>
                    <td>{seed.seed}</td>
                    <td>{stage.stage}</td>
                    <td>{stage.status}</td>
                    <td>{fixed(stage.validation_loss, 3)}</td>
                    <td>{pct(stage.validation_accuracy, 1)}</td>
                    <td>{pct(stage.mean_coverage, 1)}</td>
                    <td className="ml-lab__muted">{stage.initialized_from_previous ? "yes" : "—"}</td>
                  </tr>
                )))}
              </tbody>
            </table>
          </div>
          {Object.entries(variation).map(([metric, row]) => (
            <p key={metric} className="ml-lab__note mono">
              {metric}: mean {fixed(row.mean, 4)} · std {fixed(row.std, 4)} · range {fixed(row.min, 3)}–{fixed(row.max, 3)}
            </p>
          ))}
          <p className="ml-lab__note">{curriculum.selection_rule || "Selection uses validation data only; the test split is read after the choice is frozen."}</p>
        </>
      )}

      <h3 className="ml-lab__section-title">DAY 39 MODEL BASELINES</h3>
      {!baselines ? (
        <Empty>Baseline report unavailable. Run `npm run ml -- evaluate`.</Empty>
      ) : (
        <>
          <div className="ml-lab__facts">
            <Metric label="campaign" value={baselines.campaign ?? "—"} />
            <Metric label="split" value={baselines.split ?? "—"} />
            <Metric label="episodes" value={String(baselines.episodes ?? "—")} />
            <Metric label="policies" value={String(baselines.policies.length)} />
          </div>
          <div className="ml-lab__table-wrap">
            <table className="ml-lab__table mono">
              <thead><tr><th>policy</th><th>coverage</th><th>reward</th><th>conflicts</th><th>invalid</th></tr></thead>
              <tbody>
                {baselines.policies.map((row) => (
                  <tr key={row.name}>
                    <td>{row.name}</td>
                    <td>{pct(row.coverage, 2)}</td>
                    <td>{fixed(row.reward, 4)}</td>
                    <td>{row.conflicts}</td>
                    <td>{row.invalid_actions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="ml-lab__note">{baselines.selection_note || "Comparison only; checkpoints were selected on validation data before this test split was read."}</p>
        </>
      )}
    </section>
  );
}

function BenchmarkView({ report, promoted }: { report: MlReport; promoted: string | null }) {
  const { suite, policies } = report.benchmark;
  const shielded = policies.filter((policy) => policy.shield).length;
  return (
    <section className="ml-lab__section" data-testid="ml-benchmark-view">
      <h3 className="ml-lab__section-title">PAIRED POLICY COMPARISON</h3>
      <SuiteFacts suite={suite} />
      <div className="ml-lab__facts">
        <Metric label="policies" value={String(policies.length)} />
        <Metric label="shielded" value={`${shielded}/${policies.length}`} />
        <Metric label="winner" value={promoted ?? "—"} tone={promoted ? "pass" : undefined} />
      </div>
      <PolicyTable policies={policies} promoted={promoted} showShield />
      <p className="ml-lab__note">
        {shielded === policies.length && shielded > 0
          ? "Every row was measured behind the same independent shield; rejection is the shield's energy/state floor, not an invalid action reaching the simulator."
          : "Shield coverage differs between rows; compare rejection counts only within the same shield configuration."}
      </p>
      {report.benchmark.selection_note && <p className="ml-lab__note">{report.benchmark.selection_note}</p>}
    </section>
  );
}

function SafetyView({ shield, compatibility }: { shield: MlShield | null; compatibility: MlCompatibility | null }) {
  const summary = shield?.summary;
  return (
    <section className="ml-lab__section" data-testid="ml-safety-view">
      <h3 className="ml-lab__section-title">INDEPENDENT ACTION SHIELD</h3>
      {!shield ? (
        <Empty>Shield suite report unavailable. Run `npm run ml -- shield`.</Empty>
      ) : (
        <>
          <div className="ml-lab__facts">
            <Metric label="cases" value={`${summary?.passed ?? 0}/${summary?.cases ?? shield.cases.length}`} tone={shield.passed ? "pass" : "fail"} />
            <Metric
              label="reason codes"
              value={`${summary?.reason_codes_covered?.length ?? 0}/${(summary?.reason_codes_covered?.length ?? 0) + (summary?.reason_codes_missing?.length ?? 0)}`}
              tone={(summary?.reason_codes_missing?.length ?? 0) > 0 ? "warn" : "pass"}
            />
            <Metric label="fallback" value={summary?.fallback ?? "—"} />
            <Metric label="disable via policy" value={String(summary?.disable_via_policy_output ?? false)} tone={summary?.disable_via_policy_output ? "fail" : "pass"} />
          </div>
          <div className="ml-lab__chips">
            {(summary?.reason_codes_covered || []).map((code) => <span key={code} className="ml-lab__chip mono">{code}</span>)}
            {(summary?.reason_codes_missing || []).map((code) => (
              <span key={code} className="ml-lab__chip ml-lab__chip--warn mono" title="not exercised by this suite">{code} *</span>
            ))}
          </div>
          <div className="ml-lab__table-wrap">
            <table className="ml-lab__table mono">
              <thead><tr><th>case</th><th>result</th><th>expected reason</th><th>observed</th></tr></thead>
              <tbody>
                {shield.cases.map((item) => (
                  <tr key={item.name}>
                    <td title={item.description}>{item.name}</td>
                    <td className={item.passed ? "ml-lab__pass-text" : "ml-lab__fail-text"}>{item.passed ? "PASS" : "FAIL"}</td>
                    <td className="ml-lab__muted">{valueText(item.expected.reason)}</td>
                    <td>{Object.entries(item.observed.reason_counts).map(([code, count]) => `${code}:${count}`).join(" ") || (item.observed.fallback.length ? "fallback" : "accepted")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="ml-lab__note">
            The shield is an operational constraint layer, not a physical flight-safety proof. Unknown values are blocked, never approved.
            {summary?.unknown_kind_blocked_at_contract ? " The unknown-kind case is rejected by the contract before the shield, so it is not counted as a shield reason code." : ""}
          </p>
        </>
      )}

      <h3 className="ml-lab__section-title">MODEL COMPATIBILITY + FALLBACK</h3>
      {!compatibility ? (
        <Empty>Compatibility suite unavailable. Run `npm run ml -- compat`.</Empty>
      ) : (
        <>
          <div className="ml-lab__facts">
            {Object.entries(compatibility.counts).map(([status, count]) => (
              <Metric key={status} label={status} value={String(count)} />
            ))}
            <Metric label="suite" value={compatibility.passed ? "pass" : "fail"} tone={compatibility.passed ? "pass" : "fail"} />
          </div>
          <div className="ml-lab__envelope">
            {Object.entries(compatibility.accepted_envelope.ranges || {}).map(([field, range]) => (
              <div key={field} className="ml-lab__envelope-row mono">
                <span>{field}</span>
                <span className="ml-lab__muted">{`${valueText(range.min)} → ${valueText(range.max)}`}</span>
              </div>
            ))}
          </div>
          <div className="ml-lab__table-wrap">
            <table className="ml-lab__table mono">
              <thead><tr><th>case</th><th>status</th><th>fallback</th><th>OOD</th><th>reasons</th></tr></thead>
              <tbody>
                {compatibility.cases.map((item) => (
                  <tr key={item.name}>
                    <td title={item.name}>{item.name}</td>
                    <td>{item.status ?? "—"}</td>
                    <td>{item.fallback ?? "—"}</td>
                    <td>{fixed(item.ood_score, 3)}</td>
                    <td className="ml-lab__muted">{item.reasons.join(", ") || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="ml-lab__note">
            {compatibility.fallback_contract || "Unsupported configurations fall back to the baseline."}
            {compatibility.manifest.model_family ? ` · model ${compatibility.manifest.model_family}` : ""}
            {compatibility.manifest.checkpoint_hash ? ` · hash ${short(compatibility.manifest.checkpoint_hash, 12)}` : ""}
          </p>
        </>
      )}
    </section>
  );
}

function AblationsView({ ablations }: { ablations: MlAblations | null }) {
  if (!ablations) {
    return <Empty>Ablation report unavailable. Run `npm run ml -- ablations`.</Empty>;
  }
  return (
    <section className="ml-lab__section" data-testid="ml-ablations-view">
      <h3 className="ml-lab__section-title">REWARD / FEATURE / ALGORITHM ABLATIONS</h3>
      <div className="ml-lab__table-wrap">
        <table className="ml-lab__table mono">
          <thead><tr><th>variant</th><th>coverage</th><th>reward</th><th>conflicts</th><th>deployable</th><th>note</th></tr></thead>
          <tbody>
            {ablations.variants.map((variant) => (
              <tr key={variant.name} className={variant.diagnostic ? "" : "ml-lab__row--promoted"}>
                <td>{variant.name}</td>
                <td>{pct(variant.metrics.mean_coverage, 2)}</td>
                <td>{fixed(variant.metrics.mean_reward, 4)}</td>
                <td>{variant.metrics.conflicts}</td>
                <td>{variant.diagnostic ? <Pill tone="warn">diagnostic</Pill> : <Pill tone="pass">deployable</Pill>}</td>
                <td className="ml-lab__muted">{variant.note || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="ml-lab__note">
        {ablations.selection_note || "One factor at a time; diagnostic variants are retained as negative results."}
      </p>
      <p className="ml-lab__note">
        The no-shield row is offline measurement only. It is never a deployment option.
      </p>
    </section>
  );
}

function TrainingView({ rl, ranker }: { rl: MlRL | Record<string, never>; ranker: MlRanker | Record<string, never> }) {
  const rlView = (rl as Partial<MlRL>).run_id ? (rl as MlRL) : null;
  const rankerView = (ranker as Partial<MlRanker>).metrics ? (ranker as MlRanker) : null;
  return (
    <section className="ml-lab__section" data-testid="ml-training-view">
      <h3 className="ml-lab__section-title">MULTI-SEED RL CAMPAIGN</h3>
      {!rlView ? (
        <Empty>RL curriculum unavailable. Run `npm run ml -- rl`.</Empty>
      ) : (
        <>
          <div className="ml-lab__facts">
            <Metric label="run" value={rlView.run_id ?? "—"} />
            <Metric label="seeds" value={String(rlView.summary.seeds ?? 0)} />
            <Metric label="stages" value={String(rlView.summary.stages_completed ?? 0)} />
            <Metric
              label="validation mean"
              value={pct(rlView.summary.validation_coverage?.mean, 1)}
              tone={(rlView.summary.validation_coverage?.min ?? 0) === 0 ? "warn" : undefined}
            />
            <Metric
              label="winner"
              value={rlView.summary.winner ? `seed ${rlView.summary.winner.seed} / ${rlView.summary.winner.stage} ${pct(rlView.summary.winner.value, 1)}` : "—"}
              tone="pass"
            />
          </div>
          <div className="ml-lab__table-wrap">
            <table className="ml-lab__table mono">
              <thead><tr><th>seed</th><th>stage</th><th>status</th><th>selection</th><th>fleet / targets</th><th>transfer</th></tr></thead>
              <tbody>
                {rlView.seeds.flatMap((seed) => seed.stages.map((stage) => (
                  <tr key={`${seed.seed}-${stage.stage}`} className={stage.selection.value === 0 ? "ml-lab__row--fail" : ""}>
                    <td>{seed.seed}</td>
                    <td>{stage.stage}</td>
                    <td>{stage.status}</td>
                    <td>{`${stage.selection.metric || "—"} ${pct(stage.selection.value, 1)}`}</td>
                    <td>{`${stage.fleet_limit ?? "—"} / ${stage.target_limit ?? "—"}`}</td>
                    <td className="ml-lab__muted">{stage.initialized_from_previous ? "yes" : "—"}</td>
                  </tr>
                )))}
              </tbody>
            </table>
          </div>
          <p className="ml-lab__note">
            {rlView.selection_note || "Checkpoints are selected on validation coverage before the held-out test is read."}
            {" "}A seed that collapses is retained rather than hidden.
          </p>
        </>
      )}

      <h3 className="ml-lab__section-title">MUTATION RANKER</h3>
      {!rankerView ? (
        <Empty>Ranker unavailable. Run `npm run ml -- rank`.</Empty>
      ) : (
        <>
          <div className="ml-lab__facts">
            <Metric label="records" value={String(rankerView.records ?? 0)} />
            <Metric label="validation AUC" value={fixed(rankerView.metrics.validation_auc, 3)} tone="pass" />
            <Metric label="average precision" value={fixed(rankerView.metrics.validation_average_precision, 3)} />
            <Metric label="split" value={rankerView.family_held_out ? "family-held-out" : "random"} />
            <Metric label="top-k precision" value={pct(rankerView.metrics.top_k.precision ?? null, 1)} />
          </div>
          <p className="ml-lab__note">
            {rankerView.split_note || ""}
            {rankerView.class_balance.failures !== undefined
              ? ` · ${rankerView.class_balance.failures} failures / ${rankerView.class_balance.non_failures ?? "?"} non-failures`
              : ""}
            {rankerView.metrics.top_k.k ? ` · top-${rankerView.metrics.top_k.k} recall ${pct(rankerView.metrics.top_k.recall ?? null, 1)}` : ""}
          </p>
          <p className="ml-lab__note">The ranker is advisory. Invariant findings remain authoritative, and the searcher can run without it.</p>
        </>
      )}
    </section>
  );
}

function SearchView({ search }: { search: MlSearch | null }) {
  if (!search) {
    return <Empty>Search evidence unavailable. Run the mutation-search commands in `docs/ml-pipeline.md`.</Empty>;
  }
  const failures = search.totals.failures ?? 0;
  const records = search.totals.records ?? 0;
  return (
    <section className="ml-lab__section" data-testid="ml-search-view">
      <h3 className="ml-lab__section-title">ADVERSARIAL MUTATION CAMPAIGNS</h3>
      <div className="ml-lab__facts">
        <Metric label="mutations" value={String(records)} />
        <Metric label="failures" value={String(failures)} tone={failures ? "warn" : undefined} />
        <Metric label="non-failures" value={String(search.totals.non_failures ?? records - failures)} />
        <Metric label="failure rate" value={pct(records ? failures / records : null, 1)} />
        <Metric label="budget / campaign" value={String(search.budget_per_campaign ?? "—")} />
      </div>
      <div className="ml-lab__table-wrap">
        <table className="ml-lab__table mono">
          <thead><tr><th>campaign</th><th>family</th><th>mode</th><th>seed</th><th>records</th><th>failures</th><th>operators</th></tr></thead>
          <tbody>
            {search.campaigns.map((campaign) => (
              <tr key={campaign.name}>
                <td>{campaign.name}</td>
                <td>{campaign.base_family}</td>
                <td>{campaign.mode}</td>
                <td>{campaign.seed}</td>
                <td>{campaign.records}</td>
                <td>{campaign.failures}</td>
                <td className="ml-lab__muted" title={Object.entries(campaign.operators).map(([name, row]) => `${name}: ${row.failures}/${row.evaluations}`).join(", ")}>
                  {Object.keys(campaign.operators).length} registered
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="ml-lab__note">
        {search.note || "Non-failures are retained; the learned ranker is not required for the random baseline."}
      </p>
      <p className="ml-lab__note">
        Labels are baseline-relative: failing campaigns use distributed-greedy, passing campaigns use local-greedy, and policy identity is not a ranker input.
      </p>
    </section>
  );
}

export function MLLab({ report, error, loading, onRefresh, onClose }: MLLabProps) {
  const [tab, setTab] = useState<TabKey>("gate");
  const promoted = report?.gate?.promoted ?? null;
  const gate = report?.gate ?? null;
  const statusTone = gate?.status === "pass" ? "pass" : gate?.status === "fail" ? "fail" : "warn";

  return (
    <aside className="ml-lab" data-testid="ml-lab" aria-label="G3 learned-policy evidence lab">
      <header className="ml-lab__header">
        <div>
          <h2 className="ml-lab__title">G3 // ML EVIDENCE LAB</h2>
          <p className={`ml-lab__subtitle mono ml-lab__subtitle--${statusTone}`}>
            {gate
              ? `${gate.status.toUpperCase()} · promoted ${gate.promoted ?? "—"} · fallback ${gate.fallback ?? "—"}`
              : report
                ? "report loaded without a gate decision"
                : "loading core evidence…"}
          </p>
        </div>
        <div className="ml-lab__header-actions">
          <button type="button" className="ml-lab__icon-button" onClick={onRefresh} title="Reload evidence" aria-label="Reload ML evidence">⟳</button>
          <button type="button" className="ml-lab__icon-button" onClick={onClose} title="Close ML lab" aria-label="Close ML lab">✕</button>
        </div>
      </header>

      {report && (
        <>
          <div className="ml-lab__artifacts" aria-label="Evidence artefacts">
            {Object.values(report.artifacts).map((artifact) => (
              <span
                key={artifact.key}
                className={`ml-lab__artifact-chip ${artifact.available ? "ml-lab__artifact-chip--ok" : "ml-lab__artifact-chip--missing"}`}
                title={artifact.available ? artifact.path : `${artifact.path}: ${artifact.error || "missing"}`}
              >
                <span className="ml-lab__dot" />
                {artifactLabel[artifact.key] || artifact.key}
              </span>
            ))}
          </div>
          <nav className="ml-lab__tabs" role="tablist" aria-label="ML evidence views">
            {TABS.map((item) => (
              <button
                key={item.key}
                type="button"
                role="tab"
                aria-selected={tab === item.key}
                className={tab === item.key ? "ml-lab__tab ml-lab__tab--active" : "ml-lab__tab"}
                onClick={() => setTab(item.key)}
              >
                {item.label}
              </button>
            ))}
          </nav>
        </>
      )}

      <div className="ml-lab__body">
        {loading && !report && !error && <p className="ml-lab__quiet">Reading `/api/ml` from the core…</p>}
        {error && <p className="ml-lab__error" role="alert">{error}</p>}
        {report && tab === "gate" && (gate ? <GateView gate={gate} report={report} /> : <Empty>No G3 report. Run `npm run ml -- gate`.</Empty>)}
        {report && tab === "generalization" && <GeneralizationView report={report} promoted={promoted} />}
        {report && tab === "benchmark" && <BenchmarkView report={report} promoted={promoted} />}
        {report && tab === "pipeline" && <PipelineView pipeline={report.pipeline} />}
        {report && tab === "safety" && <SafetyView shield={report.shield} compatibility={report.compatibility} />}
        {report && tab === "ablations" && <AblationsView ablations={report.ablations} />}
        {report && tab === "training" && <TrainingView rl={report.training.rl} ranker={report.training.ranker} />}
        {report && tab === "search" && <SearchView search={report.search} />}
      </div>
    </aside>
  );
}
