/**
 * Days 46–56: read the core's read-only ML evidence payload.
 *
 * The payload is a compact view of the checked-in reports. This module only
 * validates shape (never recomputes a metric), so a UI bug cannot turn a
 * failing gate into a passing one.
 */
import { CORE_URL } from "../scenario/api";

export interface MlArtifact {
  key: string;
  path: string;
  available: boolean;
  error?: string;
  bytes?: number;
  generated_at_s?: number;
}

export interface MlInterval {
  mean: number | null;
  low: number | null;
  high: number | null;
}

export interface MlShieldSummary {
  version?: string;
  steps?: number;
  proposed?: number;
  accepted?: number;
  rejected?: number;
  acceptance_rate?: number | null;
  reason_counts?: Record<string, number>;
  fallback?: string;
  cases?: number;
  passed?: number;
  failed?: string[];
  reason_codes_covered?: string[];
  reason_codes_missing?: string[];
  disable_via_policy_output?: boolean;
  unknown_kind_blocked_at_contract?: boolean;
}

export interface MlPolicy {
  name: string;
  coverage: MlInterval;
  reward: MlInterval;
  observed: number;
  tasks_total: number;
  expired: number;
  conflicts: number;
  invalid_actions: number;
  latency_ms: { mean: number | null; p95: number | null };
  shield: MlShieldSummary | null;
  slices: Record<string, { coverage: number | null; episodes: number; expired: number; conflicts: number }>;
}

export interface MlCriterion {
  key: string;
  title: string;
  required: boolean;
  passed: boolean;
  reasons: string[];
  measured: Record<string, unknown>;
  thresholds: Record<string, unknown>;
}

export interface MlGate {
  status: "pass" | "fail" | "blocked";
  promoted: string | null;
  fallback: string | null;
  candidate: string | null;
  checkpoint: string | null;
  accepted_envelope: {
    ranges?: Record<string, { min: number | null; max: number | null }>;
    families?: string[];
    condition_classes?: string[];
  };
  criteria: MlCriterion[];
  required_failures: string[];
  retained: string[];
  rejected: string[];
  selection_note: string | null;
}

export interface MlSuite {
  split?: string;
  seed?: number;
  episodes?: number | null;
  inventory?: Record<string, unknown>;
  slice_counts?: Record<string, number>;
}

export interface MlShieldCase {
  name: string;
  description: string;
  expected: Record<string, unknown>;
  observed: { accepted: number; reason_counts: Record<string, number>; fallback: string[] };
  passed: boolean;
}

export interface MlShield {
  passed: boolean;
  source?: string;
  config: Record<string, unknown>;
  summary: MlShieldSummary;
  cases: MlShieldCase[];
}

export interface MlCompatibilityCase {
  name: string;
  kind: string;
  expected: Record<string, unknown>;
  passed: boolean;
  status: string | null;
  fallback: string | null;
  ood_score: number | null;
  reasons: string[];
  warnings: string[];
}

export interface MlCompatibility {
  passed: boolean;
  counts: Record<string, number>;
  failed: string[];
  accepted_envelope: MlGate["accepted_envelope"];
  fallback_contract: string | null;
  manifest: {
    checkpoint: string | null;
    model_family: string | null;
    checkpoint_hash: string | null;
    feature_schema_hash: string | null;
  };
  cases: MlCompatibilityCase[];
}

export interface MlAblationVariant {
  name: string;
  kind: string | null;
  transform: string | null;
  shield: boolean;
  diagnostic: boolean;
  note: string | null;
  checkpoint: string | null;
  coverage: MlInterval;
  metrics: {
    mean_coverage: number | null;
    mean_reward: number | null;
    conflicts: number;
    invalid_actions: number;
  };
}

export interface MlAblations {
  suite: Record<string, unknown>;
  selection_note: string | null;
  variants: MlAblationVariant[];
}

export interface MlRLStage {
  stage: string | null;
  status: string | null;
  selection: { metric?: string; value?: number | null; direction?: string };
  checkpoint: string | null;
  fleet_limit: number | null;
  target_limit: number | null;
  initialized_from_previous: boolean;
}

export interface MlRL {
  run_id: string | null;
  config: Record<string, unknown>;
  summary: {
    seeds?: number;
    stages_completed?: number;
    validation_coverage?: { count?: number; mean?: number | null; std?: number | null; min?: number | null; max?: number | null };
    winner?: { seed?: number; stage?: string; value?: number | null; checkpoint?: string | null } | null;
  };
  seeds: { seed: number | null; stages: MlRLStage[] }[];
  selection_note: string | null;
}

export interface MlRanker {
  version: string | null;
  records: number | null;
  train_records: number | null;
  validation_records: number | null;
  family_held_out: boolean;
  split_note: string | null;
  class_balance: Record<string, number>;
  metrics: {
    validation_loss: number | null;
    validation_auc: number | null;
    validation_average_precision: number | null;
    top_k: Record<string, number>;
    calibration: Record<string, unknown>[];
  };
  checkpoint: string | null;
  selection_note: string | null;
}

export interface MlSearchCampaign {
  name: string;
  base_episode: string;
  base_family: string;
  mode: string;
  seed: number;
  records: number;
  failures: number;
  class_balance: Record<string, number>;
  operators: Record<string, { evaluations: number; failures: number }>;
}

export interface MlSearch {
  totals: { records?: number; failures?: number; non_failures?: number };
  campaigns: MlSearchCampaign[];
  budget_per_campaign?: number;
  note?: string;
}

export interface MlBaselineRow {
  name: string;
  coverage: number | null;
  reward: number | null;
  episodes: number;
  observed: number;
  expired: number;
  conflicts: number;
  invalid_actions: number;
  tasks_total: number;
}

export interface MlBaselines {
  campaign: string | null;
  split: string | null;
  episodes: number | null;
  selection_note: string | null;
  policies: MlBaselineRow[];
}

export interface MlCurriculumStage {
  stage: string | null;
  status: string | null;
  samples: number;
  validation_loss: number | null;
  validation_accuracy: number | null;
  mean_coverage: number | null;
  initialized_from_previous: boolean;
  wall_time_s: number | null;
}

export interface MlCurriculum {
  curriculum_id: string | null;
  curriculum_version: string | null;
  selected_seed: { seed: number | null; stage: string | null; metric: string | null; value: number | null };
  selection_rule: string | null;
  seed_variation: {
    seeds: number | null;
    metrics: Record<string, { mean: number | null; std: number | null; min: number | null; max: number | null }>;
  };
  seeds: { seed: number | null; stages: MlCurriculumStage[] }[];
}

export interface MlModelManifest {
  version: string | null;
  checkpoint: string | null;
  checkpoint_hash: string | null;
  model_family: string | null;
  observation_version: string | null;
  env_version: string | null;
  feature_schema_hash: string | null;
  parameter_schema_hash: string | null;
  trained_campaign: string | null;
  model: { name: string | null; hidden: number | null; layers: number | null };
  training: {
    learning_rate: number | null;
    epochs: number | null;
    batches_per_epoch: number | null;
    graphs_per_batch: number | null;
    max_sequence_samples: number | null;
    seed: number | null;
    curriculum: Record<string, unknown>;
  };
  envelope: {
    ranges: Record<string, { min: number | null; max: number | null }>;
    families: string[];
  };
}

export interface MlPipeline {
  baselines: MlBaselines | null;
  curriculum: MlCurriculum | null;
  manifest: MlModelManifest | null;
}

export interface MlReport {
  version: string;
  request_id?: string;
  revision?: number;
  artifacts: Record<string, MlArtifact>;
  gate: MlGate | null;
  generalization: { suite: MlSuite; policies: MlPolicy[]; selection_note?: string | null };
  benchmark: { suite: MlSuite; policies: MlPolicy[]; selection_note?: string | null };
  shield: MlShield | null;
  compatibility: MlCompatibility | null;
  ablations: MlAblations | null;
  training: { rl: MlRL | Record<string, never>; ranker: MlRanker | Record<string, never> };
  search: MlSearch | null;
  pipeline: MlPipeline;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseMlReport(input: unknown): MlReport {
  if (!isObject(input) || typeof input.version !== "string") {
    throw new Error("ML report is missing a version");
  }
  if (!isObject(input.artifacts) || !isObject(input.generalization) || !isObject(input.benchmark)) {
    throw new Error("ML report is missing required sections");
  }
  if (input.gate !== null && !isObject(input.gate)) {
    throw new Error("ML report gate section is malformed");
  }
  return input as unknown as MlReport;
}

export async function fetchMlReport(): Promise<MlReport> {
  const res = await fetch(`${CORE_URL}/api/ml`);
  if (!res.ok) throw new Error(`core returned ${res.status} for /api/ml`);
  return parseMlReport(await res.json());
}
