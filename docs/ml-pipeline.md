# Learned-policy pipeline (Days 46–56)

This is the operator guide for the trained-swarm-policy half of ConstellaLab.
Everything here is offline: the renderer and the live simulation never import
`constellab_ml`, and a model checkpoint can only reach the simulator through the
environment and the independent safety shield. The plan of record is
[daywise-plan.md](../assets/docs/daywise-plan.md); the G3 decision is
[g3-gate.md](g3-gate.md) with the machine-readable report in
[g3-gate.json](../benchmarks/g3-gate.json).

## Status

**Day 56 complete; G3 passes for the promoted candidate.** The promoted policy is
the GNN+GRU imitation checkpoint selected by the Day 45 curriculum
(`runs/curriculum-demo/seed-0/03-full/checkpoint-best.pt`). The PPO path is
implemented, trained, benchmarked and retained as a measured negative result:
it did not beat the imitation candidate on the frozen held-out suite.

The current campaign is `demo-48` (48 episodes: 30 train / 9 validation /
9 test, seed 7) plus six freshly sampled held-out episodes for the Day 46 suite.
This is a smoke-scale campaign, not a published generalization claim.

## Commands

All commands run from the repository root and write machine-readable artefacts:

```bash
# Day 46: slice report (fleet size, family, condition, OOD, compatibility)
node scripts/py.js -m constellab_ml generalize configs/generalize.json \
  --out benchmarks/generalization.json

# Day 47/49: PPO and the multi-seed RL curriculum
node scripts/py.js -m constellab_ml ppo configs/ppo-long.json --out runs/ppo-long
node scripts/py.js -m constellab_ml rl configs/rl-curriculum.json --out runs/rl-demo

# Day 48: shield boundary suite (16 cases, incl. a hostile policy)
node scripts/py.js -m constellab_ml shield --campaign data/ml/demo --split test \
  --out benchmarks/shield-suite.json

# Day 51: manifest + compatibility/OOD/fallback suite
node scripts/py.js -m constellab_ml manifest \
  runs/curriculum-demo/seed-0/03-full/checkpoint-best.pt \
  --campaign data/ml/demo --out runs/g3/model-manifest.json
node scripts/py.js -m constellab_ml compat runs/g3/model-manifest.json \
  --campaign data/ml/demo --split test --limit 3 \
  --out benchmarks/compatibility.json

# Day 50: one-factor-at-a-time ablation table
node scripts/py.js -m constellab_ml ablations configs/ablations.json \
  --out benchmarks/ablations.json

# Day 52: paired comparison on identical episodes, every policy shielded
node scripts/py.js -m constellab_ml benchmark configs/benchmark-policies.json \
  --out benchmarks/policy-comparison.json

# Days 53–54: mutation search (random baseline, same mutation space)
node scripts/py.js -m constellab_ml search configs/search-partitioned-fail.json \
  --out benchmarks/search-partitioned-fail.json
node scripts/py.js -m constellab_ml search configs/search-partitioned-pass.json \
  --out benchmarks/search-partitioned-pass.json
node scripts/py.js -m constellab_ml search configs/search-faulted-fail.json \
  --out benchmarks/search-faulted-fail.json
node scripts/py.js -m constellab_ml search configs/search-faulted-pass.json \
  --out benchmarks/search-faulted-pass.json

# Day 55: mutation ranker, filtered by scenario family
node scripts/py.js -m constellab_ml rank configs/ranker.json --out runs/ranker

# Day 56: G3 promotion gate
node scripts/py.js -m constellab_ml gate configs/g3-gate.json \
  --out benchmarks/g3-gate.json --markdown docs/g3-gate.md
```

`gate` exits `0` only on `pass`, `5` when a required criterion fails, and `2`
on a malformed bundle. `--no-replay` exists for inspection; a report produced
that way fails the replay criterion by construction.

## What each day adds

### Day 46 — held-out and out-of-distribution slices
`evaluation.build_generalization_suite` freezes the test episodes plus optional
freshly sampled held-out episodes. Every episode is labelled with fleet bucket,
exact unseen fleet, scenario family, condition class, OOD status and parameter
compatibility. Reports include a 500-resample bootstrap interval per slice.
The suite deliberately includes fleet sizes absent from training. In `demo-48`
**all 15 test episodes are OOD**, so the `ood` slice is the whole suite; the
report does not pretend this is a second, independent OOD set.

### Day 47 — centralized training, decentralized execution
`ppo.ActorCritic` is a graph actor with a shared per-satellite scoring head plus
a centralized value head. `train_ppo` stores rollouts, computes GAE, applies the
clipped objective, value loss, entropy bonus, gradient clipping and writes
`checkpoint-best.pt` selected on **validation coverage**. `PPOPolicy.act` is
deterministic; sampling is only used during training. The checkpoint format is
`ppo-checkpoint-1` and is auto-detected by `evaluation.load_policy_checkpoint`.

### Day 48 — independent safety shield
Every proposed action passes `shield.SafetyShield.screen` before the simulator
sees it. Rejections carry stable reason codes: unknown satellite/task/kind,
not-in-mask, satellite unavailable/busy, task unavailable, deadline margin,
unknown resource, energy/storage floor, duplicate assignment and the per-step
cap. A blocked action produces the configured fallback (`wait` or `none`).
`ShieldedPolicy.summary()` aggregates screening statistics across every episode
of a run, not just the last one. `shield_suite()` produces a reproducible
16-case boundary table, including a hostile policy that tries to switch the
shield off; the action contract rejects invented action kinds before the shield
is even reached.

### Day 49 — multi-seed RL curriculum
`rl_curriculum` trains through declared stages (fleet/target limits, fault
requirement, step budget), carries the best checkpoint forward, evaluates on a
fixed validation set every stage and freezes the winner before any test report.
`runs/rl-demo` records **two seeds and six completed stages**; one seed
collapsed to zero validation coverage and is retained as a negative result.

### Day 50 — ablations
`ablations` evaluates one deployable `full` variant and named diagnostics
(no-shield, no-communication, no-parameter-context, no-task-features,
PPO-finetuned). Only `full` may be deployable; the no-shield row is for offline
measurement and is never presented as usable. The table is generated from raw
per-episode results, not hand-written.

### Day 51 — compatibility and fallback
`ModelManifest` records checkpoint hash, model family, observation/environment
versions, feature-schema hash, parameter-schema hash, training campaign,
configuration and the **training envelope** (fleet size, targets, faults,
partitions, energy, sensor duty cycle, families and condition classes).
`compatibility_report` distinguishes three cases: `compatible`,
`out-of-distribution` (numeric extrapolation, fallback to baseline) and
`incompatible` (unknown semantics or schema/hash mismatch, fallback to
baseline). `compatibility_suite` exercises all three on a real campaign and
checks the declared fallback deterministically.

### Day 52 — frozen policy comparison
`run_policy_benchmark` evaluates fixed-priority, greedy, wait, MLP, DeepSets,
GNN+GRU and PPO-finetuned on the **same 15 episodes**, with every policy behind
the same shield. Reports include per-episode rows, bootstrap intervals, shield
rejection counts and p95 inference cost. Candidate selection is not allowed to
read the test split.

### Days 53–54 — mutation grammar and labeled search data
Faults are typed events with target selector, start condition, duration,
intensity, dependency and scope. The mutation engine is a registry over
`shift-fault`, `extend-fault`, `change-target`, `combine-faults`,
`delay-delivery` and `trigger-at-transition`; every mutation has a deterministic
ID and parent ID, and invalid mutations are rejected with a reason. The search
objective exposes every score component (new failure, distinct failure,
severity, quickness, cost). `SearchDataset` keeps failures **and** non-failures,
deduplicates by canonical spec hash and writes an audit line per execution.
The checked-in mixed dataset is 96 records (47 failures / 49 non-failures)
from four campaigns over `partitioned` and `faulted` families; all campaigns use
the same mutation space and budget. The failing campaigns run
`distributed-greedy`, the passing campaigns run the transparent `local-greedy`
baseline, and policy identity is deliberately not a ranker input: the label is
"did this mutation fail under the baseline policy that executed it". This is a
baseline-relative dataset, not a claim that the same mutation passes for every
policy.

### Day 55 — neural candidate ranker
`ranker.MutationRanker` encodes scenario/mutation context and a masked set of
fault tokens, then predicts failure likelihood. Splitting is by **scenario
family**; `ranker-manifest.json` records class balance, calibration, top-k
precision/recall and whether the split was family-held-out. The ranker is
advisory and can be left out of search (`configs/search*.json` has no ranker
key), which is the same random path used for the baseline.

### Day 56 — the G3 gate
`gate.run_gate` reads saved evidence, verifies the checkpoint against its
manifest, and checks:

| Criterion | Threshold in `configs/g3-gate.json` |
|---|---|
| Artifact | checkpoint hash, schema, versions and non-empty envelope must match |
| Generalization | candidate within 0.02 coverage of the strongest baseline; OOD coverage ≥ 0.20; zero conflicts/invalid actions; p95 ≤ 250 ms |
| Benchmark | ≥ 3 policies; candidate within 0.02 coverage; shield rejection ≤ 35 %; candidate evaluated with the shield |
| Shield | ≥ 10 cases, all pass, fallback declared, cannot be disabled by policy output |
| Compatibility | all cases pass, incompatible and OOD cases present, baseline fallback demonstrated |
| Ablation | ≥ 4 variants, exactly `full` deployable, no-shield marked diagnostic |
| RL curriculum | ≥ 2 seeds, validation-based selection, completed stages and checkpoints |
| Replay | two clean re-executions of saved test seeds produce identical trace hashes |
| Ranker | optional/adopted evidence; not required for G3 |

Missing or malformed evidence yields `blocked`, not a pass. A failed candidate
stays in `rejected` and the declared simpler baseline remains the fallback.

## Checked-in evidence

All tables below are generated from the checked-in JSON; no number is typed by
hand. See [benchmarks/g3-gate.json](../benchmarks/g3-gate.json) for the full
report and per-criterion measurements.

### Day 46: generalization, 15 held-out episodes

| Policy | Mean task coverage | Bootstrap 95 % | p95 inference |
|---|---:|---:|---:|
| expert-rule | 0.6262 | 0.5275–0.7217 | 0.05 ms |
| first-feasible | 0.5995 | 0.4869–0.7017 | 0.03 ms |
| wait | 0.0000 | 0.0000–0.0000 | 0.02 ms |
| MLP | 0.6328 | 0.5250–0.7248 | 4.04 ms |
| DeepSets | 0.6328 | 0.5235–0.7224 | 6.81 ms |
| **GNN+GRU (promoted)** | **0.6376** | 0.5383–0.7277 | 6.14 ms |

### Day 52: paired benchmark, all policies shielded

| Policy | Coverage | Mean reward | Invalid actions | Shield rejected | p95 |
|---|---:|---:|---:|---:|---:|
| fixed-priority | 0.6170 | −0.0637 | 0 | 26 / 2219 | 0.10 ms |
| greedy | 0.5902 | −0.0692 | 0 | 25 / 2197 | 0.09 ms |
| wait | 0.0000 | −0.1421 | 0 | 0 / 2226 | 0.06 ms |
| MLP | 0.6237 | −0.0629 | 0 | 26 / 2219 | 4.40 ms |
| DeepSets | 0.6237 | −0.0629 | 0 | 26 / 2219 | 7.50 ms |
| **GNN+GRU** | **0.6284** | **−0.0619** | 0 | 26 / 2219 | 6.85 ms |
| PPO-finetuned | 0.5826 | −0.0693 | 0 | 25 / 2197 | 8.47 ms |

The shield rejects 1.17 % of proposals (energy floor) for the promoted policy;
no invalid action reaches the simulator. The unshielded reference is 0.6328
coverage, so the shield costs about 0.004 coverage on this suite.

### Day 50: ablations, 15 episodes

| Variant | Mean coverage | Deployable | Note |
|---|---:|---|---|
| full | 0.6170 | yes | promoted candidate + shield |
| no-shield | 0.6328 | no | diagnostic only |
| no-communication | 0.6170 | no | contact features removed; no decision changed on this suite |
| no-parameter-context | 0.6170 | no | parameter tokens removed; no decision changed on this suite |
| no-task-features | 0.6170 | no | task features zeroed |
| PPO-finetuned | 0.5820 | no | retained negative result |

The identical feature-ablation rows are reported as measured: the promoted
policy's argmax did not flip on this suite, so those features are not proven
useful by this experiment. They remain in the model; the table does not
manufacture a difference.

### Day 49: RL campaign

Two seeds, three stages each. Seed 0: 0.6667 → 0.8750 → 0.7500 validation
coverage. Seed 1 collapsed to 0.0 at every stage and is retained. Winner:
seed 0, stage `medium`, validation coverage 0.875. On the frozen test suite the
winner scores 0.6114, below the imitation candidate's 0.6376, so G3 promotes
the imitation candidate and keeps the RL winner as a measured negative result.

### Day 48: shield suite

16/16 cases pass. 12 of 13 reason codes are exercised directly; `unknown-kind`
is rejected earlier by the action contract (`Action.__post_init__`). The
hostile-policy case confirms a model cannot disable the shield.

### Day 51: compatibility suite

6 cases: 3 compatible, 2 incompatible, 1 out-of-distribution. Accepted envelope:
fleet 2–24, targets 1–14, faults 0–6, partitions 0–2, energy ratio 0.504–0.987,
sensor duty cycle 0.265–0.949. Incompatible and OOD cases fall back to the
baseline and record the reason.

### Days 53–55: adversarial search and ranker

96 mutations, 47 failures / 49 non-failures, all valid, deduplicated by spec
hash. Ranker: 96 records, family-held-out split, validation AUC **0.835**,
average precision **0.785**, top-16 precision **0.688** / recall **0.458** on
the held-out families.

## In the app

The evidence is read-only. Start the dev stack with `npm run dev`, then either
click `VIEW EVIDENCE` on the startup HUD or use the graph icon in the left rail.

- **Startup evidence** loads `GET /api/g2` and `GET /api/ml` and shows the G2
  verdict, held-out coverage, strongest baseline and shield result. Its buttons
  deep-link into either lab.
- **G3 / ML Evidence Lab** has eight tabs: gate, generalization, benchmark,
  pipeline (model manifest + Day 45 curriculum + Day 39 baselines), safety,
  ablations, training (RL + ranker) and search.
- **G2 Failure Lab** replays the recorded partition episode with its timeline,
  links, events and invariant violation.

`GET /api/ml` reads the fixed checked-in paths in `constellab_core.ml.ARTIFACTS`;
it never loads a checkpoint or accepts a filesystem path. Python changes require
a core restart because `npm run dev` does not hot-reload the Python service.

## Reproduce from a clean checkout

The checked-in run directories are generated artefacts, not source. To rebuild
the evidence:

```bash
npm install
# Python environment: follow README → fetch embedded Python or use Python ≥ 3.10
node scripts/py.js -m constellab_ml campaign data/ml/demo --id demo-48 --episodes 48 --seed 7
node scripts/py.js -m constellab_ml curriculum configs/curriculum.json --out runs/curriculum-demo
node scripts/py.js -m constellab_ml rl configs/rl-curriculum.json --out runs/rl-demo
node scripts/py.js -m constellab_ml ppo configs/ppo-long.json --out runs/ppo-long
# then the generalize/shield/compat/ablations/benchmark/search/rank/gate commands above
```

`runs/` and `data/ml/` are local artefacts; the JSON evidence and manifests are
checked in so a reviewer can inspect the decision without retraining.

## Known limits

- **Demo scale.** 48 episodes, 15 evaluated test episodes and a small graph
  model. Confidence intervals are wide and the OOD slice is the whole test split.
- **The promoted policy is imitation, not RL.** PPO works end-to-end and is
  reproducible, but it lost on the frozen suite. That is an honest negative
  result, not a hidden fallback.
- **No live policy in the viewport.** The app surfaces G2 and a read-only G3/ML
  evidence lab; the promoted policy still does not drive the live simulation.
- **Shield scope.** It enforces observation-level constraints (mask, state,
  deadline, resources, duplicates, cap). It is not a physical flight-safety
  proof and cannot certify values the observation does not contain.
- **Search objective is deliberately simple.** Learned ranking is evaluated
  offline; guided search itself is Day 57, not part of G3.
- **Parameter/task ablations showed no behavioral change** on this suite; that
  is reported rather than papered over.
