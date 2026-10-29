# ConstellaLab — Roadmap

**Plan of record:** [assets/docs/daywise-plan.md](assets/docs/daywise-plan.md) (70 days, gates G0–G5).  
**Current position:** Day 56 of 70 complete. Days 1–56 are implemented and tested; G3 passes for the promoted policy. Day 57 is the next planned step. The learned-policy guide is [docs/ml-pipeline.md](docs/ml-pipeline.md).

| Gate | Days | Required result | Status |
|---|---:|---|---|
| **G0** — Visual orbital simulator | 1–14 | Earth, satellites, orbit paths, camera/time controls, selection and inspector from real core state | **Implemented**; rerun `npm run g0` for the automated checklist and the app for the visual items |
| **G1** — Configurable foundation | 15–24 | Schema-driven parameters, components, validation, seeded scenarios, scalable state | **Implemented**; `npm run test:core` regenerates [g1-acceptance.json](benchmarks/g1-acceptance.json) |
| **G2** — First reproducible swarm failure | 25–34 | Tasking/comms fault reproduces a baseline-policy violation; control and replay agree | **Passing**; [g2-acceptance.json](benchmarks/g2-acceptance.json), `GET /api/g2`, G2 Fault Lab |
| **G3** — Trained swarm policy | 35–56 | Multiple learned families evaluated against baselines; independent safety shield | **Passing**; [g3-gate.json](benchmarks/g3-gate.json), [g3-gate.md](docs/g3-gate.md) |
| **G4** — Learned adversarial testing | 57–62 | Learned search compared fairly with random/fixed search, with reproducible counterexamples | **Not started** |
| **G5** — Integrated build | 63–70 | Everything runs together, evidence pack verifies, clean-checkout demo | **Not started** |

---

## What is done (Days 1–56)

### Desktop and visual simulator
- Electron desktop shell with embedded Python, splash → landing page → viewport flow.
- Lit Earth viewport, camera orbit/zoom/reset, orbit paths, satellite markers, selection and inspector.
- Instanced fleet renderer and de-duplicated orbit paths; checked-in 1,000-satellite benchmark.
- Clock controls: pause, resume, reset, step one second, time scale with bounded easing.
- Diagnostics overlays: IDs, orbital planes, simulation time, positions, snapshot age.
- Loading, invalid, disconnected and error states; stale-revision replies are discarded.
- Startup HUD, mission-parameter editor generated from the contract's `x-ui` metadata.

### Contract and simulation core
- Versioned JSON-Schema contract (v1.2.0) with generated TypeScript types and independent validators on both sides.
- One Python core owns time, orbit propagation, randomness, events, component state and snapshots.
- Keplerian two-body dynamics behind a registered dynamics key; analytic propagation checked against an independent RK4 reference.
- Unit registry with kind-safe conversion; ECI/ECEF/geodetic/RSW frames and Earth/sun geometry.
- Virtual power, storage, compute, sensor, payload, task and event-source components in a stage-ordered runtime.
- Data-driven parameter registry with thousands of fields, migrations, schema hashing and an unconsumed-parameter audit.
- Plugin lifecycle (register → validate → ordered activate) with loud duplicate/host-version failures.
- Constraint-aware scenario sampler; built-in `swarm-48/100/500/1000` fixtures and inline config validation.
- Headless `constellab_core run` streams NDJSON at roughly 110k satellite-states/s.

### Swarm testing layer
- Configurable EO targets with a strict truth/observable split.
- Sensor plugins (`sensor.conical`, `sensor.agile`) with footprint geometry, resolution proxy, cloud mask, duty cycle and access windows.
- Generic task lifecycle with imaging and change-detection handlers, deadlines and observation outcomes.
- Policy interface with `local-greedy`, `fixed-priority` and diagnostic `oracle-upper-bound` baselines.
- Inter-satellite link models (`link.uhf`, `link.optical`) and deterministic contact plans.
- Message transport with bounded queues, bandwidth, latency, seeded loss/duplication/reordering, partitions and reconnect policy.
- Invariant monitor and baseline allocation comparison report.

### Failures, replay and the G2 gate
- Fault plans as data: `satellite-failure`, `sensor-degrade`, `link-cut`, `message-delay`; manual, seeded random and `worst-moment` strategies.
- Partition-aware claim bus with delivery delay, TTL and link cuts.
- Canonical event traces, per-subsystem/entity `SeedBank`, canonical serialization and trace hashes.
- Behavioural replay: the same harness re-runs the episode and compares trace hashes instead of re-printing stored state.
- Reproducible duplicate-claim failure, partition-free control, structured first violation and replayable manifest.
- G2 Fault Lab in the app (`GET /api/g2`) with gate checks, timeline scrubber, recorded links, events and fault-vs-control metrics.
- G3/ML Evidence Lab in the app (`GET /api/ml`) with gate, held-out, benchmark, pipeline, safety, ablation, training and search views; the startup evidence modal loads the same live G2/G3 reports and deep-links into either lab.

### Learned-policy lab (offline; never in the render loop)
- Versioned domain randomization: ten parameter families, latent groups, conditional constraints, held-out spans, coverage and invalid-combination reports.
- Access-anchored scenario sampler producing validated `EpisodeSpec`s and deterministic train/validation/test splits.
- Learned-policy environment with masked joint observations, variable fleet/task/edge counts, reward terms and episode rollouts.
- Deployable rule expert, diagnostic window-order expert, candidate scoring, greedy assignment and a leakage scan.
- Metadata-aware parameter-token encoder (value, validity, type, unit, bounds, namespace, stable field id) with compatibility/OOD reporting.
- PyTorch model families: MLP, DeepSets, GNN, GNN+GRU and GNN+attention over one variable-size graph interface.
- Training framework: declarative JSON configs, abstain-aware masked imitation loss, train-only normalization, seeded loaders, checkpoints, resume, early stop and export.
- Bounded, resumable sweeps with recorded failures and validation-only selection.
- Curriculum ladder (Day 45): staged fleet/task scale, inverse-frequency balancing, checkpoint transfer, early/stall stop, seed-variation report and held-out comparison against the expert and non-neural baselines.
- Campaign writer with manifests, config/seed/schema hashes, resumable batches and deterministic regeneration.
- CLI: `campaign`, `train`, `sweep`, `curriculum`, `evaluate`, all printing JSON summaries and writing full manifests.
- Checked-in reference comparison: [ml-baselines.json](benchmarks/ml-baselines.json) (rule expert, first-feasible, wait, MLP, DeepSets, GNN+GRU on identical held-out episodes).

### Days 46–56: G3 trained swarm policy
- Frozen generalization suite with fleet/family/condition/OOD slices, bootstrap intervals and p95 inference cost ([generalization.json](benchmarks/generalization.json)).
- PPO actor-critic (`ppo-checkpoint-1`): rollout storage, GAE, clipped objective, value/entropy losses, gradient clipping, resume, validation-only checkpoint selection and deterministic deployment.
- Independent safety shield with stable reason codes, documented fallback, lifetime rejection statistics and a 16-case boundary suite including a hostile policy ([shield-suite.json](benchmarks/shield-suite.json)).
- Multi-seed RL curriculum with stage transfer, validation selection and retained seed failures (`runs/rl-demo`).
- Declared ablations with exactly one deployable `full` variant and diagnostic no-shield/no-feature/PPO rows ([ablations.json](benchmarks/ablations.json)).
- Model manifest with checkpoint/schema/parameter hashes, accepted envelope, OOD score and baseline fallback; compatibility suite covers compatible/incompatible/OOD cases ([compatibility.json](benchmarks/compatibility.json)).
- Paired comparison of fixed-priority, greedy, wait, MLP, DeepSets, GNN+GRU and PPO-finetuned on 15 identical episodes with every policy shielded ([policy-comparison.json](benchmarks/policy-comparison.json)).
- Typed mutation registry with ancestry and reason codes; 96-record labelled search dataset with 47 failures and 49 non-failures ([adversarial-search.json](benchmarks/adversarial-search.json)).
- Family-held-out mutation ranker (validation AUC 0.835, AP 0.785) usable by search or disabled for the identical random baseline.
- G3 gate over saved evidence: artifact verification, thresholds, compatibility, shield, ablation, RL, optional ranker and deterministic replay; `pass` promotes the GNN+GRU imitation candidate with `fixed-priority` as fallback.

### Verification and tooling
- `npm test`: 630 Python tests plus TypeScript contract/scene/API/scenario tests, HTTP end-to-end tests and a production Vite build.
- Goldens for orbit propagation, access windows, episode traces, contact plans, message traces and fault replay hashes.
- G0/G1/G2 acceptance report generators.
- Fleet benchmark with machine metadata.
- Portable Windows `.exe` build path (`npm run dist:win`).

---

## What is left

### Immediate next: Day 57 — guided adaptive search
Integrate the trained ranker with the mutation engine while preserving an exploration fraction. Bound candidate count, simulation budget, memory and mutation depth; log the predicted score, actual outcome and policy checkpoint for every selection; make the learned ranker disableable so the random baseline keeps the same code path.

### G3 — trained swarm policy (Days 35–56) — complete
Promoted candidate: GNN+GRU imitation checkpoint from the Day 45 curriculum, selected on validation loss before the test split was read. Held-out coverage 0.6376 vs 0.6328 (MLP/DeepSets) and 0.6262 (rule expert); OOD coverage 0.6376; zero conflicts/invalid actions; p95 6.1 ms. The shield passes all 16 boundary cases and rejects 1.17 % of proposals (energy floor). PPO and the RL curriculum are implemented, reproducible and retained as measured negative results (0.5826 and 0.6114 held-out coverage). The promotion report is [g3-gate.json](benchmarks/g3-gate.json).

### G4 — learned adversarial testing (Days 57–62)
| Day | Work |
|---|---|
| 57 | Guided adaptive search with a preserved random-exploration fraction and full decision logging |
| 58 | Failure shrinking/minimization with a generic reducer interface and rerun checks |
| 59 | Counterfactual intervention replay with no-change and unrelated-change controls |
| 60 | Differential and metamorphic testing with first-divergence reports |
| 61 | Temporal invariant language, parser, validator and monitor |
| 62 | **G4 gate:** equal-budget comparison of learned vs random/fixed search, with resource costs and a retain/reject decision |

### G5 — integrated build (Days 63–70)
| Day | Work |
|---|---|
| 63 | Neural trace encoder for failure triage; rule signatures remain authoritative |
| 64 | Failure clustering and duplicate-report detection with raw traces retained |
| 65 | Evidence pack, hash-chained ledger, signatures and clean-environment verifier |
| 66 | Multi-fidelity execution (L0/L1/L2) and failure escalation |
| 67 | CI gates: typecheck, unit/property, replay, schema, tiny training smoke, checkpoint load, invariant regressions, evidence verify |
| 68 | Scalability, stress and fault-combination campaigns with bounded queues/memory |
| 69 | End-to-end integration and defect-fix pass from a clean environment |
| 70 | **G5 gate:** freeze the candidate, clean-checkout demo, model/search reports, verified evidence pack and known limitations |

---

## Known limits and deliberate gaps

- **The learned-policy results are smoke-scale.** The reference campaign is 48 episodes with 15 evaluated test episodes; all are OOD, so the OOD slice is the whole suite. It is not a published generalization claim.
- **The promoted policy is imitation, not RL.** PPO is implemented and reproducible but lost on the frozen suite; that negative result is retained rather than hidden.
- **Learned policies are offline only.** They do not drive the live simulation yet; the app surfaces both the G2 fault lab and the read-only G3/ML evidence lab (`GET /api/ml`), including the live startup evidence summary.
- **Feature ablations showed no behavioral change** on this suite (no-communication, no-parameter-context, no-task-features all 0.6170). The table reports the measurement honestly; it does not claim those features are proven useful.
- **Guided search, shrinking and interventions are Days 57–59.** The G3 gate uses the random mutation baseline only.
- **No evidence pack, temporal-rule language or multi-fidelity path yet.** These are Days 61–66.
- **The packaged `.exe` predates the latest ML/failure work** unless `npm run dist:win` is run again.
- **Python changes require a core restart.** `npm run dev` does not hot-reload the Python service; a running core keeps its old route table (this is what caused the `/api/g2` 404).
