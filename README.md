<p align="center"><img src="assets/icons/icon.png" alt="ConstellaLab logo" width="120"></p>

# ConstellaLab — Day 56: G3 Trained Swarm Policy

Software-only crash-test lab for autonomous satellite swarms — a Three.js Earth viewport built on the versioned Python/TypeScript simulation contract. Satellites orbit Earth from snapshots produced by the local Python core, which can also run headlessly for batch work.
(Echelon)

**Status:** Day 56 of 70 complete; G3 passes for the promoted policy. See [ROADMAP.md](ROADMAP.md) for the full done/next breakdown, [docs/ml-pipeline.md](docs/ml-pipeline.md) for the learned-policy guide and [assets/docs/daywise-plan.md](assets/docs/daywise-plan.md) for the plan of record.

## Quick Start

| Action | Command |
|---|---|
| Run Python core status | `npm run status` |
| Print the epoch snapshot | `npm run snapshot` |
| Run the simulation headlessly | `npm run headless -- swarm-48 --duration 3600 --step 5` |
| Regenerate TS contract types | `npm run gen:contract` |
| Start dev mode (core + Vite + Electron) | `npm run dev` |
| Build frontend | `npm run build` |
| Generate a demo campaign | `npm run ml -- campaign data/ml/demo --id demo-48 --episodes 48 --seed 7` |
| Train one configuration | `npm run ml -- train configs/train.json --out runs/train` |
| Run a bounded sweep | `npm run ml -- sweep configs/sweep.json --out runs/sweep` |
| Train the curriculum | `npm run ml -- curriculum configs/curriculum.json --out runs/curriculum` |
| Evaluate a checkpoint | `npm run ml -- evaluate data/ml/demo --checkpoint runs/train/checkpoint-best.pt` |
| Compare policy families | `npm run ml -- compare data/ml/demo --checkpoint mlp=runs/baseline-mlp/checkpoint-best.pt --checkpoint deepsets=runs/baseline-deepsets/checkpoint-best.pt --out benchmarks/ml-baselines.json` |
| Held-out/OOD slice report | `npm run ml -- generalize configs/generalize.json --out benchmarks/generalization.json` |
| Train PPO / RL curriculum | `npm run ml -- ppo configs/ppo-long.json --out runs/ppo-long` · `npm run ml -- rl configs/rl-curriculum.json --out runs/rl-demo` |
| Shield / compatibility suites | `npm run ml -- shield --campaign data/ml/demo --out benchmarks/shield-suite.json` · `npm run ml -- compat runs/g3/model-manifest.json --campaign data/ml/demo --out benchmarks/compatibility.json` |
| Ablations / full benchmark | `npm run ml -- ablations configs/ablations.json --out benchmarks/ablations.json` · `npm run ml -- benchmark configs/benchmark-policies.json --out benchmarks/policy-comparison.json` |
| Adversarial search / ranker | `npm run ml -- search configs/search-faulted-fail.json --out benchmarks/search-faulted-fail.json` · `npm run ml -- rank configs/ranker.json --out runs/ranker` |
| G3 promotion gate | `npm run ml -- gate configs/g3-gate.json --out benchmarks/g3-gate.json --markdown docs/g3-gate.md` |
| Fetch embedded Python | `npm run fetch-python` |
| Build .exe (portable) | `npm run dist:win` |
| Regenerate app icons | `npm run icon` |

The app opens with a brief `constellab` splash, then the landing page, then the viewport.

## .exe Application

**`dist-electron\ConstellaLab-<version>-portable.exe`** — portable Windows executable.

- Native desktop app (no browser, no internet required after build)
- Embedded Python 3.14 service (autofetched via `npm run fetch-python`)
- VSCode dark theme UI with system font
- Splash → landing page → viewport, all over a rotating Earth
- `npm run dist:win` verifies icons in both portable launcher and inner exe

## Days 31–56: failure injection, replay and the learned-policy lab

Two layers sit on top of the simulation core without changing it: a deterministic
failure/replay layer (Days 31–34) that the G2 fault lab surfaces in the viewport,
and an offline learned-policy pipeline (Days 35–56) that reads stored
demonstrations and never runs inside the render or simulation loop. Days 46–56
add the held-out suite, PPO/RL path, independent safety shield, compatibility
fallback, paired benchmark, adversarial mutation search, neural ranker and the
G3 gate. In the app, the startup evidence modal loads the live G2/G3 reports and
deep-links into the G2 fault lab or the eight-view G3/ML evidence lab
(`GET /api/ml`). The full guide is [docs/ml-pipeline.md](docs/ml-pipeline.md).

| Module | What it owns |
|---|---|
| `core/constellab_core/faults.py` | Fault plans as data: `satellite-failure`, `sensor-degrade`, `link-cut`, `message-delay`; manual, seeded random and `worst-moment` strategies |
| `core/constellab_core/trace.py` | Canonical event traces, `SeedBank` per subsystem/entity, replay manifests hashed by `(config, policy, seed, fault plan)` |
| `core/constellab_core/invariants.py` | Declarative rules (duplicate claim, expired assignment, missed deadline, rejected action, failed satellite holding a task, energy budget) and baseline allocation comparisons |
| `core/constellab_core/comms.py` | Partition-aware claim bus: delivery delay, TTL, link cuts; visibility is decided here, never in the UI |
| `core/constellab_core/g2.py` | The G2 gate: one reproducible duplicate-claim failure, its partition-free control, a structured first violation and a replayable manifest |
| `core/constellab_ml/randomization.py` | Versioned distribution set with correlated latent groups, conditional constraints, held-out spans and coverage/unexplored-region reports |
| `core/constellab_ml/scenarios.py` | Draw → validated `EpisodeSpec`: orbit/fleet/sensor/energy/storage/compute/task/network/fault/time parameters, access-anchored event schedules |
| `core/constellab_ml/env.py` | The learned-policy environment: masked joint observations, greedy matching, reward terms and episode rollout |
| `core/constellab_ml/expert.py` | Deployable rule expert + diagnostic window-order expert, candidate scoring, greedy assignment, leakage scan |
| `core/constellab_ml/param_encoder.py` | Metadata-aware parameter tokens (value, validity, type, unit, bounds, namespace, stable field id) and compatibility/OOD reports |
| `core/constellab_ml/models.py` | PyTorch policy families: MLP, DeepSets, GNN, GNN+GRU, GNN+attention, all sharing one variable-size graph interface |
| `core/constellab_ml/training.py` | Declarative configs, masked imitation loss with an abstain class, normalization, checkpoint/best selection, resume, export |
| `core/constellab_ml/sweep.py` | Bounded, resumable sweeps with recorded failures and frozen selection |
| `core/constellab_ml/curriculum.py` | Day 45 ladder: stage gating, inverse-frequency balancing, transfer, stall stop and held-out comparison against expert/baseline policies |
| `core/constellab_ml/evaluation.py` | Days 46/52: frozen suites, per-slice bootstrap reports and paired policy comparison |
| `core/constellab_ml/ppo.py` | Day 47: graph actor-critic PPO, GAE, clipped objective, value/entropy losses, checkpoint recovery and deterministic deployment |
| `core/constellab_ml/shield.py` | Day 48: independent action shield, stable reject codes, configurable fallback, lifetime statistics and a 16-case boundary suite |
| `core/constellab_ml/rl_curriculum.py` | Day 49: multi-seed staged RL, validation selection, checkpoint transfer and retained seed failures |
| `core/constellab_ml/ablations.py` | Day 50: one-factor-at-a-time variants generated from raw results |
| `core/constellab_ml/compatibility.py` | Day 51: model manifest, envelope, OOD score, safe fallback and compatibility suite |
| `core/constellab_ml/adversarial.py` | Days 53–54: typed mutation grammar, ancestry, visible search objective and labeled dataset with non-failures |
| `core/constellab_ml/ranker.py` | Day 55: family-held-out neural mutation ranker with calibration and top-k metrics |
| `core/constellab_ml/gate.py` | Day 56: G3 promotion gate over saved evidence, including deterministic replay |

Key guarantees:

- **Faults are data, not branches.** A plan is a typed, time-bounded list of
evidence-carrying events; the same plan can be replayed byte-for-byte and the
trace hash proves it. A fault never edits dynamics, task rules or policy code.
- **Replay is behavioural.** `g2.replay` re-runs the episode through the same
harness and compares canonical trace hashes; it never re-prints stored state.
- **The student never sees the simulator’s truth.** Stored demonstrations contain
only observations, actions and targets; `leakage_scan` audits the student view.
- **Variable fleet sizes are the default.** Graphs are batched by edge index and
masks, not padded to one satellite count; the same weights run on 3 or 24 nodes.
- **Selection is frozen before the test split.** Sweeps, the curriculum and the
RL ladder rank checkpoints on validation metrics only; the held-out test is
evaluated once, after the winner is fixed, and every report says so.
- **Training steps are not progress.** Curriculum stages early-stop and the
ladder stops when validation stalls; supervised loss, environment reward and
parameter coverage are recorded separately.
- **A model only proposes.** The independent shield screens every action before
the simulator sees it, records a stable reject code and supplies the fallback;
the action contract rejects invented action kinds.
- **Promotion is evidence-driven.** G3 reads saved reports, verifies the
checkpoint against its manifest, re-runs two saved seeds for matching trace
hashes and refuses to promote a candidate that only trained successfully.

```bash
# 1. demonstrations from randomized, validated episodes
npm run ml -- campaign data/ml/demo --id demo-48 --episodes 48 --seed 7

# 2. one training run (declarative config)
npm run ml -- train configs/train.json --out runs/train

# 3. bounded architecture/hyper-parameter sweep
npm run ml -- sweep configs/sweep.json --out runs/sweep

# 4. Day 45 curriculum: stages, transfer, held-out evaluation
npm run ml -- curriculum configs/curriculum.json --out runs/curriculum

# 5. compare a checkpoint against the rule expert and baselines
npm run ml -- evaluate data/ml/demo --checkpoint runs/train/checkpoint-best.pt

# 6. Day 39/41 report: rule expert vs neural families on identical episodes
npm run ml -- compare data/ml/demo \
  --checkpoint mlp=runs/baseline-mlp/checkpoint-best.pt \
  --checkpoint deepsets=runs/baseline-deepsets/checkpoint-best.pt \
  --checkpoint gnn_gru=runs/curriculum/seed-0/03-full/checkpoint-best.pt \
  --out benchmarks/ml-baselines.json

# 7. Days 46–56: generalization, RL, shield, compatibility, ablations, benchmark
npm run ml -- generalize configs/generalize.json --out benchmarks/generalization.json
npm run ml -- rl configs/rl-curriculum.json --out runs/rl-demo
npm run ml -- shield --campaign data/ml/demo --out benchmarks/shield-suite.json
npm run ml -- manifest runs/curriculum-demo/seed-0/03-full/checkpoint-best.pt \
  --campaign data/ml/demo --out runs/g3/model-manifest.json
npm run ml -- compat runs/g3/model-manifest.json --campaign data/ml/demo \
  --out benchmarks/compatibility.json
npm run ml -- ablations configs/ablations.json --out benchmarks/ablations.json
npm run ml -- benchmark configs/benchmark-policies.json --out benchmarks/policy-comparison.json

# 8. Days 53–56: mutation search, ranker and the G3 gate
npm run ml -- search configs/search-faulted-fail.json --out benchmarks/search-faulted-fail.json
npm run ml -- rank configs/ranker.json --out runs/ranker
npm run ml -- gate configs/g3-gate.json --out benchmarks/g3-gate.json --markdown docs/g3-gate.md
```

Every command prints a compact JSON summary and writes its full manifest next to
the artefacts: `training-manifest.json`, `sweep.json` or
`curriculum-manifest.json`. A checkpoint records its config, dataset campaign,
feature/parameter schema hashes, normalization, code revision and library
versions, so it can only be loaded by a compatible environment.

The checked-in `benchmarks/ml-baselines.json` is the Day 39/Day 41 comparison:
the rule expert, the first-feasible and wait baselines, and MLP, DeepSets and
GNN+GRU checkpoints evaluated on exactly the same held-out episodes. It is
regenerated with `npm run ml -- compare` and its checkpoints/configs are
ordinary training runs, not special-cased numbers.

The G3 evidence is checked in as [generalization.json](benchmarks/generalization.json),
[policy-comparison.json](benchmarks/policy-comparison.json),
[ablations.json](benchmarks/ablations.json),
[shield-suite.json](benchmarks/shield-suite.json),
[compatibility.json](benchmarks/compatibility.json),
[adversarial-search.json](benchmarks/adversarial-search.json) and
[g3-gate.json](benchmarks/g3-gate.json). The gate promotes the GNN+GRU
imitation candidate (0.6376 held-out coverage vs 0.6328 for MLP/DeepSets and
0.6262 for the rule expert); PPO is retained as a measured negative result.

## Day 8–10: scenarios, parameters and the canonical API

- Scenarios load at runtime from validated configuration. `GET /api/scenarios` lists the built-in
  catalog; `POST /api/scenario` accepts `{"key": "swarm-48"}` or `{"config": {...}}`. Inline configs
  go through the same validator and a rejected load leaves the running scenario untouched.
- Mission Parameters (rail icon) is generated from the contract schema: labels, units, steps and
  bounds come from the schema's `x-ui` block through a widget registry, so a new same-type parameter
  needs no editor code. Edits stay in a draft, are validated locally, and apply as a restart at epoch.
- `GET /api/state` is the canonical stream: `{request_id, revision, scenario_id, clock, snapshot}`.
  `revision` is monotonic, and the renderer drops any reply that is older than the state it already
  shows, so a slow response can never roll the view backward. `GET /api/snapshot` stays a pure
  contract payload.
- Every reply echoes `X-Request-Id` (caller-supplied when valid) and `X-Revision`; errors are
  `{error, message, request_id, ...}` with a stable code. Bodies are bounded at 1 MiB (413), non-JSON
  content types are rejected (415) and unknown routes return a structured 404.
- `constellab_core run` advances a scenario with no server and no browser, streaming one NDJSON
  record per step (about 110k satellite states/s on a laptop), which is the entry point for batch
  work. Training code is not coupled into the visual process.
- The API is strictly renderer ↔ core at this point. Simulated inter-satellite contacts and
  message transport arrive later (Days 25–30, below); they are core-side and not yet driven by the
  visual client.

## Days 25–30: targets, sensors, tasks, policies, contacts and transport

The simulation core now carries the swarm-testing layer. Everything is config/
plugin driven, deterministic under a seed, and split so a policy can never read
the simulator's ground truth.

| Module | What it owns |
|---|---|
| `core/constellab_core/targets.py` | `EventTarget` (id, kind, location/radius, active interval, urgency, requirements, separate `truth`) and the `EventSource` registry (`hotspot`, `flood-region`, `change-detection`) |
| `core/constellab_core/sensors.py` | `SensorModel` plugins, footprint geometry (range, off-nadir, elevation, limb occlusion, resolution proxy), hash cloud mask, duty cycle, contiguous `AccessWindow`s |
| `core/constellab_core/tasks.py` | generic task lifecycle, idempotent/illegal transitions, deadlines, and observation outcomes with confidence, freshness and resource cost |
| `core/constellab_core/policy.py` | observation/action schema, `local-greedy` + `fixed-priority` baselines, diagnostic `oracle-upper-bound`, and the shared `run_episode` harness |
| `core/constellab_core/contacts.py` | `LinkModel` plugins (`link.uhf`, `link.optical`), deterministic contact plans, topology helpers, three reference arrangements |
| `core/constellab_core/transport.py` | bounded queues, bandwidth, latency, seeded loss/duplication/reordering, partitions, reconnect policy, replayable trace |

Key guarantees:

- **Ground truth stays simulator-side.** Policies receive `PolicyObservation`
  (active targets only, already-created tasks, predicted access windows) and
  nothing else; a target's `truth` payload is serialised in exactly one place,
  the simulator side, and observation construction rejects it.
- **Every outcome is reproducible.** Event traces, access windows, episode
  traces, contact plans and message traces are all functions of
  `(config, targets, seed, step)`, asserted by tests in each suite.
- **Partitions prevent delivery, not just drawing.** While a partition is
  active a message cannot cross between its groups; `flush`/`hold`/`drop`
  decide what happens to traffic that waited.
- **A policy only proposes.** The harness validates every proposal; illegal
  actions, unknown tasks and double claims are rejected and recorded.

`GET /api/swarm` publishes the Day 25–30 registries (event kinds, sensor models,
task types, policies, link models, transport defaults) for the renderer and
tests. Run the suites with `npm run test:core` or everything with `npm test`.

## Day 4–5: satellites, orbit paths and motion

- Satellite markers and their orbit paths are built from the snapshot `OrbitElements`;
  the scrubber's **Toggle Orbits** button hides or shows every path without touching state
- The core propagates position analytically (`F(t)`, two-body) at a fixed simulation-time step;
  frame rate never changes simulation state
- The renderer polls the core's state stream and moves markers; it never integrates orbits itself
- `state_from_elements_at_time(elements, 0)` equals the epoch `state_from_elements(elements)` and one
  full period returns to the starting position
- Orbit paths and satellite positions share one ECI → scene mapping (`scene/orbit.ts`), so markers
  ride their own trajectory lines
- The demo scenario runs at `time_scale` 300 (≈19 s per LEO revolution); the clock controls that
  drive it are described under Day 6 above

### Brand assets

Icons and the wordmark/symbol come from `assets/constellab_brand_assets/`. `assets/icons/icon.png`
(2048 px) is the single source for the generated `.ico` and favicon via `npm run icon`.

## Day 3: the Earth viewport

- Lit Earth sphere using the served tactical Earth texture (`frontend/public/earth-tactical-dark.png`),
  with a canvas-drawn schematic fallback if the image cannot load
- Drag to orbit; wheel/pinch or +/− buttons to zoom; Reset view restores camera
- Earth is static — camera animation never changes simulation time
- Renders even if core is unreachable; telemetry panel reports errors separately
- Valid snapshots appear in telemetry panel; expand Core state for satellite table
- Satellite meshes and orbit paths are Day 4; propagation is Day 5
- Scene follows viewport size via ResizeObserver, clamps DPR, owns one render loop
- Unmount cancels loop, disconnects resize, releases GPU resources

### Scene components

| Component | Responsibility |
|---|---|
| `frontend/src/scene/SceneShell.tsx` | React host, camera buttons, snapshot telemetry |
| `EarthScene.ts` | Three.js renderer, camera, OrbitControls |
| `EarthWorld.ts` | Earth geometry/material, lighting, disposal |
| `config.ts` | Named camera, geometry, lighting, texture settings |
| `assets.ts` | Canvas-generated Earth texture, no URL download |
| `camera.ts` | Camera reset and bounded zoom helpers |
| `lifecycle.ts` | Single animation loop, injectable runtime for tests |

## Day 2: the contract

`core/constellab_core/data/simulation_contract.schema.json` is the single versioned source of truth (v1.0.0).

| Layer | Artefact |
|---|---|
| Python typed models | `core/constellab_core/contract.py` |
| Python validation | `core/constellab_core/validation.py` — JSON-Schema-subset validator |
| TypeScript types | `frontend/src/contract/generated.ts` — generated by `npm run gen:contract` |
| TypeScript validation | `frontend/src/contract/validate.ts` + `parseSnapshot()` |
| Fixtures | `scenario_empty.json`, `scenario_four_satellites.json`, `malformed_*.json` |

### Core HTTP API (`http://127.0.0.1:8765`)

| Endpoint | Purpose |
|---|---|
| `GET /api/status`, `GET /api/health` | service metadata |
| `GET /api/contract` | JSON schema |
| `GET /api/scenario` | currently loaded SimulationConfig |
| `GET /api/scenarios` | built-in scenario catalog + active key |
| `POST /api/scenario` | load `{"key": ...}` or `{"config": ...}`; restarts at epoch |
| `GET /api/state` | canonical state stream: request id, revision, clock, snapshot |
| `GET /api/snapshot` | SimulationSnapshot at the current clock time |
| `POST /api/config/validate` | validate config |
| `POST /api/snapshot` | validate config and return its epoch snapshot |
| `GET /api/clock` | clock (epoch, simulation time, scale, paused) |
| `POST /api/clock/pause`, `/resume`, `/reset`, `/step`, `/time_scale` | simulation time controls |
| `GET /api/swarm` | Days 25–30 registries: target kinds, sensors, tasks, policies, links, transport |
| `GET /api/g2` | Cached G2 failure-gate report: checks, fault/control traces, timeline, first violation, replay |
| `GET /api/ml` | Read-only G3/ML evidence: gate, held-out, benchmark, model/curriculum pipeline, shield, compatibility, ablations, RL, ranker and search |

Responses carry `X-Request-Id` and `X-Revision`. Errors are JSON with a stable `error` code, a
human `message` and the `request_id`; validation errors also carry field-path `errors`.

Clock semantics:

- `reset` returns to the scenario's epoch: `t = 0`, clock paused, the scenario's own `time_scale`.
- `step` advances exactly one simulated second, whatever the time scale, and never changes the
  paused/running state.
- `time_scale` must be a positive JSON number (`true`, strings and `null` are rejected with
  `400 invalid_time_scale`); accepted values are clamped to `[0.001, 1 000 000]`.

## Folder Structure

```
constellab/
  package.json     npm + electron-builder config
  vite.config.ts   Vite + React config
  pyproject.toml   Python package metadata
  assets/icons/    logo + generated icon.ico
  core/            Python constellab_core package
  core/constellab_ml/  offline environment, models, training, sweeps, curriculum
  frontend/        Vite + React + TS source
  electron/        Electron main + preload
  scripts/         dev/test/helpers
  dist-electron/   build output (gitignored)
```

## Development Commands

```bash
npm run test      # Python core, contract drift, tsc, TS tests, Vite build
npm run test:core # Python tests only
npm run test:contract  # TypeScript contract tests
npm run test:scene     # Scene lifecycle tests
npm run test:scenario  # Schema-driven editor tests
npm run test:api       # State stream / stale-snapshot guard tests
npm run ml -- campaign data/ml/demo --id demo-48 --episodes 48 --seed 7  # demonstrations
npm run ml -- train CONFIG --out DIR        # one training run
npm run ml -- sweep CONFIG --out DIR        # bounded sweep
npm run ml -- curriculum CONFIG --out DIR   # staged Day 45 curriculum
npm run ml -- evaluate CAMPAIGN --checkpoint CHECKPOINT.pt
npm run ml -- compare CAMPAIGN --checkpoint NAME=PATH [...] --out benchmarks/ml-baselines.json
npm run ml -- generalize CONFIG --out FILE      # held-out/OOD slices
npm run ml -- ppo CONFIG --out DIR              # PPO run
npm run ml -- rl CONFIG --out DIR               # multi-seed RL curriculum
npm run ml -- shield --campaign CAMPAIGN --out FILE
npm run ml -- manifest CHECKPOINT --out FILE --campaign CAMPAIGN
npm run ml -- compat MANIFEST --campaign CAMPAIGN --out FILE
npm run ml -- ablations CONFIG --out FILE
npm run ml -- benchmark CONFIG --out FILE
npm run ml -- search CONFIG --out FILE
npm run ml -- rank CONFIG --out DIR
npm run ml -- gate BUNDLE --out FILE --markdown FILE
npm run headless -- swarm-48 --duration 600 --step 10  # browser-free run
npm run typecheck # tsc --noEmit
npm run dev       # Start core + Vite + Electron (restart after Python changes)
npm run status    # Print core status as JSON
npm run snapshot  # Print demo scenario's epoch snapshot
npm run fetch-python  # Download embedded Python for .exe
npm run dist:win  # Build portable .exe
npm run icon      # Regenerate icons (needs Pillow)
```

