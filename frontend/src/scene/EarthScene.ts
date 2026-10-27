import {
  ACESFilmicToneMapping, PerspectiveCamera, SRGBColorSpace, Vector3, WebGLRenderer, Scene, Mesh,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { SimulationSnapshot } from "../contract";
import type { SceneAssets } from "./assets";
import { REAL_EARTH_ASSETS } from "./assets";
import { resetCamera, zoomCamera } from "./camera";
import { SCENE_CONFIG } from "./config";
import { earthRotationAngleRad, solarDeclinationDeg } from "./earth";
import { createEarthWorld } from "./EarthWorld";
import { FleetRenderer, type FleetStats } from "./fleet";
import { BROWSER_RUNTIME, startViewport } from "./lifecycle";
import { getSatelliteColor } from "./orbit";
import { focusCamera, pickSatellite } from "./selection";
import { createTacticalEarthOverlays } from "./tacticalOverlays";

export interface SatelliteProjection {
  id: string;
  screenX: number;
  screenY: number;
  isOccluded: boolean;
  distanceKm: number;
  altitudeKm: number;
  velocityKms: number;
  colorHex: string;
  index: number;
}

export interface RenderStats {
  frames: number;
  meanMs: number;
  p50Ms: number;
  p95Ms: number;
  maxMs: number;
  fps: number;
}

const FRAME_SAMPLE_LIMIT = 240;

export class EarthScene {
  private readonly renderer: WebGLRenderer;
  private readonly camera: PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly stopViewport: () => void;
  private readonly worldScene: Scene;
  private readonly world: ReturnType<typeof createEarthWorld>;
  private readonly earthMesh: Mesh;
  private readonly fleet: FleetRenderer;
  private readonly overlays: { group: import("three").Group; dispose: () => void };
  private frameTimesMs: number[] = [];
  private lastProjectionAtMs = 0;
  private currentSnapshot: SimulationSnapshot | null = null;
  private disposed = false;
  private onRenderCallback?: (projections: SatelliteProjection[]) => void;
  private onSelectCallback?: (id: string | null) => void;
  private lastProjections: SatelliteProjection[] = [];
  private selectedId: string | null = null;
  private autoRotate = true;
  // Local estimate of simulated time. Snapshots arrive a few times per second,
  // which is far too coarse to drive the globe directly at high time-warps, so
  // the estimate is integrated every frame at the clock's rate and eased toward
  // each snapshot to absorb poll latency and core tick drift.
  private simTimeS = 0;
  private clockRate = 1;
  private clockPaused = true;
  private clockSynced = false;
  private epochUtc = "";
  private sunDeclinationDeg = Number.NaN;

  constructor(host: HTMLElement, assets: SceneAssets = REAL_EARTH_ASSETS) {
    const config = SCENE_CONFIG;
    this.renderer = new WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
      stencil: false,
      depth: true,
    });
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = config.renderer.exposure;
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.domElement.setAttribute("aria-label", "3D Earth with satellite orbits. Drag to orbit, scroll to zoom.");
    this.renderer.domElement.setAttribute("role", "img");
    this.renderer.setPixelRatio(Math.min(config.renderer.maxPixelRatio, window.devicePixelRatio));
    this.renderer.setClearColor(0x080B14, 1);
    this.camera = new PerspectiveCamera(config.camera.fovDeg, 1, config.camera.near, config.camera.far);
    resetCamera(this.camera, new Vector3());
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = config.camera.dampingFactor;
    this.controls.enablePan = false;
    this.controls.minDistance = config.camera.minDistance;
    this.controls.maxDistance = config.camera.maxDistance;
    this.controls.rotateSpeed = config.camera.rotateSpeed;
    this.controls.zoomSpeed = config.camera.zoomSpeed;
    this.reset();

    let world;
    try {
      world = createEarthWorld(assets);
    } catch (error) {
      this.controls.dispose();
      this.renderer.dispose();
      throw error;
    }
    this.world = world;
    this.worldScene = world.scene;
    this.earthMesh = world.earth;

    // Tactical Aerospace overlays: polar reticles, coordinate rings, LEO altitude circle
    this.overlays = createTacticalEarthOverlays();
    this.worldScene.add(this.overlays.group);
    // Day 11: one instanced renderer owns every marker, path and the selection.
    this.fleet = new FleetRenderer(this.worldScene);

    // Day 7: clicking the canvas picks the nearest visible satellite marker.
    this.renderer.domElement.addEventListener("click", this.handleClick);

    host.appendChild(this.renderer.domElement);
    this.stopViewport = startViewport(host, {
      resize: (width, height) => {
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(config.renderer.maxPixelRatio, window.devicePixelRatio));
      },
      render: (dt) => {
        this.frameStartedAt = performance.now();
        this.advanceSimulatedTime(dt ?? 0);
        this.applyEarthOrientation();
        this.controls.update();
        this.renderer.render(this.worldScene, this.camera);
        this.sampleProjections();
        this.recordFrame();
      },
      dispose: () => {
        this.renderer.domElement.removeEventListener("click", this.handleClick);
        this.controls.dispose();
        this.overlays.dispose();
        this.fleet.dispose();
        world.dispose();
        this.renderer.dispose();
        this.renderer.forceContextLoss();
        this.renderer.domElement.remove();
      },
    }, BROWSER_RUNTIME, config.renderer.maxPixelRatio);
  }

  /**
   * Advance the local simulated-time estimate one frame and place the globe.
   *
   * The mesh is tilted so its local +Y (the sphere's pole) is scene +Z, so a
   * spin about local Y turns the globe about the north pole, west to east, at
   * +2π per sidereal day of simulated time. Because `dt` is real seconds and the
   * rate is the clock's time-warp, the spin is frame-rate independent and got
   * faster the moment the user raised the time-warp.
   */
  private advanceSimulatedTime(dt: number) {
    if (!this.clockPaused && Number.isFinite(dt)) {
      this.simTimeS += dt * this.clockRate;
    }
  }

  /** Point the globe and the Sun at their real-world orientation for sim time. */
  private applyEarthOrientation() {
    if (this.autoRotate) {
      this.earthMesh.rotation.y = earthRotationAngleRad(this.simTimeS);
    }
    // Observable state for UI/e2e checks: the globe's real-world orientation.
    const canvas = this.renderer.domElement;
    if (canvas) {
      canvas.dataset.earthRotationDeg = ((this.earthMesh.rotation.y * 180) / Math.PI).toFixed(3);
      canvas.dataset.earthSimTimeS = this.simTimeS.toFixed(3);
    }
    if (this.epochUtc) {
      // The declination moves ~0.4°/day, so only refresh the light when it has
      // actually changed by a visible amount.
      const declination = solarDeclinationDeg(this.epochUtc, this.simTimeS);
      if (!(Math.abs(declination - this.sunDeclinationDeg) < 0.01)) {
        this.sunDeclinationDeg = declination;
        this.world.setSunDeclinationDeg(declination);
        if (canvas) canvas.dataset.sunDeclinationDeg = declination.toFixed(3);
      }
    }
  }

  /**
   * Recompute screen projections on a fixed cadence, not every frame. React
   * callouts and picking consume these; a 1,000-satellite fleet must not cause
   * 60 state updates per second with 1,000 new objects each time.
   */
  private sampleProjections() {
    const now = performance.now();
    if (now - this.lastProjectionAtMs < SCENE_CONFIG.fleet.projectionIntervalMs) return;
    this.lastProjectionAtMs = now;
    this.lastProjections = this.calculateProjections();
    if (this.onRenderCallback) this.onRenderCallback(this.lastProjections);
  }

  private recordFrame() {
    // `performance.now()` deltas around the draw call: CPU-side frame cost.
    this.frameTimesMs.push(performance.now() - this.frameStartedAt);
    if (this.frameTimesMs.length > FRAME_SAMPLE_LIMIT) this.frameTimesMs.shift();
    if (this.frameTimesMs.length % 20 === 0) {
      const stats = this.renderStats();
      const canvas = this.renderer.domElement;
      if (canvas) {
        canvas.dataset.frameP50Ms = stats.p50Ms.toFixed(2);
        canvas.dataset.frameP95Ms = stats.p95Ms.toFixed(2);
      }
    }
  }

  private frameStartedAt = typeof performance !== "undefined" ? performance.now() : 0;

  /** Distribution of recent CPU frame times (render + projections). */
  renderStats(): RenderStats {
    const samples = [...this.frameTimesMs].sort((a, b) => a - b);
    if (samples.length === 0) return { frames: 0, meanMs: 0, p50Ms: 0, p95Ms: 0, maxMs: 0, fps: 0 };
    const at = (fraction: number) => samples[Math.min(samples.length - 1, Math.floor(fraction * samples.length))];
    const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
    return {
      frames: samples.length,
      meanMs: mean,
      p50Ms: at(0.5),
      p95Ms: at(0.95),
      maxMs: samples[samples.length - 1],
      fps: mean > 0 ? 1000 / mean : 0,
    };
  }

  fleetStats(): FleetStats {
    return this.fleet.stats();
  }

  private readonly handleClick = (event: MouseEvent) => {
    if (!this.onSelectCallback) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const id = pickSatellite(this.lastProjections, event.clientX - rect.left, event.clientY - rect.top);
    if (id) this.onSelectCallback(id);
  };

  setOnRender(callback?: (projections: SatelliteProjection[]) => void) {
    this.onRenderCallback = callback;
    // Emit immediately so a surface that mounts after the first frame is not blank.
    if (callback && this.lastProjections.length > 0) callback(this.lastProjections);
  }

  /** Called with the satellite id picked on the canvas; null clears the selection. */
  setOnSelect(callback?: (id: string | null) => void) {
    this.onSelectCallback = callback;
  }

  /** Highlight the selected marker without touching simulation state. */
  setSelection(id: string | null) {
    if (this.selectedId === id) return;
    this.selectedId = id;
    this.fleet.setSelection(id);
    this.publishFleetState();
  }

  /**
   * Aim the camera at one satellite while keeping the Earth in frame.
   * Returns false when the id is not currently rendered.
   */
  focusSatellite(id: string): boolean {
    const position = this.fleet.positionOf(id);
    if (!position || this.disposed) return false;
    const distance = Math.min(
      SCENE_CONFIG.camera.maxDistance,
      Math.max(2.4, position.length() + 1.6),
    );
    focusCamera(this.camera, this.controls, position, distance);
    this.controls.update();
    // Observable state for UI/e2e checks: which satellite the camera is aimed at.
    this.renderer.domElement.dataset.focusedSatellite = id;
    return true;
  }

  setOverlaysVisible(visible: boolean) {
    this.overlays.group.visible = visible;
  }

  /** Show/hide every orbit path; satellites stay visible either way. */
  setOrbitsVisible(visible: boolean) {
    this.fleet.setOrbitsVisible(visible);
    this.renderer.domElement.dataset.orbitsVisible = String(visible);
  }

  /** Day 12 diagnostic layer: translucent orbital-plane fill for the selection. */
  setOrbitPlaneVisible(visible: boolean) {
    this.fleet.setOrbitPlaneVisible(visible);
    this.renderer.domElement.dataset.orbitPlaneVisible = String(visible);
  }

  setAutoRotate(enabled: boolean) {
    this.autoRotate = enabled;
  }

  isAutoRotating(): boolean {
    return this.autoRotate;
  }

  private calculateProjections(): SatelliteProjection[] {
    if (!this.currentSnapshot || !this.renderer.domElement) return [];
    const width = this.renderer.domElement.clientWidth;
    const height = this.renderer.domElement.clientHeight;
    if (width === 0 || height === 0) return [];

    const camPos = this.camera.position;
    const earthRadius = SCENE_CONFIG.earth.radius;
    const kmPerUnit = SCENE_CONFIG.units.kmPerSceneUnit;

    const projections: SatelliteProjection[] = [];

    this.currentSnapshot.satellites.forEach((sat, i) => {
      const satPos = this.fleet.positionOf(sat.id);
      if (!satPos) return;

      // Check occlusion by Earth sphere at (0,0,0)
      const ray = satPos.clone().sub(camPos);
      const rayLen = ray.length();
      ray.normalize();
      const dot = -camPos.dot(ray);
      let isOccluded = false;
      if (dot > 0 && dot < rayLen) {
        const perpDistSq = camPos.lengthSq() - dot * dot;
        if (perpDistSq < earthRadius * earthRadius) {
          isOccluded = true;
        }
      }

      const projected = satPos.clone().project(this.camera);
      if (projected.z > 1.0) isOccluded = true; // Behind camera

      const screenX = (projected.x * 0.5 + 0.5) * width;
      const screenY = (-projected.y * 0.5 + 0.5) * height;

      const satRadiusKm = satPos.length() * kmPerUnit;
      const altitudeKm = Math.max(0, satRadiusKm - kmPerUnit);
      const [vx, vy, vz] = sat.velocity_eci_km_s;
      const velocityKms = Math.hypot(vx, vy, vz);

      const color = getSatelliteColor(i);
      const colorHex = `#${color.toString(16).padStart(6, "0")}`;

      projections.push({
        id: sat.id,
        screenX,
        screenY,
        isOccluded,
        distanceKm: satRadiusKm,
        altitudeKm,
        velocityKms,
        colorHex,
        index: i,
      });
    });

    return projections;
  }

  setSnapshot(snapshot: SimulationSnapshot | null) {
    if (this.disposed) return;
    this.currentSnapshot = snapshot;
    const canvas = this.renderer.domElement;
    this.fleet.setSnapshot(snapshot);
    if (snapshot) {
      canvas.dataset.scenarioId = snapshot.scenario_id;
      canvas.dataset.simulationTimeS = String(snapshot.clock.simulation_time_s);
      this.syncClock(snapshot);
      // Refresh projections right away so a scenario switch is visible without
      // waiting for the next cadence tick.
      this.lastProjectionAtMs = 0;
    } else {
      delete canvas.dataset.scenarioId;
      delete canvas.dataset.simulationTimeS;
      this.clockSynced = false;
      this.clockPaused = true;
      this.lastProjections = [];
      if (this.onRenderCallback) this.onRenderCallback([]);
    }
    this.publishFleetState();
  }

  private publishFleetState() {
    const canvas = this.renderer.domElement;
    if (!canvas) return;
    const stats = this.fleet.stats();
    canvas.dataset.fleetSatellites = String(stats.satellites);
    canvas.dataset.fleetInstances = String(stats.markerInstances);
    canvas.dataset.fleetPaths = String(stats.orbitPaths);
    canvas.dataset.fleetUniqueOrbits = String(stats.uniqueOrbits);
    canvas.dataset.fleetPathSegments = String(stats.pathSegments);
    canvas.dataset.fleetMarkerDetail = stats.markerDetail;
  }

  /**
   * Adopt the snapshot clock. A reset, a step, a pause or a backwards jump is
   * authoritative and adopted exactly; otherwise the estimate is nudged toward
   * the snapshot instead of being overwritten, so the globe never stutters at
   * the poll rate while still converging on the core's time.
   */
  private syncClock(snapshot: SimulationSnapshot) {
    const clock = snapshot.clock;
    this.clockRate = clock.time_scale;
    this.clockPaused = clock.paused;
    this.epochUtc = clock.epoch_utc;
    const discontinuity =
      !this.clockSynced || clock.paused || clock.simulation_time_s < this.simTimeS;
    this.clockSynced = true;
    const error = clock.simulation_time_s - this.simTimeS;
    this.simTimeS = discontinuity ? clock.simulation_time_s : this.simTimeS + error * 0.2;
    this.applyEarthOrientation();
  }

  zoom(direction: "in" | "out") {
    if (this.disposed) return;
    zoomCamera(this.camera, this.controls.target,
      direction === "in" ? 1 / SCENE_CONFIG.camera.zoomFactor : SCENE_CONFIG.camera.zoomFactor);
    this.controls.update();
  }

  reset() {
    if (this.disposed) return;
    const damping = this.controls.enableDamping;
    this.controls.enableDamping = false;
    this.controls.reset();
    resetCamera(this.camera, this.controls.target);
    this.controls.saveState();
    this.controls.update();
    this.controls.enableDamping = damping;
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.stopViewport();
  }
}
