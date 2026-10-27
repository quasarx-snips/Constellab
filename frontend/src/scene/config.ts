export const SCENE_CONFIG = {
  background: "#080B14",
  units: {
    // Scene unit = Earth equatorial radius (WGS 84 semi-major axis). The core
    // speaks kilometres (ECI), so the renderer needs exactly one conversion; see
    // scene/orbit.ts eciToScene().
    kmPerSceneUnit: 6378.137,
    // Real-world Earth rotation: one sidereal day = 23h 56m 4.0905s of mean solar
    // time. This is the true 360° spin relative to the fixed stars (ECI), not the
    // 24h solar day. Source: mean sidereal day, IAU/USNO convention.
    secondsPerEarthSiderealDay: 86164.0905,
  },
  earth: {
    radius: 1,
    widthSegments: 128,
    heightSegments: 64,
    roughness: 0.6,
    metalness: 0.05,
  },
  // Large-fleet rendering policy (Day 11). Every number here is a visual
  // density decision, never a simulation rule: the core still owns all state.
  fleet: {
    // Markers are always one InstancedMesh draw call. Above this count the
    // marker spheres drop to a low-detail mesh and a smaller radius.
    detailMarkerLimit: 64,
    markerRadius: 0.024,
    denseMarkerRadius: 0.016,
    selectionScale: 1.7,
    // Orbit paths are de-duplicated by orbit geometry, so a Walker shell with
    // 1,000 satellites needs one path per plane, not 1,000 paths. This caps the
    // number of distinct paths that may be drawn at once.
    maxOrbitPaths: 128,
    // Tessellation of a drawn path, chosen from the fleet size.
    pathSegments: [
      { maxSatellites: 4, segments: 180 },
      { maxSatellites: 48, segments: 120 },
      { maxSatellites: 250, segments: 72 },
      { maxSatellites: 100000, segments: 48 },
    ],
    // Diagnostic ID labels drawn at once (selected ships one extra).
    maxLabels: 48,
    // Screen projections feed React callouts/picking. They are recomputed at
    // this cadence instead of once per render frame, so a 1,000-satellite fleet
    // does not turn into 60 React state updates per second.
    projectionIntervalMs: 100,
    // Swarm telemetry table rows rendered at once.
    telemetryRowLimit: 50,
  },
  texture: {
    width: 1024,
    height: 512,
    ocean: "#080B14",
    land: "#2E405A",
    ice: "#364B69",
    grid: "rgba(255, 118, 95, 0.2)",
    gridSpacingDeg: 30,
  },
  camera: {
    fovDeg: 45,
    near: 0.01,
    far: 100,
    position: [3.0, 1.8, 3.0] as const,
    target: [0, 0, 0] as const,
    minDistance: 1.35,
    maxDistance: 12,
    zoomFactor: 1.25,
    dampingFactor: 0.08,
    rotateSpeed: 0.6,
    zoomSpeed: 1.0,
  },
  lighting: {
    ambientColor: "#101827",
    ambientIntensity: 1.4,
    sunColor: "#FFF5EA",
    sunIntensity: 3.2,
    sunPosition: [4, 2.5, 4.5] as const,
  },
  renderer: {
    maxPixelRatio: 2,
    exposure: 1.05,
  },
  // Day 12 diagnostics: thresholds only, never simulation rules.
  diagnostics: {
    // A snapshot older than this is flagged stale (poll cadence is 200 ms).
    staleWarnMs: 600,
    // Beyond this the link is reported as lost even before a request fails.
    staleBadMs: 2500,
    // Clock readout refresh for the age/UTC rows.
    refreshMs: 250,
  },
  connection: {
    // Retry backoff for the local core: 400 ms, 800 ms, 1.6 s … capped at 5 s.
    baseDelayMs: 400,
    maxDelayMs: 5000,
  },
  // Real Earth orientation. The globe's spin is not a free visual parameter:
  // it is derived from the simulation clock at one turn per sidereal day (see
  // scene/earth.ts), so a time-warp change speeds the planet up too.
  earthRotation: {
    // Obliquity of the ecliptic, WGS 84 / IAU 2006. In ECI the pole is +Z, so
    // this appears as the Sun's seasonal declination (±23.44°), which tilts the
    // day/night terminator.
    axialTiltDeg: 23.44,
  },
  timeWarp: {
    // Simulation time multiplier: 1 = real time, up to 1,000,000x.
    min: 1,
    max: 1_000_000,
    default: 300,
  },
} as const;

