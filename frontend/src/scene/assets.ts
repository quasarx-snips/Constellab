import { CanvasTexture, RepeatWrapping, SRGBColorSpace, Texture, TextureLoader } from "three";
import { publicAssetUrl } from "../assetUrl";
import { SCENE_CONFIG } from "./config";

// Tactical Dark Earth texture generated from Natural Earth vector boundaries,
// styled with deep navy/void oceans (#080B14) and glowing anomaly coral outlines (#FF765F).
// Resolved against the page base so it also loads in the packaged file:// app.
export const REAL_EARTH_TEXTURE_URL = publicAssetUrl("earth-tactical-dark.png");

// Loaded on demand, never at module init, so importing this module stays safe in
// Node.js (the scene tests) where there is no DOM.
let realEarthTexture: Texture | undefined;

export function getRealEarthTexture(): Texture {
  if (!realEarthTexture) {
    const loader = new TextureLoader();
    const tex = loader.load(REAL_EARTH_TEXTURE_URL);
    tex.colorSpace = SRGBColorSpace;
    tex.wrapS = RepeatWrapping;
    tex.wrapT = RepeatWrapping;
    tex.repeat.set(1, 1);
    realEarthTexture = tex;
  }
  return realEarthTexture;
}

// Schematic land/sea outlines drawn on canvas, not a geographic dataset or a simulation model.
const LAND_OUTLINES: readonly (readonly (readonly [number, number])[])[] = [
  [[-168, 68], [-135, 72], [-115, 55], [-80, 52], [-52, 45], [-82, 25], [-97, 15], [-110, 29], [-130, 48]],
  [[-80, 12], [-50, 5], [-35, -10], [-52, -35], [-70, -55], [-78, -18]],
  [[-53, 83], [-20, 73], [-40, 59], [-62, 62]],
  [[-12, 36], [4, 59], [40, 72], [110, 72], [170, 60], [150, 40], [120, 20], [103, 0], [79, 8], [65, 27], [36, 35]],
  [[-17, 34], [12, 37], [40, 12], [50, -12], [28, -35], [13, -28], [-5, 3]],
  [[113, -22], [130, -12], [152, -26], [144, -40], [116, -34]],
  [[46, -13], [51, -17], [47, -26], [44, -21]],
];

// Generate a schematic Earth texture on a canvas (offline/failure fallback).
// This is NOT a real map — it's a simple silhouette drawing.
export function generateEarthTexture(canvas: HTMLCanvasElement): CanvasTexture {
  const settings = SCENE_CONFIG.texture;
  canvas.width = settings.width;
  canvas.height = settings.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("A 2D canvas is needed to generate the offline Earth texture.");

  const point = (lon: number, lat: number) =>
    [(lon + 180) / 360 * canvas.width, (90 - lat) / 180 * canvas.height];

  ctx.fillStyle = settings.ocean;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = settings.land;
  for (const outline of LAND_OUTLINES) {
    ctx.beginPath();
    outline.forEach(([lon, lat], i) => {
      const [x, y] = point(lon, lat);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill();
  }
  ctx.fillStyle = settings.ice;
  ctx.fillRect(0, 0, canvas.width, point(0, 80)[1]);
  ctx.fillRect(0, point(0, -75)[1], canvas.width, canvas.height);
  ctx.strokeStyle = settings.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let lon = -180; lon <= 180; lon += settings.gridSpacingDeg) {
    const [x] = point(lon, 0);
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
  }
  for (let lat = -90; lat <= 90; lat += settings.gridSpacingDeg) {
    const [, y] = point(0, lat);
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
  }
  ctx.stroke();
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return texture;
}

// Scene assets that use the real NASA Blue Marble texture (default in the app).
export const REAL_EARTH_ASSETS: SceneAssets = {
  createEarthTexture: getRealEarthTexture,
};

export const OFFLINE_ASSETS: SceneAssets = {
  createEarthTexture: () => generateEarthTexture(document.createElement("canvas")),
};

// Replaceable texture provider: the viewport only needs something that can hand
// back a Three.js texture, so tests inject one without a DOM or a network.
export interface SceneAssets {
  createEarthTexture(): Texture;
}
