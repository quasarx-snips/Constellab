import {
  AmbientLight, Color, DirectionalLight, LinearFilter, Mesh, MeshStandardMaterial, Scene, SphereGeometry,
  RepeatWrapping, SRGBColorSpace, Vector3,
} from "three";
import type { SceneAssets } from "./assets";
import { SCENE_CONFIG } from "./config";

// Builds the Earth sphere from whatever texture provider it is handed:
// REAL_EARTH_ASSETS (NASA Blue Marble image) in the app, OFFLINE_ASSETS (canvas
// schematic) or a test double. The world never chooses the texture itself.
export function createEarthWorld(assets: SceneAssets) {
  const config = SCENE_CONFIG;
  const scene = new Scene();
  scene.background = new Color(config.background);

  const texture = assets.createEarthTexture();
  const geometry = new SphereGeometry(config.earth.radius, config.earth.widthSegments, config.earth.heightSegments);
  const material = new MeshStandardMaterial({
    map: texture,
    roughness: config.earth.roughness,
    metalness: config.earth.metalness,
    color: 0xffffff,
  });
  const earth = new Mesh(geometry, material);
  // Apply sRGB color space to the material. Some Three.js versions only accept
  // colorSpace on a material that is already bound to a mesh; set it both on the
  // typed material and on the mesh's material reference to cover all cases.
  // Apply sRGB color space to the material and the mesh so the texture renders
  // in the right color space regardless of Three.js version quirks.
  // Modern Three.js exposes colorSpace on the material; older builds attach it
  // via the mesh's material slot. Use both paths guarded by duck typing.
  const applyColorSpace = (target: unknown) => {
    const maybe = target as { colorSpace?: typeof SRGBColorSpace };
    if (maybe && typeof maybe.colorSpace === "undefined") return;
    maybe.colorSpace = SRGBColorSpace;
  };
  applyColorSpace(material);
  applyColorSpace(earth.material);
  // Three.js sphere geometry is Y-up; reserve world Z for the contract's north/ECI axis.
  earth.rotation.x = Math.PI / 2;
  // Softer roughness = poles get lit instead of going fully black.
  material.roughness = config.earth.roughness;
  // Ensure texture doesn't seam or repeat to a different color at the poles.
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.repeat.set(1, 1);
  scene.add(earth);
  scene.add(new AmbientLight(config.lighting.ambientColor, config.lighting.ambientIntensity));
  const sun = new DirectionalLight(config.lighting.sunColor, config.lighting.sunIntensity);
  const sunBase = new Vector3(...config.lighting.sunPosition);
  // Keep the configured azimuth and distance; only the height above the equator
  // (the solar declination) is real, so it is driven from the simulated date.
  const sunAzimuth = Math.atan2(sunBase.y, sunBase.x);
  const sunDistance = sunBase.length();
  sun.position.set(...config.lighting.sunPosition);
  scene.add(sun);

  /**
   * Place the Sun at a real declination, in degrees, measured from the equator.
   * In this Z-up ECI scene a positive declination lights the north pole, so the
   * terminator tilts exactly as the obliquity of the ecliptic dictates.
   */
  function setSunDeclinationDeg(declinationDeg: number) {
    const dec = (declinationDeg * Math.PI) / 180;
    const equatorial = Math.cos(dec);
    sun.position.set(
      sunDistance * equatorial * Math.cos(sunAzimuth),
      sunDistance * equatorial * Math.sin(sunAzimuth),
      sunDistance * Math.sin(dec),
    );
  }

  let disposed = false;
  return {
    scene,
    earth,
    sun,
    setSunDeclinationDeg,
    dispose() {
      if (disposed) return;
      disposed = true;
      geometry.dispose();
      material.dispose();
      texture.dispose();
      scene.clear();
    },
  };
}
