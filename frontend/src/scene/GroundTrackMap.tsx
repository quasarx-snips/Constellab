import { useEffect, useRef } from "react";
import { publicAssetUrl } from "../assetUrl";
import type { SnapshotState } from "../App";
import { getSatelliteColor } from "./orbit";
import { SCENE_CONFIG } from "./config";
import { eciLatitudeDeg, eciLongitudeDeg, orbitalPeriodS } from "./earth";

interface GroundTrackMapProps {
  snapshot: SnapshotState;
  selectedSatId: string | null;
  onSelectSat: (id: string) => void;
  showCallouts?: boolean;
}

const km = (v: number) => v.toFixed(1);
const kms = (v: number) => v.toFixed(2);
const DEG = 180 / Math.PI;
const wrapLon = (deg: number) => ((((deg + 180) % 360) + 360) % 360) - 180;

export function GroundTrackMap({
  snapshot,
  selectedSatId,
  onSelectSat,
  showCallouts = true,
}: GroundTrackMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bgImgRef = useRef<HTMLImageElement | null>(null);

  // Preload tactical earth map texture
  useEffect(() => {
    const img = new Image();
    img.src = publicAssetUrl("earth-tactical-dark.png");
    img.onload = () => {
      bgImgRef.current = img;
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;

    const render = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      // 1. Draw Background
      ctx.fillStyle = "#080B14";
      ctx.fillRect(0, 0, width, height);

      if (bgImgRef.current && bgImgRef.current.complete) {
        ctx.drawImage(bgImgRef.current, 0, 0, width, height);
      }

      // 2. Tactical Graticules Overlay
      ctx.strokeStyle = "rgba(27, 43, 74, 0.6)";
      ctx.lineWidth = 1;

      // Meridians (every 30 deg)
      for (let lon = -180; lon <= 180; lon += 30) {
        const x = ((lon + 180) / 360) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        ctx.fillStyle = "rgba(125, 133, 152, 0.4)";
        ctx.font = "10px monospace";
        ctx.fillText(`${lon}°`, x + 4, height - 8);
      }

      // Parallels (every 30 deg)
      for (let lat = -60; lat <= 60; lat += 30) {
        const y = ((90 - lat) / 180) * height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();

        ctx.fillStyle = "rgba(125, 133, 152, 0.4)";
        ctx.font = "10px monospace";
        ctx.fillText(`${lat > 0 ? `+${lat}` : lat}°`, 8, y - 4);
      }

      // Highlight Equator
      const eqY = height * 0.5;
      ctx.strokeStyle = "rgba(61, 104, 178, 0.6)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, eqY);
      ctx.lineTo(width, eqY);
      ctx.stroke();

      // 3. Draw Satellites and Orbital Ground Tracks
      if (snapshot.kind === "ready") {
        const satellites = snapshot.snapshot.satellites;
        const simTimeS = snapshot.snapshot.clock.simulation_time_s;
        const siderealDayS = SCENE_CONFIG.units.secondsPerEarthSiderealDay;

        satellites.forEach((sat, i) => {
          const [x, y, z] = sat.position_eci_km;
          const [vx, vy, vz] = sat.velocity_eci_km_s;
          const r = Math.sqrt(x * x + y * y + z * z);
          const vel = Math.sqrt(vx * vx + vy * vy + vz * vz);
          const alt = r - SCENE_CONFIG.units.kmPerSceneUnit;

          // Lat is the same in both frames; longitude must subtract the angle
          // Earth has turned through, or the marker sits forever over the same
          // ground point while the planet spins underneath it.
          const lat = eciLatitudeDeg(sat.position_eci_km);
          const inertialLon = Math.atan2(y, x) * DEG;
          const lon = eciLongitudeDeg(sat.position_eci_km, simTimeS);

          const satX = ((lon + 180) / 360) * width;
          const satY = ((90 - lat) / 180) * height;

          const colorInt = getSatelliteColor(i);
          const colorHex = `#${colorInt.toString(16).padStart(6, "0")}`;
          const isSelected = selectedSatId === sat.id;
          const isAnomaly = sat.id.toLowerCase().includes("anomaly");

          // Draw Simulated Ground Track Curve
          if (sat.orbit) {
            const inc = (sat.orbit.inclination_deg * Math.PI) / 180;
            ctx.strokeStyle = colorHex;
            ctx.globalAlpha = isSelected ? 0.45 : 0.22;
            ctx.lineWidth = isSelected ? 1.75 : 1;
            ctx.setLineDash([4, 4]);
            const periodS = orbitalPeriodS(sat.orbit.semi_major_axis_km);
            ctx.beginPath();
            let first = true;
            for (let t = -180; t <= 180; t += 3) {
              // One orbit spans a full 360° of inertial longitude, but the ground
              // under it slides east-to-west by the planet's spin over the same
              // seconds, so the plotted swath drifts westward each pass.
              const trackSpinDeg = ((t / 360) * periodS * 360) / siderealDayS;
              const trackLon = wrapLon(inertialLon + t - trackSpinDeg);
              const trackLat = Math.asin(Math.sin(inc) * Math.sin((t * Math.PI) / 180)) * DEG;
              const px = ((trackLon + 180) / 360) * width;
              const py = ((90 - trackLat) / 180) * height;
              if (first) {
                ctx.moveTo(px, py);
                first = false;
              } else {
                ctx.lineTo(px, py);
              }
            }
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
          }

          // Pulsing Marker
          const now = Date.now() / 1000;
          const pulse = Math.sin(now * 3 + i) * 3 + 6;

          ctx.fillStyle = colorHex;
          ctx.beginPath();
          ctx.arc(satX, satY, 3, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = colorHex;
          ctx.lineWidth = 1.25;
          ctx.beginPath();
          ctx.arc(satX, satY, pulse, 0, Math.PI * 2);
          ctx.stroke();

          // Satellite Diamond Glyph
          const d = 5;
          ctx.beginPath();
          ctx.moveTo(satX, satY - d);
          ctx.lineTo(satX + d, satY);
          ctx.lineTo(satX, satY + d);
          ctx.lineTo(satX - d, satY);
          ctx.closePath();
          ctx.stroke();

          // Callout Badge on 2D map
          if (showCallouts && (isSelected || isAnomaly)) {
            const labelX = satX > width - 160 ? satX - 150 : satX + 16;
            const labelY = satY > height - 70 ? satY - 50 : satY - 12;

            // Leader Line
            ctx.strokeStyle = colorHex;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 2]);
            ctx.beginPath();
            ctx.moveTo(satX, satY);
            ctx.lineTo(labelX, labelY + 12);
            ctx.stroke();
            ctx.setLineDash([]);

            // Card background
            ctx.fillStyle = "rgba(16, 24, 39, 0.92)";
            ctx.strokeStyle = colorHex;
            ctx.lineWidth = 1;
            ctx.fillRect(labelX, labelY, 140, 46);
            ctx.strokeRect(labelX, labelY, 140, 46);

            // Title
            ctx.fillStyle = colorHex;
            ctx.font = "bold 11px monospace";
            ctx.fillText(isAnomaly ? "JACKAL-II (SAT-02)" : sat.id.toUpperCase(), labelX + 8, labelY + 14);

            // Status Badge
            ctx.fillStyle = isAnomaly ? "#FF765F" : "#6F9B97";
            ctx.font = "bold 9px monospace";
            ctx.fillText(isAnomaly ? "TRUE ANOMALY" : "NOMINAL", labelX + 8, labelY + 26);

            // Metrics
            ctx.fillStyle = "#B7D8C8";
            ctx.font = "9px monospace";
            ctx.fillText(`Alt ${km(alt)}km · Vel ${kms(vel)}km/s`, labelX + 8, labelY + 38);
          }
        });
      }

      animId = requestAnimationFrame(render);
    };

    render();

    const handleCanvasClick = (e: MouseEvent) => {
      if (snapshot.kind !== "ready") return;
      const rect = canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      let closestSat: string | null = null;
      let closestDist = 25; // Click radius threshold

      const simTimeS = snapshot.snapshot.clock.simulation_time_s;
      snapshot.snapshot.satellites.forEach((sat) => {
        const lat = eciLatitudeDeg(sat.position_eci_km);
        const lon = eciLongitudeDeg(sat.position_eci_km, simTimeS);
        const sx = ((lon + 180) / 360) * rect.width;
        const sy = ((90 - lat) / 180) * rect.height;

        const dist = Math.hypot(clickX - sx, clickY - sy);
        if (dist < closestDist) {
          closestDist = dist;
          closestSat = sat.id;
        }
      });

      if (closestSat) {
        onSelectSat(closestSat);
      }
    };

    canvas.addEventListener("click", handleCanvasClick);

    return () => {
      cancelAnimationFrame(animId);
      canvas.removeEventListener("click", handleCanvasClick);
    };
  }, [snapshot, selectedSatId, showCallouts, onSelectSat]);

  return (
    <div className="ground-track-container" aria-label="2D Ground Track Map">
      <div className="ground-track-header">
        <span className="ground-track-badge">2D GROUND TRACK PROJECTION</span>
        <span className="ground-track-sub">EQUIRECTANGULAR SUB-SATELLITE TRACE</span>
      </div>
      <canvas ref={canvasRef} className="ground-track-canvas" />
    </div>
  );
}
