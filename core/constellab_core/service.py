"""ConstellaLab core service: status, contract schema, config validation and snapshots.

Pure standard library. The service is the single source of truth for version,
clock and orbital state; the renderer only displays the snapshots it is given.
"""

from __future__ import annotations

import json
import math
import platform
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import SCHEMA_VERSION, __version__
from .config import (
    MAX_CONFIG_BYTES,
    BUILTIN_SCENARIOS,
    ConfigError,
    builtin_scenario,
    scenario_catalog,
    validate_config_bytes,
)
from . import components, plugins
from .frames import frames_payload
from .physics import orbit_field_payload
from .units import registry_payload as units_registry_payload
from .contract import SCHEMA, ContractError, SimulationClock, SimulationConfig
from .orbit import initial_snapshot, snapshot_at_time
if not components.registered_components():  # pragma: no cover - defensive
    plugins.activate_builtins()

ROUTES = {
    ("GET", "/api/status"): "status",
    ("GET", "/api/health"): "health",
    ("GET", "/api/contract"): "contract",
    ("GET", "/api/scenario"): "scenario",
    ("GET", "/api/scenarios"): "scenarios",
    ("POST", "/api/scenario"): "scenario_load",
    ("GET", "/api/snapshot"): "snapshot",
    ("GET", "/api/state"): "state",
    ("GET", "/api/registry"): "registry",
    ("GET", "/api/swarm"): "swarm",
    ("GET", "/api/g2"): "g2",
    ("GET", "/api/ml"): "ml",
    ("GET", "/api/clock"): "clock_get",
    ("POST", "/api/clock/pause"): "clock_pause",
    ("POST", "/api/clock/resume"): "clock_resume",
    ("POST", "/api/clock/reset"): "clock_reset",
    ("POST", "/api/clock/step"): "clock_step",
    ("POST", "/api/clock/time_scale"): "clock_time_scale",
    ("POST", "/api/config/validate"): "validate",
    ("POST", "/api/snapshot"): "snapshot_for_config",
}

# Local-only bridge: the renderer (Vite dev server or packaged file://) is a
# different origin, so answer CORS preflight for it.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Request-Id",
    "Access-Control-Expose-Headers": "X-Request-Id, X-Revision",
}

# Request ids are opaque, bounded and echoed back so a caller can match a reply
# to the exact request that produced it.
MAX_REQUEST_ID_LENGTH = 64
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,%d}" % MAX_REQUEST_ID_LENGTH)


def new_request_id() -> str:
    return uuid.uuid4().hex


def status_payload() -> dict[str, Any]:
    """Build the canonical version/status payload."""
    return {
        "name": "constellab-core",
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "endpoints": [f"{method} {path}" for method, path in ROUTES],
        "registry": registry_summary(),
    }


def registry_summary() -> dict[str, Any]:
    """Machine-readable Day 16-20 registry summary for status and /api/registry."""
    from .parameters import builtin_registry

    parameters = builtin_registry()
    return {
        "units": units_registry_payload(),
        "frames": frames_payload(),
        "orbit_fields": orbit_field_payload(),
        "parameters": {
            "schema_version": parameters.payload()["schema_version"],
            "schema_hash": parameters.schema_hash(),
            "count": len(parameters),
            "owners": parameters.owners(),
        },
        "components": components.states_payload(),
        "plugins": plugins.host().report(),
    }


def registry_payload() -> dict[str, Any]:
    """Day 22: the full registry editor payload (one spec dict per parameter).

    ``registry_summary`` keeps serving the compact Day 16-20 summary for
    ``/api/status`` and ``registry``; this payload is what a scaled editor can
    paginate, validate and hash against.
    """
    from .parameters import builtin_registry

    parameters = builtin_registry()
    version = parameters.payload()["schema_version"]
    return {
        "include": "constellab-registry",
        "schema_version": version,
        "schema_hash": parameters.schema_hash(),
        "owners": parameters.owners(),
        "count": len(parameters),
        "parameters": [spec.to_dict() for spec in parameters.specs()],
    }


def swarm_payload() -> dict[str, Any]:
    """Day 25-30 registries: targets, sensors, tasks, policies, links, transport.

    The renderer and any test client read the same descriptions the core uses,
    so a new event source, sensor, task type, policy or link model is discoverable
    without a hard-coded endpoint list.
    """
    from . import contacts, faults, invariants, policy, sensors, targets, tasks, trace, transport

    return {
        "faults": {
            "version": faults.FAULT_VERSION,
            "kinds": list(faults.FAULT_KINDS),
            "strategies": list(faults.FAULT_STRATEGIES),
        },
        "invariants": invariants.rules_payload(),
        "trace": {"version": trace.TRACE_VERSION, "seeds": trace.SEED_VERSION},
        "targets": {
            "version": targets.EVENT_SOURCE_VERSION,
            "kinds": list(targets.SUPPORTED_EVENT_KINDS),
        },
        "sensors": sensors.sensor_models_payload(),
        "tasks": tasks.task_types_payload(),
        "policies": {
            "version": policy.POLICY_VERSION,
            "policies": policy.registered_policy_metadata(),
        },
        "contacts": contacts.link_models_payload(),
        "transport": transport.message_payload(),
    }


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class CoreHandler(BaseHTTPRequestHandler):
    server_version = f"ConstellaLabCore/{__version__}"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[constellab-core] %s - %s\n" % (self.address_string(), fmt % args))

    @property
    def scenario(self) -> SimulationConfig:
        return self.server.scenario  # type: ignore[attr-defined]

    @property
    def clock(self) -> SimulationClock:
        return self.server.clock  # type: ignore[attr-defined]

    @property
    def request_id(self) -> str:
        """Echo the caller's ``X-Request-Id`` when it is valid, else generate one."""
        cached = getattr(self, "_request_id", None)
        if cached is None:
            supplied = (self.headers.get("X-Request-Id") or "").strip()
            cached = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else new_request_id()
            self._request_id = cached
        return cached

    def _send(self, code: int, payload: Any) -> None:
        body = _json_bytes(payload)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-Id", self.request_id)
        self.send_header("X-Revision", str(self.server.revision))  # type: ignore[attr-defined]
        for name, value in CORS_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, error: str, message: str, **extra: Any) -> None:
        """Predictable error body: a code, a human message and the request id."""
        self._send(code, {"error": error, "message": message,
                          "request_id": self.request_id, **extra})

    def _not_found(self) -> None:
        self._error(404, "not_found", f"no route for {self.path}")

    def _state(self):
        """Current scenario, clock and revision read as one consistent view."""
        with self.server._clock_lock:  # type: ignore[attr-defined]
            clock = self.clock
            revision = self.server.revision  # type: ignore[attr-defined]
            scenario = self.scenario
        runtime = getattr(self.server, "runtime", None)
        return revision, snapshot_at_time(
            scenario, clock.simulation_time_s, clock, runtime=runtime
        )

    def _route(self, method: str) -> str | None:
        return ROUTES.get((method, self.path.split("?", 1)[0]))

    def _read_body(self) -> bytes | None:
        """Read a bounded JSON body; ``None`` means a reply was already sent."""
        supplied = (self.headers.get("X-Request-Id") or "").strip()
        if supplied and not REQUEST_ID_PATTERN.fullmatch(supplied):
            self._error(400, "invalid_request_id",
                        f"X-Request-Id must match [A-Za-z0-9_.:-]{{1,{MAX_REQUEST_ID_LENGTH}}}")
            return None
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "invalid_content_length", "Content-Length must be an integer")
            return None
        if length < 0:
            self._error(400, "invalid_content_length", "Content-Length must not be negative")
            return None
        if length > 0:
            content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if content_type and content_type != "application/json":
                self._error(415, "unsupported_media_type", "expected application/json")
                return None
        # Read at most one byte past the limit: that is enough to reject, and
        # draining the client's write keeps it from seeing a connection reset
        # instead of the 413.
        data = self.rfile.read(min(length, MAX_CONFIG_BYTES + 1)) if length > 0 else b""
        if len(data) > MAX_CONFIG_BYTES:
            self.close_connection = True
            self._error(413, "payload_too_large",
                        f"request body exceeds the {MAX_CONFIG_BYTES} byte limit")
            return None
        return data

    def do_OPTIONS(self) -> None:  # noqa: N802 (stdlib naming)
        self.send_response(204)
        for name, value in CORS_HEADERS.items():
            self.send_header(name, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        route = self._route("GET")
        if route == "status":
            self._send(200, status_payload())
        elif route == "health":
            self._send(200, {"status": "ok", "version": __version__})
        elif route == "contract":
            self._send(200, SCHEMA)
        elif route == "scenario":
            self._send(200, self.scenario.to_dict())
        elif route == "scenarios":
            self._send(200, {"scenarios": scenario_catalog(), "active": self.scenario.scenario_id})
        elif route == "snapshot":
            _, snap = self._state()
            self._send(200, snap.to_dict())
        elif route == "state":
            # Canonical state stream: the contract snapshot plus the metadata a
            # renderer needs to drop out-of-order replies.
            revision, snap = self._state()
            self._send(200, {
                "request_id": self.request_id,
                "revision": revision,
                "scenario_id": snap.scenario_id,
                "clock": snap.clock.to_dict(),
                "snapshot": snap.to_dict(),
            })
        elif route == "clock_get":
            self._send(200, self.clock.to_dict())
        elif route == "registry":
            self._send(200, registry_payload())
        elif route == "swarm":
            self._send(200, swarm_payload())
        elif route == "g2":
            # The gate report is precomputed by the core (cached per process);
            # the renderer displays recorded state and never decides correctness.
            from .g2 import cached_demo

            report = cached_demo()
            self._send(200, {
                "request_id": self.request_id,
                "revision": self.server.revision,  # type: ignore[attr-defined]
                **report,
            })
        elif route == "ml":
            # Days 46–56: fixed-path evidence for the ML Lab. This endpoint is
            # read-only and never recomputes a metric; missing reports are
            # returned as unavailable with a reason the UI can display.
            from .ml import ml_report

            self._send(200, {
                "request_id": self.request_id,
                "revision": self.server.revision,  # type: ignore[attr-defined]
                **ml_report(),
            })
        else:
            self._not_found()

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        route = self._route("POST")
        if route == "scenario_load":
            self._load_scenario()
            return
        if route in ("validate", "snapshot_for_config"):
            data = self._read_body()
            if data is None:
                return
            try:
                effective = validate_config_bytes(data)
            except ConfigError as exc:
                self._send(422, {"valid": False, "error": "invalid_config",
                                 "request_id": self.request_id, "errors": exc.errors})
                return
            if route == "validate":
                self._send(200, {"valid": True, "request_id": self.request_id,
                                 "effective_config": effective})
            else:
                try:
                    self._send(200, initial_snapshot(SimulationConfig.from_dict(effective)).to_dict())
                except ContractError as exc:  # pragma: no cover - effective config is already valid
                    self._error(500, "snapshot_failed", "could not build a snapshot", errors=exc.errors)
            return

        if route.startswith("clock_"):
            data = self._read_body()
            if data is None:
                return
            try:
                payload = json.loads(data) if data else {}
            except json.JSONDecodeError as exc:
                self._error(400, "invalid_json", f"request body is not JSON: {exc}")
                return
            if not isinstance(payload, dict):
                self._error(400, "invalid_body", "expected a JSON object")
                return

            with self.server._clock_lock:
                clock = self.server.clock
                if route == "clock_pause":
                    if not clock.paused:
                        clock = SimulationClock(
                            epoch_utc=clock.epoch_utc,
                            simulation_time_s=clock.simulation_time_s,
                            time_scale=clock.time_scale,
                            paused=True,
                        )
                        self.server.clock = clock
                        self.server.sync_advance_rate()
                elif route == "clock_resume":
                    if clock.paused:
                        clock = SimulationClock(
                            epoch_utc=clock.epoch_utc,
                            simulation_time_s=clock.simulation_time_s,
                            time_scale=clock.time_scale,
                            paused=False,
                        )
                        self.server.clock = clock
                        self.server.sync_advance_rate()
                elif route == "clock_reset":
                    self.server.runtime = components.initial_runtime(self.server.scenario)
                    epoch = self.server.scenario.initial_clock()
                    clock = SimulationClock(
                        epoch_utc=epoch.epoch_utc,
                        simulation_time_s=0.0,
                        time_scale=epoch.time_scale,
                        paused=epoch.paused,
                    )
                    self.server.clock = clock
                    self.server.sync_advance_rate()
                elif route == "clock_step":
                    # Single step: advance by one simulation second (regardless of time_scale)
                    new_time_s = clock.simulation_time_s + 1.0
                    if self.server.runtime:
                        components.advance_runtime(
                            self.scenario, self.server.runtime,
                            time_s=new_time_s, dt_s=1.0,
                        )
                    clock = SimulationClock(
                        epoch_utc=clock.epoch_utc,
                        simulation_time_s=new_time_s,
                        time_scale=clock.time_scale,
                        paused=clock.paused,
                    )
                    self.server.clock = clock
                    self.server.sync_advance_rate()
                elif route == "clock_time_scale":
                    new_scale = payload.get("time_scale")
                    if not isinstance(new_scale, (int, float)) or isinstance(new_scale, bool) or new_scale <= 0:
                        self._error(400, "invalid_time_scale", "time_scale must be a positive number")
                        return
                    # Clamp to the documented bound and report it at once: the
                    # requested rate is now the clock's scale. Only how fast
                    # simulated time *advances* eases into it (see the advance
                    # loop), so the UI and the replied clock never lag behind.
                    new_scale = max(0.001, min(1_000_000, new_scale))
                    clock = SimulationClock(
                        epoch_utc=clock.epoch_utc,
                        simulation_time_s=clock.simulation_time_s,
                        time_scale=new_scale,
                        paused=clock.paused,
                    )
                    self.server.clock = clock
                else:
                    self._not_found()
                    return
                revision = self.server.bump_revision()  # under the clock lock
            body = dict(self.server.clock.to_dict())
            body.update({"request_id": self.request_id, "revision": revision})
            self._send(200, body)
            return

        self._not_found()

    def _load_scenario(self) -> None:
        """Load a built-in scenario by key, or a validated config body.

        Body forms: ``{"key": "swarm-48"}`` or ``{"config": {...}}``. The config
        path goes through the same contract validation as ``/api/config/validate``,
        then replaces the scenario and restarts the clock at t=0 (paused=false,
        the scenario's own time_scale).
        """
        data = self._read_body()
        if data is None:
            return
        try:
            payload = json.loads(data) if data else {}
        except json.JSONDecodeError as exc:
            self._error(400, "invalid_json", f"request body is not JSON: {exc}")
            return
        if not isinstance(payload, dict):
            self._error(400, "invalid_body", "expected a JSON object")
            return

        try:
            if "config" in payload:
                config = SimulationConfig.from_dict(validate_config_bytes(
                    json.dumps(payload["config"]).encode("utf-8")
                ))
                source = "inline"
            else:
                key = payload.get("key", payload.get("name"))
                if not isinstance(key, str) or not key:
                    self._error(400, "missing_scenario_key",
                                "expected a scenario key or an inline config",
                                available=sorted(BUILTIN_SCENARIOS))
                    return
                if key not in BUILTIN_SCENARIOS:
                    self._error(404, "unknown_scenario", f"no built-in scenario named {key!r}",
                                key=key, available=sorted(BUILTIN_SCENARIOS))
                    return
                config = builtin_scenario(key)
                source = key
        except ConfigError as exc:
            self._send(422, {"valid": False, "error": "invalid_config",
                             "request_id": self.request_id, "errors": exc.errors})
            return

        with self.server._clock_lock:
            self.server.scenario = config
            self.server.runtime = components.initial_runtime(config)
            clock = running_clock(config)
            self.server.clock = clock
            self.server.sync_advance_rate()
            revision = self.server.bump_revision()  # under the clock lock

        self._send(200, {
            "request_id": self.request_id,
            "revision": revision,
            "scenario": config.to_dict(),
            "clock": clock.to_dict(),
            "source": source,
        })


# Wall-clock time constant for easing the simulated advance rate into a new
# time-warp. ~0.4 s reads as immediate but never sees a jump.
TIME_SCALE_EASE_S = 0.4


class CoreServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a mutable simulation clock and auto-advance.

    ``revision`` is a monotonic counter of state changes. Every reply carries it,
    so a renderer can discard a late answer that predates what it already shows.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scenario = builtin_scenario()
        self.clock = running_clock(self.scenario)
        self.revision = 1
        # Day 18: per-satellite virtual component runtime, advanced each tick.
        self.runtime = components.initial_runtime(self.scenario)
        self._clock_lock = threading.Lock()
        self._advance_thread: threading.Thread | None = None
        self._stop_advance = threading.Event()
        # Rate simulated time currently advances at. It eases toward
        # ``clock.time_scale`` so a warp change never jerks the clock, but it is
        # snapped whenever the clock is re-seated (pause/resume/reset/step/load)
        # so a paused-to-running transition starts at exactly the requested rate.
        self._advance_rate: float = self.clock.time_scale

    def bump_revision(self) -> int:
        """Advance the state revision. Callers hold ``_clock_lock``."""
        self.revision += 1
        return self.revision

    def start_advance(self):
        if self._advance_thread is None or not self._advance_thread.is_alive():
            self._stop_advance.clear()
            self._advance_thread = threading.Thread(target=self._advance_loop, daemon=True)
            self._advance_thread.start()

    def stop_advance(self):
        self._stop_advance.set()
        if self._advance_thread:
            self._advance_thread.join(timeout=1.0)

    def sync_advance_rate(self) -> None:
        """Re-seat the advance rate to the clock's scale. Callers hold the lock."""
        self._advance_rate = self.clock.time_scale

    def _advance_loop(self):
        last_wall = time.monotonic()
        while not self._stop_advance.is_set():
            time.sleep(0.1)
            now = time.monotonic()
            dt = now - last_wall
            last_wall = now
            with self._clock_lock:
                if not self.clock.paused:
                    target = self.clock.time_scale
                    # Frame-rate independent exponential approach: after
                    # TIME_SCALE_EASE_S of simulated wall time the rate has
                    # covered ~63% of the gap, whatever the tick spacing. This
                    # is what makes a warp change read as a smooth ramp instead
                    # of a step, and it converges for any jump (1× <-> 10^6×).
                    weight = 1.0 - math.exp(-dt / TIME_SCALE_EASE_S)
                    self._advance_rate += (target - self._advance_rate) * weight
                    if abs(target - self._advance_rate) <= abs(target) * 1e-3:
                        self._advance_rate = target
                    self.clock = SimulationClock(
                        epoch_utc=self.clock.epoch_utc,
                        simulation_time_s=self.clock.simulation_time_s + dt * self._advance_rate,
                        time_scale=target,
                        paused=self.clock.paused,
                    )
                    if self.runtime:
                        components.advance_runtime(
                            self.scenario, self.runtime,
                            time_s=self.clock.simulation_time_s,
                            dt_s=dt * self._advance_rate,
                        )
                    self.bump_revision()


def running_clock(scenario: SimulationConfig) -> SimulationClock:
    """A clock that already advances time at the scenario's ``time_scale``.

    ``initial_snapshot`` keeps the epoch clock paused (it is the t = 0 state),
    but the running service starts immediately so the renderer sees motion.
    """
    epoch = scenario.initial_clock()
    return SimulationClock(
        epoch_utc=epoch.epoch_utc,
        simulation_time_s=0.0,
        time_scale=epoch.time_scale,
        paused=False,
    )


def serve(host: str = "127.0.0.1", port: int = 0, scenario: SimulationConfig | None = None) -> CoreServer:
    """Start the core HTTP server. port=0 picks a free ephemeral port."""
    server = CoreServer((host, port), CoreHandler)
    server.scenario = scenario or builtin_scenario()  # type: ignore[attr-defined]
    server.clock = running_clock(server.scenario)
    server.runtime = components.initial_runtime(server.scenario)  # type: ignore[attr-defined]
    server.sync_advance_rate()
    server.start_advance()
    return server