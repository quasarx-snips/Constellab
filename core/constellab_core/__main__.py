"""Command-line entry point for the ConstellaLab core.

Usage:
    python -m constellab_core status [--config PATH]
    python -m constellab_core validate PATH
    python -m constellab_core snapshot [PATH]
    python -m constellab_core run [PATH] [--duration S] [--step S] [--out FILE]
    python -m constellab_core serve [--host HOST] [--port PORT] [--scenario PATH]

PATH may be a scenario JSON file or a built-in scenario name
(``empty``, ``visual-orbit-demo``, ``swarm-48``).  ``run`` advances the simulation
without starting a server or a browser, which is what batch work uses.

Exit codes:
    0  success
    2  invalid configuration
    3  usage error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import BUILTIN_SCENARIOS, DEFAULT_SCENARIO, ConfigError, load_config_file
from .contract import SimulationClock
from .orbit import initial_snapshot, snapshot_at_time
from .service import serve, status_payload


def _print_errors(exc: ConfigError) -> int:
    print(json.dumps({"valid": False, "errors": exc.errors}, indent=2), file=sys.stderr)
    return 2


def _cmd_status(args: argparse.Namespace) -> int:
    if args.config:
        try:
            load_config_file(args.config)
        except ConfigError as exc:
            return _print_errors(exc)
    print(json.dumps(status_payload(), indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        config = load_config_file(args.path)
    except ConfigError as exc:
        return _print_errors(exc)
    print(json.dumps({"valid": True, "effective_config": config.to_dict()}, indent=2))
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    try:
        config = load_config_file(args.path)
    except ConfigError as exc:
        return _print_errors(exc)
    print(initial_snapshot(config).to_json(indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Headless simulation: emit state at a fixed cadence, no server involved."""
    if args.step <= 0 or args.duration < 0:
        print("step must be > 0 and duration >= 0", file=sys.stderr)
        return 3
    try:
        config = load_config_file(args.path)
    except ConfigError as exc:
        return _print_errors(exc)

    steps = int(args.duration // args.step) + 1
    # Records describe an advancing run, not the paused epoch clock.
    clock = SimulationClock(
        epoch_utc=config.epoch_utc,
        simulation_time_s=0.0,
        time_scale=config.time_scale,
        paused=False,
    )
    from . import components

    # Day 18: component state advances along the same clock in a headless run.
    runtime = components.initial_runtime(config)
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    started = time.perf_counter()
    try:
        for index in range(steps):
            t = index * args.step
            snapshot = snapshot_at_time(config, t, clock, runtime=runtime)
            if index + 1 < steps:
                next_t = (index + 1) * args.step
                components.advance_runtime(config, runtime, time_s=next_t, dt_s=next_t - t)
            record = {
                "t": round(t, 6),
                "scenario_id": snapshot.scenario_id,
                "clock": snapshot.clock.to_dict(),
                "satellites": [
                    {
                        "id": sat.id,
                        "position_eci_km": list(sat.position_eci_km),
                        "velocity_eci_km_s": list(sat.velocity_eci_km_s),
                        **({"components": [
                            {"type": c.type, "state": c.state} for c in sat.components
                        ]} if sat.components else {}),
                    }
                    for sat in snapshot.satellites
                ],
            }
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
    finally:
        if args.out:
            out.close()
    elapsed = time.perf_counter() - started
    summary = {
        "scenario_id": config.scenario_id,
        "satellites": len(config.satellites),
        "steps": steps,
        "step_s": args.step,
        "duration_s": args.duration,
        "wall_s": round(elapsed, 3),
        "out": args.out,
    }
    # Records own stdout; the run summary goes to stderr so it stays pipeable.
    print(json.dumps(summary), file=sys.stderr)
    return 0


def _cmd_g1(args: argparse.Namespace) -> int:
    """Day 24: generate and persist the automated G1 acceptance report."""
    from .g1 import write_report

    target = write_report(Path(args.out) if args.out else None)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["report_path"] = str(target)
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_g2(args: argparse.Namespace) -> int:
    """Day 34: generate and persist the G2 failure gate report."""
    from .g2 import write_report

    target = write_report(Path(args.out) if args.out else None)
    payload = json.loads(target.read_text(encoding="utf-8"))
    summary = {
        "report_path": str(target),
        "checks": payload["checks"],
        "fault_trace_hash": payload["fault"]["trace_hash"],
        "control_trace_hash": payload["control"]["trace_hash"],
        "first_violation": payload["first_violation"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if payload["checks"]["g2_pass"] else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        scenario = load_config_file(args.scenario)
    except ConfigError as exc:
        return _print_errors(exc)
    server = serve(args.host, args.port, scenario)
    actual = server.server_address
    print(f"constellab-core {__version__} listening on http://{actual[0]}:{actual[1]} "
          f"(scenario {scenario.scenario_id}, {len(scenario.satellites)} satellites)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    builtin = ", ".join(BUILTIN_SCENARIOS)
    parser = argparse.ArgumentParser(prog="constellab_core", description="ConstellaLab simulation core")
    parser.add_argument("--version", action="version", version=f"constellab-core {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="print the version/status payload as JSON")
    p_status.add_argument("--config", metavar="PATH", help="validate a scenario file first")
    p_status.set_defaults(func=_cmd_status)

    p_validate = sub.add_parser("validate", help="validate a scenario file and print the effective config")
    p_validate.add_argument("path", metavar="PATH", help=f"scenario JSON file or built-in name ({builtin})")
    p_validate.set_defaults(func=_cmd_validate)

    p_snapshot = sub.add_parser("snapshot", help="print the epoch snapshot of a scenario")
    p_snapshot.add_argument("path", metavar="PATH", nargs="?", default=DEFAULT_SCENARIO,
                            help=f"scenario JSON file or built-in name ({builtin})")
    p_snapshot.set_defaults(func=_cmd_snapshot)

    p_run = sub.add_parser("run", help="advance a scenario headlessly and stream state records")
    p_run.add_argument("path", metavar="PATH", nargs="?", default=DEFAULT_SCENARIO,
                       help=f"scenario JSON file or built-in name ({builtin})")
    p_run.add_argument("--duration", type=float, default=600.0, metavar="SECONDS",
                       help="simulated seconds to cover (default 600)")
    p_run.add_argument("--step", type=float, default=10.0, metavar="SECONDS",
                       help="simulated seconds between records (default 10)")
    p_run.add_argument("--out", metavar="FILE", help="write newline-delimited records here")
    p_run.set_defaults(func=_cmd_run)

    p_g1 = sub.add_parser("g1", help="generate the automated G1 acceptance report")
    p_g1.add_argument("--out", metavar="FILE", help="report target (default benchmarks/g1-acceptance.json)")
    p_g1.set_defaults(func=_cmd_g1)

    p_g2 = sub.add_parser("g2", help="generate the automated G2 failure-gate report")
    p_g2.add_argument("--out", metavar="FILE", help="report target (default benchmarks/g2-acceptance.json)")
    p_g2.set_defaults(func=_cmd_g2)

    p_serve = sub.add_parser("serve", help="run the local core HTTP service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=0)
    p_serve.add_argument("--scenario", metavar="PATH", default=DEFAULT_SCENARIO,
                         help=f"scenario served by /api/snapshot ({builtin})")
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
