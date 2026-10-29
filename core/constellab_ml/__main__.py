"""Command-line entry points for the ConstellaLab learned-policy pipeline.

Usage:
    python -m constellab_ml campaign OUT --id NAME [--episodes N] [--seed S]
    python -m constellab_ml train CONFIG --out DIR [--resume-checkpoint PATH]
    python -m constellab_ml sweep CONFIG --out DIR
    python -m constellab_ml curriculum CONFIG --out DIR
    python -m constellab_ml evaluate CAMPAIGN --checkpoint PATH [--split test]
    python -m constellab_ml compare CAMPAIGN --checkpoint NAME=PATH [...] [--out FILE]
    python -m constellab_ml generalize CONFIG [--out FILE]
    python -m constellab_ml benchmark CONFIG [--out FILE]
    python -m constellab_ml ppo CONFIG --out DIR
    python -m constellab_ml rl CONFIG --out DIR
    python -m constellab_ml ablations CONFIG [--out FILE]
    python -m constellab_ml manifest CHECKPOINT --out FILE [--campaign DIR]
    python -m constellab_ml compat MANIFEST --campaign DIR [--out FILE]
    python -m constellab_ml shield --campaign DIR [--out FILE]
    python -m constellab_ml search CONFIG [--out FILE]
    python -m constellab_ml rank CONFIG --out DIR
    python -m constellab_ml gate BUNDLE [--out FILE] [--markdown FILE] [--no-replay]

Every command writes machine-readable artefacts and prints a compact JSON
summary on stdout; the artefacts are the source of truth. ``train``, ``sweep``,
``curriculum``, ``ppo``, ``rl``, ``generalize``, ``benchmark``, ``ablations``,
``search``, ``rank`` and ``gate`` read declarative JSON configurations (the same
payloads their dataclasses read), so a run can be reproduced by re-running the
command with the same file.

Exit codes:
    0  success
    2  invalid configuration, dataset or evidence bundle
    5  the run produced no usable checkpoint or the gate did not promote
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _write(path: str | None, payload: Any) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _cmd_campaign(args: argparse.Namespace) -> int:
    from .dataset import DatasetConfig, generate_campaign

    try:
        config = DatasetConfig(
            campaign_id=args.id,
            seed=args.seed,
            expert_key=args.expert,
            splits=tuple(args.splits),
            label=args.label or args.id,
        )
        report = generate_campaign(args.out, config, args.episodes, resume=not args.fresh)
    except Exception as exc:  # configuration and dataset errors share exit code 2
        print(f"campaign failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    manifest = report["manifest"]
    _print({
        "campaign_id": manifest["campaign_id"],
        "out": str(Path(args.out).resolve()),
        "counts": manifest["counts"],
        "families": manifest["coverage"].get("scenario_families", {}),
        "parameter_schema_hash": manifest["schema"]["parameter_schema_hash"],
        "leakage_clean": manifest["leakage"]["clean"],
    })
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .training import TrainingConfig, run_training

    try:
        config = TrainingConfig.load(args.config)
        manifest = run_training(config, args.out, resume_from=args.resume_checkpoint)
    except Exception as exc:
        print(f"training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "run_id": manifest["run_id"],
        "out": str(Path(args.out).resolve()),
        "epochs": len(manifest["history"]),
        "best_metric": manifest["best_metric"],
        "selection_metric": manifest["selection_metric"],
        "checkpoints": manifest["checkpoints"],
        "export": manifest.get("export"),
        "parameters": manifest["hyperparameters"]["parameters"],
    })
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import SweepConfig, SweepRunner

    try:
        runner = SweepRunner(SweepConfig.load(args.config), args.out)
        report = runner.run()
    except Exception as exc:
        print(f"sweep failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "sweep_id": report["sweep_id"],
        "out": str(Path(args.out).resolve()),
        "planned_runs": report["planned_runs"],
        "completed": report["completed"],
        "failed": report["failed"],
        "selection": report["selection"],
    })
    return 0 if report["selection"] else 5


def _cmd_curriculum(args: argparse.Namespace) -> int:
    from .curriculum import CurriculumConfig, CurriculumRunner

    try:
        runner = CurriculumRunner(CurriculumConfig.load(args.config), args.out)
        manifest = runner.run()
    except Exception as exc:
        print(f"curriculum failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "curriculum_id": manifest["curriculum_id"],
        "out": str(Path(args.out).resolve()),
        "stages": [stage["name"] for stage in manifest["stages"]],
        "seed_reports": [
            {
                "seed": report["seed"],
                "stages_completed": report["stages_completed"],
                "stop_reason": report["stop_reason"],
                "selection": report["selection"],
                "test": (
                    report["evaluation"].get("test", {}).get("policies", {}).get("learned", {})
                    .get("aggregate")
                ),
            }
            for report in manifest["seeds"]
        ],
        "selected_seed": manifest["selected_seed"],
        "seed_variation": manifest["seed_variation"],
    })
    return 0 if manifest["selected_seed"] else 5


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare named checkpoints on identical held-out episodes (Day 39 report)."""
    from .curriculum import baseline_policies
    from .dataset import EpisodeDataset
    from .evaluation import load_policy_checkpoint
    from .policies import compare_policies

    try:
        dataset = EpisodeDataset(args.campaign)
        records = dataset.episodes(args.split)
        if not records:
            print(f"campaign has no {args.split!r} episodes", file=sys.stderr)
            return 2
        policies: dict[str, Any] = dict(baseline_policies())
        for item in args.checkpoint:
            if "=" not in item:
                raise ValueError("each --checkpoint is NAME=PATH")
            name, path = item.split("=", 1)
            if not name.strip() or not Path(path).exists():
                raise ValueError(f"bad checkpoint entry {item!r}")
            policies[name.strip()] = load_policy_checkpoint(path)
        report = compare_policies(policies, records, limit=args.limit)
    except Exception as exc:
        print(f"comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    payload = {
        "campaign": dataset.manifest["campaign_id"],
        "split": args.split,
        "episodes": report["episodes"],
        "selection_note": (
            "comparison only: checkpoints must have been selected on validation data "
            "before this report reads the test split"
        ),
        "policies": {
            name: {"aggregate": entry["aggregate"], "families": entry["families"]}
            for name, entry in report["policies"].items()
        },
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
    _print(payload)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .curriculum import baseline_policies
    from .dataset import EpisodeDataset
    from .evaluation import load_policy_checkpoint
    from .policies import compare_policies
    from .training import load_export

    try:
        dataset = EpisodeDataset(args.campaign)
        records = dataset.episodes(args.split)
        if not records:
            print(f"campaign has no {args.split!r} episodes", file=sys.stderr)
            return 2
        policies: dict[str, Any] = dict(baseline_policies())
        if args.checkpoint:
            policies["learned"] = load_policy_checkpoint(args.checkpoint)
        if args.export:
            payload = load_export(args.export)
            payload["path"] = str(args.export)
            from .policies import build_model_policy_from_checkpoint

            policies["learned-export"] = build_model_policy_from_checkpoint(payload)
        if "learned" not in policies and "learned-export" not in policies:
            print("evaluate needs --checkpoint or --export", file=sys.stderr)
            return 3
        report = compare_policies(policies, records, limit=args.limit)
    except Exception as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "campaign": dataset.manifest["campaign_id"],
        "split": args.split,
        "episodes": report["episodes"],
        "policies": {
            name: {
                "aggregate": entry["aggregate"],
                "families": entry["families"],
            }
            for name, entry in report["policies"].items()
        },
    })
    return 0


def _cmd_generalize(args: argparse.Namespace) -> int:
    from .evaluation import run_generalization

    try:
        report = run_generalization(_load(args.config), args.out)
    except Exception as exc:
        print(f"generalization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "out": str(Path(args.out).resolve()) if args.out else None,
        "suite": report["suite"]["inventory"],
        "policies": {
            name: {
                "coverage": entry["aggregate"]["coverage"],
                "ood": (entry["slices"].get("ood") or {}).get("coverage", {}).get("mean"),
            }
            for name, entry in report["policies"].items()
        },
    })
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .evaluation import run_policy_benchmark

    try:
        report = run_policy_benchmark(_load(args.config), args.out)
    except Exception as exc:
        print(f"benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "out": str(Path(args.out).resolve()) if args.out else None,
        "episodes": report["suite"]["episodes"],
        "policies": {
            name: {
                "coverage": entry["aggregate"]["coverage"]["mean"],
                "conflicts": entry["aggregate"]["conflicts"],
                "invalid_actions": entry["aggregate"]["invalid_actions"],
                "p95_latency_ms": entry["aggregate"]["latency_ms"]["p95"],
                "shield_rejection": (entry.get("shield") or {}).get("rejected"),
            }
            for name, entry in report["policies"].items()
        },
    })
    return 0


def _cmd_ppo(args: argparse.Namespace) -> int:
    from .ppo import PPOConfig, train_ppo

    try:
        manifest = train_ppo(PPOConfig.load(args.config), args.out)
    except Exception as exc:
        print(f"ppo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "out": str(Path(args.out).resolve()),
        "updates": len(manifest.get("history", [])),
        "best_validation_coverage": manifest.get("best_validation_coverage"),
        "checkpoints": manifest.get("checkpoints"),
    })
    return 0 if (manifest.get("checkpoints") or {}).get("best") else 5


def _cmd_rl(args: argparse.Namespace) -> int:
    from .rl_curriculum import RLCurriculumConfig, run_rl_curriculum

    try:
        manifest = run_rl_curriculum(RLCurriculumConfig.load(args.config), args.out)
    except Exception as exc:
        print(f"rl curriculum failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "out": str(Path(args.out).resolve()),
        "seeds": len(manifest["seed_reports"]),
        "summary": manifest["summary"],
    })
    return 0 if manifest["summary"].get("winner") else 5


def _cmd_ablations(args: argparse.Namespace) -> int:
    from .ablations import run_ablations

    try:
        report = run_ablations(_load(args.config), args.out)
    except Exception as exc:
        print(f"ablations failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "out": str(Path(args.out).resolve()) if args.out else None,
        "variants": {
            name: row["metrics"] for name, row in report["variants"].items()
        },
    })
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    from .compatibility import ModelManifest

    try:
        entry = ModelManifest.from_checkpoint(args.checkpoint, dataset_root=args.campaign or None)
        payload = entry.to_dict()
        _write(args.out, payload)
    except Exception as exc:
        print(f"manifest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "out": str(Path(args.out).resolve()),
        "checkpoint_hash": entry.checkpoint_hash,
        "model_family": entry.model_family,
        "envelope_ranges": sorted(entry.envelope.ranges),
    })
    return 0


def _cmd_compat(args: argparse.Namespace) -> int:
    from .compatibility import compatibility_suite

    try:
        report = compatibility_suite(
            _load(args.manifest), args.campaign,
            split=args.split, limit=args.limit,
        )
        _write(args.out, report)
    except Exception as exc:
        print(f"compatibility failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "passed": report["passed"],
        "counts": report["counts"],
        "failed": report["failed"],
        "out": str(Path(args.out).resolve()) if args.out else None,
    })
    return 0 if report["passed"] else 5


def _cmd_shield(args: argparse.Namespace) -> int:
    from .dataset import EpisodeDataset
    from .env import EpisodeSpec, SwarmEnvironment
    from .shield import ShieldConfig, shield_suite

    try:
        from .evaluation import baseline_set

        records = EpisodeDataset(args.campaign).episodes(args.split)
        observation = None
        episode_id = None
        for record in records[: max(1, args.search)]:
            environment = SwarmEnvironment(EpisodeSpec.from_payload(record["spec"]))
            candidate, _ = environment.reset()
            probe = baseline_set()["expert-rule"]
            probe.reset()
            # The first instant can be quiet; advance with the transparent
            # expert until the observation carries a legal assign pair.
            for _ in range(24):
                if (candidate.get("mask") or {}).get("assign"):
                    break
                if environment.done:
                    break
                actions = probe.act(candidate)
                candidate, _, _, _, _ = environment.step(actions)
            if (candidate.get("mask") or {}).get("assign"):
                observation = {key: value for key, value in candidate.items() if key not in ("context", "notes")}
                episode_id = record.get("episode_id")
                break
        if observation is None:
            raise ValueError("no episode with a legal assign pair was found")
        config = _load(args.config) if args.config else None
        suite = shield_suite(observation, config=config, source=f"episode {episode_id}")
        _write(args.out, suite)
    except Exception as exc:
        print(f"shield suite failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print(suite["summary"])
    return 0 if suite["passed"] else 5


def _cmd_search(args: argparse.Namespace) -> int:
    from .adversarial import run_search

    try:
        config = _load(args.config)
        ranker = None
        if config.get("ranker"):
            from .ranker import load_ranker

            ranker = load_ranker(config["ranker"])
        report = run_search(config, args.out, ranker=ranker)
    except Exception as exc:
        print(f"search failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "out": str(Path(args.out).resolve()) if args.out else None,
        "mode": report["mode"],
        "summary": report["summary"],
    })
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    from .ranker import RankerConfig, train_ranker

    try:
        config = RankerConfig.from_payload(_load(args.config))
        manifest = train_ranker(config, args.out)
    except Exception as exc:
        print(f"ranker training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "out": str(Path(args.out).resolve()),
        "records": manifest["records"],
        "family_held_out": manifest["family_held_out"],
        "metrics": manifest["metrics"],
        "checkpoint": manifest["checkpoint"],
    })
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from .gate import render_gate_report, run_gate_file

    try:
        report = run_gate_file(args.bundle, replay=not args.no_replay)
        _write(args.out, report)
        if args.markdown:
            target = Path(args.markdown)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render_gate_report(report), encoding="utf-8")
    except Exception as exc:
        print(f"gate failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print({
        "status": report["status"],
        "promoted": report["promoted"],
        "fallback": report["fallback_policy"],
        "required_failures": report["required_failures"],
        "out": str(Path(args.out).resolve()) if args.out else None,
    })
    return 0 if report["status"] == "pass" else 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="constellab_ml", description="ConstellaLab learned-policy pipeline"
    )
    parser.add_argument("--version", action="version", version="constellab-ml 1")
    sub = parser.add_subparsers(dest="command", required=True)

    campaign = sub.add_parser("campaign", help="generate a randomized demonstration campaign")
    campaign.add_argument("out", metavar="OUT", help="campaign directory")
    campaign.add_argument("--id", required=True, help="campaign id (stored in the ledger)")
    campaign.add_argument("--episodes", type=int, default=8, help="episodes to generate (default 8)")
    campaign.add_argument("--seed", type=int, default=0, help="campaign seed (default 0)")
    campaign.add_argument("--expert", default="expert-rule", help="demonstration expert key")
    campaign.add_argument("--splits", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                          metavar=("TRAIN", "VALIDATION", "TEST"))
    campaign.add_argument("--label", default="", help="human-readable label")
    campaign.add_argument("--fresh", action="store_true", help="refuse to resume an existing ledger")
    campaign.set_defaults(func=_cmd_campaign)

    train = sub.add_parser("train", help="train one configuration")
    train.add_argument("config", metavar="CONFIG", help="TrainingConfig JSON file")
    train.add_argument("--out", required=True, metavar="DIR", help="run directory")
    train.add_argument("--resume-checkpoint", metavar="PATH", help="continue from this checkpoint")
    train.set_defaults(func=_cmd_train)

    sweep = sub.add_parser("sweep", help="run a bounded architecture/hyper-parameter sweep")
    sweep.add_argument("config", metavar="CONFIG", help="SweepConfig JSON file")
    sweep.add_argument("--out", required=True, metavar="DIR", help="sweep directory")
    sweep.set_defaults(func=_cmd_sweep)

    curriculum = sub.add_parser("curriculum", help="train the staged Day 45 curriculum")
    curriculum.add_argument("config", metavar="CONFIG", help="CurriculumConfig JSON file")
    curriculum.add_argument("--out", required=True, metavar="DIR", help="curriculum directory")
    curriculum.set_defaults(func=_cmd_curriculum)

    compare = sub.add_parser(
        "compare", help="compare named checkpoints and baselines on identical episodes"
    )
    compare.add_argument("campaign", metavar="CAMPAIGN", help="campaign directory")
    compare.add_argument(
        "--checkpoint", action="append", default=[], metavar="NAME=PATH",
        help="named checkpoint to compare (repeatable)",
    )
    compare.add_argument("--split", default="test", choices=("train", "validation", "test"))
    compare.add_argument("--limit", type=int, default=None, help="episodes to evaluate")
    compare.add_argument("--out", metavar="FILE", help="write the report payload here")
    compare.set_defaults(func=_cmd_compare)

    evaluate = sub.add_parser("evaluate", help="evaluate a checkpoint against baselines")
    evaluate.add_argument("campaign", metavar="CAMPAIGN", help="campaign directory")
    evaluate.add_argument("--checkpoint", metavar="PATH", help="training checkpoint (.pt)")
    evaluate.add_argument("--export", metavar="DIR", help="exported policy directory")
    evaluate.add_argument("--split", default="test", choices=("train", "validation", "test"))
    evaluate.add_argument("--limit", type=int, default=None, help="episodes to evaluate")
    evaluate.set_defaults(func=_cmd_evaluate)

    generalize = sub.add_parser("generalize", help="Day 46 held-out/OOD slice report")
    generalize.add_argument("config", metavar="CONFIG", help="generalization config JSON")
    generalize.add_argument("--out", metavar="FILE", help="write the full report here")
    generalize.set_defaults(func=_cmd_generalize)

    benchmark = sub.add_parser("benchmark", help="Day 52 paired policy comparison")
    benchmark.add_argument("config", metavar="CONFIG", help="benchmark config JSON")
    benchmark.add_argument("--out", metavar="FILE", help="write the full report here")
    benchmark.set_defaults(func=_cmd_benchmark)

    ppo = sub.add_parser("ppo", help="Day 47 PPO training run")
    ppo.add_argument("config", metavar="CONFIG", help="PPOConfig JSON")
    ppo.add_argument("--out", required=True, metavar="DIR", help="run directory")
    ppo.set_defaults(func=_cmd_ppo)

    rl = sub.add_parser("rl", help="Day 49 multi-seed RL curriculum")
    rl.add_argument("config", metavar="CONFIG", help="RLCurriculumConfig JSON")
    rl.add_argument("--out", required=True, metavar="DIR", help="campaign directory")
    rl.set_defaults(func=_cmd_rl)

    ablations = sub.add_parser("ablations", help="Day 50 declared ablation table")
    ablations.add_argument("config", metavar="CONFIG", help="ablation config JSON")
    ablations.add_argument("--out", metavar="FILE", help="write the table here")
    ablations.set_defaults(func=_cmd_ablations)

    manifest = sub.add_parser("manifest", help="write the promotion manifest for a checkpoint")
    manifest.add_argument("checkpoint", metavar="CHECKPOINT", help="checkpoint .pt")
    manifest.add_argument("--out", required=True, metavar="FILE", help="manifest JSON path")
    manifest.add_argument("--campaign", metavar="DIR", help="campaign that defines the envelope")
    manifest.set_defaults(func=_cmd_manifest)

    compat = sub.add_parser("compat", help="Day 51 compatibility and fallback suite")
    compat.add_argument("manifest", metavar="MANIFEST", help="model manifest JSON")
    compat.add_argument("--campaign", required=True, metavar="DIR", help="campaign directory")
    compat.add_argument("--split", default="test", choices=("train", "validation", "test"))
    compat.add_argument("--limit", type=int, default=3, help="dataset episodes to include")
    compat.add_argument("--out", metavar="FILE", help="write the suite report here")
    compat.set_defaults(func=_cmd_compat)

    shield = sub.add_parser("shield", help="Day 48/56 shield boundary suite")
    shield.add_argument("--campaign", required=True, metavar="DIR", help="campaign directory")
    shield.add_argument("--split", default="test", choices=("train", "validation", "test"))
    shield.add_argument("--search", type=int, default=40, help="episodes to scan for a legal pair")
    shield.add_argument("--config", metavar="FILE", help="ShieldConfig JSON")
    shield.add_argument("--out", metavar="FILE", help="write the suite report here")
    shield.set_defaults(func=_cmd_shield)

    search = sub.add_parser("search", help="Day 53/54 mutation search")
    search.add_argument("config", metavar="CONFIG", help="search config JSON")
    search.add_argument("--out", metavar="FILE", help="write the full report here")
    search.set_defaults(func=_cmd_search)

    rank = sub.add_parser("rank", help="Day 55 train a mutation ranker")
    rank.add_argument("config", metavar="CONFIG", help="RankerConfig JSON")
    rank.add_argument("--out", required=True, metavar="DIR", help="ranker directory")
    rank.set_defaults(func=_cmd_rank)

    gate = sub.add_parser("gate", help="Day 56 G3 promotion gate")
    gate.add_argument("bundle", metavar="BUNDLE", help="gate bundle JSON")
    gate.add_argument("--out", metavar="FILE", help="write the gate report here")
    gate.add_argument("--markdown", metavar="FILE", help="write a Markdown summary here")
    gate.add_argument("--no-replay", action="store_true", help="skip deterministic re-execution")
    gate.set_defaults(func=_cmd_gate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
