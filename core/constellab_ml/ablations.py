"""Day 50: diagnostic ablation table for the learned policy path.

Ablations are deliberately *variants of a frozen candidate*, not a second model
selection contest. Each variant changes one factor at inference/evaluation time
and is labelled diagnostic when it removes something the deployable policy must
keep (the shield, communication features or parameter context). The report
keeps negative and unstable results instead of dropping them.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .env import Action
from .evaluation import (
    EvaluationError,
    baseline_set,
    build_generalization_suite,
    evaluate_policy_records,
    load_policy_checkpoint,
)
from .ppo import PPOPolicy
from .shield import ShieldConfig

ABLATION_VERSION = "policy-ablations-1"


class AblationError(RuntimeError):
    """An ablation variant is invalid or its checkpoint can not be loaded."""


@dataclass
class ObservationTransformPolicy:
    """Apply one declared observation ablation before the wrapped policy."""

    policy: Any
    transform: str

    @property
    def label(self) -> str:
        return f"{getattr(self.policy, 'label', 'policy')}[{self.transform}]"

    def reset(self) -> None:
        self.policy.reset()

    def act(self, observation: Mapping[str, Any]) -> list[Action]:
        return self.policy.act(self.transform_observation(observation))

    def metadata(self) -> dict[str, Any]:
        data = self.policy.metadata() if hasattr(self.policy, "metadata") else {}
        return {**dict(data), "ablated": self.transform}

    def transform_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        changed = copy.deepcopy(dict(observation))
        if self.transform == "no-communication":
            changed.setdefault("edges", {})["contact"] = []
            for row in changed.get("satellites", []):
                features = list(row.get("features") or [])
                if features:
                    # contact degree is the twelfth satellite feature
                    features[11] = 0.0
                    row["features"] = features
        elif self.transform == "no-parameter-context":
            changed["context"] = {}
        elif self.transform == "no-task-features":
            for row in changed.get("tasks", []):
                row["features"] = [0.0] * len(row.get("features") or [])
        elif self.transform == "none":
            pass
        else:
            raise AblationError(f"unknown observation transform {self.transform!r}")
        return changed


@dataclass
class AblationVariant:
    name: str
    checkpoint: str = ""
    kind: str = "ppo"  # ppo | imitation
    transform: str = "none"
    shield: bool = False
    diagnostic: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise AblationError("a variant needs a name")
        if self.kind not in ("ppo", "imitation"):
            raise AblationError("variant kind must be ppo or imitation")
        if self.transform not in ("none", "no-communication", "no-parameter-context", "no-task-features"):
            raise AblationError(f"unknown transform {self.transform!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "kind": self.kind,
            "transform": self.transform,
            "shield": self.shield,
            "diagnostic": self.diagnostic,
            "note": self.note,
        }


def default_variants(checkpoint: str, *, imitation: str = "") -> list[AblationVariant]:
    variants = [
        AblationVariant("full", checkpoint, "ppo", "none", True, False,
                        "deployable candidate with the independent shield"),
        AblationVariant("no-shield", checkpoint, "ppo", "none", False, True,
                        "diagnostic only; never deployable"),
        AblationVariant("no-communication", checkpoint, "ppo", "no-communication", True, True,
                        "diagnostic: contact relation removed at inference"),
        AblationVariant("no-parameter-context", checkpoint, "ppo", "no-parameter-context", True, True,
                        "diagnostic: parameter tokens removed"),
        AblationVariant("no-task-features", checkpoint, "ppo", "no-task-features", True, True,
                        "diagnostic: task feature columns zeroed"),
    ]
    if imitation:
        variants.append(AblationVariant(
            "imitation-only", imitation, "imitation", "none", True, False,
            "supervised candidate, no reward fine-tuning",
        ))
    return variants


def load_variant_policy(variant: AblationVariant) -> Any:
    try:
        if variant.kind == "ppo":
            policy = PPOPolicy.load(variant.checkpoint)
        else:
            policy = load_policy_checkpoint(variant.checkpoint)
    except Exception as exc:
        raise AblationError(f"variant {variant.name!r} could not load: {exc}") from exc
    if variant.transform != "none":
        policy = ObservationTransformPolicy(policy, variant.transform)
    return policy


def run_ablations(
    config: Mapping[str, Any],
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate declared variants on one frozen suite and write the table."""
    variants = [
        AblationVariant(
            name=str(item["name"]),
            checkpoint=str(item.get("checkpoint", "")),
            kind=str(item.get("kind", "ppo")),
            transform=str(item.get("transform", "none")),
            shield=bool(item.get("shield", False)),
            diagnostic=bool(item.get("diagnostic", False)),
            note=str(item.get("note", "")),
        )
        for item in config.get("variants", [])
    ]
    if not variants:
        raise AblationError("no variants declared")
    suite = build_generalization_suite(
        config["dataset_root"],
        split=str(config.get("split", "test")),
        limit=config.get("limit"),
        extra_episodes=int(config.get("extra_episodes", 0)),
        seed=int(config.get("seed", 50)),
    )
    rows: dict[str, Any] = {}
    for index, variant in enumerate(variants):
        policy = load_variant_policy(variant) if variant.checkpoint else baseline_set()["expert-rule"]
        shield_config = ShieldConfig.from_payload(config.get("shield") or {})
        report = evaluate_policy_records(
            policy,
            suite["records"],
            seed=int(config.get("seed", 50)) + index * 101,
            use_shield=variant.shield,
            shield_config=shield_config,
        )
        rows[variant.name] = {
            "variant": variant.to_dict(),
            "aggregate": report["aggregate"],
            "slices": report["slices"],
            "shield": report["shield"],
            "metrics": {
                "mean_coverage": report["aggregate"]["coverage"]["mean"],
                "coverage_ci": [report["aggregate"]["coverage"]["low"], report["aggregate"]["coverage"]["high"]],
                "mean_reward": report["aggregate"]["reward"]["mean"],
                "conflicts": report["aggregate"]["conflicts"],
                "invalid_actions": report["aggregate"]["invalid_actions"],
            },
        }
    payload = {
        "version": ABLATION_VERSION,
        "generated_at_s": __import__("time").time(),
        "suite": {
            "episodes": len(suite["records"]),
            "split": suite["split"],
            "seed": suite["seed"],
        },
        "variants": rows,
        "selection_note": (
            "one factor at a time; diagnostic variants are retained as negative "
            "results and are not deployable candidates"
        ),
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "ABLATION_VERSION",
    "AblationError",
    "ObservationTransformPolicy",
    "AblationVariant",
    "default_variants",
    "load_variant_policy",
    "run_ablations",
]
