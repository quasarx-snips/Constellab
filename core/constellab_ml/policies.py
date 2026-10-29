"""Running a trained model as a policy, and evaluating it the way it is used.

:class:`ModelPolicy` is the bridge from a checkpoint back into the environment:
it normalizes the live observation with the checkpoint's own normalization,
scores the masked candidate pairs, performs the same greedy matching the expert
demonstrations were generated with, and abstains when nothing is feasible. A
recurrent model keeps its hidden state per *episode*, and the state is dropped on
``reset`` — memory never crosses an episode boundary.

:func:`evaluate_policy` runs a policy over stored episodes and reports metrics
per scenario family, which is how the Day 39/45 comparisons against the expert
and the rule baseline are produced: same episodes, same reward terms, same
metric definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .batch import GraphSample, StepBatch, build_batch
from .env import ACTION_ASSIGN, ACTION_WAIT, Action, EnvPolicy, EpisodeSpec, SwarmEnvironment
from .expert import Candidate, greedy_assignment, score_candidates
from .features import Normalizer

POLICY_VERSION_LEARNED = "learned-policy-1"


@dataclass
class LearnedPolicyMetadata:
    key: str = "learned-gnn"
    kind: str = "learned"
    deployable: bool = True
    version: str = POLICY_VERSION_LEARNED
    checkpoint: str = ""
    model: dict[str, Any] = field(default_factory=dict)
    feature_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "deployable": self.deployable,
            "version": self.version,
            "checkpoint": self.checkpoint,
            "model": dict(self.model),
            "feature_schema": dict(self.feature_schema),
        }


class ModelPolicy(EnvPolicy):
    """A checkpoint running inside the environment as a swarm policy."""

    label = "learned-policy"
    version = POLICY_VERSION_LEARNED

    def __init__(
        self,
        model: Any,
        normalizer: Normalizer,
        *,
        metadata: LearnedPolicyMetadata | None = None,
        device: str = "cpu",
    ) -> None:
        import torch

        self.model = model
        self.normalizer = normalizer
        self.metadata_ = metadata or LearnedPolicyMetadata()
        self.device = device
        self.state: dict[str, Any] | None = None
        self.torch = torch
        self.model.eval()

    # --- lifecycle --------------------------------------------------------------------
    def reset(self) -> None:
        self.state = None

    def candidates(self, observation: Mapping[str, Any]) -> tuple[list[Candidate], StepBatch, Any]:
        sample = GraphSample(
            observation=dict(observation),
            targets=[],
            context=observation.get("context") or None,
        )
        batch = build_batch([sample])
        normalized = self.normalizer.apply(batch)
        with self.torch.no_grad():
            output = self.model(normalized, state=self.state)
        self.state = output.state or None
        candidates = []
        abstain = float(output.abstain_logits[0].item()) if output.abstain_logits.numel() else 0.0
        for edge_id, (src, dst) in enumerate(zip(batch.assign_src, batch.assign_dst)):
            candidates.append(Candidate(
                satellite_index=src,
                task_index=dst,
                satellite_id=str(observation["satellite_ids"][src]),
                task_id=str(observation["task_ids"][dst]),
                score=float(output.edge_logits[edge_id].item()),
                terms={"abstain_baseline": abstain},
            ))
        candidates.sort(key=lambda item: (-item.score, item.satellite_id, item.task_id))
        return candidates, batch, output

    def act(self, observation: Mapping[str, Any]) -> list[Action]:
        candidates, batch, output = self.candidates(observation)
        chosen = greedy_assignment(candidates)
        if not chosen:
            return [Action(kind=ACTION_WAIT)]
        return [
            Action(
                kind=ACTION_ASSIGN,
                satellite_id=candidate.satellite_id,
                task_id=candidate.task_id,
                score=candidate.score,
            )
            for candidate in chosen
        ]

    def act_with_abstain(self, observation: Mapping[str, Any]) -> list[Action]:
        """Like ``act`` but an explicit abstain class can veto every assignment.

        A satellite is only tasked when its best candidate outscores the learned
        abstain score: a policy that has learned "wait" must be able to use it.
        """
        candidates, _, output = self.candidates(observation)
        if not candidates:
            return [Action(kind=ACTION_WAIT)]
        abstain = float(output.abstain_logits[0].item()) if output.abstain_logits.numel() else 0.0
        filtered = [Candidate(**{**candidate.__dict__, "score": candidate.score}) for candidate in candidates]
        supported: list[Candidate] = []
        for candidate in filtered:
            if candidate.score > abstain:
                supported.append(candidate)
            else:
                candidate.dropped_reason = "abstain class scored higher"
        chosen = greedy_assignment(supported)
        if not chosen:
            return [Action(kind=ACTION_WAIT)]
        return [
            Action(
                kind=ACTION_ASSIGN,
                satellite_id=candidate.satellite_id,
                task_id=candidate.task_id,
                score=candidate.score,
            )
            for candidate in chosen
        ]

    def metadata(self) -> dict[str, Any]:
        return self.metadata_.to_dict()


def build_model_policy_from_checkpoint(checkpoint: Mapping[str, Any]) -> ModelPolicy:
    """Rebuild a policy from a checkpoint payload (model + normalizer + config)."""
    from .models import build_model

    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    normalizer = Normalizer.from_payload(checkpoint["normalizer"])
    metadata = LearnedPolicyMetadata(
        key=str(checkpoint.get("policy_key", "learned-gnn")),
        checkpoint=str(checkpoint.get("path", "")),
        model=dict(checkpoint["model_config"]),
        feature_schema=dict(checkpoint.get("feature_schema", {})),
    )
    return ModelPolicy(model, normalizer, metadata=metadata)


@dataclass
class EvaluationReport:
    """Outcome of a policy over a set of episodes, aggregated per family."""

    policy: dict[str, Any]
    episodes: list[dict[str, Any]] = field(default_factory=list)
    families: dict[str, dict[str, Any]] = field(default_factory=dict)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": dict(self.policy),
            "episodes": list(self.episodes),
            "families": {key: dict(value) for key, value in self.families.items()},
            "aggregate": dict(self.aggregate),
        }


def evaluate_policy(
    policy: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int | None = None,
    use_abstain_class: bool = False,
) -> EvaluationReport:
    """Run ``policy`` over stored episodes and aggregate metrics per family."""
    episodes: list[dict[str, Any]] = []
    for record in records[: limit if limit is not None else len(records)]:
        spec = EpisodeSpec.from_payload(record["spec"])
        environment = SwarmEnvironment(spec)
        target = _AbstainAware(policy) if use_abstain_class and hasattr(policy, "act_with_abstain") else policy
        outcome = environment.rollout(target)
        metrics = dict(outcome.metrics)
        episodes.append({
            "episode_id": record["episode_id"],
            "family": record.get("family"),
            "condition_class": record.get("condition_class"),
            "coverage": metrics.get("coverage", 0.0),
            "observed": metrics.get("observed", 0),
            "expired": metrics.get("expired", 0),
            "matrix_completed": metrics.get("completed", 0),
            "conflicts": metrics.get("conflicts", 0),
            "invalid_actions": metrics.get("invalid_actions", 0),
            "mean_latency_s": metrics.get("mean_latency_s"),
            "mean_reward": metrics.get("mean_reward", 0.0),
            "reward_terms": metrics.get("reward_terms", {}),
            "tasks_total": metrics.get("tasks_total", 0),
        })
    families: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        bucket = families.setdefault(str(episode["family"]), {"episodes": 0, "coverage": 0.0,
                                                              "observed": 0, "expired": 0,
                                                              "conflicts": 0, "reward": 0.0,
                                                              "tasks_total": 0})
        bucket["episodes"] += 1
        bucket["coverage"] += float(episode["coverage"])
        bucket["observed"] += int(episode["observed"])
        bucket["expired"] += int(episode["expired"])
        bucket["conflicts"] += int(episode["conflicts"])
        bucket["reward"] += float(episode["mean_reward"])
        bucket["tasks_total"] += int(episode["tasks_total"])
    for family, bucket in families.items():
        count = max(1, bucket["episodes"])
        bucket["mean_coverage"] = round(bucket["coverage"] / count, 6)
        bucket["mean_reward"] = round(bucket["reward"] / count, 6)
        bucket["coverage_by_tasks"] = (
            round(bucket["observed"] / bucket["tasks_total"], 6)
            if bucket["tasks_total"] else 0.0
        )
        del bucket["coverage"], bucket["reward"]
    total = max(1, len(episodes))
    aggregate = {
        "episodes": len(episodes),
        "mean_coverage": round(sum(float(item["coverage"]) for item in episodes) / total, 6),
        "mean_reward": round(sum(float(item["mean_reward"]) for item in episodes) / total, 6),
        "observed": sum(int(item["observed"]) for item in episodes),
        "expired": sum(int(item["expired"]) for item in episodes),
        "conflicts": sum(int(item["conflicts"]) for item in episodes),
        "invalid_actions": sum(int(item["invalid_actions"]) for item in episodes),
        "tasks_total": sum(int(item["tasks_total"]) for item in episodes),
    }
    aggregate["coverage_by_tasks"] = (
        round(aggregate["observed"] / aggregate["tasks_total"], 6) if aggregate["tasks_total"] else 0.0
    )
    return EvaluationReport(
        policy=policy.metadata() if hasattr(policy, "metadata") else {"key": policy.label},
        episodes=episodes,
        families=families,
        aggregate=aggregate,
    )


class _AbstainAware(EnvPolicy):
    """Wrap a model policy so the environment calls its abstain-aware action."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.label = getattr(policy, "label", "learned-policy")

    def reset(self) -> None:
        self.policy.reset()

    def act(self, observation: Mapping[str, Any]) -> list[Action]:
        return self.policy.act_with_abstain(observation)

    def metadata(self) -> dict[str, Any]:
        return self.policy.metadata()


def compare_policies(
    policies: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate several policies on identical episodes and compare per family."""
    reports: dict[str, Any] = {}
    for key, policy in policies.items():
        report = evaluate_policy(policy, records, limit=limit)
        reports[key] = {
            "aggregate": report.aggregate,
            "families": report.families,
        }
    return {
        "episodes": len(records[: limit if limit is not None else len(records)]),
        "policies": reports,
    }


__all__ = [
    "POLICY_VERSION_LEARNED",
    "LearnedPolicyMetadata",
    "ModelPolicy",
    "build_model_policy_from_checkpoint",
    "EvaluationReport",
    "evaluate_policy",
    "compare_policies",
]
