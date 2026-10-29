"""Day 47: centralized-training / decentralized-execution PPO.

The actor is the same permutation-invariant graph model used for imitation: it
scores feasible satellite→task edges and an explicit wait class. The critic is
*centralized* during training: it pools the graph's node embeddings together
with the global features. At execution time only the actor path is used, so a
deployed satellite never needs the critic or another agent's state.

This is deliberately a single joint decision per environment step (one edge or
wait), not an unvalidated multi-action burst. The environment still owns the
mask and the action validation. The implementation is a compact, testable PPO:
GAE(λ), clipped surrogate, value loss, entropy bonus, gradient clipping, seeded
rollouts and resumable checkpoints.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from constellab_core import trace as core_trace

from . import ENV_VERSION, GLOBAL_FEATURES, OBSERVATION_VERSION, feature_schema_payload
from .batch import GraphSample, StepBatch, build_batch
from .env import ACTION_ASSIGN, ACTION_WAIT, Action, EpisodeSpec, SwarmEnvironment
from .features import Normalizer
from .models import ModelError, build_model, model_parameter_count

PPO_VERSION = "ppo-1"
PPO_CHECKPOINT_VERSION = "ppo-checkpoint-1"


class PPOError(RuntimeError):
    """A PPO configuration, rollout or checkpoint is invalid."""


@dataclass
class PPOConfig:
    """Declarative PPO training configuration."""

    run_id: str = "ppo"
    dataset_root: str = ""
    model: dict[str, Any] = field(default_factory=lambda: {"model": "gnn", "hidden": 32, "layers": 1})
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 3
    updates: int = 4
    episodes_per_update: int = 2
    max_episode_steps: int = 12
    max_train_episodes: int = 8
    validation_episodes: int = 2
    evaluate_every: int = 1
    seed: int = 0
    reward_scale: float = 1.0
    normalize_advantages: bool = True
    resume_from: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            raise PPOError("run_id must be non-empty")
        family = str(self.model.get("model", "gnn"))
        if family in ("gnn_gru", "gnn_attention"):
            raise PPOError(
                "the compact PPO path currently uses a non-recurrent graph actor; "
                "use gnn/deepsets/mlp and keep recurrence for imitation"
            )
        if self.learning_rate <= 0.0:
            raise PPOError("learning_rate must be positive")
        if not 0.0 < self.clip_ratio < 1.0:
            raise PPOError("clip_ratio must be within (0, 1)")
        if self.updates <= 0 or self.epochs <= 0:
            raise PPOError("updates and epochs must be positive")
        if self.episodes_per_update <= 0 or self.max_episode_steps <= 0:
            raise PPOError("rollout sizes must be positive")
        if self.gamma <= 0.0 or self.gamma > 1.0:
            raise PPOError("gamma must be within (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_root": self.dataset_root,
            "model": dict(self.model),
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "epochs": self.epochs,
            "updates": self.updates,
            "episodes_per_update": self.episodes_per_update,
            "max_episode_steps": self.max_episode_steps,
            "max_train_episodes": self.max_train_episodes,
            "validation_episodes": self.validation_episodes,
            "evaluate_every": self.evaluate_every,
            "seed": self.seed,
            "reward_scale": self.reward_scale,
            "normalize_advantages": self.normalize_advantages,
            "resume_from": self.resume_from,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PPOConfig":
        return cls(
            run_id=str(payload.get("run_id", "ppo")),
            dataset_root=str(payload.get("dataset_root", "")),
            model=dict(payload.get("model") or {"model": "gnn", "hidden": 32, "layers": 1}),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            gamma=float(payload.get("gamma", 0.99)),
            gae_lambda=float(payload.get("gae_lambda", 0.95)),
            clip_ratio=float(payload.get("clip_ratio", 0.2)),
            entropy_coef=float(payload.get("entropy_coef", 0.01)),
            value_coef=float(payload.get("value_coef", 0.5)),
            max_grad_norm=float(payload.get("max_grad_norm", 0.5)),
            epochs=int(payload.get("epochs", 3)),
            updates=int(payload.get("updates", 4)),
            episodes_per_update=int(payload.get("episodes_per_update", 2)),
            max_episode_steps=int(payload.get("max_episode_steps", 12)),
            max_train_episodes=int(payload.get("max_train_episodes", 8)),
            validation_episodes=int(payload.get("validation_episodes", 2)),
            evaluate_every=int(payload.get("evaluate_every", 1)),
            seed=int(payload.get("seed", 0)),
            reward_scale=float(payload.get("reward_scale", 1.0)),
            normalize_advantages=bool(payload.get("normalize_advantages", True)),
            resume_from=str(payload.get("resume_from", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PPOConfig":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


class ActorCritic(nn.Module):
    """Graph actor plus a centralized value head over pooled graph state."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.backbone = build_model(self.config)
        hidden = int(self.config.get("hidden", 32))
        self.value_head = nn.Sequential(
            nn.Linear(hidden + len(GLOBAL_FEATURES), hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: StepBatch) -> tuple[Any, torch.Tensor]:
        output = self.backbone(batch)
        pooled = torch.zeros(batch.graph_count, output.node_embeddings.shape[-1])
        if batch.node_graph and output.node_embeddings.numel():
            index = torch.tensor(batch.node_graph, dtype=torch.long)
            pooled.index_add_(0, index, output.node_embeddings)
            counts = torch.zeros(batch.graph_count).index_add_(
                0, index, torch.ones(len(batch.node_graph))
            )
            pooled = pooled / counts.clamp(min=1.0).unsqueeze(-1)
        graph = torch.zeros(batch.graph_count, len(GLOBAL_FEATURES))
        for index, row in enumerate(batch.graph_features):
            graph[index, : len(row)] = torch.tensor(row, dtype=torch.float32)
        value = self.value_head(torch.cat([pooled, graph], dim=-1)).squeeze(-1)
        if value.numel() == 1:
            value = value[0]
        return output, value


@dataclass
class Transition:
    observation: dict[str, Any]
    action_index: int
    logprob: float
    value: float
    reward: float
    done: bool
    edge_count: int


class PPOPolicy:
    """Deployable actor wrapper; sampling is enabled only during training."""

    label = "ppo-graph"
    version = PPO_VERSION

    def __init__(
        self,
        model: ActorCritic,
        normalizer: Normalizer,
        config: PPOConfig,
        *,
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.config = config
        self.deterministic = deterministic
        self.model.eval()

    def reset(self) -> None:
        return None

    def _batch(self, observation: Mapping[str, Any]) -> StepBatch:
        sample = GraphSample(
            observation=dict(observation),
            context=observation.get("context") or None,
        )
        return self.normalizer.apply(build_batch([sample]))

    def _logits(
        self, observation: Mapping[str, Any], *, grad: bool = False
    ) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor]:
        batch = self._batch(observation)
        if grad:
            output, value = self.model(batch)
        else:
            with torch.no_grad():
                output, value = self.model(batch)
        allowed_pairs = {
            (int(pair[0]), int(pair[1])) for pair in observation["mask"]["assign"]
        }
        edge_ids = [
            edge_id
            for edge_id, (source, destination) in enumerate(
                zip(batch.assign_src, batch.assign_dst)
            )
            if (source, destination) in allowed_pairs
        ]
        wait_logit = (
            output.abstain_logits[0] if output.abstain_logits.numel() else torch.tensor(0.0)
        )
        if edge_ids:
            logits = torch.cat([output.edge_logits[edge_ids], wait_logit.reshape(1)])
        else:
            logits = wait_logit.reshape(1)
        return logits, edge_ids, value, output

    def sample(self, observation: Mapping[str, Any]) -> tuple[list[Action], Transition]:
        logits, edge_ids, value, _ = self._logits(observation)
        distribution = torch.distributions.Categorical(logits=logits)
        index = int(distribution.sample().item())
        logprob = float(distribution.log_prob(torch.tensor(index)).item())
        action = self._action(observation, index, edge_ids)
        return [action], Transition(
            observation=dict(observation),
            action_index=index,
            logprob=logprob,
            value=float(value.detach().item()),
            reward=0.0,
            done=False,
            edge_count=len(edge_ids),
        )

    def act(self, observation: Mapping[str, Any]) -> list[Action]:
        logits, edge_ids, _, _ = self._logits(observation)
        index = int(torch.argmax(logits).item())
        return [self._action(observation, index, edge_ids)]

    def act_with_abstain(self, observation: Mapping[str, Any]) -> list[Action]:
        return self.act(observation)

    @staticmethod
    def _action(observation: Mapping[str, Any], index: int, edge_ids: list[int]) -> Action:
        if index >= len(edge_ids):
            return Action(kind=ACTION_WAIT)
        edge_id = edge_ids[index]
        local_satellite, local_task = (
            int(observation["edges"]["assign"][edge_id][0]),
            int(observation["edges"]["assign"][edge_id][1]),
        )
        return Action(
            kind=ACTION_ASSIGN,
            satellite_id=str(observation["satellite_ids"][local_satellite]),
            task_id=str(observation["task_ids"][local_task]),
            score=0.0,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.label,
            "kind": "ppo",
            "deployable": True,
            "version": PPO_VERSION,
            "model": dict(self.config.model),
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "checkpoint_version": PPO_CHECKPOINT_VERSION,
            "policy_key": self.label,
            "model_config": dict(self.config.model),
            "state_dict": self.model.state_dict(),
            "normalizer": self.normalizer.payload(),
            "ppo_config": self.config.to_dict(),
            "feature_schema": feature_schema_payload(),
            "observation_version": OBSERVATION_VERSION,
            "env_version": ENV_VERSION,
            "versions": {"torch": torch.__version__, "ppo": PPO_VERSION},
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PPOPolicy":
        payload = torch.load(Path(path), weights_only=False)
        if payload.get("checkpoint_version") != PPO_CHECKPOINT_VERSION:
            raise PPOError(
                f"checkpoint version {payload.get('checkpoint_version')!r} is not "
                f"{PPO_CHECKPOINT_VERSION!r}"
            )
        config = PPOConfig.from_payload(payload["ppo_config"])
        model = ActorCritic(payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        normalizer = Normalizer.from_payload(payload["normalizer"])
        return cls(model, normalizer, config)


def compute_gae(
    transitions: Sequence[Transition],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Generalized advantage estimates for one episode (or a truncated tail)."""
    advantages = [0.0] * len(transitions)
    returns = [0.0] * len(transitions)
    last = 0.0
    next_value = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        delta = transition.reward + gamma * next_value * (0.0 if transition.done else 1.0) - transition.value
        last = delta + gamma * gae_lambda * (0.0 if transition.done else 1.0) * last
        advantages[index] = last
        returns[index] = last + transition.value
        next_value = transition.value
    return advantages, returns


def _train_transition_batch(
    policy: PPOPolicy,
    transitions: Sequence[Transition],
    advantages: Sequence[float],
    returns: Sequence[float],
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    *,
    rng: random.Random,
) -> dict[str, Any]:
    """One PPO update over a rollout chunk."""
    assert config.epochs > 0
    order = list(range(len(transitions)))
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0,
             "approx_kl": 0.0, "steps": 0}
    for epoch in range(config.epochs):
        rng.shuffle(order)
        for position in order:
            transition = transitions[position]
            policy.model.train()
            logits, edge_ids, value, _ = policy._logits(transition.observation, grad=True)
            distribution = torch.distributions.Categorical(logits=logits)
            index = min(transition.action_index, len(edge_ids))
            new_logprob = distribution.log_prob(torch.tensor(index))
            entropy = distribution.entropy()
            ratio = torch.exp(new_logprob - transition.logprob)
            advantage = float(advantages[position])
            if config.normalize_advantages and len(advantages) > 1:
                mean = statistics.fmean(advantages)
                std = statistics.pstdev(advantages) or 1.0
                advantage = (advantage - mean) / std
            surrogate = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage
            policy_loss = -torch.min(surrogate, clipped)
            value_loss = (torch.tensor(float(returns[position])) - value.reshape(())) ** 2
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.model.parameters(), config.max_grad_norm)
            optimizer.step()
            stats["policy_loss"] += float(policy_loss.detach().item())
            stats["value_loss"] += float(value_loss.detach().item())
            stats["entropy"] += float(entropy.detach().item())
            stats["clip_fraction"] += float((abs(float(ratio.detach().item()) - 1.0) > config.clip_ratio))
            stats["approx_kl"] += float((transition.logprob - new_logprob.detach().item()))
            stats["steps"] += 1
    for key in ("policy_loss", "value_loss", "entropy", "clip_fraction", "approx_kl"):
        stats[key] = round(stats[key] / max(1, stats["steps"]), 6)
    return stats


def _rollout_episode(
    env: SwarmEnvironment,
    policy: PPOPolicy,
    *,
    max_steps: int,
) -> tuple[list[Transition], dict[str, Any]]:
    observation, _ = env.reset()
    policy.reset()
    transitions: list[Transition] = []
    total_reward = 0.0
    for _ in range(max_steps):
        actions, transition = policy.sample(observation)
        next_observation, reward, terminated, truncated, info = env.step(actions)
        transition.reward = float(reward)
        transition.done = bool(terminated or truncated)
        transitions.append(transition)
        total_reward += float(reward)
        observation = next_observation
        if terminated or truncated:
            break
    metrics = env.metrics()
    metrics["episode_return"] = round(total_reward, 6)
    metrics["train_steps"] = len(transitions)
    return transitions, metrics


def _evaluate_specs(policy: PPOPolicy, specs: Sequence[EpisodeSpec], *, max_steps: int) -> dict[str, Any]:
    episodes = []
    for spec in specs:
        env = SwarmEnvironment(spec)
        outcomes = env.rollout(policy)
        metrics = dict(outcomes.metrics)
        metrics["episode_id"] = spec.episode_id
        episodes.append(metrics)
    if not episodes:
        return {"episodes": 0, "mean_coverage": None, "mean_reward": None}
    return {
        "episodes": len(episodes),
        "mean_coverage": round(statistics.fmean(float(row.get("coverage", 0.0)) for row in episodes), 6),
        "mean_reward": round(statistics.fmean(float(row.get("mean_reward", 0.0)) for row in episodes), 6),
        "episodes_detail": episodes,
    }


def train_ppo(
    config: PPOConfig,
    output_dir: str | Path,
    *,
    train_specs: Sequence[EpisodeSpec] | None = None,
    validation_specs: Sequence[EpisodeSpec] | None = None,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train a PPO policy on specs (or a campaign's train/validation splits)."""
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if train_specs is None:
        if not config.dataset_root:
            raise PPOError("train_ppo needs train_specs or a dataset_root")
        from .dataset import EpisodeDataset

        dataset = EpisodeDataset(config.dataset_root)
        train_specs = [
            EpisodeSpec.from_payload(row["spec"])
            for row in dataset.episodes("train")[: config.max_train_episodes]
        ]
        validation_specs = validation_specs or [
            EpisodeSpec.from_payload(row["spec"])
            for row in dataset.episodes("validation")[: config.validation_episodes]
        ]
    specs = list(train_specs)
    if not specs:
        raise PPOError("no training episodes")
    validation_episodes = list(
        validation_specs or specs[: max(1, config.validation_episodes)]
    )

    normalizer = Normalizer.fit(
        [build_batch([GraphSample(observation=SwarmEnvironment(spec).reset()[0])]) for spec in specs]
    )
    model = ActorCritic(config.model)
    if initial_checkpoint:
        payload = torch.load(Path(initial_checkpoint), weights_only=False)
        state = payload.get("state_dict", payload)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            model.backbone.load_state_dict(state)
        if "normalizer" in payload:
            normalizer = Normalizer.from_payload(payload["normalizer"])
    policy = PPOPolicy(model, normalizer, config, deterministic=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    history: list[dict[str, Any]] = []
    best = -math.inf
    best_path = output / "checkpoint-best.pt"
    last_path = output / "checkpoint-last.pt"
    rng = random.Random(config.seed)
    started = time.perf_counter()

    for update in range(config.updates):
        rollout_transitions: list[Transition] = []
        episode_metrics: list[dict[str, Any]] = []
        for episode_index in range(config.episodes_per_update):
            spec = specs[(update * config.episodes_per_update + episode_index) % len(specs)]
            env = SwarmEnvironment(spec)
            transitions, metrics = _rollout_episode(env, policy, max_steps=config.max_episode_steps)
            rollout_transitions.extend(transitions)
            episode_metrics.append(metrics)
        flat_advantages: list[float] = []
        flat_returns: list[float] = []
        start = 0
        for metrics in episode_metrics:
            count = int(metrics["train_steps"])
            chunk = rollout_transitions[start : start + count]
            advantages, returns = compute_gae(
                chunk, gamma=config.gamma, gae_lambda=config.gae_lambda,
            )
            flat_advantages.extend(advantages)
            flat_returns.extend(returns)
            start += count
        update_stats = _train_transition_batch(
            policy, rollout_transitions, flat_advantages, flat_returns,
            optimizer, config, rng=rng,
        )
        entry = {
            "update": update,
            "steps": len(rollout_transitions),
            "episode_return": round(statistics.fmean(row["episode_return"] for row in episode_metrics), 6),
            "coverage": round(statistics.fmean(float(row.get("coverage", 0.0)) for row in episode_metrics), 6),
            **update_stats,
        }
        if config.evaluate_every and (update % config.evaluate_every == 0 or update == config.updates - 1):
            validation_report = _evaluate_specs(
                policy, validation_episodes, max_steps=config.max_episode_steps
            )
            entry["validation"] = validation_report
            score = float(validation_report.get("mean_coverage") or 0.0)
            if score > best:
                best = score
                policy.save(best_path)
        history.append(entry)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        policy.save(last_path)

    manifest = {
        "version": PPO_VERSION,
        "run_id": config.run_id,
        "config": config.to_dict(),
        "parameters": model_parameter_count(model),
        "feature_schema": feature_schema_payload(),
        "history": history,
        "best_validation_coverage": None if best == -math.inf else round(best, 6),
        "checkpoints": {
            "last": str(last_path),
            "best": str(best_path) if best_path.exists() else None,
        },
        "wall_time_s": round(time.perf_counter() - started, 3),
        "selection_note": "checkpoints are selected on validation coverage only",
    }
    (output / "ppo-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return manifest


def ppo_report(output_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(output_dir) / "ppo-manifest.json").read_text(encoding="utf-8"))


__all__ = [
    "PPO_VERSION",
    "PPO_CHECKPOINT_VERSION",
    "PPOError",
    "PPOConfig",
    "ActorCritic",
    "Transition",
    "PPOPolicy",
    "compute_gae",
    "train_ppo",
    "ppo_report",
]
