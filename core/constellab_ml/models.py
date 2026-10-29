"""Days 39/40/41/42: the learned policy family, behind one interface.

Every family consumes the same :class:`~constellab_ml.batch.StepBatch` and
returns the same :class:`PolicyOutput`: a score per *assign edge* plus an
abstain score per actionable satellite. The families differ in how much
structure they use, which is exactly what a sweep needs to compare:

* ``mlp`` — a per-satellite and per-task MLP with a bilinear pair score; no set
  aggregation, no message passing (Day 39 baseline),
* ``deepsets`` — the same encoders plus permutation-invariant masked pooling of
  the satellite and task sets (Day 39, permutation-aware baseline),
* ``gnn`` — shared message-passing layers over the *contact* relation between
  satellites and the *assign* relation between satellites and tasks
  (Day 41),
* ``gnn_gru`` — ``gnn`` plus a per-satellite GRU hidden state that carries
  temporal context across steps of one episode and is reset per episode
  (Day 42),
* ``gnn_attention`` — the same interface with a single-head temporal
  self-attention over the last few satellite embeddings instead of a GRU
  (Day 42's optional variant; promoted only when a sweep shows a benefit).

The parameter-token encoder (Day 40) is shared by every family: it embeds the
categorical token features (type, unit family, namespace), projects the numeric
ones and pools the variable-length token set with a learned masked-attention
query. Adding registered fields changes the *number* of tokens, never a weight
shape, so one checkpoint can serve configurations with different declared
fields — and an unrecognised field is reported by the compatibility layer
rather than silently absorbed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from . import EDGE_FEATURES, GLOBAL_FEATURES, SATELLITE_FEATURES, TASK_FEATURES
from .batch import StepBatch, token_feature_lengths
from .param_encoder import TOKEN_CATEGORICAL_FEATURES, TOKEN_NUMERIC_FEATURES

MODEL_FAMILIES: tuple[str, ...] = ("mlp", "deepsets", "gnn", "gnn_gru", "gnn_attention")
TEMPORAL_FAMILIES: tuple[str, ...] = ("gnn_gru", "gnn_attention")


class ModelError(RuntimeError):
    """A model configuration is invalid or a batch does not fit the contract."""


def default_model_config(model: str = "gnn") -> dict[str, Any]:
    if model not in MODEL_FAMILIES:
        raise ModelError(f"unknown model family {model!r}; known: {list(MODEL_FAMILIES)}")
    return {
        "model": model,
        "hidden": 32,
        "layers": 2,
        "dropout": 0.0,
        "namespace_count": 8,
        "unit_kind_count": 8,
        "type_count": 5,
        "temporal_window": 4,
    }


@dataclass
class PolicyOutput:
    """Model output: a score per assign edge and a per-node abstain score."""

    edge_logits: torch.Tensor
    abstain_logits: torch.Tensor
    node_embeddings: torch.Tensor
    state: dict[str, Any] = field(default_factory=dict)

    def detached(self) -> "PolicyOutput":
        return PolicyOutput(
            edge_logits=self.edge_logits.detach(),
            abstain_logits=self.abstain_logits.detach(),
            node_embeddings=self.node_embeddings.detach(),
            state={key: value.detach() for key, value in self.state.items()},
        )


class ParamTokenEncoder(nn.Module):
    """Metadata-aware encoder for a variable number of parameter tokens."""

    def __init__(
        self,
        out_dim: int,
        *,
        namespace_count: int = 8,
        unit_kind_count: int = 8,
        type_count: int = 5,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        numeric_dim, categorical_dim = token_feature_lengths()
        if categorical_dim != len(TOKEN_CATEGORICAL_FEATURES):
            raise ModelError("categorical token schema drifted from the encoder")
        width = hidden or max(16, out_dim)
        self.numeric = nn.Linear(numeric_dim, width)
        self.type_embedding = nn.Embedding(type_count + 1, width)
        self.unit_embedding = nn.Embedding(unit_kind_count + 1, width)
        self.namespace_embedding = nn.Embedding(namespace_count + 1, width)
        self.query = nn.Parameter(torch.randn(width) / math.sqrt(width))
        self.projection = nn.Linear(width, out_dim)
        self.missing_embedding = nn.Parameter(torch.zeros(width))
        self.activation = nn.ReLU()

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        token_graph: torch.Tensor,
        graph_count: int,
    ) -> torch.Tensor:
        """Pooled context per graph: ``[graph_count, out_dim]`` (zeros when empty)."""
        width = self.numeric.out_features
        context = torch.zeros(graph_count, self.projection.out_features, dtype=self.query.dtype)
        if numeric.numel() == 0:
            return context
        numeric = numeric.to(self.query.dtype)
        categorical = categorical.to(torch.long)
        filled = numeric[:, -1].unsqueeze(-1)
        token = (
            self.numeric(numeric)
            + self.type_embedding(_clamp_index(categorical[:, 0], self.type_embedding))
            + self.unit_embedding(_clamp_index(categorical[:, 1], self.unit_embedding))
            + self.namespace_embedding(_clamp_index(categorical[:, 2], self.namespace_embedding))
        )
        # A missing field contributes its learned "unknown" offset, never a zero
        # that pretends the value was measured.
        token = token + (1.0 - filled) * self.missing_embedding.unsqueeze(0)
        weights = (token * self.query).sum(-1) / math.sqrt(width)
        pooled = torch.zeros(graph_count, width, dtype=self.query.dtype)
        for graph_index in range(graph_count):
            selected = (token_graph == graph_index).nonzero(as_tuple=True)[0]
            if selected.numel() == 0:
                continue
            attention = torch.softmax(weights[selected], dim=0)
            pooled[graph_index] = (attention.unsqueeze(-1) * token[selected]).sum(dim=0)
        return self.projection(self.activation(pooled))


def _clamp_index(codes: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
    return codes.clamp(min=0, max=embedding.num_embeddings - 1)


def _matrix(rows: Sequence[Sequence[float]], width: int) -> torch.Tensor:
    """A float tensor of a fixed column count, even when there are no rows."""
    if not rows:
        return torch.zeros(0, width, dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


class _BasePolicy(nn.Module):
    family = "base"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.hidden = int(config.get("hidden", 32))
        self.dropout = float(config.get("dropout", 0.0))
        self.satellite_encoder = nn.Sequential(
            nn.Linear(len(SATELLITE_FEATURES), self.hidden), nn.ReLU()
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(len(TASK_FEATURES), self.hidden), nn.ReLU()
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(len(EDGE_FEATURES), self.hidden), nn.ReLU()
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(len(GLOBAL_FEATURES), self.hidden), nn.ReLU()
        )
        self.parameter_encoder = ParamTokenEncoder(
            self.hidden,
            namespace_count=int(config.get("namespace_count", 8)),
            unit_kind_count=int(config.get("unit_kind_count", 8)),
            type_count=int(config.get("type_count", 5)),
        )
        self.score_head = nn.Sequential(
            nn.Linear(self.hidden * 4, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        # The abstain class sees the satellite plus its (graph, parameter and
        # set) context: two hidden blocks, exactly like the pair head minus the
        # task side.
        self.abstain_head = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        if self.dropout > 0.0:
            self.drop = nn.Dropout(self.dropout)
        else:
            self.drop = nn.Identity()

    # --- helpers ----------------------------------------------------------------------
    def _pool(self, values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
        """Masked mean over entities of each graph (permutation invariant)."""
        pooled = torch.zeros(count, values.shape[-1], dtype=values.dtype)
        if values.numel() == 0:
            return pooled
        pooled.index_add_(0, index, values)
        counts = torch.zeros(count, dtype=values.dtype)
        counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
        return pooled / counts.clamp(min=1.0).unsqueeze(-1)

    def _encode_inputs(self, batch: StepBatch) -> tuple[torch.Tensor, torch.Tensor]:
        node = self.satellite_encoder(_matrix(batch.node_features, len(SATELLITE_FEATURES)))
        task = self.task_encoder(_matrix(batch.task_features, len(TASK_FEATURES)))
        return node, task

    def _context(self, batch: StepBatch) -> torch.Tensor:
        numeric = torch.tensor(batch.token_numeric, dtype=torch.float32) if batch.token_numeric else torch.zeros(0, len(TOKEN_NUMERIC_FEATURES))
        categorical = (
            torch.tensor(batch.token_categorical, dtype=torch.long)
            if batch.token_categorical
            else torch.zeros(0, len(TOKEN_CATEGORICAL_FEATURES), dtype=torch.long)
        )
        token_graph = torch.tensor(batch.token_graph, dtype=torch.long) if batch.token_graph else torch.zeros(0, dtype=torch.long)
        return self.parameter_encoder(numeric, categorical, token_graph, max(1, batch.graph_count))

    def _graph_broadcast(self, batch: StepBatch, node_graph: torch.Tensor) -> torch.Tensor:
        encoded = self.graph_encoder(_matrix(batch.graph_features, len(GLOBAL_FEATURES)))
        if encoded.numel() == 0:
            return torch.zeros(len(batch.node_graph), self.hidden)
        return encoded[node_graph] * self.hidden ** 0  # broadcast rows per node

    def _context_broadcast(self, batch: StepBatch, node_graph: torch.Tensor) -> torch.Tensor:
        context = self._context(batch)
        return context[node_graph] if context.numel() else torch.zeros(len(batch.node_graph), self.hidden)

    # --- interface --------------------------------------------------------------------
    def initial_state(self, batch: StepBatch, *, device: Any = None) -> dict[str, Any]:
        return {}

    def forward(self, batch: StepBatch, *, state: Mapping[str, Any] | None = None) -> PolicyOutput:
        raise NotImplementedError

    def _score(
        self,
        node: torch.Tensor,
        task: torch.Tensor,
        batch: StepBatch,
        node_context: torch.Tensor,
    ) -> PolicyOutput:
        graph_context = self._graph_broadcast(batch, torch.tensor(batch.node_graph, dtype=torch.long))
        context = self._context_broadcast(batch, torch.tensor(batch.node_graph, dtype=torch.long))
        src = torch.tensor(batch.assign_src, dtype=torch.long)
        dst = torch.tensor(batch.assign_dst, dtype=torch.long)
        edge_features = self.edge_encoder(_matrix(batch.assign_features, len(EDGE_FEATURES)))
        if len(batch.assign_src) == 0:
            edge_logits = torch.zeros(0)
        else:
            pair = torch.cat(
                [
                    node[src],
                    task[dst],
                    edge_features,
                    graph_context[src] + context[src] + node_context[src],
                ],
                dim=-1,
            )
            edge_logits = self.score_head(self.drop(pair)).squeeze(-1)
        head = torch.cat(
            [node, graph_context + context + node_context],
            dim=-1,
        ) if node.numel() else torch.zeros(0, self.hidden * 2)
        abstain_logits = (
            self.abstain_head(self.drop(head)).squeeze(-1) if node.numel() else torch.zeros(0)
        )
        return PolicyOutput(
            edge_logits=edge_logits,
            abstain_logits=abstain_logits,
            node_embeddings=node,
        )


class MLPPolicy(_BasePolicy):
    """Day 39 baseline: per-satellite and per-task encoders, bilinear pair score."""

    family = "mlp"

    def forward(self, batch: StepBatch, *, state: Mapping[str, Any] | None = None) -> PolicyOutput:
        del state
        node, task = self._encode_inputs(batch)
        return self._score(node, task, batch, torch.zeros_like(node))


class DeepSetsPolicy(_BasePolicy):
    """Day 39 baseline: permutation-aware masked pooling of both entity sets."""

    family = "deepsets"

    def forward(self, batch: StepBatch, *, state: Mapping[str, Any] | None = None) -> PolicyOutput:
        del state
        node, task = self._encode_inputs(batch)
        node_graph = torch.tensor(batch.node_graph, dtype=torch.long)
        task_graph = torch.tensor(batch.task_graph, dtype=torch.long)
        if node.numel() == 0:
            return self._score(node, task, batch, node)
        graph_count = max(1, batch.graph_count)
        # Both pools are permutation invariant; the task pool is broadcast back
        # onto the satellites of the same graph.
        satellite_context = self._pool(node, node_graph, graph_count)[node_graph]
        task_pool = (
            self._pool(task, task_graph, graph_count)
            if task.numel() else torch.zeros(graph_count, self.hidden)
        )
        return self._score(node, task, batch, satellite_context + task_pool[node_graph])


class MessagePassingLayer(nn.Module):
    """One shared message-passing step over two typed relations."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.satellite_message = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.assign_message = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU())
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        node: torch.Tensor,
        task: torch.Tensor,
        *,
        contact_src: torch.Tensor,
        contact_dst: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = node.shape[0]
        accum = torch.zeros_like(node)
        weight = torch.zeros(count, 1, dtype=node.dtype)
        if contact_src.numel():
            message = self.satellite_message(torch.cat([node[contact_src], node[contact_dst]], dim=-1))
            accum.index_add_(0, contact_src, message)
            weight.index_add_(0, contact_src, torch.ones(message.shape[0], 1, dtype=node.dtype))
            accum.index_add_(0, contact_dst, message)
            weight.index_add_(0, contact_dst, torch.ones(message.shape[0], 1, dtype=node.dtype))
        if edge_src.numel():
            message = self.assign_message(
                torch.cat([node[edge_src], task[edge_dst], edge_features], dim=-1)
            )
            accum.index_add_(0, edge_src, message)
            weight.index_add_(0, edge_src, torch.ones(message.shape[0], 1, dtype=node.dtype))
        aggregated = accum / weight.clamp(min=1.0)
        node = self.norm(node + self.update(torch.cat([node, aggregated, aggregator_norm(accum)], dim=-1)))
        if edge_src.numel():
            task_accum = torch.zeros_like(task)
            task_weight = torch.zeros(task.shape[0], 1, dtype=task.dtype)
            task_message = self.satellite_message(torch.cat([node[edge_src], task[edge_dst]], dim=-1))
            task_accum.index_add_(0, edge_dst, task_message)
            task_weight.index_add_(0, edge_dst, torch.ones(task_message.shape[0], 1, dtype=task.dtype))
            aggregated_task = task_accum / task_weight.clamp(min=1.0)
            task = self.norm(
                task + self.update(torch.cat([task, aggregated_task, aggregated_task], dim=-1))
            )
        return node, task


def aggregator_norm(values: torch.Tensor) -> torch.Tensor:
    return torch.tanh(values)


class GNNPolicy(_BasePolicy):
    """Day 41: shared message passing over the contact and assign relations."""

    family = "gnn"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        layers = int(config.get("layers", 2))
        # One module applied ``layers`` times: the weights are shared across
        # hops, and a different fleet size only changes the index arrays.
        self.layer = MessagePassingLayer(self.hidden)
        self.message_layers = max(1, layers)
        self.edge_projection = nn.Linear(self.hidden, self.hidden)

    def _message_pass(self, node: torch.Tensor, task: torch.Tensor, batch: StepBatch) -> tuple[torch.Tensor, torch.Tensor]:
        contact_src = torch.tensor(batch.contact_src, dtype=torch.long)
        contact_dst = torch.tensor(batch.contact_dst, dtype=torch.long)
        edge_src = torch.tensor(batch.assign_src, dtype=torch.long)
        edge_dst = torch.tensor(batch.assign_dst, dtype=torch.long)
        edge_features = self.edge_projection(
            self.edge_encoder(_matrix(batch.assign_features, len(EDGE_FEATURES)))
        )
        for _ in range(self.message_layers):
            node, task = self.layer(
                node,
                task,
                contact_src=contact_src,
                contact_dst=contact_dst,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_features=edge_features,
            )
        return node, task

    def forward(self, batch: StepBatch, *, state: Mapping[str, Any] | None = None) -> PolicyOutput:
        del state
        node, task = self._encode_inputs(batch)
        node, task = self._message_pass(node, task, batch)
        return self._score(node, task, batch, torch.zeros_like(node))


class RecurrentGNNPolicy(GNNPolicy):
    """Day 42: the GNN plus per-satellite temporal memory.

    ``state`` holds one hidden vector per graph and satellite plus the episode id
    of each graph. Hidden state is reset whenever a graph's episode changes, so
    memory can never leak from one episode into the next.
    """

    family = "gnn_gru"
    temporal = "gru"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        window = int(config.get("temporal_window", 4))
        self.temporal_window = max(1, window)
        # Only the GRU family owns a cell, so an attention model carries no
        # parameter that never receives a gradient.
        self.cell = nn.GRUCell(self.hidden, self.hidden) if self.family == "gnn_gru" else None
        self.temporal_projection = nn.Linear(self.hidden, self.hidden)

    def initial_state(self, batch: StepBatch, *, device: Any = None) -> dict[str, Any]:
        return {
            "hidden": torch.zeros(batch.node_count, self.hidden, **( {"device": device} if device is not None else {})),
            "episodes": list(batch.graph_ids),
            "node_graph": list(batch.node_graph),
        }

    def forward(self, batch: StepBatch, *, state: Mapping[str, Any] | None = None) -> PolicyOutput:
        node, task = self._encode_inputs(batch)
        node_graph = torch.tensor(batch.node_graph, dtype=torch.long)
        if state is None:
            state = self.initial_state(batch)
        hidden = state["hidden"]
        resets = _episode_resets(state.get("episodes", []), batch.graph_ids)
        if hidden.shape[0] != batch.node_count or state.get("node_graph") != list(batch.node_graph):
            hidden = torch.zeros(batch.node_count, self.hidden)
            resets = []
        if resets:
            mask = torch.ones(batch.node_count, 1)
            for graph_index in resets:
                mask[node_graph == graph_index] = 0.0
            hidden = hidden * mask
        hidden = self.cell(node, hidden)
        node = node + self.temporal_projection(hidden) * 0.5
        node, task = self._message_pass(node, task, batch)
        output = self._score(node, task, batch, torch.zeros_like(node))
        output.state = {
            "hidden": hidden,
            "episodes": list(batch.graph_ids),
            "node_graph": list(batch.node_graph),
        }
        return output


class AttentionGNNPolicy(RecurrentGNNPolicy):
    """Day 42 optional variant: temporal self-attention instead of a GRU."""

    family = "gnn_attention"
    temporal = "attention"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.attention = nn.MultiheadAttention(self.hidden, 1, batch_first=True)
        self.memory: torch.Tensor | None = None

    def forward(self, batch: StepBatch, *, state: Mapping[str, Any] | None = None) -> PolicyOutput:
        node, task = self._encode_inputs(batch)
        if state is None:
            state = {"hidden": torch.zeros(batch.node_count, self.hidden), "episodes": list(batch.graph_ids),
                     "node_graph": list(batch.node_graph), "memory": None}
        history = state.get("memory")
        resets = _episode_resets(state.get("episodes", []), batch.graph_ids)
        current = node.unsqueeze(1)
        if history is None or state.get("memory_size") != batch.node_count:
            sequence = current
        else:
            sequence = torch.cat([history, current], dim=1)[:, -self.temporal_window:]
        if resets:
            node_graph = torch.tensor(batch.node_graph, dtype=torch.long)
            mask = torch.ones(batch.node_count, sequence.shape[1], 1)
            for graph_index in resets:
                mask[node_graph == graph_index] = 0.0
            sequence = sequence * mask
        attended, _ = self.attention(sequence, sequence, sequence)
        # Residual temporal update: the attended step refines the current node
        # embedding instead of replacing it.
        node = node + self.temporal_projection(attended[:, -1, :])
        node, task = self._message_pass(node, task, batch)
        output = self._score(node, task, batch, torch.zeros_like(node))
        output.state = {
            "hidden": node.detach(),
            "episodes": list(batch.graph_ids),
            "node_graph": list(batch.node_graph),
            "memory": sequence.detach(),
            "memory_size": batch.node_count,
        }
        return output


def _episode_resets(previous: Sequence[str], current: Sequence[str]) -> list[int]:
    """Graph indices whose episode changed (or appeared for the first time)."""
    resets: list[int] = []
    for index, episode_id in enumerate(current):
        if index >= len(previous) or previous[index] != episode_id:
            resets.append(index)
    return resets


MODEL_FACTORY: dict[str, type[_BasePolicy]] = {
    "mlp": MLPPolicy,
    "deepsets": DeepSetsPolicy,
    "gnn": GNNPolicy,
    "gnn_gru": RecurrentGNNPolicy,
    "gnn_attention": AttentionGNNPolicy,
}


def build_model(config: Mapping[str, Any] | str) -> _BasePolicy:
    """Registry factory: no ``if model == ...`` branching in callers."""
    if isinstance(config, str):
        config = default_model_config(config)
    resolved = dict(default_model_config(str(config.get("model", "gnn"))))
    resolved.update(dict(config))
    family = str(resolved["model"])
    if family not in MODEL_FACTORY:
        raise ModelError(f"unknown model family {family!r}; known: {list(MODEL_FAMILIES)}")
    return MODEL_FACTORY[family](resolved)


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "MODEL_FAMILIES",
    "TEMPORAL_FAMILIES",
    "ModelError",
    "PolicyOutput",
    "ParamTokenEncoder",
    "MLPPolicy",
    "DeepSetsPolicy",
    "GNNPolicy",
    "RecurrentGNNPolicy",
    "AttentionGNNPolicy",
    "MessagePassingLayer",
    "build_model",
    "default_model_config",
    "model_parameter_count",
]
