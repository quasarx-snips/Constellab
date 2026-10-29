"""Batch assembly for the learned policies (torch-free, fully testable).

A batch is a *set of graphs*, not a padded tensor: satellites, tasks and edges
from several episodes are concatenated and address each other through index
arrays (``assign_src``/``assign_dst``, ``contact_src``/``contact_dst``) plus a
per-entity graph id. That is why the same weights run on a 3-satellite and a
24-satellite episode, and why adding a task type never changes a tensor shape.

Two target conventions meet here:

* the environment speaks in *observation indices* (``satellite_ids[i]``), and
* the model scores *assign edges*, so a chosen task becomes an edge id.

``StepBatch.targets`` therefore maps a global satellite index to a global assign
edge id, and ``StepBatch.abstain`` marks the satellites whose target is "wait"
(either the expert abstained, or no candidate existed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import EDGE_FEATURES, GLOBAL_FEATURES, SATELLITE_FEATURES, TASK_FEATURES
from .param_encoder import TOKEN_CATEGORICAL_FEATURES, TOKEN_NUMERIC_FEATURES, build_parameter_context


class BatchError(RuntimeError):
    """A batch could not be assembled from the given samples."""


@dataclass
class GraphSample:
    """One decision point of one episode, ready to become part of a batch."""

    observation: dict[str, Any]
    targets: list[list[int]] = field(default_factory=list)
    episode_id: str = ""
    family: str = ""
    step_index: int = 0
    context: dict[str, Any] | None = None
    weight: float = 1.0
    # Episode-level metadata used for curriculum slicing and balanced sampling
    # (fleet size, target count, fault count, partitions, condition class).
    attributes: dict[str, Any] = field(default_factory=dict)

    def candidate_pairs(self) -> list[tuple[int, int]]:
        return [(int(pair[0]), int(pair[1])) for pair in self.observation["mask"]["assign"]]


@dataclass
class StepBatch:
    """Flattened tensors-as-lists for one training step (see module docstring)."""

    graph_count: int = 0
    graph_ids: list[str] = field(default_factory=list)
    node_features: list[list[float]] = field(default_factory=list)
    node_available: list[list[float]] = field(default_factory=list)
    node_graph: list[int] = field(default_factory=list)
    node_is_actionable: list[int] = field(default_factory=list)
    task_features: list[list[float]] = field(default_factory=list)
    task_graph: list[int] = field(default_factory=list)
    assign_src: list[int] = field(default_factory=list)
    assign_dst: list[int] = field(default_factory=list)
    assign_features: list[list[float]] = field(default_factory=list)
    assign_graph: list[int] = field(default_factory=list)
    contact_src: list[int] = field(default_factory=list)
    contact_dst: list[int] = field(default_factory=list)
    graph_features: list[list[float]] = field(default_factory=list)
    token_numeric: list[list[float]] = field(default_factory=list)
    token_categorical: list[list[int]] = field(default_factory=list)
    token_graph: list[int] = field(default_factory=list)
    # Targets: global satellite index -> global assign edge id (chosen task).
    targets: dict[int, int] = field(default_factory=dict)
    abstain: list[int] = field(default_factory=list)
    candidates: dict[int, list[int]] = field(default_factory=dict)
    sample_weights: list[float] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    parameter_hashes: list[str] = field(default_factory=list)
    compatibility: dict[str, Any] = field(default_factory=dict)

    @property
    def edge_count(self) -> int:
        return len(self.assign_src)

    @property
    def node_count(self) -> int:
        return len(self.node_features)

    @property
    def token_count(self) -> int:
        return len(self.token_numeric)

    def supervised_satellites(self) -> list[int]:
        """Actionable satellites that carry a target (edge id or abstain)."""
        abstain = set(self.abstain)
        return [
            node for node in sorted(self.candidates)
            if node in self.targets or node in abstain
        ]

    def to_summary(self) -> dict[str, Any]:
        return {
            "graphs": self.graph_count,
            "satellites": self.node_count,
            "tasks": len(self.task_features),
            "assign_edges": self.edge_count,
            "contact_edges": len(self.contact_src),
            "tokens": self.token_count,
            "supervised": len(self.supervised_satellites()),
            "families": sorted(set(self.families)),
        }


def build_batch(
    samples: Sequence[GraphSample],
    *,
    parameter_context: Mapping[str, Mapping[str, Any]] | None = None,
) -> StepBatch:
    """Concatenate decision samples into one graph batch.

    ``parameter_context`` optionally supplies the Day 40 token context per
    episode (keyed by episode id); when a sample carries its own ``context`` that
    wins, and when neither exists the graph gets an empty token set.
    """
    batch = StepBatch()
    node_offset = 0
    task_offset = 0
    edge_offset = 0
    token_offset = 0
    contexts: dict[str, Mapping[str, Any]] = dict(parameter_context or {})
    incompatible: list[dict[str, Any]] = []

    for graph_index, sample in enumerate(samples):
        observation = sample.observation
        if observation.get("satellites") is None or observation.get("tasks") is None:
            raise BatchError("a sample observation must carry satellites and tasks")
        batch.graph_ids.append(sample.episode_id or f"graph-{graph_index}")
        batch.families.append(sample.family)
        batch.sample_weights.append(float(sample.weight))
        batch.graph_features.append(_pad(observation.get("global"), len(GLOBAL_FEATURES), 0.0))

        satellites = observation["satellites"]
        tasks = observation["tasks"]
        for row in satellites:
            features = _pad(row.get("features"), len(SATELLITE_FEATURES), 0.0)
            available = _pad(row.get("available"), len(SATELLITE_FEATURES), 1.0)
            batch.node_features.append(features)
            batch.node_available.append(available)
            batch.node_graph.append(graph_index)
            batch.node_is_actionable.append(0)
        for row in tasks:
            batch.task_features.append(_pad(row.get("features"), len(TASK_FEATURES), 0.0))
            batch.task_graph.append(graph_index)

        # --- assign edges (satellite -> task) --------------------------------------
        edge_of: dict[tuple[int, int], int] = {}
        for edge in observation["edges"]["assign"]:
            local_satellite, local_task = int(edge[0]), int(edge[1])
            features = _pad(edge[2:], len(EDGE_FEATURES), 0.0)
            global_satellite = node_offset + local_satellite
            global_task = task_offset + local_task
            batch.assign_src.append(global_satellite)
            batch.assign_dst.append(global_task)
            batch.assign_features.append(features)
            batch.assign_graph.append(graph_index)
            edge_of[(local_satellite, local_task)] = edge_offset
            edge_offset += 1

        # The mask decides which pairs are *actionable*; a satellite with no
        # masked pair is not supervised at all (it has no legal action).
        actionable_satellites: set[int] = set()
        for pair in observation["mask"]["assign"]:
            local_satellite, local_task = int(pair[0]), int(pair[1])
            edge_id = edge_of.get((local_satellite, local_task))
            if edge_id is None:
                continue
            global_satellite = node_offset + local_satellite
            actionable_satellites.add(global_satellite)
            batch.candidates.setdefault(global_satellite, []).append(edge_id)
        for global_satellite in sorted(actionable_satellites):
            batch.node_is_actionable[global_satellite] = 1
            # Waiting is always legal, so every actionable satellite is a
            # supervised decision point with an explicit abstain class.
            batch.abstain.append(global_satellite)
        for node in list(batch.candidates):
            batch.candidates[node] = sorted(set(batch.candidates[node]))

        for pair in sample.targets:
            local_satellite, local_task = int(pair[0]), int(pair[1])
            edge_id = edge_of.get((local_satellite, local_task))
            if edge_id is None:
                continue
            global_satellite = node_offset + local_satellite
            batch.targets[global_satellite] = edge_id
            if global_satellite in batch.candidates:
                batch.abstain.remove(global_satellite)

        # --- contact edges ----------------------------------------------------------
        for edge in observation["edges"]["contact"]:
            batch.contact_src.append(node_offset + int(edge[0]))
            batch.contact_dst.append(node_offset + int(edge[1]))

        # --- parameter tokens -------------------------------------------------------
        context = sample.context or contexts.get(sample.episode_id) or _empty_context()
        report = dict(context.get("compatibility", {}))
        batch.parameter_hashes.append(str(context.get("schema_hash", "")))
        if not report.get("compatible", True):
            incompatible.append({"episode_id": sample.episode_id, **report})
        for token in context.get("tokens", []):
            batch.token_numeric.append(
                [float(token["normalized"]), float(token["log_scale"]), float(token["bound_low"]),
                 float(token["bound_span"]), float(token["share"]), float(token["extension"]),
                 0.0 if token["missing"] else 1.0]
            )
            batch.token_categorical.append([
                int(token["type"].__class__ and _type_code(token["type"])),
                int(token["unit_kind_code"]),
                int(token["namespace_code"]),
            ])
            batch.token_graph.append(graph_index)
            token_offset += 1
        node_offset += len(satellites)
        task_offset += len(tasks)
        batch.graph_count = graph_index + 1

    if incompatible:
        batch.compatibility = {
            "compatible": False,
            "graphs": incompatible,
        }
    else:
        batch.compatibility = {"compatible": True, "graphs": []}
    return batch


def samples_from_dataset(dataset: Any, split: str, *, limit: int | None = None) -> list[GraphSample]:
    """Turn stored decision samples into :class:`GraphSample` objects."""
    samples: list[GraphSample] = []
    values_by_episode: dict[str, dict[str, Any]] = {}
    for episode in dataset.episodes(split):
        values_by_episode[episode["episode_id"]] = {
            **dict(episode.get("values") or {}),
            "family": episode.get("family"),
            "condition_class": episode.get("condition_class"),
        }
    for entry in dataset.decision_samples(split, limit=limit):
        samples.append(
            GraphSample(
                observation=entry["observation"],
                targets=[list(pair) for pair in entry["targets"]],
                episode_id=entry["episode_id"],
                family=entry.get("family") or "",
                step_index=int(entry["observation"].get("step_index", 0)),
                context=entry.get("context") or None,
                attributes=dict(values_by_episode.get(entry["episode_id"], {})),
            )
        )
    return samples


def empty_parameter_context() -> dict[str, Any]:
    from constellab_ml.param_encoder import empty_context

    return empty_context()


def _empty_context() -> dict[str, Any]:
    return empty_parameter_context()


def token_feature_lengths() -> tuple[int, int]:
    return len(TOKEN_NUMERIC_FEATURES), len(TOKEN_CATEGORICAL_FEATURES)


def _type_code(name: str) -> int:
    from .param_encoder import TYPE_CODES

    return TYPE_CODES.index(name) if name in TYPE_CODES else TYPE_CODES.index("unknown")


def _pad(values: Any, length: int, fill: float) -> list[float]:
    row = [float(value) for value in (values or [])][:length]
    if len(row) < length:
        row.extend([fill] * (length - len(row)))
    return row


__all__ = [
    "BatchError",
    "GraphSample",
    "StepBatch",
    "build_batch",
    "samples_from_dataset",
    "empty_parameter_context",
    "token_feature_lengths",
]
