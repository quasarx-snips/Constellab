"""Feature normalization fitted on the **training split only**.

A learned policy must not learn from statistics of its own test set: the mean
and standard deviation of every feature column are computed from the training
decision samples, stored inside the checkpoint, and applied unchanged to
validation, test and live observations. Columns whose training variance is zero
(or whose mask is entirely missing) keep ``scale = 1`` so they are passed
through untouched instead of being amplified into noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import EDGE_FEATURES, GLOBAL_FEATURES, SATELLITE_FEATURES, TASK_FEATURES
from .batch import StepBatch
from .param_encoder import TOKEN_NUMERIC_FEATURES

NORMALIZED_SCOPES: tuple[str, ...] = ("satellite", "task", "edge", "global", "token")
SCOPE_WIDTHS: dict[str, int] = {
    "satellite": len(SATELLITE_FEATURES),
    "task": len(TASK_FEATURES),
    "edge": len(EDGE_FEATURES),
    "global": len(GLOBAL_FEATURES),
    "token": len(TOKEN_NUMERIC_FEATURES),
}
SCOPE_NAMES: dict[str, tuple[str, ...]] = {
    "satellite": SATELLITE_FEATURES,
    "task": TASK_FEATURES,
    "edge": EDGE_FEATURES,
    "global": GLOBAL_FEATURES,
    "token": TOKEN_NUMERIC_FEATURES,
}


@dataclass
class Normalizer:
    """Column-wise mean/scale per feature scope, fitted on training rows."""

    mean: dict[str, list[float]] = field(default_factory=dict)
    scale: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, batches: Iterable[StepBatch]) -> "Normalizer":
        totals = {scope: [0.0] * SCOPE_WIDTHS[scope] for scope in NORMALIZED_SCOPES}
        squares = {scope: [0.0] * SCOPE_WIDTHS[scope] for scope in NORMALIZED_SCOPES}
        counts = {scope: 0 for scope in NORMALIZED_SCOPES}
        for batch in batches:
            for scope, rows in _rows_of(batch):
                width = SCOPE_WIDTHS[scope]
                for row in rows:
                    counts[scope] += 1
                    for index in range(min(width, len(row))):
                        value = float(row[index])
                        totals[scope][index] += value
                        squares[scope][index] += value * value
        mean: dict[str, list[float]] = {}
        scale: dict[str, list[float]] = {}
        for scope in NORMALIZED_SCOPES:
            width = SCOPE_WIDTHS[scope]
            if counts[scope] == 0:
                mean[scope] = [0.0] * width
                scale[scope] = [1.0] * width
                continue
            scope_mean: list[float] = []
            scope_scale: list[float] = []
            for index in range(width):
                average = totals[scope][index] / counts[scope]
                variance = max(0.0, squares[scope][index] / counts[scope] - average * average)
                deviation = variance ** 0.5
                scope_mean.append(round(average, 6))
                scope_scale.append(round(deviation, 6) if deviation > 1e-9 else 1.0)
            mean[scope] = scope_mean
            scale[scope] = scope_scale
        return cls(mean=mean, scale=scale, counts=counts)

    # --- application ------------------------------------------------------------------
    def apply_rows(self, scope: str, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        width = SCOPE_WIDTHS[scope]
        mean = self.mean.get(scope) or [0.0] * width
        scale = self.scale.get(scope) or [1.0] * width
        normalized: list[list[float]] = []
        for row in rows:
            normalized.append([
                round((float(row[index]) - mean[index]) / (scale[index] or 1.0), 6)
                if index < len(row) else 0.0
                for index in range(width)
            ])
        return normalized

    def apply(self, batch: StepBatch) -> StepBatch:
        """Return a normalized copy of a batch (the batch itself is untouched)."""
        return StepBatch(
            graph_count=batch.graph_count,
            graph_ids=list(batch.graph_ids),
            node_features=self.apply_rows("satellite", batch.node_features),
            node_available=[list(row) for row in batch.node_available],
            node_graph=list(batch.node_graph),
            node_is_actionable=list(batch.node_is_actionable),
            task_features=self.apply_rows("task", batch.task_features),
            task_graph=list(batch.task_graph),
            assign_src=list(batch.assign_src),
            assign_dst=list(batch.assign_dst),
            assign_features=self.apply_rows("edge", batch.assign_features),
            assign_graph=list(batch.assign_graph),
            contact_src=list(batch.contact_src),
            contact_dst=list(batch.contact_dst),
            graph_features=self.apply_rows("global", batch.graph_features),
            token_numeric=self.apply_rows("token", batch.token_numeric),
            token_categorical=[list(row) for row in batch.token_categorical],
            token_graph=list(batch.token_graph),
            targets=dict(batch.targets),
            abstain=list(batch.abstain),
            candidates={key: list(value) for key, value in batch.candidates.items()},
            sample_weights=list(batch.sample_weights),
            families=list(batch.families),
            parameter_hashes=list(batch.parameter_hashes),
            compatibility=dict(batch.compatibility),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "mean": {scope: list(values) for scope, values in self.mean.items()},
            "scale": {scope: list(values) for scope, values in self.scale.items()},
            "counts": dict(self.counts),
            "features": {scope: list(SCOPE_NAMES[scope]) for scope in NORMALIZED_SCOPES},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Normalizer":
        return cls(
            mean={scope: list(values) for scope, values in dict(payload.get("mean", {})).items()},
            scale={scope: list(values) for scope, values in dict(payload.get("scale", {})).items()},
            counts={scope: int(value) for scope, value in dict(payload.get("counts", {})).items()},
        )

    def describe(self) -> dict[str, Any]:
        return {
            scope: {
                "rows": self.counts.get(scope, 0),
                "mean": self.mean.get(scope),
                "scale": self.scale.get(scope),
            }
            for scope in NORMALIZED_SCOPES
        }


def _rows_of(batch: StepBatch) -> list[tuple[str, list[list[float]]]]:
    return [
        ("satellite", batch.node_features),
        ("task", batch.task_features),
        ("edge", batch.assign_features),
        ("global", batch.graph_features),
        ("token", batch.token_numeric),
    ]


__all__ = [
    "NORMALIZED_SCOPES",
    "SCOPE_WIDTHS",
    "SCOPE_NAMES",
    "Normalizer",
]
