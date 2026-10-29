"""ConstellaLab ML package (Days 35-45).

Everything the learned policies need — the episode environment, the randomized
scenario pipeline, expert demonstrations, the PyTorch models, training, sweeps
and curriculum — lives behind this package. The simulation core
(``constellab_core``) stays the single owner of state: this package only wraps
it.

The feature schemas below are the *contract* between environment, datasets,
models and checkpoints. Names are stable; new columns may only be appended and
a change of meaning requires bumping ``OBSERVATION_VERSION``, because a
checkpoint is only valid for the schema hash it was trained on.
"""

from __future__ import annotations

ENV_VERSION = "swarm-env-1"
OBSERVATION_VERSION = "swarm-observation-1"

# Per-satellite node features. Missing values are carried by the parallel
# availability mask (never fabricated as 0).
SATELLITE_FEATURES: tuple[str, ...] = (
    "altitude_km_norm",
    "speed_km_s_norm",
    "available",
    "busy",
    "energy_fraction",
    "storage_fraction",
    "compute_duty",
    "footprint_km_norm",
    "visible_target_norm",
    "visible_task_norm",
    "time_to_window_norm",
    "contact_degree_norm",
    "claims_seen_norm",
    "max_urgency_norm",
)

# Per-task node features.
TASK_FEATURES: tuple[str, ...] = (
    "active",
    "urgency_norm",
    "deadline_norm",
    "age_norm",
    "min_confidence",
    "attempts_norm",
    "is_assigned",
    "is_in_progress",
    "is_open",
    "target_radius_km_norm",
    "is_change_detection",
)

# Bipartite satellite->task edge features (only feasible edges are emitted).
EDGE_FEATURES: tuple[str, ...] = (
    "peak_quality",
    "time_to_open_norm",
    "duration_norm",
    "resolution_norm",
)

# Global (graph-level) features.
GLOBAL_FEATURES: tuple[str, ...] = (
    "time_fraction",
    "open_task_fraction",
    "fleet_available_fraction",
    "contact_density",
    "fault_intensity",
    "deadline_pressure",
)

FEATURE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "satellite": SATELLITE_FEATURES,
    "task": TASK_FEATURES,
    "edge": EDGE_FEATURES,
    "global": GLOBAL_FEATURES,
}


def feature_schema_payload() -> dict[str, list[str]]:
    """Stable, JSON-safe view of the schema (hashed into every checkpoint)."""
    return {key: list(value) for key, value in FEATURE_SCHEMAS.items()}


def feature_index(scope: str, name: str) -> int:
    try:
        return FEATURE_SCHEMAS[scope].index(name)
    except KeyError as exc:  # pragma: no cover - programmer error
        raise KeyError(f"unknown feature scope {scope!r}") from exc
    except ValueError as exc:
        raise KeyError(f"unknown {scope} feature {name!r}") from exc


__all__ = [
    "ENV_VERSION",
    "OBSERVATION_VERSION",
    "SATELLITE_FEATURES",
    "TASK_FEATURES",
    "EDGE_FEATURES",
    "GLOBAL_FEATURES",
    "FEATURE_SCHEMAS",
    "feature_schema_payload",
    "feature_index",
]
