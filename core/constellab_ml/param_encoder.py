"""Day 40: metadata-aware encoding of the configurable parameter space.

A client configuration carries a variable number of declared variables, and the
set grows: the model must never learn "column 37 is the sensor angle". Every
configured value therefore becomes a **token** with its semantics attached:

* ``field`` — the registry key, plus a **stable field id** derived from the
  sorted registry keys and its schema hash (never from a display label),
* ``namespace`` — the owning component (sensor, power, storage, ...),
* ``unit``/``unit_kind`` — the declared unit and its quantity family,
* ``bounds`` — the declared ``[minimum, maximum]`` and whether the distribution
  is log-uniform, so a value can be interpreted relative to its own range,
* ``value`` — normalised into ``[0, 1]`` against those bounds,
* ``missing`` — 1 when a declared field carries no value (never a fake zero),
* ``extension`` — 1 when the field is *not* in the registry, i.e. a registered
  extension whose semantics the model was not trained on.

Tokensing a configuration is order-independent: tokens are sorted by
``(namespace, field, value)`` and the same configuration always produces the
same token list and hash. Fields held by a declared component that the build
does not know, values outside their declared bounds, and unknown owners are
surfaced as a **compatibility report** — a parsing success is never reported as
semantic understanding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from constellab_core import components as component_registry
from constellab_core import units
from constellab_core.parameters import ParameterRegistry, ParameterSpec, builtin_registry

PARAM_TOKEN_VERSION = "parameter-tokens-1"

# Numeric token features (float columns).
TOKEN_NUMERIC_FEATURES: tuple[str, ...] = (
    "normalized",     # value inside its own declared bounds
    "log_scale",      # 1 when the declared distribution is log-uniform
    "bound_low",      # declared minimum, scaled by the unit-family reference
    "bound_span",     # declared width, scaled by the unit-family reference
    "share",          # share of satellites carrying this value
    "extension",      # 1 when the field is not in the registry
    "filled",         # 0 for a declared-but-unset field
)

# Categorical token features (embedding ids).
TOKEN_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "type_code",
    "unit_kind_code",
    "namespace_code",
)

TYPE_CODES: tuple[str, ...] = ("number", "integer", "boolean", "string", "unknown")
UNKNOWN_TYPE_CODE = TYPE_CODES.index("unknown")


class ParameterContextError(ValueError):
    """A configuration could not be tokenised."""


@dataclass(frozen=True)
class ParameterToken:
    """One configured value with the semantics needed to interpret it."""

    field: str
    field_id: int
    owner: str
    namespace_code: int
    value: float | bool | str | None
    normalized: float
    missing: int
    extension: int
    type_code: int
    unit: str
    unit_kind: str
    unit_kind_code: int
    minimum: float | None
    maximum: float | None
    log_scale: int
    bound_low: float
    bound_span: float
    share: float = 1.0
    shortest: bool = True

    def numeric_vector(self) -> list[float]:
        return [
            float(self.normalized),
            float(self.log_scale),
            float(self.bound_low),
            float(self.bound_span),
            float(self.share),
            float(self.extension),
            0.0 if self.missing else 1.0,
        ]

    def categorical_codes(self) -> list[int]:
        return [int(self.type_code), int(self.unit_kind_code), int(self.namespace_code)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "field_id": self.field_id,
            "owner": self.owner,
            "namespace_code": self.namespace_code,
            "value": self.value,
            "normalized": round(float(self.normalized), 6),
            "missing": int(self.missing),
            "extension": int(self.extension),
            "type": TYPE_CODES[self.type_code],
            "unit": self.unit,
            "unit_kind": self.unit_kind,
            "unit_kind_code": int(self.unit_kind_code),
            "bounds": [self.minimum, self.maximum],
            "log_scale": int(self.log_scale),
            "bound_low": round(float(self.bound_low), 6),
            "bound_span": round(float(self.bound_span), 6),
            "share": round(float(self.share), 6),
            "source": "registry" if not self.extension else "extension",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterToken":
        bounds = list(payload.get("bounds", [None, None]))
        return cls(
            field=str(payload["field"]),
            field_id=int(payload["field_id"]),
            owner=str(payload["owner"]),
            namespace_code=int(payload["namespace_code"]),
            value=payload.get("value"),
            normalized=float(payload.get("normalized", 0.0)),
            missing=int(payload.get("missing", 0)),
            extension=int(payload.get("extension", 0)),
            type_code=TYPE_CODES.index(str(payload.get("type", "unknown")))
            if str(payload.get("type", "unknown")) in TYPE_CODES else UNKNOWN_TYPE_CODE,
            unit=str(payload.get("unit", "")),
            unit_kind=str(payload.get("unit_kind", "")),
            unit_kind_code=int(payload.get("unit_kind_code", 0)),
            minimum=bounds[0] if bounds else None,
            maximum=bounds[1] if len(bounds) > 1 else None,
            log_scale=int(payload.get("log_scale", 0)),
            bound_low=float(payload.get("bound_low", 0.0)),
            bound_span=float(payload.get("bound_span", 0.0)),
            share=float(payload.get("share", 1.0)),
        )


@dataclass
class ParameterVocabulary:
    """Stable ids for the categorical token features of one registry build."""

    namespaces: tuple[str, ...]
    unit_kinds: tuple[str, ...]
    schema_hash: str
    field_ids: dict[str, int]
    reference_scale: dict[str, float]

    @classmethod
    def from_registry(cls, registry: ParameterRegistry) -> "ParameterVocabulary":
        specs = sorted(registry.specs(), key=lambda spec: spec.key)
        field_ids = {spec.key: index for index, spec in enumerate(specs)}
        namespaces = tuple(sorted({spec.owner for spec in specs}))
        reference: dict[str, float] = {}
        for spec in specs:
            kind = _unit_kind(spec)
            magnitude = max(abs(spec.minimum or 0.0), abs(spec.maximum or 0.0))
            reference[kind] = max(reference.get(kind, 0.0), magnitude)
        unit_kinds = tuple(sorted(reference))
        return cls(
            namespaces=namespaces,
            unit_kinds=unit_kinds,
            schema_hash=registry.schema_hash(),
            field_ids=field_ids,
            reference_scale=reference,
        )

    @property
    def namespace_codes(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.namespaces)}

    @property
    def unit_kind_codes(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.unit_kinds)}

    def field_index(self, key: str, *, extension_buckets: int = 512) -> int:
        """Stable id for a registry key, or a stable extension bucket."""
        if key in self.field_ids:
            return self.field_ids[key]
        digest = hashlib.sha256(f"{self.schema_hash}:{key}".encode("utf-8")).hexdigest()
        return len(self.field_ids) + int(digest[:8], 16) % extension_buckets

    def payload(self) -> dict[str, Any]:
        return {
            "version": PARAM_TOKEN_VERSION,
            "schema_hash": self.schema_hash,
            "namespaces": list(self.namespaces),
            "unit_kinds": list(self.unit_kinds),
            "field_ids": dict(sorted(self.field_ids.items())),
            "reference_scale": dict(sorted(self.reference_scale.items())),
            "token_numeric_features": list(TOKEN_NUMERIC_FEATURES),
            "token_categorical_features": list(TOKEN_CATEGORICAL_FEATURES),
        }


def _unit_kind(spec: ParameterSpec) -> str:
    if spec.kind == "number":
        try:
            return units.unit_kind(spec.unit)
        except Exception:  # pragma: no cover - a unit the registry does not know
            return "unknown"
    return "categorical"


def _type_code(value: Any, kind: str) -> int:
    if kind in TYPE_CODES:
        return TYPE_CODES.index(kind)
    if isinstance(value, bool):
        return TYPE_CODES.index("boolean")
    if isinstance(value, int):
        return TYPE_CODES.index("integer")
    if isinstance(value, float):
        return TYPE_CODES.index("number")
    if isinstance(value, str):
        return TYPE_CODES.index("string")
    return UNKNOWN_TYPE_CODE


def _normalize_numeric(spec: ParameterSpec, value: float) -> float:
    if spec.minimum is not None and spec.maximum is not None and spec.maximum > spec.minimum:
        return max(0.0, min(1.0, (float(value) - spec.minimum) / (spec.maximum - spec.minimum)))
    scale = float(spec.normalization.get("scale", 1.0)) or 1.0
    offset = float(spec.normalization.get("offset", 0.0))
    return max(0.0, min(1.0, (float(value) - offset) / scale))


def _bounds_metadata(spec: ParameterSpec, vocabulary: ParameterVocabulary) -> tuple[float, float]:
    reference = vocabulary.reference_scale.get(_unit_kind(spec), 0.0)
    reference = reference if reference > 0.0 else 1.0
    low = float(spec.minimum or 0.0) / reference
    high = float(spec.maximum if spec.maximum is not None else spec.minimum or 0.0) / reference
    return round(low, 6), round(max(0.0, high - low), 6)


def _collect_declared(config: Mapping[str, Any]) -> dict[str, list[tuple[str, Any]]]:
    """Every ``(owner.key) -> [(satellite_id, value), ...]`` declared in a config."""
    declared: dict[str, list[tuple[str, Any]]] = {}
    for satellite in config.get("satellites", []) or []:
        satellite_id = str(satellite.get("id", "sat"))
        for component in satellite.get("components", []) or []:
            owner = str(component.get("type", ""))
            values = component.get("parameters") or {}
            for key, value in values.items():
                declared.setdefault(f"{owner}.{key}", []).append((satellite_id, value))
    return declared


def build_parameter_context(
    config: Mapping[str, Any],
    *,
    registry: ParameterRegistry | None = None,
    vocabulary: ParameterVocabulary | None = None,
    include_registry_defaults: bool = True,
) -> dict[str, Any]:
    """Tokenise a configuration payload (JSON-safe) plus a compatibility report.

    ``include_registry_defaults`` adds a ``missing=1`` token for every *declared
    component's* registry fields that the configuration leaves at default, so the
    encoder sees the declared shape of the configuration rather than only the
    values somebody happened to write down.
    """
    active_registry = registry or builtin_registry()
    vocab = vocabulary or ParameterVocabulary.from_registry(active_registry)
    declared = _collect_declared(config)
    owners = sorted({key.rpartition(".")[0] for key in declared})
    spec_by_key = {spec.key: spec for spec in active_registry.specs()}

    entries: list[tuple[str, Any, float, bool, int]] = []
    # (key, value, share, missing, extension)
    for key, pairs in declared.items():
        values: dict[Any, int] = {}
        for _, value in pairs:
            values[value] = values.get(value, 0) + 1
        for value, count in values.items():
            entries.append((key, value, count / max(1, len(pairs)), 0, 0 if key in spec_by_key else 1))

    if include_registry_defaults:
        for owner in owners:
            for spec in active_registry.by_owner(owner):
                if spec.key in declared:
                    continue
                entries.append((spec.key, None, 0.0, 1, 0))

    tokens: list[ParameterToken] = []
    for key, value, share, missing, extension in entries:
        spec = spec_by_key.get(key)
        owner = spec.owner if spec else key.rpartition(".")[0]
        if spec is not None:
            kind = spec.kind
            unit = spec.unit
            unit_kind = _unit_kind(spec)
            minimum, maximum = spec.minimum, spec.maximum
            log_scale = 1 if spec.distribution == "log-uniform" else 0
            bound_low, bound_span = _bounds_metadata(spec, vocab)
            if missing or value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                normalized = 0.0
            else:
                normalized = _normalize_numeric(spec, float(value))
            type_code = _type_code(value, kind)
            field_id = vocab.field_index(key)
        else:
            kind = "unknown"
            unit = ""
            unit_kind = "unknown"
            minimum = maximum = None
            log_scale = 0
            bound_low = bound_span = 0.0
            normalized = 0.0 if value is None or isinstance(value, bool) else 0.5
            type_code = _type_code(value, kind)
            field_id = vocab.field_index(key)
        tokens.append(ParameterToken(
            field=key,
            field_id=field_id,
            owner=owner,
            namespace_code=vocab.namespace_codes.get(owner, len(vocab.namespaces)),
            value=value,
            normalized=normalized,
            missing=missing,
            extension=extension,
            type_code=type_code,
            unit=unit,
            unit_kind=unit_kind,
            unit_kind_code=vocab.unit_kind_codes.get(unit_kind, len(vocab.unit_kinds)),
            minimum=minimum,
            maximum=maximum,
            log_scale=log_scale,
            bound_low=bound_low,
            bound_span=bound_span,
            share=share,
        ))

    tokens.sort(key=lambda token: (token.owner, token.field, str(token.value)))
    payload_tokens = [token.to_dict() for token in tokens]
    report = compatibility_report(payload_tokens, registry=active_registry, vocabulary=vocab)
    return {
        "version": PARAM_TOKEN_VERSION,
        "schema_hash": vocab.schema_hash,
        "vocabulary": vocab.payload(),
        "token_numeric_features": list(TOKEN_NUMERIC_FEATURES),
        "token_categorical_features": list(TOKEN_CATEGORICAL_FEATURES),
        "tokens": payload_tokens,
        "compatibility": report,
        "hash": trace_hash(payload_tokens),
    }


def compatibility_report(
    tokens: Sequence[Mapping[str, Any]],
    *,
    registry: ParameterRegistry | None = None,
    vocabulary: ParameterVocabulary | None = None,
) -> dict[str, Any]:
    """Which parts of this configuration the model is (not) entitled to trust."""
    active_registry = registry or builtin_registry()
    vocab = vocabulary or ParameterVocabulary.from_registry(active_registry)
    spec_by_key = {spec.key: spec for spec in active_registry.specs()}
    known_components = set(component_registry.registered_components())

    unknown_fields: list[str] = []
    unsupported_owners: list[str] = []
    out_of_range: list[dict[str, Any]] = []
    unset_fields: list[str] = []
    for token in tokens:
        key = str(token["field"])
        owner = str(token.get("owner", key.rpartition(".")[0]))
        spec = spec_by_key.get(key)
        if spec is None:
            unknown_fields.append(key)
            if owner not in known_components:
                unsupported_owners.append(owner)
            continue
        if token.get("missing"):
            unset_fields.append(key)
            continue
        value = token.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum = token["bounds"][0]
            maximum = token["bounds"][1]
            if minimum is not None and float(value) < float(minimum) - 1e-9:
                out_of_range.append({"field": key, "value": value, "bound": [minimum, maximum], "side": "low"})
            if maximum is not None and float(value) > float(maximum) + 1e-9:
                out_of_range.append({"field": key, "value": value, "bound": [minimum, maximum], "side": "high"})

    reasons: list[str] = []
    if unknown_fields:
        reasons.append(f"{len(unknown_fields)} field(s) have no registry semantics")
    if unsupported_owners:
        reasons.append(f"unsupported component type(s): {', '.join(sorted(set(unsupported_owners)))}")
    if out_of_range:
        reasons.append(f"{len(out_of_range)} value(s) fall outside their declared bounds")
    return {
        "compatible": not reasons,
        "schema_hash": vocab.schema_hash,
        "torch_usable": not unknown_fields and not out_of_range,
        "unknown_fields": sorted(set(unknown_fields)),
        "unsupported_owners": sorted(set(unsupported_owners)),
        "out_of_range": out_of_range,
        "unset_fields": sorted(set(unset_fields)),
        "reasons": reasons,
        "token_count": len(tokens),
    }


def trace_hash(tokens: Sequence[Mapping[str, Any]]) -> str:
    import json

    canonical = json.dumps(list(tokens), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def tokens_of(context: Mapping[str, Any]) -> list[ParameterToken]:
    return [ParameterToken.from_dict(item) for item in context.get("tokens", [])]


def numeric_matrix(context: Mapping[str, Any]) -> list[list[float]]:
    return [token.numeric_vector() for token in tokens_of(context)]


def categorical_matrix(context: Mapping[str, Any]) -> list[list[int]]:
    return [token.categorical_codes() for token in tokens_of(context)]


def empty_context(*, registry: ParameterRegistry | None = None) -> dict[str, Any]:
    active_registry = registry or builtin_registry()
    vocab = ParameterVocabulary.from_registry(active_registry)
    return {
        "version": PARAM_TOKEN_VERSION,
        "schema_hash": vocab.schema_hash,
        "vocabulary": vocab.payload(),
        "token_numeric_features": list(TOKEN_NUMERIC_FEATURES),
        "token_categorical_features": list(TOKEN_CATEGORICAL_FEATURES),
        "tokens": [],
        "compatibility": compatibility_report([], registry=active_registry, vocabulary=vocab),
        "hash": trace_hash([]),
    }


def context_is_compatible(context: Mapping[str, Any], *, expected_schema_hash: str | None = None) -> dict[str, Any]:
    """Compare a context against the schema a checkpoint was trained on."""
    report = dict(context.get("compatibility", {}))
    report["expected_schema_hash"] = expected_schema_hash
    report["schema_match"] = (
        expected_schema_hash is None or expected_schema_hash == context.get("schema_hash")
    )
    if not report["schema_match"]:
        report["compatible"] = False
        report["reasons"] = list(report.get("reasons", [])) + [
            "the configuration schema hash differs from the one the model was trained on"
        ]
    return report


def vocabulary_payload(registry: ParameterRegistry | None = None) -> dict[str, Any]:
    return ParameterVocabulary.from_registry(registry or builtin_registry()).payload()


def decode_tokens(tokens: Iterable[Mapping[str, Any]]) -> list[ParameterToken]:
    return [ParameterToken.from_dict(item) for item in tokens]


__all__ = [
    "PARAM_TOKEN_VERSION",
    "TOKEN_NUMERIC_FEATURES",
    "TOKEN_CATEGORICAL_FEATURES",
    "TYPE_CODES",
    "ParameterContextError",
    "ParameterToken",
    "ParameterVocabulary",
    "build_parameter_context",
    "compatibility_report",
    "tokens_of",
    "numeric_matrix",
    "categorical_matrix",
    "empty_context",
    "context_is_compatible",
    "vocabulary_payload",
    "decode_tokens",
    "trace_hash",
]
