"""Day 40 acceptance: metadata-aware parameter tokens and compatibility reporting."""

from __future__ import annotations

import json
import unittest

from constellab_core.parameters import builtin_registry

from constellab_ml.param_encoder import (
    TOKEN_CATEGORICAL_FEATURES,
    TOKEN_NUMERIC_FEATURES,
    ParameterToken,
    ParameterVocabulary,
    build_parameter_context,
    categorical_matrix,
    compatibility_report,
    context_is_compatible,
    empty_context,
    numeric_matrix,
    tokens_of,
)

from tests.test_ml_scenarios import fast_distribution_set
from constellab_ml.scenarios import ScenarioSampler


def sample_config(**overrides) -> dict:
    sampler = ScenarioSampler(seed=3, distribution_set=fast_distribution_set(), salt="unit")
    return sampler.sample(0, split="train").spec.config


class TestTokens(unittest.TestCase):
    def setUp(self) -> None:
        self.config = sample_config()
        self.context = build_parameter_context(self.config)
        self.tokens = tokens_of(self.context)

    def test_context_is_versioned_and_json_safe(self) -> None:
        self.assertEqual(self.context["version"], "parameter-tokens-1")
        self.assertEqual(
            self.context["token_numeric_features"], list(TOKEN_NUMERIC_FEATURES)
        )
        self.assertEqual(
            self.context["token_categorical_features"], list(TOKEN_CATEGORICAL_FEATURES)
        )
        self.assertEqual(json.loads(json.dumps(self.context))["tokens"], self.context["tokens"])

    def test_every_declared_field_becomes_a_token(self) -> None:
        fields = {token.field for token in self.tokens}
        self.assertIn("sensor.conical.half_angle_deg", fields)
        self.assertIn("power.solar-battery.battery_capacity_j", fields)
        self.assertIn("storage.buffer.capacity_bit", fields)
        self.assertIn("compute.capacity.throughput_hz", fields)
        self.assertIn("payload.imaging.data_rate_bit_s", fields)

    def test_tokens_carry_semantics_not_just_values(self) -> None:
        token = next(t for t in self.tokens if t.field == "sensor.conical.half_angle_deg")
        self.assertEqual(token.owner, "sensor.conical")
        self.assertEqual(token.unit, "deg")
        self.assertEqual(token.unit_kind, "angle")
        self.assertEqual(token.minimum, 0.1)
        self.assertEqual(token.maximum, 89.9)
        self.assertEqual(token.type_code, 0)  # number
        self.assertGreaterEqual(token.normalized, 0.0)
        self.assertLessEqual(token.normalized, 1.0)
        self.assertGreater(token.bound_span, 0.0)

    def test_log_scaled_fields_are_flagged(self) -> None:
        token = next(t for t in self.tokens if t.field == "storage.buffer.capacity_bit")
        self.assertEqual(token.log_scale, 1)

    def test_field_ids_are_stable_against_display_labels(self) -> None:
        registry = builtin_registry()
        vocabulary = ParameterVocabulary.from_registry(registry)
        renamed = type(registry)([
            spec.with_label("Totally different display label") for spec in registry.specs()
        ], source="renamed")
        other = ParameterVocabulary.from_registry(renamed)
        self.assertEqual(vocabulary.schema_hash, other.schema_hash)
        self.assertEqual(vocabulary.field_ids, other.field_ids)

    def test_field_ids_do_not_depend_on_token_order(self) -> None:
        reversed_context = build_parameter_context(
            {"satellites": list(reversed(self.config["satellites"]))}
        )
        first = {token.field: token.field_id for token in self.tokens}
        second = {token.field: token.field_id for token in tokens_of(reversed_context)}
        for field, value in first.items():
            if field in second:
                self.assertEqual(value, second[field])

    def test_tokens_are_order_independent(self) -> None:
        shuffled = json.loads(json.dumps(self.config))
        shuffled["satellites"] = list(reversed(shuffled["satellites"]))
        other = build_parameter_context(shuffled)
        self.assertEqual(other["hash"], self.context["hash"])
        self.assertEqual(
            [token.field for token in tokens_of(other)],
            [token.field for token in self.tokens],
        )

    def test_matrices_have_the_declared_shapes(self) -> None:
        numeric = numeric_matrix(self.context)
        categorical = categorical_matrix(self.context)
        self.assertEqual(len(numeric), len(self.tokens))
        self.assertEqual(len(numeric[0]), len(TOKEN_NUMERIC_FEATURES))
        self.assertEqual(len(categorical[0]), len(TOKEN_CATEGORICAL_FEATURES))

    def test_declared_but_unset_fields_are_missing_not_zero(self) -> None:
        # A component may declare only part of its registry fields; the rest are
        # present as explicit ``missing`` tokens, never as a fabricated zero.
        config = json.loads(json.dumps(self.config))
        for satellite in config["satellites"]:
            for component in satellite["components"]:
                if component["type"] == "storage.buffer":
                    component["parameters"] = {"capacity_bit": 4.0e9}
        context = build_parameter_context(config)
        missing = [token for token in tokens_of(context) if token.missing]
        self.assertTrue(missing)
        self.assertTrue(all(token.field.startswith("storage.buffer.") for token in missing))
        self.assertNotIn("storage.buffer.capacity_bit", {token.field for token in missing})
        self.assertTrue(all(token.numeric_vector()[6] == 0.0 for token in missing))

    def test_share_reports_fleet_heterogeneity(self) -> None:
        config = json.loads(json.dumps(self.config))
        second = json.loads(json.dumps(config["satellites"][0]))
        second["id"] = "ml-999"
        for component in second["components"]:
            if component["type"] == "sensor.conical":
                component["parameters"]["half_angle_deg"] = 40.0
        config["satellites"].append(second)
        context = build_parameter_context(config)
        shares = [
            token.share for token in tokens_of(context)
            if token.field == "sensor.conical.half_angle_deg"
        ]
        self.assertEqual(len(shares), 2)
        self.assertAlmostEqual(sum(shares), 1.0, places=6)


class TestCompatibility(unittest.TestCase):
    def setUp(self) -> None:
        self.config = sample_config()

    def test_known_configuration_is_compatible(self) -> None:
        context = build_parameter_context(self.config)
        report = context["compatibility"]
        self.assertTrue(report["compatible"])
        self.assertTrue(report["torch_usable"])
        self.assertEqual(report["reasons"], [])

    def test_unregistered_field_is_surfaced_as_an_extension(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["satellites"][0]["components"].append({
            "type": "sensor.conical",
            "parameters": {"half_angle_deg": 30.0, "experimental_gain": 1.5},
        })
        context = build_parameter_context(config)
        report = context["compatibility"]
        self.assertIn("sensor.conical.experimental_gain", report["unknown_fields"])
        self.assertFalse(report["compatible"])
        token = next(
            token for token in tokens_of(context) if token.field.endswith("experimental_gain")
        )
        self.assertEqual(token.extension, 1)
        self.assertGreaterEqual(token.field_id, len(builtin_registry().specs()))

    def test_unsupported_component_type_is_reported(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["satellites"][0]["components"].append({
            "type": "reactor.fusion",
            "parameters": {"output_w": 10.0},
        })
        report = build_parameter_context(config)["compatibility"]
        self.assertIn("reactor.fusion", report["unsupported_owners"])
        self.assertFalse(report["compatible"])

    def test_out_of_range_value_is_reported(self) -> None:
        config = json.loads(json.dumps(self.config))
        for component in config["satellites"][0]["components"]:
            if component["type"] == "sensor.conical":
                component["parameters"]["half_angle_deg"] = 120.0
        report = build_parameter_context(config)["compatibility"]
        self.assertTrue(report["out_of_range"])
        self.assertFalse(report["torch_usable"])

    def test_schema_mismatch_is_reported(self) -> None:
        context = build_parameter_context(self.config)
        report = context_is_compatible(context, expected_schema_hash="deadbeefdeadbeef")
        self.assertFalse(report["schema_match"])
        self.assertFalse(report["compatible"])

    def test_matching_schema_stays_compatible(self) -> None:
        context = build_parameter_context(self.config)
        report = context_is_compatible(context, expected_schema_hash=context["schema_hash"])
        self.assertTrue(report["schema_match"])
        self.assertTrue(report["compatible"])

    def test_empty_context_is_valid_and_warns_about_nothing(self) -> None:
        context = empty_context()
        self.assertEqual(context["tokens"], [])
        self.assertTrue(context["compatibility"]["compatible"])
        self.assertEqual(numeric_matrix(context), [])

    def test_report_is_serialisable(self) -> None:
        report = build_parameter_context(self.config)["compatibility"]
        self.assertEqual(json.loads(json.dumps(report)), report)

    def test_token_round_trip(self) -> None:
        context = build_parameter_context(self.config)
        token = tokens_of(context)[0]
        restored = ParameterToken.from_dict(token.to_dict())
        self.assertEqual(restored.to_dict(), token.to_dict())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
