"""Stage 4 gate math tests, including the design doc's canonical example."""

import math
import unittest

from s1core import (
    ABSTAIN_INSUFFICIENT_EVIDENCE,
    GateInputs,
    GatePolicy,
    js_divergence_bits,
    normalized_entropy,
    temperature_scale,
)
from s1core.gates import evaluate, validate_support

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fixtures import ABSTAIN_DIST


def choice_inputs(policy: GatePolicy | None = None) -> GateInputs:
    from s1core import ABSTAIN_INSUFFICIENT_EVIDENCE, ABSTAIN_OUT_OF_DOMAIN

    return GateInputs(
        question_id="q",
        primitive="choice",
        policy=policy or GatePolicy(),
        escape_keys=frozenset({ABSTAIN_INSUFFICIENT_EVIDENCE, ABSTAIN_OUT_OF_DOMAIN}),
    )


def score_inputs(policy: GatePolicy | None = None) -> GateInputs:
    return GateInputs(question_id="q", primitive="score", policy=policy or GatePolicy())


class TemperatureTests(unittest.TestCase):
    def test_identity_at_one(self):
        dist = {"a": 0.7, "b": 0.3}
        self.assertEqual(temperature_scale(dist, 1.0), dist)

    def test_high_temperature_flattens(self):
        dist = {"a": 0.9, "b": 0.1}
        flat = temperature_scale(dist, 5.0)
        self.assertGreater(normalized_entropy(flat), normalized_entropy(dist))

    def test_low_temperature_sharpens(self):
        dist = {"a": 0.6, "b": 0.4}
        sharp = temperature_scale(dist, 0.3)
        self.assertLess(normalized_entropy(sharp), normalized_entropy(dist))

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            temperature_scale({"a": 1.0}, 0.0)


class ChoiceGateTests(unittest.TestCase):
    def test_ratified(self):
        verdict = evaluate(choice_inputs(), {"a": 0.85, "b": 0.10, "c": 0.05}, "d" * 16)
        self.assertEqual(verdict.status.name, "RATIFIED")
        self.assertEqual(verdict.value, "a")
        self.assertEqual(verdict.raw_value, "a")
        self.assertAlmostEqual(verdict.top2_margin, 0.75)

    def test_low_top1_abstains(self):
        verdict = evaluate(choice_inputs(), {"a": 0.5, "b": 0.3, "c": 0.2}, "d" * 16)
        self.assertEqual(verdict.status.name, "ABSTAINED_GATE")
        self.assertIsNone(verdict.value)
        self.assertEqual(verdict.raw_value, "a")

    def test_escape_hatch_maps_to_model_abstention(self):
        verdict = evaluate(choice_inputs(), ABSTAIN_DIST, "d" * 16)
        self.assertEqual(verdict.status.name, "ABSTAINED_MODEL")
        self.assertIsNone(verdict.value)
        self.assertEqual(verdict.raw_value, ABSTAIN_INSUFFICIENT_EVIDENCE)

    def test_conformal_gate(self):
        policy = GatePolicy(conformal_tau=0.30)
        # top-2 margin 0.05 < 0.20 already abstains; use conformal alone:
        policy = GatePolicy(min_top1_prob=0.0, min_top2_margin=0.0, max_normalized_entropy=1.0,
                            conformal_tau=0.30)
        # {a: 0.5, b: 0.45} -> conformal set {a, b} size 2 -> abstain
        verdict = evaluate(choice_inputs(policy), {"a": 0.5, "b": 0.45, "c": 0.05}, "d" * 16)
        self.assertEqual(verdict.status.name, "ABSTAINED_GATE")
        self.assertEqual(verdict.conformal_set_size, 2)
        # singleton set passes
        verdict2 = evaluate(
            choice_inputs(policy), {"a": 0.9, "b": 0.05, "c": 0.05}, "d" * 16
        )
        self.assertEqual(verdict2.status.name, "RATIFIED")
        self.assertEqual(verdict2.conformal_set_size, 1)

    def test_scale_invariant_entropy(self):
        # H({0.5, 0.25, 0.25}) = 1.0397 nats; dividing by ln(m) keeps one
        # threshold meaningful across option counts.
        three = {"a": 0.5, "b": 0.25, "c": 0.25}
        expected = -(0.5 * math.log(0.5) + 0.5 * math.log(0.25)) / math.log(3)
        self.assertAlmostEqual(normalized_entropy(three), expected, places=12)


class ScoreGateTests(unittest.TestCase):
    def test_design_doc_state_a_concentrated(self):
        gi = score_inputs()
        verdict = evaluate(gi, {"1": 0.01, "2": 0.98, "3": 0.01}, "d" * 16)
        self.assertEqual(verdict.status.name, "RATIFIED")
        self.assertAlmostEqual(verdict.value, 2.0)
        self.assertAlmostEqual(verdict.score_mode_deviation, 0.02 / 2.0)

    def test_design_doc_state_b_bimodal(self):
        gi = score_inputs()
        verdict = evaluate(gi, {"1": 0.50, "2": 0.00, "3": 0.50}, "d" * 16)
        # Same expectation (2.0) as State A, but polarized: refused.
        self.assertEqual(verdict.status.name, "BIMODAL_CONFLICT")
        self.assertIsNone(verdict.value)
        self.assertAlmostEqual(verdict.raw_value, 2.0)
        self.assertAlmostEqual(verdict.score_mode_deviation, 1.0 / 2.0)

    def test_tie_break_is_deterministic(self):
        verdict = evaluate(score_inputs(), {"1": 0.5, "2": 0.5}, "d" * 16)
        self.assertEqual(verdict.status.name, "BIMODAL_CONFLICT")
        self.assertAlmostEqual(verdict.raw_value, 1.5)


class GateValidationTests(unittest.TestCase):
    def test_entropy_of_degenerate_support_is_zero(self):
        self.assertEqual(normalized_entropy({}), 0.0)
        self.assertEqual(normalized_entropy({"a": 1.0}), 0.0)

    def test_score_keys_must_be_integer_levels(self):
        with self.assertRaises(ValueError):
            evaluate(score_inputs(), {"low": 0.5, "high": 0.5}, "d" * 16)

    def test_noul_edges_must_be_ordered(self):
        gi = GateInputs("n", "noul", GatePolicy(), noul_true_edge=0.2, noul_false_edge=0.5)
        with self.assertRaises(ValueError):
            evaluate(gi, {"true": 0.9, "false": 0.1}, "d" * 16)

    def test_unknown_primitive(self):
        gi = GateInputs("q", "gossip", GatePolicy())
        with self.assertRaises(ValueError):
            evaluate(gi, {"a": 1.0}, "d" * 16)

    def test_conformal_tau_must_be_a_probability(self):
        policy = GatePolicy(min_top1_prob=0.0, min_top2_margin=0.0,
                            max_normalized_entropy=1.0, conformal_tau=1.5)
        with self.assertRaises(ValueError):
            evaluate(choice_inputs(policy), {"a": 0.9, "b": 0.1}, "d" * 16)


class NoulGateTests(unittest.TestCase):
    def noul(self, p, **policy_kwargs):
        gi = GateInputs(
            question_id="n", primitive="noul", policy=GatePolicy(**policy_kwargs)
        )
        return evaluate(gi, {"true": p, "false": 1.0 - p}, "d" * 16)

    def test_true_edge(self):
        verdict = self.noul(0.9)
        self.assertEqual(verdict.status.name, "RATIFIED")
        self.assertIs(verdict.value, True)

    def test_false_edge(self):
        verdict = self.noul(0.2)
        self.assertEqual(verdict.status.name, "RATIFIED")
        self.assertIs(verdict.value, False)

    def test_gray_zone_fails_closed(self):
        verdict = self.noul(0.5)
        self.assertEqual(verdict.status.name, "ABSTAINED_GATE")
        self.assertIsNone(verdict.value)
        self.assertAlmostEqual(verdict.raw_value, 0.5)
        self.assertAlmostEqual(verdict.confidence, 0.5)


class JSDivergenceTests(unittest.TestCase):
    def test_identical_is_zero(self):
        d = {"a": 0.5, "b": 0.5}
        self.assertEqual(js_divergence_bits(d, d), 0.0)

    def test_disjoint_singletons_are_one_bit(self):
        self.assertAlmostEqual(js_divergence_bits({"a": 1.0}, {"b": 1.0}), 1.0, places=12)

    def test_symmetric(self):
        p, q = {"a": 0.8, "b": 0.2}, {"a": 0.3, "b": 0.7}
        self.assertAlmostEqual(js_divergence_bits(p, q), js_divergence_bits(q, p), places=12)


class DeclaredSupportTests(unittest.TestCase):
    """The declared output space is the law a returned distribution is judged
    against — a key the question never declared can never win a ratification,
    and a declared key can never be silently imputed."""

    def declared_choice(self) -> GateInputs:
        return GateInputs("q", "choice", GatePolicy(), declared_keys=frozenset({"a", "b"}))

    def declared_score(self, k: int = 3) -> GateInputs:
        return GateInputs("q", "score", GatePolicy(), score_levels=k)

    def test_undeclared_key_raises(self):
        with self.assertRaises(ValueError):
            evaluate(self.declared_choice(), {"not_a_declared_option": 1.0}, "")
        with self.assertRaises(ValueError):
            evaluate(self.declared_choice(), {"a": 1.0, "b": 0.0, "stray": 0.0}, "")

    def test_incomplete_support_raises(self):
        # a confident winner with declared options missing is a backend
        # contract defect, not a ratification
        with self.assertRaises(ValueError):
            evaluate(self.declared_choice(), {"a": 1.0}, "")

    def test_complete_support_ratifies(self):
        verdict = evaluate(self.declared_choice(), {"a": 0.9, "b": 0.1}, "")
        self.assertEqual(verdict.status.name, "RATIFIED")
        self.assertEqual(verdict.value, "a")

    def test_score_support_must_equal_declared_k(self):
        with self.assertRaises(ValueError):
            evaluate(self.declared_score(3),
                     {"1": 0.5, "2": 0.25, "3": 0.125, "4": 0.125}, "")
        with self.assertRaises(ValueError):
            evaluate(self.declared_score(3), {"1": 0.5, "2": 0.5}, "")

    def test_score_level_collision_raises(self):
        # "1" and "01" are distinct wire keys that collapse to one level
        with self.assertRaises(ValueError):
            evaluate(self.declared_score(3), {"1": 0.5, "01": 0.5, "2": 0.5}, "")

    def test_non_integer_score_keys_raise(self):
        with self.assertRaises(ValueError):
            evaluate(self.declared_score(2), {"low": 0.5, "high": 0.5}, "")

    def test_legacy_inputs_without_metadata_keep_historical_semantics(self):
        # no declared metadata (v1/v2 ledger records) -> validate_support is a
        # no-op; the gate judges the distribution exactly as before
        verdict = evaluate(GateInputs("q", "choice", GatePolicy()), {"whatever": 1.0}, "")
        self.assertEqual(verdict.status.name, "RATIFIED")
        self.assertEqual(verdict.value, "whatever")
        verdict = evaluate(GateInputs("q", "score", GatePolicy()),
                           {"1": 0.98, "2": 0.01, "3": 0.01}, "")
        self.assertEqual(verdict.status.name, "RATIFIED")

    def test_noul_support_is_still_exactly_true_false(self):
        with self.assertRaises(ValueError):
            validate_support(GateInputs("n", "noul", GatePolicy()),
                             {"true": 1.0, "yes": 0.0})

    def test_unknown_primitive_raises(self):
        with self.assertRaises(ValueError):
            validate_support(GateInputs("q", "gossip", GatePolicy()), {"a": 1.0})

    def test_jsonable_round_trip_keeps_declared_support(self):
        restored = GateInputs.from_jsonable(self.declared_choice().to_jsonable())
        self.assertEqual(restored.declared_keys, frozenset({"a", "b"}))
        restored = GateInputs.from_jsonable(self.declared_score().to_jsonable())
        self.assertEqual(restored.score_levels, 3)

    def test_legacy_jsonable_loads_with_empty_defaults(self):
        legacy = GateInputs.from_jsonable({
            "question_id": "q", "primitive": "choice", "policy": {},
            "escape_keys": [], "noul_true_edge": 0.75, "noul_false_edge": 0.35,
        })
        self.assertEqual(legacy.declared_keys, frozenset())
        self.assertIsNone(legacy.score_levels)


if __name__ == "__main__":
    unittest.main()
