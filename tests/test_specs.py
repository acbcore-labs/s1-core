"""Stage 1 spec compiler tests (Pillar I: semantic rigor at compile time)."""

import unittest

from s1core import (
    ABSTAIN_INSUFFICIENT_EVIDENCE,
    ABSTAIN_OUT_OF_DOMAIN,
    ChoiceSpec,
    GatePolicy,
    NoulSpec,
    ScoreSpec,
    SpecError,
    values_match,
)


class ChoiceSpecTests(unittest.TestCase):
    def test_compile_injects_escape_hatches(self):
        spec = ChoiceSpec(
            question_id="lane",
            instructions="Which handling lane does this ticket belong in?",
            options={
                "auto": "Policy limits are satisfied and history supports approval.",
                "review": "A human must review the amount before approval.",
            },
        )
        compiled = spec.compile()
        self.assertIn(ABSTAIN_INSUFFICIENT_EVIDENCE, compiled.criteria)
        self.assertIn(ABSTAIN_OUT_OF_DOMAIN, compiled.criteria)
        self.assertEqual(
            compiled.gate_inputs.escape_keys,
            frozenset({ABSTAIN_INSUFFICIENT_EVIDENCE, ABSTAIN_OUT_OF_DOMAIN}),
        )

    def test_escape_hatches_can_be_disabled(self):
        spec = ChoiceSpec(
            question_id="lane",
            instructions="Which handling lane does this ticket belong in?",
            options={
                "auto": "Policy limits are satisfied and history supports approval.",
                "review": "A human must review the amount before approval.",
            },
            inject_escape_hatch=False,
        )
        compiled = spec.compile()
        self.assertNotIn(ABSTAIN_INSUFFICIENT_EVIDENCE, compiled.criteria)
        self.assertEqual(compiled.gate_inputs.escape_keys, frozenset())

    def test_terse_instructions_rejected(self):
        with self.assertRaises(SpecError):
            ChoiceSpec(
                question_id="lane", instructions="pick lane",
                options={"a": "one meaningful criteria line", "b": "another meaningful line here"},
            ).compile()

    def test_thin_option_criteria_rejected(self):
        with self.assertRaises(SpecError):
            ChoiceSpec(
                question_id="lane",
                instructions="Which handling lane does this ticket belong in?",
                options={"auto": "approve it", "review": "A human must review the amount first."},
            ).compile()

    def test_option_count_bounds(self):
        good = {f"opt_{i}": f"criteria number {i} in plain prose words" for i in range(1, 11)}
        spec = ChoiceSpec(
            question_id="many", instructions="Which of the following numbered options applies here?",
            options=good,
        )
        self.assertEqual(len(spec.compile().criteria), 12)  # + 2 escape hatches
        with self.assertRaises(SpecError):
            ChoiceSpec(
                question_id="one", instructions="Which of the following numbered options applies?",
                options={"only": "one meaningful criteria line of prose"},
            ).compile()


class ScoreSpecTests(unittest.TestCase):
    def test_valid_rubric(self):
        spec = ScoreSpec(
            question_id="urgency",
            instructions="Rate the urgency of this ticket on the rubric provided.",
            rubric_levels={1: "Routine, no impact.", 2: "Elevated, some impact.", 3: "Critical impact now."},
        )
        compiled = spec.compile()
        self.assertEqual(list(compiled.criteria.keys()), ["1", "2", "3"])

    def test_non_contiguous_levels_rejected(self):
        with self.assertRaises(SpecError):
            ScoreSpec(
                question_id="bad", instructions="Rate the urgency of this ticket on the rubric.",
                rubric_levels={1: "Routine no impact.", 3: "Critical impact right now today."},
            ).compile()

    def test_too_few_levels_rejected(self):
        with self.assertRaises(SpecError):
            ScoreSpec(
                question_id="bad", instructions="Rate the urgency of this ticket on the rubric.",
                rubric_levels={1: "Only a single level here."},
            ).compile()


class NoulSpecTests(unittest.TestCase):
    def test_valid(self):
        spec = NoulSpec(
            question_id="gate",
            instructions="Does the request exceed the documented spending limit?",
            true_criteria="The amount is strictly greater than the documented limit.",
            false_criteria="The amount is within the documented spending limit.",
        )
        gi = spec.compile().gate_inputs
        self.assertEqual(gi.noul_true_edge, 0.75)
        self.assertEqual(gi.noul_false_edge, 0.35)

    def test_invalid_band_rejected(self):
        with self.assertRaises(SpecError):
            NoulSpec(
                question_id="gate",
                instructions="Does the request exceed the documented spending limit?",
                true_criteria="The amount is strictly greater than the limit.",
                false_criteria="The amount is within the limit.",
                false_threshold=0.8,
                true_threshold=0.6,
            ).compile()


class DependencyTests(unittest.TestCase):
    def test_single_and_multi_conditions(self):
        spec = ChoiceSpec(
            question_id="child", instructions="Which follow-up lane should be executed now?",
            options={"x": "one meaningful criteria line here", "y": "another meaningful line of prose"},
            depends_on=("parent", "true"),
        )
        self.assertEqual(spec.dependencies(), (("parent", "true"),))
        spec2 = ChoiceSpec(
            question_id="child2", instructions="Which follow-up lane should be executed now?",
            options={"x": "one meaningful criteria line here", "y": "another meaningful line of prose"},
            depends_on=[("a", "true"), ("b", "false")],
        )
        self.assertEqual(spec2.dependencies(), (("a", "true"), ("b", "false")))

    def test_values_match(self):
        self.assertTrue(values_match(True, "true"))
        self.assertTrue(values_match(False, "false"))
        self.assertTrue(values_match(2.0, "2"))
        self.assertTrue(values_match("agent_review", "agent_review"))
        self.assertFalse(values_match(True, "True"))
        self.assertFalse(values_match("auto", "review"))


class PolicyDefaults(unittest.TestCase):
    def test_documented_defaults(self):
        policy = GatePolicy()
        self.assertEqual(policy.min_top1_prob, 0.65)
        self.assertEqual(policy.min_top2_margin, 0.20)
        self.assertEqual(policy.max_normalized_entropy, 0.55)
        self.assertEqual(policy.max_score_mode_aad, 0.25)
        self.assertEqual(policy.temperature, 1.0)
        self.assertFalse(policy.debiased_permutation)


if __name__ == "__main__":
    unittest.main()
