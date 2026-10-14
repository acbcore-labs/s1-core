"""Combiner tests: fail-closed behavior and probability semantics."""

import unittest

from s1core import (
    DecisionVerdict,
    EpistemicStatus,
    LatticeCombiner,
    NoisyANDCombiner,
    signal_probability,
)


def verdict(value, status=EpistemicStatus.RATIFIED, probabilities=None, digest="d"):
    return DecisionVerdict(
        question_id="q",
        status=status,
        value=value,
        raw_value=value,
        probabilities=probabilities or {},
        confidence=0.9,
        top2_margin=0.5,
        normalized_entropy=0.1,
        replay_digest=digest,
    )


class SignalProbabilityTests(unittest.TestCase):
    def test_choice_value_lookup(self):
        v = verdict("a", probabilities={"a": 0.8, "b": 0.2})
        self.assertAlmostEqual(signal_probability(v), 0.8)

    def test_bool_uses_noul_keys(self):
        v = verdict(True, probabilities={"true": 0.93, "false": 0.07})
        self.assertAlmostEqual(signal_probability(v), 0.93)
        v_false = verdict(False, probabilities={"true": 0.93, "false": 0.07})
        self.assertAlmostEqual(signal_probability(v_false), 0.07)

    def test_unratified_is_zero(self):
        v = verdict(None, status=EpistemicStatus.ABSTAINED_GATE)
        self.assertEqual(signal_probability(v), 0.0)


class LatticeCombinerTests(unittest.TestCase):
    def setUp(self):
        self.combiner = LatticeCombiner(
            "merge",
            required=["tests", "safe"],
            rule=lambda vs: (vs["tests"].value is True) and (vs["safe"].value == "yes"),
        )

    def test_combined_when_all_ratified(self):
        verdicts = {
            "tests": verdict(True, probabilities={"true": 0.95, "false": 0.05}),
            "safe": verdict("yes", probabilities={"yes": 0.9, "no": 0.1}),
        }
        combined = self.combiner.combine(verdicts)
        self.assertEqual(combined.status, EpistemicStatus.RATIFIED)
        self.assertIs(combined.value, True)

    def test_fails_closed_on_abstained_signal(self):
        verdicts = {
            "tests": verdict(True, probabilities={"true": 0.95, "false": 0.05}),
            "safe": verdict(None, status=EpistemicStatus.ABSTAINED_GATE),
        }
        combined = self.combiner.combine(verdicts)
        self.assertEqual(combined.status, EpistemicStatus.ABSTAINED_GATE)
        self.assertIsNone(combined.value)

    def test_fails_closed_on_missing_signal(self):
        combined = self.combiner.combine({"tests": verdict(True)})
        self.assertEqual(combined.status, EpistemicStatus.ABSTAINED_GATE)

    def test_rule_exception_is_a_defect_not_an_abstention(self):
        from s1core import CombinerError

        boom = LatticeCombiner("boom", ["tests"], lambda vs: 1 / 0)
        with self.assertRaises(CombinerError):
            boom.combine({"tests": verdict(True)})


class NoisyANDTests(unittest.TestCase):
    def test_joint_probability(self):
        combiner = NoisyANDCombiner("and", ["a", "b"])
        combined = combiner.combine(
            {
                "a": verdict(True, probabilities={"true": 0.8, "false": 0.2}),
                "b": verdict(True, probabilities={"true": 0.5, "false": 0.5}),
            }
        )
        self.assertAlmostEqual(combined.value, 0.4)


if __name__ == "__main__":
    unittest.main()
