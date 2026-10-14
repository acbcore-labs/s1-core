"""Wave-DAG compilation tests."""

import unittest

from s1core import ChoiceSpec, NoulSpec, ScoreSpec, SpecError, compile_waves


def choice(qid, deps=None):
    return ChoiceSpec(
        question_id=qid,
        instructions=f"Question {qid} instructions in a full self-contained sentence.",
        options={"a": "one meaningful criteria line here", "b": "another meaningful line of prose"},
        depends_on=deps,
    )


def noul(qid, deps=None):
    return NoulSpec(
        question_id=qid,
        instructions=f"Question {qid} instructions in a full self-contained sentence.",
        true_criteria="The affirmative criteria described in plain words.",
        false_criteria="The negative criteria described in plain words.",
        depends_on=deps,
    )


class CompileWavesTests(unittest.TestCase):
    def test_flat_questions_single_wave(self):
        waves = compile_waves([choice("a"), choice("b"), noul("c")])
        self.assertEqual(len(waves), 1)
        self.assertEqual([s.question_id for s in waves[0]], ["a", "b", "c"])

    def test_layering(self):
        waves = compile_waves([choice("c", ("b", "a")), choice("b", ("a", "true")), choice("a")])
        self.assertEqual([[s.question_id for s in w] for w in waves], [["a"], ["b"], ["c"]])

    def test_diamond(self):
        waves = compile_waves(
            [
                choice("top"),
                choice("left", ("top", "a")),
                choice("right", ("top", "a")),
                noul("bottom", [("left", "true"), ("right", "true")]),
            ]
        )
        self.assertEqual([[s.question_id for s in w] for w in waves],
                         [["top"], ["left", "right"], ["bottom"]])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(SpecError):
            compile_waves([choice("a"), choice("a")])

    def test_unknown_parent_rejected(self):
        with self.assertRaises(SpecError):
            compile_waves([choice("a", ("ghost", "true"))])

    def test_self_dependency_rejected(self):
        with self.assertRaises(SpecError):
            compile_waves([choice("a", ("a", "true"))])

    def test_cycle_rejected(self):
        with self.assertRaises(SpecError):
            compile_waves([choice("a", ("b", "true")), choice("b", ("a", "true"))])


if __name__ == "__main__":
    unittest.main()
