"""Driver-contract tests: the lexical reference backend and the shadow tee."""

import unittest
from contextlib import contextmanager

from s1core import ChoiceSpec, GatePolicy, System1Harness
from s1core.backends import BackendError, LexicalHeuristicDriver, ShadowTeeDriver
from s1core.backends.base import StateHandle
from s1core.backends.lexical import state_tokens

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fixtures import ScriptedDriver


def lane_spec():
    return ChoiceSpec(
        question_id="lane",
        instructions="Which handling lane does this ticket belong in?",
        options={
            "fast": "Automated fast-track handling with no human contact needed.",
            "slow": "Human agent reviews everything before any action is taken.",
        },
    )


class StateTokenTests(unittest.TestCase):
    def test_nested_collections_and_scalars(self):
        state = {
            "ticket": "wants refund",
            "meta": {"tags": ["vip", "urgent"], "ids": frozenset({"x"})},
            "quantity": 3,
            "note": None,
            "flag": True,
        }
        tokens = state_tokens(state)
        # top-level keys and all string leaves are tokenized; nested mapping
        # keys ("ids") are not — only values travel through _iter_strings
        for expected in ("ticket", "wants", "refund", "vip", "urgent", "meta", "x", "quantity"):
            self.assertIn(expected, tokens)
        self.assertNotIn("ids", tokens)
        # None and bool leaves contribute nothing (not even "true"/"none")
        self.assertNotIn("none", tokens)
        self.assertNotIn("true", tokens)

    def test_empty_state_has_empty_vocabulary(self):
        self.assertEqual(state_tokens({}), frozenset())


class LexicalHeuristicDriverTests(unittest.TestCase):
    def test_empty_state_yields_uniform_distribution(self):
        # No state vocabulary -> every option scores Dice 0 -> uniform softmax.
        verdicts = System1Harness(LexicalHeuristicDriver()).evaluate({}, [lane_spec()])
        probs = verdicts["lane"].probabilities
        self.assertAlmostEqual(len({round(p, 12) for p in probs.values()}), 1)


class StubDriver:
    """Minimal driver double that records pin_state and returns fixed mass."""

    model_id = "stub"

    def __init__(self, mass=0.5):
        self.mass = mass
        self.pinned: list[str] = []

    @contextmanager
    def pin_state(self, canonical_state, state_hash):
        self.pinned.append(state_hash)
        yield StateHandle(state_hash=state_hash, canonical_state=canonical_state)

    def score_branches(self, handle, batch):
        return {qid: {"a": self.mass, "b": 1.0 - self.mass} for qid in batch}


class ShadowTeeDriverTests(unittest.TestCase):
    def setUp(self):
        self.primary = StubDriver(mass=0.75)
        self.shadow = StubDriver(mass=0.25)
        self.seen: list[tuple] = []
        self.tee = ShadowTeeDriver(
            self.primary, self.shadow, rate=1.0,
            on_shadow=lambda *args: self.seen.append(args),
        )

    def test_rate_must_be_a_probability(self):
        for rate in (-0.1, 1.5):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                ShadowTeeDriver(self.primary, self.shadow, rate=rate, on_shadow=lambda *a: None)

    def test_pin_state_delegates_to_primary(self):
        with self.tee.pin_state({"s": 1}, "hash-1") as handle:
            self.assertEqual(handle.state_hash, "hash-1")
        self.assertEqual(self.primary.pinned, ["hash-1"])
        self.assertEqual(self.shadow.pinned, [])

    def test_rate_zero_never_shadows(self):
        tee = ShadowTeeDriver(self.primary, self.shadow, rate=0.0,
                              on_shadow=lambda *args: self.seen.append(args))
        handle = StateHandle(state_hash="h", canonical_state={"s": 1})
        batch = {"q": object()}
        out = tee.score_branches(handle, batch)
        self.assertEqual(out, {"q": {"a": 0.75, "b": 0.25}})
        self.assertEqual(self.shadow.pinned, [])
        self.assertEqual(self.seen, [])

    def test_rate_one_always_shadows_and_reports(self):
        handle = StateHandle(state_hash="h", canonical_state={"s": 1})
        batch = {"q": object()}
        out = self.tee.score_branches(handle, batch)
        self.assertEqual(out, {"q": {"a": 0.75, "b": 0.25}})  # primary answer wins
        self.assertEqual(self.shadow.pinned, ["h"])
        self.assertEqual(len(self.seen), 1)
        state_hash, seen_batch, shadow_results, primary_results = self.seen[0]
        self.assertEqual(state_hash, "h")
        self.assertIs(seen_batch, batch)
        self.assertEqual(shadow_results, {"q": {"a": 0.25, "b": 0.75}})
        self.assertEqual(primary_results, {"q": {"a": 0.75, "b": 0.25}})

    def test_end_to_end_through_harness(self):
        # The tee is a drop-in driver: the harness cannot tell it is wrapped.
        tee = ShadowTeeDriver(
            ScriptedDriver({"lane": {"fast": 0.9, "slow": 0.1}}), StubDriver(),
            rate=1.0, on_shadow=lambda *args: self.seen.append(args),
        )
        verdicts = System1Harness(tee).evaluate({}, [lane_spec()])
        self.assertEqual(verdicts["lane"].value, "fast")
        self.assertEqual(len(self.seen), 1)


class TypeSafeParserTests(unittest.TestCase):
    """The wire adapter absorbs documented differences (zero-filling omitted
    declared options) and rejects anything the question never declared."""

    def compiled(self):
        return lane_spec().compile()

    def body(self, probabilities):
        return {"answers": {"lane": {"type": "choice", "probabilities": probabilities}}}

    def test_declared_options_zero_fill(self):
        from s1core.backends.typesafe import typesafe_response_parser
        parsed = typesafe_response_parser(self.body({"fast": 0.9}), {"lane": self.compiled()})
        # every declared option — the escape hatches included — is filled
        self.assertEqual(parsed["lane"]["fast"], 0.9)
        self.assertEqual(parsed["lane"]["slow"], 0.0)
        self.assertEqual(set(parsed["lane"]), set(self.compiled().criteria))

    def test_undeclared_wire_key_is_rejected(self):
        from s1core.backends.typesafe import typesafe_response_parser
        with self.assertRaises(BackendError):
            typesafe_response_parser(
                self.body({"fast": 0.9, "slow": 0.1, "stray_option": 0.5}),
                {"lane": self.compiled()})


if __name__ == "__main__":
    unittest.main()
