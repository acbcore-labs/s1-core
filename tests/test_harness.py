"""Harness orchestration tests: waves, dependency gating, loop guard,
debiasing, fail-closed invariants, and byte-identical determinism."""

import unittest

from s1core import (
    ABSTAIN_INSUFFICIENT_EVIDENCE,
    ChoiceSpec,
    DecisionVerdict,
    EpistemicStatus,
    GatePolicy,
    InMemoryAuditLedger,
    JsonlAuditLedger,
    NoulSpec,
    ScoreSpec,
    SpecError,
    System1Harness,
    verify_chain,
)
from s1core.backends import LexicalHeuristicDriver

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fixtures import ABSTAIN_DIST, ScriptedDriver


def lane_spec(policy=None):
    return ChoiceSpec(
        question_id="lane",
        instructions="Which handling lane does this ticket belong in?",
        options={
            "fast": "Automated fast-track handling with no human contact needed.",
            "slow": "Human agent reviews everything before any action is taken.",
        },
        policy=policy or GatePolicy(),
    )


def followup_spec(deps):
    return NoulSpec(
        question_id="followup",
        instructions="Should the slow lane trigger a courtesy callback to the customer?",
        true_criteria="The customer explicitly asked to be called back by phone.",
        false_criteria="The customer did not ask for any callback contact.",
        depends_on=deps,
    )


class HarnessTests(unittest.TestCase):
    def test_all_questions_get_a_verdict(self):
        harness = System1Harness(ScriptedDriver({"lane": {"fast": 0.8, "slow": 0.2}}))
        verdicts = harness.evaluate({}, [lane_spec()])
        self.assertIn("lane", verdicts)
        self.assertEqual(verdicts["lane"].status, EpistemicStatus.RATIFIED)
        self.assertEqual(verdicts["lane"].value, "fast")

    def test_fail_closed_invariant_value_none_unless_ratified(self):
        harness = System1Harness(
            ScriptedDriver({"lane": {"fast": 0.45, "slow": 0.35, ABSTAIN_INSUFFICIENT_EVIDENCE: 0.20}})
        )
        verdicts = harness.evaluate({}, [lane_spec()])
        v = verdicts["lane"]
        self.assertEqual(v.status, EpistemicStatus.ABSTAINED_GATE)
        self.assertIsNone(v.value)
        for verdict in verdicts.values():
            if verdict.status is not EpistemicStatus.RATIFIED:
                self.assertIsNone(verdict.value)

    def test_dependency_run_when_parent_matches(self):
        driver = ScriptedDriver(
            {"lane": {"fast": 0.9, "slow": 0.1}, "followup": {"true": 0.9, "false": 0.1}}
        )
        harness = System1Harness(driver)
        verdicts = harness.evaluate({}, [lane_spec(), followup_spec(("lane", "fast"))])
        self.assertEqual(verdicts["followup"].status, EpistemicStatus.RATIFIED)
        self.assertIs(verdicts["followup"].value, True)
        # Wave 0: lane; Wave 1: followup
        self.assertEqual(driver.calls, [["lane"], ["followup"]])

    def test_dependency_skipped_when_parent_unmatched(self):
        driver = ScriptedDriver(
            {"lane": {"fast": 0.9, "slow": 0.1}, "followup": {"true": 0.9, "false": 0.1}}
        )
        harness = System1Harness(driver)
        verdicts = harness.evaluate({}, [lane_spec(), followup_spec(("lane", "slow"))])
        self.assertEqual(verdicts["followup"].status, EpistemicStatus.SKIPPED_DEPENDENCY)
        self.assertIsNone(verdicts["followup"].value)
        # followup never reached the model
        self.assertEqual(driver.calls, [["lane"]])

    def test_dependency_skipped_when_parent_abstained(self):
        driver = ScriptedDriver(
            {"lane": dict(ABSTAIN_DIST), "followup": {"true": 0.9, "false": 0.1}}
        )
        harness = System1Harness(driver)
        verdicts = harness.evaluate({}, [lane_spec(), followup_spec(("lane", "fast"))])
        self.assertEqual(verdicts["lane"].status, EpistemicStatus.ABSTAINED_MODEL)
        self.assertEqual(verdicts["followup"].status, EpistemicStatus.SKIPPED_DEPENDENCY)

    def test_loop_guard_trips_after_max_consecutive_abstentions(self):
        harness = System1Harness(ScriptedDriver({"lane": dict(ABSTAIN_DIST)}))
        spec = lane_spec()
        for expected_status in (
            EpistemicStatus.ABSTAINED_MODEL,
            EpistemicStatus.ABSTAINED_MODEL,
        ):
            verdicts = harness.evaluate({}, [spec], stream_id="s")
            self.assertEqual(verdicts["lane"].status, expected_status)
        verdicts = harness.evaluate({}, [spec], stream_id="s", max_consecutive_abstentions=3)
        self.assertEqual(verdicts["lane"].status, EpistemicStatus.LOOP_GUARD_TRIPPED)
        self.assertIsNone(verdicts["lane"].value)
        # raw telemetry survives the trip
        self.assertEqual(verdicts["lane"].raw_value, ABSTAIN_INSUFFICIENT_EVIDENCE)

    def test_loop_guard_resets_on_ratified(self):
        scripted = {
            "abstain": dict(ABSTAIN_DIST),
            "clear": {"fast": 0.9, "slow": 0.1},
        }
        harness = System1Harness(ScriptedDriver(scripted))
        abstain = ChoiceSpec(
            question_id="abstain",
            instructions="Which handling lane does this ticket belong in exactly?",
            options={"fast": "one meaningful criteria line here",
                     "slow": "another meaningful line of prose"},
        )
        clear = ChoiceSpec(
            question_id="clear",
            instructions="Which handling lane does this ticket belong in exactly?",
            options={"fast": "one meaningful criteria line here",
                     "slow": "another meaningful line of prose"},
        )
        harness.evaluate({}, [abstain], stream_id="s")   # count 1
        harness.evaluate({}, [clear], stream_id="s")     # ratified: reset to 0
        harness.evaluate({}, [abstain], stream_id="s")   # count 1
        verdicts = harness.evaluate({}, [abstain], stream_id="s", max_consecutive_abstentions=3)
        # only 2 consecutive: the ratified call in between reset the counter
        self.assertEqual(verdicts["abstain"].status, EpistemicStatus.ABSTAINED_MODEL)

    def test_debiased_permutation_runs_second_call_and_merges(self):
        class OrderSensitiveDriver(ScriptedDriver):
            """Gives whichever option is listed first 0.7 of the mass — a
            positional-bias caricature."""

            def score_branches(self, handle, batch):
                self.calls.append(list(batch.keys()))
                out = {}
                for question_id, compiled in batch.items():
                    keys = list(compiled.criteria.keys())
                    if question_id in self._scripted:
                        out[question_id] = dict(self._scripted[question_id])
                        continue
                    dist = {k: 0.3 / (len(keys) - 1) for k in keys[1:]}
                    dist[keys[0]] = 0.7
                    out[question_id] = dist
                return out

        harness = System1Harness(OrderSensitiveDriver({}))
        spec = lane_spec(GatePolicy(min_top1_prob=0.0, min_top2_margin=0.0,
                                    max_normalized_entropy=1.0, debiased_permutation=True,
                                    max_permutation_divergence=1.0))
        verdicts = harness.evaluate({}, [spec])
        v = verdicts["lane"]
        self.assertIsNotNone(v.permutation_divergence)
        self.assertGreater(v.permutation_divergence, 0.0)
        # forward run boosts "fast"; reversed run boosts the last escape hatch;
        # the average leaves "fast" above "slow" but far below its 0.7 raw peak
        self.assertGreater(v.probabilities["fast"], v.probabilities["slow"])
        self.assertLess(v.probabilities["fast"], 0.7)

    def test_permutation_divergence_trips_gate(self):
        class OrderSensitiveDriver(ScriptedDriver):
            def score_branches(self, handle, batch):
                out = {}
                for question_id, compiled in batch.items():
                    keys = list(compiled.criteria.keys())
                    first = 0.9 if keys[0] == "fast" else 0.1
                    out[question_id] = {"fast": first, "slow": 1.0 - first}
                return out

        harness = System1Harness(OrderSensitiveDriver({}))
        spec = lane_spec(GatePolicy(min_top1_prob=0.0, min_top2_margin=0.0,
                                    max_normalized_entropy=1.0, debiased_permutation=True,
                                    max_permutation_divergence=0.05))
        verdicts = harness.evaluate({}, [spec])
        self.assertEqual(verdicts["lane"].status, EpistemicStatus.ABSTAINED_GATE)
        self.assertIsNone(verdicts["lane"].value)

    def test_determinism_byte_identical_records(self):
        driver = LexicalHeuristicDriver()
        questions = [
            lane_spec(),
            ScoreSpec(
                question_id="urgency",
                instructions="Rate the urgency of this ticket on the rubric provided.",
                rubric_levels={1: "Routine, no impact.", 2: "Elevated, some impact.",
                               3: "Critical impact right now."},
            ),
        ]
        state = {"ticket": "customer requests refund", "order_total": 120.0}
        ledgers = []
        for _ in range(2):
            ledger = InMemoryAuditLedger()
            harness = System1Harness(LexicalHeuristicDriver(), ledger)
            harness.evaluate(state, questions, reference_time_iso="2026-10-14T12:00:00+00:00")
            ledgers.append(ledger.records)
        for r1, r2 in zip(ledgers[0], ledgers[1]):
            j1, j2 = r1.to_jsonable(), r2.to_jsonable()
            # wall-clock stamps and the chain digests that cover them are
            # per-event; everything decision-bearing must be byte-identical
            for volatile in ("created_utc", "previous_digest", "record_digest"):
                j1.pop(volatile)
                j2.pop(volatile)
            self.assertEqual(j1, j2)
        # replay digests (state + compiled question + model + calibration)
        # must match exactly — that is the content address for replay
        self.assertEqual(
            [r.verdict["replay_digest"] for r in ledgers[0]],
            [r.verdict["replay_digest"] for r in ledgers[1]],
        )

    def test_jsonl_ledger_chain_and_verification(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "audit.jsonl")
            ledger = JsonlAuditLedger(path)
            driver = ScriptedDriver({"lane": {"fast": 0.8, "slow": 0.2}})
            harness = System1Harness(driver, ledger)
            harness.evaluate({}, [lane_spec()], reference_time_iso="2026-10-14T12:00:00+00:00")
            # second harness instance continues the chain from the file
            harness2 = System1Harness(driver, JsonlAuditLedger(path))
            harness2.evaluate({}, [lane_spec()], reference_time_iso="2026-10-14T12:00:00+00:00")
            with open(path, encoding="utf-8") as fh:
                lines = [line for line in fh.read().splitlines() if line]
            self.assertEqual(len(lines), 2)
            import json
            records = [json.loads(line) for line in lines]
            self.assertTrue(verify_chain(records))
            tampered = [dict(records[0])]
            tampered[0] = {**tampered[0], "raw_probabilities": {"fast": 0.99, "slow": 0.01}}
            self.assertFalse(verify_chain(tampered + [records[1]]))


class HarnessFailClosedTests(unittest.TestCase):
    def test_non_serializable_state_raises_spec_error(self):
        harness = System1Harness(ScriptedDriver({}))
        with self.assertRaises(SpecError):
            harness.evaluate({"bad": {1, 2, 3}}, [lane_spec()])

    def test_backend_omitting_a_question_is_a_runtime_error(self):
        class OmittingDriver(ScriptedDriver):
            def score_branches(self, handle, batch):
                return {}

        harness = System1Harness(OmittingDriver({}))
        with self.assertRaises(RuntimeError):
            harness.evaluate({}, [lane_spec()])


if __name__ == "__main__":
    unittest.main()
