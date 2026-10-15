"""Replay ledger tests: every recorded verdict must re-derive bit-for-bit
from its record (wrong == 0), across all statuses and primitives."""

import unittest
import random

from s1core import (
    ChoiceSpec,
    DecisionVerdict,
    EpistemicStatus,
    InMemoryAuditLedger,
    NoulSpec,
    ScoreSpec,
    System1Harness,
    replay_verdict,
    verify_chain,
)
from s1core.backends import LexicalHeuristicDriver

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fixtures import ScriptedDriver

REPLAY_FIELDS = (
    "status", "value", "raw_value", "probabilities", "confidence",
    "top2_margin", "normalized_entropy", "score_mode_deviation",
    "conformal_set_size", "permutation_divergence", "replay_digest",
)


class ReplayTests(unittest.TestCase):
    def build_records(self):
        ledger = InMemoryAuditLedger()
        harness = System1Harness(LexicalHeuristicDriver(), ledger)
        questions = [
            ChoiceSpec(
                question_id="lane",
                instructions="Which handling lane does this refund ticket belong in?",
                options={
                    "auto_approve": "Refund is within policy limits and history supports approval.",
                    "agent_review": "A human agent must review the refund amount before approval.",
                },
            ),
            ScoreSpec(
                question_id="urgency",
                instructions="Rate the urgency of this refund ticket on the rubric provided.",
                rubric_levels={
                    1: "Routine backlog with no customer impact.",
                    2: "Standard queue handling within the day.",
                    3: "Critical; money at risk right now.",
                },
            ),
            NoulSpec(
                question_id="exceeds",
                instructions="Does the requested refund exceed the full order total?",
                true_criteria="The requested refund is greater than the order total.",
                false_criteria="The requested refund is within the order total.",
                depends_on=("lane", "agent_review"),
            ),
        ]
        state = {
            "ticket": "customer requests refund",
            "order_total": 120.0,
            "requested_refund": 150.0,
        }
        harness.evaluate(state, questions, reference_time_iso="2026-10-14T12:00:00+00:00")
        return harness, ledger.records

    def test_every_record_replays_exactly(self):
        harness, records = self.build_records()
        self.assertTrue(records)
        statuses_seen = set()
        for record in records:
            replayed = replay_verdict(record.to_jsonable())
            stored = DecisionVerdict.from_jsonable(record.verdict)
            statuses_seen.add(stored.status)
            for field_name in REPLAY_FIELDS:
                self.assertEqual(
                    getattr(replayed, field_name),
                    getattr(stored, field_name),
                    f"replay mismatch on {field_name} for {record.compiled_question['question_id']}",
                )
        # the scenario must cover ratified AND non-ratified paths
        self.assertIn(EpistemicStatus.RATIFIED, statuses_seen)
        self.assertTrue(statuses_seen - {EpistemicStatus.RATIFIED})

    def test_chain_survives_json_round_trip(self):
        import json
        _, records = self.build_records()
        serialized = [json.loads(json.dumps(r.to_jsonable())) for r in records]
        self.assertTrue(verify_chain(serialized))

    def test_tampered_distribution_changes_verdict_and_breaks_chain(self):
        _, records = self.build_records()
        target = None
        for record in records:
            if record.raw_probabilities:
                target = dict(record.to_jsonable())
                break
        self.assertIsNotNone(target)
        original = records[0]
        # flip the raw distribution: replay must disagree with the stored verdict
        keys = list(target["raw_probabilities"])
        target["raw_probabilities"] = {k: (0.02 if k == keys[0] else 0.98) for k in keys} \
            if len(keys) == 2 else {k: 1.0 / len(keys) for k in keys}
        replayed = replay_verdict(target)
        stored = DecisionVerdict.from_jsonable(
            next(r for r in records
                 if r.compiled_question["question_id"] == target["compiled_question"]["question_id"]).verdict
        )
        differs = (
            replayed.status != stored.status
            or replayed.value != stored.value
            or abs(replayed.confidence - stored.confidence) > 1e-12
        )
        self.assertTrue(differs, "tampered raw distribution must not replay to the stored verdict")

    def test_randomized_replay_sweep(self):
        rng = random.Random(20261014)
        for trial in range(50):
            n = rng.randint(2, 6)
            raw = {f"opt_{i}": rng.random() + 0.01 for i in range(n)}
            spec = ChoiceSpec(
                question_id=f"q{trial}",
                instructions=f"Synthetic question {trial} with full self-contained instructions.",
                options={f"opt_{i}": f"criteria prose number {i} in plain words" for i in range(n)},
            )
            ledger = InMemoryAuditLedger()
            harness = System1Harness(ScriptedDriver({f"q{trial}": raw}, model_id="sweep"), ledger)
            harness.evaluate({}, [spec], reference_time_iso="2026-10-14T00:00:00+00:00")
            record = ledger.records[0]
            replayed = replay_verdict(record.to_jsonable())
            stored = DecisionVerdict.from_jsonable(record.verdict)
            self.assertEqual(replayed.status, stored.status)
            self.assertEqual(replayed.value, stored.value)
            self.assertEqual(replayed.replay_digest, stored.replay_digest)


class ReplayEventValidationTests(unittest.TestCase):
    """Post-gate events recorded in v2 records must be coherent with the gate."""

    def setUp(self):
        _, self.records = ReplayTests(
            "test_every_record_replays_exactly"
        ).build_records()

    def ratified_record(self):
        return next(
            dict(r.to_jsonable()) for r in self.records
            if r.raw_probabilities
            and DecisionVerdict.from_jsonable(r.verdict).status is EpistemicStatus.RATIFIED
        )

    def test_unsupported_schema_version(self):
        record = self.ratified_record()
        record["schema_version"] = "s1core.audit/3"
        with self.assertRaises(ValueError):
            replay_verdict(record)

    def test_impossible_recorded_divergence(self):
        for divergence in (1.5, -0.5, float("nan")):
            with self.subTest(divergence=divergence):
                record = self.ratified_record()
                record["postprocessing"] = {
                    "permutation_divergence": divergence, "loop_guard_tripped": False,
                }
                with self.assertRaises(ValueError):
                    replay_verdict(record)

    def test_loop_guard_event_requires_an_abstained_gate_result(self):
        record = self.ratified_record()
        record["postprocessing"] = {
            "permutation_divergence": None, "loop_guard_tripped": True,
        }
        with self.assertRaises(ValueError):
            replay_verdict(record)


if __name__ == "__main__":
    unittest.main()
