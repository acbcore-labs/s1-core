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
        record["schema_version"] = "s1core.audit/9"
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


class SchemaV3LawTests(unittest.TestCase):
    """v3 records restate the declared output space and replay under the v3
    digest (full gate inputs); v1/v2 records keep historical semantics."""

    def ratified_record(self, primitive="choice"):
        from s1core import InMemoryAuditLedger, ScoreSpec, System1Harness
        from fixtures import ScriptedDriver
        if primitive == "choice":
            spec = ChoiceSpec(
                question_id="lane",
                instructions="Which handling lane does this ticket belong in?",
                options={"fast": "Automated fast-track handling, no human contact.",
                         "slow": "A human agent reviews before any action."},
            )
            scripted = {"lane": {"fast": 0.8, "slow": 0.2}}
        else:
            spec = ScoreSpec(
                question_id="urgency",
                instructions="Rate the urgency of this ticket on the rubric provided.",
                rubric_levels={1: "Routine backlog with no customer impact.",
                               2: "Standard queue handling within the day.",
                               3: "Critical; money at risk right now."},
            )
            scripted = {"urgency": {"1": 0.05, "2": 0.9, "3": 0.05}}
        ledger = InMemoryAuditLedger()
        harness = System1Harness(ScriptedDriver(scripted), ledger)
        harness.evaluate({"ticket": "t"}, [spec],
                         reference_time_iso="2026-10-14T12:00:00+00:00")
        record = ledger.records[0]
        self.assertEqual(
            DecisionVerdict.from_jsonable(record.verdict).status, EpistemicStatus.RATIFIED)
        return dict(record.to_jsonable())

    def test_v2_record_replays_under_the_legacy_digest(self):
        from s1core.audit import compute_replay_digest
        from s1core.types import GatePolicy
        record = self.ratified_record()
        record["schema_version"] = "s1core.audit/2"
        record.pop("spec_digest", None)
        record["verdict"]["replay_digest"] = compute_replay_digest(
            record["state_hash"], record["compiled_question"], record["model_id"],
            GatePolicy(**record["gate_inputs"]["policy"]))
        replayed = replay_verdict(record)
        self.assertEqual(replayed.to_jsonable(), record["verdict"])

    def test_v3_record_without_support_metadata_raises(self):
        record = self.ratified_record("choice")
        record["gate_inputs"].pop("declared_keys")
        with self.assertRaises(ValueError):
            replay_verdict(record)
        score_record = self.ratified_record("score")
        score_record["gate_inputs"].pop("score_levels")
        with self.assertRaises(ValueError):
            replay_verdict(score_record)

    def test_v3_record_with_disagreeing_support_raises(self):
        record = self.ratified_record("choice")
        record["gate_inputs"]["declared_keys"] = ["some_option_never_declared"]
        with self.assertRaises(ValueError):
            replay_verdict(record)
        score_record = self.ratified_record("score")
        score_record["gate_inputs"]["score_levels"] = 99
        with self.assertRaises(ValueError):
            replay_verdict(score_record)

    def test_unknown_schema_is_rejected_before_the_empty_raw_return(self):
        record = self.ratified_record()
        record["schema_version"] = "s1core.audit/9"
        record["raw_probabilities"] = {}
        with self.assertRaises(ValueError):
            replay_verdict(record)

    def _short_circuit_record(self):
        from s1core import ChoiceSpec, InMemoryAuditLedger, System1Harness
        from s1core.backends import LexicalHeuristicDriver
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).parent))
        from fixtures import ABSTAIN_DIST, ScriptedDriver
        spec = ChoiceSpec(
            question_id="abstain",
            instructions="Which handling lane does this ticket belong in exactly?",
            options={"fast": "one meaningful criteria line here",
                     "slow": "another meaningful line of prose"},
        )
        ledger = InMemoryAuditLedger()
        harness = System1Harness(ScriptedDriver({"abstain": dict(ABSTAIN_DIST)}), ledger)
        harness.evaluate({}, [spec], stream_id="s", max_consecutive_abstentions=1)
        harness.evaluate({}, [spec], stream_id="s", max_consecutive_abstentions=1)
        return next(dict(r.to_jsonable()) for r in ledger.records
                    if not r.raw_probabilities
                    and r.verdict["status"] == "LOOP_GUARD_TRIPPED")

    def test_short_circuit_record_replays(self):
        record = self._short_circuit_record()
        replayed = replay_verdict(record)
        self.assertEqual(replayed.to_jsonable(), record["verdict"])

    def test_short_circuit_contradictions_raise(self):
        base = self._short_circuit_record()
        cases = []
        missing_flag = {**base, "postprocessing": dict(base["postprocessing"],
                                                       loop_guard_short_circuit=False)}
        cases.append(missing_flag)
        ratified = {**base, "verdict": dict(base["verdict"], status="RATIFIED", value="fast")}
        cases.append(ratified)
        valued = {**base, "verdict": dict(base["verdict"], value="fast")}
        cases.append(valued)
        distributed = {**base, "verdict": dict(base["verdict"],
                                               probabilities={"fast": 0.9, "slow": 0.1})}
        cases.append(distributed)
        diverged = {**base, "postprocessing": dict(base["postprocessing"],
                                                   permutation_divergence=0.5)}
        cases.append(diverged)
        unlatched = {**base, "postprocessing": dict(base["postprocessing"],
                                                    loop_guard_tripped=False)}
        cases.append(unlatched)
        for record in cases:
            with self.subTest(verdict=record["verdict"]["status"],
                              post=record["postprocessing"]):
                with self.assertRaises(ValueError):
                    replay_verdict(record)

    def test_short_circuit_flag_on_an_inference_record_raises(self):
        record = self.ratified_record()
        record["postprocessing"] = dict(record["postprocessing"],
                                        loop_guard_short_circuit=True)
        with self.assertRaises(ValueError):
            replay_verdict(record)

    def test_short_circuit_flag_must_be_a_boolean(self):
        for flag in (1, "yes"):
            with self.subTest(flag=flag):
                record = self.ratified_record()
                record["postprocessing"] = dict(record["postprocessing"],
                                                loop_guard_short_circuit=flag)
                with self.assertRaises(ValueError):
                    replay_verdict(record)

    def _skip_record(self):
        """A genuine v3 SKIPPED_DEPENDENCY record: the dependency abstains,
        so the dependent question never reaches the backend."""
        from fixtures import ABSTAIN_DIST, ScriptedDriver
        first = ChoiceSpec(
            question_id="lane",
            instructions="Which handling lane does this ticket belong in?",
            options={"fast": "Automated fast-track handling, no human contact.",
                     "slow": "A human agent reviews before any action."},
        )
        second = ChoiceSpec(
            question_id="exceeds",
            instructions="Does the requested refund exceed the order total?",
            options={"yes": "The refund is larger than the order total.",
                     "no": "The refund is within the order total."},
            depends_on=(("lane", "fast"),),
        )
        ledger = InMemoryAuditLedger()
        harness = System1Harness(
            ScriptedDriver({"lane": dict(ABSTAIN_DIST),
                            "exceeds": {"yes": 0.9, "no": 0.1}}), ledger)
        harness.evaluate({}, [first, second],
                         reference_time_iso="2026-10-14T12:00:00+00:00")
        record = next(r for r in ledger.records
                      if r.verdict["status"] == "SKIPPED_DEPENDENCY")
        self.assertFalse(record.raw_probabilities)
        return dict(record.to_jsonable())

    def test_v3_dependency_skip_requires_support_metadata(self):
        missing = self._skip_record()
        missing["gate_inputs"].pop("declared_keys")
        with self.assertRaises(ValueError):
            replay_verdict(missing)
        disagreeing = self._skip_record()
        disagreeing["gate_inputs"]["declared_keys"] = ["some_option_never_declared"]
        with self.assertRaises(ValueError):
            replay_verdict(disagreeing)

    def test_v3_short_circuit_requires_support_metadata(self):
        missing = self._short_circuit_record()
        missing["gate_inputs"].pop("declared_keys")
        with self.assertRaises(ValueError):
            replay_verdict(missing)
        disagreeing = self._short_circuit_record()
        disagreeing["gate_inputs"]["declared_keys"] = ["some_option_never_declared"]
        with self.assertRaises(ValueError):
            replay_verdict(disagreeing)

    def test_v3_record_must_address_its_own_content(self):
        tampered_short = self._short_circuit_record()
        tampered_short["verdict"]["replay_digest"] = "0" * 32
        with self.assertRaises(ValueError):
            replay_verdict(tampered_short)
        tampered_inference = self.ratified_record()
        tampered_inference["verdict"]["replay_digest"] = "0" * 32
        with self.assertRaises(ValueError):
            replay_verdict(tampered_inference)


if __name__ == "__main__":
    unittest.main()
