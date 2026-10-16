"""Focused hardening regressions; no live backend required."""
import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from s1core.audit import (JsonlAuditLedger, make_record, replay_verdict,
                          compute_replay_digest, compute_replay_digest_v3, verify_chain)
from s1core.calibration import conformal_tau, fit_gate_policy
from s1core.combiners import CombinerError, LatticeCombiner, NoisyANDCombiner
from s1core.gates import GateInputs, evaluate, normalize, temperature_scale
from s1core.hashing import digest_of
from s1core.types import GatePolicy, EpistemicStatus

class HardeningTests(unittest.TestCase):
    def record(self, primitive="choice", raw=None, divergence=None, loop=False):
        raw = raw if raw is not None else {"a": .99, "b": .01}
        if primitive == "choice":
            gi = GateInputs("q", primitive, GatePolicy(), declared_keys=frozenset(raw))
            criteria = {key: f"criteria prose for {key}" for key in raw}
        elif primitive == "score":
            gi = GateInputs("q", primitive, GatePolicy(), score_levels=len(raw))
            criteria = {str(level): f"rubric prose for level {level}" for level in raw}
        else:
            gi = GateInputs("q", primitive, GatePolicy())
            criteria = dict(raw)
        state_hash = digest_of({})
        payload = {"question_id": "q", "type": primitive, "criteria": criteria}
        v = evaluate(gi, raw, compute_replay_digest_v3(state_hash, payload, "test", gi))
        if divergence is not None:
            v = dataclasses.replace(v, permutation_divergence=divergence)
            if divergence > gi.policy.max_permutation_divergence:
                v = dataclasses.replace(v, status=EpistemicStatus.ABSTAINED_GATE, value=None)
        if loop:
            v = dataclasses.replace(v, status=EpistemicStatus.LOOP_GUARD_TRIPPED, value=None)
        return make_record(stream_id="test", wave_index=0, model_id="test", state_hash=state_hash,
            canonical_state={}, compiled_payload=payload, gate_inputs=gi, raw_probabilities=raw, verdict=v)

    def test_invalid_weights_and_extremes(self):
        for raw in ({}, {"a": 0}, {"a": 2, "b": -1}, {"a": math.nan}, {"a": math.inf}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize(raw)
        self.assertEqual(normalize({"a": 1e308, "b": 1e308}), {"a": .5, "b": .5})
        self.assertEqual(temperature_scale({"a": .6, "b": .4}, 1e-300), {"a": 1., "b": 0.})
        for t in (0, -1, math.nan, math.inf):
            with self.subTest(t=t), self.assertRaises(ValueError):
                temperature_scale({"a": 1.}, t)

    def test_bad_support_and_policy(self):
        for primitive, raw in (("noul", {"true": .9, "other": .1}), ("score", {"1": .5, "3": .5})):
            with self.subTest(primitive=primitive), self.assertRaises(ValueError):
                evaluate(GateInputs("q", primitive, GatePolicy()), raw, "")
        with self.assertRaises(ValueError):
            evaluate(GateInputs("q", "choice", GatePolicy(min_top1_prob=math.nan)), {"a": 1.}, "")
        # the noul handler keeps its own exact-keys law for direct callers
        from s1core.gates import evaluate_noul
        with self.assertRaises(ValueError):
            evaluate_noul(GateInputs("q", "noul", GatePolicy()), {"true": 1.0}, "")

    def test_post_gate_replay_and_legacy(self):
        for options in ({"divergence": .7}, {"divergence": .01},
                        {"raw": {"a": .5, "b": .5}, "loop": True}, {"divergence": .7, "loop": True}):
            r = self.record(**options).to_jsonable()
            self.assertEqual(replay_verdict(r).to_jsonable(), r["verdict"])
            # Downgrade to a faithful v1 record: historical schema, no
            # postprocessing map, and a verdict digest computed with the
            # legacy (policy-only) algorithm the v1 replay path re-derives.
            r["schema_version"] = "s1core.audit/1"
            r.pop("postprocessing")
            r["verdict"]["replay_digest"] = compute_replay_digest(
                r["state_hash"], r["compiled_question"], "test",
                GatePolicy(**r["gate_inputs"]["policy"]))
            self.assertEqual(replay_verdict(r).to_jsonable(), r["verdict"])

    def test_replay_recomputes_and_requires_raw(self):
        r = self.record().to_jsonable()
        r["verdict"]["value"] = "forged"
        self.assertEqual(replay_verdict(r).value, "a")
        r["raw_probabilities"] = {}
        with self.assertRaises(ValueError):
            replay_verdict(r)

    def test_chain_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            JsonlAuditLedger(path).record(self.record())
            JsonlAuditLedger(path).record(self.record(divergence=.7))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(verify_chain(rows))
            rows[0]["verdict"]["value"] = "forged"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaises(ValueError):
                JsonlAuditLedger(path)

    def test_conformal_rank(self):
        self.assertEqual(conformal_tau([.9], .1), 0.)
        self.assertEqual(conformal_tau([.1, .2, .3, .4], .3), .1)
        population = [i / 11 for i in range(1, 11)]
        for alpha in (.01, .1, .15, .3, .6):
            misses = sum(value < conformal_tau(population[:i] + population[i+1:], alpha)
                         for i, value in enumerate(population))
            self.assertLessEqual(misses / len(population), alpha)

    def test_disjoint_fitting(self):
        samples = [({"a": p, "b": 1-p}, "a") for p in (.6, .7, .8, .9)]
        with patch("s1core.calibration.fit_temperature", return_value=1.) as fit:
            policy = fit_gate_policy(samples, alpha=.5)
        fit.assert_called_once_with(samples[:2])
        self.assertAlmostEqual(policy.conformal_tau, .8)
        with patch("s1core.calibration.fit_temperature", return_value=1.) as fit:
            policy = fit_gate_policy(samples[:1], calibration_samples=samples[1:], alpha=.5)
        fit.assert_called_once_with(samples[:1])
        self.assertAlmostEqual(policy.conformal_tau, .8)

    def test_combiner_semantics(self):
        v = replay_verdict(self.record("score", {"1": .01, "2": .99}).to_jsonable())
        with self.assertRaises(CombinerError):
            NoisyANDCombiner("test", ["q"]).combine({"q": v})
        self.assertTrue(LatticeCombiner("test", ["q"], lambda vs: vs["q"].value > 1.5).combine({"q": v}).value)
        v = replay_verdict(self.record("noul", {"true": .01, "false": .99}).to_jsonable())
        self.assertAlmostEqual(NoisyANDCombiner("test", ["q"]).combine({"q": v}).value, .99)
        for required in ([], ["q", "q"]):
            with self.assertRaises(ValueError):
                NoisyANDCombiner("test", required)
            with self.assertRaises(ValueError):
                LatticeCombiner("test", required, lambda vs: True)

if __name__ == "__main__":
    unittest.main()
