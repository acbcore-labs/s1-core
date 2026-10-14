"""Calibration module tests: temperature recovery, conformal coverage, ECE."""

import math
import random
import unittest

from s1core import (
    GatePolicy,
    conformal_tau,
    expected_calibration_error,
    fit_gate_policy,
    fit_temperature,
    normalize as normalize_probs,
    temperature_scale,
)


def softmax(logits: dict[str, float]) -> dict[str, float]:
    peak = max(logits.values())
    exps = {k: math.exp(v - peak) for k, v in logits.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def synthetic_population(rng: random.Random, n: int, true_temperature: float):
    """Sample (raw probs, correct key) pairs the way a ledger sees them: the
    backend reports *raw* softmax(logits) probabilities, while outcomes are
    drawn from the model's true (temperature-scaled) distribution."""
    samples = []
    for _ in range(n):
        logits = {f"opt_{i}": rng.gauss(0.0, 1.0) for i in range(4)}
        raw = softmax(logits)
        truth = softmax({k: v / true_temperature for k, v in logits.items()})
        correct = rng.choices(list(truth), weights=truth.values(), k=1)[0]
        samples.append((raw, correct))
    return samples


class FitTemperatureTests(unittest.TestCase):
    def test_recovers_true_temperature(self):
        rng = random.Random(42)
        samples = synthetic_population(rng, 4000, true_temperature=2.0)
        fitted = fit_temperature(samples, lo=0.5, hi=8.0)
        self.assertLess(abs(fitted - 2.0), 0.25)

    def test_requires_samples(self):
        with self.assertRaises(ValueError):
            fit_temperature([])


class ConformalTauTests(unittest.TestCase):
    def test_coverage_guarantee_holds(self):
        rng = random.Random(7)
        samples = synthetic_population(rng, 5000, true_temperature=1.5)
        split = 3000
        calibrated = [
            (temperature_scale(normalize_probs(probs), 1.5), correct)
            for probs, correct in samples
        ]
        calib = calibrated[:split]
        tau = conformal_tau([probs[correct] for probs, correct in calib], alpha=0.10)
        heldout = calibrated[split:]
        covered = sum(1 for probs, correct in heldout if probs[correct] >= tau)
        self.assertGreaterEqual(covered / len(heldout), 0.88)

    def test_singleton_sets_are_selective(self):
        # A sharp model (high true-class probability) yields small conformal sets.
        rng = random.Random(11)
        samples = synthetic_population(rng, 2000, true_temperature=0.4)
        calibrated = [(temperature_scale(normalize_probs(probs), 0.4), c) for probs, c in samples]
        tau = conformal_tau([p[c] for p, c in calibrated], alpha=0.10)
        sizes = [sum(1 for v in p.values() if v >= tau) for p, _ in calibrated[:500]]
        self.assertLess(sum(sizes) / len(sizes), 1.6)


class ECETests(unittest.TestCase):
    def test_perfect_confidence_zero_error(self):
        samples = [({"a": 1.0, "b": 0.0}, "a"), ({"a": 1.0, "b": 0.0}, "a")]
        self.assertAlmostEqual(expected_calibration_error(samples), 0.0)

    def test_overconfidence_detected(self):
        samples = [({"a": 0.9, "b": 0.1}, "b")] * 10
        self.assertGreater(expected_calibration_error(samples), 0.8)


class FitGatePolicyTests(unittest.TestCase):
    def test_policy_carries_fitted_values(self):
        rng = random.Random(3)
        samples = synthetic_population(rng, 3000, true_temperature=2.0)
        policy = fit_gate_policy(samples, alpha=0.10)
        self.assertIsInstance(policy, GatePolicy)
        self.assertGreater(policy.temperature, 1.0)
        self.assertIsNotNone(policy.conformal_tau)
        self.assertGreater(policy.conformal_tau, 0.0)
        self.assertLessEqual(policy.conformal_tau, 1.0)


if __name__ == "__main__":
    unittest.main()
