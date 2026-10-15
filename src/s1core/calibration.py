"""Temperature fitting and split-conformal probability-threshold (LAC) sets.

Coverage is marginal on exchangeable held-out data, not conditional accuracy
of the subset admitted by the harness. Temperature fitting and conformal
calibration must use disjoint data. ECE is a diagnostic, not a guarantee.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Mapping, Sequence
from .gates import PROB_FLOOR, normalize, temperature_scale
from .types import GatePolicy

Sample = tuple[Mapping[str, float], str]


def fit_temperature(samples: Sequence[Sample], *, lo=0.05, hi=20.0, iterations=100) -> float:
    """Minimize held-in mean NLL over a bounded positive temperature interval."""
    if not samples:
        raise ValueError("fit_temperature requires at least one sample")
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi) or iterations < 1:
        raise ValueError("invalid temperature search bounds or iterations")
    normalized = [(normalize(p), c) for p, c in samples]
    if any(c not in p for p, c in normalized):
        raise ValueError("label missing from distribution")
    def nll(t):
        return -sum(math.log(max(temperature_scale(p, t)[c], PROB_FLOOR)) for p, c in normalized) / len(normalized)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = nll(d)
    return (a + b) / 2.0


def conformal_tau(calibration_scores: Sequence[float], alpha: float) -> float:
    """Lower-tail finite-sample threshold for sets {i: p_i >= tau}.

    Use floor((n+1)*alpha), with tau=0 when the rank is zero. This is
    equivalent to the usual upper quantile of nonconformity 1-p_true.
    """
    if not calibration_scores or not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("require scores and 0 < alpha < 1")
    if any(not math.isfinite(s) or not 0.0 <= s <= 1.0 for s in calibration_scores):
        raise ValueError("calibration scores must be finite probabilities")
    rank = math.floor((len(calibration_scores) + 1) * alpha)
    return 0.0 if rank == 0 else sorted(calibration_scores)[rank - 1]


def expected_calibration_error(samples: Sequence[Sample], *, bins=15) -> float:
    if not samples or not isinstance(bins, int) or bins < 1:
        raise ValueError("require samples and a positive integer bin count")
    counts, confidence, accuracy = [0] * bins, [0.0] * bins, [0.0] * bins
    for raw, correct in samples:
        p = normalize(raw)
        if correct not in p:
            raise ValueError("label missing from distribution")
        key = max(p, key=p.get)
        index = min(int(p[key] * bins), bins - 1)
        counts[index] += 1
        confidence[index] += p[key]
        accuracy[index] += key == correct
    return sum(abs(confidence[i] - accuracy[i]) for i in range(bins)) / len(samples)


def fit_gate_policy(samples: Sequence[Sample], *, alpha=0.10, base=None,
                    calibration_samples: Sequence[Sample] | None = None) -> GatePolicy:
    """Fit on samples, calibrate on disjoint calibration_samples.

    If calibration_samples is omitted, split samples in half without shuffling:
    first half for temperature, second half for conformal calibration. Supply
    representative exchangeable data; this helper does not cure dataset shift.
    """
    if calibration_samples is None:
        if len(samples) < 2:
            raise ValueError("automatic split requires at least two samples")
        split = len(samples) // 2
        fitting, calibration = samples[:split], samples[split:]
    else:
        fitting, calibration = samples, calibration_samples
    if not calibration:
        raise ValueError("conformal calibration requires held-out samples")
    temperature = fit_temperature(fitting)
    scores = []
    for p, correct in calibration:
        scaled = temperature_scale(normalize(p), temperature)
        if correct not in scaled:
            raise ValueError("label missing from distribution")
        scores.append(scaled[correct])
    return replace(base or GatePolicy(), temperature=temperature, conformal_tau=conformal_tau(scores, alpha))

__all__ = ["Sample", "fit_temperature", "conformal_tau", "expected_calibration_error", "fit_gate_policy"]
