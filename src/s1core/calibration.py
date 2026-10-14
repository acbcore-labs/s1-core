"""Post-hoc auto-calibration (design pattern #4: shadow-traffic distillation).

Pure, dependency-free statistics over ledger-derived samples:

- :func:`fit_temperature` — per-question temperature ``T_d`` minimizing
  negative log-likelihood via golden-section search (exact 1-D optimum, no
  local minima on this unimodal objective).
- :func:`conformal_tau` — split-conformal APS threshold guaranteeing
  ``P(true-class probability >= tau) >= 1 - alpha`` on exchangeable data.
- :func:`expected_calibration_error` — binned ECE, the fitness metric these
  fits minimize.
- :func:`fit_gate_policy` — convenience: a GatePolicy with ``temperature`` and
  ``conformal_tau`` fitted from labeled samples.

Labels come from production ground truth or a strong System 2 judge; nothing
here is task-specific.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace
from typing import Mapping, Sequence

from .gates import PROB_FLOOR, normalize, temperature_scale
from .types import GatePolicy

# (raw distribution, key of the outcome that turned out to be correct)
Sample = tuple[Mapping[str, float], str]


def fit_temperature(samples: Sequence[Sample], *, lo: float = 0.05, hi: float = 20.0,
                    iterations: int = 100) -> float:
    """Golden-section search for the temperature minimizing mean NLL.

    NLL as a function of temperature is unimodal on (0, inf) for any fixed
    sample set, so golden section converges to the exact optimum without
    gradient machinery.
    """
    if not samples:
        raise ValueError("fit_temperature requires at least one sample")

    def nll(t: float) -> float:
        total = 0.0
        for probs, correct in samples:
            scaled = temperature_scale(normalize(probs), t)
            total -= math.log(max(scaled.get(correct, 0.0), PROB_FLOOR))
        return total / len(samples)

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
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
    """Split-conformal APS threshold.

    ``calibration_scores`` are the probabilities the (already
    temperature-scaled) model assigned to the true outcome on a held-out
    calibration set. Returns tau such that a prediction set
    ``{i : p_i >= tau}`` covers the true outcome with probability at least
    ``1 - alpha`` on exchangeable data: tau is the alpha-quantile of the
    scores (finite-sample rank ``ceil((n+1) * alpha)``), so the true-class
    score falls below tau at most ``alpha`` of the time.
    """
    if not calibration_scores:
        raise ValueError("conformal_tau requires at least one calibration score")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    ordered = sorted(calibration_scores)
    n = len(ordered)
    rank = min(max(math.ceil((n + 1) * alpha), 1), n)
    return ordered[rank - 1]


def expected_calibration_error(samples: Sequence[Sample], *, bins: int = 15) -> float:
    """Binned ECE over top-1 confidence vs top-1 accuracy."""
    if not samples:
        raise ValueError("expected_calibration_error requires at least one sample")
    bucket_acc = [0.0] * bins
    bucket_conf = [0.0] * bins
    bucket_n = [0] * bins
    for probs, correct in samples:
        normalized = normalize(probs)
        top_key = max(normalized, key=lambda k: normalized[k])
        conf = normalized[top_key]
        idx = min(int(conf * bins), bins - 1)
        bucket_n[idx] += 1
        bucket_conf[idx] += conf
        if top_key == correct:
            bucket_acc[idx] += 1.0
    total = len(samples)
    ece = 0.0
    for n_b, conf_b, acc_b in zip(bucket_n, bucket_conf, bucket_acc):
        if n_b:
            ece += (n_b / total) * abs((acc_b / n_b) - (conf_b / n_b))
    return ece


def fit_gate_policy(
    samples: Sequence[Sample],
    *,
    alpha: float = 0.10,
    base: GatePolicy | None = None,
) -> GatePolicy:
    """Fit temperature (min NLL), then conformal tau (at risk ``alpha``) on top."""
    base = base or GatePolicy()
    temperature = fit_temperature(samples)
    scaled = [(temperature_scale(normalize(p), temperature), c) for p, c in samples]
    tau = conformal_tau([sp[c] for sp, c in scaled], alpha)
    return replace(base, temperature=temperature, conformal_tau=tau)


__all__ = [
    "Sample",
    "fit_temperature",
    "conformal_tau",
    "expected_calibration_error",
    "fit_gate_policy",
]
