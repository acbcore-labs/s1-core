"""Stage 4: post-hoc calibration and scale-invariant admissibility gates.

Pure functions: ``(GateInputs, raw distribution, replay_digest) ->
DecisionVerdict``. No I/O, no clock, no RNG — the precondition for
byte-identical replay (``wrong == 0``).

Gate math (from the S1-Core design doc):

1. Domain-calibrated distribution: temperature scaling ``T_d`` applied to the
   raw distribution (log-space; ``T_d = 1.0`` for pre-calibrated RLCD models).
2. Scale-invariant normalized entropy: Shannon entropy divided by ``ln(m)``
   so one threshold works for 3-option and 200-option choices alike.
3. Top-2 dominance margin + conformal prediction-set cardinality.
4. Bimodality rejection for score primitives: normalized Average Absolute
   Deviation (AAD) around the mode. ``s^ = 2.0`` can come from a confident
   ``[0.01, 0.98, 0.01]`` (AAD 0.01) or a polarized ``[0.50, 0.00, 0.50]``
   (AAD 0.50) — only the latter is refused.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .types import DecisionVerdict, EpistemicStatus, GatePolicy

PROB_FLOOR = 1e-12

CHOICE = "choice"
SCORE = "score"
NOUL = "noul"


@dataclass(frozen=True, slots=True)
class GateInputs:
    """Everything Stage 4 needs besides the raw distribution.

    Kept separate from the spec objects so the gate is a pure function and the
    audit ledger can serialize exactly these inputs for replay.
    """

    question_id: str
    primitive: str  # "choice" | "score" | "noul"
    policy: GatePolicy
    escape_keys: frozenset[str] = frozenset()
    noul_true_edge: float = 0.75   # p(true) >= true_edge  -> ratified True
    noul_false_edge: float = 0.35  # p(true) <= false_edge -> ratified False; between -> abstain

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "primitive": self.primitive,
            "policy": dataclasses.asdict(self.policy),
            "escape_keys": sorted(self.escape_keys),
            "noul_true_edge": self.noul_true_edge,
            "noul_false_edge": self.noul_false_edge,
        }

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "GateInputs":
        return cls(
            question_id=data["question_id"],
            primitive=data["primitive"],
            policy=GatePolicy(**data["policy"]),
            escape_keys=frozenset(data["escape_keys"]),
            noul_true_edge=data["noul_true_edge"],
            noul_false_edge=data["noul_false_edge"],
        )


def normalize(raw: Mapping[str, float]) -> dict[str, float]:
    """Normalize a raw distribution to the probability simplex."""
    total = float(sum(raw.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"distribution does not sum to a positive finite value: {raw!r}")
    return {k: float(v) / total for k, v in raw.items()}


def temperature_scale(probs: Mapping[str, float], temperature: float) -> dict[str, float]:
    """Sharpen (T < 1) or flatten (T > 1) a distribution in log-space."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if abs(temperature - 1.0) < 1e-9:
        return dict(probs)
    scaled = {k: math.exp(math.log(max(v, PROB_FLOOR)) / temperature) for k, v in probs.items()}
    z = sum(scaled.values())
    return {k: v / z for k, v in scaled.items()}


def normalized_entropy(probs: Mapping[str, float]) -> float:
    """H_norm = -sum(p ln p) / ln(m) in [0, 1]; scale-invariant across option counts."""
    m = len(probs)
    if m <= 1:
        return 0.0
    h = -sum(p * math.log(max(p, PROB_FLOOR)) for p in probs.values())
    return h / math.log(m)


def _sorted_probs(probs: Mapping[str, float]) -> tuple[list[tuple[str, float]], float, float]:
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top1 = items[0][1]
    top2 = items[1][1] if len(items) > 1 else 0.0
    return items, top1, top2


def js_divergence_bits(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence in bits between two distributions."""
    keys = set(p) | set(q)

    def kl(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        total = 0.0
        for k, av in a.items():
            if av > 0.0:
                bv = b.get(k, 0.0)
                if bv <= 0.0:
                    return float("inf")
                total += av * math.log(av / bv)
        return total

    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    js_nats = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return js_nats / math.log(2.0)


def evaluate_choice(gi: GateInputs, raw_probs: Mapping[str, float], replay_digest: str) -> DecisionVerdict:
    probs = temperature_scale(normalize(raw_probs), gi.policy.temperature)
    items, top1_p, top2_p = _sorted_probs(probs)
    top1_key = items[0][0]
    margin = top1_p - top2_p
    h_norm = normalized_entropy(probs)
    base = {
        "probabilities": probs,
        "confidence": 1.0 - h_norm,
        "top2_margin": margin,
        "normalized_entropy": h_norm,
        "replay_digest": replay_digest,
    }

    # 1. The model explicitly chose an injected epistemic escape hatch.
    if top1_key in gi.escape_keys:
        return DecisionVerdict(gi.question_id, EpistemicStatus.ABSTAINED_MODEL, None, top1_key, **base)

    # 2. Conformal prediction-set cardinality at the fitted risk level.
    conformal_size: int | None = None
    if gi.policy.conformal_tau is not None:
        conformal_size = sum(1 for p in probs.values() if p >= gi.policy.conformal_tau)

    # 3. Fail-closed admissibility boundaries.
    if (
        top1_p < gi.policy.min_top1_prob
        or margin < gi.policy.min_top2_margin
        or h_norm > gi.policy.max_normalized_entropy
        or (conformal_size is not None and conformal_size != 1)
    ):
        return DecisionVerdict(
            gi.question_id, EpistemicStatus.ABSTAINED_GATE, None, top1_key,
            conformal_set_size=conformal_size, **base,
        )

    return DecisionVerdict(
        gi.question_id, EpistemicStatus.RATIFIED, top1_key, top1_key,
        conformal_set_size=conformal_size, **base,
    )


def evaluate_score(gi: GateInputs, raw_probs: Mapping[str, float], replay_digest: str) -> DecisionVerdict:
    probs = temperature_scale(normalize(raw_probs), gi.policy.temperature)
    try:
        levels = {int(k): probs[k] for k in probs}
    except (KeyError, ValueError) as exc:
        raise ValueError(f"score distribution keys must be rubric level integers: {probs!r}") from exc

    k = len(levels)
    expected = sum(lvl * p for lvl, p in levels.items())
    # Ties resolve to the lowest level — deterministic under iteration in
    # ascending level order.
    mode = max(sorted(levels), key=lambda lvl: levels[lvl])
    aad_mode = sum(p * abs(lvl - mode) for lvl, p in levels.items())
    norm_aad = aad_mode / max(k - 1, 1)

    _, top1_p, top2_p = _sorted_probs(probs)
    base = {
        "probabilities": probs,
        "confidence": 1.0 - norm_aad,
        "top2_margin": top1_p - top2_p,
        "normalized_entropy": normalized_entropy(probs),
        "score_mode_deviation": norm_aad,
        "replay_digest": replay_digest,
    }

    # Refuse to collapse a polarized distribution into a scalar average.
    if norm_aad > gi.policy.max_score_mode_aad:
        return DecisionVerdict(gi.question_id, EpistemicStatus.BIMODAL_CONFLICT, None, expected, **base)

    return DecisionVerdict(gi.question_id, EpistemicStatus.RATIFIED, expected, expected, **base)


def evaluate_noul(gi: GateInputs, raw_probs: Mapping[str, float], replay_digest: str) -> DecisionVerdict:
    probs = temperature_scale(normalize(raw_probs), gi.policy.temperature)
    if "true" not in probs:
        raise ValueError(f"noul distribution must contain key 'true': {probs!r}")
    p_true = min(max(probs["true"], 0.0), 1.0)
    dist = {"true": p_true, "false": 1.0 - p_true}
    base = {
        "probabilities": dist,
        # Binary uncertainty is fully determined by p(1-p); peak separation
        # and 1 - normalized entropy coincide with max(p, 1-p) here.
        "confidence": max(p_true, 1.0 - p_true),
        "top2_margin": abs(2.0 * p_true - 1.0),
        "normalized_entropy": normalized_entropy(dist),
        "replay_digest": replay_digest,
    }

    if p_true >= gi.noul_true_edge:
        return DecisionVerdict(gi.question_id, EpistemicStatus.RATIFIED, True, True, **base)
    if p_true <= gi.noul_false_edge:
        return DecisionVerdict(gi.question_id, EpistemicStatus.RATIFIED, False, False, **base)
    # Fail-closed gray zone: the value is withheld, not rounded toward an edge.
    return DecisionVerdict(gi.question_id, EpistemicStatus.ABSTAINED_GATE, None, p_true, **base)


def evaluate(gi: GateInputs, raw_probs: Mapping[str, float], replay_digest: str) -> DecisionVerdict:
    """Dispatch to the gate for the primitive named in ``gi``."""
    if gi.primitive == CHOICE:
        return evaluate_choice(gi, raw_probs, replay_digest)
    if gi.primitive == SCORE:
        return evaluate_score(gi, raw_probs, replay_digest)
    if gi.primitive == NOUL:
        return evaluate_noul(gi, raw_probs, replay_digest)
    raise ValueError(f"unknown primitive: {gi.primitive!r}")
