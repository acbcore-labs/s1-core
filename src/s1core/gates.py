"""Pure calibration and primitive-specific admissibility gates."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping
from .types import DecisionVerdict, EpistemicStatus, GatePolicy

PROB_FLOOR = 1e-12
CHOICE, SCORE, NOUL = "choice", "score", "noul"

@dataclass(frozen=True, slots=True)
class GateInputs:
    question_id: str
    primitive: str
    policy: GatePolicy
    escape_keys: frozenset[str] = frozenset()
    noul_true_edge: float = 0.75
    noul_false_edge: float = 0.35

    def to_jsonable(self) -> dict[str, Any]:
        return {"question_id": self.question_id, "primitive": self.primitive,
                "policy": dataclasses.asdict(self.policy), "escape_keys": sorted(self.escape_keys),
                "noul_true_edge": self.noul_true_edge, "noul_false_edge": self.noul_false_edge}

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "GateInputs":
        return cls(data["question_id"], data["primitive"], GatePolicy(**data["policy"]),
                   frozenset(data["escape_keys"]), data["noul_true_edge"], data["noul_false_edge"])


def normalize(raw: Mapping[str, float]) -> dict[str, float]:
    """Reject invalid weights before normalizing to the probability simplex."""
    if not raw:
        raise ValueError("distribution must not be empty")
    values = {k: float(v) for k, v in raw.items()}
    if any(not math.isfinite(v) or v < 0.0 for v in values.values()):
        raise ValueError("distribution weights must be finite and nonnegative")
    peak = max(values.values())
    if peak <= 0.0:
        raise ValueError("distribution must have positive mass")
    scaled = {k: v / peak for k, v in values.items()}
    total = math.fsum(scaled.values())
    return {k: v / total for k, v in scaled.items()}


def temperature_scale(probs: Mapping[str, float], temperature: float) -> dict[str, float]:
    """Stable probability-space temperature scaling (equivalent to scaled logits)."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    probs = normalize(probs)
    if abs(temperature - 1.0) < 1e-9:
        return dict(probs)
    logs = {k: math.log(max(v, PROB_FLOOR)) for k, v in probs.items()}
    peak = max(logs.values())
    weights = {k: math.exp((v - peak) / temperature) for k, v in logs.items()}
    total = math.fsum(weights.values())
    return {k: v / total for k, v in weights.items()}


def normalized_entropy(probs: Mapping[str, float]) -> float:
    if len(probs) <= 1:
        return 0.0
    return -sum(p * math.log(max(p, PROB_FLOOR)) for p in probs.values()) / math.log(len(probs))


def _sorted_probs(probs: Mapping[str, float]):
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return items, items[0][1], items[1][1] if len(items) > 1 else 0.0


def js_divergence_bits(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    p, q = normalize(p), normalize(q)
    terms = []
    for k in sorted(set(p) | set(q)):
        a, b = p.get(k, 0.0), q.get(k, 0.0)
        m = 0.5 * (a + b)
        if a > 0.0:
            terms.append(0.5 * a * math.log2(a / m))
        if b > 0.0:
            terms.append(0.5 * b * math.log2(b / m))
    return math.fsum(terms)


def evaluate_choice(gi: GateInputs, raw_probs: Mapping[str, float], replay_digest: str) -> DecisionVerdict:
    probs = temperature_scale(normalize(raw_probs), gi.policy.temperature)
    items, top1, top2 = _sorted_probs(probs)
    key, margin, entropy = items[0][0], top1 - top2, normalized_entropy(probs)
    base = dict(probabilities=probs, confidence=1.0 - entropy, top2_margin=margin,
                normalized_entropy=entropy, replay_digest=replay_digest)
    if key in gi.escape_keys:
        return DecisionVerdict(gi.question_id, EpistemicStatus.ABSTAINED_MODEL, None, key, **base)
    size = None if gi.policy.conformal_tau is None else sum(p >= gi.policy.conformal_tau for p in probs.values())
    refused = (top1 < gi.policy.min_top1_prob or margin < gi.policy.min_top2_margin
               or entropy > gi.policy.max_normalized_entropy or (size is not None and size != 1))
    return DecisionVerdict(gi.question_id,
                          EpistemicStatus.ABSTAINED_GATE if refused else EpistemicStatus.RATIFIED,
                          None if refused else key, key, conformal_set_size=size, **base)


def evaluate_score(gi: GateInputs, raw_probs: Mapping[str, float], replay_digest: str) -> DecisionVerdict:
    probs = temperature_scale(normalize(raw_probs), gi.policy.temperature)
    try:
        levels = {int(k): p for k, p in probs.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("score keys must be rubric level integers") from exc
    if len(levels) != len(probs) or set(levels) != set(range(1, len(levels) + 1)) or not 2 <= len(levels) <= 10:
        raise ValueError("score levels must be exactly 1..k, with 2 <= k <= 10")
    expected = sum(level * p for level, p in levels.items())
    mode = max(sorted(levels), key=lambda level: levels[level])
    aad = sum(p * abs(level - mode) for level, p in levels.items()) / (len(levels) - 1)
    _, top1, top2 = _sorted_probs(probs)
    refused = aad > gi.policy.max_score_mode_aad
    return DecisionVerdict(gi.question_id,
                          EpistemicStatus.BIMODAL_CONFLICT if refused else EpistemicStatus.RATIFIED,
                          None if refused else expected, expected, probabilities=probs,
                          confidence=1.0 - aad, top2_margin=top1 - top2,
                          normalized_entropy=normalized_entropy(probs), score_mode_deviation=aad,
                          replay_digest=replay_digest)


def evaluate_noul(gi: GateInputs, raw_probs: Mapping[str, float], replay_digest: str) -> DecisionVerdict:
    if set(raw_probs) != {"true", "false"}:
        raise ValueError("noul distribution must contain exactly 'true' and 'false'")
    if not 0.0 < gi.noul_false_edge < gi.noul_true_edge <= 1.0:
        raise ValueError("require 0 < noul false edge < true edge <= 1")
    probs = temperature_scale(normalize(raw_probs), gi.policy.temperature)
    p = probs["true"]
    dist = {"true": p, "false": 1.0 - p}
    value = True if p >= gi.noul_true_edge else False if p <= gi.noul_false_edge else None
    return DecisionVerdict(gi.question_id,
                          EpistemicStatus.RATIFIED if value is not None else EpistemicStatus.ABSTAINED_GATE,
                          value, p if value is None else value, probabilities=dist,
                          confidence=max(p, 1.0 - p), top2_margin=abs(2.0 * p - 1.0),
                          normalized_entropy=normalized_entropy(dist), replay_digest=replay_digest)


def evaluate(gi: GateInputs, raw_probs: Mapping[str, float], replay_digest: str) -> DecisionVerdict:
    handlers = {CHOICE: evaluate_choice, SCORE: evaluate_score, NOUL: evaluate_noul}
    if gi.primitive not in handlers:
        raise ValueError(f"unknown primitive: {gi.primitive!r}")
    policy = dataclasses.asdict(gi.policy)
    for key in ("min_top1_prob", "min_top2_margin", "max_normalized_entropy", "max_score_mode_aad", "max_permutation_divergence"):
        if not math.isfinite(policy[key]) or not 0.0 <= policy[key] <= 1.0:
            raise ValueError(f"{key} must be finite and in [0, 1]")
    tau = gi.policy.conformal_tau
    if tau is not None and (not math.isfinite(tau) or not 0.0 <= tau <= 1.0):
        raise ValueError("conformal_tau must be finite and in [0, 1]")
    return handlers[gi.primitive](gi, raw_probs, replay_digest)
