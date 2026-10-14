"""Epistemic vocabulary shared by every stage of the pipeline.

The :class:`DecisionVerdict` is the *only* shape application code ever
receives from the harness. Its law (fail-closed admissibility boundary,
Pillar III): ``value`` is not ``None`` if and only if ``status`` is
``RATIFIED``. Application control flow branches on ratified values; every
other status means "no decision was made" and must be handled explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class EpistemicStatus(Enum):
    """Formal epistemic status tag attached to every decision.

    ``SKIPPED_DEPENDENCY`` extends the five statuses in the source design doc:
    a declared question whose wave-DAG parent did not ratify with the required
    value is *reported*, never silently omitted (Failure Mode #2 — silent
    failures — applies to harness bookkeeping too).
    """

    RATIFIED = "RATIFIED"                    # Passed all margin/entropy/calibration gates
    ABSTAINED_MODEL = "ABSTAINED_MODEL"      # Model explicitly selected an injected escape hatch
    ABSTAINED_GATE = "ABSTAINED_GATE"        # Rejected by margin / entropy / conformal / divergence gate
    BIMODAL_CONFLICT = "BIMODAL_CONFLICT"    # Score distribution split across non-adjacent rubric levels
    SKIPPED_DEPENDENCY = "SKIPPED_DEPENDENCY"  # Wave-DAG parent did not ratify with the required value
    LOOP_GUARD_TRIPPED = "LOOP_GUARD_TRIPPED"  # Exceeded max consecutive abstentions on the stream

    @property
    def admits_value(self) -> bool:
        return self is EpistemicStatus.RATIFIED


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """Per-question admissibility thresholds. Every gate fails closed.

    ``conformal_tau`` is the split-conformal APS threshold fitted from held-out
    data (see :mod:`s1core.calibration`); when ``None`` the conformal gate is
    not applied. ``debiased_permutation`` dispatches the question twice with
    reversed option ordering and refuses the averaged result if the forward and
    reversed distributions diverge by more than ``max_permutation_divergence``
    (bits of Jensen-Shannon divergence).
    """

    min_top1_prob: float = 0.65
    min_top2_margin: float = 0.20
    max_normalized_entropy: float = 0.55
    max_score_mode_aad: float = 0.25
    temperature: float = 1.0
    conformal_tau: float | None = None
    debiased_permutation: bool = False
    max_permutation_divergence: float = 0.10


@dataclass(frozen=True, slots=True)
class DecisionVerdict:
    """Immutable, fully-gated outcome for one question against one state.

    ``normalized_entropy`` is the scale-invariant Shannon entropy (in [0, 1])
    for choice/noul; ``score_mode_deviation`` carries the normalized Average
    Absolute Deviation around the mode for score primitives (kept as a separate
    field — the two metrics have different laws and must not share a slot).
    """

    question_id: str
    status: EpistemicStatus
    value: str | float | bool | None      # None unless status is RATIFIED
    raw_value: str | float | bool | None  # Un-gated argmax/expectation for shadow telemetry
    probabilities: Mapping[str, float]
    confidence: float
    top2_margin: float
    normalized_entropy: float
    score_mode_deviation: float | None = None
    conformal_set_size: int | None = None
    permutation_divergence: float | None = None
    replay_digest: str = ""

    @property
    def ratified(self) -> bool:
        return self.status.admits_value

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "status": self.status.value,
            "value": self.value,
            "raw_value": self.raw_value,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "top2_margin": self.top2_margin,
            "normalized_entropy": self.normalized_entropy,
            "score_mode_deviation": self.score_mode_deviation,
            "conformal_set_size": self.conformal_set_size,
            "permutation_divergence": self.permutation_divergence,
            "replay_digest": self.replay_digest,
        }

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "DecisionVerdict":
        return cls(
            question_id=data["question_id"],
            status=EpistemicStatus(data["status"]),
            value=data["value"],
            raw_value=data["raw_value"],
            probabilities=dict(data.get("probabilities") or {}),
            confidence=data["confidence"],
            top2_margin=data["top2_margin"],
            normalized_entropy=data["normalized_entropy"],
            score_mode_deviation=data.get("score_mode_deviation"),
            conformal_set_size=data.get("conformal_set_size"),
            permutation_divergence=data.get("permutation_divergence"),
            replay_digest=data.get("replay_digest", ""),
        )
