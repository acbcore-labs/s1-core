"""Deterministic combiners (design pattern #1: multi-signal orthogonal
decomposition).

A single broad question ("Should this PR be merged safely?") frequently loses
to a small reasoning LLM, but 5–8 narrow orthogonal signals cost ~1.03x the
tokens of one question and — combined here in plain deterministic code — match
or beat System 2 at a fraction of the latency.

Combiners fail closed: if any required signal is not RATIFIED, the combined
decision is ``ABSTAINED_GATE`` with no value, never a guess.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

from .types import DecisionVerdict, EpistemicStatus


class CombinerError(RuntimeError):
    """A deterministic combiner rule raised — a code defect, not an epistemic state."""


def _fail_closed(name: str) -> DecisionVerdict:
    return DecisionVerdict(
        question_id=f"combined:{name}",
        status=EpistemicStatus.ABSTAINED_GATE,
        value=None,
        raw_value=None,
        probabilities={},
        confidence=0.0,
        top2_margin=0.0,
        normalized_entropy=0.0,
        replay_digest="",
    )


def signal_probability(verdict: DecisionVerdict) -> float:
    """Model probability that a ratified signal's value is correct."""
    if verdict.status is not EpistemicStatus.RATIFIED:
        return 0.0
    if verdict.value in verdict.probabilities:
        return float(verdict.probabilities[verdict.value])
    if verdict.value is True or verdict.value is False:
        key = "true" if verdict.value else "false"
        return float(verdict.probabilities.get(key, 0.0))
    return 0.0


class LatticeCombiner:
    """Boolean lattice over ratified signal values, evaluated in plain code.

    ``rule`` receives the verdict mapping and returns the combined boolean.
    Non-ratified required signals fail the whole combination closed.
    """

    def __init__(
        self,
        name: str,
        required: Sequence[str],
        rule: Callable[[Mapping[str, DecisionVerdict]], bool],
    ):
        self.name = name
        self._required = tuple(required)
        self._rule = rule

    def combine(self, verdicts: Mapping[str, DecisionVerdict]) -> DecisionVerdict:
        unsatisfied = [
            qid for qid in self._required
            if (v := verdicts.get(qid)) is None or v.status is not EpistemicStatus.RATIFIED
        ]
        if unsatisfied:
            return _fail_closed(self.name)
        try:
            outcome = bool(self._rule(verdicts))
        except Exception as exc:  # deterministic user code must not fabricate decisions
            raise CombinerError(f"combiner '{self.name}' rule raised: {exc}") from exc
        digest = "|".join(verdicts[qid].replay_digest for qid in self._required)
        return DecisionVerdict(
            question_id=f"combined:{self.name}",
            status=EpistemicStatus.RATIFIED,
            value=outcome,
            raw_value=outcome,
            probabilities={},
            confidence=min(signal_probability(verdicts[qid]) for qid in self._required),
            top2_margin=0.0,
            normalized_entropy=0.0,
            replay_digest=digest,
        )


class NoisyANDCombiner:
    """Probability that all required signals hold, as ``prod(p_signal_i)``.

    Useful as a soft pre-gate: a low product routes to a System 2 reviewer
    instead of accepting the decomposed decision.
    """

    def __init__(self, name: str, required: Sequence[str]):
        self.name = name
        self._required = tuple(required)

    def combine(self, verdicts: Mapping[str, DecisionVerdict]) -> DecisionVerdict:
        unsatisfied = [
            qid for qid in self._required
            if (v := verdicts.get(qid)) is None or v.status is not EpistemicStatus.RATIFIED
        ]
        if unsatisfied:
            return _fail_closed(self.name)
        probs = [signal_probability(verdicts[qid]) for qid in self._required]
        joint = math.prod(probs)
        return DecisionVerdict(
            question_id=f"combined:{self.name}",
            status=EpistemicStatus.RATIFIED,
            value=joint,
            raw_value=joint,
            probabilities={},
            confidence=joint,
            top2_margin=0.0,
            normalized_entropy=0.0,
            replay_digest="|".join(verdicts[qid].replay_digest for qid in self._required),
        )
