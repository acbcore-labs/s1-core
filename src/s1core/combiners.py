"""Deterministic rules and a probability-product heuristic over ratified inputs.

Ratification here means the computation had admissible inputs, not that its
output is true or exceeds an application action threshold. Products require
independence to represent joint probabilities; shared-state signals may not.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence
from .types import DecisionVerdict, EpistemicStatus


class CombinerError(RuntimeError):
    """A deterministic rule raised or a signal has no categorical probability."""


def _fail_closed(name: str) -> DecisionVerdict:
    return DecisionVerdict(f"combined:{name}", EpistemicStatus.ABSTAINED_GATE,
        None, None, probabilities={}, confidence=0.0, top2_margin=0.0, normalized_entropy=0.0)


def signal_probability(verdict: DecisionVerdict) -> float:
    """Selected-label model probability, not an independent correctness proof.

    Score expectations and combined scalars have no selected-label probability.
    """
    if verdict.status is not EpistemicStatus.RATIFIED:
        return 0.0
    if isinstance(verdict.value, bool):
        key = "true" if verdict.value else "false"
    elif isinstance(verdict.value, str):
        key = verdict.value
    else:
        raise CombinerError("probability products require choice/noul signals, not score expectations")
    if key not in verdict.probabilities:
        raise CombinerError("selected-label probability missing from signal")
    p = float(verdict.probabilities[key])
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise CombinerError("selected-label probability must be finite and in [0, 1]")
    return p


class LatticeCombiner:
    """Apply a Boolean rule to ratified inputs; confidence is a heuristic minimum.

    Numeric score inputs are permitted in the rule. The confidence field is
    the minimum input confidence, not a calibrated probability of the rule.
    """
    def __init__(self, name: str, required: Sequence[str],
                 rule: Callable[[Mapping[str, DecisionVerdict]], bool]):
        if not required or len(set(required)) != len(required):
            raise ValueError("required signals must be nonempty and unique")
        self.name, self._required, self._rule = name, tuple(required), rule

    def combine(self, verdicts):
        if any(qid not in verdicts or not verdicts[qid].ratified for qid in self._required):
            return _fail_closed(self.name)
        try:
            outcome = bool(self._rule(verdicts))
        except Exception as exc:
            raise CombinerError(f"combiner '{self.name}' rule raised: {exc}") from exc
        return DecisionVerdict(f"combined:{self.name}", EpistemicStatus.RATIFIED,
            outcome, outcome, probabilities={},
            confidence=min(verdicts[qid].confidence for qid in self._required),
            top2_margin=0.0, normalized_entropy=0.0,
            replay_digest="|".join(verdicts[qid].replay_digest for qid in self._required))


class NoisyANDCombiner:
    """Product of selected-label probabilities; not a truth/action threshold.

    A ratified False noul contributes p(false), not p(true): the product refers
    to the selected labels. Only independent events justify a joint-probability
    interpretation. Applications must apply their own action threshold.
    """
    def __init__(self, name: str, required: Sequence[str]):
        if not required or len(set(required)) != len(required):
            raise ValueError("required signals must be nonempty and unique")
        self.name, self._required = name, tuple(required)

    def combine(self, verdicts):
        if any(qid not in verdicts or not verdicts[qid].ratified for qid in self._required):
            return _fail_closed(self.name)
        joint = math.prod(signal_probability(verdicts[qid]) for qid in self._required)
        return DecisionVerdict(f"combined:{self.name}", EpistemicStatus.RATIFIED,
            joint, joint, probabilities={}, confidence=joint, top2_margin=0.0,
            normalized_entropy=0.0,
            replay_digest="|".join(verdicts[qid].replay_digest for qid in self._required))
