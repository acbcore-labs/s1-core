"""Deterministic combiners (design pattern #1: multi-signal orthogonal
decomposition).

A single broad question ("Should this PR be merged safely?") frequently loses
to a small reasoning LLM, but 5–8 narrow orthogonal signals cost ~1.03x the
tokens of one question and — combined here in plain deterministic code — match
or beat System 2 at a fraction of the latency.

Two laws keep the combiners honest:

- **Fail-closed**: if any required signal is not RATIFIED, the combined
  decision is ``ABSTAINED_GATE`` with no value, never a guess.
- **Ratification ≠ truth**: a ratified signal means the computation had
  admissible inputs, not that its output is true or exceeds an application
  action threshold. Probability products require independence to represent
  joint probabilities; shared-state signals may not.
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
    """Model probability that a ratified signal's *selected label* is correct.

    A score expectation or an already-combined scalar is not a categorical
    outcome and has no selected-label probability — asking for one is a code
    defect (:class:`CombinerError`), not a zero.
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
    """Boolean lattice over ratified signal values, evaluated in plain code.

    ``rule`` receives the verdict mapping and returns the combined boolean;
    numeric score values may be compared inside it. The resulting confidence
    field is the *minimum input confidence* — a routing heuristic, not a
    calibrated probability of the rule.
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
        except Exception as exc:  # deterministic user code must not fabricate decisions
            raise CombinerError(f"combiner '{self.name}' rule raised: {exc}") from exc
        return DecisionVerdict(f"combined:{self.name}", EpistemicStatus.RATIFIED,
            outcome, outcome, probabilities={},
            confidence=min(verdicts[qid].confidence for qid in self._required),
            top2_margin=0.0, normalized_entropy=0.0,
            replay_digest="|".join(verdicts[qid].replay_digest for qid in self._required))


class NoisyANDCombiner:
    """Probability that all required selected labels hold, as ``prod(p_i)``.

    Useful as a soft pre-gate: a low product routes to a System 2 reviewer
    instead of accepting the decomposed decision. A ratified False noul
    contributes ``p(false)``, not ``p(true)`` — the product refers to the
    selected labels, so only independent events justify reading it as a joint
    probability. The value is a scalar for the application to threshold;
    the harness does not impose an action threshold.
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
