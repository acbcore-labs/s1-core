"""Stage 1: declarative question specs and the compile-time linter.

Pillar I (Semantic Rigor) lives here. System 1 models do *not* read JSON keys
or enum identifiers (Failure Mode #3) — only the natural-language instruction
and criteria prose reaches the pointer head. The compiler therefore rejects
terse, under-specified prose at build time and injects epistemic escape hatches
so ambiguity can never force a false-positive pick (Failure Mode #1).

Every spec is a frozen dataclass with a ``compile()`` method returning a
:class:`CompiledQuestion` — the wire-format payload plus the
:class:`~s1core.gates.GateInputs` that Stage 4 will later apply.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Union

from .gates import CHOICE, NOUL, SCORE, GateInputs
from .types import GatePolicy

# Injected epistemic escape hatches (Part 2, Stage 1). Both are added to every
# non-exhaustive Choice set by default; either one maps to ABSTAINED_MODEL.
ABSTAIN_INSUFFICIENT_EVIDENCE = "__abstain_insufficient_evidence__"
ABSTAIN_OUT_OF_DOMAIN = "__abstain_out_of_domain__"
ESCAPE_HATCH_CRITERIA: dict[str, str] = {
    ABSTAIN_INSUFFICIENT_EVIDENCE: (
        "The input state is incomplete, ambiguous, truncated, or lacks sufficient "
        "factual evidence to confidently choose any of the specific options above."
    ),
    ABSTAIN_OUT_OF_DOMAIN: (
        "The input state is entirely unrelated to this question's subject matter; "
        "none of the specific options above are applicable to it."
    ),
}

MIN_INSTRUCTION_WORDS = 4
MIN_CRITERIA_WORDS = 3
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# (parent_question_id, required_value) — one condition, or a sequence of
# conditions ALL of which must hold for the question to execute.
Dependencies = Union[tuple[str, str], Sequence[tuple[str, str]], None]


class SpecError(ValueError):
    """Compile-time schema violation — the question never reaches the model."""


def _word_count(text: str) -> int:
    return len(text.strip().split())


def _check_instructions(question_id: str, instructions: str) -> None:
    if _word_count(instructions) < MIN_INSTRUCTION_WORDS:
        raise SpecError(
            f"[{question_id}] Instructions too terse ({_word_count(instructions)} words, "
            f"minimum {MIN_INSTRUCTION_WORDS}). System 1 models do not read "
            f"question_id='{question_id}'; write full, self-contained instructions."
        )


def _check_criteria(question_id: str, label: str, criteria: str) -> None:
    if _word_count(criteria) < MIN_CRITERIA_WORDS:
        raise SpecError(
            f"[{question_id}:{label}] Criteria too thin ({_word_count(criteria)} words, "
            f"minimum {MIN_CRITERIA_WORDS}). The model cannot read the key '{label}'; "
            "the prose must explicitly define what it means."
        )


def _normalize_deps(deps: Dependencies) -> tuple[tuple[str, str], ...]:
    if deps is None:
        return ()
    if (
        isinstance(deps, tuple)
        and len(deps) == 2
        and isinstance(deps[0], str)
        and isinstance(deps[1], str)
    ):
        return (deps,)
    return tuple((str(parent), str(required)) for parent, required in deps)


def _canonical_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def values_match(actual: Any, required: Any) -> bool:
    """Dependency-value comparison.

    Bools canonicalize to ``"true"``/``"false"``; integral floats collapse to
    their integer form (``2.0`` matches ``"2"``); everything else compares as
    its exact string form.
    """
    return _canonical_value(actual) == _canonical_value(required)


@dataclass(frozen=True, slots=True)
class CompiledQuestion:
    """Wire format sent to a backend driver, plus the gate that will judge it."""

    question_id: str
    primitive: str
    instructions: str
    criteria: Mapping[str, str]
    gate_inputs: GateInputs

    def to_payload(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "type": self.primitive,
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }

    def reversed_criteria(self) -> "CompiledQuestion":
        """Same question with option order reversed (dual-permutation debiasing)."""
        return dataclasses.replace(
            self, criteria=dict(reversed(list(self.criteria.items())))
        )


@dataclass(frozen=True, slots=True)
class ChoiceSpec:
    """``switch``/``match on an enum``: 2–255 predefined options.

    ``options`` maps option key -> natural-language criteria. The key is an
    identifier for *your* code; the model only ever sees the criteria prose.
    """

    question_id: str
    instructions: str
    options: Mapping[str, str]
    inject_escape_hatch: bool = True
    policy: GatePolicy = field(default_factory=GatePolicy)
    depends_on: Dependencies = None

    def dependencies(self) -> tuple[tuple[str, str], ...]:
        return _normalize_deps(self.depends_on)

    def compile(self) -> CompiledQuestion:
        _check_instructions(self.question_id, self.instructions)
        if not (2 <= len(self.options) <= MAX_CHOICE_OPTIONS):
            raise SpecError(
                f"[{self.question_id}] Choice must have between 2 and "
                f"{MAX_CHOICE_OPTIONS} options, got {len(self.options)}."
            )
        compiled = {}
        for key, criteria in self.options.items():
            if not key:
                raise SpecError(f"[{self.question_id}] Option keys must be non-empty.")
            _check_criteria(self.question_id, key, criteria)
            compiled[key] = criteria
        escape_keys: frozenset[str] = frozenset()
        if self.inject_escape_hatch:
            for key, criteria in ESCAPE_HATCH_CRITERIA.items():
                if key not in compiled:
                    compiled[key] = criteria
            escape_keys = frozenset(ESCAPE_HATCH_CRITERIA)
        gate_inputs = GateInputs(
            question_id=self.question_id,
            primitive=CHOICE,
            policy=self.policy,
            escape_keys=escape_keys,
        )
        return CompiledQuestion(self.question_id, CHOICE, self.instructions, compiled, gate_inputs)


@dataclass(frozen=True, slots=True)
class ScoreSpec:
    """Continuous threshold / sort key / rubric grade: 2–10 ordered levels.

    ``rubric_levels`` must be exactly the contiguous integers ``1..k`` mapping
    to level criteria prose. The ratified value is the probability-weighted
    expectation ``s^ = sum(p_i * i)``.
    """

    question_id: str
    instructions: str
    rubric_levels: Mapping[int, str]
    policy: GatePolicy = field(default_factory=GatePolicy)
    depends_on: Dependencies = None

    def dependencies(self) -> tuple[tuple[str, str], ...]:
        return _normalize_deps(self.depends_on)

    def compile(self) -> CompiledQuestion:
        _check_instructions(self.question_id, self.instructions)
        keys = set(self.rubric_levels.keys())
        if not all(isinstance(k, int) and not isinstance(k, bool) for k in keys):
            raise SpecError(f"[{self.question_id}] Score rubric level keys must be ints.")
        if not (MIN_SCORE_LEVELS <= len(keys) <= MAX_SCORE_LEVELS):
            raise SpecError(
                f"[{self.question_id}] Score rubric must have between "
                f"{MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels, got {len(keys)}."
            )
        if keys != set(range(1, len(keys) + 1)):
            raise SpecError(
                f"[{self.question_id}] Score rubric levels must be exactly 1..{len(keys)} "
                f"contiguous; got {sorted(keys)}."
            )
        criteria: dict[str, str] = {}
        for lvl in sorted(keys):
            _check_criteria(self.question_id, f"L{lvl}", self.rubric_levels[lvl])
            criteria[str(lvl)] = self.rubric_levels[lvl]
        gate_inputs = GateInputs(
            question_id=self.question_id, primitive=SCORE, policy=self.policy
        )
        return CompiledQuestion(self.question_id, SCORE, self.instructions, criteria, gate_inputs)


@dataclass(frozen=True, slots=True)
class NoulSpec:
    """Probabilistic if/else gate over a binary proposition.

    ``p(true) >= true_threshold`` ratifies ``True``; ``p(true) <= false_threshold``
    ratifies ``False``; the band between is a fail-closed gray zone that
    abstains. Defaults follow the source design doc (0.75 / 0.35).
    """

    question_id: str
    instructions: str
    true_criteria: str
    false_criteria: str
    true_threshold: float = 0.75
    false_threshold: float = 0.35
    policy: GatePolicy = field(default_factory=GatePolicy)
    depends_on: Dependencies = None

    def dependencies(self) -> tuple[tuple[str, str], ...]:
        return _normalize_deps(self.depends_on)

    def compile(self) -> CompiledQuestion:
        _check_instructions(self.question_id, self.instructions)
        if not (0.0 < self.false_threshold < self.true_threshold <= 1.0):
            raise SpecError(
                f"[{self.question_id}] Require 0 < false_threshold ({self.false_threshold}) "
                f"< true_threshold ({self.true_threshold}) <= 1."
            )
        _check_criteria(self.question_id, "true", self.true_criteria)
        _check_criteria(self.question_id, "false", self.false_criteria)
        criteria = {"true": self.true_criteria, "false": self.false_criteria}
        gate_inputs = GateInputs(
            question_id=self.question_id,
            primitive=NOUL,
            policy=self.policy,
            noul_true_edge=self.true_threshold,
            noul_false_edge=self.false_threshold,
        )
        return CompiledQuestion(self.question_id, NOUL, self.instructions, criteria, gate_inputs)


QuestionPrimitive = ChoiceSpec | ScoreSpec | NoulSpec
