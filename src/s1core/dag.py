"""Wave-DAG compilation (Failure Mode #4: cross-question blindness).

All questions inside one request execute in parallel behind a block-diagonal
attention mask, so Question B can never observe the model's answer to
Question A. Questions that depend on earlier outcomes are compiled into
topological *waves*: Wave 0 runs first, its ratified values gate Wave 1, and
so on. On KV-pinnable local backends the state prefix is encoded exactly once
across all waves.
"""

from __future__ import annotations

from typing import Sequence

from .specs import QuestionPrimitive, SpecError


def compile_waves(questions: Sequence[QuestionPrimitive]) -> list[list[QuestionPrimitive]]:
    """Group questions into dependency-ordered waves, preserving caller order.

    Raises :class:`SpecError` on duplicate ids, unknown parents, self- or
    cyclic dependencies — all at compile time, before any model call.
    """
    by_id: dict[str, QuestionPrimitive] = {}
    for spec in questions:
        if spec.question_id in by_id:
            raise SpecError(f"duplicate question_id: '{spec.question_id}'")
        by_id[spec.question_id] = spec

    deps: dict[str, tuple[tuple[str, str], ...]] = {
        spec.question_id: spec.dependencies() for spec in questions
    }
    for qid, conditions in deps.items():
        for parent, _required in conditions:
            if parent == qid:
                raise SpecError(f"[{qid}] cannot depend on itself")
            if parent not in by_id:
                raise SpecError(f"[{qid}] depends on undeclared question '{parent}'")

    remaining: list[str] = [spec.question_id for spec in questions]
    resolved: set[str] = set()
    waves: list[list[QuestionPrimitive]] = []
    while remaining:
        layer = [qid for qid in remaining if all(p in resolved for p, _ in deps[qid])]
        if not layer:
            raise SpecError(f"cyclic dependencies among: {sorted(remaining)}")
        waves.append([by_id[qid] for qid in layer])
        resolved.update(layer)
        remaining = [qid for qid in remaining if qid not in resolved]
    return waves
