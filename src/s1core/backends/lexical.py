"""A deterministic, order-invariant toy backend for tests and examples.

This is NOT a System 1 model. It scores lexical overlap (Dice coefficient)
between the canonical state's string content and each option's criteria prose,
then applies a mean-centered softmax. It exists so the entire harness
pipeline — waves, gates, audit, replay — runs offline with meaningful,
reproducible distributions, and so new backends have a reference for the
driver contract.

Swap it for :class:`~s1core.backends.systemone.SystemOneHTTPDriver` (hosted
models) or your own driver (local MLX/ONNX/llama.cpp models) without touching
application code.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Mapping

from .base import StateHandle

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words and harness vocabulary that carry no signal for overlap.
_STOPWORDS = frozenset(
    """
    a an and any are as at be been by can cannot choose could do does for from
    had has have how if in into is it its least less may more most must no not
    of on or other our per select selected should so specific state than that
    the their them then this to true false above input option options question
    criteria was were will with without would you your what which whether why
    all also each use used using based given show shows when
    """.split()
)


def tokenize(text: str) -> frozenset[str]:
    return frozenset(
        tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS
    )


def _iter_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            yield from _iter_strings(value)
    elif obj is None or obj is True or obj is False:
        return
    else:
        yield str(obj)


def state_tokens(canonical_state: Mapping[str, Any]) -> frozenset[str]:
    """State vocabulary: user-state keys AND values, plus derived fact keys
    and values. The harness envelope keys themselves are skipped — they are
    scaffolding, not signal."""
    from ..augment import FACTS_KEY

    tokens: set[str] = set()

    def add(key: Any, value: Any) -> None:
        if key is not None:
            tokens.update(tokenize(str(key)))
        for text in _iter_strings(value):
            tokens.update(tokenize(text))

    for key, value in canonical_state.items():
        if key in ("raw_state", FACTS_KEY) and isinstance(value, Mapping):
            for inner_key, inner_value in value.items():
                add(inner_key, inner_value)
        else:
            add(key, value)
    return frozenset(tokens)


def _dice(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


class LexicalHeuristicDriver:
    """Deterministic lexical-overlap driver. Order-invariant by construction,
    so ``debiased_permutation`` measure zero divergence against it."""

    def __init__(
        self,
        *,
        model_id: str = "lexical-heuristic-v1",
        sharpness: float = 16.0,
    ):
        self.model_id = model_id
        self._sharpness = sharpness

    @contextmanager
    def pin_state(self, canonical_state: Mapping[str, Any], state_hash: str):
        yield StateHandle(state_hash=state_hash, canonical_state=canonical_state)

    def score_branches(
        self, handle: StateHandle, batch: Mapping[str, Any]
    ) -> dict[str, dict[str, float]]:
        # Score each option's criteria prose against the state vocabulary.
        # Instructions are shared context across all options of a question and
        # carry no discriminative signal here.
        state_vocab = state_tokens(handle.canonical_state)
        results: dict[str, dict[str, float]] = {}
        for question_id, compiled in batch.items():
            overlaps = {
                key: _dice(state_vocab, tokenize(criteria))
                for key, criteria in compiled.criteria.items()
            }
            results[question_id] = self._softmax_centered(overlaps)
        return results

    def _softmax_centered(self, overlaps: Mapping[str, float]) -> dict[str, float]:
        mean = sum(overlaps.values()) / len(overlaps)
        logits = {k: (v - mean) * self._sharpness for k, v in overlaps.items()}
        peak = max(logits.values())
        exps = {k: math.exp(v - peak) for k, v in logits.items()}
        z = sum(exps.values())
        return {k: v / z for k, v in exps.items()}
