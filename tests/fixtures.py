"""Shared test fixtures: scripted stub drivers with fixed distributions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping

from s1core.backends.base import StateHandle
from s1core.specs import ABSTAIN_INSUFFICIENT_EVIDENCE


class ScriptedDriver:
    """Returns canned distributions per question_id; anything unlisted is a
    confident uniform-abstain. Deterministic and order-invariant."""

    def __init__(
        self,
        scripted: Mapping[str, Mapping[str, float]],
        *,
        model_id: str = "scripted-test-v1",
    ):
        self.model_id = model_id
        self._scripted = dict(scripted)
        self.calls: list[list[str]] = []

    @contextmanager
    def pin_state(self, canonical_state: Mapping[str, Any], state_hash: str):
        yield StateHandle(state_hash=state_hash, canonical_state=canonical_state)

    def score_branches(
        self, handle: StateHandle, batch: Mapping[str, Any]
    ) -> dict[str, dict[str, float]]:
        self.calls.append(list(batch.keys()))
        out = {}
        for question_id, compiled in batch.items():
            if question_id in self._scripted:
                # A real backend scores every option marker; scripted keys not
                # mentioned get zero mass.
                scripted = self._scripted[question_id]
                out[question_id] = {key: scripted.get(key, 0.0) for key in compiled.criteria}
            else:
                out[question_id] = {key: 1.0 / len(compiled.criteria) for key in compiled.criteria}
        return out


ABSTAIN_DIST = {ABSTAIN_INSUFFICIENT_EVIDENCE: 0.9, "alpha": 0.06, "beta": 0.04}
