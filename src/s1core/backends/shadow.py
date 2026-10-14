"""Shadow-traffic tee (design pattern #4: distillation & auto-calibration).

Wraps a primary driver and mirrors a sampled fraction of live batches to a
secondary driver (typically a cheap local model) with zero application-code
changes. Paired primary/shadow distributions land in a callback; feed them to
:mod:`s1core.calibration` once enough ratified traces accumulate, then promote
the local model by making it primary.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any, Callable, Mapping

from .base import StateHandle

ShadowCallback = Callable[
    [str, Mapping[str, Any], dict[str, dict[str, float]], dict[str, dict[str, float]]],
    None,
]
# (state_hash, batch, shadow_results, primary_results)


class ShadowTeeDriver:
    """Pass-through primary driver that samples ``rate`` of batches to ``shadow``."""

    def __init__(
        self,
        primary: Any,
        shadow: Any,
        *,
        rate: float,
        on_shadow: ShadowCallback,
        seed: int = 0,
    ):
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.model_id = primary.model_id
        self._primary = primary
        self._shadow = shadow
        self._rate = rate
        self._on_shadow = on_shadow
        self._rng = random.Random(seed)

    @contextmanager
    def pin_state(self, canonical_state: Mapping[str, Any], state_hash: str):
        with self._primary.pin_state(canonical_state, state_hash) as handle:
            yield handle

    def score_branches(
        self, handle: StateHandle, batch: Mapping[str, Any]
    ) -> dict[str, dict[str, float]]:
        primary_results = self._primary.score_branches(handle, batch)
        if self._rate > 0.0 and self._rng.random() < self._rate:
            with self._shadow.pin_state(handle.canonical_state, handle.state_hash) as shadow_handle:
                shadow_results = self._shadow.score_branches(shadow_handle, batch)
            self._on_shadow(handle.state_hash, batch, shadow_results, primary_results)
        return primary_results
