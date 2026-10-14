"""Backend driver abstraction — the seam that adapts any System 1 model.

A driver wraps one model family and implements exactly two operations:

- ``pin_state`` — encode/pin the canonical state once and hand back a handle
  whose lifetime is the ``with`` block. On KV-pinnable local backends (Kev via
  MLX/Metal, ONNX encoders, llama.cpp) this is a real prefix-pinned KV cache;
  on hosted APIs it caches the serialized canonical state so waves within one
  request reuse it. Either way the harness code is identical.
- ``score_branches`` — one parallel fan-out of question branches against the
  pinned state, returning ``{question_id: {option_key: probability}}``.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from ..specs import CompiledQuestion


class BackendError(RuntimeError):
    """A backend violated its contract or failed at the transport layer."""


@dataclass(frozen=True, slots=True)
class StateHandle:
    """Pinned state: the content hash plus the canonical envelope itself."""

    state_hash: str
    canonical_state: Mapping[str, Any]


@runtime_checkable
class BackendDriver(Protocol):
    model_id: str

    def pin_state(
        self, canonical_state: Mapping[str, Any], state_hash: str
    ) -> AbstractContextManager[StateHandle]: ...

    def score_branches(
        self, handle: StateHandle, batch: Mapping[str, CompiledQuestion]
    ) -> dict[str, dict[str, float]]: ...
