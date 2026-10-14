"""Stage 2: deterministic fact preprocessing (Pillar III — The Third Door).

System 1 models have zero scratchpad tokens and cannot reliably compute
arithmetic, calendar deltas, or multi-hop logic in a single prefill pass
(Failure Mode #5). This stage resolves those facts in native code — zero
neural cost — and appends a compact ``@deterministic_facts`` block to the
canonical state before encoding.

The harness ships general-purpose extractors; project-specific extractors are
one small class (or one callable via :class:`CallableExtractor`) away. Nothing
here is use-case-specific by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

FACTS_KEY = "@deterministic_facts"


class StateError(TypeError):
    """The raw state cannot be canonically serialized."""


@dataclass(frozen=True, slots=True)
class AugmentationContext:
    reference_time_utc: datetime


class FactExtractor(Protocol):
    def extract(
        self, state: Mapping[str, Any], facts: dict[str, Any], ctx: AugmentationContext
    ) -> None:
        """Read ``state`` and write derived facts into ``facts``. Must be pure."""


class CallableExtractor:
    """Adapt any ``fn(state, facts, ctx) -> None`` callable as an extractor."""

    def __init__(self, fn: Callable[[Mapping[str, Any], dict[str, Any], AugmentationContext], None]):
        self._fn = fn

    def extract(self, state: Mapping[str, Any], facts: dict[str, Any], ctx: AugmentationContext) -> None:
        self._fn(state, facts, ctx)


class IsoDateDeltaFacts:
    """Emit calendar-day deltas from the reference time for ISO-8601 date fields.

    With explicit ``keys``, those fields are parsed unconditionally. Without,
    string values under date-ish keys (name contains ``date`` or ends with
    ``_at``) are attempted and silently skipped when unparseable. Fact name:
    ``<key><suffix>``, default ``<key>__days_from_reference`` (signed int,
    UTC calendar days).
    """

    def __init__(
        self,
        keys: Sequence[str] | None = None,
        *,
        suffix: str = "__days_from_reference",
    ):
        self._keys = tuple(keys) if keys is not None else None
        self._suffix = suffix

    @staticmethod
    def _is_date_ish(key: str) -> bool:
        k = key.lower()
        return "date" in k or k.endswith("_at") or k.endswith("_time")

    def extract(self, state: Mapping[str, Any], facts: dict[str, Any], ctx: AugmentationContext) -> None:
        for key, value in state.items():
            if self._keys is not None:
                if key not in self._keys:
                    continue
            elif not isinstance(key, str) or not self._is_date_ish(key):
                continue
            if not isinstance(value, str):
                continue
            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            facts[key + self._suffix] = (dt.date() - ctx.reference_time_utc.date()).days


class RatioFacts:
    """Emit a numeric ratio and an optional boolean threshold flag.

    Generalizes the design doc's refund example: with
    ``RatioFacts("requested_refund", "order_total")`` the facts block gains
    ``requested_refund_over_order_total`` (float) and
    ``requested_refund_over_order_total__exceeds_1`` (bool).
    """

    def __init__(
        self,
        numerator_key: str,
        denominator_key: str,
        *,
        name: str | None = None,
        exceeds: float | None = 1.0,
        zero_floor: float = 1e-9,
        round_digits: int = 6,
    ):
        self._num = numerator_key
        self._den = denominator_key
        self._name = name
        self._exceeds = exceeds
        self._zero_floor = zero_floor
        self._round_digits = round_digits

    def extract(self, state: Mapping[str, Any], facts: dict[str, Any], ctx: AugmentationContext) -> None:
        if self._num not in state or self._den not in state:
            return
        try:
            num = float(state[self._num])
            den = max(float(state[self._den]), self._zero_floor)
        except (TypeError, ValueError):
            return
        ratio = num / den
        name = self._name or f"{self._num}_over_{self._den}"
        facts[name] = round(ratio, self._round_digits)
        if self._exceeds is not None:
            facts[f"{name}__exceeds_{self._exceeds:g}"] = ratio > self._exceeds


class Augmentor:
    """Builds the canonical state envelope: ``{"raw_state": ..., FACTS_KEY: {...}}``."""

    def __init__(self, extractors: Sequence[FactExtractor] = ()):
        self._extractors = tuple(extractors)

    def augment(self, raw_state: Any, reference_time_utc: datetime) -> dict[str, Any]:
        if reference_time_utc.tzinfo is None:
            reference_time_utc = reference_time_utc.replace(tzinfo=timezone.utc)
        envelope: dict[str, Any] = {"raw_state": raw_state, FACTS_KEY: {}}
        if isinstance(raw_state, Mapping):
            facts: dict[str, Any] = envelope[FACTS_KEY]
            ctx = AugmentationContext(reference_time_utc)
            for extractor in self._extractors:
                extractor.extract(raw_state, facts, ctx)
        return envelope
