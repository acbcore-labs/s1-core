"""The System1Harness: a deterministic control loop around a bounded
stochastic sensor (Stage 2–5 orchestration).

Pipeline per ``evaluate`` call:

1. Deterministic pre-computation (:mod:`s1core.augment`) — arithmetic, dates,
   and structural invariants resolved in native code before any neural pass.
2. Canonical serialization + content hash (:mod:`s1core.hashing`).
3. Prefix-pinned execution — the state is pinned once on the backend; question
   waves (:mod:`s1core.dag`) fan out in parallel under it. Wave N+1 questions
   run only if their Wave <= N parents RATIFIED with the required values
   (children of unmatched parents are reported as SKIPPED_DEPENDENCY, never
   silently dropped). Returned distributions are validated against the
   declared output space — both permutation orders individually before any
   debias averaging.
4. Calibration + admissibility gates (:mod:`s1core.gates`) — every verdict
   fails closed: no value unless RATIFIED. Optional dual-permutation
   debiasing averages forward/reversed option orderings and refuses
   positionally unstable distributions.
5. Content-addressed audit (:mod:`s1core.audit`) — one hash-chained record per
   question, replayable bit-for-bit.

Loop guard: consecutive abstaining *question verdicts* are counted per stream
across calls. A stream that reaches the threshold is **latched** — it
initiates no further backend operations (no ``pin_state``, no
``score_branches``) until the application acknowledges via
:meth:`System1Harness.reset_stream`. Work already dispatched is not
cancelled. Non-abstaining evaluated statuses reset the counter but never the
latch; dependency skips neither advance nor reset.
"""

from __future__ import annotations

import dataclasses
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .augment import Augmentor, IsoDateDeltaFacts
from .audit import (
    AuditLedger,
    InMemoryAuditLedger,
    compute_replay_digest_v3,
    make_record,
)
from .dag import compile_waves
from .gates import evaluate, js_divergence_bits, normalize, validate_support
from .hashing import digest_of
from .specs import QuestionPrimitive, SpecError, values_match
from .types import DecisionVerdict, EpistemicStatus

_ABSTENTION_STATUSES = frozenset(
    {EpistemicStatus.ABSTAINED_MODEL, EpistemicStatus.ABSTAINED_GATE}
)


class System1Harness:
    """Generalized System 1 execution harness.

    One harness instance wraps one backend driver; run one instance per stream
    (not thread-safe across streams — the loop guard keeps per-stream counters
    and the tripped-stream latch).
    """

    def __init__(
        self,
        driver: Any,
        audit_sink: AuditLedger | None = None,
        *,
        augmentor: Augmentor | None = None,
    ):
        self.driver = driver
        self.audit_sink: AuditLedger = audit_sink if audit_sink is not None else InMemoryAuditLedger()
        self._augmentor = augmentor if augmentor is not None else Augmentor((IsoDateDeltaFacts(),))
        self._consecutive_abstentions: dict[str, int] = {}
        self._tripped_streams: set[str] = set()

    def reset_stream(self, stream_id: str = "default") -> None:
        """Application acknowledgement that reopens a tripped stream.

        Clears both the loop-guard latch and the consecutive-abstention
        counter, allowing backend dispatch on ``stream_id`` again. This is the
        only way a latched stream re-enters: neither a ratified verdict nor a
        larger ``max_consecutive_abstentions`` clears the latch.
        """
        self._tripped_streams.discard(stream_id)
        self._consecutive_abstentions.pop(stream_id, None)

    def evaluate(
        self,
        raw_state: Any,
        questions: Sequence[QuestionPrimitive],
        *,
        reference_time_iso: str | None = None,
        stream_id: str = "default",
        max_consecutive_abstentions: int = 3,
    ) -> dict[str, DecisionVerdict]:
        if (
            not isinstance(max_consecutive_abstentions, int)
            or isinstance(max_consecutive_abstentions, bool)
            or max_consecutive_abstentions < 1
        ):
            raise ValueError("max_consecutive_abstentions must be a positive integer")
        reference_time = (
            datetime.fromisoformat(reference_time_iso)
            if reference_time_iso is not None
            else datetime.now(timezone.utc)
        )
        canonical_state = self._augmentor.augment(raw_state, reference_time)
        try:
            state_hash = digest_of(canonical_state)
        except TypeError as exc:
            raise SpecError(
                f"state must be canonically serializable (string-keyed, JSON-able): {exc}"
            ) from exc
        waves = compile_waves(questions)
        verdicts: dict[str, DecisionVerdict] = {}

        # Guarantee: once tripped, the stream initiates no further backend
        # operations until reset_stream(). ``pin_state`` encodes/allocates
        # model state, so a stream latched on entry must not enter it either;
        # short-circuit verdicts and their audit records are computed entirely
        # from local state. Work already dispatched is not cancelled.
        with ExitStack() as stack:
            handle = None
            if stream_id not in self._tripped_streams:
                handle = stack.enter_context(self.driver.pin_state(canonical_state, state_hash))
            for wave_index, wave in enumerate(waves):
                active: list[QuestionPrimitive] = []
                for spec in wave:
                    conditions = spec.dependencies()
                    if conditions and not all(
                        self._dependency_satisfied(verdicts, parent, required)
                        for parent, required in conditions
                    ):
                        skipped = self._skipped_verdict(spec, state_hash, wave_index)
                        verdicts[spec.question_id] = skipped
                        self._audit(state_hash, spec, skipped, {}, wave_index, stream_id, canonical_state)
                        continue
                    active.append(spec)
                if not active:
                    continue
                if handle is None or stream_id in self._tripped_streams:
                    for spec in active:
                        self._short_circuit(spec, verdicts, state_hash, canonical_state,
                                            wave_index, stream_id)
                    continue

                batch = {spec.question_id: spec.compile() for spec in active}
                raw_distributions = self.driver.score_branches(handle, batch)

                debias_ids = {
                    spec.question_id
                    for spec in active
                    if spec.policy.debiased_permutation
                }
                reversed_distributions: dict[str, dict[str, float]] = {}
                if debias_ids:
                    reversed_batch = {
                        qid: batch[qid].reversed_criteria() for qid in debias_ids
                    }
                    reversed_distributions = self.driver.score_branches(handle, reversed_batch)

                for spec in active:
                    compiled = batch[spec.question_id]
                    if spec.question_id not in raw_distributions:
                        raise RuntimeError(
                            f"backend did not return a distribution for '{spec.question_id}'"
                        )
                    raw_probs = raw_distributions[spec.question_id]
                    divergence: float | None = None
                    if spec.question_id in debias_ids:
                        reversed_probs = reversed_distributions[spec.question_id]
                        # Both orders must individually carry the declared
                        # support before averaging: two incomplete supports can
                        # union to a complete one, and averaging must never
                        # manufacture a valid distribution from two invalid ones.
                        validate_support(compiled.gate_inputs, raw_probs)
                        validate_support(compiled.gate_inputs, reversed_probs)
                        raw_probs, divergence = self._debias(raw_probs, reversed_probs)

                    digest = compute_replay_digest_v3(
                        state_hash, compiled.to_payload(), self.driver.model_id,
                        compiled.gate_inputs,
                    )
                    verdict = evaluate(compiled.gate_inputs, raw_probs, digest)
                    if divergence is not None:
                        verdict = dataclasses.replace(verdict, permutation_divergence=divergence)
                        if divergence > spec.policy.max_permutation_divergence:
                            verdict = dataclasses.replace(
                                verdict,
                                status=EpistemicStatus.ABSTAINED_GATE,
                                value=None,
                            )
                    verdict = self._loop_guard(
                        verdict, stream_id, max_consecutive_abstentions
                    )
                    verdicts[spec.question_id] = verdict
                    self._audit(state_hash, spec, verdict, raw_probs, wave_index, stream_id, canonical_state)

        return verdicts

    @staticmethod
    def _dependency_satisfied(
        verdicts: Mapping[str, DecisionVerdict], parent: str, required: Any
    ) -> bool:
        parent_verdict = verdicts.get(parent)
        return (
            parent_verdict is not None
            and parent_verdict.status is EpistemicStatus.RATIFIED
            and values_match(parent_verdict.value, required)
        )

    @staticmethod
    def _debias(
        forward: Mapping[str, float], reversed_probs: Mapping[str, float]
    ) -> tuple[dict[str, float], float]:
        fwd = normalize(forward)
        rev = normalize(reversed_probs)
        # Deterministic key order (forward first, then reversed-only extras) —
        # never set iteration, which varies across processes.
        keys = list(dict.fromkeys([*fwd.keys(), *rev.keys()]))
        averaged = {k: 0.5 * (fwd.get(k, 0.0) + rev.get(k, 0.0)) for k in keys}
        return averaged, js_divergence_bits(fwd, rev)

    def _loop_guard(
        self, verdict: DecisionVerdict, stream_id: str, max_consecutive: int
    ) -> DecisionVerdict:
        if verdict.status in _ABSTENTION_STATUSES:
            count = self._consecutive_abstentions.get(stream_id, 0) + 1
            self._consecutive_abstentions[stream_id] = count
            if count >= max_consecutive:
                # Latch the stream. Only reset_stream() clears it: verdicts
                # from already-dispatched waves reset the *counter* below but
                # can never reopen the stream mid-call or on later calls.
                self._tripped_streams.add(stream_id)
                return dataclasses.replace(
                    verdict, status=EpistemicStatus.LOOP_GUARD_TRIPPED, value=None
                )
        else:
            self._consecutive_abstentions[stream_id] = 0
        return verdict

    def _short_circuit(
        self,
        spec: QuestionPrimitive,
        verdicts: dict[str, DecisionVerdict],
        state_hash: str,
        canonical_state: Mapping[str, Any],
        wave_index: int,
        stream_id: str,
    ) -> None:
        """Emit an audited LOOP_GUARD_TRIPPED verdict with no backend call."""
        compiled = spec.compile()
        digest = compute_replay_digest_v3(
            state_hash, compiled.to_payload(), self.driver.model_id, compiled.gate_inputs
        )
        verdict = DecisionVerdict(
            question_id=spec.question_id,
            status=EpistemicStatus.LOOP_GUARD_TRIPPED,
            value=None,
            raw_value=None,
            probabilities={},
            confidence=0.0,
            top2_margin=0.0,
            normalized_entropy=0.0,
            replay_digest=digest,
        )
        verdicts[spec.question_id] = verdict
        self._audit(state_hash, spec, verdict, {}, wave_index, stream_id, canonical_state,
                    loop_guard_short_circuit=True)

    def _skipped_verdict(
        self, spec: QuestionPrimitive, state_hash: str, wave_index: int
    ) -> DecisionVerdict:
        compiled = spec.compile()
        digest = compute_replay_digest_v3(
            state_hash, compiled.to_payload(), self.driver.model_id, compiled.gate_inputs
        )
        return DecisionVerdict(
            question_id=spec.question_id,
            status=EpistemicStatus.SKIPPED_DEPENDENCY,
            value=None,
            raw_value=None,
            probabilities={},
            confidence=0.0,
            top2_margin=0.0,
            normalized_entropy=0.0,
            replay_digest=digest,
        )

    def _audit(
        self,
        state_hash: str,
        spec: QuestionPrimitive,
        verdict: DecisionVerdict,
        raw_probs: Mapping[str, float],
        wave_index: int,
        stream_id: str,
        canonical_state: Mapping[str, Any],
        *,
        loop_guard_short_circuit: bool = False,
    ) -> None:
        compiled = spec.compile()
        entry = make_record(
            stream_id=stream_id,
            wave_index=wave_index,
            model_id=self.driver.model_id,
            state_hash=state_hash,
            canonical_state=canonical_state,
            compiled_payload=compiled.to_payload(),
            gate_inputs=compiled.gate_inputs,
            raw_probabilities=raw_probs,
            verdict=verdict,
            loop_guard_short_circuit=loop_guard_short_circuit,
        )
        self.audit_sink.record(entry)
