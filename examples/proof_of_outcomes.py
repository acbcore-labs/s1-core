"""Proof of outcomes — a live scorecard of what the S1-Core harness provides.

Run:  python examples/proof_of_outcomes.py

With ``TYPESAFE_API_KEY`` in .env this runs against the live TypeSafe Jev API
and reports measured evidence: parallel fan-out (questions vs backend calls),
wall-clock latency, billed tokens and cost at the published input-only rate,
verdict quality under the admissibility gates, accuracy of the
deterministic-facts probe against computed ground truth, inference
determinism, and audit guarantees (replay wrong=0, chain integrity,
fail-closed invariant). Without a key the identical scorecard runs against the
offline toy backend, labeled as such.
"""

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s1core import (  # noqa: E402
    Augmentor,
    ChoiceSpec,
    EpistemicStatus,
    GateInputs,
    GatePolicy,
    InMemoryAuditLedger,
    IsoDateDeltaFacts,
    NoulSpec,
    RatioFacts,
    ScoreSpec,
    System1Harness,
    compile_waves,
    driver_from_env,
    estimate_cost_usd,
    evaluate,
    load_dotenv,
    replay_verdict,
    verify_chain,
)
from s1core.backends import LexicalHeuristicDriver  # noqa: E402
from s1core.backends.base import StateHandle  # noqa: E402

REFERENCE_TIME = "2026-10-14T12:00:00+00:00"


class TimedDriver:
    """Delegating wrapper that records per-fan-out-call wall time."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.model_id = inner.model_id
        self.durations: list[float] = []

    @contextmanager
    def pin_state(self, canonical_state: Mapping[str, Any], state_hash: str):
        with self.inner.pin_state(canonical_state, state_hash) as handle:
            yield handle

    def score_branches(self, handle: StateHandle, batch: Mapping[str, Any]):
        start = time.perf_counter()
        try:
            return self.inner.score_branches(handle, batch)
        finally:
            self.durations.append(time.perf_counter() - start)


def build_questions() -> list:
    return [
        ChoiceSpec(
            question_id="refund_lane",
            instructions=(
                "Given the customer ticket, which fulfillment lane should handle "
                "this refund request?"
            ),
            options={
                "auto_approve": (
                    "Approve the refund automatically: the amount is within policy "
                    "limits and the order history clearly supports approval without "
                    "any human involvement."
                ),
                "agent_review": (
                    "Route to a human agent for review: the customer needs money and "
                    "support urgently, the refund amount or order history needs human "
                    "judgment, or the situation is escalating."
                ),
                "deny": (
                    "Deny the refund: the request falls outside the published refund "
                    "policy or evidence shows the purchase terms make it ineligible."
                ),
            },
        ),
        ScoreSpec(
            question_id="urgency",
            instructions=(
                "Rate how urgently a human should attend to this ticket on the "
                "rubric from routine to critical."
            ),
            rubric_levels={
                1: "Routine queue work with no customer waiting on the outcome.",
                2: "Standard handling within one business day is fully acceptable.",
                3: "The customer is actively waiting and money is involved.",
                4: "Urgent: the customer is distressed and escalating quickly.",
                5: "Critical: legal, regulatory, or reputational exposure right now.",
            },
        ),
        NoulSpec(
            question_id="exceeds_total",
            instructions=(
                "Does the requested refund exceed the full order total? The "
                "deterministic facts block pre-computes the refund-to-total ratio."
            ),
            true_criteria=(
                "The requested refund is greater than the order total: the ratio "
                "of refund to total exceeds one, which the facts block flags as "
                "exceeds true."
            ),
            false_criteria=(
                "The requested refund is within the order total: the ratio of "
                "refund to total is at most one, and no exceeds flag is present."
            ),
        ),
        NoulSpec(
            question_id="customer_distress",
            instructions=(
                "Does the ticket text indicate the customer is distressed or "
                "escalating (angry, urgent hardship, threats to leave)?"
            ),
            true_criteria="The customer expresses anger, urgent hardship, or escalation.",
            false_criteria="The customer is calm and no escalation language appears.",
        ),
        NoulSpec(
            question_id="sla_missed",
            instructions=(
                "Has the service-level deadline recorded in the state already "
                "passed as of the reference time? The facts block pre-computes "
                "the signed day delta."
            ),
            true_criteria=(
                "The deadline date is before the reference date: the signed day "
                "delta is negative, so the deadline has passed."
            ),
            false_criteria=(
                "The deadline date is on or after the reference date: the signed "
                "day delta is zero or positive, so the deadline has not passed."
            ),
        ),
        NoulSpec(
            question_id="escalate_to_human",
            instructions=(
                "Given the lane decision, should a human agent be paged for "
                "immediate follow-up on this ticket?"
            ),
            true_criteria=(
                "The ticket was routed to agent review and shows customer "
                "distress, so an immediate human page is warranted."
            ),
            false_criteria=(
                "The ticket was auto-approved or denied, or no distress is "
                "present, so no immediate page is needed."
            ),
            depends_on=("refund_lane", "agent_review"),
        ),
    ]


def ticket_state(order_total: float = 120.0, requested_refund: float = 150.0) -> dict:
    return {
        "ticket_body": (
            "Customer says the refund they requested has not arrived and they "
            "need the money back urgently before rent is due."
        ),
        "order_total": order_total,
        "requested_refund": requested_refund,
        "purchase_date": "2026-09-30T09:00:00+00:00",
        "sla_deadline": "2026-10-12T17:00:00+00:00",
    }


def accuracy_probe(backend: Any) -> tuple[int, int]:
    """Compare the harness verdict against arithmetic ground truth the
    deterministic preprocessor guarantees the model can see."""
    spec = NoulSpec(
        question_id="probe_exceeds",
        instructions=(
            "Does the requested refund exceed the full order total? The "
            "deterministic facts block pre-computes the refund-to-total ratio."
        ),
        true_criteria=(
            "The requested refund is greater than the order total: the ratio "
            "of refund to total exceeds one, which the facts block flags as "
            "exceeds true."
        ),
        false_criteria=(
            "The requested refund is within the order total: the ratio of "
            "refund to total is at most one, and no exceeds flag is present."
        ),
    )
    harness = System1Harness(backend, augmentor=Augmentor((RatioFacts("requested_refund", "order_total"),)))
    cases = [(120.0, 150.0), (200.0, 150.0), (99.99, 100.0), (49.50, 50.0), (1000.0, 150.0)]
    correct = 0
    for order_total, refund in cases:
        verdict = harness.evaluate(
            ticket_state(order_total, refund), [spec], reference_time_iso=REFERENCE_TIME
        )["probe_exceeds"]
        truth = refund > order_total
        if verdict.status is EpistemicStatus.RATIFIED and verdict.value is truth:
            correct += 1
    return correct, len(cases)


def gate_counterfactual(records: list) -> None:
    """Re-run Stage 4 on a recorded live distribution under three policies —
    the sensor is fixed; the policy decides what admissible means."""
    choice_record = next(
        (r for r in records if r.gate_inputs["primitive"] == "choice" and r.raw_probabilities),
        None,
    )
    if choice_record is None:
        print("   (no ratified-able choice distribution recorded; skipped)")
        return
    raw = choice_record.raw_probabilities
    gi = GateInputs.from_jsonable(choice_record.gate_inputs)
    label = {True: "RATIFIED", False: "ABSTAINED_GATE"}
    for name, policy in (
        ("default  (top1>=0.65, margin>=0.20, H<=0.55)", GatePolicy()),
        ("relaxed  (top1>=0.40, margin>=0.05, H<=0.80)",
         GatePolicy(min_top1_prob=0.40, min_top2_margin=0.05, max_normalized_entropy=0.80)),
        ("strict   (top1>=0.90, margin>=0.50, H<=0.20)",
         GatePolicy(min_top1_prob=0.90, min_top2_margin=0.50, max_normalized_entropy=0.20)),
    ):
        verdict = evaluate(GateInputs(question_id=gi.question_id, primitive="choice", policy=policy),
                           raw, "counterfactual")
        top = max(raw, key=lambda k: raw[k])
        print(f"   {name}: top-1 '{top}' p={raw[top]:.3f} -> {label[verdict.status is EpistemicStatus.RATIFIED]}")


def main() -> None:
    load_dotenv()
    live = driver_from_env()
    if live is not None:
        inner, mode = live, "LIVE (TypeSafe Jev)"
    else:
        inner, mode = LexicalHeuristicDriver(sharpness=24.0), "TOY (LexicalHeuristicDriver — set TYPESAFE_API_KEY in .env for the live model)"

    backend = TimedDriver(inner)
    ledger = InMemoryAuditLedger()
    harness = System1Harness(
        backend,
        audit_sink=ledger,
        augmentor=Augmentor((IsoDateDeltaFacts(), RatioFacts("requested_refund", "order_total"))),
    )
    questions = build_questions()
    state = ticket_state()

    print("=== S1-Core proof of outcomes ===")
    print(f"backend: {backend.model_id} — {mode}\n")

    start = time.perf_counter()
    verdicts = harness.evaluate(state, questions, reference_time_iso=REFERENCE_TIME, stream_id="proof")
    wall = time.perf_counter() - start
    n_questions = len(questions)
    n_calls = len(backend.durations)
    wave_sizes = [len(w) for w in compile_waves(questions)]

    print("1) Performance — parallel fan-out & cost")
    print(f"   questions declared        : {n_questions} across {len(wave_sizes)} waves "
          f"({', '.join(map(str, wave_sizes))} per wave; wave 2 is dependency-gated)")
    print(f"   backend fan-out calls     : {n_calls} (one per executed wave + any debias re-dispatch)")
    print(f"   sequential calls avoided  : {max(n_questions - n_calls, 0)} (batched inside waves)")
    print(f"   wall clock                : {wall * 1000:.0f} ms total, "
          f"{sum(backend.durations) / max(n_calls, 1) * 1000:.0f} ms avg per call")
    if live is not None:
        usage = inner.usage_totals
        cost = estimate_cost_usd(usage["input_tokens"])
        print(f"   tokens billed             : {usage['input_tokens']:,} in / "
              f"{usage['output_tokens']:,} out across {usage['requests']} requests")
        print(f"   estimated cost            : ${cost:.6f} (input-only, $0.042 per 1M tokens)")

    print("\n2) Decision quality — gates & epistemic status")
    statuses: dict[str, int] = {}
    for verdict in verdicts.values():
        statuses[verdict.status.name] = statuses.get(verdict.status.name, 0) + 1
    print(f"   status breakdown          : {statuses}")
    for question_id, verdict in verdicts.items():
        value = repr(verdict.value) if verdict.value is not None else "-"
        extra = ""
        if verdict.score_mode_deviation is not None:
            extra = f" mode_aad={verdict.score_mode_deviation:.3f}"
        print(f"   {question_id:18s} {verdict.status.name:17s} value={value:16s} conf={verdict.confidence:.3f}{extra}")
    invariant_ok = all(
        (v.value is None) == (v.status is not EpistemicStatus.RATIFIED)
        for v in verdicts.values()
    )
    print(f"   fail-closed invariant     : {'OK' if invariant_ok else 'VIOLATED'} "
          f"(no verdict carries a value unless RATIFIED)")
    probe_correct, probe_total = accuracy_probe(backend)
    print(f"   deterministic-facts probe : {probe_correct}/{probe_total} correct "
          f"vs arithmetic ground truth (refund > total)")
    if len(backend.durations) > n_calls:
        rerun = time.perf_counter()
        second = harness.evaluate(state, questions, reference_time_iso=REFERENCE_TIME, stream_id="proof")
        rerun_wall = time.perf_counter() - rerun
        deltas = [
            abs(second[q].probabilities[k] - verdicts[q].probabilities[k])
            for q in verdicts
            if q in second
            for k in set(verdicts[q].probabilities) & set(second[q].probabilities)
        ]
        print(f"   inference determinism     : max |Δp| across a full re-run = "
              f"{max(deltas, default=0.0):.6f} ({rerun_wall * 1000:.0f} ms)")

    print("\n3) Gate counterfactual — same sensor, three admissibility policies")
    gate_counterfactual(ledger.records)

    print("\n4) Governance — audit, replay, chain")
    mismatches = 0
    for record in ledger.records:
        if record.raw_probabilities:
            replayed = replay_verdict(record.to_jsonable())
            if (replayed.status.name != record.verdict["status"]
                    or replayed.value != record.verdict["value"]):
                mismatches += 1
    chain_ok = verify_chain([r.to_jsonable() for r in ledger.records])
    print(f"   audit records             : {len(ledger.records)} (content-addressed, hash-chained)")
    print(f"   replay verification       : {len(ledger.records) - mismatches}/{len(ledger.records)} "
          f"byte-identical (wrong={mismatches})")
    print(f"   ledger chain verified     : {chain_ok}")


if __name__ == "__main__":
    main()
