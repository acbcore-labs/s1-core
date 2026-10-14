"""End-to-end refund-triage example: all five S1-Core stages in one file.

Run:  python examples/refund_triage.py

The LexicalHeuristicDriver is a TOY backend (lexical overlap, not a model) so
this runs offline. Swap in SystemOneHTTPDriver or your own driver and nothing
else here changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s1core import (  # noqa: E402
    Augmentor,
    ChoiceSpec,
    IsoDateDeltaFacts,
    JsonlAuditLedger,
    NoulSpec,
    RatioFacts,
    ScoreSpec,
    System1Harness,
    driver_from_env,
    load_dotenv,
    replay_verdict,
)
from s1core.backends import LexicalHeuristicDriver  # noqa: E402


def select_backend():
    """Live TypeSafe Jev when TYPESAFE_API_KEY is set (see example.env.txt),
    otherwise the offline toy backend. Nothing else in this file changes."""
    load_dotenv()
    live = driver_from_env()
    if live is not None:
        return live, f"LIVE {live.model_id}"
    return LexicalHeuristicDriver(sharpness=24.0), "TOY LexicalHeuristicDriver"


def main() -> None:
    backend, backend_label = select_backend()
    print(f"backend: {backend_label} — copy example.env.txt to .env with TYPESAFE_API_KEY for the live model")

    # Stage 2 configuration: deterministic facts resolved in native code
    # before any neural pass (Failure Mode #5: models cannot do date deltas
    # or arithmetic in a single prefill pass).
    harness = System1Harness(
        driver=backend,
        audit_sink=JsonlAuditLedger(Path(__file__).resolve().parent / "refund_triage_audit.jsonl"),
        augmentor=Augmentor(
            (IsoDateDeltaFacts(), RatioFacts("requested_refund", "order_total"))
        ),
    )

    # Stage 1 declarations: the model only reads the prose; keys are for
    # your code. Terse prose is a compile error, not a runtime surprise.
    ticket_state = {
        "ticket_body": (
            "Customer says the refund they requested has not arrived and they "
            "need the money back urgently before rent is due."
        ),
        "order_total": 120.0,
        "requested_refund": 150.0,
        "purchase_date": "2026-09-30T09:00:00+00:00",
    }

    questions = [
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
        # Cross-question blindness (Failure Mode #4): this question only runs
        # if refund_lane RATIFIED as agent_review — otherwise SKIPPED_DEPENDENCY.
        NoulSpec(
            question_id="exceeds_policy",
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
            depends_on=("refund_lane", "agent_review"),
        ),
    ]

    verdicts = harness.evaluate(
        ticket_state,
        questions,
        reference_time_iso="2026-10-14T12:00:00+00:00",
        stream_id="ticket-8471",
    )

    print("=== Verdicts (fail-closed: value is None unless RATIFIED) ===")
    for question_id, verdict in verdicts.items():
        print(
            f"{question_id:16s} {verdict.status.name:20s} "
            f"value={verdict.value!r:16} confidence={verdict.confidence:.3f} "
            f"margin={verdict.top2_margin:.3f}"
        )
        if verdict.score_mode_deviation is not None:
            print(f"{'':16s} score_mode_aad={verdict.score_mode_deviation:.3f}")

    print("\n=== Audit ledger: hash-chained, replayable bit-for-bit ===")
    if "TOY" in backend_label:
        print(
            "(exceeds_policy lands in the noul gray zone: a lexical toy cannot "
            "adjudicate the binary proposition, so the harness fails closed — "
            "that is the correct behavior being demonstrated)"
        )
    import json

    ledger_path = Path(__file__).resolve().parent / "refund_triage_audit.jsonl"
    with open(ledger_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    from s1core import verify_chain

    print(f"chain verified: {verify_chain(records)}")
    for record in records:
        qid = record["compiled_question"]["question_id"]
        replayed = replay_verdict(record)
        match = (
            replayed.status.name == record["verdict"]["status"]
            and replayed.value == record["verdict"]["value"]
        )
        print(f"{qid:16s} digest={record['record_digest'][:12]}… replay_ok={match}")


if __name__ == "__main__":
    main()
