"""Promotion gate — certify a candidate model against the production ledger.

Design pattern: model-swap certification. A production audit ledger is not
just a compliance artifact; it is a frozen evaluation set. The licence law
this operationalizes: a candidate may be promoted only where it re-derives
every pinned outcome with ``wrong == 0`` on the cases the current model is
licensed to serve.

Flow (three phases, all deterministic):

1. BASELINE — the current production model evaluates a pinned case set
   through the full harness into a JSONL ledger. In production these records
   already exist; this phase only stands in for history.
2. PREFLIGHT — the ledger must be a valid licence before anything certifies
   against it: ``verify_chain`` must pass and every recorded verdict must
   replay bit-for-bit (``replay_verdict``, wrong == 0).
3. CERTIFY — each candidate re-evaluates the same raw states under the same
   specs. A candidate is PROMOTED iff it ratifies no wrong answer on the
   pinned set. Abstentions are coverage loss, reported, never wrongness.
   Per-question probability drift (JS divergence, bits) and status flips are
   telemetry only — a drift budget would be a tuned threshold, so drift is
   not gated; wrongness is.

Why the demo is deterministic (no live backend): a certificate must
reproduce. The synthetic stand-in drivers derive every distribution from the
case content via blake2b — same state, same probabilities, forever. Swap in
real drivers (hosted or local) and the procedure is unchanged. The two
candidates simulate the classic promotion failure — a checkpoint better on
average that broke one lane — and its clean rebuild.
"""

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hashlib  # noqa: E402
import json  # noqa: E402

from s1core import (  # noqa: E402
    ABSTAIN_INSUFFICIENT_EVIDENCE,
    ABSTAIN_OUT_OF_DOMAIN,
    Augmentor,
    ChoiceSpec,
    IsoDateDeltaFacts,
    JsonlAuditLedger,
    NoulSpec,
    RatioFacts,
    System1Harness,
    js_divergence_bits,
    normalize,
    replay_verdict,
    values_match,
    verify_chain,
)
from s1core.backends.base import StateHandle  # noqa: E402
from s1core.types import EpistemicStatus  # noqa: E402

REFERENCE_TIME_ISO = "2026-10-15T12:00:00+00:00"
ESCAPE_KEYS = (ABSTAIN_INSUFFICIENT_EVIDENCE, ABSTAIN_OUT_OF_DOMAIN)

# (case_id, order_total, requested_refund, account_age_days, prior_chargebacks)
CASES = [
    ("ord-4401", 400.0, 100.0, 410, 0),
    ("ord-4402", 220.0, 110.0, 90, 0),     # exactly at both auto-approve edges
    ("ord-4403", 300.0, 158.0, 260, 0),    # ratio 0.527, just past the auto edge
    ("ord-4404", 500.0, 365.0, 700, 0),
    ("ord-4405", 100.0, 140.0, 55, 0),     # exceeds total: deny
    ("ord-4406", 120.0, 140.0, 320, 0),    # exceeds total, less blatantly
    ("ord-4407", 200.0, 60.0, 150, 2),     # repeat chargebacks: deny
    ("ord-4408", 150.0, 45.0, 30, 0),      # young account: agent review
    ("ord-4409", 700.0, 630.0, 900, 1),
    ("ord-4410", 90.0, 36.0, 45, 0),
    ("ord-4411", 260.0, 60.0, 500, 0),
    ("ord-4412", 1000.0, 940.0, 120, 0),   # ratio 0.94: review, near the deny edge
]


def pinned_exceeds(case: dict) -> bool:
    return case["requested_refund"] > case["order_total"]


def pinned_lane(case: dict) -> str:
    ratio = case["requested_refund"] / case["order_total"]
    if ratio > 1.0 or case["prior_chargebacks"] >= 2:
        return "deny"
    if (ratio <= 0.5 and case["account_age_days"] >= 90
            and case["prior_chargebacks"] == 0):
        return "auto_approve"
    return "agent_review"


def lane_margin(case: dict) -> float:
    """Distance to the nearest rule edge, normalized to [0, 1] — the stand-in
    for how far a case sits from a decision boundary."""
    ratio = case["requested_refund"] / case["order_total"]
    edges = [abs(ratio - 1.0) / 0.5, abs(ratio - 0.5) / 0.5,
             abs(case["account_age_days"] - 90) / 210.0]
    if case["prior_chargebacks"]:
        edges.append(min(case["prior_chargebacks"], 2) / 2.0)
    return min(edges)


def case_state(case) -> dict:
    case_id, total, refund, age, chargebacks = case
    return {
        "case_id": case_id,
        "order_total": total,
        "requested_refund": refund,
        "account_age_days": age,
        "prior_chargebacks": chargebacks,
    }


def build_specs():
    return [
        ChoiceSpec(
            question_id="refund_lane",
            instructions=(
                "Given this refund request and the deterministic facts about the "
                "order, which fulfillment lane should handle it?"
            ),
            options={
                "auto_approve": (
                    "Approve automatically because the refund is small relative to "
                    "the order total and the account history is clean."
                ),
                "agent_review": (
                    "A human agent must review because the amount, account age, or "
                    "dispute history needs judgment beyond the policy rule."
                ),
                "deny": (
                    "Deny because the request exceeds the order total or the account "
                    "shows repeat chargeback abuse."
                ),
            },
        ),
        NoulSpec(
            question_id="exceeds_total",
            instructions=(
                "Does the requested refund exceed the full order total for this "
                "purchase? The facts block pre-computes the refund-to-total ratio."
            ),
            true_criteria="The requested refund amount is greater than the order total.",
            false_criteria="The requested refund amount is within the order total.",
        ),
    ]


class SyntheticJev:
    """Stand-in sensor: distributions derived from case content via blake2b.

    Quality tracks the distance to the nearest rule edge (real sensors are
    weakest at boundaries too). Same state in, same probabilities out — the
    property a certification run requires. The optional ``lane_regression``
    simulates the classic bad candidate: better on average, but confidently
    wrong on one lane (over-total refunds routed to review instead of deny).
    """

    def __init__(self, model_id: str, *, quality_lift: float = 0.0,
                 lane_regression: bool = False, salt: str = "prod"):
        self.model_id = model_id
        self._lift = quality_lift
        self._lane_regression = lane_regression
        self._salt = salt

    def _draw(self, state_hash: str, tag: str) -> float:
        digest = hashlib.blake2b(
            f"{self._salt}|{tag}|{state_hash}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") / 2**64  # uniform [0, 1)

    @contextmanager
    def pin_state(self, canonical_state, state_hash: str):
        yield StateHandle(state_hash=state_hash, canonical_state=canonical_state)

    def _peaked(self, handle, question_id: str, truth: str, p_correct: float):
        jitter = (self._draw(handle.state_hash, question_id) - 0.5) * 0.06
        q = min(0.97, max(0.35, p_correct + self._lift + jitter))
        escape = 0.02 if q > 0.7 else 0.05
        return q, escape

    def score_branches(self, handle, batch):
        case = handle.canonical_state["raw_state"]
        out = {}
        for question_id, compiled in batch.items():
            if question_id == "refund_lane":
                truth = pinned_lane(case)
                p_correct = 0.45 + 0.50 * min(1.0, lane_margin(case))
                if self._lane_regression and case["requested_refund"] > case["order_total"]:
                    out[question_id] = {
                        "auto_approve": 0.02, "agent_review": 0.88, "deny": 0.06,
                        ESCAPE_KEYS[0]: 0.02, ESCAPE_KEYS[1]: 0.02,
                    }
                    continue
                q, escape = self._peaked(handle, question_id, truth, p_correct)
                others = [k for k in compiled.criteria
                          if k != truth and k not in ESCAPE_KEYS]
                dist = {k: 0.0 for k in compiled.criteria}
                dist[truth] = q
                for key in ESCAPE_KEYS:
                    dist[key] = escape / len(ESCAPE_KEYS)
                for key in others:
                    dist[key] = (1.0 - q - escape) / len(others)
                out[question_id] = dist
            else:  # exceeds_total (noul)
                truth = pinned_exceeds(case)
                ratio = case["requested_refund"] / case["order_total"]
                margin = min(1.0, abs(ratio - 1.0) / 0.4)
                p_correct = 0.45 + 0.50 * margin
                q, _ = self._peaked(handle, question_id, "true" if truth else "false", p_correct)
                p_true = q if truth else 1.0 - q
                out[question_id] = {"true": p_true, "false": 1.0 - p_true}
        return out


def load_ledger(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def replay_wrong_count(records: list[dict]) -> int:
    wrong = 0
    for record in records:
        replayed = replay_verdict(record)
        stored = record["verdict"]
        if replayed.status.value != stored["status"] or replayed.value != stored["value"]:
            wrong += 1
    return wrong


def certify(candidate: SyntheticJev, specs, records: list[dict]) -> bool:
    """Re-evaluate the pinned cases under the candidate and issue a verdict.

    Certificate law: wrong == 0, where wrong means the candidate RATIFIED a
    value that differs from the pinned outcome. Abstention is coverage loss.
    Baseline wrongs are context only — the certificate binds the candidate.
    """
    harness = System1Harness(candidate, augmentor=AUGMENTOR)
    baseline = {
        (r["canonical_state"]["raw_state"]["case_id"],
         r["compiled_question"]["question_id"]): r
        for r in records
    }
    stats: dict[str, dict] = {}
    wrongs: list[str] = []
    pinned = 0
    ratified = 0
    for case in CASES:
        state = case_state(case)
        verdicts = harness.evaluate(
            state, specs, reference_time_iso=REFERENCE_TIME_ISO,
            stream_id=case[0],
        )
        for spec in specs:
            verdict = verdicts[spec.question_id]
            state = case_state(case)
            truth = (pinned_lane(state) if isinstance(spec, ChoiceSpec)
                     else pinned_exceeds(state))
            pinned += 1
            key = (case[0], spec.question_id)
            base = baseline[key]
            stat = stats.setdefault(spec.question_id, {
                "wrong": 0, "ratified": 0, "drift": [], "flips": 0})
            if verdict.status is EpistemicStatus.RATIFIED:
                ratified += 1
                stat["ratified"] += 1
                if not values_match(verdict.value, truth):
                    stat["wrong"] += 1
                    wrongs.append(
                        f"{spec.question_id}/{case[0]} ratified {verdict.value!r}, "
                        f"pinned {truth!r}")
            stat["drift"].append(js_divergence_bits(
                normalize(base["raw_probabilities"]), verdict.probabilities))
            if base["verdict"]["status"] != verdict.status.name:
                stat["flips"] += 1

    promoted = not wrongs
    print(f"\ncandidate {candidate.model_id}"
          f"{' (lane regression injected)' if candidate._lane_regression else ''}")
    for question_id, stat in stats.items():
        drift = stat["drift"]
        print(f"  {question_id:14s} wrong={stat['wrong']}  "
              f"ratified {stat['ratified']}/{pinned // len(stats)}  "
              f"drift mean {sum(drift) / len(drift):.3f} bits, max {max(drift):.3f}  "
              f"status-flips {stat['flips']}")
    if promoted:
        print(f"  CERTIFICATE: PROMOTE — wrong=0 on {pinned} pinned case-questions "
              f"(ratified {ratified}, abstained {pinned - ratified}; "
              f"abstention is coverage loss, not wrongness)")
    else:
        print(f"  CERTIFICATE: REJECT — wrong={len(wrongs)} (must be 0): "
              f"{'; '.join(wrongs)}")
    return promoted


AUGMENTOR = Augmentor((IsoDateDeltaFacts(), RatioFacts("requested_refund", "order_total")))


def main() -> None:
    specs = build_specs()

    with tempfile.TemporaryDirectory(prefix="s1core_model_swap_") as tmp:
        # Phase 1 — BASELINE: production model writes the licence ledger.
        ledger_path = Path(tmp) / "baseline_ledger.jsonl"
        baseline = SyntheticJev("prod-jev-snapshot")
        harness = System1Harness(
            baseline, audit_sink=JsonlAuditLedger(ledger_path), augmentor=AUGMENTOR,
        )
        for case in CASES:
            harness.evaluate(
                case_state(case), specs,
                reference_time_iso=REFERENCE_TIME_ISO, stream_id=case[0],
            )
        records = load_ledger(ledger_path)

        # Phase 2 — PREFLIGHT: the ledger must be a valid licence.
        chain_ok = verify_chain(records)
        replay_wrong = replay_wrong_count(records)
        baseline_wrong = []
        for record in records:
            case_id = record["canonical_state"]["raw_state"]["case_id"]
            case = case_state(next(c for c in CASES if c[0] == case_id))
            spec = next(s for s in specs
                        if s.question_id == record["compiled_question"]["question_id"])
            truth = pinned_lane(case) if isinstance(spec, ChoiceSpec) else pinned_exceeds(case)
            stored = record["verdict"]
            if (stored["status"] == "RATIFIED"
                    and not values_match(stored["value"], truth)):
                baseline_wrong.append(case_id)
        print("=== model-swap certification ===")
        print(f"licence preflight: chain verified={chain_ok}  "
              f"replay {len(records) - replay_wrong}/{len(records)} wrong={replay_wrong}")
        print(f"baseline prod-jev-snapshot: wrong-while-ratified {len(baseline_wrong)} "
              f"(context only — the certificate binds the candidate)")
        if not chain_ok or replay_wrong:
            raise SystemExit("ledger failed preflight: not a valid licence")

        # Phase 3 — CERTIFY both candidates against the same licence.
        certify(SyntheticJev("jev-next-rc1", quality_lift=0.22,
                             lane_regression=True, salt="rc1"),
                specs, records)
        certify(SyntheticJev("jev-next-rc2", quality_lift=0.22,
                             lane_regression=False, salt="rc1"),
                specs, records)


if __name__ == "__main__":
    main()
