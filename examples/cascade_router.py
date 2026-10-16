"""Cascade router — the admissibility gate as a cost dial.

Design pattern: System 1 fast lane with System 2 escalation. A frontier
reasoning model resolves anything, at frontier latency and cost. A System 1
decider resolves most things in one ~100 ms pass — if its answer is
admissible. The harness gate is the dial between them: RATIFIED ships the
cheap answer; every other status escalates to System 2. Sweeping the gate
tiers over one labeled case set turns "how aggressive can we be" into a
measured table instead of a vibe: per tier — handled share, customer-facing
wrong answers, overall accuracy, average latency, and cost per 1,000 cases.

The dial has four settings here: three hand-declared GatePolicies
(permissive / default / strict, the same axes the proof scorecard sweeps)
plus one fitted tier — conformal tau at α = 0.10 from a disjoint
calibration window, which is what the nightly recalibration loop
(``nightly_recalibration.py``) produces. The operations loop is:
calibrate nightly, route daily.

Declared stand-ins (replace with your provider's real numbers): the S1
latency/token figures are the live TypeSafe measurements from the README
proof run; the System 2 figures are typical frontier pricing. Escalated
cases are modeled as resolved correctly by construction — System 2 has its
own error rate, measurable the same way (its own ledger); the scorecard
isolates the S1 gate's contribution.

Deterministic end to end: the case set is seeded, the sensor derives every
distribution from case content via blake2b. No live backend — the routing
logic is unchanged under real drivers.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hashlib  # noqa: E402
import random  # noqa: E402

from s1core import (  # noqa: E402
    Augmentor,
    ChoiceSpec,
    GatePolicy,
    IsoDateDeltaFacts,
    RatioFacts,
    System1Harness,
    fit_gate_policy,
    values_match,
)
from s1core.backends.base import StateHandle  # noqa: E402
from s1core.types import EpistemicStatus  # noqa: E402

REFERENCE_ISO = "2026-10-15T12:00:00+00:00"
ROUTED_CASES = 80
CALIBRATION_CASES = 120

# --- declared cost/latency stand-ins (replace with your provider's numbers) ---
S1_LATENCY_MS = 110          # live TypeSafe measured: 112 ms avg per call
S1_TOKENS_IN = 264           # 1,584 tokens / 6 questions, README proof run
S1_PRICE_PER_MTOK = 0.042    # input-only billing, as advertised
S2_LATENCY_MS = 2400
S2_TOKENS_IN, S2_IN_PRICE = 2900, 2.50
S2_TOKENS_OUT, S2_OUT_PRICE = 350, 10.00

S1_COST = S1_TOKENS_IN * S1_PRICE_PER_MTOK / 1e6
S2_COST = (S2_TOKENS_IN * S2_IN_PRICE + S2_TOKENS_OUT * S2_OUT_PRICE) / 1e6


def make_cases(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        total = rng.choice([80.0, 120.0, 200.0, 300.0, 500.0, 800.0])
        # Case mix: mostly clear-cut, some medium, a real boundary slice —
        # a routing tier is only interesting when the traffic spans the dial.
        ratio = rng.choice([0.15, 0.3, 0.45, 0.5, 0.55, 0.8, 0.97, 1.05, 1.6, 2.2])
        cases.append({
            "ticket_id": f"r-{seed}-{i:03d}",
            "order_total": total,
            "requested_refund": round(total * ratio, 2),
            "account_age_days": rng.choice([15, 60, 200, 800]),
            "prior_chargebacks": rng.choices([0, 1, 2], weights=[0.85, 0.1, 0.05])[0],
        })
    return cases


def pinned_lane(case: dict) -> str:
    ratio = case["requested_refund"] / case["order_total"]
    if ratio > 1.0 or case["prior_chargebacks"] >= 2:
        return "deny"
    if ratio <= 0.5 and case["account_age_days"] >= 90 and case["prior_chargebacks"] == 0:
        return "auto_approve"
    return "agent_review"


def lane_margin(case: dict) -> float:
    ratio = case["requested_refund"] / case["order_total"]
    edges = [abs(ratio - 1.0) / 0.5, abs(ratio - 0.5) / 0.5,
             abs(case["account_age_days"] - 90) / 210.0]
    if case["prior_chargebacks"]:
        edges.append(min(case["prior_chargebacks"], 2) / 2.0)
    return min(edges)


class CascadeSensor:
    """The System 1 decider: quality tracks boundary distance, same law real
    sensors obey — confident on clear cases, dithering near rule edges — and
    on confused cases it dithers between two plausible lanes instead of
    committing to either. What the gate does with that dither is the dial
    this scorecard sweeps."""

    model_id = "cascade-jev-2026.10"
    FLIP_RATE = 0.10

    @staticmethod
    def _draw(state_hash: str, tag: str) -> float:
        digest = hashlib.blake2b(
            f"{CascadeSensor.model_id}|{tag}|{state_hash}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") / 2**64  # uniform [0, 1)

    @staticmethod
    def _confusion(truth: str) -> str:
        return "agent_review" if truth == "auto_approve" else "auto_approve"

    @contextmanager
    def pin_state(self, canonical_state, state_hash: str):
        yield StateHandle(state_hash=state_hash, canonical_state=canonical_state)

    def score_branches(self, handle, batch):
        case = handle.canonical_state["raw_state"]
        truth = pinned_lane(case)
        flip = self._draw(handle.state_hash, "flip") < self.FLIP_RATE
        out = {}
        for question_id, compiled in batch.items():
            dist = {k: 0.0 for k in compiled.criteria}
            for k in compiled.criteria:
                if k.startswith("__abstain"):
                    dist[k] = 0.03
            if flip:
                # Dithering confusion: near-tied pair — no lane is defensible.
                other = self._confusion(truth)
                rest = [k for k in compiled.criteria
                        if k not in (truth, other) and not k.startswith("__abstain")]
                dist[truth] = 0.40
                dist[other] = 0.44
                for k in rest:
                    dist[k] = 0.05
            else:
                jitter = (self._draw(handle.state_hash, question_id) - 0.5) * 0.06
                q = min(0.97, max(0.38, 0.45 + 0.50 * lane_margin(case) + jitter))
                escape = 0.02 if q > 0.7 else 0.05
                others = [k for k in compiled.criteria
                          if not k.startswith("__abstain") and k != truth]
                dist[truth] = q
                for key in others:
                    dist[key] = (1.0 - q - escape) / len(others)
                for k in compiled.criteria:
                    if k.startswith("__abstain"):
                        dist[k] = escape / 2
            out[question_id] = dist
        return out


def lane_spec(policy: GatePolicy) -> ChoiceSpec:
    return ChoiceSpec(
        question_id="refund_lane",
        policy=policy,
        instructions=(
            "Given this refund ticket and the deterministic facts about the "
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
    )


AUGMENTOR = Augmentor((IsoDateDeltaFacts(), RatioFacts("requested_refund", "order_total")))


def fit_conformal_tier(cases: list[dict]) -> GatePolicy:
    """The tier the nightly loop would have shipped: temperature on the
    older 60% of the calibration window, conformal tau at α = 0.10 on the
    newer 40% (disjoint, as the fit requires). The (raw distribution, label)
    samples come straight from the in-memory audit records."""
    harness = System1Harness(CascadeSensor(), augmentor=AUGMENTOR)
    spec = lane_spec(GatePolicy())
    for case in cases:
        harness.evaluate(
            case, [spec], reference_time_iso=REFERENCE_ISO,
            stream_id=case["ticket_id"],
        )
    samples = [(record.raw_probabilities, pinned_lane(record.canonical_state["raw_state"]))
               for record in harness.audit_sink.records]
    split = (len(samples) * 3) // 5
    return fit_gate_policy(samples[:split], calibration_samples=samples[split:], alpha=0.10)


def route(cases: list[dict], policy: GatePolicy) -> dict:
    harness = System1Harness(CascadeSensor(), augmentor=AUGMENTOR)
    spec = lane_spec(policy)
    handled = shipped_wrong = escalated = 0
    for case in cases:
        verdict = harness.evaluate(
            case, [spec], reference_time_iso=REFERENCE_ISO,
            stream_id=case["ticket_id"],
        )["refund_lane"]
        if verdict.status is EpistemicStatus.RATIFIED:
            handled += 1
            if not values_match(verdict.value, pinned_lane(case)):
                shipped_wrong += 1
        else:
            escalated += 1  # gray zone, model escape hatch, or gate refusal
    return {"handled": handled, "shipped_wrong": shipped_wrong, "escalated": escalated}


def main() -> None:
    calibration = make_cases(CALIBRATION_CASES, seed=771)
    routed = make_cases(ROUTED_CASES, seed=882)
    tiers = [
        ("permissive", GatePolicy(min_top1_prob=0.40, min_top2_margin=0.02,
                                  max_normalized_entropy=0.85)),
        ("default", GatePolicy()),
        ("strict", GatePolicy(min_top1_prob=0.85, min_top2_margin=0.45,
                              max_normalized_entropy=0.25)),
        ("conformal@α=0.10", fit_conformal_tier(calibration)),
    ]

    print("=== cascade router — the admissibility gate as a cost dial ===")
    print(f"routed cases: {ROUTED_CASES}  calibration window: {CALIBRATION_CASES} "
          f"(fitted tier: T={tiers[3][1].temperature:.3f}, "
          f"tau={tiers[3][1].conformal_tau:.3f})")
    print(f"\n{'tier':18s} {'S1 handled':>10s} {'shipped wrong':>14s} "
          f"{'overall acc':>12s} {'avg latency':>12s} {'$ / 1k cases':>13s}")
    zero_wrong = []
    for name, policy in tiers:
        r = route(routed, policy)
        accuracy = (r["handled"] - r["shipped_wrong"] + r["escalated"]) / len(routed)
        latency = (r["handled"] * S1_LATENCY_MS
                   + r["escalated"] * (S1_LATENCY_MS + S2_LATENCY_MS)) / len(routed)
        cost = (r["handled"] * S1_COST
                + r["escalated"] * (S1_COST + S2_COST)) * 1000 / len(routed)
        if r["shipped_wrong"] == 0:
            zero_wrong.append((cost, name))
        print(f"{name:18s} {r['handled'] / len(routed):9.1%} "
              f"{r['shipped_wrong']:14d} {accuracy:11.1%} {latency:10.0f} ms "
              f"{cost:12.4f}")
    cheapest_safe = min(zero_wrong) if zero_wrong else None
    print(f"\ncheapest tier that ships zero wrong answers: "
          f"{cheapest_safe[1]} at ${cheapest_safe[0]:.4f} per 1,000 cases"
          if cheapest_safe else
          "\nno tier ships zero wrong answers on this window — do not automate")
    print("assumptions: escalated cases are modeled as correctly resolved by "
          "System 2 (measure its error rate the same way, from its own "
          "ledger); S1 cost/latency are the README live-run figures, S2 are "
          "declared stand-ins for your provider's pricing.")


if __name__ == "__main__":
    main()
