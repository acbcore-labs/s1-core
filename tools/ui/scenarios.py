"""Scenario adapters for the S1-Core UI.

Each adapter re-runs a flow from ``examples/`` against in-memory ledgers and
returns a JSON payload the UI renders. Data-bearing pieces (case tables,
synthetic sensors, truth functions) are imported from the example modules;
the flows are re-composed here so the UI gets record-level detail the
print-oriented examples never return.

Division of labor (the law the whole UI obeys): everything computed here
comes from s1core itself — statuses, drift figures, fitted policies, chain
verification. The browser only draws.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_TOOLS_UI = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_UI.parents[1]
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "examples"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import math  # noqa: E402

from s1core import (  # noqa: E402
    Augmentor,
    ChoiceSpec,
    EpistemicStatus,
    GateInputs,
    GatePolicy,
    InMemoryAuditLedger,
    IsoDateDeltaFacts,
    LexicalHeuristicDriver,
    NoulSpec,
    RatioFacts,
    ScoreSpec,
    System1Harness,
    compile_waves,
    driver_from_env,
    digest_of,
    evaluate,
    expected_calibration_error,
    fit_gate_policy,
    fit_temperature,
    js_divergence_bits,
    load_dotenv,
    normalize,
    replay_verdict,
    temperature_scale,
    values_match,
    verify_chain,
)

import cascade_router as crc  # noqa: E402
import model_swap_certification as msc  # noqa: E402
import nightly_recalibration as nrc  # noqa: E402

REFERENCE_TIME_ISO = "2026-10-15T12:00:00+00:00"
ALPHA = 0.10

AUGMENTOR = Augmentor((IsoDateDeltaFacts(), RatioFacts("requested_refund", "order_total")))


class ApiError(RuntimeError):
    """An expected, user-facing failure mapped to an HTTP status by the server."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- backends

_cache: dict[str, Any] = {}


def backends() -> tuple[Any | None, Any]:
    """(live, toy): live is None without a configured key."""
    if "pair" not in _cache:
        load_dotenv()
        _cache["pair"] = (driver_from_env(),
                          LexicalHeuristicDriver(sharpness=24.0))
    return _cache["pair"]


def resolve_backend(preference: str) -> tuple[Any, str]:
    live, toy = backends()
    if preference == "toy":
        return toy, "toy"
    if preference == "live":
        if live is None:
            raise ApiError(422, "no TYPESAFE_API_KEY configured — the live "
                                "backend is unavailable (copy example.env.txt "
                                "to .env, or pick the toy backend)")
        return live, "live"
    return (live, "live") if live is not None else (toy, "toy")


# ------------------------------------------------------------ run building

def check_replay(records: list[dict]) -> dict:
    checked = wrong = 0
    rows = []
    for idx, record in enumerate(records):
        checked += 1
        try:
            replayed = replay_verdict(record)
            ok = (replayed.status.value == record["verdict"]["status"]
                  and replayed.value == record["verdict"]["value"])
        except Exception:
            ok = False
        wrong += 0 if ok else 1
        rows.append({"record_idx": idx, "ok": ok})
    return {"checked": checked, "wrong": wrong, "rows": rows}


def verdict_rows(records: list[dict], labels: dict[int, str] | None = None) -> list[dict]:
    """The universal row shape every view renders. One row per record; the
    row points at its record so any view can open the inspector."""
    rows = []
    for idx, rec in enumerate(records):
        verdict = rec.get("verdict") or {}
        compiled = rec.get("compiled_question") or {}
        state = (rec.get("canonical_state") or {}).get("raw_state") or {}
        rows.append({
            "record_idx": idx,
            "key": str(state.get("case_id") or state.get("ticket_id")
                       or rec.get("stream_id") or "-"),
            "question_id": compiled.get("question_id", "?"),
            "status": verdict.get("status", "?"),
            "value": verdict.get("value"),
            "confidence": verdict.get("confidence"),
            "top2_margin": verdict.get("top2_margin"),
            "normalized_entropy": verdict.get("normalized_entropy"),
            "score_mode_deviation": verdict.get("score_mode_deviation"),
            "conformal_set_size": verdict.get("conformal_set_size"),
            "model_id": rec.get("model_id", ""),
            "digest": (rec.get("record_digest") or "")[:12],
            "label": (labels or {}).get(idx),
        })
    return rows


def build_run(*, kind: str, driver: Any, backend_kind: str, records: list[Any],
              labels: dict[int, str], reference_time_iso: str | None = None,
              **extras) -> dict:
    records_json = [r.to_jsonable() if hasattr(r, "to_jsonable") else r
                    for r in records]
    payload = {
        "kind": kind,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": {"kind": backend_kind, "model_id": driver.model_id},
        "reference_time_iso": reference_time_iso,
        "records": records_json,
        "labels": {str(k): v for k, v in labels.items()},
        "rows": verdict_rows(records_json, labels),
        "chain_verified": verify_chain(records_json),
        "replay": check_replay(records_json),
    }
    payload.update(extras)
    return payload


def run_cases(driver, states, specs, truth_fn: Callable[[dict, str], str],
              *, reference_time_iso: str = REFERENCE_TIME_ISO,
              augmentor=AUGMENTOR):
    """Evaluate every state on one harness with an in-memory ledger, labeling
    each record with the pinned outcome for its question."""
    ledger = InMemoryAuditLedger()
    harness = System1Harness(driver, audit_sink=ledger, augmentor=augmentor)
    labels: dict[int, str] = {}
    for i, state in enumerate(states):
        start = len(ledger.records)
        harness.evaluate(
            state, specs, reference_time_iso=reference_time_iso,
            stream_id=str(state.get("case_id") or state.get("ticket_id")
                          or f"case-{i}"),
        )
        for record in ledger.records[start:]:
            labels[len(labels)] = truth_fn(
                state, record.compiled_question["question_id"])
    return ledger, labels


# ---------------------------------------------------------------- triage

_TRIAGE_LANES = {
    "auto_approve": (
        "Approve automatically because the refund is small relative to the "
        "order total and the account history is clean."
    ),
    "agent_review": (
        "A human agent must review because the amount, dispute history, or "
        "situation needs judgment beyond the policy rule."
    ),
    "deny": (
        "Deny because the request exceeds the order total or the account "
        "shows repeat chargeback abuse."
    ),
}


def triage_specs() -> list:
    return [
        ChoiceSpec(
            question_id="refund_lane",
            instructions=(
                "Given this refund ticket and the deterministic facts about "
                "the order, which fulfillment lane should handle it?"
            ),
            options=_TRIAGE_LANES,
        ),
        ScoreSpec(
            question_id="urgency",
            instructions=(
                "Rate how urgently a human should attend to this ticket on "
                "the rubric from routine to critical."
            ),
            rubric_levels={
                1: "Routine queue work with no customer waiting on the outcome.",
                2: "Standard handling within one business day is fully acceptable.",
                3: "The customer is waiting and the matter is approaching a deadline.",
                4: "Urgent: the customer is distressed or the deadline has passed.",
                5: "Critical: legal, regulatory, or reputational exposure right now.",
            },
        ),
        NoulSpec(
            question_id="exceeds_total",
            instructions=(
                "Does the requested refund exceed the full order total for "
                "this purchase? The facts block pre-computes the ratio."
            ),
            true_criteria="The requested refund amount is greater than the order total.",
            false_criteria="The requested refund amount is within the order total.",
            depends_on=("refund_lane", "agent_review"),
        ),
    ]


TRIAGE_TICKETS = [
    {
        "case_id": "tkt-8471",
        "ticket_body": (
            "Customer says the refund they requested has not arrived and they "
            "need the money back urgently before rent is due."
        ),
        "order_total": 120.0,
        "requested_refund": 150.0,
        "purchase_date": "2026-09-30T09:00:00+00:00",
        "sla_deadline_at": "2026-10-13T17:00:00+00:00",
    },
    {
        "case_id": "tkt-8472",
        "ticket_body": (
            "Long-standing customer requests a small partial refund for a "
            "shipping delay and is polite about the wait."
        ),
        "order_total": 400.0,
        "requested_refund": 60.0,
        "purchase_date": "2026-05-11T09:00:00+00:00",
        "sla_deadline_at": "2026-10-20T17:00:00+00:00",
    },
    {
        "case_id": "tkt-8473",
        "ticket_body": (
            "Customer disputes the entire charge and mentions their bank is "
            "already involved after a prior chargeback last month."
        ),
        "order_total": 260.0,
        "requested_refund": 260.0,
        "purchase_date": "2026-08-02T09:00:00+00:00",
        "sla_deadline_at": "2026-10-10T17:00:00+00:00",
        "prior_chargebacks": 3,
    },
]


def _triage_truth(state: dict, question_id: str) -> str:
    ratio = state["requested_refund"] / state["order_total"]
    chargebacks = state.get("prior_chargebacks", 0)
    exceeds = state["requested_refund"] > state["order_total"]
    sla_days = int((datetime.fromisoformat(state["sla_deadline_at"]).date()
                    - datetime.fromisoformat(REFERENCE_TIME_ISO).date()).days)
    if question_id == "refund_lane":
        if exceeds or chargebacks >= 2:
            return "deny"
        if ratio <= 0.25 and chargebacks == 0:
            return "auto_approve"
        return "agent_review"
    if question_id == "urgency":
        level = 1
        level += 2 if sla_days < 0 else 0
        level += 1 if chargebacks else 0
        level += 1 if exceeds else 0
        return str(min(5, level))
    return "true" if exceeds else "false"


def run_triage(backend_preference: str = "auto") -> dict:
    driver, kind = resolve_backend(backend_preference)
    specs = triage_specs()
    ledger, labels = run_cases(driver, TRIAGE_TICKETS, specs, _triage_truth)
    return build_run(
        kind="evaluate", driver=driver, backend_kind=kind,
        records=ledger.records, labels=labels,
        reference_time_iso=REFERENCE_TIME_ISO,
        waves=[[s.question_id for s in wave] for wave in compile_waves(specs)],
        questions_meta=[
            {"question_id": s.question_id,
             "depends_on": [list(c) for c in s.dependencies()]}
            for s in specs
        ],
        scenario="triage",
    )


# ---------------------------------------------------------- certification

def run_certification() -> dict:
    specs = msc.build_specs()
    states = [msc.case_state(case) for case in msc.CASES]
    baseline_driver = msc.SyntheticJev("prod-jev-snapshot")

    def truth(state, qid):
        return (msc.pinned_lane(state) if qid == "refund_lane"
                else msc.pinned_exceeds(state))

    ledger, labels = run_cases(baseline_driver, states, specs, truth)
    base_by_key = {}
    for idx, record in enumerate(ledger.records):
        case_id = record.canonical_state["raw_state"]["case_id"]
        base_by_key[(case_id, record.compiled_question["question_id"])] = (
            idx, record)

    candidates = []
    for driver in (
        msc.SyntheticJev("jev-next-rc1", quality_lift=0.22,
                         lane_regression=True, salt="rc1"),
        msc.SyntheticJev("jev-next-rc2", quality_lift=0.22,
                         lane_regression=False, salt="rc1"),
    ):
        harness = System1Harness(driver, augmentor=AUGMENTOR)
        rows, per_q, wrongs = [], {}, []
        for state in states:
            verdicts = harness.evaluate(
                state, specs, reference_time_iso=msc.REFERENCE_TIME_ISO,
                stream_id=state["case_id"])
            for spec in specs:
                verdict = verdicts[spec.question_id]
                pinned = truth(state, spec.question_id)
                base_idx, base_rec = base_by_key[
                    (state["case_id"], spec.question_id)]
                drift = js_divergence_bits(
                    normalize(base_rec.raw_probabilities),
                    verdict.probabilities)
                wrong = (verdict.status is EpistemicStatus.RATIFIED
                         and not values_match(verdict.value, pinned))
                if wrong:
                    wrongs.append({
                        "case_id": state["case_id"],
                        "question_id": spec.question_id,
                        "ratified": verdict.value,
                        "pinned": pinned,
                    })
                rows.append({
                    "case_id": state["case_id"],
                    "question_id": spec.question_id,
                    "base_status": base_rec.verdict["status"],
                    "cand_status": verdict.status.value,
                    "base_value": base_rec.verdict["value"],
                    "cand_value": verdict.value,
                    "pinned": pinned,
                    "drift": round(drift, 4),
                    "wrong": wrong,
                })
                stat = per_q.setdefault(spec.question_id, {
                    "wrong": 0, "ratified": 0, "drifts": []})
                if verdict.status is EpistemicStatus.RATIFIED:
                    stat["ratified"] += 1
                stat["wrong"] += 1 if wrong else 0
                stat["drifts"].append(drift)
        candidates.append({
            "model_id": driver.model_id,
            "promoted": not wrongs,
            "wrong_count": len(wrongs),
            "wrongs": wrongs,
            "per_question": {
                qid: {
                    "wrong": s["wrong"],
                    "ratified": s["ratified"],
                    "mean_drift": round(sum(s["drifts"]) / len(s["drifts"]), 4),
                    "max_drift": round(max(s["drifts"]), 4),
                }
                for qid, s in per_q.items()
            },
            "rows": rows,
        })

    return build_run(
        kind="certification", driver=baseline_driver, backend_kind="toy",
        records=ledger.records, labels=labels,
        reference_time_iso=msc.REFERENCE_TIME_ISO,
        pinned_cases=len(states) * len(specs),
        candidates=candidates,
        scenario="model_swap",
    )


# --------------------------------------------------------------- cascade

def run_cascade() -> dict:
    routed = crc.make_cases(crc.ROUTED_CASES, seed=882)
    calibration = crc.make_cases(crc.CALIBRATION_CASES, seed=771)
    fitted = crc.fit_conformal_tier(calibration)

    tier_policies = [
        ("permissive", GatePolicy(min_top1_prob=0.40, min_top2_margin=0.02,
                                  max_normalized_entropy=0.85)),
        ("default", GatePolicy()),
        ("strict", GatePolicy(min_top1_prob=0.85, min_top2_margin=0.45,
                              max_normalized_entropy=0.25)),
        ("conformal@α=0.10 (fitted)", fitted),
    ]
    tiers = []
    for name, policy in tier_policies:
        result = crc.route(routed, policy)
        cost = (result["handled"] * crc.S1_COST
                + result["escalated"] * (crc.S1_COST + crc.S2_COST))
        latency = (result["handled"] * crc.S1_LATENCY_MS
                   + result["escalated"] * (crc.S1_LATENCY_MS + crc.S2_LATENCY_MS))
        tiers.append({
            "name": name,
            "policy": asdict(policy),
            "handled": result["handled"],
            "wrong": result["wrong"],
            "escalated": result["escalated"],
            "cost_per_1k": round(cost * 1000 / len(routed), 4),
            "avg_latency_ms": round(latency / len(routed)),
        })

    driver = crc.CascadeSensor()
    spec = crc.lane_spec(GatePolicy())
    ledger, labels = run_cases(
        driver, routed, [spec],
        lambda state, qid: crc.pinned_lane(state))

    zero_wrong = [t for t in tiers if t["wrong"] == 0]
    return build_run(
        kind="cascade", driver=driver, backend_kind="toy",
        records=ledger.records, labels=labels,
        reference_time_iso=REFERENCE_TIME_ISO,
        tiers=tiers,
        fitted={"temperature": round(fitted.temperature, 4),
                "conformal_tau": round(fitted.conformal_tau, 4)},
        cheapest_zero_wrong=(min(zero_wrong, key=lambda t: t["cost_per_1k"])["name"]
                             if zero_wrong else None),
        calibration_window=len(calibration),
        routed_cases=len(routed),
        scenario="cascade",
    )


# ---------------------------------------------------------- recalibration

def collect_samples(records: list[dict], labels: dict[int, str],
                    question_id: str) -> list[tuple[dict, str]]:
    return [(rec["raw_probabilities"], labels[idx])
            for idx, rec in enumerate(records)
            if idx in labels and rec.get("raw_probabilities")
            and rec["compiled_question"]["question_id"] == question_id]


def reliability_bins(samples, temperature: float, n_bins: int = 10) -> list[dict]:
    bins = [{"conf": 0.0, "acc": 0.0, "n": 0} for _ in range(n_bins)]
    for raw, label in samples:
        dist = temperature_scale(normalize(raw), temperature)
        top = max(dist, key=dist.get)
        index = min(int(dist[top] * n_bins), n_bins - 1)
        bins[index]["n"] += 1
        bins[index]["conf"] += dist[top]
        bins[index]["acc"] += 1 if top == label else 0
    for b in bins:
        if b["n"]:
            b["conf"] = round(b["conf"] / b["n"], 4)
            b["acc"] = round(b["acc"] / b["n"], 4)
    return bins


def fit_samples(samples, primitive: str, alpha: float = ALPHA) -> dict:
    """The nightly split law: temperature on the older 60%, conformal tau on
    the newer 40% (choice only — tau gates choice primitives)."""
    if len(samples) < 4:
        raise ApiError(422, f"only {len(samples)} labeled samples for this "
                            f"question; calibration needs at least 4")
    split = math.floor(len(samples) * 0.6)
    fitting, calibration = samples[:split], samples[split:]
    if primitive == "choice":
        policy = fit_gate_policy(fitting, calibration_samples=calibration,
                                 alpha=alpha)
        scaled = [(temperature_scale(normalize(p), policy.temperature), lbl)
                  for p, lbl in calibration]
        coverage = round(sum(p[lbl] >= policy.conformal_tau
                             for p, lbl in scaled) / len(scaled), 4)
    else:
        policy = replace(GatePolicy(), temperature=fit_temperature(fitting))
        coverage = None
    ece_before = expected_calibration_error(samples)
    ece_after = expected_calibration_error(
        [(temperature_scale(normalize(p), policy.temperature), lbl)
         for p, lbl in samples])
    return {
        "primitive": primitive,
        "temperature": round(policy.temperature, 6),
        "conformal_tau": (None if policy.conformal_tau is None
                          else round(policy.conformal_tau, 6)),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "coverage": coverage,
        "n_fit": len(fitting),
        "n_cal": len(calibration),
        "alpha": alpha,
        "bins": reliability_bins(samples, policy.temperature),
        "policy": asdict(policy),
    }


def run_recalibration() -> dict:
    cases = nrc.make_window(60, seed=20261015)
    specs = nrc.build_specs()
    driver = nrc.UnderconfidentJev()
    ledger, labels = run_cases(
        driver, cases, specs,
        lambda state, qid: nrc.truth(state)[qid])

    records = [r.to_jsonable() for r in ledger.records]
    fits = {}
    for qid, primitive in nrc.QUESTIONS.items():
        fits[qid] = fit_samples(collect_samples(records, labels, qid), primitive)
    digest = digest_of({qid: {k: f[k] for k in ("temperature", "conformal_tau")}
                        for qid, f in fits.items()})

    counterfactual = {}
    for qid, primitive in nrc.QUESTIONS.items():
        policy = replace(GatePolicy(), temperature=fits[qid]["temperature"],
                         conformal_tau=fits[qid]["conformal_tau"])
        counts = {"def_rat": 0, "def_wrong": 0, "fit_rat": 0, "fit_wrong": 0,
                  "n": 0}
        for idx, record in enumerate(records):
            if idx not in labels or qid != record["compiled_question"]["question_id"]:
                continue
            counts["n"] += 1
            gi = GateInputs.from_jsonable(record["gate_inputs"])
            for tag, pol in (("def", GatePolicy()), ("fit", policy)):
                verdict = evaluate(replace(gi, policy=pol),
                                   record["raw_probabilities"], "counterfactual")
                if verdict.status is EpistemicStatus.RATIFIED:
                    counts[f"{tag}_rat"] += 1
                    label = labels[idx]
                    if primitive in ("choice", "noul"):
                        ok = values_match(verdict.value, label)
                    else:
                        ok = round(verdict.value) == int(label)
                    counts[f"{tag}_wrong"] += 0 if ok else 1
        counterfactual[qid] = counts

    return build_run(
        kind="recalibration", driver=driver, backend_kind="toy",
        records=records, labels=labels,
        reference_time_iso=nrc.REFERENCE_ISO,
        fits=fits,
        snapshot_digest=digest,
        counterfactual=counterfactual,
        window_cases=len(cases),
        labeled_cases=len(labels) // len(nrc.QUESTIONS),
        scenario="recalibration",
    )


# --------------------------------------------------------------- registry

SCENARIOS = [
    {
        "id": "triage",
        "name": "Refund triage",
        "kind": "evaluate",
        "deterministic": False,
        "description": (
            "Three tickets through the full five-stage pipeline: lane choice, "
            "urgency rubric, and a wave-2 dependency gate. Runs the live model "
            "when a key is configured, the deterministic toy otherwise."
        ),
    },
    {
        "id": "model_swap",
        "name": "Model-swap certification",
        "kind": "certification",
        "deterministic": True,
        "description": (
            "Twelve pinned orders certified against the ledger: baseline "
            "replays wrong=0, then two candidates are judged by the only law "
            "that promotes — wrong == 0."
        ),
    },
    {
        "id": "cascade",
        "name": "Cascade router",
        "kind": "cascade",
        "deterministic": True,
        "description": (
            "Eighty tickets routed by the admissibility gate: RATIFIED ships "
            "the ~110 ms answer, everything else escalates. Four gate tiers "
            "swept, including the fitted conformal tier."
        ),
    },
    {
        "id": "recalibration",
        "name": "Nightly recalibration",
        "kind": "recalibration",
        "deterministic": True,
        "description": (
            "A labeled traffic window through an under-confident sensor: "
            "temperature + conformal fits per question, ECE before/after, and "
            "the counterfactual the fitted policy would ship tonight."
        ),
    },
]

_RUNNERS = {
    "triage": run_triage,
    "model_swap": run_certification,
    "cascade": run_cascade,
    "recalibration": run_recalibration,
}


def run_scenario(scenario_id: str, backend_preference: str = "auto") -> dict:
    runner = _RUNNERS.get(scenario_id)
    if runner is None:
        raise ApiError(404, f"unknown scenario: {scenario_id}")
    if scenario_id == "triage":
        return runner(backend_preference)
    return runner()
