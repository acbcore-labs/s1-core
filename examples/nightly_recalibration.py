"""Nightly recalibration loop — outcome labels in, gate-policy snapshots out.

Design pattern: outcome-fed auto-recalibration. The audit ledger plus the
business system's outcome labels turn "we automated this lane" into a
measured statement: temperature fitted on one slice, conformal tau on a
disjoint later slice (the recency half — tomorrow's traffic looks like
yesterday evening's, not like last week's), and a policy snapshot whose
content digest makes the nightly diff itself auditable.

Flow (one run = one night, deterministic):

1. Two sealed traffic windows (yesterday, tonight) run through the full
   harness into JSONL ledgers. Labels land for ~75% of cases — outcomes are
   always late; unlabeled cases are excluded from fitting, not imputed.
2. Labels enter as a provenance-carrying table — label, source kind,
   labeler id, observed-at time, and the two join keys that tie it to the
   exact decision: ``record_digest`` (the audited decision) and
   ``spec_digest`` (the task+sensor identity). Correspondence is an
   enforced join requirement: missing, spec-mismatched, or conflicting
   labels are excluded with counts, never silently joined. Per question
   group, samples are fitted: ``fit_gate_policy`` with explicit disjoint
   splits for choice (temperature + conformal tau); ``fit_temperature``
   alone for score and noul — tau is a choice-primitive gate, and pretending
   otherwise would be a policy that does not obey its own law.
3. The fitted policy is counterfactually applied to tonight's raw
   distributions with the pure Stage 4 gate: ratified rate, wrong-while-
   ratified, and ECE before/after — the ops tradeoff, not a promise.
   Residual wrongness is the sensor's own confusion rate living inside the
   α budget; calibration prices it, the cascade and certification cure it.
4. Tonight's snapshot is written and diffed against yesterday's.

Honesty lines the output keeps: measured coverage is on the calibration
slice and marginal under exchangeability — dataset shift is why this job
runs nightly. The traffic sensor is deliberately under-confident (peak mass
flattened toward uniform), the common production pathology the loop fixes.

Deterministic end to end: windows come from a seeded generator, sensor
distributions from blake2b of case content. No live backend — swap real
drivers in and the loop is unchanged.
"""

import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
from dataclasses import replace  # noqa: E402

from s1core import (  # noqa: E402
    Augmentor,
    ChoiceSpec,
    IsoDateDeltaFacts,
    JsonlAuditLedger,
    NoulSpec,
    RatioFacts,
    ScoreSpec,
    System1Harness,
    expected_calibration_error,
    fit_gate_policy,
    fit_temperature,
    normalize,
    temperature_scale,
    values_match,
)
from s1core.backends.base import StateHandle  # noqa: E402
from s1core.calibration import Sample  # noqa: E402
from s1core.gates import GateInputs, evaluate  # noqa: E402
from s1core.hashing import canonical_json, digest_of  # noqa: E402
from s1core.types import EpistemicStatus, GatePolicy  # noqa: E402

REFERENCE_ISO = "2026-10-15T12:00:00+00:00"
REFERENCE = datetime.fromisoformat(REFERENCE_ISO)
ALPHA = 0.10
WINDOW_CASES = 200
UNIFORM_MIX = 0.15  # mild under-confidence: the pathology temperature fixing recovers
FLIP_RATE = 0.08    # genuine ranking errors: the sensor sometimes peaks on a
                    # plausible confusion — what the conformal gate must refuse

# Label provenance (fixed stamps keep the run deterministic). A production
# stream would carry these on every label row: where the truth came from,
# who produced it, and when it was observed relative to the decision.
LABEL_SOURCE = "business_rule"  # rule | human | event | model_judge
LABELER_ID = "refund-policy-rules/v1"
LABEL_OBSERVED_ISO = "2026-10-16T09:00:00+00:00"

QUESTIONS = {
    "refund_lane": "choice",
    "urgency": "score",
    "sla_missed": "noul",
}


def make_window(n: int, seed: int) -> list[dict]:
    """Sealed traffic window: seeded field generation (traffic is random;
    the seed makes tonight reproducible)."""
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        total = rng.choice([80.0, 120.0, 150.0, 200.0, 300.0, 450.0, 600.0, 900.0])
        ratio = rng.choice([0.2, 0.35, 0.48, 0.5, 0.55, 0.7, 0.9, 1.05, 1.3, 1.6])
        age = rng.choice([10, 45, 89, 90, 120, 400, 800])
        chargebacks = rng.choices([0, 1, 2, 3], weights=[0.80, 0.10, 0.06, 0.04])[0]
        distress = rng.random() < 0.35
        sla_days = rng.choice([-6, -2, -1, 0, 1, 3, 9])
        cases.append({
            "ticket_id": f"t-{seed}-{i:03d}",
            "order_total": total,
            "requested_refund": round(total * ratio, 2),
            "account_age_days": age,
            "prior_chargebacks": chargebacks,
            "customer_distress": distress,
            "sla_deadline_at": (REFERENCE + timedelta(days=sla_days)).isoformat(),
        })
    return cases


def truth(case: dict) -> dict[str, str]:
    """The business system's outcome labels, computed from the state — the
    ground truth that arrives after resolution."""
    total = case["order_total"]
    refund = case["requested_refund"]
    ratio = refund / total
    if ratio > 1.0 or case["prior_chargebacks"] >= 2:
        lane = "deny"
    elif ratio <= 0.5 and case["account_age_days"] >= 90 and case["prior_chargebacks"] == 0:
        lane = "auto_approve"
    else:
        lane = "agent_review"
    sla_days = (datetime.fromisoformat(case["sla_deadline_at"]).date()
                - REFERENCE.date()).days
    if sla_days < 0:
        level = min(5, 1 + (2 if case["customer_distress"] else 0) + 2)
    elif sla_days <= 1:
        level = min(5, 1 + (2 if case["customer_distress"] else 0) + 1)
    else:
        level = 1 + (2 if case["customer_distress"] else 0)
    return {
        "refund_lane": lane,
        "urgency": str(level),
        "sla_missed": "true" if sla_days < 0 else "false",
    }


def rule_margin(case: dict, question_id: str) -> float:
    """Distance to the nearest decision edge, normalized to [0, 1] — sensors
    are weakest near boundaries and mid-rubric."""
    ratio = case["requested_refund"] / case["order_total"]
    if question_id == "refund_lane":
        edges = [abs(ratio - 1.0) / 0.5, abs(ratio - 0.5) / 0.5,
                 abs(case["account_age_days"] - 90) / 210.0]
        if case["prior_chargebacks"]:
            edges.append(min(case["prior_chargebacks"], 2) / 2.0)
        return min(edges)
    sla_days = (datetime.fromisoformat(case["sla_deadline_at"]).date()
                - REFERENCE.date()).days
    if question_id == "sla_missed":
        return min(1.0, abs(sla_days) / 4.0)
    level = int(truth(case)["urgency"])
    return 0.3 + 0.7 * (abs(level - 3) / 2)  # rubric extremes are easy to judge


class UnderconfidentJev:
    """Stand-in traffic sensor: peaked at the truth, then flattened toward
    uniform — under-confident on well-ranked answers, which is what
    temperature fitting recovers — and occasionally peaking on a plausible
    confusion, which is what the conformal gate must refuse."""

    model_id = "traffic-jev-2026.10"

    @staticmethod
    def _draw(state_hash: str, tag: str) -> float:
        digest = hashlib.blake2b(
            f"{UnderconfidentJev.model_id}|{tag}|{state_hash}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "big") / 2**64  # uniform [0, 1)

    def _confusion(self, question_id: str, truth_key: str) -> str:
        if question_id == "refund_lane":
            return "agent_review" if truth_key == "auto_approve" else "auto_approve"
        if question_id == "sla_missed":
            return "false" if truth_key == "true" else "true"
        level = min(5, max(1, int(truth_key) + (1 if int(truth_key) < 3 else -1)))
        return str(level)

    @contextmanager
    def pin_state(self, canonical_state, state_hash: str):
        yield StateHandle(state_hash=state_hash, canonical_state=canonical_state)

    def score_branches(self, handle, batch):
        case = handle.canonical_state["raw_state"]
        out = {}
        for question_id, compiled in batch.items():
            labels = [k for k in compiled.criteria
                      if not k.startswith("__abstain")]
            truth_key = truth(case)[question_id]
            peak_key = (self._confusion(question_id, truth_key)
                        if self._draw(handle.state_hash, f"flip|{question_id}") < FLIP_RATE
                        else truth_key)
            jitter = (self._draw(handle.state_hash, question_id) - 0.5) * 0.08
            q = min(0.95, max(0.45, 0.55 + 0.40 * rule_margin(case, question_id) + jitter))
            m = len(labels)
            dist = {k: 0.0 for k in compiled.criteria}
            for k in labels:
                peak = q if k == peak_key else (1.0 - q) / (m - 1)
                dist[k] = (1.0 - UNIFORM_MIX) * peak + UNIFORM_MIX / m
            out[question_id] = dist
        return out


def build_specs() -> list:
    return [
        ChoiceSpec(
            question_id="refund_lane",
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
                3: "The customer is waiting and the matter is approaching a deadline.",
                4: "Urgent: the customer is distressed or the deadline has passed.",
                5: "Critical: legal, regulatory, or reputational exposure right now.",
            },
        ),
        NoulSpec(
            question_id="sla_missed",
            instructions=(
                "Has the service-level deadline for this ticket already passed as "
                "of the reference time? The facts block pre-computes the day delta."
            ),
            true_criteria="The deadline date is before the reference date.",
            false_criteria="The deadline date is today or after the reference date.",
        ),
    ]


def run_window(cases: list[dict], ledger_path: Path) -> None:
    harness = System1Harness(
        UnderconfidentJev(), audit_sink=JsonlAuditLedger(ledger_path),
        augmentor=AUGMENTOR,
    )
    specs = build_specs()
    for case in cases:
        harness.evaluate(
            case, specs, reference_time_iso=REFERENCE_ISO,
            stream_id=case["ticket_id"],
        )


def load_records(ledger_path: Path) -> list[dict]:
    with ledger_path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_label_table(records: list[dict], cases: list[dict]):
    """The business system's outcome labels, as a provenance-carrying table.

    One row per labeled decision: the label, where it came from, who produced
    it, when it was observed, and the two join keys that tie it to the exact
    decision — ``record_digest`` (the audited decision) and ``spec_digest``
    (the task+sensor identity). Two label rows that disagree about the same
    decision poison that digest: the conflict is counted and the decision is
    excluded at join time.
    """
    labeled = {c["ticket_id"]: truth(c) for c in cases if int(c["ticket_id"][-3:]) % 4 != 3}
    rows: dict[str, dict] = {}
    conflicted: set[str] = set()
    for record in records:
        if not record.get("raw_probabilities"):
            continue  # dependency skips are decisions that never ran
        ticket = record["canonical_state"]["raw_state"]["ticket_id"]
        if ticket not in labeled:
            continue  # outcome not yet observed: no label row exists
        digest = record["record_digest"]
        qid = record["compiled_question"]["question_id"]
        row = {
            "record_digest": digest,
            "spec_digest": record["spec_digest"],
            "question_id": qid,
            "label": labeled[ticket][qid],
            "source": LABEL_SOURCE,
            "labeler_id": LABELER_ID,
            "observed_utc": LABEL_OBSERVED_ISO,
        }
        if digest in rows and rows[digest]["label"] != row["label"]:
            conflicted.add(digest)
            continue
        rows.setdefault(digest, row)
    return rows, conflicted


def load_joined(ledger_path: Path, cases: list[dict]) -> tuple[list[tuple[dict, dict]], dict[str, int]]:
    """Join ledger records to label rows under enforced correspondence.

    A record joins only when a label row exists for its ``record_digest``
    AND that row's ``spec_digest`` equals the record's. Everything else is
    excluded with a count — unobserved outcomes, spec mismatches (a policy
    change or model swap must start a fresh sample set, not silently
    re-label a different task), and conflicting labels — never silently
    joined.
    """
    records = load_records(ledger_path)
    label_rows, conflicted = build_label_table(records, cases)
    labeled_ids = {c["ticket_id"] for c in cases if int(c["ticket_id"][-3:]) % 4 != 3}
    joined: list[tuple[dict, dict]] = []
    unobserved = mismatched = 0
    for record in records:
        if not record.get("raw_probabilities"):
            continue
        digest = record["record_digest"]
        if digest in conflicted:
            continue  # counted as conflicting in the provenance summary
        ticket = record["canonical_state"]["raw_state"]["ticket_id"]
        row = label_rows.get(digest)
        if row is None or ticket not in labeled_ids:
            unobserved += 1
            continue
        if row["spec_digest"] != record["spec_digest"]:
            mismatched += 1
            continue
        joined.append((record, row))
    provenance = {"joined": len(joined), "outcome_unobserved_or_missing": unobserved,
                  "spec_mismatch_excluded": mismatched,
                  "conflicting_excluded": len(conflicted)}
    return joined, provenance


def extract_samples(ledger_path: Path, cases: list[dict]) -> tuple[dict[str, dict], dict[str, int]]:
    """Group joined samples by ``spec_digest`` — the (task, sensor) identity —
    so a policy edit or model swap starts a fresh sample set instead of
    silently re-labeling a different task."""
    joined, provenance = load_joined(ledger_path, cases)
    groups: dict[str, dict] = {}
    for record, row in joined:
        group = groups.setdefault(row["spec_digest"],
                                  {"question_id": row["question_id"], "samples": []})
        group["samples"].append((record["raw_probabilities"], row["label"]))
    return groups, provenance


def fit_question(primitive: str, samples: list[Sample]):
    """Temperature on the older 60%, conformal tau on the newer 40% — the
    disjointness fit_gate_policy requires, with recency where it matters."""
    split = math.floor(len(samples) * 0.6)
    fitting, calibration = samples[:split], samples[split:]
    if primitive == "choice":
        policy = fit_gate_policy(fitting, calibration_samples=calibration, alpha=ALPHA)
        scaled = [(temperature_scale(normalize(p), policy.temperature), lbl)
                  for p, lbl in calibration]
        coverage = sum(p[lbl] >= policy.conformal_tau for p, lbl in scaled) / len(scaled)
    else:
        policy = replace(GatePolicy(), temperature=fit_temperature(fitting))
        coverage = None  # tau gates choice primitives only
    ece_after = expected_calibration_error(
        [(temperature_scale(normalize(p), policy.temperature), lbl) for p, lbl in samples])
    return policy, len(fitting), len(calibration), coverage, ece_after


def snapshot_for(window_name: str, cases: list[dict], ledger_path: Path) -> dict:
    groups, provenance = extract_samples(ledger_path, cases)
    questions = {}
    for spec_digest, group in groups.items():
        qid = group["question_id"]
        primitive = QUESTIONS[qid]
        policy, n_fit, n_cal, coverage, ece_after = fit_question(primitive, group["samples"])
        questions[qid] = {
            "primitive": primitive,
            "spec_digest": spec_digest,
            "temperature": round(policy.temperature, 6),
            "conformal_tau": (None if policy.conformal_tau is None
                              else round(policy.conformal_tau, 6)),
            "samples_temperature_fit": n_fit,
            "samples_conformal": n_cal,
            "measured_coverage": (None if coverage is None else round(coverage, 4)),
            "ece_at_fitted_temperature": round(ece_after, 6),
        }
    snapshot = {
        "schema": "s1core.gate-policy-snapshot/1",
        "window": window_name,
        "alpha": ALPHA,
        "window_cases": len(cases),
        "label_provenance": provenance,
        "questions": questions,
    }
    snapshot["snapshot_digest"] = digest_of(snapshot)
    return snapshot


def counterfactual(ledger_path: Path, cases: list[dict], tonight: dict) -> None:
    """Apply default vs fitted policies to tonight's raw distributions with
    the pure gate — ratified rate and wrong-while-ratified per question."""
    fitted = {qid: replace(GatePolicy(), temperature=q["temperature"],
                           conformal_tau=q["conformal_tau"])
              for qid, q in tonight["questions"].items()}
    joined, _ = load_joined(ledger_path, cases)
    rows: dict[str, dict] = {}
    for record, row_label in joined:
        qid = row_label["question_id"]
        primitive = QUESTIONS[qid]
        label = row_label["label"]
        gi = GateInputs.from_jsonable(record["gate_inputs"])
        row = rows.setdefault(qid, {"n": 0, "def_rat": 0, "def_wrong": 0,
                                    "fit_rat": 0, "fit_wrong": 0,
                                    "samples": []})
        row["n"] += 1
        row["samples"].append((record["raw_probabilities"], label))
        for policy, rat_key, wrong_key in (
            (GatePolicy(), "def_rat", "def_wrong"),
            (fitted[qid], "fit_rat", "fit_wrong"),
        ):
            verdict = evaluate(replace(gi, policy=policy),
                               record["raw_probabilities"], "counterfactual")
            if verdict.status is EpistemicStatus.RATIFIED:
                row[rat_key] += 1
                if primitive in ("choice", "noul"):
                    if not values_match(verdict.value, label):
                        row[wrong_key] += 1
                elif round(verdict.value) != int(label):
                    row[wrong_key] += 1

    print(f"\n--- tonight's counterfactual (n={len(joined)} labeled of {len(cases)}) ---")
    print(f"  {'question':12s} {'ECE raw':>8s} {'ECE fit':>8s}   "
          f"{'ratified def→fit':>18s}   {'wrong-while-ratified def→fit':>28s}")
    for qid, primitive in QUESTIONS.items():
        row = rows[qid]
        ece_raw = expected_calibration_error(row["samples"])
        ece_fit = tonight["questions"][qid]["ece_at_fitted_temperature"]
        print(f"  {qid:12s} {ece_raw:8.3f} {ece_fit:8.3f}   "
              f"{row['def_rat'] / row['n']:7.1%} →{row['fit_rat'] / row['n']:6.1%}   "
              f"{row['def_wrong']:14d} →{row['fit_wrong']:5d}"
              f"{'   (tau gates choice only)' if primitive != 'choice' else ''}")


AUGMENTOR = Augmentor((IsoDateDeltaFacts(), RatioFacts("requested_refund", "order_total")))


def diff_snapshots(yesterday: dict, tonight: dict) -> None:
    print(f"\n--- policy snapshot diff (auditable: content-addressed) ---")
    print(f"  digest {yesterday['snapshot_digest'][:12]}… → "
          f"{tonight['snapshot_digest'][:12]}…")
    for qid in QUESTIONS:
        old, new = yesterday["questions"][qid], tonight["questions"][qid]
        changes = [f"{field} {old[field]} → {new[field]}"
                   for field in ("temperature", "conformal_tau", "measured_coverage")
                   if old[field] != new[field]]
        print(f"  {qid:12s} {'| '.join(changes) if changes else 'unchanged'}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="s1core_recalibration_") as tmp:
        tmp_path = Path(tmp)
        yesterday_cases = make_window(WINDOW_CASES, seed=20261014)
        tonight_cases = make_window(WINDOW_CASES, seed=20261015)
        run_window(yesterday_cases, tmp_path / "yesterday.jsonl")
        run_window(tonight_cases, tmp_path / "tonight.jsonl")

        print("=== nightly recalibration ===")
        yesterday = snapshot_for("2026-10-14", yesterday_cases, tmp_path / "yesterday.jsonl")
        tonight = snapshot_for("2026-10-15", tonight_cases, tmp_path / "tonight.jsonl")
        for name, snapshot in (("yesterday", yesterday), ("tonight", tonight)):
            (tmp_path / f"gate-policy-snapshot-{name}.json").write_text(
                canonical_json(snapshot) + "\n", encoding="utf-8")

        for qid, q in tonight["questions"].items():
            extra = (f"  measured coverage {q['measured_coverage']:.3f} "
                     f"on {q['samples_conformal']} (α={ALPHA}, marginal, "
                     f"exchangeability assumed)"
                     if q["measured_coverage"] is not None else "")
            tau = ("—" if q["conformal_tau"] is None
                   else f"tau={q['conformal_tau']:.3f}")
            print(f"{qid:12s} ({q['primitive']:6s}) T={q['temperature']:.3f} {tau}{extra}")
        prov = tonight["label_provenance"]
        print(f"\nlabel provenance: {prov['joined']} joined · "
              f"{prov['outcome_unobserved_or_missing']} unobserved/missing · "
              f"{prov['spec_mismatch_excluded']} spec-mismatched · "
              f"{prov['conflicting_excluded']} conflicting — "
              f"source={LABEL_SOURCE} labeler={LABELER_ID} "
              f"(excluded with counts, never silently joined)")

        diff_snapshots(yesterday, tonight)
        counterfactual(tmp_path / "tonight.jsonl", tonight_cases, tonight)
        print("\nreading: temperature recovered the automation the default gate "
              "refused, at fitted ECE; the residual wrong-while-ratified is this "
              "sensor's confusion rate sitting inside the α risk budget — "
              "calibration prices errors, it does not cure them. Cure paths: "
              "certify a better sensor (model_swap_certification.py) and "
              "escalate what the gate refuses (cascade_router.py). Coverage is "
              "marginal on the calibration slice, not a guarantee under drift — "
              "which is why the loop runs nightly.")


if __name__ == "__main__":
    main()
