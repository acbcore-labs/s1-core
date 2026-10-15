# S1-Core — a System 1 Inference Model Harness

[![CI](https://github.com/acbcore-labs/s1-core/actions/workflows/tests.yml/badge.svg)](https://github.com/acbcore-labs/s1-core/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-148%20passing-brightgreen.svg)]()

**S1-Core is a Python harness for System 1 inference models** (decision
models like TypeSafe's Jev): models that classify, score, and gate in a
single forward pass instead of generating text. It wraps any such backend —
hosted APIs, local MLX/ONNX decoders, llama.cpp logit extractors — in a
deterministic control loop with fail-closed calibration gates, conformal
prediction thresholds, topological wave batching, and a hash-chained,
replayable audit ledger.

The harness treats the model as a **bounded stochastic sensor**, never an
oracle: every output passes through deterministic preprocessing, mathematical
admissibility gates, and a content-addressed audit ledger before application
code sees anything. **A value is released if and only if the harness can
defend it** (fail-closed, `wrong == 0` admissibility boundary).

Python ≥ 3.14, zero dependencies (stdlib only). Licensed under
[MIT](LICENSE).

**Contents:** [60-second version](#the-60-second-version) ·
[Five-stage pipeline](#the-five-stage-pipeline--module-map) ·
[Installation & tests](#installation--tests) ·
[The UI](#the-ui--watch-the-harness-think) ·
[The examples](#the-examples--a-guided-tour) ·
[Configuration (.env)](#configuration--env-live-backend) ·
[Semantics and limits](#semantics-and-limits) ·
[Proof of outcomes](#proof-of-outcomes--what-the-harness-demonstrably-provides) ·
[Adapt a use case](#adapting-to-a-new-use-case-your-project) ·
[Adapt a backend](#adapting-to-a-new-system-1-model-the-backend-seam) ·
[Failure modes](#failure-mode-coverage-design-doc-part-1-3) ·
[Design notes](#design-law-notes-mechanical-sympathy--semantic-rigor--third-door) ·
[Design rationale](docs/DESIGN.md) ·
[Layout](#layout)

---

## The 60-second version

```python
import s1core
from s1core import System1Harness, ChoiceSpec, NoulSpec, EpistemicStatus
from s1core.backends import LexicalHeuristicDriver   # toy backend, runs offline

harness = System1Harness(LexicalHeuristicDriver())

verdicts = harness.evaluate(
    raw_state={"ticket": "refund requested", "order_total": 120.0,
               "requested_refund": 150.0},
    questions=[
        ChoiceSpec(
            question_id="refund_lane",
            instructions="Which fulfillment lane should handle this refund request?",
            options={
                "auto_approve": "Refund is within policy limits and history supports automatic approval.",
                "agent_review": "A human agent must review the refund amount before approval.",
                "deny":          "The request falls outside the refund policy and must be denied.",
            },
        ),
        NoulSpec(
            question_id="exceeds_policy",
            instructions="Does the requested refund exceed the full order total?",
            true_criteria="The requested refund is greater than the order total.",
            false_criteria="The requested refund is within the order total.",
            depends_on=("refund_lane", "agent_review"),   # wave DAG: runs only if parent ratified
        ),
    ],
    reference_time_iso="2026-10-14T12:00:00+00:00",
)

v = verdicts["refund_lane"]
if v.status is EpistemicStatus.RATIFIED:
    route(v.value)                 # e.g. "agent_review"
else:
    escalate_to_human(v)           # ABSTAINED_*, BIMODAL_CONFLICT, SKIPPED_DEPENDENCY, ...
```

Every `DecisionVerdict` carries: `status`, `value` (None unless RATIFIED),
`raw_value` (ungated telemetry), full `probabilities`, `confidence`,
`top2_margin`, `normalized_entropy`, and a `replay_digest` that content-addresses
(state + compiled question + model + calibration profile).

---

## The five-stage pipeline → module map

| Stage | Design-doc name | Module | What it enforces |
|---|---|---|---|
| 1. Compile-time schema lint + escape hatches | `SpecCompiler` | `s1core.specs` | Terse prose is a **compile error** (models don't read JSON keys). Epistemic escape hatches injected into every choice set so ambiguity abstains instead of guessing. |
| 2. Deterministic fact preprocessing | `StateAugmentor` | `s1core.augment` | Date deltas, ratios, structural invariants resolved in native code (zero neural cost) into `@deterministic_facts` before encoding. |
| 3. Prefix-pinned driver execution | `System1Driver` | `s1core.harness`, `s1core.dag`, `s1core.backends` | State pinned once; topological question **waves** fan out in parallel; Wave N+1 runs only where Wave ≤ N parents ratified with the required values. Optional dual-permutation debiasing. |
| 4. Calibration + admissibility gate | `AdmissibilityGate` | `s1core.gates`, `s1core.calibration` | Temperature scaling, scale-invariant normalized entropy, top-2 margin, conformal set cardinality, score-mode AAD (bimodality rejection). All pure functions; all inputs validated. |
| 5. Content-addressed replay ledger | `AuditSink` | `s1core.audit` | Hash-chained JSONL records (canonical state, compiled question, raw distribution, post-gate events, verdict; schema `s1core.audit/2`). `replay_verdict()` re-derives any verdict. |

**The three primitives** (see design doc Part 1): `choice` (2–255 options,
discrete distribution + confidence), `score` (2–10 ordered rubric levels →
expectation `s^`, gated on mode dispersion), `noul` (binary proposition →
`p(true)`, fail-closed gray zone between `false_threshold` and `true_threshold`).

---

## Installation & tests

**With [uv](https://docs.astral.sh/uv/) (recommended):**

```bash
brew install uv                # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                        # .venv + editable s1core + dev tools, per uv.lock
uv run pytest                  # 148 tests + enforced 100% coverage gate
uv run python examples/refund_triage.py
uv run python examples/orthogonal_pr_gate.py
uv run python examples/calibrate_from_ledger.py
uv run python examples/model_swap_certification.py
uv run python examples/nightly_recalibration.py
uv run python examples/cascade_router.py
uv run python examples/proof_of_outcomes.py   # live scorecard when a key is configured
```

**Without uv — the zero-dependency path (stdlib only):**

```bash
PYTHONPATH=src python3 -m unittest discover -s tests   # 148 tests, offline
```

CI (`.github/workflows/tests.yml`) runs both paths on Python 3.14 via
`astral-sh/setup-uv`: the stdlib-only core suite, then the full suite under
the enforced 100% line-coverage gate — no API keys, with the randomized
`wrong == 0` replay sweep riding along in both.

## The UI — watch the harness think

```bash
uv run python tools/ui/app.py        # opens http://127.0.0.1:8765 in your browser
```

A local, zero-dependency web console (stdlib `http.server`, vanilla JS, no
build step) for running, inspecting, and interrogating the harness. If the
README sections below are the *what*, the UI is the *see it happen*.

**What to do with it, in the order you'll probably want to:**

1. **Run a scenario.** The home screen runs the four workflows — refund
   triage, model-swap certification, cascade routing, nightly
   recalibration — as one click each. Every run lands on a *run view* whose
   every number came from `s1core` this time you clicked.
2. **Open the verdict inspector.** Click any row (or navigate with ↑/↓ and
   Enter; Esc closes). You get the raw distribution with its gate boundaries
   drawn *on* the chart — the top-1 floor line over the probability bars,
   the noul gray zone as a shaded band, the score expectation marker — plus
   pass/fail gauges for every threshold, the exact prose the model saw, and
   the `@deterministic_facts`. When a verdict abstains, the one red gauge is
   the reason; nothing else about the gate is hidden from you.
3. **Drag the gate lab dials.** Sliders for top-1 floor, margin, entropy,
   score AAD, temperature, and conformal τ re-run the pure Stage-4 gate over
   the run's stored raw distributions (debounced, server-side) and show
   exactly which verdicts flip. This is the fastest way to internalize what
   `GatePolicy` does — and why thresholds are policy, not tuning knobs over
   truth.
4. **Fit calibration.** Runs with pinned outcome labels carry a calibration
   panel: temperature on the older 60% of labeled samples, conformal τ on
   the newer 40% (disjoint, as the fit requires), ECE before/after, measured
   coverage, and the reliability chart.
5. **Audit a ledger.** The loader's dropdown lists every audit ledger under
   the repo (files whose first record carries the audit schema's digest —
   not just any `.jsonl`), newest first, with record counts. Loading one
   gives you chain verification, per-record replay checks, a stream filter,
   and the hash-chained spine. `--allow DIR` extends the search to other
   directories.
6. **Compose.** The composer is a guided editor for declaring new
   `choice`/`score`/`noul` questions — the compile-time prose lint fires
   inline as a 422 with the exact message — plus a raw-state JSON pane and a
   toy/live backend switch. It is the quickest way to feel Stage 1's
   discipline before writing any code.

**How it works** — the law is *the gate never lives in the browser*:

- `tools/ui/app.py` is a stdlib `http.server` exposing a small JSON API —
  evaluate, scenario runners, ledger load, gate sweep, calibration fit,
  replay — where every route is a thin wrapper over the same `s1core` calls
  the examples make. `static/app.js` is render-only: it draws what the
  server computed and computes nothing itself, so what you see is exactly
  what replay defends.
- Runs live in an in-memory registry; `#/run/:id` links are shareable while
  the server runs. Nothing is written to disk except by the harness's own
  ledgers.
- Security posture: binds `127.0.0.1` only; ledger paths must resolve under
  the repo root or an `--allow` directory; the API key is never sent to the
  page — only the model id and whether a live backend exists.

## The examples — a guided tour

| Example | Proves | See it in the UI |
|---|---|---|
| `refund_triage.py` | All five stages on real tickets | Home → *Refund triage*; writes `refund_triage_audit.jsonl`, which the ledger dropdown lists |
| `orthogonal_pr_gate.py` | Orthogonal decomposition + combiners | Read alongside the composer's question model |
| `calibrate_from_ledger.py` | Shadow-traffic distillation loop | Home → *Nightly recalibration* renders the same fits |
| `model_swap_certification.py` | The promotion licence law | Home → *Model-swap certification* |
| `nightly_recalibration.py` | Outcome-fed recalibration | Home → *Nightly recalibration* |
| `cascade_router.py` | The gate as a cost dial | Home → *Cascade router* + the gate lab |
| `proof_of_outcomes.py` | The measured scorecard | Standalone printout |

Reading notes — what to look for in each:

- **`refund_triage.py`** — under the toy sensor, `exceeds_total` lands in
  the noul gray zone and the harness *withholds* the value: that
  `ABSTAINED_GATE` is fail-closed working, not a bug. A dependency-gated
  question comes back `SKIPPED_DEPENDENCY` — reported, never dropped. With
  a key configured the same tickets run against live Jev unchanged.
- **`orthogonal_pr_gate.py`** — five narrow signals in one parallel call
  (~1.03× the tokens of one question), combined by `LatticeCombiner` /
  `NoisyANDCombiner`. Against the live model, the merge-lane question chose
  the escape hatch at p = 0.81 and the whole cascade abstained — the
  recorded run in [Proof of outcomes](#proof-of-outcomes--what-the-harness-demonstrably-provides).
- **`calibrate_from_ledger.py`** — `ShadowTeeDriver` mirrors production
  traffic into a local model, paired with outcomes, then
  `fit_gate_policy` promotes it: ECE 0.145 → 0.018, coverage 0.900 at
  α = 0.10, zero application changes.
- **`model_swap_certification.py`** — the production ledger is a frozen
  eval set. After preflight (chain verified, every verdict replays
  `wrong == 0`), candidates are judged by one law: ratify no wrong answer.
  The regressed candidate is rejected with its wrongs *attributed* —
  `refund_lane` on the two over-total orders; drift (JS divergence) is
  reported as telemetry, never gated.
- **`nightly_recalibration.py`** — temperature recovers the automation the
  default gate refused (ratified 12.7% → 98.7% at ECE 0.36 → 0.03). The
  residual wrong-while-ratified equals the sensor's confusion rate living
  inside the α budget: calibration *prices* errors, it does not cure them —
  which is what the other two workflows are for.
- **`cascade_router.py`** — the permissive tier is cheapest and ships 5
  wrong answers per 80 cases; default/strict ship zero at ~10× the cost;
  the fitted conformal tier ships zero at ~7% of the default's cost by
  catching dithering confusions as two-element prediction sets.
- **`proof_of_outcomes.py`** — the scorecard above; every line is labeled
  with what it is proof *of*. Deterministic offline, live with a key.

All deterministic examples reproduce bit-for-bit (seeded traffic,
content-derived sensors). `refund_triage` and `proof_of_outcomes` upgrade
to the live model when `TYPESAFE_API_KEY` is configured; the certification,
cascade, and recalibration flows are deliberately deterministic because
certificates and scorecards must reproduce.

## Configuration — `.env` (live backend)

The repo ships **`example.env.txt`**, a commented template. Copy it to **`.env`**
(gitignored — never commit real keys) and fill in your key:

```bash
cp example.env.txt .env    # then edit TYPESAFE_API_KEY
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `TYPESAFE_API_KEY` | for live use | — | TypeSafe (Jev) API key |
| `TYPESAFE_BASE_URL` | no | `https://api.typesafe.ai` | Point at a proxy/self-hosted gateway (use trusted HTTPS) |
| `TYPESAFE_MODEL` | no | `jev-latest` | Any model served by `/v1/systemone` |

`s1core.load_dotenv()` parses `.env` with the stdlib (no python-dotenv);
real environment variables always win over file values. Then one line picks
the backend — live model when configured, offline toy otherwise — with zero
other code changes:

```python
from s1core import load_dotenv, driver_from_env
from s1core.backends import LexicalHeuristicDriver

load_dotenv()
backend = driver_from_env() or LexicalHeuristicDriver()
```

`driver_from_env()` builds a `SystemOneHTTPDriver` pre-loaded with the
TypeSafe wire adapter (see below) — it returns `None` (never raises) when no
key is present, which is what makes the fallback one-liner work.

---

## Semantics and limits

The precise meaning of what the harness releases — the part that keeps the
marketing honest:

- **`RATIFIED` means a policy passed, not that an answer is correct or
  safe.** Enforce permissions and computable business invariants in
  application code. Prose word counts are compile-time lint, not semantic
  proof.
- **Confidence is defined per primitive**: `1 − normalized_entropy` for
  choice, `1 − mode deviation` for score, the selected Boolean probability
  for noul. None of these is a universal correctness probability.
- **Gates are primitive-scoped**: choice margin/entropy/conformal gates do
  not apply to score/noul. Invalid distributions and backend failures raise
  exceptions; gate refusals abstain.
- **Score expectations are averages.** `s^ = Σ i·p_i` is not a measured or
  modal level — polarized distributions are refused as `BIMODAL_CONFLICT`
  rather than collapsed.
- **`@deterministic_facts` is context, not a constraint.** The original live
  example ratified a false SLA answer despite a computed overdue deadline;
  logging that error does not correct it. Date facts use calendar days;
  write custom extractors for exact elapsed-time rules. Dependencies control
  execution, not injection of parent answers into child context.
- **Calibration**: use
  `fit_gate_policy(samples, calibration_samples=held_out, alpha=0.10)` with
  disjoint labeled data (omit `calibration_samples` and the helper splits in
  half). Probability-threshold/LAC sets are not APS; their rank is
  `floor((n+1)·α)` (rank zero → τ 0, full support). Coverage is marginal
  under exchangeability, not accuracy conditional on singleton acceptance.
- **Combiners**: `LatticeCombiner` confidence is a heuristic input minimum;
  `NoisyANDCombiner` multiplies selected-label probabilities (choice/noul
  only — score expectations are a `CombinerError`) and uses `p(false)` for a
  ratified False noul. Joint-probability readings require independence, and
  combiners are not automatically audited.
- **Audit**: `s1core.audit/2` records capture permutation divergence and the
  loop-guard event; replay reapplies gates and these events — not inference,
  and not the hidden stream-counter history. Legacy v1 events are inferred
  from stored telemetry. Gate-code changes can change old replay results;
  retain the code revision for long-term bit-identical replay. Verify the
  chain separately.
- **Operations**: one harness per stream, not concurrently. The loop guard
  counts consecutive model/gate abstentions across calls; other evaluated
  statuses reset it and dependency skips do not advance it. The JSONL ledger
  verifies existing chains before appending but is single-writer — no
  locking, fsync durability, signing, encryption, or external anchor. An
  unkeyed chain cannot detect a full-file rewrite; anchor its head digest
  externally if you need that. Protect raw state/PII and define retention.

---

## Proof of outcomes — what the harness demonstrably provides

`examples/proof_of_outcomes.py` prints a measured scorecard. With
`TYPESAFE_API_KEY` configured it runs against the live TypeSafe Jev API;
without, the identical scorecard runs on the offline toy backend, clearly
labeled. The figures below are **one recorded live run** (2026-10-14), not a
benchmark:

```text
=== S1-Core proof of outcomes ===
backend: jev-latest — LIVE (TypeSafe Jev)

1) Performance — parallel fan-out & cost
   questions declared        : 6 across 2 waves (5, 1 per wave; wave 2 is dependency-gated)
   backend fan-out calls     : 2 (one per executed wave + any debias re-dispatch)
   sequential calls avoided  : 4 (batched inside waves)
   wall clock                : 225 ms total, 112 ms avg per call
   tokens billed             : 1,584 in / 169 out across 2 requests
   estimated cost            : $0.000067 (input-only, $0.042 per 1M tokens)

2) Decision quality — gates & epistemic status
   status breakdown          : {'RATIFIED': 5, 'ABSTAINED_GATE': 1}
   refund_lane        RATIFIED          value='agent_review'   conf=0.930
   urgency            RATIFIED          value=3.63             conf=0.907 mode_aad=0.092
   exceeds_total      RATIFIED          value=True             conf=0.980
   customer_distress  RATIFIED          value=True             conf=0.910
   sla_missed         RATIFIED          value=False            conf=0.650
   escalate_to_human  ABSTAINED_GATE    value=-                conf=0.710
   fail-closed invariant     : OK (no verdict carries a value unless RATIFIED)
   deterministic-facts probe : 5/5 correct vs arithmetic ground truth (refund > total)
   inference determinism     : max |Δp| across a full re-run = 0.020000 (210 ms)

3) Gate counterfactual — same sensor, three admissibility policies
   default  (top1>=0.65, margin>=0.20, H<=0.55): top-1 'agent_review' p=0.980 -> RATIFIED
   relaxed  (top1>=0.40, margin>=0.05, H<=0.80): top-1 'agent_review' p=0.980 -> RATIFIED
   strict   (top1>=0.90, margin>=0.50, H<=0.20): top-1 'agent_review' p=0.980 -> RATIFIED

4) Governance — audit, replay, chain
   audit records             : 12 (content-addressed, hash-chained)
   replay verification       : 12/12 byte-identical (wrong=0)
   ledger chain verified     : True
```

What each line is proof **of**:

- **Fan-out & cost** — six questions consumed two API calls (one per wave),
  matching the design doc's single-pass economics; latency sits inside Jev's
  documented 70–500 ms envelope and billing is input-only as advertised.
- **Fail-closed quality** — `escalate_to_human` scored 0.71, inside the 0.35–0.75
  noul gray zone: the harness withheld the value rather than releasing a
  coin-flip decision. The invariant check proves no unratified verdict ever
  carries a value.
- **Deterministic-facts probe** — five refund-vs-total questions with
  computable ground truth: 5/5 correct, because Stage 2 resolved the
  arithmetic in native code before the model ever saw the state.
- **Live disagreement, surfaced honestly** — `sla_missed` returned False
  (conf 0.65) while the state's deterministic fact shows the deadline passed
  two days prior. The harness does not hide sensor error; the ledger records
  the raw distribution, the facts, and the verdict so per-question accuracy is
  measurable in production (that measurement is what feeds auto-calibration).
- **Determinism** — a full re-run moved probabilities by at most 0.02; the
  harness reports model-side variance instead of assuming it away.
- **Replay `wrong == 0`** — every verdict re-derives from its audit record;
  the hash chain over all records verifies.

`examples/orthogonal_pr_gate.py` against the live model additionally shows the
escape hatches earning their keep: asked to pick a merge lane with only a
title, CI line, and diff summary, **Jev chose `__abstain_insufficient_evidence__`
at p=0.81** — the cascade then skipped the dependent question
(`SKIPPED_DEPENDENCY`) and the auto-merge lattice refused to fire. A raw
integration would have guessed; the harness abstained, and every step of that
decision is auditable.

---

## Adapting to a new use case (your project)

Nothing in the core knows about refunds or pull requests. A new use case is:

1. **Declare questions.** Frozen dataclasses with full-sentence instructions
   and criteria prose (≥ 4 words instructions, ≥ 3 words criteria — enforced).
   Keys are for *your* code; only prose reaches the model.
2. **Add fact extractors if the state needs arithmetic/dates:**

   ```python
   from s1core import Augmentor, IsoDateDeltaFacts, RatioFacts, CallableExtractor

   def sla_hours(state, facts, ctx):
       facts["hours_to_sla"] = (parse(state["sla_at"]) - ctx.reference_time_utc) // 3600

   augmentor = Augmentor((
       IsoDateDeltaFacts(),                      # auto: *_date/_at fields → __days_from_reference
       RatioFacts("requested_refund", "order_total"),
       CallableExtractor(sla_hours),             # anything you can write as a function
   ))
   harness = System1Harness(driver, augmentor=augmentor)
   ```

3. **Tune gate policy per question** — `GatePolicy` is per-spec. High-stakes
   questions get strict thresholds (`min_top1_prob=0.8`); high-frequency
   routing gets loose ones. Set `conformal_tau` via `fit_gate_policy` once
   you have labels (on data disjoint from the temperature fit).
4. **Compose signals in code** where a single broad question would lose to a
   reasoning model: 5–8 orthogonal `noul`/`score` signals in one parallel call
   (~1.03x tokens of one question), combined with `LatticeCombiner` /
   `NoisyANDCombiner` (both fail closed on any non-RATIFIED input). See
   `examples/orthogonal_pr_gate.py`.
5. **Chain decisions with `depends_on`** — a single `(parent_id, required_value)`
   or a list of conditions (all must hold). Children of unmatched parents come
   back `SKIPPED_DEPENDENCY` — reported, never silently dropped.

Beyond single decisions, three promotion-and-operations workflows — model-swap
certification, nightly recalibration, cascade routing — compose the stages
into a system that decides what may be automated, with evidence. See
[The examples — a guided tour](#the-examples--a-guided-tour) for what each
proves and how to read it in the UI.

## Adapting to a new System 1 model (the backend seam)

Implement two methods — that is the whole contract
(`s1core.backends.base.BackendDriver`):

```python
class MyLocalKevDriver:
    model_id = "kev-4b-mlx"

    @contextmanager
    def pin_state(self, canonical_state, state_hash):
        handle = kev.encode_prefix(canonical_state)     # real KV pin on local runtimes
        try:
            yield StateHandle(state_hash, canonical_state)
        finally:
            kev.release(handle)

    def score_branches(self, handle, batch):
        # one parallel fan-out; batch maps qid -> CompiledQuestion
        return {qid: kev.pointer_head_readout(handle, q) for qid, q in batch.items()}
```

- `score_branches` returns `{question_id: {option_key: probability}}` — score
  **every** criteria key (escape hatches included); unnormalized vectors are
  fine, the gate normalizes and validates.
- Hosted APIs: `driver_from_env()` (or `typesafe_driver(api_key=...)`) gives
  you the live TypeSafe/Jev adapter on `POST /v1/systemone` — it absorbs the
  wire differences (questions as a JSON map, score criteria as an ordered
  array with 0-indexed returned levels mapped back to the harness's 1..k,
  scalar `noul` answers expanded to `{true, false}`, missing choice options
  zero-filled, 429/529 retried with backoff) and accumulates
  `driver.usage_totals` for cost accounting. Any other hosted decider adapts
  via `SystemOneHTTPDriver`'s `request_builder` / `response_parser` hooks.
- Local models (Kev/MLX, Von/Laya via ONNX, SemIf/llama.cpp logit extraction):
  `pin_state` is where the prefix-pinned KV cache lives. Hosted drivers just
  re-send state; the harness code above them is identical.
- Order-sensitive backends: set `GatePolicy(debiased_permutation=True)` on
  high-stakes questions — the harness dispatches canonical + reversed option
  order in the same batch, averages, and refuses when
  `js_divergence_bits(fwd, rev) > max_permutation_divergence` (divergence is
  in bits, bounded by 1.0; `1.0` is the lawful "never refuse").
- Shadow-traffic distillation: wrap a driver in `ShadowTeeDriver(primary,
  shadow_local, rate=0.05, on_shadow=...)`, accumulate paired distributions +
  production outcomes, then `fit_gate_policy(samples, calibration_samples=held_out)`
  → promote the local model with zero application changes
  (`examples/calibrate_from_ledger.py`: ECE 0.145 → 0.018, coverage 0.900 at
  α = 0.10).

## Failure-mode coverage (design doc Part 1, §3)

| Failure mode | Mitigation in this codebase |
|---|---|
| 1. Zero native abstention | Injected escape hatches (`ABSTAINED_MODEL`) + fail-closed gates (`ABSTAINED_GATE`); `value` is None unless RATIFIED. |
| 2. Silent failures (no forensic trace) | Hash-chained audit records with full raw distributions; `replay_verdict()`; `SKIPPED_DEPENDENCY` reported explicitly. |
| 3. Blindness to schema keys | Compile-time minimum-prose lint; option keys never reach the model, criteria prose does. |
| 4. Cross-question blindness | Topological wave DAG; children execute only after parents ratify with required values. |
| 5. Weak arithmetic / date deltas | `Augmentor` fact extractors compute in native code pre-encode. |
| 6. Downstream prompt-cache destruction | Out of scope here by design (this harness supplements, not replaces, LLM harnesses) — but when routing context for a downstream LLM, keep its static prefix byte-identical and inject pruned context only into the dynamic suffix. |

## Deviations from the source design doc (all deliberate)

- **`SKIPPED_DEPENDENCY` status added** — a declared question whose parent
  didn't match is *reported*, not omitted (the doc's own Failure Mode #2
  applies to harness bookkeeping too).
- **`score_mode_deviation` is its own verdict field** — the doc's pseudocode
  stuffed AAD into the `normalized_entropy` slot; different laws, different
  fields.
- **Both escape hatches injected** (`insufficient_evidence` +
  `out_of_domain`), per the doc's Stage-1 prose rather than its example code.
- **Fact extraction is a registry, not hardcoded fields** — the doc's
  refund-specific example became pluggable `FactExtractor`s.
- **`depends_on` accepts multiple conditions** (all must hold).
- **blake2b (stdlib) for digests** instead of blake3 — zero dependencies;
  identical content-addressing semantics.
- **Conformal tau is the finite-sample `floor((n+1)α)` rank** of the sorted
  true-class scores (τ = 0 when the rank is zero), which the coverage tests
  verify empirically at ≥ 1 − α for every α in the sweep.

## Design-law notes (mechanical sympathy / semantic rigor / third door)

> **Why is it built this way?** The full argument — the problem, the
> sensor-not-oracle thesis, the five stages, and the derivations behind
> every gate (temperature scaling, scale-invariant entropy, the conformal
> rank theorem and its honest limits, AAD polarization, JS-divergence
> debiasing, hash chaining) — is in
> **[docs/DESIGN.md](docs/DESIGN.md)**.

- **Pure Stage 4**: gates are functions of `(GateInputs, raw distribution)` —
  no I/O, clock, or RNG. Replay `wrong == 0` is a test, not an aspiration
  (`tests/test_replay.py`, randomized sweep included).
- **Deterministic key ordering everywhere** (canonical JSON, `dict.fromkeys`
  in debias averaging) — no set-iteration order leaks into digests.
- **Overflow-safe normalization** — distributions are peak-scaled before
  summing and combined with `math.fsum`; `1e308`-scale raw scores normalize
  to the same simplex as probabilities.
- **O(n²) wave scheduling** is fine at decision-batch scale (n ≲ 10³); the
  cost center is the model call, and the state KV prefix is encoded once
  across all waves.
- **Single harness instance per stream**; the loop guard keeps per-stream
  consecutive-abstention counters and is not thread-safe across streams by
  design.

## Layout

```
src/s1core/
├── types.py        EpistemicStatus, GatePolicy, DecisionVerdict
├── specs.py        Stage 1: ChoiceSpec / ScoreSpec / NoulSpec + compile-time lint
├── augment.py      Stage 2: Augmentor + fact extractors
├── dag.py          topological wave compilation
├── gates.py        Stage 4: pure admissibility-gate math
├── calibration.py  temperature fitting (golden-section), conformal tau, ECE
├── harness.py      Stage 3: orchestration, debiasing, loop guard
├── audit.py        Stage 5: hash-chained ledger + replay
├── combiners.py    deterministic signal combiners (fail-closed)
├── hashing.py      canonical JSON + blake2b content addressing
├── config.py       .env loader + driver_from_env factory
└── backends/
    ├── base.py         BackendDriver protocol, StateHandle
    ├── lexical.py      deterministic toy backend (tests/examples)
    ├── systemone.py    hosted /v1/systemone HTTP transport (usage + retries)
    ├── typesafe.py     live TypeSafe/Jev wire adapter
    └── shadow.py       ShadowTeeDriver (distillation pattern)
tools/ui/          local web UI (not packaged): app.py = stdlib server +
                   JSON API over s1core; scenarios.py = example adapters;
                   static/ = render-only single-page frontend
docs/DESIGN.md     design rationale: the problem, the thesis, the
                   mathematics behind every gate, the epistemology
```

Source design doc: *System 1 Inference Model Harness* (S1-Core architecture:
SpecCompiler → StateAugmentor → System1Driver → AdmissibilityGate → AuditSink;
primitives `choice` / `score` / `noul`; three pillars — Semantic Rigor,
Mechanical Sympathy, Deterministic Admissibility & Fail-Closed Governance).
