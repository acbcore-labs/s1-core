# S1-Core — design rationale

*Why the harness is built the way it is, and why that shape produces the
guarantees it produces.*

This document is the narrative companion to the README: the README says
**what** each stage enforces; this file argues **why** — the problem, the
thesis, the mathematics, and the epistemology — and ends every argument at
the test that proves it. The origin design doc (*System 1 Inference Model
Harness*) lives outside the repository, so this file is written to stand
alone. Where this document and the code disagree, the code wins and this
file is a bug.

---

## 1. The problem

A **System 1 decision model** (Jev-style `/v1/systemone` endpoints, local
pointer-head decoders, encoder scorers) classifies, scores, and gates in a
single forward pass. No scratchpad tokens, no chain of thought, no
self-correction: the model reads a state once and emits a distribution.

That shape is enormously cheap — measured at ~110 ms and input-only billing
for a six-question batch — and enormously fragile. Six failure modes
recur, and each has a concrete incident shape:

| # | Failure mode | Incident shape |
|---|---|---|
| 1 | Zero native abstention | The model must emit *some* answer; a coin-flip distribution surfaces as a confident-sounding pick that application code routes on. |
| 2 | Silent failures | A refused or skipped question is indistinguishable from an answered one unless the pipeline reports it; the forensic trace simply doesn't exist. |
| 3 | Blindness to schema keys | The model never reads your JSON keys — only prose. An option named `auto_approve` is invisible; undeclared criteria prose is invisible. Terse specs silently degrade. |
| 4 | Cross-question blindness | Question N+1 cannot see question N's answer, so pipelines that *should* branch on an earlier decision run unconditionally. |
| 5 | Weak arithmetic and dates | A single prefill pass cannot reliably compute "is 2026-09-30 plus 14 days before the SLA?" — the model guesses at exactly the facts that are mechanically checkable. |
| 6 | Downstream prompt-cache destruction | (Out of scope here by design — this harness supplements, not replaces, LLM harnesses — but when routing context downstream, keep static prefixes byte-identical.) |

A System 2 pattern — "wrap it in a reasoning model, ask it to be careful" —
doesn't fix any of these; it re-implements them at 100× the latency and
cost. The failure modes are structural, so the fix must be structural too.

## 2. The thesis

> **The model is a bounded stochastic sensor. Admissibility is a separate,
> deterministic authority.**

Everything in S1-Core follows from taking that sentence literally.

A sensor produces *readings*, not *decisions*. A thermometer reading of
"104.2°" is not a diagnosis; it is an input to a decision procedure with
its own law. Likewise a decision model's output —
`{"agent_review": 0.61, "auto_approve": 0.24, ...}` — is not a routing
decision. It is a reading, and readings have quality: they can be sharp or
diffuse, stable or position-sensitive, calibrated or overconfident.

So the harness splits responsibilities along that line:

- **The model** (any `BackendDriver`) answers *questions* — declared as
  prose, compiled, and dispatched in dependency-ordered waves.
- **The harness** owns *admissibility* — whether any given reading is good
  enough to release a value to application code — by pure, deterministic
  mathematics that does not consult the model about its own quality.
- **The ledger** owns *accountability* — a content-addressed record of
  every reading and every verdict, replayable bit-for-bit.

The consequence that shapes everything else: **abstention is a first-class
output.** A sensor that can't tell the harness *how confident it is* is
useless; a harness that can't *refuse to act* on a bad reading is a pass-through,
not an authority. `ABSTAINED_*` is not an error state — it is the correct
answer to "should we act on this?" when the honest response is "not on
this evidence." Because abstention is measurable (gray-zone rate, refusal
rate, coverage), it becomes **priceable**: escalation budgets, review
staffing, and automation coverage all become arithmetic instead of vibes.
The cascade-router example is that arithmetic, worked.

## 3. The five stages, and what breaks without each

| Stage | Module | Threat it neutralizes | If you removed it |
|---|---|---|---|
| 1. Compile-time spec lint + escape hatches | `specs.py` | FM #1, #3 | Terse specs degrade silently; the model is forced to guess instead of abstain. |
| 2. Deterministic fact preprocessing | `augment.py` | FM #5 | Date deltas and ratios are guessed by a neural net that has a calculator available — the live proof run caught exactly this: the model ratified `sla_missed = False` while the facts block had the deadline two days stale. |
| 3. Wave-DAG execution | `dag.py`, `harness.py` | FM #4 | Questions run unconditionally; branch logic leaks into prompts. |
| 4. Calibration + admissibility gates | `gates.py`, `calibration.py` | FM #1 (again, structurally) | Raw distributions ship to application code; confidence is whatever the model says it is. |
| 5. Content-addressed audit ledger | `audit.py` | FM #2 | No forensic trace; no replay; no eval set. |

Two details of Stage 1 deserve their own paragraphs, because they are the
most commonly under-appreciated:

**Only prose reaches the model — and the lint enforces that you know it.**
Options and criteria are `(key, prose)` pairs where the key is an
identifier for *your* code and the prose is the *entire* stimulus. The
compile-time lint (≥ 4 words per instruction, ≥ 3 per criterion) is not
style policing; it is the mechanical check for "did you write a sentence
the model could possibly act on, or did you write a JSON key and hope?"
`tests/test_specs.py` pins it.

**Escape hatches are injected, not requested.** Every non-exhaustive
choice set gets two extra options — *insufficient evidence* and *out of
domain* — whose selection the gate maps to `ABSTAINED_MODEL`. This is the
only channel by which the model is *allowed* to refuse, and it exists so
that a model which honestly cannot discriminate isn't forced to
manufacture confidence. The recorded live run in the README (a merge-lane
question answered `__abstain_insufficient_evidence__` at p = 0.81) is
this working.

## 4. The mathematics

Every computation below is a pure function of `(GateInputs, raw
distribution)` — no I/O, no clock, no RNG (`tests/test_harness.py` proves
byte-identical determinism). That purity is what makes everything later —
replay, sweep dials, certification — trustworthy.

### 4.1 Temperature scaling, in log space

Given a normalized reading `p`, temperature `T` produces
`q_i = p_i^(1/T) / Σ_j p_j^(1/T)` — exactly the softmax-with-temperature
formula on logits, since logits are `log p` up to a constant. `T < 1`
sharpens (the fitted cure for an under-confident sensor); `T > 1` flattens.
It is computed by subtracting the peak log before exponentiating, so no
achievable `T` can overflow (`gates.py::temperature_scale`).

`T` is *fitted, not tuned*: golden-section search on mean negative
log-likelihood over labeled samples (`calibration.py::fit_temperature`).
NLL is unimodal in `T` for a fixed sample set, so golden section converges
to the exact optimum — no gradients, no learning rate, 100 evaluations,
bounds (0.05, 20).

### 4.2 Scale-invariant entropy

Raw Shannon entropy scales with the number of options — uniform-on-3 and
uniform-on-200 would need different thresholds. Dividing by `ln m`
normalizes: **uniform → exactly 1.0, one-hot → exactly 0.0, for any `m`.**
One threshold (`max_normalized_entropy`, default 0.55) then means the same
thing on a 3-option lane choice as on a 200-option routing table
(`gates.py::normalized_entropy`). Degenerate supports have zero entropy by
law: with one option there is nothing to be uncertain about.

### 4.3 Margin, and the order of checks

The top-2 margin `p₁ − p₂` measures separation from the *nearest rival* —
the quantity that decides whether "the winner" is a stable statement. Note
the check order in `evaluate_choice`: **model escape hatches are consulted
first**, then conformal cardinality, then the threshold gates. A model that
selected the abstention option is reporting an epistemic state, not failing
one — conflating those two refusals is how dashboards start lying.

### 4.4 Conformal τ — the only threshold with a theorem behind it

Every hand-set threshold in `GatePolicy` is policy. `conformal_tau` is
different: it is chosen so that a coverage statement is *true by
construction*, not by calibration folklore.

Split-conformal, LAC-style: on a held-out calibration set of `n` labeled
readings, take the temperature-scaled probability the model assigned to the
**true** outcome — scores `s₁ … s_n`. Set
`τ = the ⌊(n+1)·α⌋-th smallest` (τ = 0 if that rank is 0), and admit only
singletons of the set `{i : q_i ≥ τ}`.

Why that covers: under **exchangeability**, a new reading's true-class
score is uniform-rank among `{s₁ … s_n, s_new}` — it is equally likely to
fall at any position. So

```
P(s_new < τ) = P(s_new ranks among the ⌊(n+1)α⌋ smallest of n+1)
             = ⌊(n+1)α⌋ / (n+1)  ≤  α
```

and therefore `P(p_true ≥ τ) ≥ 1 − α`. No distributional assumption beyond
exchangeability; no fitting that could overfit; the rank statistic *is*
the guarantee. `calibration.py::conformal_tau` implements exactly this
rank, and `tests/test_calibration.py` verifies measured coverage ≥ 1 − α
empirically across an α sweep.

Two honesty clauses come with the theorem, and both are load-bearing:

- **Marginal, not conditional.** Coverage holds *on average over
  exchangeable draws* — not "of the cases we ratified, 90% are right."
  Selection by the gate itself breaks conditioning; that is why the README
  refuses to promise it.
- **Exchangeability, not forever.** Dataset shift dissolves the guarantee,
  which is precisely why recalibration is a *loop* (nightly, on recency)
  rather than a one-time ceremony. The `nightly_recalibration` example is
  the theorem's maintenance schedule.

Related and often muddled: **temperature fitting and conformal calibration
must see disjoint samples** (`fit_gate_policy` enforces it — explicit
`calibration_samples`, or a no-shuffle 50/50 split). Fitting `T` on data
and then measuring true-class scores on the same data reuses the fit and
the τ guarantee is void. ECE is reported as a *diagnostic* — a fitness
number — never as a promise.

### 4.5 Scores: expectations are averages, and averages can describe nothing

A rubric score ratifies the expectation `s^ = Σ i·p_i`. Averages are
honest only when the distribution has one home. Consider two readings
with the **identical expectation**:

```
[0.01, 0.98, 0.01]   →  s^ = 2.00,  mode AAD = 0.01   → ratified
[0.50, 0.00, 0.50]   →  s^ = 2.00,  mode AAD = 0.50   → BIMODAL_CONFLICT
```

The second reading's expectation describes *no actual case* — it is half
a level-1 verdict plus half a level-3 verdict, a decision nobody can
execute, indistinguishable from the first by its average alone. The
normalized Average Absolute Deviation around the mode
(`AAD = Σ p_i·|i − mode| / (k−1)`) detects polarization, and the gate
refuses rather than collapse (`gates.py::evaluate_score`). This is
Semantic Rigor in miniature: a scalar that silently means two different
things is a defect even when tests are green. `tests/test_gates.py` runs
the design doc's canonical example.

The noul gate is the binary cousin: `p(true) ≥ 0.75` ratifies True,
`p(true) ≤ 0.35` ratifies False, and the band between is a gray zone where
the value is **withheld, not rounded toward an edge**. A 0.71 "yes" is not
a weak yes; it is an unanswered question (`evaluate_noul`; the README's
proof run shows `escalate_to_human` landing exactly there).

### 4.6 Order sensitivity, measured and refused

Pointer-head decoders can be positionally biased: the same question with
options in a different order yields a different distribution. That is
sensor error, not signal. With `debiased_permutation` on, the harness
dispatches the question twice — canonical and reversed option order —
averages the two readings (in deterministic key order; never set
iteration), and reports the Jensen-Shannon divergence between them in
bits:

```
JS(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m),   m = (p+q)/2
```

symmetric, support-independent, bounded by exactly 1.0 bit — which is why
`max_permutation_divergence = 1.0` is the *lawful* "never refuse" rather
than a magic sentinel. Divergence over budget ⇒ `ABSTAINED_GATE`
(`harness.py::_debias`, `gates.py::js_divergence_bits`). And critically,
the measured divergence is recorded in the audit's `postprocessing` map,
so replay re-applies the refusal — a verdict that was refused for
instability cannot resurrect itself on replay.

### 4.7 Content addressing and the chain

Everything content-addressed runs through one function:
`digest_of = blake2b(canonical_json(obj), 16 bytes)`, where
`canonical_json` sorts keys and compacts separators. Deterministic key
ordering everywhere — including `dict.fromkeys` in the debias averaging —
means no digest ever depends on set-iteration order (`hashing.py`,
`tests/test_harness.py` determinism checks).

Each ledger record carries `previous_digest` and
`record_digest = digest_of(payload + previous_digest)`. Recomputing the
chain detects any local edit; **appending is the only operation that
preserves a verified chain**, which is why the ledger is append-only and
single-writer by law. The honest limit: an *unkeyed* chain cannot detect a
wholesale rewrite of the whole file — if you need tamper-*proof* rather
than tamper-*evident*, anchor the head digest with an external party.
`verify_chain` and `replay_verdict` are the two readers;
`tests/test_replay.py` enforces `wrong == 0` over every recorded verdict,
including a randomized sweep.

Why blake2b and not blake3: identical content-addressing semantics, zero
dependencies. The chain is for integrity of decision records, not
adversarial cryptography — and the zero-dependency law is what keeps the
core auditable by reading.

### 4.8 Combiners, and their honest caveat

`NoisyANDCombiner` multiplies selected-label probabilities across
signals; `LatticeCombiner` applies a deterministic rule with the minimum
input confidence. Both **fail closed**: any non-RATIFIED input yields an
`ABSTAINED_GATE` combined verdict — a combiner can never manufacture
certainty from an abstention. The caveat the README states and this doc
reinforces: reading the product as a *joint* probability assumes
independence across signals, which is a property of your question
*decomposition*, not of the combiner. Decompose orthogonally or the
number is decorative. Score expectations are refused outright —
multiplying averages is a category error (`tests/test_combiners.py`).

### 4.9 Mechanical sympathy notes

The hot path is small and the cost center is the model call, so the
optimization targets are: **encode the state prefix once** per evaluate
call and fan all waves through it (that is where the single-pass
economics live); accept the O(n²) wave scheduler because decision batches
are n ≲ 10³ and the quadratic term is bookkeeping, not I/O; normalize
with peak-scaling and `math.fsum` so `1e308`-scale raw scores land on the
same simplex without overflow; and keep one harness per stream, because
the loop guard's per-stream counters are state — shared mutable state
across streams would be a race masquerading as a convenience.

## 5. The epistemology

Three sentences the whole system is willing to defend:

1. **RATIFIED means a policy passed — nothing more.** It is not "correct,"
   not "safe," not "true." Correctness is measurable only against outcomes,
   which arrive later and are the business system's to supply. Permissions
   and computable invariants are enforced in application code, which is
   where they were already living.
2. **A value exists if and only if the harness can defend it.** Every
   non-RATIFIED status carries `value = None` — structurally, in the type
   (`tests/test_harness.py` invariant) — so downstream code cannot consume
   an undischarged judgment even by accident.
3. **Wrongness is gateable; drift is telemetry.** The certificate law for
   promoting a model is exactly `wrong == 0` against pinned outcomes where
   a licence already exists — a binary, reproducible identity. Distribution
   drift (JS divergence, probability deltas) is *reported*, never gated: a
   drift budget would be a tuned threshold pretending to be a law. The
   model-swap example is the two columns side by side — REJECT for wrong=2,
   PROMOTE for wrong=0, drift printed as context.

And the loop guard ties the epistemology to operations: consecutive
model/gate abstentions on a stream are counted, and a stream that cannot
stop abstaining is *stopped* (`LOOP_GUARD_TRIPPED`) rather than allowed to
spin an agent forever. An authority that can always say "no" must
occasionally also say "stop."

## 6. The three pillars, applied

- **Semantic Rigor** — every term obeys its law: RATIFIED ≠ correct;
  entropy is normalized so one threshold means one thing; an expectation
  that describes no case is refused, not averaged harder; a name
  (`score_mode_deviation` as its own field, `SKIPPED_DEPENDENCY` as its own
  status) that doesn't obey its law is a defect even when tests are green.
- **Mechanical Sympathy** — the KV prefix is encoded once per call; the
  cost center is the model, so bookkeeping is allowed to be O(n²); the
  math is arranged (peak-scaled normalization, log-space temperature,
  deterministic ordering) so the machine's failure modes — overflow,
  nondeterministic iteration — cannot reach the output.
- **The Third Door** — where the world offers "use what exists" or "cut a
  corner," neither is taken: a registry of fact extractors instead of
  hardcoded refund fields; blake2b instead of a dependency; the conformal
  rank statistic instead of a tuned threshold; the production ledger
  instead of a curated eval set.

### Deviations from the source design doc (all deliberate)

| Deviation | Why |
|---|---|
| `SKIPPED_DEPENDENCY` status added | Failure Mode #2 applies to harness bookkeeping too — a skipped question is reported, never silently omitted. |
| `score_mode_deviation` is its own verdict field | The doc's pseudocode stuffed AAD into the entropy slot; different laws, different fields. |
| Both escape hatches injected | Per the doc's Stage-1 *prose* rather than its example code. |
| Fact extraction is a registry | The doc's refund example became pluggable `FactExtractor`s — the general mechanism, not the instance. |
| `depends_on` accepts multiple conditions | Real branch conditions are conjunctive. |
| blake2b (stdlib) instead of blake3 | Identical semantics, zero dependencies. |
| Conformal τ = floor((n+1)α) rank | The finite-sample statement the doc gestures at, made exact and coverage-tested per α. |

---

*Everything asserted here is executable: `uv run pytest` (148 tests, 100%
coverage gate) or `PYTHONPATH=src python3 -m unittest discover -s tests`
on the stdlib path. Claims require re-derivable evidence; the test suite
is where this document goes to be checked.*
