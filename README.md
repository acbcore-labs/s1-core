# S1-Core

[![Tests](https://github.com/acbcore-labs/s1-core/actions/workflows/tests.yml/badge.svg)](https://github.com/acbcore-labs/s1-core/actions/workflows/tests.yml)

A small control loop around decision models: compile questions, compute facts,
batch dependency waves, gate distributions, and record verdicts. Python >= 3.14;
zero runtime dependencies; [MIT license](LICENSE).

## Start

    pip install -e .
    PYTHONPATH=src python3 -m unittest discover -s tests
    python3 examples/refund_triage.py

`System1Harness.evaluate(raw_state, questions, reference_time_iso=...)` accepts
`ChoiceSpec`, `ScoreSpec`, and `NoulSpec`. Read `value` only for `RATIFIED` results;
otherwise escalate. `raw_value` is telemetry. See `examples/` for PR gates,
calibration, and replay. `LexicalHeuristicDriver` is an offline toy, not a model.

For TypeSafe/Jev, copy `example.env.txt` to gitignored `.env`, set
`TYPESAFE_API_KEY`, then use `load_dotenv()` and `driver_from_env()`. The factory
returns `None` without a key; choose any fallback explicitly. Optional
`TYPESAFE_BASE_URL`/`TYPESAFE_MODEL` configure the service; use trusted HTTPS.
Drivers implement `pin_state` and `score_branches`. Hosted drivers resend state;
local KV pinning is an extension seam, not an included MLX/ONNX implementation.

## Policies

`RATIFIED` means a policy passed, not that an answer is correct or safe. Enforce
permissions and computable business invariants in application code.

- Choice: top-1 probability, top-2 margin, entropy, optional conformal singleton
  sets, and abstention choices. Prose word counts are lint, not semantic proof.
- Score: expected level on `1..k`, gated by mode absolute deviation. An
  expectation is not a measured or modal level.
- Noul: true/false thresholds with abstention in between.

Temperature applies to all. Choice margin/entropy/conformal gates do not apply
to score/noul. Confidence is respectively `1 - entropy`, `1 - mode_deviation`,
or selected Boolean probability, not a universal correctness probability.
Invalid distributions and backend failures raise exceptions. Dependencies control
execution, not injection of parent answers into child context.

`@deterministic_facts` is context, not a constraint. The original live example
ratified a false SLA answer despite a computed overdue deadline. Logging that
error does not correct it. Date facts use calendar days; use custom extractors
for exact elapsed-time rules. Ratio facts are not general signed/zero arithmetic.

## Calibration and composition

Use `fit_gate_policy(samples, calibration_samples=heldout, alpha=0.10)` with
separate labeled data. Without explicit calibration data, the helper splits samples
in half, fitting temperature on the first half. Keep final evaluation separate.
Scaling log probabilities is valid temperature scaling; raw logits are unnecessary.
Probability-threshold/LAC sets are not APS. Their rank is `floor((n+1)*alpha)`;
rank zero uses threshold zero and includes all labels. Coverage is marginal under
exchangeability, not accuracy conditional on singleton acceptance. Dataset shift
and selection require separate evaluation.

`LatticeCombiner` supports score comparisons; confidence is a heuristic input
minimum. `NoisyANDCombiner` multiplies selected-label probabilities for choice/noul,
rejects score expectations, and uses `p(false)` for False noul. Joint-probability
interpretations require independence. Apply an application action threshold;
a low product remains a valid scalar. Required lists must be nonempty and unique.
Combiners are not automatically audited.

## Audit and operation

New `s1core.audit/2` records capture permutation divergence and the loop-guard event.
Replay reapplies gates and these events, not inference or independent reconstruction
of hidden stream-counter history. Legacy v1 events are inferred from stored telemetry.
Gate-code changes can change old replay results; retain the code revision for
long-term bit-identical replay. Dependency skips return stored verdicts. Verify the
chain separately. Pin reference time for repeatable preprocessing; audit timestamps
are observational.

Use one harness per stream, not concurrently. The guard counts consecutive
model/gate abstention question results across calls, not whole batches; other
evaluated statuses reset it and dependency skips do not advance it.
JSONL verifies existing chains before appending and requires a single writer.
No locking, fsync durability, signing, encryption, or external anchor is provided.
A full-file rewrite can recompute an unkeyed chain; detection needs a trusted
external digest. Protect raw state/PII and define retention/access policy.

Shadow teeing collects distributions, not trained/promoted models. Historical live
cost/timing figures were one run, not benchmarks or pricing promises. CI runs
offline stdlib unittests on Python 3.14 without API keys, not live model evaluation.
Keep the core small: regression tests and explicit limits, no new framework.
