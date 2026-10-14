"""Design pattern #4 — shadow-traffic distillation & auto-calibration.

Run:  python examples/calibrate_from_ledger.py

Simulates the workflow: a local decider reports raw (uncalibrated) softmax
distributions; production outcomes accumulate in the audit ledger; once enough
labeled traces exist, the harness fits an exact per-question temperature
(minimizing NLL) and a conformal prediction-set threshold (guaranteeing
1 - alpha coverage), then hands you a GatePolicy — zero application-code
changes when you promote the calibrated policy.
"""

import random
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s1core import (  # noqa: E402
    ChoiceSpec,
    expected_calibration_error,
    fit_gate_policy,
    temperature_scale,
    normalize,
)

OPTIONS = {
    "billing": "Issues about invoices, charges, refunds, or payment methods.",
    "technical": "Issues about product defects, errors, crashes, or outages.",
    "account": "Issues about sign-in, access, profile, or account settings.",
}
LEVELS = {k: f"Customer issue belongs to the {k} category described." for k in OPTIONS}


def softmax(logits: dict[str, float]) -> dict[str, float]:
    peak = max(logits.values())
    exps = {k: math.exp(v - peak) for k, v in logits.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


class RawLocalDecider:
    """Caricature of an uncalibrated local encoder: reports softmax(logits)
    (effective temperature 1.0) while its true accuracy follows a sharper
    distribution (true temperature 2.0)."""

    model_id = "local-decider-uncalibrated-v1"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def _logits(self, question_id: str) -> dict[str, float]:
        rng = random.Random(hash((self.model_id, question_id)) & 0xFFFF)
        return {key: rng.gauss(0.0, 1.4) for key in OPTIONS}

    def sample_outcome(self, question_id: str) -> str:
        truth = softmax({k: v / 2.0 for k, v in self._logits(question_id).items()})
        return self._rng.choices(list(truth), weights=truth.values(), k=1)[0]

    def score(self, question_id: str) -> dict[str, float]:
        return softmax(self._logits(question_id))


def main() -> None:
    backend = RawLocalDecider()

    # 1. Before: raw softmax, effective temperature 1.0 — overconfident
    #    relative to the model's true skill (temperature 2.0).
    samples = []
    for i in range(4000):
        qid = f"route_ticket_{i}"
        raw = backend.score(qid)
        correct = backend.sample_outcome(qid)
        samples.append((raw, correct))

    ece_before = expected_calibration_error(samples)
    print(f"ECE before calibration : {ece_before:.4f}  (overconfident raw softmax)")

    # 2. Fit: exact per-question temperature (golden-section NLL minimum),
    #    then a conformal tau at risk alpha = 0.10 on top of the scaled
    #    distribution.
    policy = fit_gate_policy(samples, alpha=0.10)
    print(f"fitted temperature     : {policy.temperature:.4f}")
    print(f"conformal tau @ a=0.10 : {policy.conformal_tau:.4f}")

    # 3. After: same raw distributions under the fitted policy.
    scaled = [(temperature_scale(normalize(p), policy.temperature), c) for p, c in samples]
    ece_after = expected_calibration_error(scaled)
    print(f"ECE after calibration  : {ece_after:.4f}")

    # 4. What the calibrated policy does inside the admissibility gate: the
    #    conformal set {i : p_i >= tau} covers the true outcome >= 90% and
    #    must be a singleton before any value is released.
    covered = sum(
        1 for probs, correct in scaled
        if sum(1 for v in probs.values() if v >= policy.conformal_tau) >= 1
        and probs[correct] >= policy.conformal_tau
    )
    print(f"empirical coverage     : {covered / len(scaled):.4f}  (target >= 0.90)")

    # 5. Promotion: drop the fitted policy into the harness spec. Zero
    #    application-code changes — the gate now reasons over calibrated
    #    probabilities with a formal coverage guarantee.
    spec = ChoiceSpec(
        question_id="route_ticket",
        instructions="Which support category does this customer ticket belong in?",
        options=OPTIONS,
        policy=policy,
    )
    print(f"\npromoted policy for '{spec.question_id}': temperature={spec.policy.temperature:.3f}, "
          f"conformal_tau={spec.policy.conformal_tau:.3f}")


if __name__ == "__main__":
    main()
