"""Design pattern #1 — multi-signal orthogonal decomposition with
deterministic combiners, plus dual-permutation debiasing.

Run:  python examples/orthogonal_pr_gate.py

A single broad question ("Should this PR be merged safely?") frequently loses
to a small reasoning LLM. Decomposing it into narrow orthogonal signals in one
parallel call (~1.03x the tokens of one question) and combining the calibrated
outputs in plain deterministic code matches or beats System 2 at a fraction of
the latency.

The scripted driver stands in for your real backend (Jev, Kev, a local
encoder): it returns fixed distributions per question id.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contextlib import contextmanager  # noqa: E402
from typing import Any, Mapping  # noqa: E402

from s1core import (  # noqa: E402
    ChoiceSpec,
    GatePolicy,
    LatticeCombiner,
    NoisyANDCombiner,
    NoulSpec,
    System1Harness,
    driver_from_env,
    load_dotenv,
)
from s1core.backends.base import StateHandle  # noqa: E402


class ScriptedBackend:
    """Minimal reference for the BackendDriver protocol: canned distributions."""

    model_id = "scripted-decider-v1"

    def __init__(self, scripted: Mapping[str, Mapping[str, float]]):
        self._scripted = dict(scripted)

    @contextmanager
    def pin_state(self, canonical_state: Mapping[str, Any], state_hash: str):
        yield StateHandle(state_hash=state_hash, canonical_state=canonical_state)

    def score_branches(
        self, handle: StateHandle, batch: Mapping[str, Any]
    ) -> dict[str, dict[str, float]]:
        out = {}
        for question_id, compiled in batch.items():
            scripted = self._scripted.get(question_id, {})
            out[question_id] = {
                key: scripted.get(key, 0.0) for key in compiled.criteria
            }
        return out


def backend_script() -> dict[str, dict[str, float]]:
    """What a well-calibrated decider might return for one pull request. The
    merge_safety noul is deliberately borderline-ish; the noisy-AND shows it."""
    return {
        "tests_green": {"true": 0.97, "false": 0.03},
        "scope_creep": {"true": 0.15, "false": 0.85},
        "destructive_migration": {"true": 0.02, "false": 0.98},
        "security_sensitive": {"true": 0.04, "false": 0.96},
        "merge_lane": {
            "merge_now": 0.88,
            "merge_after_fix": 0.07,
            "request_changes": 0.05,
            "__abstain_insufficient_evidence__": 0.0,
            "__abstain_out_of_domain__": 0.0,
        },
        "merge_safety": {"true": 0.71, "false": 0.29},
    }


def main() -> None:
    # Live TypeSafe Jev when TYPESAFE_API_KEY is set (see example.env.txt);
    # the scripted backend stands in offline so the demo is deterministic.
    load_dotenv()
    live = driver_from_env()
    if live is not None:
        backend = live
        print(f"backend: LIVE {live.model_id}\n")
    else:
        backend = ScriptedBackend(backend_script())
        print("backend: TOY ScriptedBackend — copy example.env.txt to .env with TYPESAFE_API_KEY for the live model\n")

    questions = [
        NoulSpec(
            question_id="tests_green",
            instructions=(
                "Does the continuous integration report show every automated "
                "test passing on the head commit of this pull request?"
            ),
            true_criteria="All automated tests pass on the head commit.",
            false_criteria="At least one automated test fails on the head commit.",
        ),
        NoulSpec(
            question_id="scope_creep",
            instructions=(
                "Does this pull request introduce changes beyond the feature "
                "area stated in its title and description?"
            ),
            true_criteria="Changes extend well beyond the stated feature area.",
            false_criteria="Changes stay within the stated feature area.",
        ),
        NoulSpec(
            question_id="destructive_migration",
            instructions=(
                "Does this pull request contain a database or schema migration "
                "that irreversibly deletes or rewrites existing data?"
            ),
            true_criteria="A migration irreversibly deletes or rewrites data.",
            false_criteria="No migration deletes or rewrites existing data.",
        ),
        NoulSpec(
            question_id="security_sensitive",
            instructions=(
                "Does this pull request modify authentication, authorization, "
                "secret handling, or any trust boundary in the codebase?"
            ),
            true_criteria="Auth, secrets, or a trust boundary is modified.",
            false_criteria="No security-relevant surface is touched.",
        ),
        ChoiceSpec(
            question_id="merge_lane",
            instructions=(
                "Which merge lane applies to this pull request given its "
                "review state and test status?"
            ),
            options={
                "merge_now": (
                    "All checks pass and the change is self-contained: merge "
                    "immediately without further review."
                ),
                "merge_after_fix": (
                    "The intent is sound but small fixes are required before "
                    "the change can merge."
                ),
                "request_changes": (
                    "The change needs substantive rework or human review "
                    "before it can merge."
                ),
            },
            # High-stakes + order-sensitive backend: dispatch the question
            # twice (canonical + reversed option order) and refuse divergent
            # distributions (design pattern #3).
            policy=GatePolicy(
                debiased_permutation=True,
                min_top1_prob=0.70,
                min_top2_margin=0.50,
            ),
        ),
        NoulSpec(
            question_id="merge_safety",
            instructions=(
                "Taking the whole pull request together, is merging it now "
                "the safe action for production traffic?"
            ),
            true_criteria="Merging now is safe for production traffic.",
            false_criteria="Merging now risks production stability.",
            depends_on=("merge_lane", "merge_now"),
        ),
    ]

    harness = System1Harness(backend)
    verdicts = harness.evaluate(
        {
            "pr_title": "Add rate limiting to the public API gateway",
            "ci_status": "142 passed, 0 failed",
            "diff_summary": "3 files in services/gateway, plus a limiter module",
        },
        questions,
        reference_time_iso="2026-10-14T12:00:00+00:00",
    )

    print("=== Decomposed signals ===")
    for question_id in ("tests_green", "scope_creep", "destructive_migration",
                        "security_sensitive", "merge_lane", "merge_safety"):
        v = verdicts[question_id]
        extra = ""
        if v.permutation_divergence is not None:
            extra = f" js_div={v.permutation_divergence:.4f}bits"
        print(
            f"{question_id:22s} {v.status.name:14s} value={v.value!r:12} "
            f"p={max(v.probabilities.values()) if v.probabilities else 0:.2f}{extra}"
        )

    lattice = LatticeCombiner(
        "auto_merge_gate",
        required=["tests_green", "scope_creep", "destructive_migration",
                  "security_sensitive", "merge_lane"],
        rule=lambda vs: (
            vs["tests_green"].value is True
            and vs["scope_creep"].value is False
            and vs["destructive_migration"].value is False
            and vs["security_sensitive"].value is False
            and vs["merge_lane"].value == "merge_now"
        ),
    )
    noisy_and = NoisyANDCombiner("safe_to_proceed", required=[
        "tests_green", "scope_creep", "destructive_migration", "security_sensitive",
    ])

    gate = lattice.combine(verdicts)
    joint = noisy_and.combine(verdicts)
    print("\n=== Deterministic combiners (fail closed on any non-RATIFIED) ===")
    print(f"lattice auto-merge gate : {gate.status.name} value={gate.value}")
    print(f"noisy-AND joint safety  : {joint.status.name} p={joint.value:.4f}")
    if live is None:
        print(
            "\nNote merge_safety: at p(true)=0.71 it sits inside the 0.35-0.75 gray\n"
            "zone, so the harness withholds the value instead of guessing. The\n"
            "lattice gate requires only the five hard signals, so auto-merge still\n"
            "ratifies — and merge_safety's abstention is exactly what you would\n"
            "route to a System 2 reviewer as a second opinion."
        )
    print(
        "\nRouter: auto-merge only if the lattice ratifies; route to a System 2 "
        "reviewer whenever the joint safety probability is low."
    )


if __name__ == "__main__":
    main()
