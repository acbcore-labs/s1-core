"""S1-Core: a generalized System 1 inference harness.

A deterministic control loop around bounded stochastic decision models
(Jev-style /v1/systemone APIs, KV-pinned local decoders, encoder scorers).
The model is a sensor; the harness owns admissibility.

Quickstart::

    from s1core import System1Harness, ChoiceSpec, NoulSpec
    from s1core.backends import LexicalHeuristicDriver

    harness = System1Harness(LexicalHeuristicDriver())
    verdicts = harness.evaluate(
        raw_state={"ticket": "refund requested", "order_total": 120.0},
        questions=[
            ChoiceSpec(
                question_id="lane",
                instructions="Which handling lane does this refund ticket belong in?",
                options={
                    "auto_approve": "Refund is within policy and history supports automatic approval.",
                    "agent_review": "A human agent must review the refund before any approval.",
                },
            ),
            NoulSpec(
                question_id="exceeds",
                instructions="Does the requested refund exceed the order total?",
                true_criteria="The requested refund amount is greater than the order total.",
                false_criteria="The requested refund amount is within the order total.",
                depends_on=("lane", "agent_review"),
            ),
        ],
    )
"""

from .audit import (
    AuditLedger,
    AuditRecord,
    InMemoryAuditLedger,
    JsonlAuditLedger,
    compute_spec_digest,
    replay_verdict,
    verify_chain,
)
from .augment import (
    AugmentationContext,
    Augmentor,
    CallableExtractor,
    FactExtractor,
    IsoDateDeltaFacts,
    RatioFacts,
    StateError,
)
from .backends import (
    BackendDriver,
    BackendError,
    LexicalHeuristicDriver,
    ShadowTeeDriver,
    StateHandle,
    SystemOneHTTPDriver,
    estimate_cost_usd,
    typesafe_driver,
)
from .calibration import (
    conformal_tau,
    expected_calibration_error,
    fit_gate_policy,
    fit_temperature,
)
from .combiners import CombinerError, LatticeCombiner, NoisyANDCombiner, signal_probability
from .config import driver_from_env, load_dotenv
from .dag import compile_waves
from .gates import (
    GateInputs,
    evaluate,
    js_divergence_bits,
    normalized_entropy,
    normalize,
    temperature_scale,
    validate_support,
)
from .hashing import canonical_json, digest_of
from .harness import System1Harness
from .specs import (
    ABSTAIN_INSUFFICIENT_EVIDENCE,
    ABSTAIN_OUT_OF_DOMAIN,
    ChoiceSpec,
    CompiledQuestion,
    NoulSpec,
    QuestionPrimitive,
    ScoreSpec,
    SpecError,
    values_match,
)
from .types import DecisionVerdict, EpistemicStatus, GatePolicy

__version__ = "0.1.0"

__all__ = [
    # types
    "DecisionVerdict",
    "EpistemicStatus",
    "GatePolicy",
    # specs (Stage 1)
    "ChoiceSpec",
    "ScoreSpec",
    "NoulSpec",
    "QuestionPrimitive",
    "CompiledQuestion",
    "SpecError",
    "values_match",
    "ABSTAIN_INSUFFICIENT_EVIDENCE",
    "ABSTAIN_OUT_OF_DOMAIN",
    # augment (Stage 2)
    "Augmentor",
    "AugmentationContext",
    "FactExtractor",
    "CallableExtractor",
    "IsoDateDeltaFacts",
    "RatioFacts",
    "StateError",
    # dag
    "compile_waves",
    # gates (Stage 4)
    "GateInputs",
    "evaluate",
    "validate_support",
    "normalize",
    "temperature_scale",
    "normalized_entropy",
    "js_divergence_bits",
    # harness (Stage 3)
    "System1Harness",
    # audit (Stage 5)
    "AuditLedger",
    "AuditRecord",
    "InMemoryAuditLedger",
    "JsonlAuditLedger",
    "compute_spec_digest",
    "replay_verdict",
    "verify_chain",
    # backends
    "BackendDriver",
    "BackendError",
    "StateHandle",
    "LexicalHeuristicDriver",
    "SystemOneHTTPDriver",
    "ShadowTeeDriver",
    "typesafe_driver",
    "estimate_cost_usd",
    # configuration
    "load_dotenv",
    "driver_from_env",
    # calibration
    "fit_temperature",
    "conformal_tau",
    "expected_calibration_error",
    "fit_gate_policy",
    # combiners
    "LatticeCombiner",
    "NoisyANDCombiner",
    "signal_probability",
    "CombinerError",
    # hashing
    "canonical_json",
    "digest_of",
]
