"""Hash-chained audit records and per-record verdict replay.

Replay repeats the gate and recorded post-gate control events. It does not
rerun model inference or independently reconstruct a stream's loop counter.
The unkeyed chain needs a trusted external head to detect wholesale rewrites.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol
from .gates import GateInputs, evaluate
from .hashing import canonical_json, digest_of
from .types import DecisionVerdict, EpistemicStatus, GatePolicy

SCHEMA_VERSION = "s1core.audit/2"


def compute_replay_digest(state_hash, compiled_payload, model_id, policy: GatePolicy) -> str:
    return digest_of([state_hash, dict(compiled_payload), model_id, dataclasses.asdict(policy)])


@dataclass(frozen=True, slots=True)
class AuditRecord:
    schema_version: str = SCHEMA_VERSION
    created_utc: str = ""
    stream_id: str = "default"
    wave_index: int = 0
    model_id: str = ""
    state_hash: str = ""
    canonical_state: Mapping[str, Any] = field(default_factory=dict)
    compiled_question: Mapping[str, Any] = field(default_factory=dict)
    gate_inputs: Mapping[str, Any] = field(default_factory=dict)
    raw_probabilities: Mapping[str, float] = field(default_factory=dict)
    verdict: Mapping[str, Any] = field(default_factory=dict)
    postprocessing: Mapping[str, Any] = field(default_factory=dict)
    previous_digest: str = ""
    record_digest: str = ""

    def payload(self):
        data = dataclasses.asdict(self)
        data.pop("record_digest")
        return data

    def to_jsonable(self):
        return dataclasses.asdict(self)


class AuditLedger(Protocol):
    def record(self, entry: AuditRecord) -> AuditRecord: ...


class InMemoryAuditLedger:
    def __init__(self):
        self.records: list[AuditRecord] = []

    def record(self, entry):
        previous = self.records[-1].record_digest if self.records else ""
        chained = dataclasses.replace(entry, previous_digest=previous,
            record_digest=digest_of({**entry.payload(), "previous_digest": previous}))
        self.records.append(chained)
        return chained


class JsonlAuditLedger:
    """Single-writer JSONL ledger; verify existing records before appending.

    Concurrent writers, fsync durability, and external anchoring are not provided.
    """
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._previous = self._load_last_digest()

    def _load_last_digest(self):
        if not self._path.exists():
            return ""
        with self._path.open(encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        if not verify_chain(records):
            raise ValueError("existing audit ledger failed chain verification")
        return records[-1]["record_digest"] if records else ""

    def record(self, entry):
        chained = dataclasses.replace(entry, previous_digest=self._previous,
            record_digest=digest_of({**entry.payload(), "previous_digest": self._previous}))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(chained.to_jsonable()) + "\n")
        self._previous = chained.record_digest
        return chained


def make_record(*, stream_id, wave_index, model_id, state_hash, canonical_state,
                compiled_payload, gate_inputs, raw_probabilities, verdict):
    return AuditRecord(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        stream_id=stream_id, wave_index=wave_index, model_id=model_id,
        state_hash=state_hash, canonical_state=dict(canonical_state),
        compiled_question=dict(compiled_payload), gate_inputs=gate_inputs.to_jsonable(),
        raw_probabilities=dict(raw_probabilities), verdict=verdict.to_jsonable(),
        postprocessing={"permutation_divergence": verdict.permutation_divergence,
                        "loop_guard_tripped": verdict.status is EpistemicStatus.LOOP_GUARD_TRIPPED})


def replay_verdict(record: Mapping[str, Any]) -> DecisionVerdict:
    """Reapply pure gates and recorded control events; verify chain separately.

    Legacy v1 records lack explicit events, so infer those two events from the
    stored verdict. Neither format proves the hidden loop-counter history.
    Skipped dependencies have no distribution and return the stored verdict.
    """
    raw = record.get("raw_probabilities") or {}
    stored = DecisionVerdict.from_jsonable(record["verdict"])
    if not raw:
        if stored.status is not EpistemicStatus.SKIPPED_DEPENDENCY:
            raise ValueError("only dependency skips may have no raw distribution")
        return stored
    gi = GateInputs.from_jsonable(record["gate_inputs"])
    digest = compute_replay_digest(record["state_hash"], record["compiled_question"], record["model_id"], gi.policy)
    result = evaluate(gi, raw, digest)
    if record.get("schema_version", "s1core.audit/1") == "s1core.audit/1":
        events = {"permutation_divergence": stored.permutation_divergence,
                  "loop_guard_tripped": stored.status is EpistemicStatus.LOOP_GUARD_TRIPPED}
    else:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported audit schema")
        events = record["postprocessing"]
    divergence = events.get("permutation_divergence")
    if divergence is not None:
        import math
        if not math.isfinite(divergence) or not 0.0 <= divergence <= 1.0:
            raise ValueError("invalid recorded permutation divergence")
        result = dataclasses.replace(result, permutation_divergence=divergence)
        if divergence > gi.policy.max_permutation_divergence:
            result = dataclasses.replace(result, status=EpistemicStatus.ABSTAINED_GATE, value=None)
    if events.get("loop_guard_tripped", False):
        if result.status not in {EpistemicStatus.ABSTAINED_MODEL, EpistemicStatus.ABSTAINED_GATE}:
            raise ValueError("loop guard event requires an abstained gate result")
        result = dataclasses.replace(result, status=EpistemicStatus.LOOP_GUARD_TRIPPED, value=None)
    return result


def verify_chain(records: Iterable[Mapping[str, Any]]) -> bool:
    previous = ""
    for record in records:
        payload = {k: v for k, v in record.items() if k != "record_digest"}
        if record.get("previous_digest", "") != previous or record.get("record_digest", "") != digest_of(payload):
            return False
        previous = record["record_digest"]
    return True
