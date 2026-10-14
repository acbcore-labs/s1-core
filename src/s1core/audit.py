"""Stage 5: content-addressed replay ledger.

Every evaluated question emits an immutable audit record containing the
canonical state, compiled question payload, gate inputs, raw distribution,
and the gate verdict — content-addressed and hash-chained so tampering is
detectable and any verdict can be re-derived bit-for-bit from the record
(``wrong == 0`` replay; see :func:`replay_verdict`).

A System 1 failure has no chain-of-thought forensic trace (Failure Mode #2);
this ledger *is* the forensic trace.
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
from .types import DecisionVerdict, GatePolicy

SCHEMA_VERSION = "s1core.audit/1"


def compute_replay_digest(
    state_hash: str,
    compiled_payload: Mapping[str, Any],
    model_id: str,
    policy: GatePolicy,
) -> str:
    """blake2b of (state hash + compiled question + model + calibration profile)."""
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
    previous_digest: str = ""
    record_digest: str = ""

    def payload(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data.pop("record_digest")
        return data

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class AuditLedger(Protocol):
    def record(self, entry: AuditRecord) -> AuditRecord:
        """Persist one entry (extending the tamper-evident chain) and return it."""
        ...


class InMemoryAuditLedger:
    """Keeps records in a list — for tests, examples, and calibration runs."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def record(self, entry: AuditRecord) -> AuditRecord:
        previous = self.records[-1].record_digest if self.records else ""
        chained = dataclasses.replace(
            entry,
            previous_digest=previous,
            record_digest=digest_of({**entry.payload(), "previous_digest": previous}),
        )
        self.records.append(chained)
        return chained


class JsonlAuditLedger:
    """Append-only JSONL file with a per-line hash chain.

    The chain continues from the last record already in the file, so restarting
    a process does not break verifiability.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._previous = self._load_last_digest()

    def _load_last_digest(self) -> str:
        if not self._path.exists():
            return ""
        last = ""
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = line
        if not last:
            return ""
        return str(json.loads(last).get("record_digest", ""))

    def record(self, entry: AuditRecord) -> AuditRecord:
        chained = dataclasses.replace(
            entry,
            previous_digest=self._previous,
            record_digest=digest_of({**entry.payload(), "previous_digest": self._previous}),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(chained.to_jsonable()) + "\n")
        self._previous = chained.record_digest
        return chained


def make_record(
    *,
    stream_id: str,
    wave_index: int,
    model_id: str,
    state_hash: str,
    canonical_state: Mapping[str, Any],
    compiled_payload: Mapping[str, Any],
    gate_inputs: GateInputs,
    raw_probabilities: Mapping[str, float],
    verdict: DecisionVerdict,
) -> AuditRecord:
    return AuditRecord(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        stream_id=stream_id,
        wave_index=wave_index,
        model_id=model_id,
        state_hash=state_hash,
        canonical_state=dict(canonical_state),
        compiled_question=dict(compiled_payload),
        gate_inputs=gate_inputs.to_jsonable(),
        raw_probabilities=dict(raw_probabilities),
        verdict=verdict.to_jsonable(),
    )


def replay_verdict(record: Mapping[str, Any]) -> DecisionVerdict:
    """Re-run Stage 4 from a ledger record and re-derive the verdict.

    Pure function of the recorded raw distribution and recorded gate inputs.
    Must reproduce the recorded verdict exactly — if it does not, the ledger
    was tampered with or the gate implementation drifted. Records without a
    raw distribution (dependency skips) replay to their stored verdict.
    """
    raw = record.get("raw_probabilities") or {}
    stored = DecisionVerdict.from_jsonable(record["verdict"])
    if not raw:
        return stored
    gi = GateInputs.from_jsonable(record["gate_inputs"])
    digest = compute_replay_digest(
        state_hash=record["state_hash"],
        compiled_payload=record["compiled_question"],
        model_id=record["model_id"],
        policy=gi.policy,
    )
    return evaluate(gi, raw, digest)


def verify_chain(records: Iterable[Mapping[str, Any]]) -> bool:
    """Recompute the hash chain over an ordered sequence of ledger records."""
    previous = ""
    for record in records:
        expected = digest_of({**{k: v for k, v in record.items() if k != "record_digest"},
                              "previous_digest": record.get("previous_digest", "")})
        if record.get("previous_digest", "") != previous:
            return False
        if record.get("record_digest", "") != expected:
            return False
        previous = record["record_digest"]
    return True
