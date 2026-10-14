"""Canonical serialization and content addressing.

Every digest in S1-Core is derived from *canonical* JSON (sorted keys, compact
separators, ASCII escaping) so byte-identical inputs always produce
byte-identical digests on any platform. This is the precondition for the
byte-identical audit replayability required by Pillar III.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# blake2b from the stdlib is used instead of blake3 so the core has zero
# dependencies; both are keyed-free, fast, and cryptographically sound for
# content addressing. 16 bytes (128 bits) is far beyond collision-proof for
# per-decision records.
DIGEST_SIZE = 16


def canonical_json(obj: Any) -> str:
    """Serialize to a canonical, platform-stable JSON string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def blake2b_hex(payload: bytes, digest_size: int = DIGEST_SIZE) -> str:
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


def digest_of(obj: Any, digest_size: int = DIGEST_SIZE) -> str:
    """Content address: blake2b over the canonical JSON of ``obj``."""
    return blake2b_hex(canonical_json(obj).encode("utf-8"), digest_size)
