"""Cryptographic identity primitives.

Normative: 12_training_system.md §6 — "Artifact Identity Model".
Every artifact is identified by content hash, never by path or count:

    "Sample counts are NOT identity."  (§6.2)
    "Paths Are Machine-Local, Never Part of Identity." (§6.3)

Canonicalization: JSON, sorted keys, no insignificant whitespace, UTF-8.
Two structurally equal objects must produce the same hash regardless of
construction order.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

ALGO = "sha256"


def canonical_json(obj: Any) -> bytes:
    """Deterministic byte representation of a JSON-compatible object."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    """`sha256:<hex>` identity of a JSON-compatible object."""
    return f"{ALGO}:{sha256_hex(canonical_json(obj))}"


def run_spec_hash(run_spec: dict[str, Any]) -> str:
    """Semantic identity of an experiment (12 §4).

    The run_spec is the SEMANTIC IDENTITY layer (12 §25): it changes only
    via fork, never by hardware change or lifecycle progression.
    """
    return content_hash(run_spec)


def file_hash(path: str) -> str:
    """Streaming SHA-256 of a file's bytes (for dataset/artifact manifests)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{ALGO}:{h.hexdigest()}"
