"""Transforms — deterministic, registry-gated, code-hashed (12 §6.4).

A transform turns VERIFIED source datasets into a prepared artifact:

    output = transform(input)                      # deterministic
    cache_key = SHA256(inputs + transform + config + environment)
                                                    # (12 §6.4)

Identity has four independent parts — a change in ANY of them invalidates
the cache (no stale reuse, no guessing):

  * input   — each source dataset's cryptographic identity
  * code    — hash of the transform implementation itself (`inspect`-based)
  * config  — hash of the transform configuration (empty today, wired for
              when per-model options arrive)
  * env     — python/platform fingerprint (placeholder until the OCI
              image digest lands: an image digest is a *better* env
              identity, same mechanism)

Unknown transforms are a ValidationBlock (exit 1) — the registry is
closed; we never "run whatever is there".
"""

from __future__ import annotations

import inspect
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mlforge.errors import ValidationBlock
from mlforge.hashing import content_hash, content_hash_bytes
from mlforge.ingest.dag import ModelPlan
from mlforge.ingest.identity import FileEntry

#: Prepared artifact envelope schema (what `mlforge prepare` persists).
PREPARED_SCHEMA = "mlforge.prepared_dataset.v1"


@dataclass(frozen=True)
class ResolvedSource:
    """One VERIFIED source dataset, ready for a transform."""

    ref: str                    # "coco_2017:train"
    split: str | None
    dataset_id: str
    path: Path
    identity: str               # cryptographic identity (sha256:...)
    entries: tuple[FileEntry, ...]

    @property
    def file_count(self) -> int:
        return len(self.entries)


def coco_detection(sources: list[ResolvedSource]) -> dict[str, Any]:
    """COCO-style detection data: one record per file, canonical re-hash
    of every JSON annotation (proves annotations PARSE, not just exist)."""
    records: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for src in sources:
        for entry in src.entries:
            records.append({
                "source": src.ref,
                "split": src.split,
                "relative_path": entry.relative_path,
                "sha256": entry.sha256,
                "size": entry.size,
            })
            if entry.relative_path.endswith(".json"):
                raw = (src.path / entry.relative_path).read_text(encoding="utf-8")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValidationBlock(
                        f"annotation does not parse as JSON: "
                        f"{src.ref}:{entry.relative_path}: {exc}"
                    ) from exc
                annotations.append({
                    "source": src.ref,
                    "relative_path": entry.relative_path,
                    "canonical_sha256": content_hash(parsed),
                })
    return {
        "output_schema": "coco_detection.v1",
        "records": records,
        "annotations": annotations,
    }


_REGISTRY: dict[str, Callable[[list[ResolvedSource]], dict[str, Any]]] = {
    "coco_detection": coco_detection,
}


def registry_names() -> list[str]:
    return sorted(_REGISTRY)


def get_transform(name: str) -> Callable[[list[ResolvedSource]], dict[str, Any]]:
    fn = _REGISTRY.get(name)
    if fn is None:
        raise ValidationBlock(
            f"unknown transform {name!r} (registry: {', '.join(registry_names())})",
            hint="fix `transform:` in ingestion.yaml — unregistered "
                 "transforms never run (fail-closed)",
        )
    return fn


def transform_identity(name: str, config: dict[str, Any] | None = None) -> dict[str, str]:
    """Four-field identity: name + code hash + config hash (12 §6.4)."""
    fn = get_transform(name)
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:  # source unavailable ⇒ identity unknown
        raise ValidationBlock(
            f"transform {name!r}: implementation source unavailable — "
            f"cannot compute code identity ({exc})"
        ) from exc
    return {
        "name": name,
        "code_hash": content_hash_bytes(source.encode("utf-8")),
        "config_hash": content_hash(config or {}),
    }


def env_fingerprint() -> str:
    """Environment identity — python + platform until the OCI digest lands
    (12 §6.4: environment_hash is a first-class cache-key field)."""
    return content_hash({
        "python": sys.version,
        "platform": platform.platform(),
    })


def cache_key(
    inputs: list[dict[str, str]],
    transform_identity_data: dict[str, str],
    environment: str,
) -> str:
    """cache_key = SHA256(input hashes + transform + config + env) (12 §6.4)."""
    return content_hash({
        "inputs": sorted(inputs, key=lambda d: d["ref"]),
        "transform": transform_identity_data,
        "environment": environment,
    })


def run_transform(
    plan: ModelPlan, sources: list[ResolvedSource]
) -> dict[str, Any]:
    """Execute + validate the transform output (fail-closed on empties)."""
    fn = get_transform(plan.transform)
    out = fn(sources)
    if not isinstance(out, dict) or "records" not in out:
        raise ValidationBlock(
            f"transform {plan.transform!r} returned no records field "
            f"(fail-closed: unverifiable output == failed output)"
        )
    records = out["records"]
    if not records:
        raise ValidationBlock(
            f"transform {plan.transform!r} produced 0 records — an empty "
            f"prepared dataset has no identity (fail-closed)"
        )
    # Every source must actually contribute (a silently skipped source
    # would make the artifact a guess).
    covered = {r.get("source") for r in records}
    missing = [s.ref for s in sources if s.ref not in covered]
    if missing:
        raise ValidationBlock(
            f"transform {plan.transform!r} produced no records for: "
            f"{', '.join(missing)}"
        )
    return out


__all__ = [
    "PREPARED_SCHEMA",
    "ResolvedSource",
    "cache_key",
    "coco_detection",
    "env_fingerprint",
    "get_transform",
    "registry_names",
    "run_transform",
    "transform_identity",
]
