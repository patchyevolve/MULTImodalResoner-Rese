"""Package — build the inference bundle (13 §4.1 `package <MODEL>`).

Normative: 12_training_system.md §15.3 (model/ bundle file list; "A
model without an inference contract cannot be packaged for reuse —
`mlforge package` BLOCKS"; package → validate → infer), 13 §7
(secrets detected in artifacts → BLOCK packaging/export).

Design:
  * packaging REQUIRES a contract (from a prior export or an import) —
    absent ⇒ BLOCK with the exact next command (`export ... --format`);
  * the bundle is an immutable artifact under `bundles/<bdl_...>/` with
    `model.safetensors` (the real weights — 12 §15.3 package file
    family) + `bundle.json` (identity) + `model_spec.json` (contract) +
    `provenance.json` (lineage) + `integrity.json` (component hashes);
    the weights are the WHOLE POINT of the family: `model import`
    (`read_package`) refuses a package without them, so a bundle that
    carried only hashes could never be imported;
  * a secrets scan runs over the bundle's SOURCE directories before
    anything is written — findings BLOCK the whole package (13 §7, no
    partial execution).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlforge.errors import ValidationBlock
from mlforge.hashing import content_hash

BUNDLES_DIR = "bundles"
BUNDLE_SCHEMA = 1


def build_bundle(
    *,
    bundle_id: str,
    model_entry: dict[str, Any],
    model_hash: str,
    model_spec: dict[str, Any],
    provenance: dict[str, Any],
    integrity: dict[str, str],
) -> dict[str, Any]:
    """Bundle manifest with its own identity (12 §15.3 integrity.json)."""
    contract = model_spec["inference_contract"]
    record: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        "created_ts": None,  # stamped by the workflow
        "model_id": model_entry.get("model_id"),
        "model_ref": (f"{model_entry.get('name')}:"
                      f"{model_entry.get('version')}"),
        "model_hash": model_hash,
        "contract_hash": content_hash(contract),
        "artifact_hash": model_entry.get("artifact_hash"),
        "files": ["model.safetensors", "bundle.json", "model_spec.json",
                  "provenance.json", "integrity.json"],
        "integrity": integrity,
        "provenance": provenance,
        "harness": model_spec.get("harness", "scaffold"),
    }
    record["identity"] = content_hash(
        {
            "model_hash": record["model_hash"],
            "contract_hash": record["contract_hash"],
            "artifact_hash": record["artifact_hash"],
            "integrity": integrity,
        }
    )
    return record


def write_bundle(
    root: Path,
    *,
    record: dict[str, Any],
    model_spec: dict[str, Any],
    provenance: dict[str, Any],
    integrity: dict[str, str],
    weights: bytes,
) -> Path:
    """Write the bundle atomically, weights FIRST.

    `model.safetensors` lands before any JSON so a crash mid-write can
    never leave a complete-looking bundle that is missing (or lying
    about) its weights — `model import` hashes the file it finds
    (`read_package`), and an interrupted package is retried, never
    imported half-written.
    """
    d = Path(root) / BUNDLES_DIR / str(record["bundle_id"])
    d.mkdir(parents=True, exist_ok=True)

    def _write_bytes(p: Path, payload: bytes) -> None:
        if p.exists():
            raise ValidationBlock(
                f"bundle file already exists: {p} — bundles are immutable")
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(p)

    def _write_json(p: Path, payload: dict[str, Any]) -> None:
        _write_bytes(p, json.dumps(payload, indent=2, sort_keys=True)
                     .encode("utf-8"))

    from mlforge.ops.importing import WEIGHTS_FILE

    _write_bytes(d / WEIGHTS_FILE, weights)
    _write_json(d / "bundle.json", record)
    _write_json(d / "model_spec.json", model_spec)
    _write_json(d / "provenance.json", provenance)
    _write_json(d / "integrity.json", integrity)
    return d


def load_bundle(root: Path, bundle_id: str) -> dict[str, Any]:
    from mlforge.errors import NotFound

    p = Path(root) / BUNDLES_DIR / bundle_id / "bundle.json"
    if not p.is_file():
        raise NotFound(f"bundle {bundle_id} not found",
                       hint="mlforge bundles are under bundles/")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationBlock(f"bundle unreadable: {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationBlock(f"bundle must be a JSON object: {p}")
    return data


def list_bundles(root: Path, *, model_id: str | None = None) -> list[dict[str, Any]]:
    base = Path(root) / BUNDLES_DIR
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(base.iterdir()):
        p = d / "bundle.json"
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationBlock(f"bundle unreadable: {p}: {exc}") from exc
        if model_id is not None and data.get("model_id") != model_id:
            continue
        out.append(data)
    return out


def component_integrity(
    model_spec: dict[str, Any],
    provenance: dict[str, Any],
    artifact_hash: str | None,
) -> dict[str, str]:
    """Per-component hashes for integrity.json (12 §15.3)."""
    from mlforge.hashing import content_hash as ch

    integrity = {
        "model_spec": ch(model_spec),
        "provenance": ch(provenance),
    }
    if artifact_hash:
        integrity["weights"] = artifact_hash
    return integrity


__all__ = [
    "BUNDLE_SCHEMA",
    "BUNDLES_DIR",
    "build_bundle",
    "component_integrity",
    "list_bundles",
    "load_bundle",
    "write_bundle",
]
