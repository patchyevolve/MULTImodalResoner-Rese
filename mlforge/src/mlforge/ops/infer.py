"""One-shot inference — contract check before execution (12 §15.3).

Normative: 12_training_system.md §15.3 ("Consumers verify against this
contract BEFORE loading ... SAFE | BLOCK with exact mismatch named";
no training-environment assumptions), 13_product_specification §6.8
(INFER sequence: validate model → validate input schema → execute),
§7 (input schema invalid → BLOCK before execution, list violations;
model/input not resolved → exit 2).

Design:
  * the contract is loaded from the model's export/import (see
    exporting.contract_source_dir); no contract ⇒ BLOCK (fail-closed);
  * EVERY input_schema entry is checked against the single one-shot
    INPUT; zero violations ⇒ SAFE, any violation ⇒ BLOCK listing each
    (never "probably fine");
  * execution is a deterministic scaffold (like `ScaffoldTrainer`): it
    proves the plumbing (contract → input → output artifact with its own
    identity) and explicitly does NOT run a model. The output records
    `harness: scaffold` — the real engine replaces it at integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound, ValidationBlock
from mlforge.hashing import content_hash, file_hash

OUTPUTS_DIR = "outputs"
OUTPUT_SCHEMA = 1

#: Extensions a contract `type: image` entry accepts (fail-closed list).
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


def check_input(contract_doc: dict[str, Any], input_path: str | Path) -> dict[str, Any]:
    """Validate INPUT against every input_schema entry (13 §6.8).

    Returns {"status": "SAFE", "input": {...}} or raises ValidationBlock
    with the full violation list. Missing file → NotFound (exit 2).
    """
    p = Path(input_path)
    if not p.is_file():
        raise NotFound(
            f"input not found: {input_path}",
            hint="one-shot infer takes a single existing file (13 §6.8)",
        )
    schema = contract_doc["inference_contract"].get("input_schema") or []
    if not schema:
        raise ValidationBlock(
            "inference contract declares no input_schema — "
            "cannot validate input (12 §15.3)",
        )
    size = p.stat().st_size
    violations: list[str] = []
    if size == 0:
        violations.append(f"{p.name}: empty file")
    ext = p.suffix.lower()
    for entry in schema:
        name = entry.get("name", "<unnamed>")
        etype = entry.get("type")
        if etype == "image":
            if ext not in IMAGE_EXTENSIONS:
                violations.append(
                    f"{name}: type image requires one of "
                    f"{', '.join(sorted(IMAGE_EXTENSIONS))}, got {ext or 'no extension'}"
                )
            if size and size > (64 << 20):
                violations.append(f"{name}: image larger than 64 MiB ({size} bytes)")
        elif etype == "text":
            try:
                p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                violations.append(
                    f"{name}: type text requires UTF-8 decodable content"
                )
        else:
            violations.append(
                f"{name}: unknown input type {etype!r} in contract "
                "(fail-closed — cannot verify)"
            )
    if violations:
        raise ValidationBlock(
            "input schema violations: " + "; ".join(violations),
            hint="BLOCK before execution (13 §7) — the contract at "
                 "model_spec.json is the authority (12 §15.3)",
        )
    return {
        "status": "SAFE",
        "input": {
            "path": str(p),
            "bytes": size,
            "extension": ext,
            "identity": file_hash(p),
        },
    }


def execute_scaffold(
    *,
    output_id: str,
    model_entry: dict[str, Any],
    contract_doc: dict[str, Any],
    checked: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic stand-in execution (honest scaffold — NOT a model).

    The artifact contract is real: own identity, contract hash, input
    identity; the `result` envelope is explicitly marked scaffold and
    carries no fabricated detections — a real inference engine fills it
    at integration (same pattern as `ScaffoldTrainer`)."""
    contract = contract_doc["inference_contract"]
    record: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "output_id": output_id,
        "created_ts": None,  # stamped by the workflow
        "model_id": model_entry.get("model_id"),
        "model_ref": (f"{model_entry.get('name')}:"
                      f"{model_entry.get('version')}"),
        "contract_hash": content_hash(contract),
        "runtime": contract.get("runtime"),
        "input": checked["input"],
        "result": {
            "note": "scaffold execution — no model was run; the real "
                    "inference engine replaces this envelope at integration",
        },
        "harness": "scaffold",
    }
    record["identity"] = content_hash(
        {
            "contract_hash": record["contract_hash"],
            "input_identity": record["input"]["identity"],
        }
    )
    return record


def output_dir(root: Path) -> Path:
    return Path(root) / OUTPUTS_DIR


def write_output(root: Path, record: dict[str, Any]) -> Path:
    d = output_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{record['output_id']}.json"
    if p.exists():
        raise ValidationBlock(f"output already exists: {p} — outputs are immutable")
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def load_output(root: Path, output_id: str) -> dict[str, Any]:
    p = output_dir(root) / f"{output_id}.json"
    if not p.is_file():
        raise NotFound(f"output {output_id} not found",
                       hint="one-shot outputs are under outputs/")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationBlock(f"output unreadable: {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationBlock(f"output must be a JSON object: {p}")
    return data


__all__ = [
    "IMAGE_EXTENSIONS",
    "OUTPUT_SCHEMA",
    "OUTPUTS_DIR",
    "check_input",
    "execute_scaffold",
    "load_output",
    "output_dir",
    "write_output",
]
