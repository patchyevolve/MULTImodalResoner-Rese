"""Model import — external package → registry entry with a real contract.

Normative: 13_product_specification §4.1 (`model import <PATH>`),
§4.3 (object reference syntax), 12_training_system.md §15.3 (the model
package file family: model_spec.json with inference_contract, weights,
integrity).

Design (fail-closed — an import we cannot verify is refused):
  * PATH must be a directory containing `model_spec.json` (with a full
    `inference_contract` — the whole point of importing) and the weights
    file `model.safetensors`;
  * name/version come from the spec (never from the directory name —
    identity is declared, not discovered);
  * the weights are content-hashed at import → `artifact_hash`, so the
    imported model participates in every later integrity check exactly
    like a published one;
  * a `name:version` that is already registered BLOCKs — versions are
    immutable (12 §4 spirit, 13 §4.3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound, ValidationBlock
from mlforge.hashing import file_hash

WEIGHTS_FILE = "model.safetensors"
SPEC_FILE = "model_spec.json"


def read_package(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the external package → (spec document, weights info)."""
    p = Path(path)
    if not p.is_dir():
        raise NotFound(
            f"import path not found or not a directory: {path}",
            hint="model import takes a packaged model directory containing "
                 f"{SPEC_FILE} + {WEIGHTS_FILE} (12 §15.3)",
        )
    spec_path = p / SPEC_FILE
    if not spec_path.is_file():
        raise ValidationBlock(
            f"package has no {SPEC_FILE}: {spec_path}",
            hint="a model package must declare its inference contract "
                 "(12 §15.3) — refusing an unverifiable import",
        )
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationBlock(f"{SPEC_FILE} is not valid JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValidationBlock(f"{SPEC_FILE} must be a JSON object")
    if not isinstance(spec.get("inference_contract"), dict):
        raise ValidationBlock(
            f"{SPEC_FILE} lacks an `inference_contract` object — "
            "the contract is what makes an import usable (12 §15.3)",
            hint="package the model with its contract, then import",
        )
    contract = spec["inference_contract"]
    for key in ("input_schema", "output_schema", "runtime"):
        if key not in contract:
            raise ValidationBlock(
                f"inference_contract lacks `{key}` — contract incomplete "
                "(12 §15.3)",
            )
    name, version = spec.get("name"), spec.get("version")
    if not name or not version:
        raise ValidationBlock(
            f"{SPEC_FILE} must declare `name` and `version` — identity is "
            "declared, never inferred from the directory name",
        )
    if not str(version).startswith("v"):
        raise ValidationBlock(
            f"declared version {version!r} must be `vN` (13 §4.3)",
        )
    weights = p / WEIGHTS_FILE
    if not weights.is_file():
        raise ValidationBlock(
            f"package has no weights file: {weights}",
            hint=f"{WEIGHTS_FILE} is required — an import without weights "
                 "is not a model (12 §15.3)",
        )
    return spec, {
        "path": weights,
        "artifact_hash": file_hash(weights),
        "bytes": weights.stat().st_size,
    }


__all__ = ["SPEC_FILE", "WEIGHTS_FILE", "read_package"]
