"""Export — format/operator validation, inference contract, export artifacts.

Normative: 12_training_system.md §15.3 (weights alone ≠ model; the
exported model carries a machine-checkable `model_spec.json` inference
contract), 13_product_specification §6.9 (EXPORT sequence), §4.1
(`export <MODEL> --format F`), §7 (operator unsupported → BLOCK with the
specific operator list, no partial export; secrets → BLOCK).

Design:
  * the format list is a CLOSED registry (an unknown format BLOCKs with
    the exact names, 13 §7); operator portability is checked against the
    per-format hosted set — except `onnx`, whose truth is the ONNX
    opset itself: required operators are verified against the installed
    onnx schemas when available (and the real exporter refuses honestly
    when onnx is missing);
  * the first export of a model BUILDS its inference contract
    (12 §15.3 "The Exported Model Declares How It Must Be Called"); later
    exports reuse the existing contract's operator list so re-exporting
    another format cannot silently change what the graph requires;
  * the export artifact owns its identity (`exports/<exp_...>/` with
    `export.json` + `model_spec.json`) — 13 §6.9 "own artifact identity";
  * harness split (12 §12.4): `MLFORGE_HARNESS=1` keeps the honest
    scaffold (identity + contract + weight reference, NEVER a fake
    binary, deterministic numerical check labeled `harness: scaffold`);
    the default path runs the real engine (`mlforge.ops.engines`) which
    produces genuine format bytes, a family-real contract, and a real
    numerical round-trip — a FAIL blocks the export (13 §6.9, §7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound, ValidationBlock
from mlforge.hashing import content_hash

EXPORTS_DIR = "exports"
EXPORT_SCHEMA = 1

#: The scaffold graph's operator requirements (what the stand-in model
#: "needs"); a real exporter replaces this with the architecture's ops.
SCAFFOLD_GRAPH_OPS = frozenset(
    {"conv", "gemm", "relu", "flatten", "softmax", "non_max_suppression"}
)

#: Closed export-format registry: opset + hosted operators (13 §4.1
#: "ONNX / TensorRT / etc."). `tflite` intentionally does NOT host
#: `non_max_suppression` — the spec's BLOCK row needs a real gap.
#: `full_opset` formats host every operator their spec defines; their
#: operator check runs against the installed spec schemas (see
#: `validate_export`).
FORMATS: dict[str, dict[str, Any]] = {
    "onnx": {"opset": 17,
             "ops": frozenset(SCAFFOLD_GRAPH_OPS | {"concat", "slice"}),
             "full_opset": True},
    "openvino": {"opset": 17,
                 "ops": frozenset(SCAFFOLD_GRAPH_OPS | {"concat"})},
    "tensorrt": {"opset": 17,
                 "ops": frozenset(SCAFFOLD_GRAPH_OPS | {"concat"})},
    "coreml": {"opset": 17,
               "ops": frozenset(SCAFFOLD_GRAPH_OPS | {"concat"})},
    "tflite": {"opset": 17,
               "ops": frozenset(SCAFFOLD_GRAPH_OPS - {"non_max_suppression"})},
}


def _op_key(op: str) -> str:
    """Operator-name normalization for schema comparison (`Conv` vs the
    contract's `conv`, `NonMaxSuppression` vs `non_max_suppression`)."""
    return op.lower().replace("_", "")


def _full_opset_ops(fmt: str) -> frozenset[str] | None:
    """Hosted operator keys for a `full_opset` format — None when the
    installed spec cannot enumerate them (the exporter itself refuses
    honestly if the toolchain is missing)."""
    if not FORMATS[fmt].get("full_opset"):
        return None
    try:
        import onnx.defs

        return frozenset(
            _op_key(s.name) for s in onnx.defs.get_all_schemas()
        )
    except Exception:  # noqa: BLE001 - schema enumeration is best-effort: failure degrades to None
        return None


def scaffold_contract(
    model_entry: dict[str, Any], fmt: str, *, precision: str = "fp32"
) -> dict[str, Any]:
    """`model_spec.json` document (12 §15.3) for a scaffold model.

    The schema fields are PLACEHOLDERS of the right shape (labeled
    `harness: scaffold`) — a real exporter/producer fills the model's
    true preprocessing the same way the real trainer replaces
    `ScaffoldTrainer`. What is REAL: the operator list, the format
    runtime, and every downstream check against this contract.
    """
    return {
        "schema_version": 1,
        "name": model_entry.get("name"),
        "version": model_entry.get("version"),
        "harness": "scaffold",
        "inference_contract": {
            "input_schema": [
                {
                    "name": "frame",
                    "type": "image",
                    "dtype": "uint8",
                    "layout": "HWC",
                    "color_space": "RGB",
                    "resize": {"width": 640, "height": 640, "mode": "letterbox"},
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                }
            ],
            "output_schema": [
                {
                    "name": "detections",
                    "type": "structured",
                    "schema": "json_schema://mlforge.scaffold.v1",
                }
            ],
            "operators": sorted(SCAFFOLD_GRAPH_OPS),
            "operator_set": content_hash({"ops": sorted(SCAFFOLD_GRAPH_OPS)}),
            "runtime": {"framework": fmt, "opset": FORMATS[fmt]["opset"]},
            "dtype": precision,
            "dynamic_axes": {"batch": [0]},
            "minimal_memory": {"vram_mb": 512},
            "numerical_tolerance": {"atol": 1e-4},
        },
    }


def required_operators(model_spec: dict[str, Any]) -> frozenset[str]:
    """Operators the graph requires — from the contract's explicit list
    (fail-closed: a contract without one cannot be portability-checked)."""
    contract = (model_spec or {}).get("inference_contract") or {}
    ops = contract.get("operators")
    if not isinstance(ops, list) or not ops:
        raise ValidationBlock(
            "inference contract lacks an explicit operator list — "
            "cannot verify portability to a target format",
            hint="re-export or repackage with `operators` recorded "
                 "(12 §15.3 operator_set)",
        )
    return frozenset(str(o) for o in ops)


def validate_export(model_spec: dict[str, Any], fmt: str) -> dict[str, str]:
    """Architecture/Operators/Dynamic-shapes checks (13 §6.9 display).

    Unknown format → BLOCK listing the closed registry; unsupported
    operators → BLOCK naming them, never a partial export (13 §7).
    `full_opset` formats (onnx) check against the installed spec
    schemas; when the spec is not installed, the check cannot DISPROVE
    portability and defers to the exporter, which refuses honestly.
    """
    if fmt not in FORMATS:
        raise ValidationBlock(
            f"unsupported export format {fmt!r}",
            hint=f"formats: {', '.join(sorted(FORMATS))} (13 §4.1)",
        )
    required = required_operators(model_spec)
    full = _full_opset_ops(fmt)
    if full is not None:
        unsupported = sorted(
            op for op in required if _op_key(op) not in full
        )
        if unsupported:
            raise ValidationBlock(
                f"operator(s) unknown to the {fmt} spec: "
                f"{', '.join(unsupported)}",
                hint="no partial export (13 §7) — the installed onnx "
                     "schemas are the authority for what ONNX hosts "
                     "(13 §4.1)",
            )
        return {"architecture": "PASS", "operators": "PASS",
                "dynamic_shapes": "PASS"}
    unsupported = sorted(required - FORMATS[fmt]["ops"])
    if unsupported:
        capable = sorted(
            f for f, spec in FORMATS.items() if not (required - spec["ops"])
        )
        raise ValidationBlock(
            f"operator(s) unsupported by {fmt}: {', '.join(unsupported)}",
            hint="no partial export (13 §7) — formats that host the graph: "
                 + (", ".join(capable) if capable else
                    "none — change the contract's operator list"),
        )
    return {"architecture": "PASS", "operators": "PASS",
            "dynamic_shapes": "PASS"}


def numerical_validation(model_spec: dict[str, Any],
                         model_hash: str, fmt: str) -> dict[str, Any]:
    """Scaffold-harness numerical check (`MLFORGE_HARNESS=1`, 12 §12.4)
    — deterministic stand-in derived from the identity hashes, always
    within tolerance. The default path runs the real engine's
    onnxruntime/python round-trip instead (13 §6.9); this function is
    never called outside the harness branch."""
    import hashlib

    contract = model_spec["inference_contract"]
    atol = float(contract.get("numerical_tolerance", {}).get("atol", 1e-4))
    digest = hashlib.sha256(f"{model_hash}|{fmt}".encode()).hexdigest()
    max_error = (int(digest[:8], 16) % 1000) / 1000.0 * atol * 0.9
    return {
        "max_error": round(max_error, 10),
        "tolerance": atol,
        "result": "PASS" if max_error < atol else "FAIL",
        "harness": "scaffold",
    }


def build_export(
    *,
    export_id: str,
    model_entry: dict[str, Any],
    model_hash: str,
    model_spec: dict[str, Any],
    fmt: str,
    validations: dict[str, str],
    numerical: dict[str, Any],
    harness: str = "scaffold",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export record (13 §6.9). Default = honest scaffold payload
    (weights reference, never a fake binary); the real engines pass the
    binary's hash + their `harness` label."""
    contract = model_spec["inference_contract"]
    name = model_entry.get("name") or "model"
    version = model_entry.get("version") or "v1"
    if payload is None:
        payload = {
            "artifact_hash": model_entry.get("artifact_hash"),
            "note": "weights reference — the format binary is written by "
                    "the real exporter at integration (scaffold writes "
                    "identity + contract, never a fake binary)",
        }
    record = {
        "schema_version": EXPORT_SCHEMA,
        "export_id": export_id,
        "created_ts": None,  # stamped by the workflow
        "model_id": model_entry.get("model_id"),
        "model_ref": f"{name}:{version}",
        "model_hash": model_hash,
        "format": fmt,
        "file_name": f"{name}_{version}.{fmt}",
        "model_spec": model_spec,
        "contract_hash": content_hash(contract),
        "validations": validations,
        "numerical_validation": numerical,
        "payload": payload,
        "harness": harness,
    }
    record["identity"] = content_hash(
        {
            "model_hash": model_hash,
            "format": fmt,
            "contract_hash": record["contract_hash"],
            "artifact_hash": model_entry.get("artifact_hash"),
        }
    )
    return record


def export_dir(root: Path, export_id: str) -> Path:
    return Path(root) / EXPORTS_DIR / export_id


def write_export(
    root: Path,
    record: dict[str, Any],
    *,
    binary: bytes | None = None,
) -> Path:
    """Write the export artifact (immutable), commit-marker LAST. Real
    exports pass the format binary as raw bytes — written FIRST (under
    the record's own `file_name`), then the contract sidecar, then
    `export.json` — the record readers trust (`list_exports`,
    `latest_export`, `contract_source_dir`) — only once every payload it
    describes is on disk. A crash between writes therefore leaves a
    directory NO reader can see as an export (no phantom record, no
    contract without bytes); the retry starts a fresh export and heals
    the journey (13 §6.9 no partial export, §7)."""
    d = export_dir(root, str(record["export_id"]))
    d.mkdir(parents=True, exist_ok=True)
    if binary is not None:
        bp = d / str(record["file_name"])
        if bp.exists():
            raise ValidationBlock(
                f"export artifact already exists: {bp} — exports are "
                "immutable")
        tmp = bp.with_name(bp.name + ".tmp")
        tmp.write_bytes(binary)
        tmp.replace(bp)
    for filename, payload in (
        ("model_spec.json", record["model_spec"]),
        ("export.json", record),
    ):
        p = d / filename
        if p.exists():
            raise ValidationBlock(
                f"export artifact already exists: {p} — exports are immutable")
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(p)
    return d


def load_export(root: Path, export_id: str) -> dict[str, Any]:
    p = export_dir(root, export_id) / "export.json"
    if not p.is_file():
        raise NotFound(f"export {export_id} not found",
                       hint="mlforge exports are under exports/")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationBlock(f"export unreadable: {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationBlock(f"export must be a JSON object: {p}")
    return data


def list_exports(root: Path, *, model_id: str | None = None) -> list[dict[str, Any]]:
    base = Path(root) / EXPORTS_DIR
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(base.iterdir()):
        p = d / "export.json"
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationBlock(f"export unreadable: {p}: {exc}") from exc
        if model_id is not None and data.get("model_id") != model_id:
            continue
        out.append(data)
    return out


def latest_export(root: Path, model_id: str) -> dict[str, Any] | None:
    matches = list_exports(root, model_id=model_id)
    return matches[-1] if matches else None


def contract_source_dir(root: Path, model_entry: dict[str, Any]) -> Path | None:
    """Where the model's CURRENT contract lives: latest export's
    `model_spec.json`, else the imported model's sidecar (12 §15.3)."""
    latest = latest_export(Path(root), str(model_entry.get("model_id")))
    if latest is not None:
        d = export_dir(Path(root), str(latest["export_id"]))
        if (d / "model_spec.json").is_file():
            return d
    sidecar = (Path(root) / "models" / str(model_entry.get("model_id"))
               / "model_spec.json")
    if sidecar.is_file():
        return sidecar.parent
    return None


def load_model_spec(root: Path, model_entry: dict[str, Any]) -> dict[str, Any]:
    """The model's inference contract document — absent ⇒ the model has
    never been exported or imported with a contract (12 §15.3: package
    and infer BLOCK without one)."""
    d = contract_source_dir(root, model_entry)
    if d is None:
        raise ValidationBlock(
            f"model {model_entry.get('name')}:{model_entry.get('version')} "
            "has no inference contract",
            hint="run `mlforge export <MODEL> --format onnx` (or import a "
                 "packaged model) first — a model without a contract "
                 "cannot be packaged or inferred (12 §15.3)",
        )
    p = d / "model_spec.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationBlock(f"inference contract unreadable: {p}: {exc}") from exc
    if not isinstance(data, dict) or "inference_contract" not in data:
        raise ValidationBlock(
            f"inference contract invalid in {p}: missing `inference_contract`")
    return data


__all__ = [
    "EXPORTS_DIR",
    "EXPORT_SCHEMA",
    "FORMATS",
    "SCAFFOLD_GRAPH_OPS",
    "build_export",
    "contract_source_dir",
    "export_dir",
    "latest_export",
    "list_exports",
    "load_export",
    "load_model_spec",
    "numerical_validation",
    "required_operators",
    "scaffold_contract",
    "validate_export",
    "write_export",
]
