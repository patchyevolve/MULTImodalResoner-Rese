"""Evaluation artifacts — five-component identity + protocol comparability.

Normative: 12_training_system.md §15.4 (evaluation references immutable
artifacts; EVALUATION IDENTITY = model_hash + dataset_hash + code_hash +
environment_hash + evaluation_protocol_hash), 13_product_specification
§6.7 (EVALUATE sequence), §6.10 (compare: NOT_COMPARABLE on protocol
mismatch), §7 (model/dataset not resolved → exit 2, nothing run).

Design:
  * an evaluation is an IMMUTABLE artifact under
    `<root>/evaluations/<eval_id>/evaluation.json` — a later evaluation
    of the same model with a different protocol is a DIFFERENT artifact,
    never an update (12 §15.4);
  * `evaluation_protocol_hash` is computed from the protocol document
    itself (split, metric definitions, aggregation, thresholds, nms,
    sample_subset, seed, harness_code_hash) — `mlforge compare` gates
    comparability on it (13 §6.10);
  * the metric HARNESS is pluggable like the Trainer (12 §12.4): the
    REAL engine runs by default (mlforge.ops.engines — family metrics
    over the prepared dataset); the deterministic `scaffold_metrics`
    stands behind `MLFORGE_HARNESS=1` (system tests) — derived from the
    identity hashes so the reproducibility contract stays real. Either
    way `harness_code_hash` covers the harness implementation (scaffold
    source + the engine package), so swapping engines changes
    comparability exactly as it should.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mlforge.errors import ValidationBlock
from mlforge.hashing import content_hash, content_hash_bytes

EVALUATIONS_DIR = "evaluations"
EVALUATION_SCHEMA = 1

#: Metric names the scaffold harness emits (the protocol's
#: `metric_definitions` hashes whatever `metric_names` the selected
#: harness declares — the default is this scaffold set).
DEFAULT_METRIC_NAMES = ("mAP", "AP50")

#: Default evaluation protocol (12 §15.4). `split` is never "train".
#: `metric_definitions` + `harness_code_hash` are DERIVED by
#: `build_protocol` — they may not be overridden from a file (faking the
#: harness identity would fake comparability).
DEFAULT_PROTOCOL: dict[str, Any] = {
    "split": "val",
    "metric_definitions": None,  # filled by `build_protocol` (hash below)
    "aggregation": "macro",
    "thresholds": {"iou": 0.5, "conf": 0.05},
    "nms": {"iou_thresh": 0.5, "max_det": 100},
    "sample_subset": None,
    "seed": 123456,
    "harness_code_hash": None,  # filled by `build_protocol`
}

#: Fields computed from real code — never user-overridable.
DERIVED_PROTOCOL_FIELDS = frozenset({"metric_definitions", "harness_code_hash"})


def harness_code_hash() -> str:
    """Code identity of THIS harness (12 §15.4 `harness_code_hash`) —
    real: sha256 over the scaffold implementation source PLUS every
    engine module under `mlforge.ops.engines` (the metric engines ARE
    the harness; swapping one changes comparability)."""
    import mlforge.ops.engines as engines_pkg

    h = hashlib.sha256()
    h.update(inspect.getsource(scaffold_metrics).encode("utf-8"))
    h.update(b"\0")
    pkg = Path(engines_pkg.__file__).parent
    for p in sorted(pkg.rglob("*.py")):
        h.update(str(p.relative_to(pkg)).encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
    return content_hash_bytes(h.digest())


def source_tree_hash() -> str:
    """`code_hash` component of the evaluation identity (12 §15.4) —
    real: sha256 over every MLForge source file (path + bytes), so a
    code change makes later evaluations non-reproducible against
    earlier ones, which is exactly the point of recording it."""
    import hashlib

    import mlforge

    pkg = Path(mlforge.__file__).parent
    h = hashlib.sha256()
    for p in sorted(pkg.rglob("*.py")):
        h.update(str(p.relative_to(pkg)).encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
    return content_hash_bytes(h.digest())


def build_protocol(
    overrides: dict[str, Any] | None = None,
    *,
    metric_names: Sequence[str] = DEFAULT_METRIC_NAMES,
) -> dict[str, Any]:
    """Complete protocol document + derived metric/harness hashes.

    `metric_names` is what the SELECTED harness measures (the real
    engines pass their family metrics; the scaffold path passes
    `DEFAULT_METRIC_NAMES`) — it feeds the derived `metric_definitions`
    hash, so a different metric set is a different protocol (12 §15.4).
    """
    if not metric_names:
        raise ValidationBlock(
            "metric_names must be non-empty — a protocol measures "
            "something (12 §15.4)")
    protocol = json.loads(json.dumps(DEFAULT_PROTOCOL))  # deep copy
    if overrides:
        unknown = set(overrides) - set(DEFAULT_PROTOCOL)
        if unknown:
            raise ValidationBlock(
                f"unknown evaluation protocol keys: {', '.join(sorted(unknown))}",
                hint=f"protocol keys: {', '.join(sorted(DEFAULT_PROTOCOL))}",
            )
        derived = set(overrides) & DERIVED_PROTOCOL_FIELDS
        if derived:
            raise ValidationBlock(
                f"derived protocol fields cannot be overridden: "
                f"{', '.join(sorted(derived))}",
                hint="metric definitions and the harness code hash come from "
                     "the harness itself (12 §15.4) — faking them would fake "
                     "comparability",
            )
        protocol.update(overrides)
    protocol["metric_definitions"] = content_hash(
        {"metrics": list(metric_names)}
    )
    protocol["harness_code_hash"] = harness_code_hash()
    return protocol


def protocol_hash(protocol: dict[str, Any]) -> str:
    return content_hash(protocol)


def model_entry_hash(entry: dict[str, Any]) -> str:
    """`model_hash` — identity of the model registry entry (weights
    artifact when present, else the immutable sidecar fields)."""
    return content_hash(
        {
            "name": entry.get("name"),
            "version": entry.get("version"),
            "artifact_hash": entry.get("artifact_hash"),
            "run_spec_hash": entry.get("run_spec_hash"),
        }
    )


def scaffold_metrics(
    model_hash: str, dataset_hash: str, protocol_hash_value: str
) -> dict[str, float]:
    """Deterministic stand-in metrics (honest scaffold — NOT a model).

    Derived from sha256(model_hash | dataset_hash | protocol_hash | seed)
    so the reproducibility contract is REAL: identical identity inputs
    always produce identical numbers; any component change changes them.
    Values are plausible in [0.2, 0.9] and AP50 >= mAP by construction.
    """
    import hashlib

    digest = hashlib.sha256(
        f"{model_hash}|{dataset_hash}|{protocol_hash_value}".encode()
    ).hexdigest()
    map_value = 0.2 + (int(digest[0:8], 16) % 60000) / 100000.0  # [0.2, 0.8)
    ap50 = min(0.99, map_value + 0.1 + (int(digest[8:16], 16) % 1000) / 10000.0)
    return {"mAP": round(map_value, 4), "AP50": round(ap50, 4)}


def build_evaluation(
    *,
    eval_id: str,
    model_entry: dict[str, Any],
    dataset_hash: str,
    dataset_ref: str,
    code_hash: str,
    environment_hash: str,
    protocol: dict[str, Any],
    metrics: dict[str, float] | None = None,
    harness: str = "scaffold",
    note: str | None = None,
) -> dict[str, Any]:
    """Full five-component evaluation record (12 §15.4).

    `metrics=None` ⇒ the deterministic scaffold harness (honest stand-in
    — identity, protocol, comparability are real, measurement is not);
    a real engine passes its measured numbers + its own `harness` label.
    """
    ph = protocol_hash(protocol)
    mh = model_entry_hash(model_entry)
    if metrics is None:
        metrics = scaffold_metrics(mh, dataset_hash, ph)
        note = note or (
            "deterministic stand-in metrics (scaffold harness) — "
            "identity, protocol, and comparability are real; the "
            "measurement engine plugs in at integration"
        )
    else:
        note = note or (
            f"measured by the real {harness} engine over the "
            "evaluation dataset (12 §12.4)"
        )
    return {
        "schema_version": EVALUATION_SCHEMA,
        "eval_id": eval_id,
        "created_ts": None,  # stamped by the workflow on write
        "model_id": model_entry.get("model_id"),
        "model_ref": f"{model_entry.get('name')}:{model_entry.get('version')}",
        "model_hash": mh,
        "model_artifact_hash": model_entry.get("artifact_hash"),
        "dataset_ref": dataset_ref,
        "dataset_hash": dataset_hash,
        "code_hash": code_hash,
        "environment_hash": environment_hash,
        "evaluation_protocol": protocol,
        "evaluation_protocol_hash": ph,
        "identity": content_hash(
            {
                "model_hash": mh,
                "dataset_hash": dataset_hash,
                "code_hash": code_hash,
                "environment_hash": environment_hash,
                "evaluation_protocol_hash": ph,
            }
        ),
        "metrics": {k: metrics[k] for k in metrics},
        "harness": harness,
        "note": note,
    }


# -- artifact storage (immutable, write-once) --------------------------------


def evaluation_dir(root: Path, eval_id: str) -> Path:
    return Path(root) / EVALUATIONS_DIR / eval_id


def write_evaluation(root: Path, record: dict[str, Any]) -> Path:
    """Write-once: an evaluation artifact is immutable once produced."""
    d = evaluation_dir(root, str(record["eval_id"]))
    d.mkdir(parents=True, exist_ok=True)
    p = d / "evaluation.json"
    if p.exists():
        raise ValidationBlock(
            f"evaluation artifact already exists: {p} — artifacts are immutable",
            hint="a different protocol is a DIFFERENT evaluation (12 §15.4), "
                 "never an update",
        )
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def load_evaluation(root: Path, eval_id: str) -> dict[str, Any]:
    p = evaluation_dir(root, eval_id) / "evaluation.json"
    if not p.is_file():
        from mlforge.errors import NotFound

        raise NotFound(f"evaluation {eval_id} not found",
                       hint="mlforge evaluations are under evaluations/")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationBlock(f"evaluation unreadable: {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationBlock(f"evaluation must be a JSON object: {p}")
    return data


def list_evaluations(root: Path, *, model_id: str | None = None) -> list[dict[str, Any]]:
    """All evaluation records (optionally one model's), newest last."""
    base = Path(root) / EVALUATIONS_DIR
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(base.iterdir()):
        p = d / "evaluation.json"
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationBlock(f"evaluation unreadable: {p}: {exc}") from exc
        if model_id is not None and data.get("model_id") != model_id:
            continue
        out.append(data)
    return out


def latest_evaluation(root: Path, model_id: str) -> dict[str, Any] | None:
    matches = list_evaluations(root, model_id=model_id)
    return matches[-1] if matches else None


def comparability_groups(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Split into comparable groups + the NOT_COMPARABLE remainder.

    13 §6.10: different `evaluation_protocol_hash` ⇒ columns marked
    NOT_COMPARABLE — shown, never averaged together.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(str(rec.get("evaluation_protocol_hash")), []).append(rec)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if not ordered:
        return {}, {}
    if len(records) >= 2 and all(len(rs) == 1 for _, rs in ordered):
        # every column used a different protocol — "any two" differ (§6.10)
        return {}, dict(ordered)
    main_hash, main = ordered[0]
    comparable = {main_hash: main}
    not_comparable = {h: rs for h, rs in ordered[1:]}
    return comparable, not_comparable


__all__ = [
    "DEFAULT_METRIC_NAMES",
    "DEFAULT_PROTOCOL",
    "EVALUATIONS_DIR",
    "EVALUATION_SCHEMA",
    "build_evaluation",
    "build_protocol",
    "comparability_groups",
    "evaluation_dir",
    "harness_code_hash",
    "latest_evaluation",
    "list_evaluations",
    "load_evaluation",
    "model_entry_hash",
    "protocol_hash",
    "scaffold_metrics",
    "source_tree_hash",
    "write_evaluation",
]
