"""Calibrator engine — temperature/conformal inference, ECE/NLL/coverage
metrics, and a real ONNX export (03_training_pipeline.md Model 5).

Normative: 10_training_plan/03_training_pipeline.md Model 5 (temperature
scaling on logits, conformal thresholds at alpha ∈ {0.05, 0.10},
decomposition weights), 06_benchmarking_plan.md (ECE < 0.05), 13
Product Specification §6.7–§6.9, 12 §15.3 (contract declares how the
model is called), §12.4 (real harness — no scaffold).

Family contract (stdlib core; `onnx`/`onnxruntime` imported only inside
export functions — the module is import-safe everywhere):
  * execute    — one-shot structured JSON `{"logits": [...]}` (or
                 `{"probs": [...]}`) → temperature-scaled probabilities,
                 the predicted class, and the conformal prediction sets
                 at every fitted alpha;
  * metrics    — `ece` + `nll` + `mce` + `brier` computed with the
                 TRAINER's own formulas (`_ece`/`_nll`/`_mce`/`_brier`)
                 at the fitted temperature, plus per-alpha coverage and
                 average prediction-set size for EVERY fitted conformal
                 alpha (`coverage_alpha_05`, `avg_set_size_alpha_10`, …
                 — 06 §5's `evaluate_calibration`, spec targets :349-353);
  * export     — a genuine ONNX graph (Div by T → Softmax) round-tripped
                 against the python implementation; FAIL blocks export.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash

FAMILY = "calibrator"

#: Models this engine runs (12 §10.2 `calibrator`).
MODEL_NAMES = frozenset({"calibrator"})

#: Numerical round-trip tolerance for the ONNX export (13 §6.9).
ATOL = 1e-6

#: Contract operator list — the two ops the calibration graph needs.
OPERATORS: tuple[str, ...] = ("Div", "Softmax")


def dependency_error() -> str | None:
    """None = runnable here (the fit/apply math is stdlib-only; onnx is
    checked at export time so inference/metrics work everywhere)."""
    return None


def input_types() -> frozenset[str]:
    return frozenset({"structured"})


def _alpha_key(alpha: float) -> str:
    """Metric-key fragment for a conformal alpha — the spec's literal
    naming (`06:351-353`: alpha 0.05 → `05`, 0.10 → `10`): the digits
    after the decimal point, zero-padded to 2 (`0.20` → `20`, `0.50` →
    `50`), never the raw concatenation `005`."""
    return f"{float(alpha):.2f}".removeprefix("0.").replace(".", "")


def _alpha_keys(alphas: Sequence[str]) -> list[str]:
    """Key per fitted alpha label (same order), fail-closed on any key
    collision — two alphas that round to the same 0.01 granularity can
    never share one metric silently."""
    keys: list[str] = []
    for label in alphas:
        key = _alpha_key(float(label))
        if key in keys:
            raise ValidationBlock(
                f"conformal alpha {label!r} collides with another fitted "
                f"alpha at metric-key precision ({key})",
                hint="metric keys use 2 decimals (spec's alpha_05/alpha_10) "
                     "— fit alphas distinct at 0.01 granularity")
        keys.append(key)
    return keys


def metric_names(weights: bytes | None = None) -> tuple[str, ...]:
    """Protocol `metric_definitions` for THIS model (12 §15.4): base
    calibration quality + one coverage / average-set-size pair per
    FITTED conformal alpha (06 §5 evaluate_calibration, targets
    06:349-353). Without weights the spec defaults [0.05, 0.10] are
    assumed; a custom `semantic.alphas` model gets ITS OWN keys
    (`coverage_alpha_20`, …) so its evaluations stay honest instead of
    being forced into metrics its thresholds cannot answer."""
    state = _state(weights, {}) if weights is not None else None
    labels = (["0.05", "0.10"] if state is None
              else [str(label) for label, _ in _alphas(state)])
    keys = _alpha_keys(labels)
    return (
        ("ece", "nll", "mce", "brier")
        + tuple(f"coverage_alpha_{k}" for k in keys)
        + tuple(f"avg_set_size_alpha_{k}" for k in keys)
    )


def _state(weights: bytes, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Checkpoint `model` blob → fitted calibrator state (fail-closed —
    a blob that is not this family's JSON state is refused)."""
    name = f"{entry.get('name')}:{entry.get('version')}"
    try:
        state = json.loads(weights.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationBlock(
            f"model {name}: checkpoint model blob is not calibrator "
            f"JSON state ({exc})",
            hint="the calibrator's `model` component is the fitted "
                 "temperature/threshold JSON (03 Model 5)",
        ) from exc
    if not isinstance(state, dict):
        raise ValidationBlock(
            f"model {name}: calibrator state must be a JSON object")
    temp = state.get("temperature")
    if isinstance(temp, bool) or not isinstance(temp, (int, float)) \
            or not math.isfinite(float(temp)) or float(temp) <= 0:
        raise ValidationBlock(
            f"model {name}: state.temperature {temp!r} is not a positive "
            "finite float")
    classes = state.get("classes")
    if isinstance(classes, bool) or not isinstance(classes, int) \
            or classes < 2:
        raise ValidationBlock(
            f"model {name}: state.classes {classes!r} must be an int >= 2")
    state["temperature"] = float(temp)
    state["classes"] = classes
    return state


def _alphas(state: Mapping[str, Any]) -> list[tuple[str, float]]:
    """Fitted (alpha_label, threshold) pairs from state.thresholds."""
    raw = state.get("thresholds")
    if not isinstance(raw, dict) or not raw:
        return []
    out: list[tuple[str, float]] = []
    for key in sorted(raw, key=lambda k: float(k)):
        val = raw[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)) \
                or not math.isfinite(float(val)):
            raise ValidationBlock(
                f"state.thresholds[{key!r}] {val!r} is not a finite float")
        out.append((str(key), float(val)))
    return out


def _probs_from_doc(doc: Any, classes: int) -> list[float]:
    """One-shot JSON → raw class scores as logits (fail-closed)."""
    if not isinstance(doc, dict):
        raise ValidationBlock(
            "structured calibrator input must be a JSON object with "
            '`logits` or `probs`',
            hint="13 §6.8 — the contract at model_spec.json is the "
                 "authority")
    if "logits" in doc:
        values = doc["logits"]
    elif "probs" in doc:
        values = doc["probs"]
    else:
        raise ValidationBlock(
            "calibrator input lacks `logits` (or `probs`) — expected "
            '{"logits": [number, ...]}')
    if not isinstance(values, list) or not values:
        raise ValidationBlock(
            f"`logits` must be a non-empty list of numbers, got "
            f"{type(values).__name__}")
    if len(values) != classes:
        raise ValidationBlock(
            f"`logits` has {len(values)} values but the calibrator was "
            f"fitted for {classes} classes")
    out: list[float] = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(float(v)):
            raise ValidationBlock(
                f"logits[{i}] must be a finite number, got {v!r}")
        out.append(float(v))
    if "probs" in doc and "logits" not in doc:
        # probability input: temperature scaling of log p is p^(1/T),
        # renormalized — express as logits so the fitted T applies.
        out = [math.log(max(p, 1e-300)) for p in out]
    return out


def _rows_from_records(records: Sequence[Any],
                       classes: int) -> list[dict[str, Any]]:
    """Prepared `calibration` records → validated {logits, label} rows."""
    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValidationBlock(
                f"record {i}: expected a prediction object, got "
                f"{type(rec).__name__}")
        logits = rec.get("logits")
        if not isinstance(logits, list) or not logits:
            continue
        if len(logits) != classes:
            raise ValidationBlock(
                f"record {i}: {len(logits)} logits but the calibrator "
                f"was fitted for {classes} classes — one eval pool is "
                "one label space")
        label = rec.get("label")
        if isinstance(label, bool) or not isinstance(label, (int, float)) \
                or int(label) != label or not 0 <= int(label) < classes:
            raise ValidationBlock(
                f"record {i}: label {label!r} is not a valid class index "
                f"for {classes} classes")
        clean: list[float] = []
        for j, v in enumerate(logits):
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(float(v)):
                raise ValidationBlock(
                    f"record {i}: logits[{j}] must be a finite number")
            clean.append(float(v))
        rows.append({"logits": clean, "label": int(label)})
    if not rows:
        raise ValidationBlock(
            "prepared dataset contains no prediction rows "
            "({logits, label}) — prepare with transform `calibration`")
    return rows


def contract(
    entry: Mapping[str, Any],
    fmt: str,
    *,
    semantic: Mapping[str, Any] | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """First-export `model_spec.json` for the calibrator (12 §15.3)."""
    from mlforge.ops.exporting import FORMATS

    ops = list(OPERATORS)
    return {
        "schema_version": 1,
        "name": entry.get("name"),
        "version": entry.get("version"),
        "harness": FAMILY,
        "inference_contract": {
            "input_schema": [
                {
                    "name": "prediction",
                    "type": "structured",
                    "schema": "json_schema://mlforge.calibrator.prediction.v1",
                    "fields": [
                        {"name": "logits", "dtype": "float[]"},
                        {"name": "probs", "dtype": "float[]"},
                    ],
                }
            ],
            "output_schema": [
                {
                    "name": "calibrated",
                    "type": "structured",
                    "schema": "json_schema://mlforge.calibrated.v1",
                }
            ],
            "operators": ops,
            "operator_set": content_hash({"ops": sorted(ops)}),
            "runtime": {
                "framework": fmt,
                "opset": FORMATS[fmt]["opset"] if fmt in FORMATS else 17,
            },
            "dtype": precision,
            "dynamic_axes": {"batch": [0]},
            "minimal_memory": {"vram_mb": 32},
            "numerical_tolerance": {"atol": ATOL},
        },
    }


def supports_format(fmt: str) -> str | None:
    if fmt == "onnx":
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except Exception as exc:
            return (f"onnx export needs the onnx + onnxruntime packages "
                    f"({type(exc).__name__}: {exc}) — pip install onnx "
                    "onnxruntime")
        return None
    return (f"the {fmt} converter is not integrated for the calibrator "
            "(13 §4.1 names ONNX as the portable target for exported "
            "models) — use --format onnx")


def execute(
    *,
    entry: Mapping[str, Any],
    contract_doc: Mapping[str, Any],
    checked: Mapping[str, Any],
    weights: bytes,
) -> dict[str, Any]:
    """Real one-shot calibration: temperature-scaled probabilities +
    conformal prediction sets from the fitted thresholds."""
    from mlforge.trainers.calibrator import _probs_at

    state = _state(weights, entry)
    t = state["temperature"]
    doc = _load_doc(checked)
    logits = _probs_from_doc(doc, state["classes"])
    probs = _probs_at([logits], t)[0]
    predicted = max(range(len(probs)), key=lambda k: probs[k])
    sets: dict[str, list[int]] = {}
    for label, threshold in _alphas(state):
        sets[label] = [c for c in range(len(probs))
                       if 1.0 - probs[c] <= threshold]
    return {
        "probs": [round(p, 6) for p in probs],
        "predicted_class": int(predicted),
        "confidence": round(float(probs[predicted]), 6),
        "temperature": t,
        "prediction_sets": sets,
    }


def _load_doc(checked: Mapping[str, Any]) -> Any:
    try:
        return json.loads(Path(str(checked["input"]["path"])).read_text(
            encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationBlock(
            f"structured input is not valid JSON: {exc}",
            hint='{"logits": [number, ...]} (13 §6.8 — schema invalid ⇒ '
                 "BLOCK before execution)") from exc


def compute_metrics(
    *,
    entry: Mapping[str, Any],
    weights: bytes,
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    semantic: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Real ECE / NLL / MCE / Brier + per-alpha coverage and average
    prediction-set size over prepared `calibration` records — the
    trainer's own formulas at the fitted temperature (13 §6.7,
    06:317-353). Returns EXACTLY `metric_names(weights)` (12 §15.4)."""
    from mlforge.trainers.calibrator import (
        _brier,
        _ece,
        _mce,
        _nll,
        _probs_at,
    )

    transform = dataset.get("transform")
    if transform and str(transform) != "calibration":
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} was prepared with transform "
            f"{transform!r} — the calibrator measures calibration quality "
            "over `calibration` prediction rows (logits + label)",
            hint="mlforge prepare <MODEL> with the calibration "
                 "transform for the eval split (12 §7)",
        )
    records = dataset.get("records")
    if not records:
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} has no prepared records — "
            "the calibrator needs `calibration` prediction rows",
            hint="mlforge prepare <MODEL> registers the eval split as a "
                 "store artifact first (12 §7: nothing runs on "
                 "unresolved data)")
    state = _state(weights, entry)
    t = state["temperature"]
    rows = _rows_from_records(records, state["classes"])
    subset = protocol.get("sample_subset")
    if subset is not None:
        if isinstance(subset, bool) or not isinstance(subset, int) \
                or subset < 1:
            raise ValidationBlock(
                f"protocol.sample_subset must be a positive integer "
                f"(first N rows), got {subset!r}")
        rows = rows[:subset]

    probs = _probs_at([r["logits"] for r in rows], t)
    alphas = _alphas(state)  # (label, threshold), sorted by alpha
    keys = _alpha_keys([label for label, _ in alphas])
    out: dict[str, float] = {
        "ece": round(_ece(rows, t), 6),
        "nll": round(_nll(rows, t), 6),
        "mce": round(_mce(rows, t), 6),
        "brier": round(_brier(rows, t), 6),
    }
    for (_label, threshold), key in zip(alphas, keys):
        # empirical coverage of the fitted threshold at T: the TRUE
        # class must fall inside the prediction set (06:351-352)
        in_set = sum(1 for i, r in enumerate(rows)
                     if 1.0 - probs[i][r["label"]] <= threshold)
        out[f"coverage_alpha_{key}"] = round(in_set / len(rows), 6)
    for (_label, threshold), key in zip(alphas, keys):
        # average prediction-set size at this alpha (06:353, target
        # 1-3 at alpha=0.05): every class whose 1 - p clears the
        # threshold is in the set; empty sets count 0, honestly.
        # Emitted coverage-first, then set-size, so the key ORDER
        # equals `metric_names(weights)` exactly (12 §15.4).
        sizes = [
            sum(1 for c in range(len(probs[i]))
                if 1.0 - probs[i][c] <= threshold)
            for i in range(len(rows))
        ]
        out[f"avg_set_size_alpha_{key}"] = round(
            sum(sizes) / len(sizes), 6)
    return out


# -- export: logits → ONNX (Div by T → Softmax) ------------------------------


def _build_onnx(state: Mapping[str, Any], entry: Mapping[str, Any]):
    """The calibration math as a real ONNX graph — checked before bytes."""
    import onnx
    from onnx import TensorProto, helper

    classes = int(state["classes"])
    t = float(state["temperature"])
    temp = helper.make_tensor("temperature", TensorProto.FLOAT, [], [t])
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Div", ["logits", "temperature"], ["scaled"],
                             name="temperature_scale"),
            helper.make_node("Softmax", ["scaled"], ["probs"],
                             name="calibrated_probs", axis=-1),
        ],
        name=f"{entry.get('name')}_{entry.get('version')}",
        inputs=[helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT, ["batch", classes])],
        outputs=[helper.make_tensor_value_info(
            "probs", TensorProto.FLOAT, ["batch", classes])],
        initializer=[temp],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17),
                       helper.make_opsetid("ai.onnx.ml", 1)],
    )
    model.ir_version = 8
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ValidationBlock(
            f"calibrator export: generated ONNX graph failed the checker "
            f"({exc}) — refusing to emit a graph we cannot verify (13 "
            "§7: never a partial export)") from exc
    return model


def _probe_logits(classes: int) -> list[list[float]]:
    """Deterministic probe inputs (basis vectors + fixed patterns)."""
    rows = [[0.0] * classes, [2.0] * classes]
    for c in range(classes):
        basis = [0.0] * classes
        basis[c] = 4.0
        rows.append(basis)
    for k in range(4):
        rows.append([(k + 1) * (c + 1) / (classes + 2) - 1.0
                     for c in range(classes)])
    return rows


def _roundtrip(state: Mapping[str, Any], proto: bytes) -> dict[str, Any]:
    """python `_probs_at` vs onnxruntime on the probe logits (13 §6.9)."""
    import numpy as np
    import onnxruntime as ort

    from mlforge.trainers.calibrator import _probs_at

    t = float(state["temperature"])
    probe = _probe_logits(int(state["classes"]))
    expected = _probs_at(probe, t)
    session = ort.InferenceSession(proto, providers=["CPUExecutionProvider"])
    got = session.run(None, {"logits": np.asarray(
        probe, dtype=np.float32)})[0]
    flat = np.asarray(got, dtype=np.float64)
    max_error = max(
        abs(float(expected[i][j]) - float(flat[i][j]))
        for i in range(len(probe)) for j in range(len(probe[0]))
    )
    return {
        "max_error": round(max_error, 10),
        "tolerance": ATOL,
        "result": "PASS" if max_error < ATOL else "FAIL",
        "harness": "python-onnx-roundtrip",
        "n_probe": len(probe),
    }


def export_bytes(
    *,
    entry: Mapping[str, Any],
    weights: bytes,
    fmt: str,
) -> tuple[bytes, dict[str, Any]]:
    """Real export: calibration graph → ONNX bytes + round-trip check."""
    if fmt != "onnx":
        raise PreconditionFailed(
            f"cannot export the calibrator to {fmt} — {supports_format(fmt)}",
            hint="13 §6.9: only formats with a real converter produce "
                 "bytes; nothing partial is ever written (13 §7)",
        )
    err = supports_format(fmt)
    if err:
        raise PreconditionFailed(f"cannot export the calibrator — {err}")
    state = _state(weights, entry)
    model = _build_onnx(state, entry)
    proto = model.SerializeToString()
    numerical = _roundtrip(state, proto)
    return proto, numerical


__all__ = [
    "ATOL",
    "FAMILY",
    "MODEL_NAMES",
    "OPERATORS",
    "compute_metrics",
    "contract",
    "dependency_error",
    "execute",
    "export_bytes",
    "input_types",
    "metric_names",
    "supports_format",
]
