"""Hypothesis ranker engine — real LambdaMART inference, NDCG@k metrics,
and a genuine tree→ONNX export (03_training_pipeline.md Model 4;
06_benchmarking_plan.md NDCG targets).

Normative: 10_training_plan/03_training_pipeline.md Model 4 (LightGBM
lambdarank, eval_at {3,5,10}, FEATURE_NAMES, num_leaves 31, the 10
evidence-graph features), 06_benchmarking_plan.md (NDCG@3 ≥ 0.85 /
NDCG@5 ≥ 0.80), 13_product_specification §6.7–§6.9, 12 §15.3 (contract
declares how the model is called), §12.4 (real harness — no scaffold).

Family contract (this module IS the harness; `lightgbm`/`onnx` are
imported only inside functions — the module is import-safe):
  * execute    — one-shot structured JSON `{"hypotheses": [{feature:
                 number, ...}, ...]}` (or one flat feature object) →
                 per-hypothesis scores from the fitted booster;
  * metrics    — `ndcg@3` + `ndcg@5`: group-mean NDCG with exponential
                 gain, computed with the TRAINER's own `_mean_ndcg` so
                 train-time and eval-time definitions cannot drift;
  * export     — the booster's trees are converted to a real ONNX
                 `TreeEnsembleRegressor` graph; numerical validation
                 compares onnxruntime against lightgbm on deterministic
                 probe vectors (a FAIL blocks the export, 13 §6.9).

Inputs are always finite (validated) — the ONNX missing-value branches
are recorded from LightGBM's `default_left` but never exercised here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash

FAMILY = "ranker"

#: Models this engine runs (03 Model 4).
MODEL_NAMES = frozenset({"hypothesis_ranker"})

#: Numerical round-trip tolerance for the ONNX export (13 §6.9 —
#: float32 thresholds/features vs LightGBM's float64 arithmetic).
ATOL = 1e-5

#: Contract operator list — the ONE op the converted graph needs.
OPERATORS: tuple[str, ...] = ("TreeEnsembleRegressor",)


def dependency_error() -> str | None:
    """None = runnable here (lightgbm present; onnx is checked at export
    time so inference/metrics work on a lightgbm-only machine)."""
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:
        return f"pip install lightgbm ({type(exc).__name__}: {exc})"
    return None


def input_types() -> frozenset[str]:
    return frozenset({"structured"})


def metric_names(weights: bytes | None = None) -> tuple[str, ...]:
    """Fixed metric set — `weights` carries no metric information here
    (engine contract: the calibrator derives per-model keys from it)."""
    return ("ndcg@3", "ndcg@5")


def _feature_names(semantic: Mapping[str, Any] | None) -> list[str]:
    raw = (semantic or {}).get("feature_columns")
    if raw is None:
        from mlforge.trainers.gbdt import FEATURE_NAMES

        return list(FEATURE_NAMES)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValidationBlock(
            "semantic.feature_columns must be a non-empty list of "
            f"column names, got {raw!r}"
        )
    return [str(c) for c in raw]


def _columns(semantic: Mapping[str, Any] | None) -> tuple[str, str]:
    sem = semantic or {}
    label_col = str(sem.get("label_column", "label")).strip()
    group_col = str(sem.get("group_column", "group")).strip()
    if not label_col or not group_col or label_col == group_col:
        raise ValidationBlock(
            f"semantic label/group columns are invalid "
            f"(label={label_col!r}, group={group_col!r}) — they must be "
            "non-empty and distinct"
        )
    return label_col, group_col


def _booster(weights: bytes, entry: Mapping[str, Any]):
    """Weights bytes → fitted LightGBM Booster (fail-closed — a blob that
    is not this family's model format is refused, never re-initialized)."""
    import lightgbm as lgb

    name = f"{entry.get('name')}:{entry.get('version')}"
    try:
        text = weights.decode("utf-8")
        if not text.strip():
            raise ValueError("weights blob is empty")
        booster = lgb.Booster(model_str=text)
    except Exception as exc:
        raise ValidationBlock(
            f"model {name}: checkpoint model blob is not a LightGBM "
            f"booster ({exc})",
            hint="the ranker's `model` component is LightGBM "
                 "model_to_string() text (03 Model 4) — this checkpoint "
                 "was produced by a different trainer",
        ) from exc
    if booster.current_iteration() <= 0:
        raise ValidationBlock(
            f"model {name}: booster has no trees — nothing to run"
        )
    return booster


def _contract_features(contract_doc: Mapping[str, Any]) -> list[str]:
    schema = (contract_doc.get("inference_contract") or {}).get(
        "input_schema") or []
    for entry in schema:
        fields = entry.get("fields")
        if isinstance(fields, list) and fields:
            out = [str(f.get("name")) for f in fields
                   if isinstance(f, dict) and f.get("name")]
            if out:
                return out
    raise ValidationBlock(
        "ranker contract declares no feature `fields` — cannot map the "
        "one-shot input onto the booster",
        hint="re-export the model so the contract records its features "
             "(12 §15.3)",
    )


def _parse_structured(path: Path) -> Any:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationBlock(
            f"structured input is not valid JSON: {exc}",
            hint='ranker input: {"hypotheses": [{feature: number, ...}, '
                 "...]} (13 §6.8 — schema invalid ⇒ BLOCK before "
                 "execution)",
        ) from exc
    return doc


def _hypotheses(doc: Any) -> list[dict[str, Any]]:
    """One-shot JSON → the hypothesis rows to score (fail-closed)."""
    if isinstance(doc, dict) and "hypotheses" in doc:
        rows = doc["hypotheses"]
        if not isinstance(rows, list) or not rows:
            raise ValidationBlock(
                "`hypotheses` must be a non-empty list of feature objects"
            )
        if any(not isinstance(r, dict) for r in rows):
            raise ValidationBlock(
                "`hypotheses` entries must each be a feature object "
                "(JSON object)"
            )
        return [dict(r) for r in rows]
    if isinstance(doc, dict):
        return [dict(doc)]  # a single flat feature object
    raise ValidationBlock(
        "structured ranker input must be a JSON object — a flat feature "
        'object or {"hypotheses": [...]}',
        hint="13 §6.8 — the contract at model_spec.json is the authority",
    )


def _row_vector(row: Mapping[str, Any], fields: Sequence[str],
                where: str) -> list[float]:
    vec: list[float] = []
    for f in fields:
        if f not in row:
            raise ValidationBlock(
                f"{where}: missing feature {f!r} — the contract declares "
                f"exactly: {', '.join(fields)}"
            )
        v = row[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(float(v)):
            raise ValidationBlock(
                f"{where}: feature {f!r} must be a finite number, "
                f"got {v!r}"
            )
        vec.append(float(v))
    return vec


def contract(
    entry: Mapping[str, Any],
    fmt: str,
    *,
    semantic: Mapping[str, Any] | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """First-export `model_spec.json` for the ranker (12 §15.3) — real
    feature schema, real operator list, real runtime."""
    from mlforge.ops.exporting import FORMATS

    features = _feature_names(semantic)
    ops = list(OPERATORS)
    return {
        "schema_version": 1,
        "name": entry.get("name"),
        "version": entry.get("version"),
        "harness": FAMILY,
        "inference_contract": {
            "input_schema": [
                {
                    "name": "hypotheses",
                    "type": "structured",
                    "schema": "json_schema://mlforge.ranker.hypotheses.v1",
                    "fields": [
                        {"name": f, "dtype": "float"} for f in features
                    ],
                }
            ],
            "output_schema": [
                {
                    "name": "scores",
                    "type": "structured",
                    "schema": "json_schema://mlforge.ranker.scores.v1",
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
            "minimal_memory": {"vram_mb": 128},
            "numerical_tolerance": {"atol": ATOL},
        },
    }


def supports_format(fmt: str) -> str | None:
    """None = this format can be produced HERE (13 §6.9 gate)."""
    if fmt == "onnx":
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except Exception as exc:
            return (f"onnx export needs the onnx + onnxruntime packages "
                    f"({type(exc).__name__}: {exc}) — pip install onnx "
                    "onnxruntime")
        return None
    return (f"the {fmt} converter is not integrated for the ranker "
            "(13 §4.1 names ONNX as the portable target for exported "
            "models) — use --format onnx")


def execute(
    *,
    entry: Mapping[str, Any],
    contract_doc: Mapping[str, Any],
    checked: Mapping[str, Any],
    weights: bytes,
) -> dict[str, Any]:
    """Real one-shot scoring — the booster runs on the checked input."""
    booster = _booster(weights, entry)
    fields = _contract_features(contract_doc)
    doc = _parse_structured(Path(str(checked["input"]["path"])))
    rows = _hypotheses(doc)
    vectors = [_row_vector(row, fields, f"hypothesis {i}")
               for i, row in enumerate(rows)]
    import numpy as np

    raw = booster.predict(np.asarray(vectors, dtype=np.float64))
    scores = [round(float(s), 6) for s in np.asarray(raw).reshape(-1)]
    return {
        "scores": scores,
        "n": len(scores),
        "features": list(fields),
        "model_iterations": int(booster.current_iteration()),
    }


def compute_metrics(
    *,
    entry: Mapping[str, Any],
    weights: bytes,
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    semantic: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Real NDCG@3/@5 over prepared `tabular` records (13 §6.7: the
    harness runs over the dataset — same grouping and the same NDCG
    definition the trainer optimizes).

    Row rules mirror `load_ranker_rows`: columns live in the record's
    `data` dict (ingested row shape); the protocol's `split: val`
    decides which rows count — `valid`/`val`/`validation` rows and rows
    without a split column (a dedicated eval artifact), never train or
    holdout rows (12 §15.4: `split` is never "train").
    """
    from mlforge.trainers.gbdt import _EXCLUDED_SPLITS, _VAL_SPLITS
    from mlforge.trainers.gbdt import _mean_ndcg  # trainer's own metric

    transform = dataset.get("transform")
    if transform and str(transform) != "tabular":
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} was prepared with transform "
            f"{transform!r} — the ranker measures NDCG over `tabular` "
            "rows (label + group + feature columns)",
            hint="mlforge prepare <MODEL> with the tabular transform "
                 "for the eval split (12 §7)",
        )
    records = dataset.get("records")
    if not records:
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} has no prepared records — "
            "the ranker measures NDCG over prepared `tabular` rows "
            "(label + group + feature columns)",
            hint="mlforge prepare <MODEL> registers the eval split as a "
                 "store artifact first (12 §7: nothing runs on "
                 "unresolved data)",
        )
    label_col, group_col = _columns(semantic)
    features = _feature_names(semantic)

    order: list[str] = []
    buckets: dict[str, list[tuple[int, list[float]]]] = {}
    split_counts: dict[str, int] = {}
    used = 0
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValidationBlock(
                f"record {i}: expected a JSON object row, got "
                f"{type(rec).__name__}")
        data = rec.get("data")
        if not isinstance(data, dict) or not data:
            data = rec  # flat row (no ingested-file wrapper)
        split = str(rec.get("split") or "").strip().lower()
        if split:
            split_counts[split] = split_counts.get(split, 0) + 1
            if split in _EXCLUDED_SPLITS or split not in _VAL_SPLITS:
                continue  # train/holdout rows are never eval rows
        used += 1
        graw = str(data.get(group_col, "")).strip()
        if not graw:
            raise ValidationBlock(
                f"record {i}: group column {group_col!r} is empty — "
                "ranking rows belong to a hypothesis group (query)",
                hint=f"prepare the eval split with transform `tabular` "
                     f"and a non-empty {group_col!r} column",
            )
        if graw not in buckets:
            buckets[graw] = []
            order.append(graw)
        lraw = data.get(label_col)
        if isinstance(lraw, bool) or not isinstance(lraw, (int, float)) \
                or int(lraw) != lraw or int(lraw) < 0:
            # tabular rows carry strings — accept numeric text too
            try:
                fv = float(str(lraw).strip())
            except (TypeError, ValueError):
                fv = float("nan")
            if math.isnan(fv) or int(fv) != fv or fv < 0:
                raise ValidationBlock(
                    f"record {i}: label column {label_col!r} value "
                    f"{lraw!r} — LambdaMART labels are integers >= 0")
            lraw = fv
        vec: list[float] = []
        for f in features:
            if f not in data:
                raise ValidationBlock(
                    f"record {i}: missing feature column {f!r} — the "
                    "eval split must share the training feature schema")
            v = data[f]
            if isinstance(v, bool):
                raise ValidationBlock(
                    f"record {i}: feature {f!r} must be a finite number")
            try:
                fv = float(str(v).strip())
            except (TypeError, ValueError):
                raise ValidationBlock(
                    f"record {i}: feature {f!r} value {v!r} is not a "
                    "finite number") from None
            if not math.isfinite(fv):
                raise ValidationBlock(
                    f"record {i}: feature {f!r} value {v!r} is not finite")
            vec.append(fv)
        buckets[graw].append((int(lraw), vec))
    if used == 0:
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} has no eval-split rows — "
            f"recorded splits: "
            + (", ".join(f"{k}×{v}" for k, v in sorted(split_counts.items()))
               or "(none)"),
            hint="the protocol evaluates split `val` (12 §15.4) — "
                 "prepare the eval split (split: valid) as its own "
                 "artifact; train/holdout rows are never eval rows",
        )

    subset = protocol.get("sample_subset")
    if subset is not None:
        if isinstance(subset, bool) or not isinstance(subset, int) \
                or subset < 1:
            raise ValidationBlock(
                f"protocol.sample_subset must be a positive integer "
                f"(first N groups), got {subset!r}")
        order = order[:subset]
    if not order:
        raise ValidationBlock(
            "evaluation dataset yields no hypothesis groups")

    booster = _booster(weights, entry)
    ys: list[int] = []
    scores: list[float] = []
    groups: list[int] = []
    vectors: list[list[float]] = []
    for g in order:
        for label, vec in buckets[g]:
            ys.append(label)
            vectors.append(vec)
        groups.append(len(buckets[g]))
    import numpy as np

    raw = booster.predict(np.asarray(vectors, dtype=np.float64))
    scores = [float(s) for s in np.asarray(raw).reshape(-1)]
    return {
        "ndcg@3": round(_mean_ndcg(ys, scores, groups, 3), 6),
        "ndcg@5": round(_mean_ndcg(ys, scores, groups, 5), 6),
    }


# -- export: booster trees → ONNX TreeEnsembleRegressor ----------------------


def _build_onnx(booster, entry: Mapping[str, Any]):
    """The booster's trees → a checked ONNX proto (real bytes source)."""
    import onnx
    from onnx import TensorProto, helper

    dump = booster.dump_model()
    tree_info = dump.get("tree_info") or []
    if not tree_info:
        raise ValidationBlock(
            "ranker export: booster dump contains no trees")
    nodes: dict[str, list[Any]] = {
        "tree": [], "node": [], "feature": [], "value": [],
        "mode": [], "true": [], "false": [], "miss": [],
    }
    leaves: list[tuple[int, int, float]] = []

    def walk(struct: Mapping[str, Any], tree_id: int,
             counter: list[int]) -> int:
        """One dump node → its attribute row(s); returns this node's
        PER-TREE id. Child pointers must be node ids (the counter's
        numbering), never global array positions: ids restart per tree
        while the attribute arrays are global — ORT resolves pointers as
        (tree, node) pairs ("Unable to find node 1-6" otherwise)."""
        nid = counter[0]
        counter[0] += 1
        if "leaf_value" in struct:
            nodes["tree"].append(tree_id)
            nodes["node"].append(nid)
            nodes["feature"].append(0)
            nodes["value"].append(0.0)
            nodes["mode"].append("LEAF")
            nodes["true"].append(0)
            nodes["false"].append(0)
            nodes["miss"].append(0)
            leaves.append((tree_id, nid, float(struct["leaf_value"])))
            return nid
        decision = str(struct.get("decision_type", "<="))
        if "categorical" in decision:
            raise ValidationBlock(
                "ranker export: categorical splits are not "
                "convertible to ONNX TreeEnsembleRegressor — 03 "
                "Model 4's feature schema is numeric-only",
            )
        pos = len(nodes["tree"])
        nodes["tree"].append(tree_id)
        nodes["node"].append(nid)
        nodes["feature"].append(int(struct["split_feature"]))
        nodes["value"].append(float(struct["threshold"]))
        nodes["mode"].append("BRANCH_LEQ")
        nodes["true"].append(-1)
        nodes["false"].append(-1)
        nodes["miss"].append(1 if struct.get("default_left") else 0)
        # pre-order: this row is already appended, children extend the
        # arrays (never shifting `pos`) — then wire their returned ids.
        nodes["true"][pos] = walk(struct["left_child"], tree_id, counter)
        nodes["false"][pos] = walk(struct["right_child"], tree_id, counter)
        return nid

    for ti, tree in enumerate(tree_info):
        struct = tree.get("tree_structure")
        if not isinstance(struct, Mapping):
            raise ValidationBlock(
                "ranker export: tree_structure missing from dump")
        walk(struct, ti, [0])

    n_features = int(booster.num_feature())
    graph = helper.make_graph(
        nodes=[helper.make_node(
            "TreeEnsembleRegressor",
            inputs=["features"],
            outputs=["scores"],
            name="mlforge_ranker",
            # ai.onnx.ml@1 — the checker resolves the node type against
            # the node's OWN domain, not the default ai.onnx domain.
            domain="ai.onnx.ml",
            aggregate_function="SUM",
            n_targets=1,
            post_transform="NONE",
            nodes_treeids=[int(v) for v in nodes["tree"]],
            nodes_nodeids=[int(v) for v in nodes["node"]],
            nodes_featureids=[int(v) for v in nodes["feature"]],
            nodes_values=[float(v) for v in nodes["value"]],
            nodes_modes=[str(v) for v in nodes["mode"]],
            nodes_truenodeids=[int(v) for v in nodes["true"]],
            nodes_falsenodeids=[int(v) for v in nodes["false"]],
            nodes_missing_value_tracks_true=[int(v) for v in nodes["miss"]],
            target_treeids=[int(t) for t, _n, _w in leaves],
            target_nodeids=[int(n) for _t, n, _w in leaves],
            target_weights=[float(w) for _t, _n, w in leaves],
            # Parallel target index — optional in the schema but
            # REQUIRED by onnxruntime 1.30's attribute reader (it checks
            # target_ids against target_nodeids): all leaves feed the
            # single output (n_targets=1) → index 0 everywhere.
            target_ids=[0] * len(leaves),
        )],
        name=f"{entry.get('name')}_{entry.get('version')}",
        inputs=[helper.make_tensor_value_info(
            "features", TensorProto.FLOAT, ["batch", n_features])],
        outputs=[helper.make_tensor_value_info(
            "scores", TensorProto.FLOAT, ["batch", 1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17),
                       helper.make_opsetid("ai.onnx.ml", 1)],
    )
    model.ir_version = 8  # portable IR level (onnxruntime-friendly)
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ValidationBlock(
            f"ranker export: generated ONNX graph failed the checker "
            f"({exc}) — refusing to emit a graph we cannot verify (13 "
            "§7: never a partial export)") from exc
    return model


def _probe_vectors(n_features: int) -> list[list[float]]:
    """Deterministic probe inputs for the numerical round-trip — no
    randomness (same bytes ⇒ same validation, 12 §15.4 spirit)."""
    vecs = [[0.0] * n_features, [1.0] * n_features]
    for k in range(6):
        vecs.append([((k + 1) * (j + 3) % 11) / 11.0
                     for j in range(n_features)])
    return vecs


def _roundtrip(booster, proto: bytes) -> dict[str, Any]:
    """lightgbm vs onnxruntime on the probe vectors (13 §6.9: max error
    vs tolerance → PASS/FAIL — a FAIL blocks the export)."""
    import numpy as np
    import onnxruntime as ort

    n_features = int(booster.num_feature())
    vectors = _probe_vectors(n_features)
    expected = [float(v) for v in np.asarray(
        booster.predict(np.asarray(vectors, dtype=np.float64))).reshape(-1)]
    session = ort.InferenceSession(proto, providers=["CPUExecutionProvider"])
    got = session.run(None, {"features": np.asarray(
        vectors, dtype=np.float32)})[0]
    flat = [float(v) for v in np.asarray(got).reshape(-1)]
    max_error = max((abs(e - g) for e, g in zip(expected, flat)),
                    default=0.0)
    return {
        "max_error": round(max_error, 10),
        "tolerance": ATOL,
        "result": "PASS" if max_error < ATOL else "FAIL",
        "harness": "lightgbm-onnx-roundtrip",
        "n_probe": len(vectors),
    }


def export_bytes(
    *,
    entry: Mapping[str, Any],
    weights: bytes,
    fmt: str,
) -> tuple[bytes, dict[str, Any]]:
    """Real export: booster trees → ONNX bytes + round-trip validation.
    Returns (binary, numerical_validation); the workflow blocks on FAIL."""
    if fmt != "onnx":
        raise PreconditionFailed(
            f"cannot export the ranker to {fmt} — {supports_format(fmt)}",
            hint="13 §6.9: only formats with a real converter produce "
                 "bytes; nothing partial is ever written (13 §7)",
        )
    err = supports_format(fmt)
    if err:
        raise PreconditionFailed(f"cannot export the ranker — {err}")
    booster = _booster(weights, entry)
    model = _build_onnx(booster, entry)
    proto = model.SerializeToString()
    numerical = _roundtrip(booster, proto)
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
