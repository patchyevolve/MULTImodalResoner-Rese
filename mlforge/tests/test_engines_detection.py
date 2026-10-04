"""Tier-3 real DETECTION engine — evaluate / infer / export with NO
harness (12 §12.4): tiny REAL RF-DETR checkpoints (RFDETRSmall at
resolution 64, RFDETRSegSmall at 96 — seeded random init, no download,
crafted in-test with the trainer's own checkpoint payload shape) drive
every seam:

  * family surface (FAMILY / MODEL_NAMES / input_types / metric_names)
    against the trainer's registry, honest format refusals naming ONNX
    as the portable target (13 §4.1), and the engine-registry wiring
    (`build_engine("rf_detr_*")` resolves this module);
  * contract() -> model_spec: real image schema (family-default resize,
    ImageNet normalization), execute output schema, the measured
    44-operator list `validate_export` accepts at opset 17 (12 §15.3,
    13 §6.9); family resolutions 512/704/384; segmentation adds masks;
  * infer -> real detections (sigmoid top-k, NMS-free, score gate 0.5):
    deterministic, JSON-ready, contract drift / unchecked input /
    undecodable pixels / foreign-family weights all BLOCK (13 §6.8);
  * evaluate -> real COCO `COCOeval` mAP/AP50: perfect boxes score 1.0
    through the `_detect_all` seam, the real model runs over prepared
    `coco_detection` records, unmapped labels are dropped (rfdetr's
    `_resolve_category_id` rule), and the split/identity/subset guards
    fire before any weights load (13 §6.7, 12 §15.4);
  * export -> genuine ONNX bytes (ir_version 8), PASS round-trip at
    ATOL = 1e-3 (measured drift ~1e-6 / ~4e-5), graph ops ⊆ contract
    operators, dynamic batch beyond the traced batch, FAIL/under-declared
    graphs blocked, non-onnx refused (13 §6.9).

The harness gate is lifted for every test (conftest opts IN; these opt
OUT — a seam regressing to the scaffold breaks them). Deterministic
(seeded init + generated images), CPU-only, engine functions called
DIRECTLY (no registry/workflow edits).
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.ops.engines import detection, harness_active

#: Handcrafted registry entry — the shape the workflow hands the engine.
ENTRY: dict = {
    "name": "rf_detr_s",
    "version": "v1",
    "model_id": "mdl_det_unit",
    "run_id": None,
    "artifact_hash": "sha256:" + "0" * 64,
}

SEG_ENTRY: dict = {**ENTRY, "name": "rf_detr_seg_s"}


# -- fixtures / helpers ------------------------------------------------------


@pytest.fixture(autouse=True)
def real_path(monkeypatch):
    """These tests must never see the scaffold harness (12 §12.4)."""
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    assert harness_active() is False


def _build_checkpoint(cls_name: str, resolution: int) -> bytes:
    """Tiny REAL rfdetr checkpoint in the trainer's payload shape —
    `{"model": bare state_dict, "args", "model_name", "model_config"}`,
    seeded random init, no download. The resolution must be a multiple
    of the variant's block size (patch x windows): 16x2 = 32 for
    detection, 12x2 = 24 for segmentation."""
    import warnings

    import rfdetr
    import torch

    torch.manual_seed(7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det = getattr(rfdetr, cls_name)(
            pretrain_weights=None, resolution=resolution,
            num_classes=3, num_queries=10)
    nn = det.model.model
    nn.eval()
    ckpt = {
        "model": nn.state_dict(),
        "args": {"pretrain_weights": "none", "num_classes": 3},
        "model_name": cls_name,
        "model_config": det.model_config.model_dump(mode="json"),
    }
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    blob = buf.getvalue()
    assert len(blob) > 50_000_000  # full DINOv2-base backbone, not a stub
    return blob


@pytest.fixture(scope="module")
def weights() -> bytes:
    return _build_checkpoint("RFDETRSmall", 64)


@pytest.fixture(scope="module")
def weights_seg() -> bytes:
    return _build_checkpoint("RFDETRSegSmall", 96)


@pytest.fixture(scope="module")
def contract_doc() -> dict:
    return detection.contract(ENTRY, "onnx")


def _write_image(path: Path, *, seed: int = 0) -> Path:
    """Deterministic 61x47 RGB PNG (non-square: the resize path runs)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 61, 47
    im = Image.new("RGB", (w, h))
    px = im.load()
    for i in range(w):
        for j in range(h):
            px[i, j] = (
                (i * 7 + seed) % 256,
                (j * 11 + seed * 3) % 256,
                (i * j + seed * 5) % 256,
            )
    im.save(path, "PNG")
    return path


def _checked(path: Path, **over) -> dict:
    """check_input's SAFE envelope (hand-built — the happy-path test
    below uses the real `mlforge.ops.infer.check_input`)."""
    return {"status": "SAFE",
            "input": {"path": str(path), "bytes": 4096,
                      "extension": path.suffix,
                      "identity": "sha256:" + "f" * 64, **over}}


def _protocol(**over) -> dict:
    from mlforge.ops.evaluation import build_protocol

    return build_protocol(over or None,
                          metric_names=detection.metric_names())


def _ann_doc(n: int = 3) -> dict:
    """Tiny COCO doc — one 20x20 box per 61x47 image, single category
    (COCO self-parent convention: `supercategory` == `name`, so rfdetr's
    parent filter keeps it)."""
    return {
        "images": [{"id": i + 1, "file_name": f"im{i}.png",
                    "width": 61, "height": 47} for i in range(n)],
        "annotations": [{"id": i + 1, "image_id": i + 1, "category_id": 1,
                         "bbox": [5.0, 5.0, 20.0, 20.0], "area": 400.0,
                         "iscrowd": 0} for i in range(n)],
        "categories": [{"id": 1, "name": "widget",
                        "supercategory": "widget"}],
    }


def _prepared(tmp_path: Path, *, transform: str = "coco_detection",
              skip_train: bool = False, skip_eval_ann: bool = False,
              skip_eval_file: bool = False) -> dict:
    """Prepared `coco_detection` dataset in the roboflow layout
    (train/ + valid/ + sha256 records) — the shape
    `WorkflowAPI._resolve_eval_dataset` hands the engine."""
    from mlforge.hashing import content_hash_bytes

    def rec(rel: str) -> dict:
        return {
            "source": "widgets:coco",
            "split": rel.split("/", 1)[0],
            "relative_path": rel,
            "sha256": content_hash_bytes((base / rel).read_bytes()),
        }

    base = tmp_path / "widgets"
    (base / "train").mkdir(parents=True, exist_ok=True)
    (base / "valid").mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if not skip_train:
        (base / "train" / "_annotations.coco.json").write_bytes(
            json.dumps(_ann_doc()).encode("utf-8"))
        records.append(rec("train/_annotations.coco.json"))
    if not skip_eval_ann:
        (base / "valid" / "_annotations.coco.json").write_bytes(
            json.dumps(_ann_doc()).encode("utf-8"))
        records.append(rec("valid/_annotations.coco.json"))
    for i in range(3):
        _write_image(base / "valid" / f"im{i}.png", seed=i)
        records.append(rec(f"valid/im{i}.png"))
    if skip_eval_file:
        (base / "valid" / "_annotations.coco.json").unlink()
    return {"name": "widgets", "version": "v1", "ref": "widgets:v1",
            "identity": "sha256:" + "d" * 64, "path": str(base),
            "records": records, "transform": transform}


def _gt_box() -> dict:
    """One detection dict matching `_ann_doc` ground truth exactly
    (xyxy pixel box + label 0 -> category_id 1)."""
    return {"boxes": [[5.0, 5.0, 25.0, 25.0]], "scores": [0.99],
            "labels": [0], "count": 1, "threshold": None,
            "image": {"width": 61, "height": 47}, "resolution": 64}


class _StubCtx:
    """Stand-in checkpoint context for seam tests (no model load)."""

    resolution = 64

    def cleanup(self) -> None:  # pragma: no cover - trivial
        pass


# -- family surface ----------------------------------------------------------


def test_dependency_error_is_none():
    assert detection.FAMILY == "detection"
    assert detection.dependency_error() is None


def test_family_surface_matches_the_trainer_and_the_benchmark():
    from mlforge.trainers.rfdetr import MODEL_REGISTRY

    assert detection.input_types() == frozenset({"image"})
    # 06_benchmarking_plan.md §4: COCO mAP / AP50 from
    # COCOeval.stats[0:2] — the trainer's registry names the family.
    assert detection.metric_names() == ("mAP", "AP50")
    assert detection.MODEL_NAMES == frozenset(MODEL_REGISTRY)
    assert detection.MODEL_NAMES == {
        "rf_detr_s", "rf_detr_l", "rf_detr_seg_s"}
    assert detection.ATOL == 1e-3
    assert detection.SCORE_THRESHOLD == 0.5
    assert detection.FAMILY_RESOLUTION == {
        "rf_detr_s": 512, "rf_detr_l": 704, "rf_detr_seg_s": 384}


def test_detection_registered_in_engine_registry():
    """The engine registry (and hence the workflow) resolves every
    RF-DETR variant to this module — one registry, two lookups
    (`registered_models` + `build_engine`, mirroring the trainers)."""
    from mlforge.ops.engines import build_engine, registered_models

    models = registered_models()
    for name in ("rf_detr_s", "rf_detr_l", "rf_detr_seg_s"):
        assert name in models
        assert build_engine(name) is detection
    with pytest.raises(PreconditionFailed) as ei:
        build_engine("some_future_model")
    assert "no real engine" in str(ei.value)


def test_supports_format_honest_refusals():
    assert detection.supports_format("onnx") is None
    for fmt in ("tflite", "openvino"):
        msg = detection.supports_format(fmt)
        assert msg is not None
        assert fmt in msg and "onnx" in msg.lower()
        assert "4.1" in msg  # 13 §4.1 names ONNX as the portable target


# -- contract (12 §15.3) ------------------------------------------------------


def test_contract_declares_real_detection_contract(contract_doc):
    import onnx.defs

    from mlforge.hashing import content_hash
    from mlforge.ops.exporting import validate_export

    assert contract_doc["schema_version"] == 1
    assert contract_doc["name"] == "rf_detr_s"
    assert contract_doc["harness"] == "detection"
    ic = contract_doc["inference_contract"]

    schema = ic["input_schema"][0]
    assert schema["name"] == "image" and schema["type"] == "image"
    assert schema["dtype"] == "uint8" and schema["layout"] == "HWC"
    assert schema["color_space"] == "RGB"
    assert schema["resize"] == {"width": 512, "height": 512,
                                "mode": "bilinear", "antialias": False}
    assert schema["normalization"] == {"mean": detection.MEANS,
                                       "std": detection.STDS}

    out = ic["output_schema"][0]
    fields = {f["name"] for f in out["fields"]}
    assert {"boxes", "scores", "labels"} <= fields
    assert "masks" not in fields  # detection task, not segmentation
    assert out["parameters"]["score_threshold"] == 0.5

    assert ic["runtime"] == {"framework": "onnx", "opset": 17}
    assert ic["dtype"] == "fp32"
    assert ic["dynamic_axes"] == {"batch": [0]}
    assert ic["minimal_memory"] == {"vram_mb": 512}
    assert ic["numerical_tolerance"]["atol"] == detection.ATOL == 1e-3

    ops = ic["operators"]
    assert set(ops) == set(detection.OPERATORS)
    assert {"Conv", "Gemm", "MatMul", "Sigmoid", "TopK",
            "Resize"} <= set(ops)
    # every declared op exists in the installed ONNX spec (13 §6.9 —
    # the workflow checks declared ⊆ schemas before writing a binary)
    schemas = {s.name.lower().replace("_", "")
               for s in onnx.defs.get_all_schemas()}
    for op in ops:
        assert op.lower().replace("_", "") in schemas, op
    assert validate_export(contract_doc, "onnx") == {
        "architecture": "PASS", "operators": "PASS",
        "dynamic_shapes": "PASS"}
    assert ic["operator_set"] == content_hash({"ops": sorted(ops)})


def test_contract_family_resolutions_and_seg_masks():
    for name, res in (("rf_detr_s", 512), ("rf_detr_l", 704),
                      ("rf_detr_seg_s", 384)):
        doc = detection.contract({**ENTRY, "name": name}, "onnx")
        resize = doc["inference_contract"]["input_schema"][0]["resize"]
        assert (resize["width"], resize["height"]) == (res, res), name
    seg_fields = {f["name"] for f in detection.contract(
        SEG_ENTRY, "onnx")["inference_contract"]["output_schema"][0][
            "fields"]}
    assert "masks" in seg_fields
    assert {"boxes", "scores", "labels"} <= seg_fields


def test_contract_unknown_model_blocks():
    with pytest.raises(ValidationBlock) as ei:
        detection.contract({**ENTRY, "name": "resnet50"}, "onnx")
    out = ei.value.render()
    assert "not a detection-family entry" in out
    assert "rf_detr_s" in out  # the refusal names what it does run


def test_check_contract_blocks_drift(contract_doc):
    def _doc() -> dict:
        return json.loads(json.dumps(contract_doc))

    # no image input at all
    doc = _doc()
    doc["inference_contract"]["input_schema"] = [
        {"name": "text", "type": "text"}]
    with pytest.raises(ValidationBlock) as ei:
        detection._check_contract(doc, ENTRY)
    assert "no `image` input_schema entry" in ei.value.render()

    # resize minted for another model — contract and weights disagree
    doc = _doc()
    doc["inference_contract"]["input_schema"][0]["resize"] = {
        "width": 640, "height": 640, "mode": "bilinear"}
    with pytest.raises(ValidationBlock) as ei:
        detection._check_contract(doc, ENTRY)
    out = ei.value.render()
    assert "640x640" in out and "512x512" in out
    assert "re-export" in (ei.value.hint or "")

    # another harness's contract
    doc = _doc()
    doc["harness"] = "text"
    with pytest.raises(ValidationBlock) as ei:
        detection._check_contract(doc, ENTRY)
    assert "is not the detection family" in ei.value.render()

    # the real contract passes (the execute path calls this gate)
    detection._check_contract(contract_doc, ENTRY)


# -- execute (13 §6.8) --------------------------------------------------------


def test_execute_real_pipeline_is_deterministic(tmp_path, weights,
                                                contract_doc):
    from mlforge.ops.infer import check_input

    img = _write_image(tmp_path / "query.png")
    checked = check_input(contract_doc, str(img))  # real input check
    assert checked["status"] == "SAFE"
    out = detection.execute(entry=ENTRY, contract_doc=contract_doc,
                            checked=checked, weights=weights)
    assert set(out) == {"boxes", "scores", "labels", "count", "threshold",
                        "image", "resolution"}
    assert out["image"] == {"width": 61, "height": 47}
    assert out["resolution"] == 64  # the CHECKPOINT's size, not the 512
    # the contract's family default; every result records the real one
    assert out["threshold"] == 0.5
    assert 0 <= out["count"] <= 40  # 10 queries x 4 logit slots
    assert (len(out["boxes"]) == out["count"] == len(out["scores"])
            == len(out["labels"]))
    # labels are raw logit slots: nc=3 classes + the background slot
    assert all(0 <= lab <= 3 for lab in out["labels"])
    for x1, y1, x2, y2 in out["boxes"]:
        assert 0.0 <= x1 <= 61.0 and 0.0 <= x2 <= 61.0
        assert 0.0 <= y1 <= 47.0 and 0.0 <= y2 <= 47.0
    json.dumps(out)  # JSON-ready result envelope (13 §6.8)
    again = detection.execute(entry=ENTRY, contract_doc=contract_doc,
                              checked=checked, weights=weights)
    assert again == out  # same bytes + same input ⇒ same detections


def test_execute_runs_the_contract_check(tmp_path, weights, contract_doc):
    """execute() invokes the drift gate before touching the pixels."""
    doc = json.loads(json.dumps(contract_doc))
    doc["inference_contract"]["input_schema"] = [
        {"name": "structured", "type": "structured"}]
    img = _write_image(tmp_path / "q.png")
    with pytest.raises(ValidationBlock) as ei:
        detection.execute(entry=ENTRY, contract_doc=doc,
                          checked=_checked(img), weights=weights)
    assert "no `image` input_schema entry" in ei.value.render()


def test_detect_one_unthresholded_keeps_every_row(tmp_path, weights):
    """The evaluation path keeps every top-k row (rank-based AP): the
    score gate belongs to execute only."""
    img = _write_image(tmp_path / "q.png", seed=2)
    ctx = detection._load_context(weights, ENTRY)
    try:
        out = detection._detect_one(ctx, str(img), score_threshold=None)
    finally:
        ctx.cleanup()
    assert out["threshold"] is None
    assert out["count"] == 40  # min(num_select=300, 10 queries x 4 slots)
    assert len(out["scores"]) == 40
    assert out["scores"] == sorted(out["scores"], reverse=True)
    assert set(out["labels"]) <= {0, 1, 2, 3}
    assert "masks" not in out  # detection task


def test_execute_blocks_empty_or_foreign_weights(tmp_path, weights,
                                                 weights_seg,
                                                 contract_doc):
    img = _write_image(tmp_path / "q.png")

    with pytest.raises(ValidationBlock) as ei:
        detection.execute(entry=ENTRY, contract_doc=contract_doc,
                          checked=_checked(img), weights=b"")
    assert "empty" in ei.value.render()

    # segmentation checkpoint handed to the detection entry — the class
    # reconstructed from `model_name` must match the registered variant
    with pytest.raises(ValidationBlock) as ei:
        detection.execute(entry=ENTRY, contract_doc=contract_doc,
                          checked=_checked(img), weights=weights_seg)
    out = ei.value.render()
    assert "RFDETRSegSmall" in out and "RFDETRSmall" in out
    assert "different family" in out

    with pytest.raises(ValidationBlock) as ei:
        detection.execute(entry={**ENTRY, "name": "resnet50"},
                          contract_doc=contract_doc, checked=_checked(img),
                          weights=weights)
    assert "not a detection-family entry" in ei.value.render()


def test_execute_blocks_unchecked_or_missing_input(tmp_path, weights,
                                                   contract_doc):
    with pytest.raises(ValidationBlock) as ei:
        detection.execute(entry=ENTRY, contract_doc=contract_doc,
                          checked={"status": "SAFE", "input": {}},
                          weights=weights)
    out = ei.value.render()
    assert "carries no path" in out
    assert "6.8" in out

    with pytest.raises(ValidationBlock) as ei:
        detection.execute(entry=ENTRY, contract_doc=contract_doc,
                          checked=_checked(tmp_path / "gone.png"),
                          weights=weights)
    assert "input image not found" in ei.value.render()


def test_execute_blocks_undecodable_image(tmp_path, weights, contract_doc):
    """Content that passed the extension/size check but cannot be
    decoded is a content failure, never an empty prediction (13 §6.8)."""
    bad = tmp_path / "not-an-image.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xff" * 64)
    with pytest.raises(ValidationBlock) as ei:
        detection.execute(entry=ENTRY, contract_doc=contract_doc,
                          checked=_checked(bad), weights=weights)
    out = ei.value.render()
    assert "input image unreadable" in out
    assert "6.8" in out


# -- evaluate (13 §6.7, 12 §15.4) ---------------------------------------------


def test_compute_metrics_perfect_boxes_score_one(tmp_path, monkeypatch):
    """GT-matching detections through the real COCOeval path ⇒ AP 1.0
    (mirrors 06 §4's own COCO/COCOeval sequence)."""
    dataset = _prepared(tmp_path)
    seen: list[dict] = []

    def fake_detect_all(ctx, items):
        assert isinstance(ctx, _StubCtx)
        seen.extend(items)
        return [_gt_box() for _ in items]

    monkeypatch.setattr(detection, "_load_context",
                        lambda _w, _e: _StubCtx())
    monkeypatch.setattr(detection, "_detect_all", fake_detect_all)
    metrics = detection.compute_metrics(
        entry=ENTRY, weights=b"stub", dataset=dataset,
        protocol=_protocol(), semantic=None)
    assert metrics == {"mAP": 1.0, "AP50": 1.0}
    assert len(seen) == 3  # all three eval images ran


def test_compute_metrics_honors_sample_subset(tmp_path, monkeypatch):
    dataset = _prepared(tmp_path)
    seen: list[dict] = []

    def fake_detect_all(ctx, items):
        seen.extend(items)
        return [_gt_box() for _ in items]

    monkeypatch.setattr(detection, "_load_context",
                        lambda _w, _e: _StubCtx())
    monkeypatch.setattr(detection, "_detect_all", fake_detect_all)
    metrics = detection.compute_metrics(
        entry=ENTRY, weights=b"stub", dataset=dataset,
        protocol=_protocol(sample_subset=1), semantic=None)
    assert len(seen) == 1  # first N images, annotation order
    assert metrics == {"mAP": 1.0, "AP50": 1.0}


def test_compute_metrics_real_model_over_prepared_records(tmp_path,
                                                          weights):
    """The unseamed path: real weights decode the real image files and
    COCOeval runs over the prepared records (rank-based, ungated)."""
    dataset = _prepared(tmp_path)
    metrics = detection.compute_metrics(
        entry=ENTRY, weights=weights, dataset=dataset,
        protocol=_protocol(), semantic=None)
    assert set(metrics) == {"mAP", "AP50"}
    for key, value in metrics.items():
        assert math.isfinite(value) and 0.0 <= value <= 1.0, (key, value)


def test_coco_results_drops_unmapped_labels_and_clips():
    items = [{"image_id": 7, "width": 61, "height": 47}]
    dets = [{"boxes": [[-5.0, -5.0, 90.0, 100.0],   # clipped to image
                       [10.0, 10.0, 30.0, 30.0]],
             "scores": [0.9, 0.8],
             "labels": [0, 5]}]  # 5 is outside the train label space
    rows = detection._coco_results(items, dets, {0: 1}, {1})
    assert len(rows) == 1
    assert rows[0]["image_id"] == 7 and rows[0]["category_id"] == 1
    assert rows[0]["bbox"] == [0.0, 0.0, 61.0, 47.0]  # xywh after clip
    assert rows[0]["score"] == 0.9
    # a mapped label whose category is absent from the eval doc drops too
    assert detection._coco_results(items, dets, {0: 9}, {1}) == []
    # ragged detector output blocks — a misaligned run is never scored
    with pytest.raises(ValidationBlock) as ei:
        detection._coco_results(
            items, [{"boxes": [[0, 0, 1, 1]], "scores": [], "labels": []}],
            {0: 1}, {1})
    assert "ragged" in ei.value.render()


def test_run_cocoeval_empty_results_report_zero(tmp_path):
    dataset = _prepared(tmp_path)
    ann = Path(dataset["path"]) / "valid" / "_annotations.coco.json"
    assert detection._run_cocoeval(ann, []) == (0.0, 0.0)


def test_compute_metrics_wrong_transform_blocks(tmp_path):
    dataset = _prepared(tmp_path, transform="tabular")
    with pytest.raises(ValidationBlock) as ei:
        detection.compute_metrics(entry=ENTRY, weights=b"x",
                                  dataset=dataset, protocol=_protocol(),
                                  semantic=None)
    out = ei.value.render()
    assert "transform" in out and "tabular" in out
    assert "coco_detection" in out
    assert "mlforge prepare" in (ei.value.hint or "")


def test_compute_metrics_requires_records_and_path(tmp_path):
    empty = _prepared(tmp_path / "empty")
    empty["records"] = []
    with pytest.raises(ValidationBlock) as ei:
        detection.compute_metrics(entry=ENTRY, weights=b"x", dataset=empty,
                                  protocol=_protocol(), semantic=None)
    assert "no prepared records" in ei.value.render()

    no_path = _prepared(tmp_path / "nopath")
    no_path["path"] = None
    with pytest.raises(PreconditionFailed) as ei:
        detection.compute_metrics(entry=ENTRY, weights=b"x",
                                  dataset=no_path, protocol=_protocol(),
                                  semantic=None)
    out = ei.value.render()
    assert "no machine-local path" in out
    assert "dataset add" in (ei.value.hint or "")


def test_compute_metrics_refuses_train_and_unknown_splits(tmp_path):
    dataset = _prepared(tmp_path)
    with pytest.raises(ValidationBlock) as ei:
        detection.compute_metrics(entry=ENTRY, weights=b"x",
                                  dataset=dataset,
                                  protocol=_protocol(split="train"),
                                  semantic=None)
    out = ei.value.render()
    assert "never eval rows" in out  # 12 §15.4: train is never eval

    with pytest.raises(ValidationBlock) as ei:
        detection.compute_metrics(entry=ENTRY, weights=b"x",
                                  dataset=dataset,
                                  protocol=_protocol(split="holdout"),
                                  semantic=None)
    assert "provides train/valid/test" in ei.value.render()


def test_compute_metrics_annotation_identity_guards(tmp_path):
    # annotation drift: record sha no longer matches the file bytes
    drifted = _prepared(tmp_path / "drift")
    drifted["records"][1] = {**drifted["records"][1],
                             "sha256": "sha256:" + "b" * 64}
    with pytest.raises(ValidationBlock) as ei:
        detection.compute_metrics(entry=ENTRY, weights=b"x",
                                  dataset=drifted, protocol=_protocol(),
                                  semantic=None)
    assert "no longer matches its prepared sha256" in ei.value.render()

    # record present, file gone (content changed since registration)
    missing = _prepared(tmp_path / "missing", skip_eval_file=True)
    with pytest.raises(ValidationBlock) as ei:
        detection.compute_metrics(entry=ENTRY, weights=b"x",
                                  dataset=missing, protocol=_protocol(),
                                  semantic=None)
    assert "annotation missing at registered path" in ei.value.render()

    # eval split's own annotations are the measurement target
    no_eval = _prepared(tmp_path / "noeval", skip_eval_ann=True)
    with pytest.raises(ValidationBlock) as ei:
        detection.compute_metrics(entry=ENTRY, weights=b"x",
                                  dataset=no_eval, protocol=_protocol(),
                                  semantic=None)
    assert "dataset has no valid/_annotations.coco.json record" in (
        ei.value.render())

    # a valid-only tree has no layout the trainer can read (the train
    # annotation file defines the roboflow layout, rfdetr_data's rule)
    no_train = _prepared(tmp_path / "notrain", skip_train=True)
    with pytest.raises(ValidationBlock) as ei:
        detection.compute_metrics(entry=ENTRY, weights=b"x",
                                  dataset=no_train, protocol=_protocol(),
                                  semantic=None)
    assert "does not match a layout" in ei.value.render()


def test_compute_metrics_bad_subset_blocks(tmp_path):
    dataset = _prepared(tmp_path)
    with pytest.raises(ValidationBlock) as ei:
        detection.compute_metrics(entry=ENTRY, weights=b"x",
                                  dataset=dataset,
                                  protocol=_protocol(sample_subset=0),
                                  semantic=None)
    assert "sample_subset" in ei.value.render()


# -- export (13 §6.9) ---------------------------------------------------------


def test_export_bytes_real_onnx_and_roundtrip(weights, contract_doc):
    import numpy as np
    import onnx
    import onnxruntime as ort

    from mlforge.ops.exporting import validate_export

    proto, numerical = detection.export_bytes(
        entry=ENTRY, weights=weights, fmt="onnx")
    # genuine bytes: portable IR8 header, not a placeholder
    assert isinstance(proto, bytes) and proto[:2] == b"\x08\x08"
    assert len(proto) > 10_000_000

    assert set(numerical) == {"max_error", "tolerance", "result",
                              "harness", "n_probe"}
    assert numerical["result"] == "PASS"
    assert numerical["harness"] == "python-onnx-roundtrip"
    assert numerical["n_probe"] == 3  # two seeded probes + a black frame
    assert float(numerical["max_error"]) < float(numerical["tolerance"])
    assert float(numerical["tolerance"]) == detection.ATOL

    graph = onnx.load_from_string(proto)
    onnx.checker.check_model(graph)
    assert graph.ir_version == 8
    assert graph.opset_import[0].version == 17
    ops = {n.op_type for n in graph.graph.node}
    assert ops <= set(detection.OPERATORS)  # never under-declared
    assert {"Conv", "Gemm", "MatMul", "Sigmoid", "TopK"} <= ops
    assert validate_export(contract_doc, "onnx")["operators"] == "PASS"

    shape_in = [d.dim_param or d.dim_value
                for d in graph.graph.input[0].type.tensor_type.shape.dim]
    assert shape_in == ["batch", 3, 64, 64]  # spatially static, batch
    # dynamic (the honest claim: DINOv2's positional encoding bakes H/W)
    assert [o.name for o in graph.graph.output] == ["dets", "labels"]

    # dynamic batch works beyond the traced batch of 1
    session = ort.InferenceSession(proto, providers=["CPUExecutionProvider"])
    got = session.run(None,
                      {"input": np.zeros((2, 3, 64, 64),
                                         dtype=np.float32)})
    assert [o.shape for o in got] == [(2, 10, 4), (2, 10, 4)]

    # the contract the workflow records matches the binary
    assert (contract_doc["inference_contract"]["numerical_tolerance"]
            ["atol"] == numerical["tolerance"])


def test_export_bytes_seg_emits_masks_output(weights_seg):
    import onnx

    proto, numerical = detection.export_bytes(
        entry=SEG_ENTRY, weights=weights_seg, fmt="onnx")
    assert numerical["result"] == "PASS"
    assert float(numerical["max_error"]) < detection.ATOL
    graph = onnx.load_from_string(bytes(proto))
    onnx.checker.check_model(graph)
    assert [o.name for o in graph.graph.output] == ["dets", "labels",
                                                    "masks"]
    ops = {n.op_type for n in graph.graph.node}
    assert ops <= set(detection.OPERATORS)
    assert {"Resize", "Einsum"} <= ops  # mask head + attention paths


def test_export_refuses_non_onnx_formats():
    for fmt in ("openvino", "tflite"):
        with pytest.raises(PreconditionFailed) as ei:
            detection.export_bytes(entry=ENTRY, weights=b"x", fmt=fmt)
        out = ei.value.render()
        assert fmt in out.lower()
        assert "onnx" in out.lower()  # refusal names the portable target


def test_verify_operators_blocks_undeclared_and_unreadable():
    from onnx import TensorProto, helper

    node = helper.make_node("NotAnOperator", ["x"], ["y"])
    graph = helper.make_graph(
        [node], "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    model = helper.make_model(graph)
    with pytest.raises(ValidationBlock) as ei:
        detection._verify_operators(model.SerializeToString(), "unit")
    out = ei.value.render()
    assert "NotAnOperator" in out and "does not declare" in out
    assert "15.3" in (ei.value.hint or "")

    with pytest.raises(ValidationBlock) as ei:
        detection._verify_operators(b"not onnx bytes", "unit")
    assert "not readable ONNX" in ei.value.render()
