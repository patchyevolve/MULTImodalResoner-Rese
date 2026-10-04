"""Detection engine — RF-DETR (`rf_detr_s`, `rf_detr_l`,
`rf_detr_seg_s`) real execution: one-shot image inference, COCO
mAP/AP50 evaluation, and a genuine ONNX export round-tripped against
the model itself.

Normative: 10_training_plan/06_benchmarking_plan.md §4 (COCO accuracy
target — mAP >= 53.0 for RF-DETR-S, measured with COCO/COCOeval),
13_product_specification §6.7 (EVALUATE runs a harness over the
dataset), §6.8 (INFER: contract check -> execute), §6.9 (EXPORT:
operator validation, real binary, numerical round-trip — FAIL blocks,
no partial export), 12_training_system.md §12.4 (real harness — no
scaffold), §15.3 (the contract declares how the model is called),
§15.4 (train is never an eval split).

Design (import-safe: torch / rfdetr / torchvision / onnx /
onnxruntime / PIL / pycocotools are imported inside functions — the
stdlib core imports cleanly everywhere):
  * weights — the family checkpoint loaded through rfdetr's canonical
    `RFDETR.from_checkpoint(path, trust_checkpoint=True)` (the bytes
    arrive from the workflow's verified checkpoint store or registry —
    12 §15.3, so trust is earned upstream); a checkpoint whose class is
    not the entry's registered variant is refused, never guessed;
  * inference mirrors `RFDETR.predict` exactly — RGB -> float/255 CHW
    -> square bilinear resize (antialias=False, the training-matched
    resampler) -> ImageNet mean/std — then rfdetr's own `PostProcess`
    (sigmoid top-k, NMS-free: RF-DETR's design) with detections kept at
    `score > SCORE_THRESHOLD`;
  * execute — JSON-ready detections: xyxy pixel boxes, scores, labels;
    segmentation models add COCO RLE `masks`;
  * metrics — COCO `COCOeval` bbox AP over the prepared
    `coco_detection` records. Model label -> category_id comes from the
    TRAIN split's filtered category list (rfdetr's own rule — labels
    are positions in `filter_parent_categories`, and the train split
    alone defines them so a valid-only category cannot shift indices);
    a label outside that mapping is DROPPED, never guessed (rfdetr's
    `_resolve_category_id` drops it too). The protocol's `split` and
    `sample_subset` are honored; `thresholds`/`nms` are deliberately
    ignored — COCO AP is rank-based and RF-DETR's postprocessor is
    NMS-free, so a conf/NMS knob would not change the ranking (it
    would only drop true positives whose score is under the knob);
  * export — trace the real graph (batch-only dynamic axes: the DINOv2
    positional encoding is baked to the export shape, so spatial
    dynamism would be a lie), verify the produced op set is a subset of
    the contract's OPERATORS, then round-trip deterministic probes
    (python forward vs onnxruntime): max error < ATOL or FAIL (13 §6.9).

Measured on this toolchain (rfdetr 11.1 / torch 2.14 / onnxruntime
1.30): torch-vs-ONNX Runtime drift <= 3.1e-5 at contract probe scale
and ~8.5e-5 on a real RF-DETR-S checkpoint at 512 — ATOL = 1e-3 keeps
~12x headroom while a graph defect (or a numerically unstable model)
lands orders of magnitude above it and blocks the export.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash, content_hash_bytes

FAMILY = "detection"

#: Models this engine runs (12 §10.2 `detection` family — the same keys
#: the RF-DETR trainer registers in `MODEL_REGISTRY`).
MODEL_NAMES = frozenset({"rf_detr_s", "rf_detr_l", "rf_detr_seg_s"})

#: Numerical round-trip tolerance for the ONNX export (13 §6.9):
#: measured torch-vs-ort drift is <= 3.1e-5 at probe scale and ~8.5e-5
#: on a real checkpoint; a genuine export defect lands far above this.
ATOL = 1e-3

#: Detection score gate recorded in the contract and applied by
#: execute (rfdetr's `predict()` default threshold).
SCORE_THRESHOLD = 0.5

#: Family-default input resolution per model, read from rfdetr's own
#: config classes (RFDETRSmallConfig.resolution = 512, RFDETRLarge bare
#: construction = 704, RFDETRSegSmallConfig.resolution = 384). The
#: run-spec semantic carries no resolution override (the trainer reads
#: none), so a first export declares the family default; execution
#: always resizes to the CHECKPOINT's own resolution, which every
#: result records.
FAMILY_RESOLUTION: dict[str, int] = {
    "rf_detr_s": 512,
    "rf_detr_l": 704,
    "rf_detr_seg_s": 384,
}

#: ImageNet normalization rfdetr's RFDETR class applies in predict().
MEANS: list[float] = [0.485, 0.456, 0.406]
STDS: list[float] = [0.229, 0.224, 0.225]

#: Contract operator list — the ONNX op types observed across the
#: rf_detr_s / rf_detr_l / rf_detr_seg_s graphs on this toolchain
#: (detection graphs use 42; Einsum + Resize come from the large /
#: segmentation variants). The export refuses any graph op outside
#: this list: an under-declared contract would promise portability it
#: cannot deliver (12 §15.3).
OPERATORS: tuple[str, ...] = (
    "Add", "And", "Cast", "Concat", "Constant", "ConstantOfShape",
    "Conv", "Cos", "Div", "Einsum", "Equal", "Erf", "Exp", "Expand",
    "Gather", "GatherElements", "Gemm", "Greater", "GridSample",
    "Identity", "LayerNormalization", "Less", "MatMul", "Mul", "Not",
    "Range", "ReduceMax", "ReduceSum", "Relu", "Reshape", "Resize",
    "Shape", "Sigmoid", "Sin", "Slice", "Softmax", "Split", "Sqrt",
    "Sub", "Tile", "TopK", "Transpose", "Unsqueeze", "Where",
)

#: Deterministic round-trip probes (image-space seeds — normalized
#: before they reach the graph so the probes exercise the operating
#: region `predict()` feeds the model).
PROBE_SEEDS: tuple[int, ...] = (1234, 99)

#: Memoized dependency verdict (the environment cannot change within
#: a process — mirrors the RF-DETR trainer).
_DEP_CACHE: list[Any] = []


def dependency_error() -> str | None:
    """None = the detection stack (rfdetr + torch + Pillow) is here."""
    if _DEP_CACHE:
        return _DEP_CACHE[0]
    err: str | None = None
    try:
        import rfdetr  # noqa: F401
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception as exc:  # pragma: no cover — broken installation
        err = (
            f"the detection engine needs rfdetr + torch + Pillow "
            f"({type(exc).__name__}: {exc}) — pip install rfdetr"
        )
    _DEP_CACHE.append(err)
    return err


def input_types() -> frozenset[str]:
    """Contracts this engine executes — one RGB image file."""
    return frozenset({"image"})


def metric_names(weights: bytes | None = None) -> tuple[str, ...]:
    """COCO bbox metrics (06 §4: mAP/AP50 from COCOeval.stats[0:2]).
    `weights` is unused here (engine contract — the calibrator derives
    per-model keys from it)."""
    return ("mAP", "AP50")


def _model_name(entry: Mapping[str, Any]) -> str:
    """The entry's model name, fail-closed to this family's registry."""
    name = str(entry.get("name") or "")
    if name not in MODEL_NAMES:
        raise ValidationBlock(
            f"model {name!r} is not a detection-family entry — the "
            f"{FAMILY} engine runs: {', '.join(sorted(MODEL_NAMES))}",
        )
    return name


# -- checkpoint context ------------------------------------------------------


@dataclass
class _Context:
    """A loaded RF-DETR checkpoint + the scratch dir that holds it.

    `from_checkpoint` re-reads the file while constructing, so the
    temp copy lives as long as the context; `cleanup()` removes it.
    """

    det: Any
    name: str
    task: str
    resolution: int
    means: list[float]
    stds: list[float]
    tmpdir: str
    _closed: bool = field(default=False, repr=False)

    def cleanup(self) -> None:
        if not self._closed:
            self._closed = True
            shutil.rmtree(self.tmpdir, ignore_errors=True)


def _load_context(weights: bytes, entry: Mapping[str, Any]) -> _Context:
    """Checkpoint bytes -> verified RF-DETR instance (fail-closed).

    The workflow only ever hands this engine bytes whose integrity the
    checkpoint store / registry already verified (12 §15.3), so the
    load mirrors `RFDETR.from_checkpoint(..., trust_checkpoint=True)`.
    A checkpoint whose reconstructed class is not the entry's
    registered variant BLOCKs — weights of a different family are
    never guessed into place.
    """
    import rfdetr

    name = _model_name(entry)
    if not weights:
        raise ValidationBlock(
            f"model {name}: checkpoint weights are empty",
            hint="weights come from the run's committed checkpoints "
                 "(12 §11) or the registry (12 §15.3)",
        )
    from mlforge.trainers.rfdetr import MODEL_REGISTRY

    variant = MODEL_REGISTRY[name]
    tmpdir = tempfile.mkdtemp(prefix="mlforge_detection_")
    path = Path(tmpdir) / "mlforge_checkpoint.ckpt"
    try:
        path.write_bytes(weights)
        try:
            import warnings

            with warnings.catch_warnings():
                # rfdetr scans its variant table (deprecated proxies raise
                # FutureWarnings) on every from_checkpoint — load noise
                # must not drown the test/CLI output.
                warnings.simplefilter("ignore")
                det = rfdetr.RFDETR.from_checkpoint(
                    str(path), trust_checkpoint=True, device="cpu")
        except Exception as exc:
            raise ValidationBlock(
                f"model {name}: checkpoint cannot be loaded as an "
                f"RF-DETR model ({type(exc).__name__}: {exc})",
                hint="the `model` component must be an rfdetr "
                     "checkpoint (args + model_name + model_config + "
                     "weights) — 12 §15.3",
            ) from exc
        actual = type(det).__name__
        if actual != variant.class_name:
            raise ValidationBlock(
                f"model {name}: checkpoint is a {actual}, but the "
                f"entry registers {variant.class_name} — refusing to "
                "run weights from a different family",
            )
        nn = getattr(getattr(det, "model", None), "model", None)
        if nn is None:
            raise ValidationBlock(
                f"model {name}: checkpoint loaded without a model — "
                "the weights blob is incomplete (12 §15.3)")
        nn.eval()
        return _Context(
            det=det,
            name=name,
            task=variant.task,
            resolution=int(det.model_config.resolution),
            means=[float(v) for v in det.means],
            stds=[float(v) for v in det.stds],
            tmpdir=tmpdir,
        )
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


# -- preprocessing + detection ----------------------------------------------


def _preprocess(ctx: _Context, img: Any) -> Any:
    """PIL RGB image -> normalized CHW tensor (`RFDETR.predict`'s own
    pipeline: float/255, square bilinear resize antialias=False, then
    ImageNet mean/std)."""
    import torch
    import torchvision.transforms.functional as TF

    arr = __import__("numpy").array(img, dtype="uint8")
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
    t = t.div(255.0)
    t = TF.resize(t, [ctx.resolution, ctx.resolution], antialias=False)
    return TF.normalize(t, ctx.means, ctx.stds)


def _forward(ctx: _Context, batch: Any) -> dict[str, Any]:
    """One native forward pass -> rfdetr's postprocess input dict.

    Mirrors `RFDETR.predict`'s tuple->dict mapping: element 0 is
    `pred_boxes` (normalized cxcywh), element 1 `pred_logits`, and a
    third element is `pred_masks` (our families never emit
    keypoints).
    """
    import torch

    with torch.no_grad():
        pred = ctx.det.model.model(batch)
    if isinstance(pred, dict):
        return pred
    if isinstance(pred, tuple):
        outputs = {"pred_boxes": pred[0], "pred_logits": pred[1]}
        if len(pred) >= 3:
            outputs["pred_masks"] = pred[2]
        return outputs
    raise ValidationBlock(
        f"model {ctx.name}: unexpected forward output "
        f"{type(pred).__name__} — expected tensors or a dict (13 §6.8)")


def _postprocess(ctx: _Context, outputs: dict[str, Any],
                 height: int, width: int,
                 score_threshold: float | None) -> dict[str, Any]:
    """rfdetr's own PostProcess + the predict-side score filter.

    Returns the per-image detection dict; `score_threshold=None`
    keeps every top-k row (evaluation is rank-based).
    """
    import torch

    target_sizes = torch.tensor([[int(height), int(width)]])
    results = ctx.det.model.postprocess(
        outputs, target_sizes=target_sizes,
        score_threshold=score_threshold)
    res = results[0]
    scores = res["scores"]
    if score_threshold is not None:
        # The box-only path never pre-filters inside PostProcess (its own
        # doc: evaluation needs every selected row), so predict()'s own
        # `scores > threshold` keep-index is mirrored here — masks rows
        # (K, 1, h, w) share the leading K and travel with the survivors.
        keep = (scores > score_threshold).nonzero(as_tuple=True)[0]
        res = {k: (v[keep] if isinstance(v, torch.Tensor)
                   and v.shape[:1] == scores.shape[:1] else v)
               for k, v in res.items()}
        scores = res["scores"]
    return res


def _detect_one(ctx: _Context, path: str,
                *, score_threshold: float | None) -> dict[str, Any]:
    """Run ONE image through the model -> JSON-ready detection dict.

    Unreadable content BLOCKs before execution (13 §6.8/§7): a file
    that passed the schema check but cannot be decoded is a content
    failure, not an empty prediction.
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            img = im.convert("RGB")
            width, height = img.size
            batch = _preprocess(ctx, img).unsqueeze(0)
    except Exception as exc:
        raise ValidationBlock(
            f"input image unreadable: {path} "
            f"({type(exc).__name__}: {exc})",
            hint="13 §6.8 — content validation BLOCKs before execution",
        ) from exc
    if width < 1 or height < 1:
        raise ValidationBlock(
            f"input image has invalid dimensions {width}x{height}")

    outputs = _forward(ctx, batch)
    res = _postprocess(ctx, outputs, height, width, score_threshold)
    boxes = res["boxes"]
    scores = res["scores"]
    labels = res["labels"]
    out: dict[str, Any] = {
        "boxes": [[round(float(v), 6) for v in row]
                  for row in boxes.tolist()],
        "scores": [round(float(s), 6) for s in scores.tolist()],
        "labels": [int(v) for v in labels.tolist()],
        "count": len(scores),
        "threshold": (float(score_threshold)
                      if score_threshold is not None else None),
        "image": {"width": int(width), "height": int(height)},
        "resolution": int(ctx.resolution),
    }
    if "masks" in res:
        out["masks"] = _encode_masks(res["masks"], height, width)
    return out


def _encode_masks(masks: Any, height: int, width: int) -> list[dict]:
    """Boolean per-detection masks -> COCO RLE dicts (JSON-safe).

    PostProcess already resized the masks to the source image size;
    a defensive nearest-resize covers an upsample-disabled config.
    """
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils

    out: list[dict] = []
    planes = masks
    for i in range(int(planes.shape[0])):
        plane = planes[i]
        while isinstance(plane, torch.Tensor) and plane.ndim > 2:
            plane = plane[0]
        binary = (plane > 0).to(torch.uint8)
        if tuple(binary.shape) != (int(height), int(width)):
            binary = torch.nn.functional.interpolate(
                binary[None, None].float(), size=(int(height), int(width)),
                mode="nearest")[0, 0].to(torch.uint8)
        arr = np.asfortranarray(binary.numpy())
        rle = mask_utils.encode(arr)
        counts = rle.get("counts")
        if isinstance(counts, (bytes, bytearray)):
            counts = counts.decode("ascii")
        out.append({"size": [int(height), int(width)],
                    "counts": str(counts)})
    return out


def _detect_all(ctx: _Context,
                items: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Evaluate every item (the seam tests replace — the real path is
    one `_detect_one` per image)."""
    return [_detect_one(ctx, str(it["path"]), score_threshold=None)
            for it in items]


# -- contract (12 §15.3, 13 §6.9) --------------------------------------------


def contract(
    entry: Mapping[str, Any],
    fmt: str,
    *,
    semantic: Mapping[str, Any] | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """First-export `model_spec.json` for the detection family.

    `semantic` is the run's semantic identity (12 §15.4) — it carries
    no resolution, so the resize block declares FAMILY_RESOLUTION; the
    executing consumer always learns the model's real input size from
    the checkpoint (recorded as `resolution` in every result).
    """
    from mlforge.ops.exporting import FORMATS
    from mlforge.trainers.rfdetr import MODEL_REGISTRY

    name = _model_name(entry)
    resolution = FAMILY_RESOLUTION[name]
    ops = list(OPERATORS)
    fields: list[dict[str, Any]] = [
        {"name": "boxes", "dtype": "float[][]",
         "layout": "xyxy pixels per image"},
        {"name": "scores", "dtype": "float[]"},
        {"name": "labels", "dtype": "int[]",
         "note": "contiguous train-split label indices"},
    ]
    if MODEL_REGISTRY[name].task == "segmentation":
        fields.append({"name": "masks", "dtype": "rle[]",
                       "layout": "COCO RLE over the source image"})
    return {
        "schema_version": 1,
        "name": entry.get("name"),
        "version": entry.get("version"),
        "harness": FAMILY,
        "inference_contract": {
            "input_schema": [
                {
                    "name": "image",
                    "type": "image",
                    "dtype": "uint8",
                    "layout": "HWC",
                    "color_space": "RGB",
                    "resize": {"width": resolution, "height": resolution,
                               "mode": "bilinear", "antialias": False},
                    "normalization": {"mean": list(MEANS),
                                      "std": list(STDS)},
                }
            ],
            "output_schema": [
                {
                    "name": "detections",
                    "type": "structured",
                    "schema": "json_schema://mlforge.detections.v1",
                    "fields": fields,
                    "parameters": {"score_threshold": SCORE_THRESHOLD},
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
            "minimal_memory": {"vram_mb": 512},
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
    return (f"the {fmt} converter is not integrated for the detection "
            "engine (13 §4.1 names ONNX as the portable target for "
            "exported models) — use --format onnx")


def _check_contract(contract_doc: Mapping[str, Any],
                    entry: Mapping[str, Any]) -> None:
    """Consumer-side contract verification (12 §15.3 — verify BEFORE
    execution): the contract must declare this family's image input, and
    a declared resize must be the family default `contract()` produces —
    a contract minted for another model (or another harness) is a BLOCK,
    never a silent re-interpretation.

    The resize check is against FAMILY_RESOLUTION, not the checkpoint:
    the checkpoint's own resolution is the executor's authority (every
    result records it), while `contract()` has no weights to read — it
    declares the family default (see `contract()`'s docstring).
    """
    name = _model_name(entry)
    doc = contract_doc or {}
    if doc.get("harness") not in (None, FAMILY):
        raise ValidationBlock(
            f"contract harness {doc.get('harness')!r} is not the "
            f"{FAMILY} family — cannot execute detection weights against "
            "it (12 §15.3)",
            hint="re-export the model so model_spec.json is rebuilt "
                 "(`mlforge export <MODEL> --format onnx`)",
        )
    contract = doc.get("inference_contract") or {}
    declared = None
    for item in contract.get("input_schema") or []:
        if isinstance(item, dict) and item.get("type") == "image":
            declared = item
            break
    if declared is None:
        raise ValidationBlock(
            "detection contract declares no `image` input_schema entry "
            "— cannot execute (12 §15.3)",
            hint="re-export the model so the contract records its image "
                 "input (`mlforge export <MODEL> --format onnx`)",
        )
    resize = declared.get("resize")
    if not isinstance(resize, dict):
        return
    if resize.get("height") is None or resize.get("width") is None:
        return
    try:
        got = (int(resize["height"]), int(resize["width"]))
    except (TypeError, ValueError) as exc:
        raise ValidationBlock(
            f"contract resize {resize!r} is not height/width integers",
            hint="the contract at model_spec.json is the authority "
                 "(12 §15.3) — re-export to rebuild it",
        ) from exc
    want = (FAMILY_RESOLUTION[name], FAMILY_RESOLUTION[name])
    if got != want:
        raise ValidationBlock(
            f"contract declares resize {got[0]}x{got[1]} (HxW) but "
            f"{name} is contracted at {want[0]}x{want[1]}",
            hint="contract and model disagree (12 §15.3) — re-export the "
                 "model so the contract is rebuilt from the family "
                 "definition",
        )


# -- execute (13 §6.8) -------------------------------------------------------


def execute(
    *,
    entry: Mapping[str, Any],
    contract_doc: Mapping[str, Any],
    checked: Mapping[str, Any],
    weights: bytes,
) -> dict[str, Any]:
    """Real one-shot inference: RGB image -> detections (JSON-ready).

    Family ordering (mirrors the re-ID engine): the weights blob first
    (a checkpoint that is not this entry's registered variant is
    refused), then the contract check (12 §15.3), then the checked
    input — the score gate is the contract's own recorded parameter
    (single source: `SCORE_THRESHOLD`).
    """
    ctx = _load_context(weights, entry)
    try:
        _check_contract(contract_doc, entry)
        try:
            raw = checked["input"]["path"]
        except (TypeError, KeyError, IndexError) as exc:
            raise ValidationBlock(
                "checked input carries no path — the input schema "
                "check did not run (13 §6.8)",
                hint="infer validates against model_spec.json first",
            ) from exc
        if not str(raw or "").strip():
            raise ValidationBlock(
                "checked input carries no path — the input schema "
                "check did not run (13 §6.8)",
                hint="infer validates against model_spec.json first")
        path = Path(str(raw))
        if not path.is_file():
            raise ValidationBlock(
                f"input image not found: {path}",
                hint="one-shot infer takes a single existing file "
                     "(13 §6.8)",
            )
        return _detect_one(ctx, str(path), score_threshold=SCORE_THRESHOLD)
    finally:
        ctx.cleanup()


# -- evaluate (13 §6.7, 12 §15.4) --------------------------------------------


def _read_ann(base: Path, rec: Mapping[str, Any]) -> tuple[Path, dict]:
    """Prepared record -> verified COCO annotation file (fail-closed)."""
    rel = str(rec.get("relative_path") or "")
    path = base / rel
    if not path.is_file():
        raise ValidationBlock(
            f"annotation missing at registered path: {path}",
            hint="dataset content changed since registration — "
                 "mlforge dataset add <NAME> <PATH> --force")
    data = path.read_bytes()
    expected = str(rec.get("sha256") or "")
    if expected and expected != content_hash_bytes(data):
        raise ValidationBlock(
            f"annotation {rel} no longer matches its prepared sha256 "
            "— content drift since prepare",
            hint="re-run `mlforge prepare <MODEL>` (identity is "
                 "verified, never reinterpreted)")
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationBlock(
            f"annotation {rel} is not readable COCO JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise ValidationBlock(
            f"annotation {rel} must be a JSON object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(doc.get(key), list) or not doc[key]:
            raise ValidationBlock(
                f"annotation {rel} has no {key} — an empty COCO "
                "section cannot ground a measurement")
    return path, doc


def _annotation_files(
    records: Sequence[Mapping[str, Any]], base: Path, split: str,
) -> tuple[Path, dict, dict | None, str]:
    """Locate + identity-verify the eval (and train) annotation files.

    Returns `(eval_path, eval_doc, train_doc_or_None, split_dir)`.
    The train file defines the label space (rfdetr's rule); when the
    dataset has no train record the eval split's own categories stand
    in — the same fallback `_train_split_cat2label` takes.
    """
    from mlforge.trainers.rfdetr_data import _detect_layout, _find_ann

    layout = _detect_layout(records)
    key = str(split).strip().lower()
    if key.startswith("train"):
        raise ValidationBlock(
            f"protocol split {split!r} — train rows are never eval "
            "rows (12 §15.4)",
            hint="evaluate on the val/valid split (13 §6.7)")
    if layout == "roboflow":
        name = {"val": "valid", "valid": "valid",
                "validation": "valid", "test": "test"}.get(key)
        if name is None:
            raise ValidationBlock(
                f"protocol split {split!r} — a roboflow layout "
                "provides train/valid/test")
        eval_suffix = f"{name}/_annotations.coco.json"
        train_suffix = "train/_annotations.coco.json"
        split_dir = name
    else:
        name = {"val": "val", "valid": "val",
                "validation": "val", "test": "test"}.get(key)
        if name is None:
            raise ValidationBlock(
                f"protocol split {split!r} — a coco2017 layout "
                "provides train/val/test")
        eval_suffix = f"annotations/instances_{name}2017.json"
        train_suffix = "annotations/instances_train2017.json"
        split_dir = f"{name}2017"
    eval_rec = _find_ann(records, eval_suffix)
    if eval_rec is None:
        raise ValidationBlock(
            f"dataset has no {eval_suffix} record — the eval split's "
            "annotations are the measurement target (13 §6.7)",
            hint="prepare a dataset whose records include the eval "
                 "split (`mlforge dataset types` lists the layouts)")
    eval_path, eval_doc = _read_ann(base, eval_rec)
    train_doc: dict | None = None
    train_rec = _find_ann(records, train_suffix)
    if train_rec is not None:
        _, train_doc = _read_ann(base, train_rec)
    return eval_path, eval_doc, train_doc, split_dir


def _label2cat(train_doc: dict | None, eval_doc: dict) -> dict[int, int]:
    """Model label -> COCO category_id (rfdetr's own derivation).

    Labels are positions in the TRAIN split's filtered category list;
    unknown labels are dropped downstream, never guessed.
    """
    from rfdetr.datasets.coco import (
        annotated_category_ids,
        filter_parent_categories,
    )

    src = train_doc if train_doc is not None else eval_doc
    try:
        kept = filter_parent_categories(
            list(src.get("categories") or []), annotated_category_ids(src))
        mapping = {label: int(cat["id"]) for label, cat in enumerate(kept)}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationBlock(
            f"annotation categories are not COCO-shaped ({exc}) — every "
            "category needs an `id` and a `name` to ground a "
            "measurement") from exc
    if not mapping:
        raise ValidationBlock(
            "no evaluable categories remain after rfdetr's parent "
            "filter — the dataset cannot ground a measurement")
    return mapping


def _eval_items(
    eval_doc: dict, base: Path, split_dir: str,
    records: Sequence[Mapping[str, Any]], ref: str,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Eval-split images -> validated items (path resolved under the
    registered base, membership-checked against the prepared records)."""
    from mlforge.trainers.rfdetr_data import _resolve_image

    rec_rels = {str(r.get("relative_path") or "") for r in records}
    items: list[dict[str, Any]] = []
    for i, im in enumerate(eval_doc.get("images") or []):
        if not isinstance(im, dict):
            raise ValidationBlock(
                f"annotation image entry {i} is not an object")
        try:
            image_id = int(im["id"])
            width = int(im["width"])
            height = int(im["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationBlock(
                f"annotation image entry {i} lacks a valid "
                f"id/width/height ({exc})") from exc
        file_name = str(im.get("file_name") or "")
        if not file_name:
            raise ValidationBlock(
                f"annotation image {image_id} has no file_name")
        if width < 1 or height < 1:
            raise ValidationBlock(
                f"annotation image {image_id} has invalid dimensions "
                f"{width}x{height}")
        path, rel = _resolve_image(base, file_name, (split_dir,))
        if rel not in rec_rels:
            raise ValidationBlock(
                f"annotation references image {file_name!r} outside "
                "the prepared records — content drift since prepare",
                hint="re-run `mlforge prepare <MODEL>` (records "
                     "identity-cover every file)")
        items.append({"image_id": image_id, "path": str(path),
                      "width": width, "height": height})
    if not items:
        raise ValidationBlock(
            f"dataset {ref} has no eval-split images — nothing to "
            "measure (13 §6.7)")
    subset = protocol.get("sample_subset")
    if subset is not None:
        if isinstance(subset, bool) or not isinstance(subset, int) \
                or subset < 1:
            raise ValidationBlock(
                f"protocol.sample_subset must be a positive integer "
                f"(first N images), got {subset!r}")
        items = items[:subset]
    return items


def _coco_results(
    items: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    label2cat: Mapping[int, int], cat_ids: set[int],
) -> list[dict[str, Any]]:
    """Detection dicts -> COCO result records.

    An unmapped label (background slot or a label outside the train
    split's label space) is dropped — rfdetr's `_resolve_category_id`
    returns None for it too; a wrong guess here would score phantom
    categories.
    """
    results: list[dict[str, Any]] = []
    for item, det in zip(items, detections):
        image_id = int(item["image_id"])
        iw = float(item["width"])
        ih = float(item["height"])
        boxes = det.get("boxes") or []
        scores = det.get("scores") or []
        labels = det.get("labels") or []
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValidationBlock(
                f"detector returned ragged output for image "
                f"{image_id} (boxes={len(boxes)} scores={len(scores)} "
                f"labels={len(labels)})")
        for box, score, label in zip(boxes, scores, labels):
            category_id = label2cat.get(int(label))
            if category_id is None or int(category_id) not in cat_ids:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError) as exc:
                raise ValidationBlock(
                    f"image {image_id}: box {box!r} is not numeric"
                ) from exc
            x1 = min(max(x1, 0.0), iw)
            x2 = min(max(x2, 0.0), iw)
            y1 = min(max(y1, 0.0), ih)
            y2 = min(max(y2, 0.0), ih)
            results.append({
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [x1, y1, max(0.0, x2 - x1),
                         max(0.0, y2 - y1)],
                "score": float(score),
            })
    return results


def _run_cocoeval(
    ann_path: Path,
    results: list[dict[str, Any]],
    img_ids: Sequence[int] | None = None,
) -> tuple[float, float]:
    """COCO bbox eval (06 §4's own COCO/COCOeval sequence).

    `img_ids` restricts the evaluation index to exactly the images the
    detector ran over — with `sample_subset` the protocol's "first N
    images" must both run AND be the measurement set, or every
    unsampled ground-truth image would count as a miss and dilute AP
    (the sibling engines score over precisely their subset).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    try:
        with redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(ann_path))
            if not results:
                # No detections at all (every row filtered or the
                # model found nothing): AP is 0 by definition, and
                # pycocotools cannot load an empty result list.
                return 0.0, 0.0
            if "info" not in coco_gt.dataset:
                coco_gt.dataset["info"] = {}
            coco_dt = coco_gt.loadRes(results)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            if img_ids is not None:
                ev.params.imgIds = sorted(int(i) for i in img_ids)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
    except ValidationBlock:
        raise
    except Exception as exc:
        raise ValidationBlock(
            f"COCOeval failed over {ann_path.name} "
            f"({type(exc).__name__}: {exc}) — refusing to report a "
            "metric the evaluation itself could not compute") from exc
    stats = [float(s) for s in ev.stats]
    if not stats or stats[0] < 0 or stats[1] < 0:
        raise ValidationBlock(
            "COCOeval produced no valid AP (stats[0]/stats[1] = "
            f"{stats[:2]}) — the ground truth does not match the "
            "detections' category space")
    return round(stats[0], 6), round(stats[1], 6)


def compute_metrics(
    *,
    entry: Mapping[str, Any],
    weights: bytes,
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    semantic: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Real mAP / AP50 — COCO `COCOeval` bbox over prepared records.

    Honors protocol `split` (never train, 12 §15.4) and
    `sample_subset` (first N images, annotation order — COCOeval is
    indexed to exactly those images); `thresholds`
    and `nms` are ignored by design (see the module docstring —
    rank-based AP with an NMS-free postprocessor). Segmentation entries
    measure bbox AP over their boxes (the metric set is the family's).
    `semantic` is the run's semantic identity, recorded by the
    protocol, not consumed by the metric.
    """
    from mlforge.trainers.rfdetr import MODEL_REGISTRY

    name = _model_name(entry)
    task = MODEL_REGISTRY[name].task
    if task not in ("detection", "segmentation"):  # pragma: no cover
        # Registry cross-check — a mis-registered task must never
        # silently measure the wrong thing.
        raise ValidationBlock(
            f"model {name}: registry task {task!r} has no COCO bbox "
            "metric in this engine")
    transform = dataset.get("transform")
    if transform and str(transform) != "coco_detection":
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} was prepared with "
            f"transform {transform!r} — the detection engine measures "
            "COCO mAP over `coco_detection` records",
            hint="mlforge prepare <MODEL> with the coco_detection "
                 "transform for the eval split (12 §7)",
        )
    records = [r for r in (dataset.get("records") or [])
               if isinstance(r, dict)]
    if not records:
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} has no prepared records — "
            "the detection engine needs `coco_detection` file records",
            hint="mlforge prepare <MODEL> registers the eval split as "
                 "a store artifact first (12 §7: nothing runs on "
                 "unresolved data)")
    raw_path = dataset.get("path")
    if not raw_path:
        raise PreconditionFailed(
            f"dataset {dataset.get('ref')} has no machine-local path "
            "— images cannot be read",
            hint="register the dataset with `mlforge dataset add "
                 "<NAME> <PATH>` so its files resolve here (12 §7)")
    base = Path(str(raw_path))
    split = str(protocol.get("split") or "val")
    ann_path, eval_doc, train_doc, split_dir = _annotation_files(
        records, base, split)
    items = _eval_items(eval_doc, base, split_dir, records,
                        str(dataset.get("ref")), protocol)
    label2cat = _label2cat(train_doc, eval_doc)
    cat_ids: set[int] = set()
    for c in eval_doc["categories"]:
        try:
            cat_ids.add(int(c["id"]))
        except (KeyError, TypeError, ValueError):  # pragma: no cover
            raise ValidationBlock(
                "eval annotation has a category without a numeric id — "
                "cannot ground a COCO measurement") from None

    ctx = _load_context(weights, entry)
    try:
        detections = _detect_all(ctx, items)
    finally:
        ctx.cleanup()
    if len(detections) != len(items):
        raise ValidationBlock(
            f"detector returned {len(detections)} result sets for "
            f"{len(items)} images — the measurement would misalign")
    results = _coco_results(items, detections, label2cat, cat_ids)
    # Evaluate over exactly the images that ran: a `sample_subset`
    # restricts the measurement set, never dilutes it (sibling
    # engines score over precisely their subset).
    m_ap, ap50 = _run_cocoeval(
        ann_path, results, [int(i["image_id"]) for i in items])
    return {"mAP": m_ap, "AP50": ap50}


# -- export (13 §6.9) --------------------------------------------------------


def _probe_inputs(ctx: _Context) -> list[Any]:
    """Deterministic probes in the region predict() feeds the graph:
    two seeded uniform images + a black frame, all normalized."""
    import torch
    import torchvision.transforms.functional as TF

    probes: list[Any] = []
    side = int(ctx.resolution)
    for seed in PROBE_SEEDS:
        gen = torch.Generator().manual_seed(seed)
        img = torch.rand(1, 3, side, side, generator=gen)
        probes.append(TF.normalize(img, ctx.means, ctx.stds))
    black = torch.zeros(1, 3, side, side)
    probes.append(TF.normalize(black, ctx.means, ctx.stds))
    return probes


def _export_graph(ctx: _Context) -> tuple[bytes, Any]:
    """Trace the real graph -> (ONNX bytes, export graph).

    Batch-only dynamic axes are the honest claim: the backbone's
    positional encoding is baked to `shape`, so declaring spatial
    dynamism would export a graph that fails at any other size.
    """
    import warnings

    import torch
    from rfdetr.export._backend import _switch_to_export_mode
    from rfdetr.export.prepare import prepare_export_graph

    side = int(ctx.resolution)
    try:
        graph = prepare_export_graph(
            ctx.det.model.model, ctx.det.model_config,
            shape=(side, side), device="cpu", dynamic_batch=True)
        _switch_to_export_mode(graph.model)
        buf = io.BytesIO()
        with warnings.catch_warnings():
            # rfdetr's traced control flow (Python-level asserts and
            # shape branches) is expected — its own exporter traces
            # the same graph.
            warnings.simplefilter("ignore")
            torch.onnx.export(
                graph.model, (graph.input_tensors,), buf,
                input_names=list(graph.input_names),
                output_names=list(graph.output_names),
                export_params=True, keep_initializers_as_inputs=False,
                do_constant_folding=True, opset_version=17,
                dynamic_axes=graph.dynamic_axes, dynamo=False)
    except ValidationBlock:
        raise
    except Exception as exc:
        raise ValidationBlock(
            f"model {ctx.name}: ONNX trace failed "
            f"({type(exc).__name__}: {exc}) — refusing to emit a graph "
            "we cannot build (13 §7: never a partial export)") from exc
    return buf.getvalue(), graph


def _verify_operators(proto: bytes, name: str) -> None:
    """The produced graph must sit INSIDE the contract's operator list.

    The workflow checks the declared list against the ONNX spec; this
    closes the other direction — an op the contract forgot would ship
    a graph that later formats were never promised (12 §15.3).
    """
    import onnx

    try:
        model = onnx.load_from_string(proto)
    except Exception as exc:
        raise ValidationBlock(
            f"model {name}: exported bytes are not readable ONNX "
            f"({exc}) — refusing to ship an unverifiable graph") from exc
    actual = sorted({node.op_type for node in model.graph.node})
    undeclared = [op for op in actual if op not in OPERATORS]
    if undeclared:
        raise ValidationBlock(
            f"model {name}: graph uses operator(s) the contract does "
            f"not declare: {', '.join(undeclared)}",
            hint="the operator list is the portability promise "
                 "(12 §15.3) — no partial export (13 §7)")


def _roundtrip(ctx: _Context, proto: bytes, graph: Any) -> dict[str, Any]:
    """python forward vs onnxruntime over the probes (13 §6.9)."""
    import numpy as np
    import onnxruntime as ort
    import torch

    probes = _probe_inputs(ctx)
    try:
        session = ort.InferenceSession(
            proto, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ValidationBlock(
            f"model {ctx.name}: onnxruntime refused the graph "
            f"({type(exc).__name__}: {exc})") from exc
    input_name = str(graph.input_names[0])
    max_error = 0.0
    with torch.no_grad():
        for probe in probes:
            ref = graph.model(probe)
            got = session.run(None, {input_name: probe.numpy()})
            if len(got) != len(ref):
                raise ValidationBlock(
                    f"model {ctx.name}: graph emits {len(got)} output"
                    f"{'' if len(got) == 1 else 's'}, python emits "
                    f"{len(ref)} — the round-trip would compare "
                    "different things")
            for out, expected in zip(got, ref):
                diff = np.abs(np.asarray(out, dtype=np.float64)
                              - expected.numpy().astype(np.float64))
                if diff.size:
                    max_error = max(max_error, float(diff.max()))
    return {
        "max_error": round(max_error, 10),
        "tolerance": ATOL,
        "result": "PASS" if max_error < ATOL else "FAIL",
        "harness": "python-onnx-roundtrip",
        "n_probe": len(probes),
    }


def export_bytes(
    *,
    entry: Mapping[str, Any],
    weights: bytes,
    fmt: str,
) -> tuple[bytes, dict[str, Any]]:
    """Real export: trace the model -> ONNX bytes + round-trip check.

    Returns (binary, numerical_validation) — the workflow blocks on a
    FAIL result before any file is written (13 §6.9: no partial export,
    13 §7). Trace failures and graphs using operators the contract did
    not declare raise here, before any bytes leave this function.
    """
    if fmt != "onnx":
        raise PreconditionFailed(
            f"cannot export the detection engine to {fmt} — "
            f"{supports_format(fmt)}",
            hint="13 §6.9: only formats with a real converter produce "
                 "bytes; nothing partial is ever written (13 §7)")
    err = supports_format(fmt)
    if err:
        raise PreconditionFailed(
            f"cannot export the detection engine — {err}")
    ctx = _load_context(weights, entry)
    try:
        proto, graph = _export_graph(ctx)
        _verify_operators(proto, ctx.name)
        numerical = _roundtrip(ctx, proto, graph)
    finally:
        ctx.cleanup()
    return proto, numerical


__all__ = [
    "ATOL",
    "FAMILY",
    "FAMILY_RESOLUTION",
    "MODEL_NAMES",
    "OPERATORS",
    "SCORE_THRESHOLD",
    "compute_metrics",
    "contract",
    "dependency_error",
    "execute",
    "export_bytes",
    "input_types",
    "metric_names",
    "supports_format",
]
