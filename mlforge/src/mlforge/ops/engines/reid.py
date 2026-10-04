"""Re-ID engine — real OSNet embedding inference, Rank-1 / Rank-5 / mAP
metrics over prepared `reid_crops` records, and a genuine ONNX export
(11_model_weights §M5/§C2; 12 §10.2 `osnet` + `reid_crops`).

Normative: 10_training_plan/11_model_weights_and_disk_space.md §M5 (OSNet
x1_0, 2.19M backbone params, 256×128 input, Market1501 Rank-1 94.2% /
mAP 87.0%), 02_dataset_preparation.md §5 (query + gallery → Rank-1,
Rank-5, mAP), 06_benchmarking_plan.md §"Re-ID Accuracy" (metric keys
`rank1`/`rank5`/`mAP`, targets Rank-1 > 95% / mAP > 85%), 13
Product Specification §6.7–§6.9, 12 §15.3 (contract declares how the
model is called), §12.4 (real harness — no scaffold).

Family contract (this module IS the harness; `torch`/`PIL`/`onnx` are
imported only inside functions — the module is import-safe):
  * execute    — one-shot image → the model's 512-d L2-normalized
                 embedding, decoded and preprocessed with the TRAINER's
                 own inference transform (RGB → 256×128 bilinear resize →
                 ImageNet normalization), so train-time and infer-time
                 tensors cannot drift (11 §M5);
  * metrics    — `rank1` + `rank5` + `mAP` over prepared query/gallery
                 records with the standard Market1501 protocol (cosine
                 ranking, same-identity/same-camera junk and 0000/-1
                 distractors removed, AP averaged over queries) —
                 implemented here because the trainer has no validation
                 loop (`early_stopping_state.enabled: false`);
  * export     — the embedding forward (embed + L2 normalization) as a
                 real ONNX graph; numerical validation compares
                 onnxruntime against torch on deterministic probe images
                 (a FAIL blocks the export, 13 §6.9).
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash

FAMILY = "reid"

#: Models this engine runs — mirrors `mlforge.trainers.reid.MODEL_REGISTRY`
#: keys (kept literal so this module stays import-safe without torch; the
#: test suite asserts the two never drift).
MODEL_NAMES = frozenset({"osnet_x1_0"})

#: Numerical round-trip tolerance for the ONNX export (13 §6.9). Measured
#: torch-vs-onnxruntime max error over the probe batches is ~8e-8 — 1e-6
#: is an honest order-of-magnitude margin above the measurement, never a
#: rubber stamp (see tests/test_engines_reid.py).
ATOL = 1e-6

#: Contract operator list — the OSNet embedding graph exactly as
#: torch.onnx.export(opset 17, dynamic batch) emits it: the architecture
#: ops plus the L2-normalization and the dynamic-batch shape plumbing.
OPERATORS: tuple[str, ...] = (
    "Add",
    "AveragePool",
    "BatchNormalization",
    "Clip",
    "Concat",
    "Constant",
    "Conv",
    "Div",
    "Gather",
    "Gemm",
    "GlobalAveragePool",
    "Identity",
    "MaxPool",
    "Mul",
    "ReduceSum",
    "Relu",
    "Reshape",
    "Shape",
    "Sigmoid",
    "Sqrt",
    "Unsqueeze",
)

#: Record split a prepared `reid_crops` artifact uses for queries.
_QUERY_SPLIT = "query"

#: Gallery splits (`bounding_box_test/`, `test/`, `val/`) — train rows are
#: never eval rows (12 §15.4: `split` is never "train").
_GALLERY_SPLITS = frozenset({"test", "val"})

#: Images encoded per forward pass during evaluation (bounded memory).
_EVAL_BATCH = 16


def dependency_error() -> str | None:
    """None = runnable here — the trainer's own dependency probe (torch +
    pillow); onnx is checked at export time so inference/metrics work on
    a torch-only machine."""
    from mlforge.trainers.reid import dependency_error as _trainer_error

    return _trainer_error()


def input_types() -> frozenset[str]:
    return frozenset({"image"})


def metric_names(weights: bytes | None = None) -> tuple[str, ...]:
    """The re-ID benchmark's metric keys — 06_benchmarking_plan.md
    §"Re-ID Accuracy" returns exactly `{"rank1", "rank5", "mAP"}` with
    targets Rank-1 > 95% / mAP > 85% (02 §5: "query + gallery → Rank-1,
    Rank-5, mAP"; 03_training_pipeline.md: `--metrics rank1 rank5 mAP`).
    `weights` is unused here (engine contract — the calibrator derives
    per-model keys from it)."""
    return ("rank1", "rank5", "mAP")


def _arch_for(entry: Mapping[str, Any]) -> Any:
    """The registered architecture for this model name (identity comes
    from the name, exactly as the trainer resolves it — 11 §M5)."""
    from mlforge.trainers.reid import MODEL_REGISTRY

    name = str(entry.get("name") or "")
    arch = MODEL_REGISTRY.get(name)
    if arch is None:
        raise ValidationBlock(
            f"model {name!r} has no registered re-ID architecture",
            hint="runnable here: "
            + ", ".join(sorted(MODEL_REGISTRY))
            + " (11 §M5 — the engine runs the trainer's own architecture, "
            "never an approximation)",
        )
    return arch


def _load_model(weights: bytes, entry: Mapping[str, Any]) -> tuple[Any, Any]:
    """Weights bytes → OSNet in eval mode (fail-closed — a blob that is
    not this family's state_dict is refused, never re-initialized)."""
    import torch

    from mlforge.trainers.reid import OSNet

    name = f"{entry.get('name')}:{entry.get('version')}"
    arch = _arch_for(entry)
    try:
        state = torch.load(
            io.BytesIO(weights), weights_only=True, map_location="cpu"
        )
    except Exception as exc:
        raise ValidationBlock(
            f"model {name}: checkpoint model blob is not a torch state_dict "
            f"({type(exc).__name__}: {exc})",
            hint="the re-ID `model` component is the OSNet state_dict saved "
            "by ReidTrainer.checkpoint_payload (11 §M5 / 12 §11.4)",
        ) from exc
    if not isinstance(state, dict) or not state:
        raise ValidationBlock(
            f"model {name}: checkpoint model blob must be a non-empty "
            f"state_dict, got {type(state).__name__}",
            hint="the re-ID `model` component is the OSNet "
            "state_dict (11 §M5)",
        )
    cleaned = {
        (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
        for k, v in state.items()
    }
    head = cleaned.get("classifier.weight")
    shape = getattr(head, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValidationBlock(
            f"model {name}: state_dict has no OSNet classifier head "
            "(`classifier.weight`) — not this family's checkpoint",
            hint="this checkpoint was produced by a different trainer — "
            "weights are never re-initialized to fit (12 §15.3)",
        )
    num_classes = int(shape[0])
    if num_classes < 2:
        raise ValidationBlock(
            f"model {name}: classifier head has {num_classes} identities — "
            "the trainer requires at least 2 (one class teaches nothing)",
        )
    model = OSNet(arch, num_classes)
    try:
        model.load_state_dict(cleaned, strict=True)
    except Exception as exc:
        raise ValidationBlock(
            f"model {name}: state_dict does not fit OSNet x1_0 "
            f"({type(exc).__name__}: {exc})",
            hint="the parent model must be this architecture — weights are "
            "never partially loaded or re-initialized (12 §15.3)",
        ) from exc
    model.eval()
    return model, arch


def _preprocess(path: Path, arch: Any, *, where: str) -> Any:
    """Image file → the trainer's inference tensor (11 §M5 parity).

    Byte-identical to `ReidTrainer._gather`'s decode — RGB convert →
    bilinear resize to `arch.input_hw` → /255 → ImageNet mean/std (the
    PNG round-trip in `_gather` reproduces exactly these bytes). The
    horizontal flip there is training augmentation, never inference.
    """
    import torch
    from PIL import Image

    from mlforge.trainers.reid import _IM_MEAN, _IM_STD

    h, w = int(arch.input_hw[0]), int(arch.input_hw[1])
    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((w, h), Image.BILINEAR)
            data = im.tobytes()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValidationBlock(
            f"cannot decode {where} {path} as an RGB image "
            f"({type(exc).__name__}: {exc})",
            hint="input type `image` accepts png/jpg/jpeg/bmp/webp — 13 §6.8: "
            "an input that does not decode BLOCKs before execution "
            "(check_input verified the extension, the pixels fail here)",
        ) from exc
    t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    t = t.view(h, w, 3).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(_IM_MEAN).view(3, 1, 1)
    std = torch.tensor(_IM_STD).view(3, 1, 1)
    return (t - mean) / std


def _encode_paths(
    model: Any,
    paths: list[Any],
    *,
    arch: Any,
    where: str,
    batch: int = _EVAL_BATCH,
) -> list[list[float]]:
    """Image paths → raw (un-normalized) embedding vectors.

    The single encode seam: `execute` and `compute_metrics` both run
    through here, so one preprocessing definition cannot drift between
    the two seams."""
    import torch

    model.eval()  # never encode through a train-mode BatchNorm
    size = max(1, int(batch))
    out: list[list[float]] = []
    for start in range(0, len(paths), size):
        chunk = paths[start : start + size]
        x = torch.stack(
            [_preprocess(Path(p), arch, where=where) for p in chunk]
        )
        with torch.no_grad():
            feats = model.embed(x)
        out.extend([[float(v) for v in row] for row in feats.tolist()])
    return out


def _normalize(feats: Any) -> Any:
    """L2-normalize rows (zero rows stay zero — never NaN/inf)."""
    import torch

    norms = torch.sqrt(torch.sum(feats * feats, dim=1, keepdim=True))
    return feats / norms.clamp(min=1e-12)


def _check_contract(contract_doc: Mapping[str, Any], arch: Any) -> None:
    """Consumer-side contract verification (12 §15.3 — verify BEFORE
    execution): the contract must declare this model's image input, and
    a declared resize must be the architecture's — a drifted contract is
    a BLOCK, never a silent re-interpretation."""
    contract = (contract_doc or {}).get("inference_contract") or {}
    declared = None
    for item in contract.get("input_schema") or []:
        if isinstance(item, dict) and item.get("type") == "image":
            declared = item
            break
    if declared is None:
        raise ValidationBlock(
            "re-ID contract declares no `image` input_schema entry — "
            "cannot execute (12 §15.3)",
            hint="re-export the model so the contract records its image "
            "input (`mlforge export <MODEL> --format onnx`)",
        )
    resize = declared.get("resize")
    if not isinstance(resize, dict):
        return
    if resize.get("height") is None or resize.get("width") is None:
        return
    try:
        want = (int(resize["height"]), int(resize["width"]))
    except (TypeError, ValueError) as exc:
        raise ValidationBlock(
            f"contract resize {resize!r} is not height/width integers",
            hint="the contract at model_spec.json is the authority "
            "(12 §15.3) — re-export to rebuild it",
        ) from exc
    got = (int(arch.input_hw[0]), int(arch.input_hw[1]))
    if want != got:
        raise ValidationBlock(
            f"contract declares resize {want[0]}x{want[1]} (HxW) but this "
            f"model runs {got[0]}x{got[1]}",
            hint="contract and weights disagree (12 §15.3) — re-export the "
            "model so the contract is rebuilt from the architecture",
        )


def contract(
    entry: Mapping[str, Any],
    fmt: str,
    *,
    semantic: Mapping[str, Any] | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """First-export `model_spec.json` for the re-ID model (12 §15.3) —
    real image schema (the trainer's preprocessing), real output schema
    (the `execute` result), real operator list, real runtime.

    `semantic` is accepted for engine-signature parity; the architecture
    identity comes from the model NAME (the same key the trainer and the
    registry use — 11 §M5), never from a mutable semantic field."""
    from mlforge.ops.exporting import FORMATS
    from mlforge.trainers.reid import _IM_MEAN, _IM_STD

    arch = _arch_for(entry)
    h, w = int(arch.input_hw[0]), int(arch.input_hw[1])
    ops = list(OPERATORS)
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
                    "resize": {"width": w, "height": h, "mode": "bilinear"},
                    "normalization": {
                        "mean": list(_IM_MEAN),
                        "std": list(_IM_STD),
                    },
                    "preprocess": "mlforge.trainers.reid inference "
                    "transform (11 §M5) — the engine reuses it verbatim",
                }
            ],
            "output_schema": [
                {
                    "name": "embedding",
                    "type": "structured",
                    "schema": "json_schema://mlforge.reid.embedding.v1",
                    "fields": [
                        {
                            "name": "embedding",
                            "dtype": "float",
                            "length": int(arch.feature_dim),
                            "normalized": "l2",
                        },
                        {"name": "dim", "dtype": "int"},
                        {"name": "raw_norm", "dtype": "float"},
                        {"name": "model", "dtype": "str"},
                    ],
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
            return (
                "onnx export needs the onnx + onnxruntime packages "
                f"({type(exc).__name__}: {exc}) — pip install onnx "
                "onnxruntime"
            )
        return None
    return (
        f"the {fmt} converter is not integrated for the re-ID model "
        "(13 §4.1 names ONNX as the portable target for exported models) "
        "— use --format onnx"
    )


def execute(
    *,
    entry: Mapping[str, Any],
    contract_doc: Mapping[str, Any],
    checked: Mapping[str, Any],
    weights: bytes,
) -> dict[str, Any]:
    """Real one-shot embedding — the checked image runs through OSNet.

    Model blob first (family pattern: a blob that is not this family's
    state_dict is refused), then the contract check (12 §15.3), then the
    pixels — an image that does not decode BLOCKs with a hint (13 §6.8).
    """
    model, arch = _load_model(weights, entry)
    _check_contract(contract_doc, arch)
    try:
        path = Path(str(checked["input"]["path"]))
    except (TypeError, KeyError, IndexError) as exc:
        raise ValidationBlock(
            "infer input was not checked — expected "
            "`checked['input']['path']`",
            hint="13 §6.8: validate → contract check → execute — run the "
            "input through `mlforge.ops.infer.check_input` first",
        ) from exc
    if not path.is_file():
        raise ValidationBlock(
            f"input image not found: {path}",
            hint="one-shot infer takes a single existing file (13 §6.8)",
        )
    vec = _encode_paths(model, [path], arch=arch, where="input")[0]
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        raise ValidationBlock(
            f"model {entry.get('name')} produced a zero-length embedding",
            hint="refusing to normalize an all-zero vector — never emit a "
            "NaN result (13 §6.9: no unverifiable output)",
        )
    return {
        "embedding": [round(v / norm, 6) for v in vec],
        "dim": len(vec),
        "raw_norm": round(norm, 6),
        "model": str(entry.get("name")),
    }


def _junk(pid: str) -> bool:
    """Market1501 junk identities — `0000` / `-1` distractor detections
    are neither ranked nor counted as ground truth."""
    try:
        return int(pid) <= 0
    except (TypeError, ValueError):
        return False


def _split_records(
    records: list[Any], base: Path, ref: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepared `reid_crops` records → validated query/gallery samples.

    Row rules mirror `load_reid_samples`: `relative_path` + `identity`
    are mandatory for rows we evaluate, `camera` must be an integer, and
    images must exist at the resolved path (existence only — full content
    hashing stays the gate's job, as the trainer documents)."""
    queries: list[dict[str, Any]] = []
    gallery: list[dict[str, Any]] = []
    splits: dict[str, int] = {}
    missing: list[str] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValidationBlock(
                f"record {i}: expected a re-ID record object, got "
                f"{type(rec).__name__}"
            )
        split = str(rec.get("split") or "").strip().lower()
        key = split or "(none)"
        splits[key] = splits.get(key, 0) + 1
        if split == _QUERY_SPLIT:
            role = "query"
        elif split in _GALLERY_SPLITS:
            role = "gallery"
        else:
            continue  # train/none rows are never eval rows (12 §15.4)
        rel = rec.get("relative_path")
        ident = rec.get("identity")
        if not rel or ident is None or not str(ident).strip():
            raise ValidationBlock(
                f"record {i}: re-ID records need `relative_path` and "
                f"`identity` (relative_path={rel!r}, identity={ident!r})",
                hint="prepare with the `reid_crops` transform — identity is "
                "parsed from the Market1501 filename, never guessed (02 §5)",
            )
        raw_cam = rec.get("camera")
        try:
            cam = int(raw_cam or 0)
        except (TypeError, ValueError):
            raise ValidationBlock(
                f"record {i}: camera {raw_cam!r} is not an integer"
            ) from None
        path = base / str(rel)
        if not path.is_file():
            missing.append(str(path))
            continue
        sample = {"path": path, "pid": str(ident), "cam": cam}
        if role == "query":
            queries.append(sample)
        else:
            gallery.append(sample)
    if missing:
        raise ValidationBlock(
            f"{len(missing)} re-ID image(s) missing at the resolved path "
            f"(first: {missing[0]}) — the dataset content changed since "
            "registration",
            hint="re-register with `mlforge dataset add --force` (12 §7: "
            "nothing runs on unresolved or changed data)",
        )
    seen = (
        ", ".join(f"{k}×{v}" for k, v in sorted(splits.items())) or "(none)"
    )
    if not queries:
        raise ValidationBlock(
            f"dataset {ref} has no query records — re-ID evaluation ranks "
            f"query images against a gallery (recorded splits: {seen})",
            hint="prepare a Market1501-style `query/` folder with the "
            "`reid_crops` transform for the eval split (02 §5, 12 §7)",
        )
    if not gallery:
        raise ValidationBlock(
            f"dataset {ref} has no gallery records — re-ID evaluation ranks "
            f"query images against a gallery (recorded splits: {seen})",
            hint="prepare `bounding_box_test/` (or `test/`) as the gallery "
            "with the `reid_crops` transform (02 §5, 12 §7)",
        )
    return queries, gallery


def _rank_metrics(
    queries: list[dict[str, Any]],
    gallery: list[dict[str, Any]],
    sims: Any,
) -> dict[str, float]:
    """Market1501 ranking protocol over the cosine similarity matrix.

    Per query: drop 0000/-1 distractors, drop same-identity+same-camera
    items (they are the trivial match), rank the rest by descending
    similarity (ties broken by record index — deterministic). Rank-k =
    a correct match inside the top k; AP = (mean precision at each correct
    match) / #ground truth; mAP = mean AP. A query with no usable ground
    truth contributes nothing (standard); if none survive, the evaluation
    refuses rather than reporting a silent 0.0.
    """
    used = 0
    hit1 = 0.0
    hit5 = 0.0
    aps: list[float] = []
    for qi, q in enumerate(queries):
        if _junk(q["pid"]):
            continue
        row = sims[qi].tolist()
        order = sorted(
            (
                gi
                for gi, g in enumerate(gallery)
                if not _junk(g["pid"])
                and not (g["pid"] == q["pid"] and g["cam"] == q["cam"])
            ),
            key=lambda gi: (-row[gi], gi),
        )
        if not order:
            continue
        matches = [gallery[gi]["pid"] == q["pid"] for gi in order]
        total = sum(matches)
        if total == 0:
            continue  # no ground truth for this query after junk removal
        used += 1
        if matches[0]:
            hit1 += 1.0
        if any(matches[:5]):
            hit5 += 1.0
        found = 0
        precisions: list[float] = []
        for pos, matched in enumerate(matches, start=1):
            if matched:
                found += 1
                precisions.append(found / pos)
        aps.append(sum(precisions) / total)
    if used == 0:
        raise ValidationBlock(
            "re-ID evaluation found no usable query→gallery pairs — every "
            "query lacks a gallery match after junk removal (same-identity/"
            "same-camera items and 0000/-1 distractors are never matches)",
            hint="evaluate a Market1501-style query/ + bounding_box_test/ "
            "pair whose identities overlap (02 §5; 06_benchmarking_plan.md "
            "Rank-1 > 95%, mAP > 85%)",
        )
    return {
        "rank1": round(hit1 / used, 6),
        "rank5": round(hit5 / used, 6),
        "mAP": round(sum(aps) / len(aps), 6),
    }


def compute_metrics(
    *,
    entry: Mapping[str, Any],
    weights: bytes,
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    semantic: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Real Rank-1 / Rank-5 / mAP over prepared `reid_crops` records
    (13 §6.7: the harness runs over the dataset) — the standard
    Market1501 query→gallery protocol with the trainer's own loading
    rules and the trainer's own inference preprocessing.

    Deterministic by construction: no RNG anywhere in this path, so the
    same records + weights + protocol always produce the same numbers
    whatever `protocol.seed` is; `protocol.sample_subset` selects the
    FIRST N queries — never a random sample.

    Guards fire before any model work (13 §6.7: a wrong transform or an
    unresolved dataset BLOCKs — no weights are even loaded)."""
    transform = dataset.get("transform")
    if transform and str(transform) != "reid_crops":
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} was prepared with transform "
            f"{transform!r} — the re-ID engine measures Rank-1/Rank-5/mAP "
            "over `reid_crops` query + gallery image records (identity + "
            "camera)",
            hint="mlforge prepare <MODEL> with the reid_crops transform for "
            "the eval split (12 §7)",
        )
    records = dataset.get("records")
    if not records:
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} has no prepared records — the "
            "re-ID engine needs `reid_crops` query + gallery records",
            hint="mlforge prepare <MODEL> registers the eval split as a "
            "store artifact first (12 §7: nothing runs on unresolved data)",
        )
    base = dataset.get("path")
    if not base:
        raise ValidationBlock(
            f"dataset {dataset.get('ref')} has no machine-local path — the "
            "re-ID engine reads the registered image files",
            hint="mlforge dataset add <NAME> <PATH> — paths are explicit, "
            "never discovered (12 §6.3)",
        )
    ref = dataset.get("ref") or dataset.get("name")
    queries, gallery = _split_records(list(records), Path(str(base)), ref)
    subset = protocol.get("sample_subset")
    if subset is not None:
        if (
            isinstance(subset, bool)
            or not isinstance(subset, int)
            or subset < 1
        ):
            raise ValidationBlock(
                f"protocol.sample_subset must be a positive integer "
                f"(first N queries), got {subset!r}"
            )
        queries = queries[:subset]
        # `_split_records` guarantees a non-empty query list, and subset
        # >= 1 — the slice can never empty it (asserted in the tests).

    import torch

    model, arch = _load_model(weights, entry)
    paths = [q["path"] for q in queries] + [g["path"] for g in gallery]
    vectors = _encode_paths(model, paths, arch=arch, where="query/gallery")
    q_vecs = vectors[: len(queries)]
    g_vecs = vectors[len(queries) :]
    feats = torch.tensor(q_vecs + g_vecs, dtype=torch.float64)
    norms = feats.norm(dim=1)
    if float(norms.max()) < 1e-12:
        raise ValidationBlock(
            f"model {entry.get('name')} produced only zero-length "
            "embeddings on this dataset",
            hint="refusing to rank vectors that carry no information — "
            "never report a meaningless number (13 §6.7: a real harness "
            "or an honest refusal)",
        )
    n_q = len(queries)
    q_unit = _normalize(feats[:n_q])
    g_unit = _normalize(feats[n_q:])
    sims = q_unit @ g_unit.T
    return _rank_metrics(queries, gallery, sims)


# -- export: embedding forward → ONNX ----------------------------------------


def _build_onnx(model: Any, arch: Any, entry: Mapping[str, Any]) -> bytes:
    """The embedding forward (embed + L2 normalize) as a checked ONNX
    proto — the graph produces exactly what `execute` returns, so the
    exported artifact IS the contract's output (12 §15.3)."""
    import onnx
    import torch

    h, w = int(arch.input_hw[0]), int(arch.input_hw[1])

    class _EmbeddingGraph(torch.nn.Module):
        def __init__(self, net: Any) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: Any) -> Any:
            return _normalize(self.net.embed(x))

    graph = _EmbeddingGraph(model)
    graph.eval()
    idx = torch.arange(3 * h * w, dtype=torch.float32)
    dummy = ((idx * 17.0 + 37.0) % 251.0).div(255.0).view(1, 3, h, w)
    buf = io.BytesIO()
    try:
        with torch.no_grad():
            torch.onnx.export(
                graph,
                dummy,
                buf,
                input_names=["image"],
                output_names=["embedding"],
                dynamic_axes={
                    "image": {0: "batch"},
                    "embedding": {0: "batch"},
                },
                opset_version=17,
                dynamo=False,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
            )
    except Exception as exc:
        raise ValidationBlock(
            f"reid export: ONNX export failed "
            f"({type(exc).__name__}: {exc}) — refusing to emit a graph we "
            "cannot verify (13 §7: never a partial export)",
        ) from exc
    finally:
        model.eval()  # torch.onnx.export leaves the module in TRAIN mode
    try:
        proto = onnx.load_from_string(buf.getvalue())
        proto.ir_version = 8  # portable IR level (onnxruntime-friendly)
        onnx.checker.check_model(proto)
    except Exception as exc:
        raise ValidationBlock(
            f"reid export: generated ONNX graph failed the checker ({exc}) "
            "— refusing to emit a graph we cannot verify (13 §7: never a "
            "partial export)",
        ) from exc
    return proto.SerializeToString()


def _probe_batch(arch: Any, n: int = 4) -> Any:
    """Deterministic probe images for the numerical round-trip — no RNG
    (same bytes ⇒ same validation, 12 §15.4 spirit)."""
    import torch

    h, w = int(arch.input_hw[0]), int(arch.input_hw[1])
    idx = torch.arange(3 * h * w, dtype=torch.float64)
    rows = [((idx * 17.0 + 37.0 * (k + 1)) % 251.0) / 255.0 for k in range(n)]
    return torch.stack(
        [row.to(torch.float32).view(1, 3, h, w).squeeze(0) for row in rows]
    )


def _roundtrip(model: Any, arch: Any, proto: bytes) -> dict[str, Any]:
    """torch vs onnxruntime on the probe images (13 §6.9: max error vs
    tolerance → PASS/FAIL — a FAIL blocks the export)."""
    import onnxruntime as ort
    import torch

    probe = _probe_batch(arch)
    model.eval()  # export leaves the module in train mode — never that
    with torch.no_grad():
        expected = _normalize(model.embed(probe)).numpy()
    session = ort.InferenceSession(proto, providers=["CPUExecutionProvider"])
    got = session.run(None, {"image": probe.numpy()})[0]
    max_error = float(abs(expected - got).max())
    return {
        "max_error": round(max_error, 10),
        "tolerance": ATOL,
        "result": "PASS" if max_error < ATOL else "FAIL",
        "harness": "torch-onnx-roundtrip",
        "n_probe": int(probe.shape[0]),
    }


def export_bytes(
    *,
    entry: Mapping[str, Any],
    weights: bytes,
    fmt: str,
) -> tuple[bytes, dict[str, Any]]:
    """Real export: embedding graph → ONNX bytes + round-trip validation.
    Returns (binary, numerical_validation); the workflow blocks on FAIL."""
    if fmt != "onnx":
        raise PreconditionFailed(
            f"cannot export the re-ID model to {fmt} — {supports_format(fmt)}",
            hint="13 §6.9: only formats with a real converter produce bytes; "
            "nothing partial is ever written (13 §7)",
        )
    err = supports_format(fmt)
    if err:
        raise PreconditionFailed(f"cannot export the re-ID model — {err}")
    model, arch = _load_model(weights, entry)
    proto = _build_onnx(model, arch, entry)
    numerical = _roundtrip(model, arch, proto)
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
