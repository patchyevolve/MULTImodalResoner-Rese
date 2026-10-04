"""Tier-3 real REID engine — evaluate / infer / export with NO harness
(12 §12.4): tiny REAL OSNet x1_0 weights are built in-test from
`MODEL_REGISTRY` (seeded random init — no download, no pretrained), then
every seam runs against those weights:

  * family surface (FAMILY / MODEL_NAMES / input_types / metric_names)
    and honest format refusals naming ONNX as the portable target
    (13 §4.1);
  * contract() → model_spec with the trainer's image schema, the execute
    output schema, and an operator list `validate_export` accepts (13
    §6.9, 12 §15.3);
  * infer → real 512-d L2 embedding; contract drift, unchecked input and
    undecodable pixels all BLOCK; preprocess is byte-identical to the
    trainer's own `_gather` (13 §6.8);
  * evaluate → Market1501 Rank-1/Rank-5/mAP with KNOWN values over
    crafted embeddings (perfect separation ⇒ 1.0; one deliberate miss ⇒
    rank1 0.5 / mAP 0.75), plus the real encode path and its guards
    (13 §6.7);
  * export → genuine ONNX bytes (ir_version 8 head), PASS round-trip,
    graph ops ⊆ contract operators, non-onnx refused (13 §6.9).

The harness gate is the trainers' gate: the conftest opt-in is lifted
for every test here, so a seam silently regressing to the scaffold
breaks these tests. Deterministic (seeded init + generated images),
CPU-only, no registry edits (engine functions are called directly).
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import pytest

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.ops.engines import harness_active
from mlforge.ops.engines import reid as reid_engine

#: Handcrafted registry entry — the engine resolves the architecture from
#: `entry["name"]` exactly as the workflow passes it.
ENTRY = {"name": "osnet_x1_0", "version": "v1", "model_id": "m_reid_test"}

#: The trainer's own semantic identity for the parity fixture (11 §M5).
REID_SEM = {
    "optimizer": "adamw",
    "learning_rate": 3e-3,
    "scheduler": "cosine",
    "loss": "cross_entropy",
    "seed": 42,
    "global_batch": 4,
    "epochs": 1,
    "precision_policy": "fp32",
    "pretrained": False,  # never a hidden download in unit tests
}


# -- fixtures / helpers ------------------------------------------------------


@pytest.fixture(autouse=True)
def real_path(monkeypatch):
    """These tests must never see the scaffold harness (12 §12.4)."""
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    assert harness_active() is False


@pytest.fixture(scope="module")
def weights() -> bytes:
    """REAL tiny weights: the registry architecture, seeded random init,
    `torch.save(state_dict)` — the same bytes shape
    `ReidTrainer.checkpoint_payload()['model']` holds (no download)."""
    import torch

    from mlforge.trainers.reid import MODEL_REGISTRY, OSNet

    torch.manual_seed(7)
    model = OSNet(MODEL_REGISTRY["osnet_x1_0"], 3)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    blob = buf.getvalue()
    assert len(blob) > 1_000_000  # 567 tensors / 2.19M backbone params
    return blob


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """Machine-local dataset paths live outside the workspace (13 §10)."""
    h = tmp_path / "mlforge-home"
    monkeypatch.setenv("MLFORGE_HOME", str(h))
    return h


def _write_image(path: Path, *, seed: int = 0) -> Path:
    """Deterministic 40×20 JPEG (varying pixels per seed — never a
    shared fixture file)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 40, 20
    im = Image.new("RGB", (w, h))
    px = im.load()
    for i in range(w):
        for j in range(h):
            px[i, j] = (
                (i * 7 + seed) % 256,
                (j * 11 + seed * 3) % 256,
                (i * j + seed * 5) % 256,
            )
    im.save(path, "JPEG", quality=85)
    return path


def _rec(split: str, rel: str, pid: str, cam: int) -> dict:
    """One prepared `reid_crops` record (transform's own shape, 12 §10.2)."""
    return {
        "source": "market:eval",
        "split": split,
        "relative_path": rel,
        "identity": pid,
        "camera": cam,
    }


def _prepared(tmp_path: Path, records: list[dict],
              *, transform: str | None = "reid_crops") -> dict:
    """Eval dataset mapping with the shape
    `WorkflowAPI._resolve_eval_dataset` hands the engine, plus the image
    files its records point at."""
    base = tmp_path / "market"
    base.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(records):
        _write_image(base / str(rec["relative_path"]), seed=i + 1)
    return {
        "name": "market",
        "version": "v1",
        "ref": "market:v1",
        "identity": "c" * 8,
        "path": str(base),
        "records": records,
        "transform": transform,
    }


def _protocol(**over) -> dict:
    """The evaluation protocol keys `compute_metrics` reads (12 §15.4)."""
    proto = {
        "split": "val",
        "sample_subset": None,
        "seed": 123456,
        "aggregation": "macro",
    }
    proto.update(over)
    return proto


def _crafted(vectors: dict[str, list[float]]):
    """Encode seam stand-in: file stem → crafted vector.

    The RANK MATH is what these tests pin down (the real forward is
    covered by test_execute_returns_unit_embedding and
    test_compute_metrics_real_encode_deterministic); the model still
    loads from the real weights first, so the weight path stays real."""

    def encode(model, paths, *, arch, where, batch=16):
        out = []
        for p in paths:
            key = Path(p).stem
            if key not in vectors:
                raise AssertionError(f"no crafted vector for {key!r}")
            out.append(list(vectors[key]))
        return out

    return encode


def _reid_project(root: Path) -> None:
    """Registered + store-backed prepared `reid_crops` dataset + the
    machine-local image path (what `dataset add` + `prepare` leave
    behind) — enough to construct the real trainer for parity."""
    import json

    from mlforge.ingest import config as ingest_config
    from mlforge.store import ContentStore

    tree = Path(root) / "market"
    records = []
    for pid in (1, 2, 3):
        for i in range(4):
            rel = f"{pid}_c1_{i:04d}.jpg"
            _write_image(tree / rel, seed=pid * 10 + i)
            records.append(
                {
                    "source": "market:train",
                    "split": "train",
                    "relative_path": rel,
                    "identity": str(pid),
                    "camera": 1,
                }
            )
    doc = {
        "artifact_schema": "mlforge.prepared_dataset.v1",
        "model": "osnet_x1_0",
        "transform": {"name": "reid_crops"},
        "records": records,
    }
    ident = ContentStore(Path(root) / "store").put_bytes(
        json.dumps(doc, sort_keys=True).encode("utf-8")
    )
    d = Path(root) / "datasets" / "market"
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity.json").write_text(
        json.dumps(
            {
                "dataset_id": "market",
                "identity": ident,
                "version": "v1",
                "file_count": len(records),
                "total_bytes": 1,
            }
        ),
        encoding="utf-8",
    )
    ingest_config.set_path(root, "market", tree)


# -- family surface / refusals ------------------------------------------------


def test_dependency_error_is_none():
    """torch + pillow are here — the engine is runnable (the trainer's
    own dependency probe, so the two cannot disagree)."""
    assert reid_engine.dependency_error() is None


def test_family_surface_matches_the_trainer_and_the_benchmark():
    from mlforge.trainers.reid import MODEL_REGISTRY

    assert reid_engine.FAMILY == "reid"
    # drift alarm: the engine's model set IS the trainer's registry
    assert reid_engine.MODEL_NAMES == frozenset(MODEL_REGISTRY)
    assert reid_engine.input_types() == frozenset({"image"})
    assert reid_engine.metric_names() == ("rank1", "rank5", "mAP")


def test_supports_format_honest_refusals():
    assert reid_engine.supports_format("onnx") is None
    for fmt in ("openvino", "tflite", "tensorrt"):
        msg = reid_engine.supports_format(fmt)
        assert msg is not None
        assert fmt in msg and "onnx" in msg  # names the portable target


# -- contract (12 §15.3) --------------------------------------------------


def test_contract_structure_is_family_real():
    from mlforge.ops.exporting import validate_export
    from mlforge.trainers.reid import _IM_MEAN, _IM_STD

    spec = reid_engine.contract(ENTRY, "onnx")
    assert spec["schema_version"] == 1
    assert spec["name"] == ENTRY["name"]
    assert spec["version"] == ENTRY["version"]
    assert spec["harness"] == "reid"
    contract = spec["inference_contract"]

    inp = contract["input_schema"][0]
    assert inp["name"] == "image" and inp["type"] == "image"
    assert inp["color_space"] == "RGB" and inp["layout"] == "HWC"
    assert inp["resize"] == {"width": 128, "height": 256, "mode": "bilinear"}
    assert inp["normalization"] == {
        "mean": list(_IM_MEAN),
        "std": list(_IM_STD),
    }

    out = contract["output_schema"][0]
    assert out["name"] == "embedding" and out["type"] == "structured"
    fields = {f["name"]: f for f in out["fields"]}
    assert fields["embedding"]["length"] == 512
    assert fields["embedding"]["normalized"] == "l2"
    assert {"dim", "raw_norm", "model"} <= set(fields)

    ops = contract["operators"]
    assert ops == list(reid_engine.OPERATORS) and len(ops) > 0
    assert isinstance(contract["operator_set"], str)
    assert contract["runtime"] == {"framework": "onnx", "opset": 17}
    assert contract["dtype"] == "fp32"
    assert contract["dynamic_axes"] == {"batch": [0]}
    assert contract["numerical_tolerance"] == {"atol": reid_engine.ATOL}
    # the operator list must survive the exporter's own gate (13 §6.9)
    assert validate_export(spec, "onnx") == {
        "architecture": "PASS",
        "operators": "PASS",
        "dynamic_shapes": "PASS",
    }
    # precision flows through, semantic is signature parity only
    fp16 = reid_engine.contract(ENTRY, "onnx", semantic={"seed": 42},
                                precision="fp16")
    assert fp16["inference_contract"]["dtype"] == "fp16"


def test_contract_unknown_model_blocks():
    with pytest.raises(ValidationBlock) as ei:
        reid_engine.contract({"name": "nope", "version": "v1"}, "onnx")
    assert "no registered re-ID architecture" in str(ei.value)
    assert "osnet_x1_0" in (ei.value.hint or "")


# -- infer (13 §6.8) -------------------------------------------------------


def test_execute_returns_unit_embedding(tmp_path, weights):
    spec = reid_engine.contract(ENTRY, "onnx")
    img = _write_image(tmp_path / "probe.jpg", seed=3)
    out = reid_engine.execute(
        entry=ENTRY,
        contract_doc=spec,
        checked={"input": {"path": str(img)}},
        weights=weights,
    )
    assert set(out) == {"embedding", "dim", "raw_norm", "model"}
    assert out["dim"] == 512 and len(out["embedding"]) == 512
    assert out["model"] == "osnet_x1_0"
    assert all(math.isfinite(float(v)) for v in out["embedding"])
    assert out["raw_norm"] > 0.0
    norm = math.sqrt(sum(v * v for v in out["embedding"]))
    assert abs(norm - 1.0) < 1e-3  # L2-normalized (6-decimal rounding)

    again = reid_engine.execute(
        entry=ENTRY,
        contract_doc=spec,
        checked={"input": {"path": str(img)}},
        weights=weights,
    )
    assert again == out  # same input ⇒ same result (deterministic)


def test_execute_blocks_undecodable_image(tmp_path, weights):
    spec = reid_engine.contract(ENTRY, "onnx")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(ValidationBlock) as ei:
        reid_engine.execute(
            entry=ENTRY,
            contract_doc=spec,
            checked={"input": {"path": str(bad)}},
            weights=weights,
        )
    out = ei.value.render()
    assert "cannot decode" in out and "bad.png" in out
    assert "13 §6.8" in (ei.value.hint or "")


def test_execute_blocks_unchecked_or_missing_input(tmp_path, weights):
    spec = reid_engine.contract(ENTRY, "onnx")
    with pytest.raises(ValidationBlock) as ei:
        reid_engine.execute(entry=ENTRY, contract_doc=spec, checked={},
                            weights=weights)
    assert "not checked" in str(ei.value)

    with pytest.raises(ValidationBlock) as ei2:
        reid_engine.execute(
            entry=ENTRY,
            contract_doc=spec,
            checked={"input": {"path": str(tmp_path / "nope.jpg")}},
            weights=weights,
        )
    assert "not found" in str(ei2.value)


def test_execute_blocks_contract_drift(tmp_path, weights):
    """Consumer-side verification (12 §15.3): a contract that disagrees
    with the weights' architecture BLOCKs, never silently re-interprets."""
    spec = reid_engine.contract(ENTRY, "onnx")
    img = _write_image(tmp_path / "drift.jpg", seed=4)

    drifted = json_deepcopy(spec)
    drifted["inference_contract"]["input_schema"][0]["resize"] = {
        "width": 64, "height": 64, "mode": "bilinear"
    }
    with pytest.raises(ValidationBlock) as ei:
        reid_engine.execute(
            entry=ENTRY, contract_doc=drifted,
            checked={"input": {"path": str(img)}}, weights=weights,
        )
    assert "contract declares resize 64x64" in str(ei.value)
    assert "256x128" in str(ei.value)

    foreign = json_deepcopy(spec)
    foreign["inference_contract"]["input_schema"][0]["type"] = "structured"
    with pytest.raises(ValidationBlock) as ei2:
        reid_engine.execute(
            entry=ENTRY, contract_doc=foreign,
            checked={"input": {"path": str(img)}}, weights=weights,
        )
    assert "no `image` input_schema entry" in str(ei2.value)


def json_deepcopy(doc: dict) -> dict:
    import json

    return json.loads(json.dumps(doc))


def test_preprocess_is_byte_identical_to_the_trainer(tmp_path, home):
    """Infer-time tensors ARE the trainer's own `_gather` output minus
    the training-only flip — byte-for-byte, not "close enough" (11 §M5
    parity: the eval numbers mean the same thing as train)."""
    import random

    import torch

    from mlforge.trainers.reid import MODEL_REGISTRY, ReidTrainer

    _reid_project(tmp_path)
    trainer = ReidTrainer(
        model_name="osnet_x1_0",
        semantic=dict(REID_SEM),
        runtime={},
        train_datasets=("market:v1",),
        root=tmp_path,
    )
    sample = trainer.samples[0]
    random.seed(0)  # first draw 0.844 ≥ 0.5 → the trainer's flip stays off
    batch, _labels = trainer._gather(torch.tensor([0]))
    ours = reid_engine._preprocess(
        sample["path"], MODEL_REGISTRY["osnet_x1_0"], where="input"
    )
    assert torch.equal(batch[0], ours)


# -- evaluate (13 §6.7) ----------------------------------------------------


def test_compute_metrics_perfect_separation(tmp_path, weights, monkeypatch):
    """Crafted embeddings with exact query→gallery correspondence ⇒ every
    metric is 1.0 (the Market1501 protocol arithmetic, pinned)."""
    records = [
        _rec("query", "q1.jpg", "1", 1),
        _rec("query", "q2.jpg", "2", 1),
        _rec("query", "q3.jpg", "3", 1),
        _rec("test", "g1.jpg", "1", 2),
        _rec("test", "g2.jpg", "2", 2),
        _rec("test", "g3.jpg", "3", 2),
    ]
    dataset = _prepared(tmp_path, records)
    vectors = {
        "q1": [1.0, 0.0, 0.0],
        "g1": [1.0, 0.0, 0.0],
        "q2": [0.0, 1.0, 0.0],
        "g2": [0.0, 1.0, 0.0],
        "q3": [0.0, 0.0, 1.0],
        "g3": [0.0, 0.0, 1.0],
    }
    monkeypatch.setattr(reid_engine, "_encode_paths", _crafted(vectors))
    metrics = reid_engine.compute_metrics(
        entry=ENTRY,
        weights=weights,
        dataset=dataset,
        protocol=_protocol(),
        semantic={"seed": 42},
    )
    assert metrics == {"rank1": 1.0, "rank5": 1.0, "mAP": 1.0}


def test_compute_metrics_known_miss_and_first_n_subset(
    tmp_path, weights, monkeypatch
):
    """q2 points at the wrong identity → its only match ranks 2nd:
    rank1 = 1/2, rank5 = 1, AP = 1/2 ⇒ mAP = 0.75. `sample_subset`
    takes the FIRST queries — never a random sample."""
    records = [
        _rec("query", "q1.jpg", "1", 1),
        _rec("query", "q2.jpg", "2", 1),
        _rec("test", "g1.jpg", "1", 2),
        _rec("test", "g2.jpg", "2", 2),
    ]
    dataset = _prepared(tmp_path, records)
    vectors = {
        "q1": [1.0, 0.0],
        "q2": [1.0, 0.0],  # aligns with g1 (pid 1) — wrong on purpose
        "g1": [1.0, 0.0],
        "g2": [0.0, 1.0],
    }
    monkeypatch.setattr(reid_engine, "_encode_paths", _crafted(vectors))
    metrics = reid_engine.compute_metrics(
        entry=ENTRY,
        weights=weights,
        dataset=dataset,
        protocol=_protocol(),
    )
    assert metrics == {"rank1": 0.5, "rank5": 1.0, "mAP": 0.75}

    subset = reid_engine.compute_metrics(
        entry=ENTRY,
        weights=weights,
        dataset=dataset,
        protocol=_protocol(sample_subset=1),
    )
    assert subset == {"rank1": 1.0, "rank5": 1.0, "mAP": 1.0}


def test_compute_metrics_real_encode_is_deterministic(tmp_path, weights):
    """The un-stubbed path: real OSNet forward over real images. The
    random-init weights cannot be asserted to rank anything in
    particular — what MUST hold is the contract: the exact metric keys,
    values inside [0, 1], and identical numbers for identical inputs
    whatever the protocol seed says."""
    records = [
        _rec("query", "q1.jpg", "1", 1),
        _rec("query", "q2.jpg", "2", 1),
        _rec("test", "g1.jpg", "1", 2),
        _rec("test", "g2.jpg", "2", 2),
        _rec("test", "g3.jpg", "1", 3),
    ]
    dataset = _prepared(tmp_path, records)
    first = reid_engine.compute_metrics(
        entry=ENTRY, weights=weights, dataset=dataset, protocol=_protocol()
    )
    assert set(first) == {"rank1", "rank5", "mAP"}
    for key, value in first.items():
        assert math.isfinite(value) and 0.0 <= value <= 1.0, (key, value)
    again = reid_engine.compute_metrics(
        entry=ENTRY,
        weights=weights,
        dataset=dataset,
        protocol=_protocol(seed=999),
    )
    assert again == first  # no RNG in the metric path — seed cannot drift


def test_compute_metrics_wrong_transform_blocks(tmp_path):
    """Guard fires BEFORE any weights are touched (13 §6.7) — even a
    garbage blob cannot reach the model."""
    dataset = _prepared(
        tmp_path,
        [_rec("query", "q1.jpg", "1", 1), _rec("test", "g1.jpg", "1", 2)],
        transform="tabular",
    )
    with pytest.raises(ValidationBlock) as ei:
        reid_engine.compute_metrics(
            entry=ENTRY,
            weights=b"not-a-checkpoint",
            dataset=dataset,
            protocol=_protocol(),
        )
    out = ei.value.render()
    assert "transform" in out and "tabular" in out
    assert "reid_crops" in out
    assert "mlforge prepare" in (ei.value.hint or "")


def test_compute_metrics_empty_records_block(tmp_path, weights):
    dataset = _prepared(tmp_path, [], transform="reid_crops")
    dataset["records"] = []
    with pytest.raises(ValidationBlock) as ei:
        reid_engine.compute_metrics(
            entry=ENTRY, weights=weights, dataset=dataset,
            protocol=_protocol(),
        )
    out = ei.value.render()
    assert "no prepared records" in out
    assert "mlforge prepare" in (ei.value.hint or "")


def test_compute_metrics_missing_query_or_gallery_blocks(tmp_path, weights):
    """Split discipline: re-ID evaluation is query → gallery, and the
    refusal names the splits it actually saw (12 §15.4 spirit)."""
    train_only = [
        _rec("train", "g1.jpg", "1", 1),
        _rec("train", "g2.jpg", "2", 1),
    ]
    with pytest.raises(ValidationBlock) as ei:
        reid_engine.compute_metrics(
            entry=ENTRY,
            weights=weights,
            dataset=_prepared(tmp_path, train_only),
            protocol=_protocol(),
        )
    out = ei.value.render()
    assert "no query records" in out and "train" in out

    query_only = [_rec("query", "q1.jpg", "1", 1)]
    with pytest.raises(ValidationBlock) as ei2:
        reid_engine.compute_metrics(
            entry=ENTRY,
            weights=weights,
            dataset=_prepared(tmp_path / "g", query_only),
            protocol=_protocol(),
        )
    assert "no gallery records" in str(ei2.value)


def test_compute_metrics_missing_image_and_bad_subset_block(
    tmp_path, weights
):
    records = [_rec("query", "q1.jpg", "1", 1), _rec("test", "g1.jpg", "1", 2)]
    dataset = _prepared(tmp_path, records)
    (Path(dataset["path"]) / "g1.jpg").unlink()  # content changed since
    with pytest.raises(ValidationBlock) as ei:
        reid_engine.compute_metrics(
            entry=ENTRY, weights=weights, dataset=dataset,
            protocol=_protocol(),
        )
    out = ei.value.render()
    assert "missing" in out and "g1.jpg" in out
    assert "dataset add --force" in (ei.value.hint or "")

    dataset = _prepared(tmp_path / "s", records)
    with pytest.raises(ValidationBlock) as ei2:
        reid_engine.compute_metrics(
            entry=ENTRY, weights=weights, dataset=dataset,
            protocol=_protocol(sample_subset=0),
        )
    assert "sample_subset" in str(ei2.value)


def test_compute_metrics_blocks_dataset_without_path(tmp_path, weights):
    records = [_rec("query", "q1.jpg", "1", 1), _rec("test", "g1.jpg", "1", 2)]
    dataset = _prepared(tmp_path, records)
    dataset["path"] = None
    with pytest.raises(ValidationBlock) as ei:
        reid_engine.compute_metrics(
            entry=ENTRY, weights=weights, dataset=dataset,
            protocol=_protocol(),
        )
    assert "no machine-local path" in str(ei.value)
    assert "dataset add" in (ei.value.hint or "")


# -- export (13 §6.9) ------------------------------------------------------


def test_export_bytes_real_onnx_and_roundtrip(tmp_path, weights):
    from mlforge.ops.exporting import validate_export

    spec = reid_engine.contract(ENTRY, "onnx")
    proto, numerical = reid_engine.export_bytes(
        entry=ENTRY, weights=weights, fmt="onnx"
    )
    # genuine bytes: portable IR8 header, not a placeholder
    assert isinstance(proto, bytes) and proto[:2] == b"\x08\x08"
    assert len(proto) > 1_000_000

    assert set(numerical) == {
        "max_error", "tolerance", "result", "harness", "n_probe",
    }
    assert numerical["result"] == "PASS"
    assert numerical["harness"] == "torch-onnx-roundtrip"
    assert float(numerical["max_error"]) <= float(numerical["tolerance"])
    assert float(numerical["tolerance"]) == reid_engine.ATOL
    assert numerical["n_probe"] >= 2

    import numpy as np
    import onnx
    import onnxruntime as ort

    graph = onnx.load_from_string(proto)
    assert graph.ir_version == 8
    ops = {n.op_type for n in graph.graph.node}
    # every real op is declared (no under-declared contract) ...
    assert ops <= set(reid_engine.OPERATORS)
    assert {"Conv", "Gemm", "BatchNormalization"} <= ops
    # ... and the declared list survives the exporter's schema gate
    assert validate_export(spec, "onnx")["operators"] == "PASS"
    shape_in = [
        d.dim_param or d.dim_value
        for d in graph.graph.input[0].type.tensor_type.shape.dim
    ]
    shape_out = [
        d.dim_param or d.dim_value
        for d in graph.graph.output[0].type.tensor_type.shape.dim
    ]
    assert shape_in == ["batch", 3, 256, 128]
    assert shape_out == ["batch", 512]

    # dynamic batch axis works beyond the traced batch of 1
    session = ort.InferenceSession(proto, providers=["CPUExecutionProvider"])
    out = session.run(
        None, {"image": np.zeros((2, 3, 256, 128), dtype=np.float32)}
    )[0]
    assert out.shape == (2, 512)


def test_export_bytes_refuses_other_formats(weights):
    for fmt in ("openvino", "tflite"):
        with pytest.raises(PreconditionFailed) as ei:
            reid_engine.export_bytes(entry=ENTRY, weights=weights, fmt=fmt)
        out = ei.value.render()
        assert fmt in out.lower()
        assert "onnx" in out.lower()  # refusal names the portable target
