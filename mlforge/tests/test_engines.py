"""Tier-3 real execution engines — evaluate / infer / export with NO
harness (12 §12.4): a module-scoped workspace trains real models once
(Worker + real trainers), then every seam runs against those weights:

  * registry + honest refusals (unknown model, unsupported format)
  * export → genuine ONNX bytes, passing numerical round-trip, family
    contract operators (13 §6.9)
  * infer → family execution, deterministic input identity, input-schema
    blocks (13 §6.8)
  * evaluate → family metric names, preview == recorded, the ranker's
    split discipline and transform guard (13 §6.7, 12 §15.4)

The harness gate is the trainers' gate: the conftest opt-in is lifted
for every test here, so a seam silently regressing to the scaffold
breaks these tests. Deterministic (seeded synthetic pools), CPU-only.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash_bytes
from mlforge.leases import RunLeaseManager, provide_run_lease
from mlforge.ops.engines import (
    availability_error,
    build_engine,
    engine_input_types,
    harness_active,
    load_weights,
    registered_models,
)
from mlforge.run_spec import RunSpec
from mlforge.runtime.worker import Worker
from mlforge.store import ContentStore
from mlforge.trainers.calibrator import CalibratorTrainer
from mlforge.trainers.gbdt import FEATURE_NAMES, GbdtRankerTrainer
from mlforge.validation import RESUME_GATE_STEPS, provide_pass
from mlforge.workflow import WorkflowAPI

GATE_KEYS = [s.id for s in RESUME_GATE_STEPS
             if s.id not in ("manifest", "schema")]

RANKER_SEM = {
    "optimizer": "gbdt", "learning_rate": 0.05, "scheduler": "constant",
    "loss": "lambdarank", "seed": 42, "global_batch": 64, "epochs": 6,
    "precision_policy": "fp32",
}
CALIB_SEM = {
    "optimizer": "none", "learning_rate": 1.0, "scheduler": "constant",
    "loss": "nll", "seed": 42, "global_batch": 1, "epochs": 2,
    "precision_policy": "fp32",
}


# -- fixtures / helpers ------------------------------------------------------


def _wf(root: Path) -> WorkflowAPI:
    """An operator environment with wired gate providers (the gate
    itself is not under test here — the seams behind it are)."""
    providers = {k: provide_pass(f"{k} verified") for k in GATE_KEYS}
    providers["lease"] = provide_run_lease(RunLeaseManager(root))
    return WorkflowAPI(root, gate_providers=providers)


def _artifact(root: Path, dataset: str, model: str, transform: str,
              rows: list[dict]) -> str:
    """Registered + store-backed prepared dataset (the shape
    `mlforge prepare` leaves behind — tests/test_gbdt_trainer)."""
    doc = {
        "artifact_schema": "mlforge.prepared_dataset.v1",
        "model": model,
        "transform": {"name": transform},
        "records": rows,
    }
    ident = ContentStore(root / "store").put_bytes(
        json.dumps(doc, sort_keys=True).encode("utf-8"))
    d = root / "datasets" / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity.json").write_text(json.dumps({
        "dataset_id": dataset, "identity": ident, "version": "v1",
        "file_count": len(rows), "total_bytes": 1,
    }), encoding="utf-8")
    return f"{dataset}:v1"


def _ranker_rows(*, groups: int, split: str, seed: int,
                 per: int = 6) -> list[dict]:
    """Synthetic tabular records — label correlated to r_score so the
    booster has real signal (same shape as the trainer tests)."""
    import random

    rng = random.Random(seed)
    rows: list[dict] = []
    counter = 0
    for gi in range(groups):
        for _ in range(per):
            data = {c: str(round(rng.random(), 3)) for c in FEATURE_NAMES}
            r = rng.random()
            data["r_score"] = str(round(r, 3))
            data["label"] = str(min(3, int(r * 4)))
            data["group"] = f"g{gi}"
            rows.append({"split": split, "relative_path": "g.csv",
                         "row": counter, "data": data})
            counter += 1
    return rows


def _calib_rows(n: int = 40, scale: float = 3.0, flip: float = 0.25,
                seed: int = 7) -> list[dict]:
    """Overconfident binary pool — temperature scaling has work to do."""
    import random

    rng = random.Random(seed)
    rows = []
    for i in range(n):
        pred = i % 2
        label = 1 - pred if rng.random() < flip else pred
        logits = [scale, -scale] if pred == 0 else [-scale, scale]
        rows.append({"logits": logits, "label": label})
    return rows


def _train(root: Path, trainer, datasets: tuple[str, ...],
           semantic: dict) -> str:
    """create_run → gate → Worker runs the REAL trainer to completion →
    model ref (version bumps per name, 13 §5.5). The trainer is passed
    EXPLICITLY: the module fixture must not depend on the session
    harness env (a function-scoped monkeypatch cannot back a module
    fixture)."""
    wf = _wf(root)
    h = wf.create_run(RunSpec(model=trainer.model_name,
                              train_datasets=datasets, semantic=semantic))
    report = wf.validate_run(h.run_id)
    assert not report.blocked, report.render()
    token = RunLeaseManager(root).status(h.run_id).session_token
    assert token, "gate should leave a lease session token"
    code = Worker(root, h.run_id, token, trainer=trainer,
                  poll_interval=0.0, heartbeat_interval=60.0,
                  checkpoint_interval=1).run()
    assert code == 0, f"worker exit {code}"
    assert wf.get_run_state(h.run_id) == "COMPLETED"
    entries = [m for m in wf.list_models() if m.get("run_id") == h.run_id]
    assert len(entries) == 1 and entries[0].get("artifact_hash")
    return f"{entries[0]['name']}:{entries[0]['version']}"


class _Ws:
    def __init__(self, root: Path, ranker: str, calib: str,
                 nocontract: str) -> None:
        self.root = root
        self.ranker = ranker
        self.calib = calib
        self.nocontract = nocontract  # calibrator v2 — never exported


@pytest.fixture(autouse=True)
def real_path(monkeypatch):
    """These tests must never see the scaffold harness (12 §12.4)."""
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    assert harness_active() is False


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> _Ws:
    """One workspace: a real trained ranker + calibrator (both exported
    so every seam test is order-independent), plus a third model that
    was never exported (no inference contract)."""
    root = tmp_path_factory.mktemp("engines")
    train_ref = _artifact(root, "ranker", "hypothesis_ranker", "tabular",
                          _ranker_rows(groups=16, split="train", seed=7))
    eval_ref = _artifact(root, "ranker_eval", "hypothesis_ranker", "tabular",
                         _ranker_rows(groups=8, split="valid", seed=99))
    preds_ref = _artifact(root, "preds", "calibrator", "calibration",
                          _calib_rows())
    ranker_ref = _train(
        root,
        GbdtRankerTrainer(model_name="hypothesis_ranker",
                          semantic=RANKER_SEM, runtime={},
                          train_datasets=(train_ref,), root=root),
        (train_ref,), RANKER_SEM)
    calib_ref = _train(
        root,
        CalibratorTrainer(model_name="calibrator", semantic=CALIB_SEM,
                          runtime={}, train_datasets=(preds_ref,),
                          root=root),
        (preds_ref,), CALIB_SEM)
    nocontract = _train(
        root,
        CalibratorTrainer(model_name="calibrator", semantic=CALIB_SEM,
                          runtime={}, train_datasets=(preds_ref,),
                          root=root),
        (preds_ref,), CALIB_SEM)
    # The conftest harness opt-in is function-scoped and cannot back a
    # module fixture — lift it around the exports so they are REAL
    # (binary + round-trip), then restore the session's value.
    saved = os.environ.pop("MLFORGE_HARNESS", None)
    try:
        wf = _wf(root)
        wf.export_model(ranker_ref, "onnx")
        wf.export_model(calib_ref, "onnx")
    finally:
        if saved is not None:
            os.environ["MLFORGE_HARNESS"] = saved
    assert eval_ref and preds_ref  # registered for the evaluate tests
    return _Ws(root, ranker_ref, calib_ref, nocontract)


def _entry(root: Path, ref: str) -> dict:
    name, _, ver = ref.partition(":")
    for m in _wf(root).list_models():
        if m.get("name") == name and m.get("version") == ver:
            return m
    raise AssertionError(f"model {ref} not registered")


# -- registry / gate ---------------------------------------------------------


def test_registry_lists_integrated_models():
    assert registered_models() == [
        "calibrator", "hypothesis_ranker", "osnet_x1_0", "reasoner_s",
        "rf_detr_l", "rf_detr_s", "rf_detr_seg_s",
    ]
    assert engine_input_types("hypothesis_ranker") == frozenset({"structured"})
    assert engine_input_types("calibrator") == frozenset({"structured"})


def test_unknown_model_refuses_honestly():
    with pytest.raises(PreconditionFailed) as ei:
        build_engine("some_future_model")
    out = ei.value.render()
    assert "no real engine" in out
    assert "runnable here" in out
    assert "hypothesis_ranker" in out and "calibrator" in out
    assert availability_error("some_future_model") is not None
    assert availability_error("hypothesis_ranker") is None  # lightgbm here


def test_harness_gate_matches_trainers(monkeypatch):
    assert harness_active() is False  # autouse delenv above
    monkeypatch.setenv("MLFORGE_HARNESS", "1")
    assert harness_active() is True


# -- weights -----------------------------------------------------------------


def test_load_weights_serves_committed_bytes(ws):
    entry = _entry(ws.root, ws.ranker)
    data = load_weights(ws.root, entry)
    assert isinstance(data, bytes) and data

    tampered = {**entry, "artifact_hash": "sha256:" + "0" * 64}
    with pytest.raises(PreconditionFailed) as ei:
        load_weights(ws.root, tampered)
    assert "does not match" in ei.value.render()

    ghost = {"name": "hypothesis_ranker", "version": "v9",
             "model_id": "ghost", "run_id": None}
    with pytest.raises(PreconditionFailed) as ei2:
        load_weights(ws.root, ghost)
    assert "no stored weights file" in ei2.value.render()


# -- export (13 §6.9) ---------------------------------------------------------


def test_export_onnx_ranker_real_bytes(ws):
    rec = _wf(ws.root).export_model(ws.ranker, "onnx")
    binary = ws.root / "exports" / rec["export_id"] / rec["file_name"]
    assert rec["harness"] == "ranker"
    assert rec["format"] == "onnx"
    nv = rec["numerical_validation"]
    assert nv["result"] == "PASS"
    assert float(nv["max_error"]) <= float(nv["tolerance"])
    assert set(rec["validations"].values()) == {"PASS"}
    assert binary.is_file() and binary.stat().st_size > 0
    head = binary.read_bytes()[:2]
    assert head == b"\x08\x08"  # ir_version = 8 (portable)
    assert content_hash_bytes(binary.read_bytes()) == \
        rec["payload"]["binary_hash"]
    spec = json.loads(
        (ws.root / "exports" / rec["export_id"] / "model_spec.json")
        .read_text(encoding="utf-8"))
    contract = spec["inference_contract"]
    assert "TreeEnsembleRegressor" in contract["operators"]
    assert contract["input_schema"][0]["type"] == "structured"


def test_export_onnx_calibrator_real_bytes(ws):
    rec = _wf(ws.root).export_model(ws.calib, "onnx")
    binary = ws.root / "exports" / rec["export_id"] / rec["file_name"]
    assert rec["harness"] == "calibrator"
    assert rec["numerical_validation"]["result"] == "PASS"
    assert binary.is_file() and binary.stat().st_size > 0
    spec = json.loads(
        (ws.root / "exports" / rec["export_id"] / "model_spec.json")
        .read_text(encoding="utf-8"))
    ops = spec["inference_contract"]["operators"]
    assert "Div" in ops and "Softmax" in ops


def test_export_refuses_real_conversions_not_integrated(ws):
    wf = _wf(ws.root)
    with pytest.raises(PreconditionFailed) as ei:
        wf.export_model(ws.ranker, "openvino")
    out = ei.value.render()
    assert "openvino" in out.lower()
    assert "onnx" in out.lower()  # the refusal names the portable target

    with pytest.raises(ValidationBlock) as ei2:
        wf.export_model(ws.ranker, "madeup")
    assert "unsupported export format" in ei2.value.render()


# -- infer (13 §6.8) ----------------------------------------------------------


def test_infer_ranker_structured_deterministic(ws, tmp_path):
    wf = _wf(ws.root)
    q = {"hypotheses": [
        {c: 0.5 for c in FEATURE_NAMES} | {"r_score": 0.9},
        {c: 0.1 for c in FEATURE_NAMES} | {"r_score": 0.2},
    ]}
    p = tmp_path / "query.json"
    p.write_text(json.dumps(q), encoding="utf-8")
    out = wf.infer_model(ws.ranker, str(p))
    assert out["harness"] == "ranker"
    scores = out["result"]["scores"]
    assert len(scores) == 2
    assert all(math.isfinite(float(s)) for s in scores)

    again = wf.infer_model(ws.ranker, str(p))
    assert again["input"]["identity"] == out["input"]["identity"]

    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hypotheses": [q["hypotheses"][0]]}),
                     encoding="utf-8")
    diff = wf.infer_model(ws.ranker, str(other))
    assert diff["input"]["identity"] != out["input"]["identity"]


def test_infer_calibrator_scales_logits(ws, tmp_path):
    wf = _wf(ws.root)
    p = tmp_path / "logits.json"
    p.write_text(json.dumps({"logits": [3.0, -3.0]}), encoding="utf-8")
    out = wf.infer_model(ws.calib, str(p))
    assert out["harness"] == "calibrator"
    probs = [float(x) for x in out["result"]["probs"]]
    assert len(probs) == 2
    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_infer_input_schema_blocks(ws, tmp_path):
    wf = _wf(ws.root)

    bad = tmp_path / "bad.json"
    bad.write_text("not json {", encoding="utf-8")
    with pytest.raises(ValidationBlock) as ei:
        wf.infer_model(ws.ranker, str(bad))
    assert "valid JSON" in ei.value.render()

    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValidationBlock) as ei2:
        wf.infer_model(ws.ranker, str(arr))
    assert "JSON object" in ei2.value.render()

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"hypotheses": []}), encoding="utf-8")
    with pytest.raises(ValidationBlock) as ei3:
        wf.infer_model(ws.ranker, str(empty))
    assert "non-empty" in ei3.value.render()

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValidationBlock) as ei4:
        wf.infer_model(ws.ranker, str(missing))
    assert "missing feature" in ei4.value.render()


def test_infer_without_contract_blocks(ws, tmp_path):
    """A model never exported has no contract — infer BLOCKS before any
    execution (12 §15.3: contract required, harness or not)."""
    p = tmp_path / "logits.json"
    p.write_text(json.dumps({"logits": [1.0, -1.0]}), encoding="utf-8")
    with pytest.raises(ValidationBlock) as ei:
        _wf(ws.root).infer_model(ws.nocontract, str(p))
    out = ei.value.render()
    assert "no inference contract" in out
    assert "mlforge export" in out


# -- evaluate (13 §6.7) -------------------------------------------------------


def test_evaluate_ranker_preview_equals_record(ws):
    wf = _wf(ws.root)
    ref = "ranker_eval:v1"
    prev = wf.preview_evaluation(ws.ranker, ref)
    rec = wf.evaluate_model(ws.ranker, ref)
    assert prev["identity"] == rec["identity"]
    assert prev["metrics"] == rec["metrics"]
    assert set(rec["metrics"]) == {"ndcg@3", "ndcg@5"}
    assert rec["harness"] == "ranker"
    assert all(0.0 <= float(v) <= 1.0 for v in rec["metrics"].values())


def test_evaluate_calibrator_metric_family(ws):
    rec = _wf(ws.root).evaluate_model(ws.calib, "preds:v1")
    assert set(rec["metrics"]) == {"ece", "nll", "coverage"}
    assert rec["harness"] == "calibrator"
    ece = float(rec["metrics"]["ece"])
    assert 0.0 <= ece <= 1.0


def test_evaluate_ranker_refuses_train_pool(ws):
    """Split discipline (12 §15.4): train rows are never eval rows — a
    pool with no val-split rows BLOCKs and lists what it saw."""
    ref = _artifact(ws.root, "ranker_alltrain", "hypothesis_ranker",
                    "tabular", _ranker_rows(groups=4, split="train", seed=5))
    with pytest.raises(ValidationBlock) as ei:
        _wf(ws.root).evaluate_model(ws.ranker, ref)
    out = ei.value.render()
    assert "no eval-split rows" in out
    assert "train" in out
    assert "never eval rows" in out


def test_evaluate_ranker_refuses_wrong_transform(ws):
    ref = _artifact(ws.root, "ranker_calib_tf", "hypothesis_ranker",
                    "calibration", _calib_rows())
    with pytest.raises(ValidationBlock) as ei:
        _wf(ws.root).evaluate_model(ws.ranker, ref)
    out = ei.value.render()
    assert "transform" in out and "calibration" in out
    assert "tabular" in out
    assert "mlforge prepare" in out
