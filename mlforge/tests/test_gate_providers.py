"""Default gate providers — steps 4-store-backed / 5 / 6 + disk/gpu
derivation shared by gate step 16 and preflight.

Fail-closed rules under test:
  * raw dataset (no transform artifact) ⇒ FAIL with the prepare hint —
    transform identity is EXACT-required (12 §6.4/§8.4)
  * tampered store blob ⇒ FAIL at identity (step 4) and integrity (5)
  * unknown transform name ⇒ FAIL (closed registry)
  * finetune base: missing registry entry / unpublished / missing
    weights COMMIT marker ⇒ FAIL; import digest recorded ⇒ PASS
  * retrain parent must be AVAILABLE
  * runtime disk overrides reach BOTH the estimate and preflight (one
    formula, 12 §7.3); CPU plan does not demand nvidia-smi (12 §14)
"""

from __future__ import annotations

import json

import pytest

from mlforge.cli.main import _gpu_required_for
from mlforge.planner import (
    PLAN_FILENAME,
    build_plan,
    detect_capabilities,
    disk_estimate_components,
    estimate_required_disk_bytes,
)
from mlforge.run_spec import RunSpec
from mlforge.store import ContentStore
from mlforge.validation import GateContext, provide_dataset_identity, provide_transform
from mlforge.validation.gate import provide_model
from mlforge.workflow import WorkflowAPI

from test_ingest import COCO_FILES, _add_verified, _ingestion, _tree
from test_validation import make_spec

GiB = 1 << 30


@pytest.fixture
def home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFORGE_HOME", str(tmp_path / "mlforge_home"))


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


def _spec_refs(*refs: str) -> RunSpec:
    base = make_spec()
    return RunSpec(model=base.model, train_datasets=tuple(refs),
                   semantic=dict(base.semantic))


def _register(root, name: str, identity: str, *, version: str = "v1") -> None:
    d = root / "datasets" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity.json").write_text(json.dumps({
        "dataset_id": name, "identity": identity, "version": version,
        "file_count": 1, "total_bytes": 1,
    }), encoding="utf-8")


def _store_doc(root, payload: dict) -> str:
    return ContentStore(root / "store").put_bytes(
        json.dumps(payload).encode("utf-8"))


# -- step 4: store-backed datasets (no machine-local path) ----------------

def test_step4_store_backed_passes_without_path(ws):
    ident = _store_doc(ws, {"artifact_schema": "mlforge.prepared.v1",
                            "transform": {"name": "coco_detection"}})
    _register(ws, "m_prepared", ident)
    check = provide_dataset_identity(ws)(
        GateContext("run_x", ws, _spec_refs("m_prepared:v1")))
    assert check.verdict == "PASS"
    assert "m_prepared:v1" in check.data["datasets"]


def test_step4_store_backed_corrupt_blob_fails(ws):
    ident = _store_doc(ws, {"artifact_schema": "mlforge.prepared.v1"})
    _register(ws, "m_prepared", ident)
    ContentStore(ws / "store").path_for(ident).write_bytes(b"evil")
    check = provide_dataset_identity(ws)(
        GateContext("run_x", ws, _spec_refs("m_prepared:v1")))
    assert check.verdict == "FAIL"
    assert "integrity" in check.detail


def test_step4_raw_without_path_still_fails(ws):
    _register(ws, "raw_ds", "sha256:" + "0" * 64)
    check = provide_dataset_identity(ws)(
        GateContext("run_x", ws, _spec_refs("raw_ds:v1")))
    assert check.verdict == "FAIL"
    assert "no machine-local path" in check.detail


# -- step 5: transform identity -------------------------------------------

def test_transform_prepared_passes(ws):
    ident = _store_doc(ws, {"artifact_schema": "mlforge.prepared.v1",
                            "transform": {"name": "coco_detection",
                                          "code_hash": "sha256:a",
                                          "config_hash": "sha256:b"}})
    _register(ws, "m_prepared", ident)
    check = provide_transform(ws)(
        GateContext("run_x", ws, _spec_refs("m_prepared:v1")))
    assert check.verdict == "PASS"
    assert "m_prepared→coco_detection" in check.detail


def test_transform_raw_dataset_fails_with_prepare_hint(ws):
    _register(ws, "raw_ds", "sha256:" + "0" * 64)
    check = provide_transform(ws)(
        GateContext("run_x", ws, _spec_refs("raw_ds:v1")))
    assert check.verdict == "FAIL"
    assert "raw dataset" in check.detail
    # actionable: the REAL prepare command + the derived ref to train on
    assert "mlforge prepare rf_detr_s" in check.detail
    assert "rf_detr_s_prepared:v1" in check.detail


def test_transform_tampered_artifact_fails(ws):
    ident = _store_doc(ws, {"transform": {"name": "coco_detection"}})
    _register(ws, "m_prepared", ident)
    ContentStore(ws / "store").path_for(ident).write_bytes(b"tampered")
    check = provide_transform(ws)(
        GateContext("run_x", ws, _spec_refs("m_prepared:v1")))
    assert check.verdict == "FAIL"
    assert "integrity" in check.detail


def test_transform_unknown_registry_name_fails(ws):
    ident = _store_doc(ws, {"transform": {"name": "ghost_transform"}})
    _register(ws, "m_prepared", ident)
    check = provide_transform(ws)(
        GateContext("run_x", ws, _spec_refs("m_prepared:v1")))
    assert check.verdict == "FAIL"
    assert "not in the closed registry" in check.detail


def test_transform_unregistered_dataset_fails(ws):
    check = provide_transform(ws)(
        GateContext("run_x", ws, _spec_refs("ghost:v1")))
    assert check.verdict == "FAIL"
    assert "not registered" in check.detail


# -- step 6: model architecture + base weights ----------------------------

def _model_entry(root, name, version, *, artifact=None, origin="run",
                 run_id=None) -> None:
    d = root / "models" / f"model_{name}_{version}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").write_text(json.dumps({
        "schema_version": 1, "model_id": d.name, "name": name,
        "version": version, "artifact_hash": artifact, "run_id": run_id,
        "run_spec_hash": None, "origin": origin, "created_ts": 0.0,
    }), encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"state": "AVAILABLE"}),
                                   encoding="utf-8")


def test_model_from_scratch_passes(ws):
    wf = WorkflowAPI(ws)
    run_id = wf.create_run(make_spec()).run_id  # lineage: train, no base
    check = provide_model(ws)(GateContext(run_id, ws, make_spec()))
    assert check.verdict == "PASS"
    assert "train from scratch" in check.detail


def test_model_finetune_base_weights_verified(ws):
    wf = WorkflowAPI(ws)
    digest = "sha256:" + "ab" * 32
    _model_entry(ws, "rf_detr_s", "v1", artifact=digest, run_id="run_parent")
    marker = ws / "runs" / "run_parent" / "checkpoints" / "ckpt-000001" / "COMMIT"
    marker.parent.mkdir(parents=True)
    marker.write_text(digest, encoding="utf-8")
    run_id = wf.create_run(make_spec(), lineage={"origin": "finetune", "parent_model": "rf_detr_s:v1"}).run_id
    check = provide_model(ws)(GateContext(run_id, ws, make_spec()))
    assert check.verdict == "PASS"
    assert "base weights verified" in check.detail


def test_model_finetune_missing_base_fails(ws):
    wf = WorkflowAPI(ws)
    run_id = wf.create_run(make_spec(), lineage={"origin": "finetune", "parent_model": "ghost:v9"}).run_id
    check = provide_model(ws)(GateContext(run_id, ws, make_spec()))
    assert check.verdict == "FAIL"
    assert "not in the registry" in check.detail


def test_model_finetune_missing_weights_fails(ws):
    wf = WorkflowAPI(ws)
    digest = "sha256:" + "cd" * 32
    _model_entry(ws, "rf_detr_s", "v1", artifact=digest, run_id="run_parent")
    run_id = wf.create_run(make_spec(), lineage={"origin": "finetune", "parent_model": "rf_detr_s:v1"}).run_id
    check = provide_model(ws)(GateContext(run_id, ws, make_spec()))
    assert check.verdict == "FAIL"
    assert "COMMIT marker" in check.detail


def test_model_finetune_no_artifact_fails(ws):
    wf = WorkflowAPI(ws)
    _model_entry(ws, "rf_detr_s", "v1", artifact=None)
    run_id = wf.create_run(make_spec(), lineage={"origin": "finetune", "parent_model": "rf_detr_s:v1"}).run_id
    check = provide_model(ws)(GateContext(run_id, ws, make_spec()))
    assert check.verdict == "FAIL"
    assert "no weights artifact" in check.detail


def test_model_retrain_parent_available_passes(ws):
    wf = WorkflowAPI(ws)
    _model_entry(ws, "rf_detr_s", "v2", artifact=None)
    run_id = wf.create_run(make_spec(), lineage={"origin": "retrain", "parent_model": "rf_detr_s:v2"}).run_id
    check = provide_model(ws)(GateContext(run_id, ws, make_spec()))
    assert check.verdict == "PASS"
    assert "parent model rf_detr_s:v2 available" in check.detail


def test_model_finetune_imported_digest_passes(ws):
    wf = WorkflowAPI(ws)
    digest = "sha256:" + "ef" * 32
    _model_entry(ws, "rf_detr_s", "v1", artifact=digest, origin="import")
    run_id = wf.create_run(make_spec(), lineage={"origin": "finetune", "parent_model": "rf_detr_s:v1"}).run_id
    check = provide_model(ws)(GateContext(run_id, ws, make_spec()))
    assert check.verdict == "PASS"
    assert "imported" in check.detail


# -- disk estimate: one formula for gate step 16 + preflight (12 §7.3) ----

def test_disk_components_honor_runtime_overrides(tmp_path):
    spec = make_spec()
    runtime = {"checkpoint_bytes": 123, "log_bytes": 77, "safety_margin_bytes": 5}
    c = disk_estimate_components(tmp_path, spec, runtime=runtime)
    assert c["checkpoint_bytes"] == 123
    assert c["log_bytes"] == 77
    assert c["safety_margin_bytes"] == 5
    assert estimate_required_disk_bytes(tmp_path, spec, runtime=runtime) == (
        2 * 123 + 0 + 77 + 5
    )


def test_disk_components_default_shape(tmp_path):
    c = disk_estimate_components(tmp_path, make_spec())
    assert c["checkpoint_bytes"] == 4 * GiB
    assert c["log_bytes"] == 2 * GiB
    assert c["safety_margin_bytes"] == 2 * GiB


# -- preflight GPU expectation follows the execution plan (12 §14) --------

def test_gpu_required_follows_execution_plan(tmp_path):
    run_dir = tmp_path / "runs" / "run_p"
    run_dir.mkdir(parents=True)
    # no plan yet ⇒ live host detection: a CPU laptop never demands a
    # GPU it cannot have; broken detection ⇒ ValidationBlock ⇒ True
    host_needs_gpu = detect_capabilities().gpu_count > 0
    assert _gpu_required_for(tmp_path, "run_p", {}) is host_needs_gpu
    caps = detect_capabilities()
    plan = build_plan(make_spec(), caps)
    plan.write(run_dir / PLAN_FILENAME)
    # plan measured on THIS host decides; explicit runtime wins
    assert _gpu_required_for(tmp_path, "run_p", {}) is (caps.gpu_count > 0)
    assert _gpu_required_for(tmp_path, "run_p", {"gpu": True}) is True
    assert _gpu_required_for(tmp_path, "run_p", {"gpu": False}) is False


# -- integration: the default provider set passes a prepared project ------

def test_default_gate_full_pass_on_prepared_project(home, ws, tmp_path):
    from mlforge.cli.main import main

    coco = _tree(tmp_path / "src", "coco", COCO_FILES)
    _add_verified(ws, "coco_2017", coco)  # add + verify (asserts inside)
    _ingestion(ws, "models:\n  rf_detr_s:\n    transform: coco_detection\n"
                   "    train_sources: [coco_2017:train]\n")
    assert main(["--root", str(ws), "prepare", "rf_detr_s"]) == 0

    wf = WorkflowAPI(ws)
    spec = RunSpec(
        model="rf_detr_s",
        train_datasets=("rf_detr_s_prepared:v1",),
        semantic=dict(make_spec().semantic),
    )
    handle = wf.create_run(spec)
    # toy-scale disk budget (state/runtime.json — written by `train` in
    # production; written here so step 16's headroom math is test-sized)
    state = ws / "runs" / handle.run_id / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "runtime.json").write_text(json.dumps({
        "checkpoint_bytes": 1 << 20, "log_bytes": 1 << 20,
        "safety_margin_bytes": 1 << 20,
    }), encoding="utf-8")

    report = wf.validate_run(handle.run_id)
    assert not report.blocked, (
        report.first_failure.id, report.first_failure.detail)
    assert len(report.checks) == 16
    assert all(c.verdict == "PASS" for c in report.checks)
    assert wf.get_run_state(handle.run_id) == "READY"
