"""Run identity captures (12 §6.4) + gate providers 3/7/8/9/14.

Specs as executable checks:
  * every run records source snapshot + tree hash + env fingerprint at
    creation (12 §6.4) — a run that cannot record identity does not exist
  * source gate (3): snapshot digest match, tree frozen at creation,
    missing capture ⇒ FAIL — fail-closed (12 §13.3)
  * environment gate (7): fingerprint frozen at creation
  * checkpoint gate (14): TRAIN fresh = PASS with that fact; RESUME =
    newest-valid predicate (12 §11.2), corrupt ⇒ FAIL, never guess
  * hardware (9) / driver (8): live detection facts (12 §13.1); CPU box
    has no driver to check — PASS with that honest detail, never a
    fabricated PASS
  * mutable state (runs/, artifacts/, store/) never counts as source
"""

from __future__ import annotations

import json

import pytest

from mlforge.errors import ValidationBlock
from mlforge.run_spec import RunSpec
from mlforge.validation import GateContext, provide_checkpoint, provide_driver, provide_hardware
from mlforge.workflow import WorkflowAPI

from test_validation import make_spec


@pytest.fixture
def wf(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFORGE_HOME", str(tmp_path / "mlforge_home"))
    return WorkflowAPI(tmp_path)


def _run(wf) -> str:
    return wf.create_run(make_spec()).run_id


def _ctx(wf, run_id, flow="TRAIN") -> GateContext:
    return GateContext(run_id, wf.root, make_spec(), flow=flow)


# -- creation captures (12 §6.4) ------------------------------------------

def test_create_run_records_captures(wf):
    run_id = _run(wf)
    d = wf.root / "runs" / run_id
    assert (d / "code" / "source.snapshot.tar.gz").is_file()
    assert (d / "code" / "source.json").is_file()
    assert (d / "environment" / "fingerprint.json").is_file()


def test_source_meta_shape_and_digest(wf, tmp_path):
    from mlforge.hashing import file_hash

    (wf.root / "app.py").write_text("print('hello')\n")  # some source to hash
    run_id = _run(wf)
    d = wf.root / "runs" / run_id
    meta = json.loads((d / "code" / "source.json").read_text())
    assert meta["schema"] == "mlforge.source_artifact.v1"
    assert meta["source_tree_hash"].startswith("sha256:")
    assert meta["file_count"] >= 1
    snap = d / "code" / "source.snapshot.tar.gz"
    assert meta["snapshot"]["sha256"] == file_hash(snap)
    assert meta["snapshot"]["bytes"] == snap.stat().st_size
    assert "git" in meta  # None outside a repo, dict inside — provenance


def test_env_meta_shape(wf):
    run_id = _run(wf)
    meta = json.loads(
        (wf.root / "runs" / run_id / "environment" / "fingerprint.json").read_text()
    )
    assert meta["schema"] == "mlforge.environment.v1"
    assert meta["fingerprint"]
    assert meta["python"] and meta["platform"]


def test_capture_failure_blocks_creation(wf, monkeypatch):
    """Fail-closed: a run that cannot record identity does not exist."""
    def _boom(*args, **kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr("mlforge.capture.capture_run_identity", _boom)
    with pytest.raises(ValidationBlock) as exc:
        wf.create_run(make_spec())
    assert "identity capture failed" in str(exc.value)


# -- mutable state never counts as source ---------------------------------

def test_mutable_dirs_excluded_from_tree(wf):
    from mlforge.capture import source_tree_hash

    run_id = _run(wf)
    h1, n1 = source_tree_hash(wf.root)
    (wf.root / "artifacts").mkdir(exist_ok=True)
    (wf.root / "artifacts" / "junk.bin").write_bytes(b"junk")
    (wf.root / "runs" / run_id / "events.jsonl").write_text('{"x":1}\n')
    h2, n2 = source_tree_hash(wf.root)
    assert (h1, n1) == (h2, n2)


# -- gate step 3: source_code ---------------------------------------------

def test_source_code_passes_on_fresh_run(wf):
    from mlforge.capture import provide_source_code

    run_id = _run(wf)
    check = provide_source_code(wf.root)(_ctx(wf, run_id))
    assert check.verdict == "PASS"
    assert check.step == 3
    assert "source tree verified" in check.detail
    assert check.data["source_tree_hash"]


def test_source_snapshot_tamper_fails(wf):
    from mlforge.capture import provide_source_code

    run_id = _run(wf)
    snap = wf.root / "runs" / run_id / "code" / "source.snapshot.tar.gz"
    snap.write_bytes(snap.read_bytes() + b"tamper")
    check = provide_source_code(wf.root)(_ctx(wf, run_id))
    assert check.verdict == "FAIL"
    assert "digest mismatch" in check.detail


def test_source_tree_changed_after_creation_fails(wf):
    from mlforge.capture import provide_source_code

    run_id = _run(wf)
    (wf.root / "new_module.py").write_text("# drift\n")
    check = provide_source_code(wf.root)(_ctx(wf, run_id))
    assert check.verdict == "FAIL"
    assert "changed since run creation" in check.detail
    assert "frozen" in check.detail


def test_missing_source_capture_fails(wf):
    from mlforge.capture import provide_source_code

    run_id = _run(wf)
    (wf.root / "runs" / run_id / "code" / "source.json").unlink()
    check = provide_source_code(wf.root)(_ctx(wf, run_id))
    assert check.verdict == "FAIL"
    assert "no source capture" in check.detail


def test_missing_snapshot_file_fails(wf):
    from mlforge.capture import provide_source_code

    run_id = _run(wf)
    (wf.root / "runs" / run_id / "code" / "source.snapshot.tar.gz").unlink()
    check = provide_source_code(wf.root)(_ctx(wf, run_id))
    assert check.verdict == "FAIL"
    assert "snapshot missing" in check.detail


# -- gate step 7: environment ---------------------------------------------

def test_environment_passes(wf):
    from mlforge.capture import provide_environment

    run_id = _run(wf)
    check = provide_environment(wf.root)(_ctx(wf, run_id))
    assert check.verdict == "PASS"
    assert check.step == 7
    assert "environment verified" in check.detail


def test_environment_tamper_fails(wf):
    from mlforge.capture import provide_environment

    run_id = _run(wf)
    path = wf.root / "runs" / run_id / "environment" / "fingerprint.json"
    meta = json.loads(path.read_text())
    meta["fingerprint"] = "deadbeef" * 8
    path.write_text(json.dumps(meta))
    check = provide_environment(wf.root)(_ctx(wf, run_id))
    assert check.verdict == "FAIL"
    assert "environment changed" in check.detail


def test_environment_missing_capture_fails(wf):
    from mlforge.capture import provide_environment

    run_id = _run(wf)
    (wf.root / "runs" / run_id / "environment" / "fingerprint.json").unlink()
    check = provide_environment(wf.root)(_ctx(wf, run_id))
    assert check.verdict == "FAIL"
    assert "no environment capture" in check.detail


# -- gate step 14: checkpoint (newest-valid predicate, 12 §11.2) ----------

def test_checkpoint_train_fresh_passes(wf):
    run_id = _run(wf)
    check = provide_checkpoint()(_ctx(wf, run_id, flow="TRAIN"))
    assert check.verdict == "PASS"
    assert "fresh run" in check.detail


def test_checkpoint_resume_without_store_fails(wf):
    run_id = _run(wf)
    check = provide_checkpoint()(_ctx(wf, run_id, flow="RESUME"))
    assert check.verdict == "FAIL"
    assert "no newest-valid checkpoint" in check.detail


def test_checkpoint_resume_with_valid_store_passes(wf):
    from mlforge.runtime import REQUIRED_COMPONENTS, CheckpointStore

    run_id = _run(wf)
    store = CheckpointStore(wf.root / "runs" / run_id)
    payload = {"model.bin": b"w", "optimizer.bin": b"o",
               "state.json": b'{"global_step": 7}'}
    store.write(1, payload, global_step=7, epoch=1,
                components=set(REQUIRED_COMPONENTS))
    check = provide_checkpoint()(_ctx(wf, run_id, flow="RESUME"))
    assert check.verdict == "PASS"
    assert "ckpt-000001 verified" in check.detail
    assert check.data["resume_ordinal"] == 1


def test_checkpoint_resume_corrupt_store_fails(wf):
    from mlforge.runtime import REQUIRED_COMPONENTS, CheckpointStore

    run_id = _run(wf)
    store = CheckpointStore(wf.root / "runs" / run_id)
    payload = {"model.bin": b"w", "optimizer.bin": b"o",
               "state.json": b'{"global_step": 7}'}
    store.write(1, payload, global_step=7, epoch=1,
                components=set(REQUIRED_COMPONENTS))
    # corrupt the payload — verification must catch it (12 §11.3)
    (wf.root / "runs" / run_id / "checkpoints" / "ckpt-000001" / "model.bin").write_bytes(b"evil")
    check = provide_checkpoint()(_ctx(wf, run_id, flow="RESUME"))
    assert check.verdict == "FAIL"
    assert "no newest-valid" in check.detail


# -- gate steps 8/9: driver + hardware (live detection, 12 §13.1) ---------

def test_hardware_passes_with_facts(wf):
    check = provide_hardware()(GateContext("run_x", wf.root, make_spec()))
    assert check.verdict == "PASS"
    assert "detected:" in check.detail
    assert check.data["cpu_count"] >= 1
    assert check.data["source"] in ("nvidia-smi", "cpu-only")


def test_driver_cpu_host_passes_honestly(wf):
    from mlforge.planner import detect_capabilities

    caps = detect_capabilities()
    check = provide_driver()(GateContext("run_x", wf.root, make_spec()))
    assert check.verdict == "PASS"
    if not caps.gpus:
        assert "not applicable" in check.detail
        assert check.data["gpu_count"] == 0
    else:  # GPU host: detection went through a responding driver
        assert "driver responding" in check.detail
        assert check.data["gpu_count"] >= 1


# -- workflow wiring -------------------------------------------------------

def test_default_providers_cover_new_steps(wf):
    providers = wf._providers()
    for key in ("source_code", "dataset", "environment", "plan", "driver",
                "hardware", "checkpoint"):
        assert key in providers, key


def test_explicit_gate_providers_still_win(wf):
    from mlforge.validation import provide_pass

    marker = provide_pass("explicit hardware")
    wf2 = WorkflowAPI(wf.root, gate_providers={"hardware": marker})
    assert wf2._providers()["hardware"] is marker  # setdefault never overrides
