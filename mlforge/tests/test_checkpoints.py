"""Build step 6 tests — transactional checkpoints (12 §11) + heartbeat.

Specs as executable checks:
  * write protocol §11.1: staging → hashes → fsync → rename → manifest →
    fsync → commit marker; crash before marker ⇒ incomplete, reported
  * write-once: never overwrite a generation
  * recovery is a PREDICATE OVER IDENTITIES (§11.2): newest VALID by
    (global_step, ordinal), never ordinal arithmetic; corrupt newest is
    walked over with a reason; NO VALID → explicit, never silent restart
  * verification during selection (§11.3): corrupt → BLOCK
  * component completeness (§11.4) recorded and checkable
  * heartbeat: atomic write, lease renewed with owner check, clear on exit
"""

from __future__ import annotations

import json
import os

import pytest

from mlforge.errors import NotFound, ValidationBlock
from mlforge.leases import RunLeaseManager
from mlforge.runtime import (
    REQUIRED_COMPONENTS,
    CheckpointStore,
    HeartbeatWriter,
)
from mlforge.run_spec import RunSpec
from mlforge.workflow import WorkflowAPI

ALL_COMPONENTS = set(REQUIRED_COMPONENTS)


def _payload(step_seed: int = 0) -> dict[str, bytes]:
    """A complete §11.4-style payload (framework-agnostic bytes)."""
    return {
        "model.bin": f"weights-{step_seed}".encode(),
        "optimizer.bin": f"opt-{step_seed}".encode(),
        "state.json": json.dumps({"global_step": step_seed}).encode(),
    }


@pytest.fixture
def store(tmp_path):
    run_dir = tmp_path / "runs" / "run_X"
    run_dir.mkdir(parents=True)
    return CheckpointStore(run_dir)


# -- write protocol (§11.1) -----------------------------------------------

def test_write_commit_cycle(store):
    dest = store.write(
        21, _payload(1), global_step=41200, epoch=7, components=ALL_COMPONENTS
    )
    assert dest.is_dir()
    assert (dest / "manifest.json").is_file()
    assert (dest / "COMMIT").is_file()
    ok, reason = store.verify(21)
    assert ok and reason == "VERIFIED"
    # no staging leftovers
    assert list(store.root.glob("*.staging")) == []


def test_write_once_never_overwrites(store):
    store.write(1, _payload(1), global_step=100, epoch=1, components=ALL_COMPONENTS)
    with pytest.raises(ValidationBlock) as exc:
        store.write(1, _payload(2), global_step=200, epoch=2, components=ALL_COMPONENTS)
    assert "write-once" in str(exc.value)
    # original payload intact
    assert store.load_payload(1, "model.bin") == b"weights-1"


def test_empty_payload_refused(store):
    with pytest.raises(ValidationBlock):
        store.write(1, {}, global_step=0, epoch=0)


def test_illegal_payload_path_refused(store):
    with pytest.raises(ValidationBlock):
        store.write(1, {"../escape": b"x"}, global_step=0, epoch=0)


def test_incomplete_without_commit_marker_reported(store):
    store.write(23, _payload(23), global_step=41500, epoch=8,
                components=ALL_COMPONENTS)
    # simulate power loss: remove the commit marker
    (store.dir_for(23) / "COMMIT").unlink()
    ok, reason = store.verify(23)
    assert not ok and "NO COMMIT MARKER" in reason
    selection = store.newest_valid()
    assert selection.no_valid_checkpoint           # nothing else exists
    assert selection.skips[0].reason.startswith("NO COMMIT MARKER")


def test_missing_manifest_is_incomplete(store):
    store.write(5, _payload(), global_step=50, epoch=1, components=ALL_COMPONENTS)
    (store.dir_for(5) / "manifest.json").unlink()
    ok, reason = store.verify(5)
    assert not ok and "MANIFEST" in reason


def test_staging_leftover_reported_not_hidden(store):
    staging = store.root / "ckpt-000099.staging"
    staging.mkdir(parents=True)
    (staging / "half.bin").write_bytes(b"partial")
    selection = store.newest_valid()
    assert selection.no_valid_checkpoint
    assert any("STAGING LEFTOVER" in s.reason for s in selection.skips)


# -- newest-valid predicate (§11.2) ---------------------------------------

def test_selection_walks_over_corrupt_newest(tmp_path):
    """The §11.2 example: 23 no marker, 22 corrupt → resume 21."""
    store = CheckpointStore(tmp_path)
    for ordinal, step in [(21, 41200), (22, 41300), (23, 41400)]:
        store.write(ordinal, _payload(ordinal), global_step=step, epoch=5,
                    components=ALL_COMPONENTS)
    (store.dir_for(23) / "COMMIT").unlink()                       # 23: incomplete
    (store.dir_for(22) / "model.bin").write_bytes(b"TAMPERED")    # 22: hash mismatch

    selection = store.newest_valid()
    assert selection.resume_ordinal == 21
    assert selection.resume_point == "ckpt-000021"
    reasons = {s.ordinal: s.reason for s in selection.skips}
    assert "NO COMMIT MARKER" in reasons[23]
    assert "FILE HASH MISMATCH" in reasons[22]
    text = "\n".join(selection.report_lines())
    assert "checkpoint-000023: NO COMMIT MARKER" in text
    assert "checkpoint-000022: MANIFEST HASH MISMATCH" not in text  # file mismatch
    assert "FILE HASH MISMATCH" in text
    assert "RESUME POINT: checkpoint-000021" in text
    assert "recovery_from: 23 → 22 → 21 (2 skips" in text


def test_selection_uses_global_step_not_ordinal(tmp_path):
    """argmax (global_step, ordinal) — ordinal arithmetic is the bug."""
    store = CheckpointStore(tmp_path)
    store.write(17, _payload(17), global_step=99999, epoch=30,
                components=ALL_COMPONENTS)   # lower ordinal, HIGHER step
    store.write(18, _payload(18), global_step=41200, epoch=10,
                components=ALL_COMPONENTS)   # higher ordinal, lower step
    selection = store.newest_valid()
    assert selection.resume_ordinal == 17


def test_no_valid_checkpoint_is_explicit(store):
    store.write(1, _payload(), global_step=10, epoch=1, components=ALL_COMPONENTS)
    (store.dir_for(1) / "COMMIT").unlink()
    selection = store.newest_valid()
    assert selection.no_valid_checkpoint
    text = "\n".join(selection.report_lines())
    assert "NO VALID CHECKPOINT" in text
    assert "fork" in text  # never "resumed from step 0"


def test_non_contiguous_generations(tmp_path):
    """Fork history: 17 and 41 exist, 18–40 never did — still fine."""
    store = CheckpointStore(tmp_path)
    store.write(17, _payload(17), global_step=100, epoch=1, components=ALL_COMPONENTS)
    store.write(41, _payload(41), global_step=900, epoch=9, components=ALL_COMPONENTS)
    selection = store.newest_valid()
    assert selection.resume_ordinal == 41
    assert selection.skips == ()


def test_scan_of_empty_store(store):
    selection = store.newest_valid()
    assert selection.no_valid_checkpoint
    assert selection.attempted_newest is None


# -- verification on load (§11.3) -----------------------------------------

def test_corrupt_payload_blocks_load(store):
    store.write(9, _payload(), global_step=90, epoch=1, components=ALL_COMPONENTS)
    (store.dir_for(9) / "optimizer.bin").write_bytes(b"clobbered")
    with pytest.raises(ValidationBlock) as exc:
        store.load_payload(9, "model.bin")
    assert "failed verification" in str(exc.value)


def test_missing_payload_file_not_found(store):
    store.write(9, _payload(), global_step=90, epoch=1, components=ALL_COMPONENTS)
    with pytest.raises(NotFound):
        store.load_payload(9, "nope.bin")


def test_manifest_tamper_detected(store):
    store.write(3, _payload(), global_step=30, epoch=1, components=ALL_COMPONENTS)
    manifest_path = store.dir_for(3) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["global_step"] = 999999  # rewrite history
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    ok, reason = store.verify(3)
    assert not ok and "MANIFEST HASH MISMATCH" in reason


# -- component completeness (§11.4) ---------------------------------------

def test_missing_components_detected(store):
    store.write(1, _payload(), global_step=1, epoch=1,
                components={"model", "optimizer"})  # no rng/sampler...
    missing = store.missing_components(1)
    assert "rng_hierarchy" in missing
    assert "sampler_state" in missing
    assert "model" not in missing


def test_complete_components_empty_missing(store):
    store.write(1, _payload(), global_step=1, epoch=1, components=ALL_COMPONENTS)
    assert store.missing_components(1) == []


# -- prune (keep top-K) ----------------------------------------------------

def test_prune_keeps_top_k_valid(store):
    for i in range(1, 6):
        store.write(i, _payload(i), global_step=i * 100, epoch=i,
                    components=ALL_COMPONENTS)
    removed = store.prune(keep=2)
    assert removed == [1, 2, 3]
    remaining = {c.ordinal for c in store.scan_candidates() if c.valid}
    assert remaining == {4, 5}


# -- heartbeat --------------------------------------------------------------

def test_heartbeat_atomic_and_readable(tmp_path):
    run_dir = tmp_path / "runs" / "run_H"
    (run_dir / "state").mkdir(parents=True)
    hb = HeartbeatWriter(tmp_path, "run_H", "token-1", clock=lambda: 123.5)
    payload = hb.beat()
    on_disk = json.loads((run_dir / "state" / "heartbeat.json").read_text())
    assert on_disk == payload
    assert on_disk["ts"] == 123.5 and on_disk["session_token"] == "token-1"
    assert not list((run_dir / "state").glob("*.tmp"))


def test_heartbeat_renews_lease_with_owner_check(tmp_path):
    (tmp_path / "runs" / "run_H").mkdir(parents=True)
    mgr = RunLeaseManager(tmp_path)
    info = mgr.acquire("run_H")
    hb = HeartbeatWriter(tmp_path, "run_H", info.session_token, lease=mgr)
    hb.beat()
    assert mgr.status("run_H").state.value == "HELD"
    # wrong token → lease renew refuses → beat raises (worker must stop)
    bad = HeartbeatWriter(tmp_path, "run_H", "not-the-token", lease=mgr)
    from mlforge.errors import PreconditionFailed

    with pytest.raises(PreconditionFailed):
        bad.beat()


def test_heartbeat_due_threshold(tmp_path):
    clock = {"now": 1000.0}
    hb = HeartbeatWriter(tmp_path, "run_H", "t", clock=lambda: clock["now"])
    assert hb.due(None, interval=30) is True
    assert hb.due(980.0, interval=30) is False
    assert hb.due(969.0, interval=30) is True


def test_heartbeat_clear(tmp_path):
    (tmp_path / "runs" / "run_H" / "state").mkdir(parents=True)
    hb = HeartbeatWriter(tmp_path, "run_H", "t")
    hb.beat()
    assert hb.path.is_file()
    hb.clear()
    assert not hb.path.is_file()


# -- deterministic reconciliation scan used by workflow --------------------

def _crashed_run(wf) -> str:
    spec = RunSpec(
        model="rf_detr_s",
        train_datasets=("coco_2017:v1",),
        semantic={k: v for k, v in {
            "optimizer": "adamw", "learning_rate": 1e-4,
            "scheduler": "cosine", "loss": "l1", "seed": 42,
            "global_batch": 32, "epochs": 50,
            "precision_policy": "bf16",
        }.items()},
    )
    run_id = wf.create_run(spec).run_id
    wf.begin_validation(run_id)
    wf.validation_pass(run_id)
    wf.preflight_pass(run_id)
    wf.crash(run_id, "power loss")
    return run_id


def test_reconcile_from_disk_full_flow(tmp_path):
    wf = WorkflowAPI(tmp_path)
    run_id = _crashed_run(wf)
    store = CheckpointStore(tmp_path / "runs" / run_id)
    store.write(21, _payload(21), global_step=41200, epoch=5,
                components=ALL_COMPONENTS)
    store.write(22, _payload(22), global_step=41300, epoch=5,
                components=ALL_COMPONENTS)
    (store.dir_for(22) / "COMMIT").unlink()  # crash mid-checkpoint

    report = wf.reconcile_from_disk(run_id)
    assert report["resume_point"] == "ckpt-000021"
    assert report["state_after"] == "RECONCILING"   # waits for explicit resume
    assert report["consistent"] is True
    assert len(report["skips"]) == 1

    events = [e["event"] for e in wf.get_run_events(run_id)]
    # §11.2 rule 4: skip recorded BEFORE the reconciled report
    assert events.index("CHECKPOINT_SKIPPED") < events.index("RECONCILED")
    assert events[-1] == "RECONCILED" and events[-2] == "reconciliation_scan"
    skipped = wf.get_run_events(run_id)[events.index("CHECKPOINT_SKIPPED")]
    assert skipped["ordinal"] == 22 and "NO COMMIT MARKER" in skipped["reason"]
    reconciled = wf.get_run_events(run_id)[-1]
    assert reconciled["reconciliation_id"]

    # resume works from the derived point (disposition via last recon scan)
    wf.resume(run_id)
    assert wf.get_run_state(run_id) == "VALIDATING"


def test_reconcile_from_disk_idempotent(tmp_path):
    """12 §12.3: running twice → identical results, no duplicate events."""
    wf = WorkflowAPI(tmp_path)
    run_id = _crashed_run(wf)
    store = CheckpointStore(tmp_path / "runs" / run_id)
    store.write(1, _payload(), global_step=10, epoch=1, components=ALL_COMPONENTS)

    first = wf.reconcile_from_disk(run_id)
    n_events = len(wf.get_run_events(run_id))
    second = wf.reconcile_from_disk(run_id)
    assert second.get("stored") is True
    assert second["reconciliation_id"] == first["reconciliation_id"]
    assert second["resume_point"] == first["resume_point"]
    assert len(wf.get_run_events(run_id)) == n_events  # no duplicates


def test_reconcile_from_disk_no_checkpoint_marks_failed_fork_only(tmp_path):
    """INTERRUPTED + no valid checkpoint → FAILED(FORK_ONLY), never a
    silent restart (12 §12.3 invariant)."""
    wf = WorkflowAPI(tmp_path)
    run_id = _crashed_run(wf)
    report = wf.reconcile_from_disk(run_id)
    assert report["no_valid_checkpoint"] is True
    assert report["state_after"] == "FAILED"
    status = wf.get_run_status(run_id)
    assert status["failure"]["recovery"] == "FORK_ONLY"
    from mlforge.errors import NoValidContinuation

    with pytest.raises(NoValidContinuation):
        wf.resume(run_id)


def test_reconcile_from_disk_stale_projection_rebuilds(tmp_path):
    """Journal wins: projection lying about state → reconciled to truth
    (13 §9.4) and the divergence is journaled as a crash."""
    wf = WorkflowAPI(tmp_path)
    run_id = _crashed_run(wf)
    store = CheckpointStore(tmp_path / "runs" / run_id)
    store.write(1, _payload(), global_step=10, epoch=1, components=ALL_COMPONENTS)
    # tamper the projection to a different state (simulates torn write)
    status_path = tmp_path / "runs" / run_id / "status.json"
    data = json.loads(status_path.read_text())
    data["state"] = "CREATED"          # journal says INTERRUPTED
    status_path.write_text(json.dumps(data))
    assert wf.get_run_state(run_id) == "CREATED"  # stale reading

    report = wf.reconcile_from_disk(run_id)
    assert report["consistent"] is False
    # journal authority = INTERRUPTED (not live) → truth wins, no fake crash
    assert report["state_after"] == "RECONCILING"
    assert wf.get_run_state(run_id) == "RECONCILING"  # projection rebuilt


def test_reconcile_requires_interrupted(tmp_path):
    wf = WorkflowAPI(tmp_path)
    spec = RunSpec(
        model="m", train_datasets=("d:v1",),
        semantic={k: 1 for k in (
            "optimizer", "learning_rate", "scheduler", "loss",
            "seed", "global_batch", "epochs", "precision_policy")},
    )
    run_id = wf.create_run(spec).run_id  # CREATED — nothing to reconcile
    from mlforge.errors import ValidationBlock

    with pytest.raises(ValidationBlock):
        wf.reconcile_from_disk(run_id)


def test_resume_auto_reconciles_interrupted(tmp_path):
    """13 §6.2: for INTERRUPTED, reconciliation runs before validation."""
    wf = WorkflowAPI(tmp_path)
    run_id = _crashed_run(wf)  # state INTERRUPTED
    store = CheckpointStore(tmp_path / "runs" / run_id)
    store.write(7, _payload(7), global_step=70, epoch=1, components=ALL_COMPONENTS)

    wf.resume(run_id)  # no explicit reconcile needed
    assert wf.get_run_state(run_id) == "VALIDATING"
    events = [e["event"] for e in wf.get_run_events(run_id)]
    assert "CHECKPOINT_SKIPPED" not in events            # nothing skipped
    assert "reconciliation_scan" in events
    assert "RECONCILED" in events
    # disposition: has_valid_checkpoint=True from the reconciliation
    assert wf._last_reconciliation(run_id) is True


def test_resume_interrupted_without_checkpoint_blocks_fork_only(tmp_path):
    wf = WorkflowAPI(tmp_path)
    run_id = _crashed_run(wf)  # INTERRUPTED, no checkpoints at all

    from mlforge.errors import NoValidContinuation

    with pytest.raises(NoValidContinuation):
        wf.resume(run_id)
    status = wf.get_run_status(run_id)
    assert status["state"] == "FAILED"
    assert status["failure"]["recovery"] == "FORK_ONLY"
