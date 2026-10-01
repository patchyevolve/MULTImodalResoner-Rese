import json

import pytest

from mlforge.errors import (
    EXIT_PRECONDITION,
    NoValidContinuation,
    ValidationBlock,
)
from mlforge.states import RunState


def _to_running(wf, spec):
    h = wf.create_run(spec)
    wf.begin_validation(h.run_id)
    wf.validation_pass(h.run_id)
    wf.preflight_pass(h.run_id)
    return h


# -- lifecycle end-to-end ------------------------------------------------

def test_happy_path_to_completed(wf, spec):
    h = _to_running(wf, spec)
    assert wf.get_run_state(h.run_id) == RunState.RUNNING.value
    wf.checkpoint_begin(h.run_id)
    wf.checkpoint_commit(h.run_id, ordinal=1)
    assert wf.get_run_state(h.run_id) == RunState.RUNNING.value
    wf.complete(h.run_id)
    assert wf.get_run_state(h.run_id) == RunState.COMPLETED.value
    assert wf.get_run_status(h.run_id)["run_spec_hash"] == h.run_spec_hash


def test_pause_then_resume_then_complete(wf, spec):
    h = _to_running(wf, spec)
    wf.pause(h.run_id)
    wf.pause_committed(h.run_id, checkpoint="ckpt-000001")
    assert wf.get_run_state(h.run_id) == RunState.PAUSED.value
    wf.resume(h.run_id)
    assert wf.get_run_state(h.run_id) == RunState.VALIDATING.value
    wf.validation_pass(h.run_id)
    wf.preflight_pass(h.run_id)
    wf.complete(h.run_id)
    assert wf.get_run_state(h.run_id) == RunState.COMPLETED.value


def test_stop_spends_resume_capability(wf, spec):
    h = _to_running(wf, spec)
    wf.stop(h.run_id)
    wf.stop_committed(h.run_id, checkpoint="ckpt-000001")
    assert wf.get_run_state(h.run_id) == RunState.STOPPED.value
    from mlforge.errors import InvalidTransition

    with pytest.raises(InvalidTransition):
        wf.resume(h.run_id)  # STOPPED: no resume transition exists (13 §5.3)


# -- FAILED disposition (13 §5.3) ---------------------------------------

def test_runtime_error_with_checkpoint_is_resumable(wf, spec):
    h = _to_running(wf, spec)
    wf.runtime_error(h.run_id, "CUDA OOM at step 41200", has_valid_checkpoint=True)
    status = wf.get_run_status(h.run_id)
    assert status["state"] == RunState.FAILED.value
    assert status["failure"]["recovery"] == "RESUME"
    wf.resume(h.run_id, has_valid_checkpoint=True)
    assert wf.get_run_state(h.run_id) == RunState.VALIDATING.value


def test_semantic_failure_is_fork_only_and_resume_blocks(wf, spec):
    h = _to_running(wf, spec)
    wf.runtime_error(
        h.run_id, "run_spec invariant violation", cause_is_semantic=True
    )
    status = wf.get_run_status(h.run_id)
    assert status["failure"]["recovery"] == "FORK_ONLY"
    with pytest.raises(NoValidContinuation) as e:
        wf.resume(h.run_id)
    assert e.value.exit_code == EXIT_PRECONDITION


def test_gate_failure_before_checkpoint_is_fork_only(wf, spec):
    h = wf.create_run(spec)
    wf.begin_validation(h.run_id)
    wf.validation_fail(h.run_id, "dataset identity mismatch")
    status = wf.get_run_status(h.run_id)
    assert status["state"] == RunState.FAILED.value
    assert status["failure"]["recovery"] == "FORK_ONLY"


# -- crash → reconciliation (12 §12.3) ----------------------------------

def test_crash_reconcile_resume(wf, spec):
    h = _to_running(wf, spec)
    wf.crash(h.run_id, reason="power loss")
    assert wf.get_run_state(h.run_id) == RunState.INTERRUPTED.value
    status = wf.reconcile(h.run_id, has_valid_checkpoint=True, resume_point="ckpt-000017")
    assert status["state"] == RunState.RECONCILING.value  # waits for explicit resume
    wf.resume(h.run_id)
    assert wf.get_run_state(h.run_id) == RunState.VALIDATING.value


def test_crash_reconcile_no_checkpoint_is_fork_only(wf, spec):
    h = _to_running(wf, spec)
    wf.crash(h.run_id, reason="kill -9")
    status = wf.reconcile(h.run_id, has_valid_checkpoint=False)
    assert status["state"] == RunState.FAILED.value
    assert status["failure"]["recovery"] == "FORK_ONLY"
    with pytest.raises(NoValidContinuation):
        wf.resume(h.run_id)


# -- authority & projection (12 §16.1) -----------------------------------

def test_status_is_projection_rebuildable_from_journal(wf, spec):
    h = _to_running(wf, spec)
    wf.pause(h.run_id)
    wf.pause_committed(h.run_id, checkpoint="ckpt-1")
    status_path = wf.root / "runs" / h.run_id / "status.json"
    status_path.unlink()  # projection destroyed — authority intact
    rebuilt = wf.get_run_status(h.run_id)
    assert rebuilt["state"] == RunState.PAUSED.value


def test_run_spec_written_once_and_immutable(wf, spec):
    h = wf.create_run(spec)
    with pytest.raises(ValidationBlock):
        spec.write_once(wf.root / "runs" / h.run_id / "run_spec.json")


def test_journal_is_append_only_api(wf, spec):
    h = wf.create_run(spec)
    from mlforge.journal import EventJournal

    j = EventJournal(wf.root / "runs" / h.run_id / "events.jsonl")
    assert not hasattr(j, "delete") and not hasattr(j, "rewrite")
    before = len(j.read())
    j.append("note", detail="manual")
    assert len(j.read()) == before + 1


def test_events_record_from_to_action(wf, spec):
    h = _to_running(wf, spec)
    events = wf.get_run_events(h.run_id)
    transitions = [e for e in events if e["event"] not in ("run_created",)]
    assert transitions[0]["event"] == "validation_started"
    assert transitions[0]["frm"] == RunState.CREATED.value
    assert transitions[0]["to"] == RunState.VALIDATING.value
    # every run event carries run_spec_hash → identity always traceable
    assert all("run_spec_hash" in e for e in events)


def test_not_found(wf):
    from mlforge.errors import NotFound

    with pytest.raises(NotFound):
        wf.get_run_status("run_MISSING")


# -- project / dataset / model machines ----------------------------------

def test_project_flow(wf):
    wf.create_project("demo")
    wf.configure_project("demo")
    assert wf._load_projection("project", "demo")["state"] == "CONFIGURED"
    wf.mark_project_ready("demo")
    assert wf._load_projection("project", "demo")["state"] == "READY"


def test_dataset_verify_reject_prepare(wf):
    wf.register_dataset("coco_2017", "sha256:abc")
    wf.verify_dataset("coco_2017", identity_matches=True, found_identity="sha256:abc")
    wf.prepare_dataset("coco_2017")
    assert wf._load_projection("dataset", "coco_2017")["state"] == "PREPARED"

    wf.register_dataset("broken", "sha256:def")
    wf.verify_dataset("broken", identity_matches=False, found_identity="sha256:xyz")
    assert wf._load_projection("dataset", "broken")["state"] == "REJECTED"


def test_model_flow_and_immutability(wf):
    m = wf.create_model(artifact_hash="sha256:model1")
    wf.validate_model(m)
    wf.publish_model(m)
    assert wf._load_projection("model", m)["state"] == "AVAILABLE"
    wf.record_model_event(m, "evaluation")
    assert wf._load_projection("model", m)["state"] == "EVALUATED"
    # no RETRAIN action exists on models (13 §5.4)
    from mlforge.errors import InvalidTransition

    with pytest.raises(InvalidTransition):
        wf._fire("model", m, "retrain", "model_retrained")


def test_list_runs(wf, spec):
    _to_running(wf, spec)
    runs = wf.list_runs()
    assert len(runs) == 1
    assert runs[0]["state"] == RunState.RUNNING.value
    assert json.loads((wf.root / "runs" / runs[0]["id"] / "status.json").read_text())[
        "state"
    ] == "RUNNING"
