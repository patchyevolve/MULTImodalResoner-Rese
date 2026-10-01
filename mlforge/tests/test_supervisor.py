"""Build step 5 tests — supervisor crash detection (12 §12.3, 13 §5.3, §9.4).

Specs as executable checks:
  * heartbeat expired → INTERRUPTED (never PAUSED/RUNNING)
  * missing heartbeat + recent activity → grace (JUST entered RUNNING)
  * missing heartbeat + silence > timeout → INTERRUPTED
  * non-RUNNING runs are never touched
  * supervisor NEVER starts/resumes/reconciles — crash detection only
  * a broken run/workspace is reported in `errors`, never kills the scan
"""

from __future__ import annotations

import json
import time

import pytest

from mlforge.run_spec import RunSpec
from mlforge.supervisor import Supervisor
from mlforge.workflow import WorkflowAPI


class FakeClock:
    """Starts at the real epoch — journal timestamps are real, so the
    supervisor's silence calculation must be on the same timeline."""

    def __init__(self, now: float | None = None):
        self.now = time.time() if now is None else now

    def __call__(self) -> float:
        return self.now

    def advance(self, s: float) -> None:
        self.now += s


def make_spec() -> RunSpec:
    return RunSpec(
        model="rf_detr_s",
        train_datasets=("coco_2017:v1",),
        semantic={k: v for k, v in {
            "optimizer": "adamw", "learning_rate": 1e-4,
            "scheduler": "cosine", "loss": "l1", "seed": 42,
            "global_batch": 32, "epochs": 50,
            "precision_policy": "bf16",
        }.items()},
    )


@pytest.fixture
def wf(tmp_path):
    return WorkflowAPI(tmp_path)


def _running(wf) -> str:
    run_id = wf.create_run(make_spec()).run_id
    wf.begin_validation(run_id)
    wf.validation_pass(run_id)
    wf.preflight_pass(run_id)
    assert wf.get_run_state(run_id) == "RUNNING"
    return run_id


def _write_heartbeat(tmp_path, run_id, ts, pid=4242):
    state = tmp_path / "runs" / run_id / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "heartbeat.json").write_text(
        json.dumps({"ts": ts, "pid": pid, "host": "lab-1", "session_token": "t"})
    )


def test_fresh_heartbeat_is_alive(tmp_path, wf):
    clock = FakeClock()
    run_id = _running(wf)
    _write_heartbeat(tmp_path, run_id, ts=clock.now - 10)
    report = Supervisor(tmp_path, workflow=wf, timeout=120, clock=clock).scan_once()
    assert report["crashed"] == []
    assert report["checked"] == [run_id]
    assert wf.get_run_state(run_id) == "RUNNING"


def test_expired_heartbeat_crashes_to_interrupted(tmp_path, wf):
    """13 §9.4: stale RUNNING after a crash is INTERRUPTED the moment the
    heartbeat expires."""
    clock = FakeClock()
    run_id = _running(wf)
    _write_heartbeat(tmp_path, run_id, ts=clock.now - 121)
    report = Supervisor(tmp_path, workflow=wf, timeout=120, clock=clock).scan_once()
    assert [c["run_id"] for c in report["crashed"]] == [run_id]
    assert "heartbeat expired" in report["crashed"][0]["reason"]
    assert wf.get_run_state(run_id) == "INTERRUPTED"
    events = [e["event"] for e in wf.get_run_events(run_id)]
    assert events[-1] == "crash_detected"
    # supervisor never reconciles — INTERRUPTED waits for the explicit scan
    assert wf.get_run_state(run_id) != "RECONCILING"


def test_missing_heartbeat_within_grace_not_crashed(tmp_path, wf):
    clock = FakeClock()
    run_id = _running(wf)   # last journal activity ≈ now (events use real time)
    report = Supervisor(tmp_path, workflow=wf, timeout=120, clock=clock).scan_once()
    assert report["crashed"] == []


def test_missing_heartbeat_beyond_grace_crashes(tmp_path, wf):
    clock = FakeClock()
    run_id = _running(wf)
    _write_heartbeat(tmp_path, run_id, ts=clock.now)  # then file vanishes
    (tmp_path / "runs" / run_id / "state" / "heartbeat.json").unlink()
    # push the clock far past timeout — journal activity (real time ~now)
    # is now older than the timeout
    clock.advance(10_000)
    report = Supervisor(tmp_path, workflow=wf, timeout=120, clock=clock).scan_once()
    assert [c["run_id"] for c in report["crashed"]] == [run_id]
    assert "heartbeat missing" in report["crashed"][0]["reason"]
    assert wf.get_run_state(run_id) == "INTERRUPTED"


def test_non_running_runs_untouched(tmp_path, wf):
    clock = FakeClock()
    run_id = wf.create_run(make_spec()).run_id   # CREATED, no heartbeat
    report = Supervisor(tmp_path, workflow=wf, timeout=120, clock=clock).scan_once()
    assert report["checked"] == [] and report["crashed"] == []
    assert wf.get_run_state(run_id) == "CREATED"


def test_unreadable_heartbeat_counts_as_dead_signal(tmp_path, wf):
    clock = FakeClock()
    run_id = _running(wf)
    state = tmp_path / "runs" / run_id / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "heartbeat.json").write_text("{garbage")
    clock.advance(10_000)
    report = Supervisor(tmp_path, workflow=wf, timeout=120, clock=clock).scan_once()
    assert [c["run_id"] for c in report["crashed"]] == [run_id]


def test_broken_workspace_reported_not_fatal(tmp_path, wf):
    clock = FakeClock()
    run_id = _running(wf)
    # corrupt BOTH projection and authority → list_runs raises → reported
    (tmp_path / "runs" / run_id / "status.json").unlink()
    (tmp_path / "runs" / run_id / "events.jsonl").write_text("{broken\n")
    report = Supervisor(tmp_path, workflow=wf, clock=clock).scan_once()
    assert report["crashed"] == []
    assert report["errors"], "corrupt run must appear in errors, not kill the scan"


def test_supervisor_never_auto_continues(tmp_path, wf):
    """13 §1: Automatic continuation is NO — after the crash the supervisor
    does nothing else (no resume, no reconciliation)."""
    clock = FakeClock()
    run_id = _running(wf)
    _write_heartbeat(tmp_path, run_id, ts=clock.now - 500)
    sup = Supervisor(tmp_path, workflow=wf, timeout=120, clock=clock)
    sup.scan_once()
    sup.scan_once()  # second scan: already INTERRUPTED, not RUNNING → ignored
    assert wf.get_run_state(run_id) == "INTERRUPTED"
    events = [e["event"] for e in wf.get_run_events(run_id)]
    assert events.count("crash_detected") == 1


def test_run_loop_bounds_scans(tmp_path, wf):
    clock = FakeClock()
    run_id = _running(wf)
    _write_heartbeat(tmp_path, run_id, ts=clock.now)
    sup = Supervisor(tmp_path, workflow=wf, timeout=120, interval=0.0, clock=clock)
    report = sup.run(max_scans=3)
    assert report["crashed"] == []
    assert len(report["checked"]) == 3  # each scan reports the run
