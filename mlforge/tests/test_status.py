"""Build step 7 tests — status layer (13 §9, §4.1 OBSERVABILITY).

Specs as executable checks:
  * 13 §4.1: `status [RUN] [--verbose]`, `watch [RUN]`, `hardware`,
    `events <RUN> [--follow]` — all read-only, always safe.
  * 13 §9.4: state comes from the orchestrator, NEVER from telemetry;
    a stale heartbeat is reported (WARNING) but status never transitions
    a run — reconciliation belongs to resume/supervisor (12 §12.3).
  * 13 §9.5: L1 block (state/model/progress/speed/GPU/checkpoint),
    FAILED runs always display their disposition, L2 adds training
    detail, L3 hardware is diagnostic only.
  * 13 §9.6: `watch` is a VIEWER — keys write control intents; `q` quits
    the viewer only.
"""

from __future__ import annotations

import json
import time

from test_train_cli import _spec, wf_factory  # shared pass-gate helpers
from test_worker import _worker, make_ready

from mlforge.cli.main import main
from mlforge.runtime import ScaffoldTrainer, read_control
from mlforge.states import RunState
from mlforge.workflow import WorkflowAPI

# -- L1 single-run block ------------------------------------------------------

def test_status_single_run_l1_block(tmp_path, capsys):
    wf = wf_factory(tmp_path)
    h = wf.create_run(_spec())
    assert main(["--root", str(tmp_path), "status", h.run_id]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "CREATED"
    assert f"Model {_spec().model} · Run {h.run_id}" in out
    assert "Checkpoint none" in out


def test_status_failed_run_shows_disposition(tmp_path, capsys):
    """13 §9.5: FAILED runs always display cause + recovery + action."""
    wf = WorkflowAPI(tmp_path)  # default fail-closed gate → BLOCK
    h = wf.create_run(_spec())
    assert wf.validate_run(h.run_id).blocked
    assert wf.get_run_status(h.run_id)["failure"]["recovery"] == "FORK_ONLY"
    assert main(["--root", str(tmp_path), "status", h.run_id]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "FAILED"
    assert "Cause:" in out and "gate BLOCK" in out
    assert "Recovery:   FORK_ONLY" in out
    assert f"mlforge fork {h.run_id}" in out


def test_status_json_single_run(tmp_path, capsys):
    wf = wf_factory(tmp_path)
    h = wf.create_run(_spec())
    assert main(["--root", str(tmp_path), "status", h.run_id, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == h.run_id
    assert data["state"] == "CREATED"
    assert data["model"] == _spec().model


# -- stale heartbeat: reported, never acted on (13 §9.4, read-only) -----------

def _make_running(wf, spec):
    """CREATED → gate PASS (READY) → RUNNING, as the train flow would."""
    h = wf.create_run(spec)
    assert not wf.validate_run(h.run_id).blocked
    wf.preflight_pass(h.run_id)
    return h.run_id


def test_status_stale_heartbeat_warns_but_never_transitions(tmp_path, capsys):
    wf = wf_factory(tmp_path)
    run_id = _make_running(wf, _spec())
    hb = tmp_path / "runs" / run_id / "state" / "heartbeat.json"
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.write_text(json.dumps({"ts": time.time() - 500.0, "pid": 1, "host": "h"}))
    events_before = [e["event"] for e in wf.get_run_events(run_id)]

    assert main(["--root", str(tmp_path), "status", run_id]) == 0
    out = capsys.readouterr().out
    assert "WARNING — no heartbeat for" in out
    assert "INTERRUPTED (not failed)" in out  # expected disposition (13 §9.4)
    assert "awaiting `mlforge resume" in out

    # READ-ONLY invariant: status changed nothing (13 §4.1)
    assert wf.get_run_state(run_id) == RunState.RUNNING.value
    assert [e["event"] for e in wf.get_run_events(run_id)] == events_before


def test_status_fresh_heartbeat_has_no_warning(tmp_path, capsys):
    wf = wf_factory(tmp_path)
    run_id = _make_running(wf, _spec())
    hb = tmp_path / "runs" / run_id / "state" / "heartbeat.json"
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.write_text(json.dumps({"ts": time.time(), "pid": 1, "host": "h"}))
    assert main(["--root", str(tmp_path), "status", run_id]) == 0
    assert "WARNING" not in capsys.readouterr().out


# -- stage-aware (13 §9.5) -----------------------------------------------------

def test_status_stage_aware_block(tmp_path, capsys):
    wf = wf_factory(tmp_path)
    run_id = _make_running(wf, _spec())
    live = tmp_path / "runs" / run_id / "state" / "live.json"
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_text(json.dumps({"stage": "CHECKPOINTING",
                                "saving": "checkpoint-18492",
                                "global_step": 18492}))
    assert main(["--root", str(tmp_path), "status", run_id]) == 0
    out = capsys.readouterr().out
    assert "Stage: CHECKPOINTING" in out
    assert "Saving: checkpoint-18492" in out
    assert "Step 18,492" in out


# -- L2 verbose ----------------------------------------------------------------

def test_status_verbose_shows_training_detail(tmp_path, capsys):
    wf, run_id, _token = make_ready(tmp_path, runtime={"max_steps": 10})
    assert main(["--root", str(tmp_path), "status", run_id, "-v"]) == 0
    out = capsys.readouterr().out
    assert "— verbose —" in out
    assert "optimizer adamw" in out          # semantic config (L2)
    assert "max_steps 10" in out             # runtime knobs (L2)
    assert "Train data" in out and "coco_2017:v1" in out
    assert "Lease HELD" in out


def test_status_verbose_checkpoint_after_training(tmp_path, capsys):
    wf, run_id, token = make_ready(tmp_path)
    code = _worker(tmp_path, run_id, token,
                   ScaffoldTrainer(max_steps=4, steps_per_epoch=2),
                   checkpoint_interval=2).run()
    assert code == 0
    assert main(["--root", str(tmp_path), "status", run_id, "-v"]) == 0
    out = capsys.readouterr().out
    assert "Checkpoint last: 2" in out
    assert "integrity VALID" in out          # bounded verification (§9.5)
    assert out.splitlines()[0] == "COMPLETED"


# -- overview (L1, all runs) ---------------------------------------------------

def test_status_overview_lists_failed_recovery(tmp_path, capsys):
    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    wf.validate_run(h.run_id)  # fail-closed → FAILED[FORK_ONLY]
    assert main(["--root", str(tmp_path), "status"]) == 0
    out = capsys.readouterr().out
    assert h.run_id in out and "FAILED" in out
    assert "recovery=FORK_ONLY" in out


# -- L3 hardware ---------------------------------------------------------------

def test_hardware_telemetry(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "hardware"]) == 0
    out = capsys.readouterr().out
    assert "GPU" in out  # present or "unavailable" — either way diagnostic
    assert main(["--root", str(tmp_path), "hardware", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "gpu" in data and "disk" in data and "load_avg" in data


# -- watch (13 §9.6: viewer only) ----------------------------------------------

def test_watch_renders_one_frame_noninteractive(tmp_path, capsys):
    wf = wf_factory(tmp_path)
    h = wf.create_run(_spec())
    assert main(["--root", str(tmp_path), "watch", h.run_id]) == 0  # not a TTY
    out = capsys.readouterr().out
    assert f"MLForge — {h.run_id}" in out
    assert "[q]uit viewer" in out
    # a viewer never transitions anything
    assert wf.get_run_state(h.run_id) == "CREATED"


def test_watch_json_one_shot(tmp_path, capsys):
    wf = wf_factory(tmp_path)
    h = wf.create_run(_spec())
    assert main(["--root", str(tmp_path), "watch", h.run_id, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == h.run_id and data["state"] == "CREATED"


def test_watch_without_runs_exits_2(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "watch"]) == 2
    assert "NOT_FOUND" in capsys.readouterr().err


def test_watch_keys_write_control_intents_only(tmp_path):
    """The viewer never reaches into the worker — keys become intents."""
    from mlforge.cli.main import _watch_keys

    wf = WorkflowAPI(tmp_path)
    wf.create_run(_spec())
    run_id = wf.list_runs()[0]["id"]
    assert _watch_keys(wf, run_id, "q") == "quit"
    for key, want in (("p", "pause"), ("s", "stop"), ("c", "checkpoint")):
        assert _watch_keys(wf, run_id, key) == key
        ctrl = read_control(tmp_path, run_id)
        assert ctrl["action"] == want
        assert ctrl["requested_by"] == "cli:watch"


# -- events --follow -----------------------------------------------------------

def test_events_follow_prints_new_events_until_interrupt(tmp_path, capsys, monkeypatch):
    from mlforge.cli import main as cli_main

    wf = wf_factory(tmp_path)
    h = wf.create_run(_spec())

    calls = {"n": 0}

    def _sleep(_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            # an event arriving while --follow is tailing (13 §4.4 dedupe)
            wf.execute_idempotent("cmdFollow", "noop", lambda: 1, run_id=h.run_id)
            wf.execute_idempotent("cmdFollow", "noop", lambda: 1, run_id=h.run_id)
            return
        raise KeyboardInterrupt  # Ctrl+C detaches the viewer

    class _FakeTime:
        sleep = staticmethod(_sleep)

    monkeypatch.setattr(cli_main, "time", _FakeTime)

    assert main(["--root", str(tmp_path), "events", h.run_id, "--follow"]) == 0
    lines = [json.loads(l)["event"] for l in capsys.readouterr().out.splitlines() if l]
    assert "run_created" in lines
    assert "COMMAND_DEDUPED" in lines  # arrived DURING --follow


def test_events_missing_run_exits_2(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "events", "run_MISSING"]) == 2
