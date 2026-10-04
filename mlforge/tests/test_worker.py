"""Build step 6b tests — worker runtime, control channel, spawn queue.

Specs as executable checks:
  * 12 §12.4: the worker proves lease ownership before touching anything;
    spawn requests are queued for the SUPERVISOR, never spawned by a shell;
    duplicate spawn → stand down (never evict the live worker's lease)
  * 13 §6.1/§7: READY → RUNNING only from the runtime; checkpoint cycle
    (RUNNING ⇄ CHECKPOINTING); write failure → FAILED[RESUME] with retry;
    graceful pause/stop only via explicit control intents (no auto-*)
  * 13 §9.1: metrics are observations (metrics.jsonl), never state
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from mlforge.errors import PreconditionFailed
from mlforge.leases import LeaseState, RunLeaseManager, provide_run_lease
from mlforge.run_spec import RunSpec
from mlforge.runtime import ScaffoldTrainer, write_control
from mlforge.runtime.worker import Worker
from mlforge.runtime.worker import main as worker_main
from mlforge.states import RunState
from mlforge.supervisor import (
    drain_pending,
    enqueue_spawn,
    ensure_supervisor,
)
from mlforge.validation import RESUME_GATE_STEPS, provide_pass
from mlforge.workflow import WorkflowAPI

GATE_PASS_KEYS = [s.id for s in RESUME_GATE_STEPS if s.id not in ("manifest", "schema")]


def _spec() -> RunSpec:
    return RunSpec(
        model="rf_detr_s",
        train_datasets=("coco_2017:v1",),
        semantic={
            "optimizer": "adamw", "learning_rate": 1e-4, "scheduler": "cosine",
            "loss": "l1", "seed": 42, "global_batch": 32, "epochs": 50,
            "precision_policy": "bf16",
        },
    )


def wf_with_providers(root) -> WorkflowAPI:
    providers = {k: provide_pass(f"{k} verified") for k in GATE_PASS_KEYS}
    providers["lease"] = provide_run_lease(RunLeaseManager(root))
    return WorkflowAPI(root, gate_providers=providers)


def make_ready(root, *, runtime: dict | None = None):
    """CREATED → gate PASS → READY with the gate's lease HELD (what the
    train/resume CLI leaves behind before the worker is spawned)."""
    wf = wf_with_providers(root)
    h = wf.create_run(_spec())
    if runtime:
        state = root / "runs" / h.run_id / "state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "runtime.json").write_text(json.dumps(runtime))
    report = wf.validate_run(h.run_id)
    assert not report.blocked, report.render()
    token = RunLeaseManager(root).status(h.run_id).session_token
    assert token
    return wf, h.run_id, token


def _worker(root, run_id, token, trainer, **kw) -> Worker:
    kw.setdefault("poll_interval", 0.0)
    kw.setdefault("heartbeat_interval", 60.0)
    return Worker(root, run_id, token, trainer=trainer, **kw)


# -- lifecycle -------------------------------------------------------------

def test_worker_ready_to_completed(tmp_path):
    wf, run_id, token = make_ready(tmp_path)
    code = _worker(
        tmp_path, run_id, token,
        ScaffoldTrainer(max_steps=6, steps_per_epoch=3),
        checkpoint_interval=2,
    ).run()
    assert code == 0
    assert wf.get_run_state(run_id) == RunState.COMPLETED.value
    events = [e["event"] for e in wf.get_run_events(run_id)]
    assert "preflight_passed" in events          # READY → RUNNING (runtime)
    assert events.count("checkpoint_committed") == 3   # steps 2, 4, 6
    assert events[-1] == "completed"
    # 13 §9.1: metrics are observations, one per step
    lines = (tmp_path / "runs" / run_id / "metrics" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 6
    assert json.loads(lines[-1])["global_step"] == 6
    # graceful exit leaves no single-writer residue
    assert RunLeaseManager(tmp_path).status(run_id).state == LeaseState.FREE
    assert not (tmp_path / "runs" / run_id / "state" / "heartbeat.json").is_file()


def test_worker_refuses_wrong_token(tmp_path):
    wf, run_id, _token = make_ready(tmp_path)
    code = _worker(tmp_path, run_id, "not-the-token", ScaffoldTrainer(max_steps=1)).run()
    assert code == 3
    assert wf.get_run_state(run_id) == RunState.READY.value  # nothing touched
    assert RunLeaseManager(tmp_path).status(run_id).state == LeaseState.HELD


def test_worker_refuses_non_ready_state(tmp_path):
    wf, run_id, token = make_ready(tmp_path)
    # rewind to CREATED (as if validation never happened)
    wf2 = WorkflowAPI(tmp_path)
    from mlforge.states import RunState as RS
    # a run can only rewind via projection tampering — simulate directly
    status = tmp_path / "runs" / run_id / "status.json"
    data = json.loads(status.read_text())
    data["state"] = RS.CREATED.value
    status.write_text(json.dumps(data))
    code = _worker(tmp_path, run_id, token, ScaffoldTrainer(max_steps=1)).run()
    assert code == 3
    # lease was ours to renew, so cleanup released it — but state never moved
    assert json.loads(status.read_text())["state"] == RS.CREATED.value


def test_worker_duplicate_spawn_stands_down(tmp_path):
    """A second spawn of a live run must neither evict the lease nor
    heartbeat nor change state (single writer, 12 §23.2)."""
    wf, run_id, token = make_ready(tmp_path)
    wf.preflight_pass(run_id)  # now RUNNING, as a live worker would be
    hb = tmp_path / "runs" / run_id / "state" / "heartbeat.json"
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.write_text(json.dumps({"ts": time.time(), "pid": 1, "host": "h"}))
    code = _worker(tmp_path, run_id, token, ScaffoldTrainer(max_steps=1)).run()
    assert code == 0
    assert wf.get_run_state(run_id) == RunState.RUNNING.value
    assert hb.is_file()                          # live worker's heartbeat intact
    assert RunLeaseManager(tmp_path).status(run_id).state == LeaseState.HELD


# -- no silent scaffold (correction: never fake training) -------------------

def test_resolve_trainer_fails_closed_without_opt_in(monkeypatch):
    from mlforge.runtime.trainer import resolve_trainer

    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    with pytest.raises(PreconditionFailed) as exc:
        resolve_trainer({})
    out = exc.value.render()
    assert "no real trainer integrated" in out
    assert "harness-scaffold" in out            # hint names the only opt-in
    # explicit PER-RUN opt-in works without the env (auditable runtime.json)
    assert isinstance(resolve_trainer({"trainer": "harness-scaffold"}),
                      ScaffoldTrainer)
    # session-wide opt-in (system tests)
    monkeypatch.setenv("MLFORGE_HARNESS", "1")
    assert isinstance(resolve_trainer({}), ScaffoldTrainer)
    # an explicit unknown id never falls back to anything
    with pytest.raises(PreconditionFailed) as exc2:
        resolve_trainer({"trainer": "torch"})
    assert "unknown trainer" in str(exc2.value)


def test_worker_without_trainer_refuses_leaving_run_ready(tmp_path, monkeypatch):
    """No real trainer + no opt-in ⇒ exit 3 BEFORE READY → RUNNING: the
    run stays READY (untouched, lease released), no metrics, no fake loss."""
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    wf, run_id, token = make_ready(tmp_path)
    code = _worker(tmp_path, run_id, token, None).run()
    assert code == 3
    assert wf.get_run_state(run_id) == RunState.READY.value  # never dirtied
    assert RunLeaseManager(tmp_path).status(run_id).state == LeaseState.FREE
    assert not (tmp_path / "runs" / run_id / "metrics" / "metrics.jsonl").exists()
    events = [e["event"] for e in wf.get_run_events(run_id)]
    assert "preflight_passed" not in events     # transition never fired
    assert "completed" not in events


def test_worker_harness_opt_in_still_runs(tmp_path, monkeypatch):
    """The explicit per-run opt-in keeps the system harness usable."""
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    wf, run_id, token = make_ready(
        tmp_path, runtime={"trainer": "harness-scaffold"}
    )
    code = _worker(tmp_path, run_id, token, None, checkpoint_interval=2).run()
    assert code == 0
    assert wf.get_run_state(run_id) == RunState.COMPLETED.value
    lines = (tmp_path / "runs" / run_id / "metrics" /
             "metrics.jsonl").read_text().splitlines()
    assert len(lines) > 0  # harness ran — loudly labeled, never silent


# -- pause / stop via control intents (13 §1: explicit only) ---------------

def _run_in_thread(worker: Worker):
    box = {}

    def _target():
        box["code"] = worker.run()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t, box


def _wait_state(wf, run_id, want, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if wf.get_run_state(run_id) in want:
            return wf.get_run_state(run_id)
        time.sleep(0.01)
    return wf.get_run_state(run_id)


def test_worker_pause_via_control_intent(tmp_path):
    wf, run_id, token = make_ready(tmp_path)
    w = _worker(tmp_path, run_id, token, ScaffoldTrainer(max_steps=10_000),
                poll_interval=0.01, heartbeat_interval=0.05)
    t, box = _run_in_thread(w)
    assert _wait_state(wf, run_id, {"RUNNING"}) == "RUNNING"
    write_control(tmp_path, run_id, "pause")
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert box["code"] == 0
    assert wf.get_run_state(run_id) == RunState.PAUSED.value
    events = [e["event"] for e in wf.get_run_events(run_id)]
    # final checkpoint committed BEFORE PAUSED (graceful pause)
    assert events[-1] == "pause_checkpoint_committed"
    assert RunLeaseManager(tmp_path).status(run_id).state == LeaseState.FREE
    assert not (tmp_path / "runs" / run_id / "state" / "control.json").is_file()
    ckpts = list((tmp_path / "runs" / run_id / "checkpoints").glob("ckpt-*"))
    assert ckpts, "pause must commit a final checkpoint"


def test_worker_stop_via_control_intent(tmp_path):
    wf, run_id, token = make_ready(tmp_path)
    w = _worker(tmp_path, run_id, token, ScaffoldTrainer(max_steps=10_000),
                poll_interval=0.01, heartbeat_interval=0.05)
    t, box = _run_in_thread(w)
    assert _wait_state(wf, run_id, {"RUNNING"}) == "RUNNING"
    write_control(tmp_path, run_id, "stop")
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert box["code"] == 0
    assert wf.get_run_state(run_id) == RunState.STOPPED.value
    assert RunLeaseManager(tmp_path).status(run_id).state == LeaseState.FREE


# -- failures land in defined states (13 §7) --------------------------------

class _ExplodingTrainer(ScaffoldTrainer):
    def step(self, state):
        if state.global_step >= 1:
            raise RuntimeError("CUDA OOM (simulated)")
        return super().step(state)


def test_worker_runtime_error_marks_failed_resume(tmp_path):
    wf, run_id, token = make_ready(tmp_path)
    code = _worker(tmp_path, run_id, token, _ExplodingTrainer(max_steps=10),
                   checkpoint_interval=1).run()
    assert code == 4
    status = wf.get_run_status(run_id)
    assert status["state"] == RunState.FAILED.value
    assert "CUDA OOM" in status["failure"]["cause"]
    assert status["failure"]["recovery"] == "RESUME"  # checkpoint @ step 1 exists
    assert RunLeaseManager(tmp_path).status(run_id).state == LeaseState.FREE


class _BadPayloadTrainer(ScaffoldTrainer):
    def checkpoint_payload(self, state):
        raise OSError("disk full")


def test_worker_checkpoint_failure_marks_failed(tmp_path):
    wf, run_id, token = make_ready(tmp_path)
    code = _worker(tmp_path, run_id, token, _BadPayloadTrainer(max_steps=10),
                   checkpoint_interval=1).run()
    assert code == 4
    events = [e["event"] for e in wf.get_run_events(run_id)]
    assert "checkpoint_started" in events and "checkpoint_failed" in events
    assert wf.get_run_state(run_id) == RunState.FAILED.value
    assert RunLeaseManager(tmp_path).status(run_id).state == LeaseState.FREE


def test_worker_control_poll_accepts_resume_of_corrupt_intent(tmp_path):
    """A torn control write is quarantined, not obeyed, not fatal."""
    from mlforge.runtime.control import read_control

    state = tmp_path / "runs" / "run_X" / "state"
    state.mkdir(parents=True)
    (state / "control.json").write_text("{not json")
    assert read_control(tmp_path, "run_X")["action"].startswith("corrupt")
    assert (state / "control.corrupt.json").is_file()
    assert not (state / "control.json").is_file()


# -- live state (13 §9.4: heartbeat + live.json are observations) -------------

def test_heartbeat_carries_progress_fields(tmp_path):
    from mlforge.runtime.heartbeat import HeartbeatWriter

    hb = HeartbeatWriter(tmp_path, "run_A", "tok")  # no lease → pure write
    payload = hb.beat(global_step=7, state="RUNNING")
    assert payload["global_step"] == 7 and payload["state"] == "RUNNING"
    data = json.loads(hb.path.read_text())
    assert data["global_step"] == 7 and data["state"] == "RUNNING"
    hb.clear()
    assert not hb.path.is_file()


def test_worker_writes_live_json_and_clears_on_exit(tmp_path):
    """state/live.json = current transient metrics; dies with the process."""
    wf, run_id, token = make_ready(tmp_path)
    w = _worker(tmp_path, run_id, token, ScaffoldTrainer(max_steps=10_000),
                poll_interval=0.01, heartbeat_interval=0.05)
    t, box = _run_in_thread(w)
    assert _wait_state(wf, run_id, {"RUNNING"}) == "RUNNING"
    live = tmp_path / "runs" / run_id / "state" / "live.json"
    deadline = time.time() + 5.0
    data = None
    while time.time() < deadline and data is None:
        if live.is_file():
            data = json.loads(live.read_text())
        time.sleep(0.01)
    assert data is not None, "worker must publish live.json while running"
    assert data["stage"] == "TRAINING"
    assert "global_step" in data and "loss" in data

    write_control(tmp_path, run_id, "stop")
    t.join(timeout=5.0)
    assert not t.is_alive() and box["code"] == 0
    assert not live.is_file()  # graceful exit removes live state


def test_worker_checkpoint_intent_forces_one_checkpoint(tmp_path):
    """13 §9.6 watch keybinding: the viewer writes an intent; the worker
    commits a checkpoint ONCE and clears the intent (not a mode)."""
    wf, run_id, token = make_ready(tmp_path)
    w = _worker(tmp_path, run_id, token, ScaffoldTrainer(max_steps=10_000),
                poll_interval=0.01, heartbeat_interval=0.05,
                checkpoint_interval=10_000)  # never fires periodically
    t, box = _run_in_thread(w)
    assert _wait_state(wf, run_id, {"RUNNING"}) == "RUNNING"
    write_control(tmp_path, run_id, "checkpoint")
    deadline = time.time() + 5.0
    events: list = []
    while time.time() < deadline:
        events = [e["event"] for e in wf.get_run_events(run_id)]
        if "checkpoint_committed" in events:
            break
        time.sleep(0.01)
    assert "checkpoint_committed" in events  # only the INTENT could cause it
    assert not (tmp_path / "runs" / run_id / "state" / "control.json").is_file()
    write_control(tmp_path, run_id, "stop")
    t.join(timeout=5.0)
    assert not t.is_alive() and box["code"] == 0
    assert wf.get_run_state(run_id) == RunState.STOPPED.value


# -- spawn queue (12 §12.4: CLI queues, supervisor spawns) -------------------

def test_drain_spawns_and_journals(tmp_path):
    WorkflowAPI(tmp_path).create_run(_spec())  # the run the request targets
    run_id = WorkflowAPI(tmp_path).list_runs()[0]["id"]
    enqueue_spawn(tmp_path, run_id, "tok-1")
    calls = []

    def fake_spawner(root, rid, token):
        calls.append((rid, token))
        return 4242

    spawned = drain_pending(tmp_path, spawner=fake_spawner)
    assert calls == [(run_id, "tok-1")]
    assert spawned == [{"run_id": run_id, "pid": 4242}]
    assert not list((tmp_path / "state" / "pending").glob("*.json"))
    wf = WorkflowAPI(tmp_path)
    events = wf.get_run_events(run_id)
    assert events[-1]["event"] == "WORKER_SPAWNED"
    assert "to" not in events[-1]  # a spawn never changes run state
    assert wf.get_run_state(run_id) == RunState.CREATED.value


def test_drain_retains_failed_spawn_for_retry(tmp_path):
    enqueue_spawn(tmp_path, "run_1", "tok-1")

    def boom(root, run_id, token):
        raise OSError("fork failed")

    assert drain_pending(tmp_path, spawner=boom) == []
    assert (tmp_path / "state" / "pending" / "run_1.json").is_file()  # retried


def test_drain_quarantines_corrupt_request(tmp_path):
    d = tmp_path / "state" / "pending"
    d.mkdir(parents=True)
    (d / "run_bad.json").write_text("{{{")
    assert drain_pending(tmp_path, spawner=lambda *a: 1) == []
    assert (d / "run_bad.json.corrupt").is_file()


def test_ensure_supervisor_noop_when_alive(tmp_path):
    import os

    sup = tmp_path / "state" / "supervisor.json"
    sup.parent.mkdir(parents=True)
    sup.write_text(json.dumps({"pid": os.getpid(), "host": "h",
                               "last_seen": time.time()}))

    def must_not_launch(root):
        raise AssertionError("launcher must not be called when alive")

    info = ensure_supervisor(tmp_path, launcher=must_not_launch)
    assert info == {"started": False, "pid": os.getpid()}


def test_ensure_supervisor_launches_when_absent(tmp_path):
    launched = []

    def launcher(root):
        import os as _os

        launched.append(str(root))
        (root / "state").mkdir(parents=True, exist_ok=True)
        (root / "state" / "supervisor.json").write_text(json.dumps(
            {"pid": _os.getpid(), "host": "h", "last_seen": time.time()}))
        return _os.getpid()

    info = ensure_supervisor(tmp_path, launcher=launcher, wait=2.0, poll=0.001)
    assert info["started"] is True and launched == [str(tmp_path)]


# -- worker process entry ----------------------------------------------------

def test_worker_main_exit_codes(tmp_path, capsys):
    wf, run_id, token = make_ready(tmp_path)
    # happy path via the module entry (in-process)
    assert worker_main(["--root", str(tmp_path), "--run", run_id,
                        "--token", token, "--poll-interval", "0"]) == 0
    assert wf.get_run_state(run_id) == "COMPLETED"
    assert worker_main(["--root", str(tmp_path), "--run", "run_MISSING",
                        "--token", "t"]) == 2
    # lease was released at completion → renew fails (precondition, exit 3)
    assert worker_main(["--root", str(tmp_path), "--run", run_id,
                        "--token", "wrong"]) == 3
