"""Build step 6b tests — `train`/`resume`/`pause`/`stop` CLI (13 §4, §6).

Specs as executable checks:
  * §4.2 async by default: train queues a spawn for the supervisor and
    returns; the CLI never becomes the training process (12 §12.4)
  * §4.4 idempotency: duplicate --command-id returns the ORIGINAL run id
  * fail-closed default: without wired providers the gate BLOCKS train
    (exit 1) — never a silent start
  * §6.2 resume: INTERRUPTED → reconcile first; FORK_ONLY → exit 3;
    gate/preflight BLOCK → run untouched
  * pause/stop: control intent for a live worker; synchronous for PAUSED
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from mlforge.cli.main import main
from mlforge.errors import NoValidContinuation, PreconditionFailed
from mlforge.leases import RunLeaseManager, provide_run_lease
from mlforge.runtime import ScaffoldTrainer, write_control
from mlforge.runtime.worker import Worker
from mlforge.run_spec import RunSpec
from mlforge.states import RunState
from mlforge.supervisor import enqueue_spawn  # noqa: F401 (contract import)
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


class _TestWorkflow(WorkflowAPI):
    """Environment-adjusted preflight for tests: /tmp test roots are small
    and CI boxes may be GPU-less — the PROBES still run (real flow), only
    the sizing floors are set to test scale. Product defaults stay 12 §7.3
    worst-case (4 GiB checkpoint, 8 GiB RAM, GPU required)."""

    def preflight_run(self, run_id, preflight=None, **kw):
        kw["gpu_required"] = False  # CI boxes may be GPU-less; probe still runs
        kw.setdefault("checkpoint_bytes", 1 << 20)
        kw.setdefault("log_bytes", 1 << 20)
        kw.setdefault("safety_margin_bytes", 1 << 20)
        kw.setdefault("min_ram_bytes", 1 << 20)
        return super().preflight_run(run_id, preflight, **kw)


def wf_factory(root):
    """What an operator environment with wired providers looks like —
    the default CLI (no factory) stays fail-closed."""
    providers = {k: provide_pass(f"{k} verified") for k in GATE_PASS_KEYS}
    providers["lease"] = provide_run_lease(RunLeaseManager(root))
    return _TestWorkflow(root, gate_providers=providers)


@pytest.fixture
def no_supervisor(monkeypatch):
    """Queueing is real; daemon spawning is not (tests are not daemons)."""
    import mlforge.cli.main as cli

    monkeypatch.setattr(
        cli, "ensure_supervisor",
        lambda root, **kw: {"started": False, "pid": 99999},
    )


def _write_config(tmp_path, **extra) -> str:
    data = {
        "schema_version": 1,
        "model": "rf_detr_s",
        "train_datasets": ["coco_2017:v1"],
        "semantic": {
            "optimizer": "adamw", "learning_rate": 1e-4, "scheduler": "cosine",
            "loss": "l1", "seed": 42, "global_batch": 32, "epochs": 50,
            "precision_policy": "bf16",
        },
        **extra,
    }
    p = tmp_path / "run.json"
    p.write_text(json.dumps(data))
    return str(p)


# -- train -------------------------------------------------------------------

def test_train_fails_closed_without_providers(tmp_path, capsys, no_supervisor):
    """Default gate has no identity providers ⇒ BLOCK (exit 1), run
    FAILED[FORK_ONLY], nothing spawned — fail-closed, never fake start."""
    cfg = _write_config(tmp_path)
    code = main(["--root", str(tmp_path), "train", "--config", cfg, "--yes"])
    assert code == 1
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "Source artifact" in out
    wf = WorkflowAPI(tmp_path)
    (run_id,) = [r["id"] for r in wf.list_runs()]
    assert wf.get_run_state(run_id) == "FAILED"
    assert wf.get_run_status(run_id)["failure"]["recovery"] == "FORK_ONLY"
    assert not list((tmp_path / "state" / "pending").glob("*.json"))  # no spawn


def test_train_queues_supervisor_spawn(tmp_path, capsys, no_supervisor):
    cfg = _write_config(tmp_path, runtime={"max_steps": 7, "checkpoint_interval": 3})
    code = main(["--root", str(tmp_path), "train", "--config", cfg, "--yes", "--json"],
                wf_factory=wf_factory)
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "started" and result["supervisor_pid"] == 99999
    run_id = result["run_id"]

    wf = WorkflowAPI(tmp_path)
    assert wf.get_run_state(run_id) == "READY"      # worker not started yet
    pending = json.loads((tmp_path / "state" / "pending" / f"{run_id}.json").read_text())
    assert pending["session_token"]                 # the lease's token
    assert RunLeaseManager(tmp_path).status(run_id).session_token == \
        pending["session_token"]
    # runtime knobs reach the worker via the run folder
    runtime = json.loads(
        (tmp_path / "runs" / run_id / "state" / "runtime.json").read_text())
    assert runtime["max_steps"] == 7


def test_train_confirmation_no_declines(tmp_path, capsys, monkeypatch, no_supervisor):
    cfg = _write_config(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    code = main(["--root", str(tmp_path), "train", "--config", cfg])
    assert code == 0
    assert "Cancelled — nothing was created" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())


def test_train_missing_config_exits_3(tmp_path, capsys):
    code = main(["--root", str(tmp_path), "train",
                 "--config", str(tmp_path / "nope.json"), "--yes"])
    assert code == 3
    assert "config not found" in capsys.readouterr().err


def test_train_command_id_dedupes(tmp_path, capsys, no_supervisor):
    cfg = _write_config(tmp_path)
    cid = "01JCOMMANDTRAIN"
    args = ["--root", str(tmp_path), "train", "--config", cfg, "--yes",
            "--command-id", cid, "--json"]
    first = json.loads(_out(capsys, lambda: main(args, wf_factory=wf_factory))["out"])
    second = json.loads(_out(capsys, lambda: main(args, wf_factory=wf_factory))["out"])
    assert first["run_id"] == second["run_id"]          # never a second run
    assert len(WorkflowAPI(tmp_path).list_runs()) == 1
    events = WorkflowAPI(tmp_path).get_run_events(first["run_id"])
    assert events[-1]["event"] == "COMMAND_DEDUPED"


def _out(capsys, fn):
    code = fn()
    captured = capsys.readouterr()
    return {"code": code, "out": captured.out, "err": captured.err}


# -- resume ------------------------------------------------------------------

def _paused_run(tmp_path) -> str:
    """READY → RUNNING → PAUSED with the lease released (what a real
    graceful pause leaves behind)."""
    wf = wf_factory(tmp_path)
    h = wf.create_run(_spec())
    assert not wf.validate_run(h.run_id).blocked
    wf.preflight_pass(h.run_id)
    wf.pause(h.run_id)
    wf.pause_committed(h.run_id, "ckpt-000000")
    token = RunLeaseManager(tmp_path).status(h.run_id).session_token
    RunLeaseManager(tmp_path).release(h.run_id, token)
    return h.run_id


def test_resume_paused_launches_worker(tmp_path, capsys, no_supervisor):
    run_id = _paused_run(tmp_path)
    code = main(["--root", str(tmp_path), "resume", run_id, "--json"],
                wf_factory=wf_factory)
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "started"
    assert WorkflowAPI(tmp_path).get_run_state(run_id) == "READY"  # gate fired
    assert (tmp_path / "state" / "pending" / f"{run_id}.json").is_file()


def test_resume_fork_only_exits_3(tmp_path, capsys):
    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    wf.begin_validation(h.run_id)
    wf.validation_fail(h.run_id, "invariant violation")
    assert wf.get_run_status(h.run_id)["failure"]["recovery"] == "FORK_ONLY"
    code = main(["--root", str(tmp_path), "resume", h.run_id])
    assert code == 3
    err = capsys.readouterr().err
    assert "NO VALID CONTINUATION" in err or "FORK_ONLY" in err
    assert wf.get_run_state(h.run_id) == "FAILED"  # untouched


def test_resume_interrupted_reconciles_first(tmp_path, capsys, no_supervisor):
    """Crash flow: RUNNING → (worker dies, lease operator-broken) →
    INTERRUPTED → resume reconciles BEFORE validation (13 §6.2)."""
    wf = wf_factory(tmp_path)
    h = wf.create_run(_spec())
    assert not wf.validate_run(h.run_id).blocked
    run_id = h.run_id
    wf.preflight_pass(run_id)          # RUNNING (as the worker would)
    wf.crash(run_id, "power loss")     # INTERRUPTED
    # crashed workers never release — the operator breaks the lease (13 §7)
    lease = RunLeaseManager(tmp_path)
    lease.release(run_id, lease.status(run_id).session_token)
    # commit one checkpoint so reconciliation has a resume point
    from mlforge.runtime import REQUIRED_COMPONENTS, CheckpointStore

    store = CheckpointStore(tmp_path / "runs" / run_id)
    payload = {c: b"x" for c in REQUIRED_COMPONENTS}
    store.write(1, payload, global_step=10, epoch=1, components=set(REQUIRED_COMPONENTS))

    code = main(["--root", str(tmp_path), "resume", run_id, "--json"],
                wf_factory=wf_factory)
    assert code == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["status"] == "started"
    events = [e["event"] for e in wf.get_run_events(run_id)]
    assert "RECONCILED" in events and "CHECKPOINT_SKIPPED" not in events
    # ...and the reconciliation ran BEFORE resume validation (13 §6.2)
    assert events.index("RECONCILED") < events.index("resume_requested")
    assert wf.get_run_state(run_id) == "READY"


def test_resume_gate_block_leaves_state_unchanged(tmp_path, capsys, no_supervisor):
    """Gate BLOCK on resume ⇒ exit 1, no state change (13 §7). Preflight
    is satisfied here so the GATE is what refuses."""
    run_id = _paused_run(tmp_path)  # PAUSED
    wf = WorkflowAPI(tmp_path)      # default (fail-closed) gate

    class _PassPreflight:
        blocked = False
        failed_step = None
        first_failure = None

        def to_dict(self):
            return {"blocked": False}

    real = WorkflowAPI.preflight_run
    try:
        WorkflowAPI.preflight_run = lambda self, rid, **kw: _PassPreflight()
        code = main(["--root", str(tmp_path), "resume", run_id])
    finally:
        WorkflowAPI.preflight_run = real
    assert code == 1
    out = capsys.readouterr().out
    assert "RESUME BLOCKED" in out and "unchanged" in out
    assert wf.get_run_state(run_id) == "PAUSED"      # stays PAUSED (13 §7)
    assert not (tmp_path / "state" / "pending" / f"{run_id}.json").exists()


def test_resume_preflight_block_leaves_state_unchanged(tmp_path, capsys, no_supervisor):
    """Preflight runs BEFORE the gate on resume: both blocks are no-ops
    on run state ("No changes were made")."""
    run_id = _paused_run(tmp_path)

    import mlforge.cli.main as cli

    class _BlockedPreflight:
        blocked = True
        failed_step = 9
        first_failure = type("C", (), {"label": "Disk", "detail": "insufficient"})()

        def to_dict(self):
            return {"blocked": True}

        def render(self):
            return "PREFLIGHT REPORT\n[FAIL] Disk space — insufficient"

    real = WorkflowAPI.preflight_run
    try:
        WorkflowAPI.preflight_run = lambda self, rid, **kw: _BlockedPreflight()
        code = main(["--root", str(tmp_path), "resume", run_id])
    finally:
        WorkflowAPI.preflight_run = real
    assert code == 1
    assert "unchanged" in capsys.readouterr().out
    assert WorkflowAPI(tmp_path).get_run_state(run_id) == "PAUSED"
    assert not (tmp_path / "state" / "pending" / f"{run_id}.json").exists()


def test_resume_missing_run_exits_2(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "resume", "run_MISSING"]) == 2


# -- pause ---------------------------------------------------------------------

def _live_worker(tmp_path) -> tuple[WorkflowAPI, str, threading.Thread, dict]:
    wf = wf_factory(tmp_path)
    h = wf.create_run(_spec())
    assert not wf.validate_run(h.run_id).blocked
    token = RunLeaseManager(tmp_path).status(h.run_id).session_token
    worker = Worker(tmp_path, h.run_id, token,
                    trainer=ScaffoldTrainer(max_steps=10_000),
                    poll_interval=0.01, heartbeat_interval=0.05)
    box = {}

    def _run():
        box["code"] = worker.run()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    deadline = time.time() + 5
    while time.time() < deadline and wf.get_run_state(h.run_id) != "RUNNING":
        time.sleep(0.01)
    assert wf.get_run_state(h.run_id) == "RUNNING"
    return wf, h.run_id, t, box


def test_pause_running_worker_via_control(tmp_path, capsys):
    wf, run_id, t, box = _live_worker(tmp_path)
    code = main(["--root", str(tmp_path), "pause", run_id, "--timeout", "8"])
    assert code == 0
    t.join(timeout=5)
    assert box["code"] == 0
    assert wf.get_run_state(run_id) == "PAUSED"
    assert "PAUSED" in capsys.readouterr().out
    assert RunLeaseManager(tmp_path).status(run_id).state.value == "FREE"


def test_pause_already_paused_is_noop(tmp_path, capsys):
    run_id = _paused_run(tmp_path)
    assert main(["--root", str(tmp_path), "pause", run_id]) == 0
    assert "already PAUSED" in capsys.readouterr().out


def test_pause_from_created_exits_3(tmp_path, capsys):
    run_id = WorkflowAPI(tmp_path).create_run(_spec()).run_id
    assert main(["--root", str(tmp_path), "pause", run_id]) == 3
    assert "cannot pause" in capsys.readouterr().err


def test_pause_not_acknowledged_exits_3(tmp_path, capsys):
    """RUNNING on disk but no worker to obey — honest timeout (exit 3),
    never a fake PAUSED."""
    wf = wf_factory(tmp_path)
    h = wf.create_run(_spec())
    assert not wf.validate_run(h.run_id).blocked
    wf.preflight_pass(h.run_id)  # RUNNING, but no worker process
    code = main(["--root", str(tmp_path), "pause", h.run_id, "--timeout", "0.2"])
    assert code == 3
    assert "not acknowledged" in capsys.readouterr().out
    assert wf.get_run_state(h.run_id) == "RUNNING"   # intent recorded only


# -- stop ----------------------------------------------------------------------

def test_stop_paused_is_synchronous(tmp_path, capsys):
    run_id = _paused_run(tmp_path)
    code = main(["--root", str(tmp_path), "stop", run_id])
    assert code == 0
    assert WorkflowAPI(tmp_path).get_run_state(run_id) == "STOPPED"
    assert "no resume" in capsys.readouterr().out


def test_stop_running_worker_via_control(tmp_path, capsys):
    wf, run_id, t, box = _live_worker(tmp_path)
    code = main(["--root", str(tmp_path), "stop", run_id, "--timeout", "8"])
    assert code == 0
    t.join(timeout=5)
    assert box["code"] == 0
    assert wf.get_run_state(run_id) == "STOPPED"
    assert RunLeaseManager(tmp_path).status(run_id).state.value == "FREE"


def test_stop_from_created_exits_3(tmp_path, capsys):
    run_id = WorkflowAPI(tmp_path).create_run(_spec()).run_id
    assert main(["--root", str(tmp_path), "stop", run_id]) == 3
    assert "cannot stop" in capsys.readouterr().err
