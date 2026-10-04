"""Fault-injection journey harness — every operator flow must FINISH or
FAIL CLEANLY with a recovery path, never wedge mid-task (12 §12.3
crash/reconcile, 13 §6.2–§6.4 resume/retrain/finetune, §6.7–§6.10
export/package/import, §7 failure matrix).

Method: drive the REAL journeys on this box, then attack the
transitions — SIGKILL mid-train, torn checkpoint payloads, stale leases,
interrupted exports, resume/crash of terminal runs — and assert each
attack either completes the journey or refuses with guidance while
leaving state intact, with the NEXT operation still working.

  J1  real-ranker train → SIGKILL → crash → planted corruptions
      (staging leftover + truncated newest checkpoint) → reconcile
      reports them → stale-lease refusal → lease break → gate → resume →
      COMPLETED; metrics CONTINUE from the resume point (global_step
      never restarts at 0) and the model publishes with loadable weights.
  J2  fine-tune: parent v1 → plan → run → real trainer receives the
      parent's committed weights (init_weights spy) → v2 + lineage,
      parent untouched.
  J3  artifact: train → real ONNX export → package (bundle carries the
      weights) → import into a FRESH workspace → validate → infer.
  J4  interrupted export between file writes → NO phantom export record,
      honest "run `mlforge export`" guidance → retry heals the journey.
  J5  retrain: the real trainer runs with NO parent weights (fresh init
      by design) + lineage parent + v2 coexisting with v1.
  J6  guards do not wedge: resume/crash of a COMPLETED run and an
      unknown model refuse with hints, state unchanged, export still
      works afterwards.

J1/J2/J3/J5 lift the harness opt-in (monkeypatch.delenv) so
`resolve_trainer` / `export` / `infer` take the REAL lightgbm path; J4/J6
run the honest scaffold — their subject is the state machine, not the
trainer.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_gbdt_trainer import make_ranker_project, ranker_rows, sem
from test_ops import _publish

from mlforge.errors import (
    InvalidTransition,
    NotFound,
    RunAlreadyExecuting,
    ValidationBlock,
)
from mlforge.leases import LeaseState, RunLeaseManager, provide_run_lease
from mlforge.ops.engines import load_weights
from mlforge.ops.exporting import list_exports
from mlforge.planner import detect_capabilities
from mlforge.run_spec import RunSpec
from mlforge.runtime.checkpoints import CheckpointStore
from mlforge.runtime.worker import Worker
from mlforge.trainers.gbdt import FEATURE_NAMES
from mlforge.validation import RESUME_GATE_STEPS, provide_pass
from mlforge.workflow import WorkflowAPI

GATE_KEYS = [s.id for s in RESUME_GATE_STEPS if s.id not in ("manifest", "schema")]

RANKER_EPOCHS_J1 = 150  # kill window: commits at 25/50/75 of 150 steps


def _wf(root: Path) -> WorkflowAPI:
    """Journey workspace: environment-only gate steps pass (these
    journeys are not about host attestation); the single-writer lease is
    REAL — the fault attacks depend on it."""
    providers = {k: provide_pass(f"{k} verified") for k in GATE_KEYS}
    providers["lease"] = provide_run_lease(RunLeaseManager(root))
    return WorkflowAPI(root, gate_providers=providers)


def _ranker_workspace(tmp_path: Path) -> tuple[WorkflowAPI, Path]:
    root = Path(tmp_path) / "ws"
    make_ranker_project(root, ranker_rows(groups=40, per=6, seed=7, split="train"))
    return _wf(root), root


def _train_real(
    wf: WorkflowAPI, root: Path, *, epochs: int, interval: int = 1
) -> tuple[str, str, dict]:
    """create → gate → in-process REAL trainer to completion → published
    model. Returns (model ref, run_id, registry entry)."""
    h = wf.create_run(
        RunSpec(
            model="hypothesis_ranker",
            train_datasets=("ranker:v1",),
            semantic=sem(epochs=epochs),
        )
    )
    report = wf.validate_run(h.run_id)
    assert not report.blocked, report.render()
    token = RunLeaseManager(root).status(h.run_id).session_token
    assert token, "gate did not acquire the run lease"
    rc = Worker(
        root,
        h.run_id,
        token,
        poll_interval=0.0,
        heartbeat_interval=60.0,
        checkpoint_interval=interval,
    ).run()
    assert rc == 0, f"worker exited {rc}"
    assert wf.get_run_state(h.run_id) == "COMPLETED"
    entries = [m for m in wf.list_models() if m.get("run_id") == h.run_id]
    assert len(entries) == 1, entries
    entry = entries[0]
    return f"{entry['name']}:{entry['version']}", h.run_id, entry


def _spawn_worker(
    root: Path, run_id: str, token: str, *, ckpt: str = "25"
) -> subprocess.Popen:
    """The REAL worker as a subprocess (so the journey can SIGKILL it)."""
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlforge.runtime.worker",
            "--root", str(root),
            "--run", run_id,
            "--token", token,
            "--checkpoint-interval", ckpt,
            "--heartbeat-interval", "0.5",
            "--poll-interval", "0.002",
        ],
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _wait_for_commits(
    proc: subprocess.Popen, run_dir: Path, n: int, *, timeout: float = 60.0
) -> None:
    """Wait until `n` COMPLETE checkpoints (COMMIT marker present) exist;
    fail loudly if the worker dies first (a refusal must not be mistaken
    for progress)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            _out, err = proc.communicate()
            raise AssertionError(
                f"worker exited early rc={proc.returncode}: "
                f"{(err or b'').decode(errors='replace')[-2000:]}"
            )
        if len(_commits(run_dir)) >= n:
            return
        time.sleep(0.001)
    proc.kill()
    raise AssertionError(f"no {n} checkpoint commit appeared within {timeout}s")


def _commits(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("checkpoints/ckpt-*/COMMIT"))


def _metrics(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics" / "metrics.jsonl"
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _spy_build_trainer(monkeypatch) -> dict:
    """Record the weights channel resolve_trainer → build_trainer."""
    from mlforge import trainers

    captured: dict = {}
    real = trainers.build_trainer

    def spy(model, **kw):
        captured["called"] = True
        captured["init_weights"] = kw.get("init_weights")
        return real(model, **kw)

    monkeypatch.setattr(trainers, "build_trainer", spy)
    return captured


# ---------------------------------------------------------------------------
# J1 — crash mid-train: kill -9, corruption, stale lease, reconcile, resume
# ---------------------------------------------------------------------------


def test_j1_kill_mid_train_reconcile_and_resume_finishes(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    wf, root = _ranker_workspace(tmp_path)
    h = wf.create_run(
        RunSpec(
            model="hypothesis_ranker",
            train_datasets=("ranker:v1",),
            semantic=sem(epochs=RANKER_EPOCHS_J1),
        )
    )
    report = wf.validate_run(h.run_id)
    assert not report.blocked, report.render()
    lease = RunLeaseManager(root)
    token1 = lease.status(h.run_id).session_token
    assert token1
    run_dir = root / "runs" / h.run_id

    # -- train as a real subprocess, then kill -9 after ≥3 commits -------
    proc = _spawn_worker(root, h.run_id, token1, ckpt="25")
    _wait_for_commits(proc, run_dir, n=3, timeout=60.0)
    proc.send_signal(signal.SIGKILL)
    assert proc.wait(timeout=30) == -signal.SIGKILL
    proc.communicate()  # reap the pipes
    # the kill was NOT a graceful shutdown — the run is still LIVE
    # (RUNNING, or CHECKPOINTING if the kill landed mid-write)
    assert wf.get_run_state(h.run_id) in ("RUNNING", "CHECKPOINTING"), (
        wf.get_run_state(h.run_id)
    )
    n1 = len(_metrics(run_dir))
    assert n1 >= 3, f"metrics before kill: {n1}"

    # -- plant the two artifacts a real kill mid-write leaves ------------
    ckpt_dir = run_dir / "checkpoints"
    committed = sorted(d for d in ckpt_dir.glob("ckpt-*") if (d / "COMMIT").is_file())
    assert len(committed) >= 3, [d.name for d in committed]
    newest = committed[-1]
    # 1) staging leftover: killed between staging write and rename
    ordinals = [
        int(m.group(1))
        for d in ckpt_dir.iterdir()
        if (m := re.match(r"ckpt-(\d{6})", d.name))
    ]
    staging = ckpt_dir / f"ckpt-{max(ordinals) + 1:06d}.staging"
    staging.mkdir()
    (staging / "model").write_bytes(b"half-written payload: kill -9 landed here")
    # 2) torn payload: COMMIT exists but the newest model file is truncated
    torn = newest / "model"
    torn.write_bytes(torn.read_bytes()[:16])

    # -- operator flow: heartbeat expiry → crash → reconcile from disk ---
    wf.crash(h.run_id, "journey fault: worker killed with SIGKILL")
    assert wf.get_run_state(h.run_id) == "INTERRUPTED"
    rep = wf.reconcile_from_disk(h.run_id)
    assert wf.get_run_state(h.run_id) == "RECONCILING"
    reasons = [str(s.get("reason", "")) for s in rep.get("skips", [])]
    assert any("STAGING LEFTOVER" in r for r in reasons), reasons
    assert any("FILE HASH MISMATCH" in r for r in reasons), reasons
    resume_point = rep["resume_point"]
    assert resume_point and resume_point != newest.name, (
        "the torn checkpoint was chosen as the resume point",
        resume_point,
        newest.name,
    )
    sel = CheckpointStore(run_dir).newest_valid()
    assert sel.selected is not None and sel.selected.path.name == resume_point
    resume_step = int(sel.selected.manifest["global_step"])
    assert 0 < resume_step < RANKER_EPOCHS_J1

    # -- stale lease ≠ free: the dead worker still guards the run --------
    with pytest.raises(RunAlreadyExecuting):
        wf.resume(h.run_id)
    assert wf.get_run_state(h.run_id) == "RECONCILING"  # refusal changed nothing

    # operator confirms the worker is dead → explicit break (12 §23.2)
    lease.force_release(h.run_id)
    assert lease.status(h.run_id).state == LeaseState.FREE

    # -- resume journey: preflight → gate → READY with a FRESH lease -----
    pre = wf.preflight_run(
        h.run_id, gpu_required=detect_capabilities().gpu_count > 0
    )
    assert not pre.blocked, pre.render()
    report = wf.validate_run(h.run_id)
    assert not report.blocked, report.render()
    assert wf.get_run_state(h.run_id) == "READY"
    token2 = lease.status(h.run_id).session_token
    assert token2 and token2 != token1

    # -- second worker finishes the SAME run from the resume point -------
    proc2 = _spawn_worker(root, h.run_id, token2, ckpt="25")
    _out, err = proc2.communicate(timeout=180)
    assert proc2.returncode == 0, (err or b"").decode(errors="replace")[-2000:]
    assert wf.get_run_state(h.run_id) == "COMPLETED"

    # metrics CONTINUE — never restart at 0, exactly from the resume point
    steps = [int(m["global_step"]) for m in _metrics(run_dir)]
    assert steps and min(steps) >= 1, steps[:5]
    tail = steps[n1:]
    assert tail, "the resumed worker wrote no metrics"
    assert tail[0] == resume_step + 1, (tail[:5], resume_step)
    assert tail == sorted(tail), "resume leg is not monotone"
    assert steps[-1] == RANKER_EPOCHS_J1, steps[-5:]

    final = CheckpointStore(run_dir).newest_valid()
    assert final.selected is not None
    assert int(final.selected.manifest["global_step"]) == RANKER_EPOCHS_J1

    # the journey ends with a publishable, loadable model and a clean
    # terminal workspace (lease released, live state gone)
    entries = [m for m in wf.list_models() if m.get("run_id") == h.run_id]
    assert len(entries) == 1
    assert load_weights(root, entries[0])
    assert lease.status(h.run_id).state == LeaseState.FREE
    assert not (run_dir / "state" / "live.json").exists()


# ---------------------------------------------------------------------------
# J2 — fine-tune: parent weights reach the REAL trainer
# ---------------------------------------------------------------------------


def test_j2_finetune_journey_parent_weights_reach_trainer(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    wf, root = _ranker_workspace(tmp_path)
    ref1, run1, entry1 = _train_real(wf, root, epochs=6)

    captured = _spy_build_trainer(monkeypatch)
    plan = wf.preview_finetune(ref1, strategy="full", overrides={"epochs": 4})
    assert plan.kind == "finetune" and plan.source_run_id == run1
    assert plan.lineage.get("origin") == "finetune"
    assert plan.lineage.get("parent_model") == ref1
    h2 = wf.create_from_plan(plan)
    report = wf.validate_run(h2.run_id)
    assert not report.blocked, report.render()
    token = RunLeaseManager(root).status(h2.run_id).session_token
    assert token
    rc = Worker(
        root, h2.run_id, token, poll_interval=0.0, heartbeat_interval=60.0,
        checkpoint_interval=1,
    ).run()
    assert rc == 0
    assert wf.get_run_state(h2.run_id) == "COMPLETED"

    # the parent's committed model bytes REACHED the real trainer
    assert captured.get("called"), "real build_trainer never ran"
    parent_sel = CheckpointStore(root / "runs" / run1).newest_valid()
    assert parent_sel.selected is not None
    assert captured["init_weights"] == (parent_sel.selected.path / "model").read_bytes()

    # new model + lineage; the base model is never modified (13 §6.4)
    entries = {f"{m['name']}:{m['version']}": m for m in wf.list_models()}
    assert ref1 in entries
    assert entries[ref1]["artifact_hash"] == entry1["artifact_hash"]
    ref2 = [k for k, m in entries.items() if m.get("run_id") == h2.run_id]
    assert len(ref2) == 1 and ref2[0] != ref1
    chain = wf.lineage_chain(h2.run_id)
    assert [c["run_id"] for c in chain] == [run1, h2.run_id]
    assert load_weights(root, entries[ref2[0]])


# ---------------------------------------------------------------------------
# J3 — export → package → import (fresh workspace) → validate → infer
# ---------------------------------------------------------------------------


def test_j3_export_package_import_validate_infer_journey(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    src = Path(tmp_path) / "src"
    make_ranker_project(src, ranker_rows(groups=16, per=6, seed=7, split="train"))
    wf = _wf(src)
    ref, _run, _entry = _train_real(wf, src, epochs=6)

    # export: real ONNX bytes + contract sidecar on disk
    rec = wf.export_model(ref, "onnx")
    export_dir = src / "exports" / rec["export_id"]
    binary = export_dir / rec["file_name"]
    assert binary.is_file() and binary.stat().st_size > 500
    assert (export_dir / "export.json").is_file()
    assert (export_dir / "model_spec.json").is_file()

    # package: the bundle carries the REAL weights (12 §15.3) — without
    # them `model import` could never consume it
    bundle = wf.package_model(ref)
    bdir = src / "bundles" / bundle["bundle_id"]
    pkg_weights = (bdir / "model.safetensors").read_bytes()
    assert pkg_weights
    assert (bdir / "model_spec.json").is_file()

    # the journey's point: a DIFFERENT workspace consumes the package
    ws2 = Path(tmp_path) / "importer"
    wf2 = _wf(ws2)
    imported = wf2.import_external_model(str(bdir))
    assert imported.get("origin") == "import"
    stored = ws2 / "models" / str(imported["model_id"]) / "model.safetensors"
    assert stored.read_bytes() == pkg_weights  # hash-verified at import

    verdict = wf2.model_integrity_report(ref)
    assert not verdict["blocked"], verdict["checks"]
    assert all(c["status"] == "PASS" for c in verdict["checks"]), verdict["checks"]

    # infer on the IMPORTED model: sidecar contract + registry weights
    query = {
        "hypotheses": [
            {c: 0.5 for c in FEATURE_NAMES} | {"r_score": 0.9},
            {c: 0.1 for c in FEATURE_NAMES} | {"r_score": 0.2},
        ]
    }
    qpath = ws2 / "query.json"
    qpath.write_text(json.dumps(query))
    out = wf2.infer_model(ref, str(qpath))
    scores = (out.get("result") or {}).get("scores")
    assert isinstance(scores, list) and len(scores) == 2, out
    assert all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores)

    # import never touched the source workspace — the original still works
    out0 = wf.infer_model(ref, str(qpath))
    assert (out0.get("result") or {}).get("scores")


# ---------------------------------------------------------------------------
# J4 — interrupted export: no phantom record, honest guidance, retry heals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fault_at", ["model_spec.json", "export.json"])
def test_j4_interrupted_export_no_phantom_record_then_retry_heals(
    tmp_path, monkeypatch, fault_at
):
    root = Path(tmp_path) / "ws"
    wf = _wf(root)
    entry = _publish(wf)
    ref = f"{entry['name']}:{entry['version']}"

    # crash between file writes: the atomic rename of `fault_at` fails
    real_replace = Path.replace

    def _boom(self, target):
        if str(target).endswith(fault_at):
            raise OSError(28, f"journey fault: ENOSPC writing {fault_at}")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _boom)
    with pytest.raises(OSError, match="journey fault"):
        wf.export_model(ref, "onnx")
    monkeypatch.setattr(Path, "replace", real_replace)  # "power restored"

    # nothing claims an export exists — the failed attempt is invisible
    assert list_exports(root) == [], list_exports(root)
    assert not list((root / "exports").glob("*/export.json"))
    # the contract gate still gives the recovery path (not a wedge)
    with pytest.raises(ValidationBlock) as exc:
        wf.package_model(ref)
    assert "mlforge export" in exc.value.render()

    # retry succeeds and yields exactly one COMPLETE record
    rec = wf.export_model(ref, "onnx")
    d = root / "exports" / rec["export_id"]
    for filename in ("export.json", "model_spec.json"):
        assert (d / filename).is_file(), filename
    # scaffold exports write no fake binary (12 §12.4); a payload with a
    # binary_hash means a real engine produced the named artifact file
    if (rec.get("payload") or {}).get("binary_hash"):
        assert (d / rec["file_name"]).is_file()
    assert len(list_exports(root)) == 1

    # and the journey continues: packaging after recovery works
    bundle = wf.package_model(ref)
    assert (root / "bundles" / bundle["bundle_id"] / "model.safetensors").is_file()


# ---------------------------------------------------------------------------
# J5 — retrain: fresh init (no parent weights) + lineage parent + v2
# ---------------------------------------------------------------------------


def test_j5_retrain_journey_fresh_init_with_parent_lineage(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFORGE_HARNESS", raising=False)
    wf, root = _ranker_workspace(tmp_path)
    ref1, run1, entry1 = _train_real(wf, root, epochs=6)

    captured = _spy_build_trainer(monkeypatch)
    plan = wf.preview_retrain("hypothesis_ranker", overrides={"epochs": 8})
    assert plan.kind == "retrain" and plan.source_run_id == run1
    assert plan.lineage.get("origin") == "retrain"
    assert plan.lineage.get("parent_run") == run1
    h2 = wf.create_from_plan(plan)
    report = wf.validate_run(h2.run_id)
    assert not report.blocked, report.render()
    token = RunLeaseManager(root).status(h2.run_id).session_token
    assert token
    rc = Worker(
        root, h2.run_id, token, poll_interval=0.0, heartbeat_interval=60.0,
        checkpoint_interval=1,
    ).run()
    assert rc == 0
    assert wf.get_run_state(h2.run_id) == "COMPLETED"

    # 13 §6.3 fresh init: the REAL trainer ran, with NO parent weights
    assert captured.get("called"), "real build_trainer never ran"
    assert captured.get("init_weights") is None, "retrain must not inherit weights"

    # both models coexist; the parent entry is untouched
    entries = {f"{m['name']}:{m['version']}": m for m in wf.list_models()}
    assert ref1 in entries
    assert entries[ref1]["artifact_hash"] == entry1["artifact_hash"]
    ref2 = [k for k, m in entries.items() if m.get("run_id") == h2.run_id]
    assert len(ref2) == 1 and ref2[0] != ref1
    assert [c["run_id"] for c in wf.lineage_chain(h2.run_id)] == [run1, h2.run_id]
    assert load_weights(root, entries[ref2[0]])


# ---------------------------------------------------------------------------
# J6 — guards refuse cleanly and the system keeps working
# ---------------------------------------------------------------------------


def test_j6_guards_refuse_cleanly_and_system_keeps_working(tmp_path):
    root = Path(tmp_path) / "ws"
    wf = _wf(root)
    entry = _publish(wf)
    ref = f"{entry['name']}:{entry['version']}"
    run_id = entry["run_id"]

    # resume of a COMPLETED run → honest refusal listing legal actions
    with pytest.raises(InvalidTransition) as exc:
        wf.resume(run_id)
    assert "legal actions" in exc.value.render()
    # crash of a COMPLETED run → same class of refusal
    with pytest.raises(InvalidTransition):
        wf.crash(run_id, "operator error: crash on a terminal run")
    # both refusals left the terminal state and lease untouched
    assert wf.get_run_state(run_id) == "COMPLETED"
    assert RunLeaseManager(root).status(run_id).state == LeaseState.FREE

    # unknown model → NotFound with discovery guidance (exit 2, not a wedge)
    with pytest.raises(NotFound) as exc2:
        wf.export_model("no_such_net:v9", "onnx")
    assert "model list" in exc2.value.render()

    # after every refusal, the next operation still works
    rec = wf.export_model(ref, "onnx")
    assert (root / "exports" / rec["export_id"] / "export.json").is_file()
