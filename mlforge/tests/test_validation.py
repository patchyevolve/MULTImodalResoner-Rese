"""Build step 3 tests — validation gate + preflight (fail-closed core).

Specs as executable checks:
  * missing provider / throwing provider / no verdict ⇒ FAIL (unverifiable
    never passes) — 12 §7.1 "probably compatible does not exist"
  * first FAIL stops the run; side-effect step 15 (lease) never runs after
    an earlier failure — 12 §18 "Any failure in 1–16 → training does not
    start"
  * EXACT only on explicit exact_compatible signals; otherwise PORTABLE
  * train path: gate BLOCK → FAILED[FORK_ONLY], never READY (13 §7)
  * resume path: gate BLOCK → NO state change, "No changes were made"
    (13 §7: Stays PAUSED / stays FAILED)
  * preflight: disk worst-case mandatory; host probes before identity;
    any FAIL ⇒ PREFLIGHT FAILED, report-only (no RUNNING without runtime)
"""

from __future__ import annotations

import json

import pytest

from mlforge.errors import NoValidContinuation
from mlforge.run_spec import RunSpec
from mlforge.validation import (
    FAIL,
    PASS,
    Check,
    GateContext,
    Preflight,
    PreflightContext,
    ValidationGate,
    make_gate,
    provide_fail,
    provide_pass,
)
from mlforge.validation.gate import (
    RESUME_GATE_STEPS,
    _builtin_manifest,
    _builtin_schema,
)
from mlforge.workflow import WorkflowAPI

#: Every step that needs an external provider (3..16).
PROVIDER_KEYS = [s.id for s in RESUME_GATE_STEPS if s.id not in ("manifest", "schema")]


def full_pass_providers(**per_key_overrides):
    """PASS providers for steps 3–16 (tests may override individual keys)."""
    p = {k: provide_pass(f"{k} verified") for k in PROVIDER_KEYS}
    p.update(per_key_overrides)
    return p


def make_spec() -> RunSpec:
    return RunSpec(
        model="rf_detr_s",
        train_datasets=("coco_2017:v1",),
        semantic={
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "scheduler": "cosine",
            "loss": "l1",
            "seed": 42,
            "global_batch": 32,
            "epochs": 50,
            "precision_policy": "bf16",
        },
    )


@pytest.fixture
def wf(tmp_path):
    return WorkflowAPI(tmp_path)


def _run(wf) -> str:
    return wf.create_run(make_spec()).run_id


# ---------------------------------------------------------------------------
# Gate engine: fail-closed core
# ---------------------------------------------------------------------------

def test_missing_provider_fails_unverifiable(wf):
    run_id = _run(wf)
    report = ValidationGate().run(
        GateContext(run_id, wf.root, make_spec(), flow="TRAIN")
    )
    assert report.blocked
    f = report.first_failure
    assert f.step == 3 and f.id == "source_code"
    assert "unverifiable" in f.detail and "fail-closed" in f.detail
    assert report.mode is None  # blocked reports carry no mode


def test_all_providers_pass(wf):
    run_id = _run(wf)
    gate = make_gate(full_pass_providers())
    report = gate.run(GateContext(run_id, wf.root, make_spec(), flow="RESUME"))
    assert not report.blocked
    assert report.mode == "PORTABLE"  # no exact signals → safe default
    assert report.result == "SAFE TO RESUME"
    assert len(report.checks) == 16


def test_exact_mode_requires_all_signals_true(wf):
    run_id = _run(wf)
    exact = {
        k: provide_pass(f"{k} identical", exact_compatible=True)
        for k in PROVIDER_KEYS
    }
    report = ValidationGate(exact).run(
        GateContext(run_id, wf.root, make_spec())
    )
    assert not report.blocked and report.mode == "EXACT"

    # one explicit False ⇒ never EXACT (12 §8: PORTABLE is the fallback)
    exact["topology"] = provide_pass("world size changed", exact_compatible=False)
    report = ValidationGate(exact).run(
        GateContext(run_id, wf.root, make_spec())
    )
    assert report.mode == "PORTABLE"


def test_first_fail_stops_and_lease_never_runs(wf):
    """Side-effect ordering: step 15 (lease) must not execute after an
    earlier failure (12 §18)."""
    run_id = _run(wf)
    lease_calls = []

    def lease_provider(ctx):
        lease_calls.append(ctx.run_id)
        return Check(15, "lease", "Acquire run lease (single-writer)", PASS, "held")

    providers = full_pass_providers(
        dataset=provide_fail(
            "content hash mismatch: expected sha256:abc, found sha256:def"
        ),
        lease=lease_provider,
    )
    report = ValidationGate(providers).run(
        GateContext(run_id, wf.root, make_spec())
    )
    assert report.blocked and report.failed_step == 4
    assert lease_calls == []          # never acquired
    steps = [c.step for c in report.checks]
    assert steps == [1, 2, 3, 4]      # nothing after the first FAIL ran


def test_throwing_provider_is_fail_not_crash(wf):
    run_id = _run(wf)

    def boom(ctx):
        raise RuntimeError("provider exploded")

    report = ValidationGate(full_pass_providers(transform=boom)).run(
        GateContext(run_id, wf.root, make_spec())
    )
    assert report.blocked
    f = report.first_failure
    assert f.id == "transform"
    assert "provider exploded" in f.detail


def test_provider_returning_none_fails(wf):
    run_id = _run(wf)
    report = ValidationGate(full_pass_providers(model=lambda ctx: None)).run(
        GateContext(run_id, wf.root, make_spec())
    )
    assert report.blocked and report.first_failure.id == "model"


def test_invalid_verdict_rejected():
    with pytest.raises(ValueError):
        Check(1, "x", "X", "MAYBE")  # type: ignore[arg-type]


def test_builtin_manifest_passes_on_real_run(wf):
    run_id = _run(wf)
    ctx = GateContext(run_id, wf.root, make_spec(), flow="TRAIN")
    c = _builtin_manifest(ctx)
    assert c.verdict == PASS and "sha256:" in c.detail


def test_builtin_manifest_detects_journal_tamper(wf):
    run_id = _run(wf)
    ev_path = wf.root / "runs" / run_id / "events.jsonl"
    lines = ev_path.read_text(encoding="utf-8").splitlines()
    tampered = []
    for line in lines:
        ev = json.loads(line)
        if "run_spec_hash" in ev:
            ev["run_spec_hash"] = "sha256:" + "0" * 64
        tampered.append(json.dumps(ev))
    ev_path.write_text("\n".join(tampered) + "\n", encoding="utf-8")
    c = _builtin_manifest(GateContext(run_id, wf.root, make_spec()))
    assert c.verdict == FAIL and "journal recorded" in c.detail


def test_builtin_manifest_missing_file(wf):
    run_id = _run(wf)
    (wf.root / "runs" / run_id / "run_spec.json").unlink()
    c = _builtin_manifest(GateContext(run_id, wf.root, make_spec()))
    assert c.verdict == FAIL and "missing" in c.detail


def test_builtin_schema_unsupported_version():
    bad = RunSpec(
        model="m",
        train_datasets=("d:v1",),
        semantic={k: 1 for k in (
            "optimizer", "learning_rate", "scheduler", "loss",
            "seed", "global_batch", "epochs", "precision_policy",
        )},
        schema_version=99,
    )
    c = _builtin_schema(GateContext("run_x", "/tmp", bad))
    assert c.verdict == FAIL and "unsupported" in c.detail


def test_report_render_format(wf):
    run_id = _run(wf)
    report = make_gate(full_pass_providers()).run(
        GateContext(run_id, wf.root, make_spec(), flow="RESUME", segment="0002")
    )
    text = report.render()
    assert "VALIDATION REPORT" in text
    assert "RESULT:" in text
    assert "SAFE TO RESUME" in text
    assert "MODE: PORTABLE CONTINUATION" in text
    assert "SEGMENT: 0002" in text

    blocked = ValidationGate().run(
        GateContext(run_id, wf.root, make_spec(), flow="RESUME")
    )
    text = blocked.render()
    assert "RESUME BLOCKED" in text
    assert "[FAIL] Source artifact + code hash" in text
    assert "MODE:" not in text  # no mode on a blocked report


# ---------------------------------------------------------------------------
# Workflow: train path (CREATED → VALIDATING → READY | FAILED)
# ---------------------------------------------------------------------------

def test_train_path_gate_pass_reaches_ready(wf):
    run_id = _run(wf)
    report = wf.validate_run(run_id, make_gate(full_pass_providers()))
    assert not report.blocked
    assert wf.get_run_state(run_id) == "READY"
    events = [e["event"] for e in wf.get_run_events(run_id)]
    assert events == ["run_created", "validation_started", "validation_passed"]
    # report is journaled with the pass event (audit trail)
    passed = wf.get_run_events(run_id)[-1]
    assert passed["report"]["result"] == "SAFE TO TRAIN"


def test_train_path_gate_block_is_failed_fork_only(wf):
    """13 §7: VALIDATING → READY, any invariant fails → FAILED (FORK_ONLY,
    exit 1). Never reaches READY."""
    run_id = _run(wf)
    gate = make_gate(full_pass_providers(dataset=provide_fail("hash mismatch")))
    report = wf.validate_run(run_id, gate)
    assert report.blocked
    assert wf.get_run_state(run_id) == "FAILED"
    status = wf.get_run_status(run_id)
    assert status["failure"]["recovery"] == "FORK_ONLY"
    assert "gate BLOCK at step 4" in status["failure"]["cause"]
    events = [e["event"] for e in wf.get_run_events(run_id)]
    assert "validation_blocked" in events and "validation_failed" in events
    # resume must then be refused with exit 3 semantics
    with pytest.raises(NoValidContinuation):
        wf.resume(run_id)


# ---------------------------------------------------------------------------
# Workflow: resume path (BLOCK = no state change)
# ---------------------------------------------------------------------------

def _parked_run(wf, state: str) -> str:
    """Create a run and park it in `state` (PAUSED or FAILED[RESUME])."""
    run_id = _run(wf)
    wf.begin_validation(run_id)
    wf.validation_pass(run_id)
    wf.preflight_pass(run_id)  # READY → RUNNING
    assert wf.get_run_state(run_id) == "RUNNING"
    if state == "PAUSED":
        wf.pause(run_id)
        wf.pause_committed(run_id, "ckpt-000001")
    elif state == "FAILED_RESUME":
        wf.runtime_error(run_id, "CUDA OOM at step 41200")
        assert wf.get_run_status(run_id)["failure"]["recovery"] == "RESUME"
    else:
        raise ValueError(state)
    return run_id


def test_resume_path_block_stays_paused(wf):
    """13 §7: resume, dataset hash mismatch → BLOCK. Stays PAUSED.
    Exit 1. No changes."""
    run_id = _parked_run(wf, "PAUSED")
    gate = make_gate(full_pass_providers(dataset=provide_fail("different content")))
    before = len(wf.get_run_events(run_id))
    report = wf.validate_run(run_id, gate)
    assert report.blocked
    assert wf.get_run_state(run_id) == "PAUSED"          # unchanged
    events = wf.get_run_events(run_id)
    assert [e["event"] for e in events[before:]] == ["validation_blocked"]
    assert events[-1].get("to") is None                   # no transition recorded
    # run is still resumable once the cause is fixed
    report2 = wf.validate_run(run_id, make_gate(full_pass_providers()))
    assert not report2.blocked
    assert wf.get_run_state(run_id) == "READY"
    resumed = [e["event"] for e in wf.get_run_events(run_id)]
    assert resumed[-2:] == ["resume_requested", "validation_passed"]


def test_resume_path_block_stays_failed_resume(wf):
    """13 §7: FAILED(resume: RESUME), cause unresolved → BLOCK, stays
    FAILED (disposition intact)."""
    run_id = _parked_run(wf, "FAILED_RESUME")
    gate = make_gate(full_pass_providers(environment=provide_fail("image missing")))
    report = wf.validate_run(run_id, gate)
    assert report.blocked
    assert wf.get_run_state(run_id) == "FAILED"
    assert wf.get_run_status(run_id)["failure"]["recovery"] == "RESUME"

    # cause fixed → gate passes → resume succeeds and failure clears
    report2 = wf.validate_run(run_id, make_gate(full_pass_providers()))
    assert not report2.blocked
    assert wf.get_run_state(run_id) == "READY"
    assert wf.get_run_status(run_id)["failure"] is None


def test_reconciling_resume_needs_valid_checkpoint(wf):
    run_id = _run(wf)
    wf.begin_validation(run_id)
    wf.validation_pass(run_id)
    wf.preflight_pass(run_id)
    wf.crash(run_id, "power loss")
    wf.reconcile(run_id, has_valid_checkpoint=True, resume_point="ckpt-17")
    assert wf.get_run_state(run_id) == "RECONCILING"
    report = wf.validate_run(run_id, make_gate(full_pass_providers()))
    assert not report.blocked
    assert wf.get_run_state(run_id) == "READY"


def test_report_only_states_never_move(wf):
    run_id = _parked_run(wf, "PAUSED")
    wf.resume(run_id)
    wf.validation_pass(run_id)
    wf.preflight_pass(run_id)
    assert wf.get_run_state(run_id) == "RUNNING"
    gate = make_gate(full_pass_providers(dataset=provide_fail("mismatch")))
    report = wf.validate_run(run_id, gate)
    assert report.blocked
    assert wf.get_run_state(run_id) == "RUNNING"  # report-only: untouched


def test_validate_missing_run_exits_2_path(wf):
    from mlforge.errors import NotFound

    with pytest.raises(NotFound):
        wf.validate_run("run_MISSING")


# ---------------------------------------------------------------------------
# Preflight (12 §7.3)
# ---------------------------------------------------------------------------

def _ok_probe(verdict=PASS, detail="ok"):
    def _p(ctx: PreflightContext):
        return verdict, detail, {}
    return _p


def _host_ok():
    return {key: _ok_probe() for key, _, _ in
            __import__("mlforge.validation.preflight", fromlist=["HOST_PROBES"]).HOST_PROBES}


def test_preflight_all_pass(wf):
    run_id = _run(wf)
    pf = Preflight(
        identity_providers=full_pass_providers(),
        probes=_host_ok(),
    )
    report = wf.preflight_run(run_id, pf)
    assert not report.blocked
    assert report.result == "PREFLIGHT PASSED"
    assert report.title == "PREFLIGHT REPORT"
    # report-only: no state change, no event (runtime wires the transition)
    assert wf.get_run_state(run_id) == "CREATED"
    assert [e["event"] for e in wf.get_run_events(run_id)] == ["run_created"]


def test_preflight_disk_fail_stops_before_identity(wf):
    """Mandatory worst-case disk check fails → identity probes never run."""
    run_id = _run(wf)
    identity_calls = []

    def tracking_provider(ctx):
        identity_calls.append(1)
        return Check(0, "", "", PASS, "x")

    probes = _host_ok()
    probes["disk"] = _ok_probe(FAIL, "worst-case need 999 GiB, free 1 GiB — insufficient")
    pf = Preflight(
        identity_providers={k: tracking_provider for k in PROVIDER_KEYS},
        probes=probes,
    )
    report = wf.preflight_run(run_id, pf)
    assert report.blocked
    assert report.first_failure.id == "disk"
    assert identity_calls == []


def test_preflight_identity_missing_provider_fails(wf):
    run_id = _run(wf)
    pf = Preflight(probes=_host_ok())  # no identity providers
    report = wf.preflight_run(run_id, pf)
    assert report.blocked
    assert report.first_failure.id == "source_code"
    assert "unverifiable" in report.first_failure.detail


def test_preflight_default_disk_calculation():
    ctx = PreflightContext(
        run_id="run_x", root="/tmp", run_spec=make_spec(),
        checkpoint_bytes=4 << 30, dataset_cache_bytes=1 << 30,
        log_bytes=2 << 30, safety_margin_bytes=2 << 30,
    )
    # 2×ckpt + cache + logs + margin = 8+1+2+2 = 13 GiB
    assert ctx.required_disk_bytes == 13 * (1 << 30)


def test_preflight_failure_hints_name_the_runtime_knobs(tmp_path):
    """Disk/RAM FAILs must tell a small-laptop user exactly what to set."""
    from mlforge.validation.preflight import probe_disk, probe_ram

    ctx = PreflightContext(
        run_id="run_x", root=tmp_path, run_spec=make_spec(),
        checkpoint_bytes=1 << 40,  # unreachable worst case ⇒ FAIL anywhere
    )
    verdict, detail, _ = probe_disk(ctx)
    assert verdict == FAIL and "runtime.checkpoint_bytes" in detail
    assert "runtime.safety_margin_bytes" in detail

    ctx_ram = PreflightContext(
        run_id="run_x", root=tmp_path, run_spec=make_spec(),
        min_ram_bytes=1 << 45,  # 32 TiB ⇒ fails on any real host
    )
    verdict, detail, _ = probe_ram(ctx_ram)
    assert verdict == FAIL and "runtime.min_ram_bytes" in detail


def test_preflight_runtime_min_ram_reaches_the_context(wf):
    """state/runtime.json (train config) lowers the RAM floor the same way
    it lowers disk — one story for gate step 16 and preflight (12 §7.3)."""
    run_id = _run(wf)
    state = wf.root / "runs" / run_id / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "runtime.json").write_text(json.dumps({
        "min_ram_bytes": 1 << 45,  # force a FAIL no default 8 GiB floor would
    }), encoding="utf-8")
    probes = _host_ok()
    del probes["ram"]  # let the real probe read ctx.min_ram_bytes
    report = wf.preflight_run(run_id, Preflight(probes=probes))
    assert report.blocked and report.first_failure.id == "ram"
    assert "runtime.min_ram_bytes" in report.first_failure.detail
    assert "32768.0 GiB" in report.first_failure.detail


def test_preflight_gpu_required_fails_without_smi(wf, monkeypatch):
    """gpu_required and no nvidia-smi ⇒ FAIL (unverifiable, fail-closed)."""
    import mlforge.validation.preflight as pfmod

    monkeypatch.setattr(pfmod.shutil, "which", lambda name: None)
    run_id = _run(wf)
    probes = _host_ok()
    del probes["gpu"]  # fall back to the real probe
    del probes["driver"]
    pf = Preflight(probes=probes)
    report = wf.preflight_run(run_id, pf, gpu_required=True)
    assert report.blocked
    assert report.first_failure.id == "gpu"
    assert "fail-closed" in report.first_failure.detail


def test_preflight_gpu_not_required_warns_not_blocks(wf):
    run_id = _run(wf)
    probes = _host_ok()
    del probes["gpu"]       # real probe must decide (WARN when not required)
    del probes["driver"]
    pf = Preflight(probes=probes)
    report = wf.preflight_run(run_id, pf, gpu_required=False)
    gpu_check = next(c for c in report.checks if c.id == "gpu")
    assert gpu_check.verdict == "WARN"          # WARN never blocks (§7.1)
    assert report.first_failure.id == "source_code"  # identity still fail-closed
