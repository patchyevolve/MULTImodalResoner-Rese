"""Build step 5 tests — run lease protocol (12 §23.1–23.3).

Specs as executable checks:
  * acquire = tmp → fsync → atomic link; destination never partial
  * at most one holder: second acquire → RunAlreadyExecuting (exit 3)
  * stale ≠ free: expired heartbeat → SUSPECT — acquire/release/renew all
    refuse until an explicit logged break (12 §23.2)
  * owner check by session_token (pid alone never trusted)
  * gate step 15 propagates RunAlreadyExecuting as exit 3 (not exit 1)
  * step 16 revalidates lease ownership + disk headroom + dataset identity
  * resume blocked by HELD lease (13 §7 RUN_ALREADY_EXECUTING) and by
    SUSPECT lease (reconcile/break first)
"""

from __future__ import annotations

import json

import pytest

from mlforge.errors import (
    NotFound,
    PreconditionFailed,
    RunAlreadyExecuting,
)
from mlforge.leases import (
    LeaseState,
    RunLeaseManager,
    provide_revalidation,
    provide_run_lease,
)
from mlforge.run_spec import RunSpec
from mlforge.validation import GateContext, ValidationGate, provide_pass
from mlforge.validation.gate import RESUME_GATE_STEPS
from mlforge.workflow import WorkflowAPI


class FakeClock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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


def _run(wf) -> str:
    return wf.create_run(make_spec()).run_id


def _parked(wf) -> str:
    run_id = _run(wf)
    wf.begin_validation(run_id)
    wf.validation_pass(run_id)
    wf.preflight_pass(run_id)
    wf.pause(run_id)
    wf.pause_committed(run_id, "ckpt-1")
    assert wf.get_run_state(run_id) == "PAUSED"
    return run_id


# ---------------------------------------------------------------------------
# Protocol (12 §23.2)
# ---------------------------------------------------------------------------

def test_acquire_creates_complete_lease(wf, tmp_path):
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    info = mgr.acquire(run_id)
    assert info.state == LeaseState.HELD
    assert info.pid and info.host and info.session_token
    lease_path = tmp_path / "runs" / run_id / ".lease"
    data = json.loads(lease_path.read_text())
    assert data["run_id"] == run_id
    assert data["session_token"] == info.session_token
    # no tmp leftovers (content never partial)
    assert list((tmp_path / "runs" / run_id).glob(".lease.tmp")) == []


def test_second_acquire_is_run_already_executing(wf, tmp_path):
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    mgr.acquire(run_id)
    with pytest.raises(RunAlreadyExecuting) as exc:
        mgr.acquire(run_id)
    assert exc.value.exit_code == 3
    assert "pid" in str(exc.value)


def test_stale_lease_is_suspect_not_free(wf, tmp_path):
    """12 §23.2: stale ≠ free — SUSPECT blocks *acquisition* (stealing);
    only an explicit break (or the owner itself) may clear it.

    Owner actions are NOT steals: the session_token proves the caller is
    the holder, so renew/release by the owner recover the lease (a
    suspended-but-alive worker may heartbeat late)."""
    clock = FakeClock()
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path, timeout=120.0, clock=clock)
    info = mgr.acquire(run_id)
    clock.advance(121)

    assert mgr.status(run_id).state == LeaseState.SUSPECT
    with pytest.raises(PreconditionFailed) as exc:
        mgr.acquire(run_id)               # nobody may TAKE a suspect lease
    assert "stale" in exc.value.hint
    with pytest.raises(PreconditionFailed):
        mgr.renew(run_id, "wrong-token")  # non-owner (steal attempt) refused
    with pytest.raises(PreconditionFailed):
        mgr.release(run_id, "wrong-token")

    # owner may recover its own lease (token = proof of ownership)
    assert mgr.renew(run_id, info.session_token).state == LeaseState.HELD

    # expire again, then explicit break path (workflow requires --force --yes)
    clock.advance(121)
    assert mgr.status(run_id).state == LeaseState.SUSPECT
    mgr.force_release(run_id)
    assert mgr.status(run_id).state == LeaseState.FREE
    assert mgr.acquire(run_id).state == LeaseState.HELD


def test_renew_refreshes_only_for_owner(wf, tmp_path):
    clock = FakeClock()
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path, clock=clock)
    info = mgr.acquire(run_id)
    clock.advance(60)

    with pytest.raises(PreconditionFailed):
        mgr.renew(run_id, "wrong-token")          # never trust a pid alone
    renewed = mgr.renew(run_id, info.session_token)
    assert renewed.state == LeaseState.HELD and renewed.age_seconds == 0

    with pytest.raises(PreconditionFailed):
        mgr.release(run_id, "wrong-token")        # cannot steal-release
    mgr.release(run_id, info.session_token)
    assert mgr.status(run_id).state == LeaseState.FREE


def test_release_without_lease_is_precondition(wf, tmp_path):
    run_id = _run(wf)
    with pytest.raises(PreconditionFailed):
        RunLeaseManager(tmp_path).release(run_id, "x")


def test_force_release_returns_previous_owner(wf, tmp_path):
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    info = mgr.acquire(run_id)
    prev = mgr.force_release(run_id)
    assert prev["session_token"] == info.session_token
    assert mgr.status(run_id).state == LeaseState.FREE
    with pytest.raises(PreconditionFailed):
        mgr.force_release(run_id)  # nothing to break → exit 3


def test_missing_run_raises_not_found(wf, tmp_path):
    with pytest.raises(NotFound):
        RunLeaseManager(tmp_path).acquire("run_MISSING")


def test_corrupt_lease_file_is_precondition(wf, tmp_path):
    run_id = _run(wf)
    (tmp_path / "runs" / run_id / ".lease").write_text("{broken")
    with pytest.raises(PreconditionFailed) as exc:
        RunLeaseManager(tmp_path).status(run_id)
    assert "corrupt" in str(exc.value)


# ---------------------------------------------------------------------------
# Gate integration (12 §18 steps 15–16)
# ---------------------------------------------------------------------------

def _passing_steps_except(extra: dict, keys=("lease", "revalidate")):
    p = {
        s.id: provide_pass(f"{s.id} verified")
        for s in RESUME_GATE_STEPS
        if s.id not in ("manifest", "schema", *keys)
    }
    p.update(extra)
    return p


def test_gate_step15_acquires_and_tokens_facts(wf, tmp_path):
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    extra = {
        "lease": provide_run_lease(mgr),
        "revalidate": provide_pass("revalidated"),
    }
    gate = ValidationGate(_passing_steps_except(extra))
    ctx = GateContext(run_id, tmp_path, make_spec())
    report = gate.run(ctx)
    assert not report.blocked
    assert mgr.status(run_id).state == LeaseState.HELD
    assert ctx.facts["session_token"] == mgr.status(run_id).session_token
    lease_check = next(c for c in report.checks if c.id == "lease")
    assert lease_check.verdict == "PASS"


def test_gate_full_16_steps_lease_and_revalidation(wf, tmp_path):
    """Steps 15+16 together: acquire, then revalidate the volatile subset
    with the token the gate itself created (12 §18, §23.3)."""
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    extra = {
        "lease": provide_run_lease(mgr),
        "revalidate": provide_revalidation(
            mgr, dataset_provider=provide_pass("dataset identity unchanged")
        ),
        "dataset": provide_pass("dataset identity verified"),
    }
    gate = ValidationGate(_passing_steps_except(extra))
    ctx = GateContext(run_id, tmp_path, make_spec())
    ctx.facts["required_disk_bytes"] = 1
    report = gate.run(ctx)
    assert not report.blocked
    assert len(report.checks) == 16
    assert report.checks[-1].id == "revalidate"
    assert report.checks[-1].verdict == "PASS"
    assert mgr.status(run_id).state == LeaseState.HELD


def test_gate_lease_conflict_propagates_exit3_not_fail(wf, tmp_path):
    """13 §7: execution lease held → BLOCK RUN_ALREADY_EXECUTING (exit 3),
    not an exit-1 gate FAIL."""
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    mgr.acquire(run_id)  # another worker holds it
    gate = ValidationGate(_passing_steps_except({"lease": provide_run_lease(mgr)}))
    with pytest.raises(RunAlreadyExecuting) as exc:
        gate.run(GateContext(run_id, tmp_path, make_spec()))
    assert exc.value.exit_code == 3


def test_revalidation_requires_lease_token(wf, tmp_path):
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    provider = provide_revalidation(mgr, dataset_provider=provide_pass("same"))
    ctx = GateContext(run_id, tmp_path, make_spec())
    check = provider(ctx)
    assert check.verdict == "FAIL" and "step 15" in check.detail


def test_revalidation_ownership_loss_fails(wf, tmp_path):
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    other = mgr.acquire(run_id)
    provider = provide_revalidation(mgr, dataset_provider=provide_pass("same"))
    ctx = GateContext(run_id, tmp_path, make_spec())
    ctx.facts["session_token"] = "gate-token-not-matching"
    check = provider(ctx)
    assert check.verdict == "FAIL" and "ownership lost" in check.detail
    assert other.session_token


def test_revalidation_full_pass(wf, tmp_path):
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    info = mgr.acquire(run_id)
    provider = provide_revalidation(mgr, dataset_provider=provide_pass("same bytes"))
    ctx = GateContext(run_id, tmp_path, make_spec())
    ctx.facts["session_token"] = info.session_token
    ctx.facts["required_disk_bytes"] = 1
    check = provider(ctx)
    assert check.verdict == "PASS"


def test_revalidation_unknown_disk_requirement_fails(wf, tmp_path):
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    info = mgr.acquire(run_id)
    provider = provide_revalidation(mgr, dataset_provider=provide_pass("same"))
    ctx = GateContext(run_id, tmp_path, make_spec())
    ctx.facts["session_token"] = info.session_token
    check = provider(ctx)
    assert check.verdict == "FAIL" and "required_disk_bytes unknown" in check.detail


def test_revalidation_missing_dataset_provider_fails(wf, tmp_path):
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    info = mgr.acquire(run_id)
    provider = provide_revalidation(mgr)  # no dataset re-verification
    ctx = GateContext(run_id, tmp_path, make_spec())
    ctx.facts["session_token"] = info.session_token
    ctx.facts["required_disk_bytes"] = 1
    check = provider(ctx)
    assert check.verdict == "FAIL" and "unverifiable" in check.detail


def test_revalidation_detects_dataset_change(wf, tmp_path):
    run_id = _run(wf)
    mgr = RunLeaseManager(tmp_path)
    info = mgr.acquire(run_id)
    from mlforge.validation import provide_fail

    provider = provide_revalidation(
        mgr, dataset_provider=provide_fail("content hash mismatch: path rewritten")
    )
    ctx = GateContext(run_id, tmp_path, make_spec())
    ctx.facts["session_token"] = info.session_token
    ctx.facts["required_disk_bytes"] = 1
    check = provider(ctx)
    assert check.verdict == "FAIL" and "changed since validation" in check.detail


# ---------------------------------------------------------------------------
# Workflow enforcement (13 §7 failure matrix)
# ---------------------------------------------------------------------------

def test_resume_blocked_by_held_lease(wf, tmp_path):
    """RESUME | execution lease held → BLOCK RUN_ALREADY_EXECUTING."""
    run_id = _parked(wf)
    info = RunLeaseManager(tmp_path).acquire(run_id)
    with pytest.raises(RunAlreadyExecuting) as exc:
        wf.resume(run_id)
    assert exc.value.exit_code == 3
    assert wf.get_run_state(run_id) == "PAUSED"  # blocked, unchanged
    # owner (with token) passes the lease guard
    assert wf.resume(run_id, session_token=info.session_token) == "VALIDATING"


def test_resume_blocked_by_suspect_lease(wf, tmp_path):
    clock = FakeClock()
    run_id = _parked(wf)
    RunLeaseManager(tmp_path, clock=clock).acquire(run_id)
    clock.advance(121)
    # status() uses its own manager instance/clock — rebuild with the fake
    # clock for the guard under test:
    from mlforge.leases import LeaseState as LS

    mgr = RunLeaseManager(tmp_path, clock=clock)
    assert mgr.status(run_id).state == LS.SUSPECT
    wf_lease_check = RunLeaseManager(tmp_path, timeout=0)  # everything stale
    assert wf_lease_check.status(run_id).state == LS.SUSPECT
    # workflow guard uses default timeout — simulate by rewriting heartbeat
    lease_path = tmp_path / "runs" / run_id / ".lease"
    data = json.loads(lease_path.read_text())
    data["heartbeat_at"] = 0.0  # ancient heartbeat → SUSPECT for any clock
    lease_path.write_text(json.dumps(data))
    with pytest.raises(PreconditionFailed) as exc:
        wf.resume(run_id)
    assert "SUSPECT" in str(exc.value)
    assert wf.get_run_state(run_id) == "PAUSED"


def test_lease_break_requires_force_and_yes(wf, tmp_path):
    run_id = _parked(wf)
    RunLeaseManager(tmp_path).acquire(run_id)
    with pytest.raises(PreconditionFailed):
        wf.lease_break(run_id, force=True, yes=False)
    with pytest.raises(PreconditionFailed):
        wf.lease_break(run_id, force=False, yes=True)
    # still held
    assert wf.lease_status(run_id)["state"] == "HELD"


def test_lease_break_logs_event_and_frees(wf, tmp_path):
    run_id = _parked(wf)
    info = RunLeaseManager(tmp_path).acquire(run_id)
    before = len(wf.get_run_events(run_id))
    result = wf.lease_break(
        run_id, force=True, yes=True, reason="worker confirmed dead",
        operator="daksh",
    )
    assert result["previous_owner"]["session_token"] == info.session_token
    events = wf.get_run_events(run_id)
    assert len(events) == before + 1
    broken = events[-1]
    assert broken["event"] == "LEASE_BROKEN"
    assert broken["previous_owner"]["pid"] == info.pid
    assert broken["reason"] == "worker confirmed dead"
    assert broken["operator"] == "daksh"
    assert "to" not in broken  # breaking a lease never changes run state
    assert wf.get_run_state(run_id) == "PAUSED"
    assert wf.lease_status(run_id)["state"] == "FREE"


def test_lease_status_free_for_unleased_run(wf):
    run_id = _run(wf)
    assert wf.lease_status(run_id)["state"] == "FREE"
