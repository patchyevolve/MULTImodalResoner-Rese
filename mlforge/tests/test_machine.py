import pytest

from mlforge.errors import (
    EXIT_PRECONDITION,
    InvalidTransition,
    NoValidContinuation,
    PreconditionFailed,
)
from mlforge.machine import RUN_MACHINE, StateMachine
from mlforge.states import FailureRecovery, RunState

V = lambda s: s.value


def fire(state: RunState | str, action: str, **ctx) -> str:
    state = state.value if isinstance(state, RunState) else state
    return RUN_MACHINE.fire(state, action, ctx)


# -- happy path ---------------------------------------------------------

def test_full_happy_path():
    s = RunState.CREATED
    for action, expected in [
        ("begin_validation", RunState.VALIDATING),
        ("validation_pass", RunState.READY),
        ("preflight_pass", RunState.RUNNING),
        ("checkpoint", RunState.CHECKPOINTING),
        ("checkpoint_done", RunState.RUNNING),
        ("complete", RunState.COMPLETED),
    ]:
        s = fire(s, action)
        assert s == V(expected)


def test_pause_resume_cycle():
    s = RunState.RUNNING
    s = fire(s, "pause", explicit=True)
    assert s == V(RunState.PAUSING)
    s = fire(s, "pause_committed")
    assert s == V(RunState.PAUSED)
    s = fire(s, "resume")
    assert s == V(RunState.VALIDATING)


def test_stop_is_terminal_sink():
    s = fire(RunState.RUNNING, "stop", explicit=True)
    assert s == V(RunState.STOPPING)
    s = fire(RunState.STOPPING, "stop_committed")
    assert s == V(RunState.STOPPED)
    with pytest.raises(InvalidTransition) as e:
        fire(RunState.STOPPED, "resume")
    assert "resume" not in e.value.legal_actions  # 13 §5.3: no resume from STOPPED


def test_completed_is_sink():
    with pytest.raises(InvalidTransition):
        fire(RunState.COMPLETED, "resume")


# -- failure disposition guards (13 §5.3) --------------------------------

def test_resume_blocked_on_failed_fork_only():
    with pytest.raises(NoValidContinuation) as e:
        fire(
            RunState.FAILED,
            "resume",
            failure_recovery=FailureRecovery.FORK_ONLY.value,
            run_id="run_X",
        )
    assert e.value.exit_code == EXIT_PRECONDITION
    assert "fork" in (e.value.hint or "")


def test_resume_allowed_on_failed_resume():
    s = fire(
        RunState.FAILED,
        "resume",
        failure_recovery=FailureRecovery.RESUME.value,
        run_id="run_X",
    )
    assert s == V(RunState.VALIDATING)


# -- crash & reconciliation (12 §12.3) -----------------------------------

def test_crash_from_running_goes_interrupted_not_paused():
    s = fire(RunState.RUNNING, "crash", explicit=True)
    assert s == V(RunState.INTERRUPTED)


@pytest.mark.parametrize(
    "state",
    [RunState.CREATED, RunState.PAUSED, RunState.FAILED, RunState.STOPPED],
)
def test_crash_illegal_from_non_live_states(state):
    with pytest.raises(InvalidTransition):
        fire(state, "crash", explicit=True)


def test_reconciliation_with_checkpoint_waits_for_resume():
    s = fire(RunState.INTERRUPTED, "begin_reconciliation")
    assert s == V(RunState.RECONCILING)
    # reconciliation never starts training: no path to RUNNING here
    with pytest.raises(InvalidTransition):
        fire(RunState.RECONCILING, "preflight_pass")
    s = fire(RunState.RECONCILING, "resume", has_valid_checkpoint=True)
    assert s == V(RunState.VALIDATING)


def test_reconciliation_without_checkpoint_is_fork_only_failure():
    fire(RunState.INTERRUPTED, "begin_reconciliation")
    s = fire(RunState.RECONCILING, "reconcile_no_checkpoint", has_valid_checkpoint=False)
    assert s == V(RunState.FAILED)


def test_resume_from_reconciling_requires_checkpoint():
    fire(RunState.INTERRUPTED, "begin_reconciliation")
    with pytest.raises(NoValidContinuation):
        fire(RunState.RECONCILING, "resume", has_valid_checkpoint=False)


def test_reconcile_no_checkpoint_illegal_when_checkpoint_exists():
    fire(RunState.INTERRUPTED, "begin_reconciliation")
    with pytest.raises(PreconditionFailed):
        fire(RunState.RECONCILING, "reconcile_no_checkpoint", has_valid_checkpoint=True)


# -- guards -------------------------------------------------------------

def test_pause_requires_explicit_initiation():
    """System never pauses/stops on its own (13 §1 decision boundary)."""
    with pytest.raises(PreconditionFailed):
        fire(RunState.RUNNING, "pause")  # no explicit=True


def test_illegal_transition_lists_legal_actions():
    with pytest.raises(InvalidTransition) as e:
        fire(RunState.CREATED, "complete")
    assert e.value.legal_actions == ["begin_validation"]
    assert e.value.exit_code == EXIT_PRECONDITION


def test_machine_is_immutable_definition():
    assert isinstance(RUN_MACHINE, StateMachine)
    assert RUN_MACHINE.initial == V(RunState.CREATED)
