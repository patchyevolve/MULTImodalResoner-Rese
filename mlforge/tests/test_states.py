from mlforge.states import (
    DatasetState,
    FailureRecovery,
    ModelState,
    ProjectState,
    RunState,
    compute_recovery,
    has_automatic_continuation,
)


def test_run_state_values_match_spec():
    """13 §5.3 — state names are contract (CLI/scripts depend on them)."""
    assert [s.value for s in RunState] == [
        "CREATED",
        "VALIDATING",
        "PREPARING",
        "READY",
        "RUNNING",
        "PAUSING",
        "PAUSED",
        "STOPPING",
        "STOPPED",
        "CHECKPOINTING",
        "INTERRUPTED",
        "RECONCILING",
        "FAILED",
        "COMPLETED",
    ]


def test_other_state_enums_match_spec():
    assert [s.value for s in ProjectState] == ["CREATED", "CONFIGURED", "READY"]
    assert [s.value for s in DatasetState] == [
        "REGISTERED",
        "VERIFIED",
        "PREPARED",
        "REJECTED",
    ]
    assert ModelState.AVAILABLE.value == "AVAILABLE"


def test_recovery_rule_no_checkpoint_is_fork_only():
    assert (
        compute_recovery(has_valid_checkpoint=False, cause_is_semantic=False)
        is FailureRecovery.FORK_ONLY
    )


def test_recovery_rule_semantic_cause_is_fork_only():
    assert (
        compute_recovery(has_valid_checkpoint=True, cause_is_semantic=True)
        is FailureRecovery.FORK_ONLY
    )


def test_recovery_rule_resumable_failure():
    assert (
        compute_recovery(has_valid_checkpoint=True, cause_is_semantic=False)
        is FailureRecovery.RESUME
    )


def test_automatic_continuation_only_for_running():
    """13 §5.3: Automatic continuation is NO for every non-RUNNING state."""
    for state in RunState:
        assert has_automatic_continuation(state) is (state is RunState.RUNNING)
