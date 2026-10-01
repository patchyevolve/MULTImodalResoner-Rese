"""Lifecycle states and the FAILED recovery disposition.

Normative: 13_product_specification.md §5 (state machines), §5.3 lifecycle
terms; 12_training_system.md §12.

Two vocabulary rules the specs made non-negotiable after audit:

1. States are never interchangeable:
     PAUSED       user, resumable (ordinary gate path)
     INTERRUPTED  unexpected, recoverable only via reconciliation scan
     FAILED       error stopped the run; exits per recorded disposition
     STOPPED      user, no resume (continue only via retrain/finetune)
     COMPLETED    end condition, no resume

2. FAILED has exactly ONE meaning with a recorded disposition — never both
   "terminal" and "resumable":

     failure.recovery =
         FORK_ONLY  if NO valid checkpoint exists OR the cause is semantic
                    (fixing it would change the experiment → must fork)
         RESUME     otherwise (valid checkpoint + cause resolvable without
                    semantic change)

   Resume on FAILED(RESUME) re-runs the full validation gate — the
   disposition never bypasses validation (13 §5.3).
"""

from __future__ import annotations

import enum


class ProjectState(str, enum.Enum):
    """13 §5.1: CREATED → CONFIGURED → READY"""

    CREATED = "CREATED"
    CONFIGURED = "CONFIGURED"
    READY = "READY"


class DatasetState(str, enum.Enum):
    """13 §5.2: REGISTERED → VERIFIED → PREPARED; hash mismatch → REJECTED"""

    REGISTERED = "REGISTERED"
    VERIFIED = "VERIFIED"
    PREPARED = "PREPARED"
    REJECTED = "REJECTED"


class ModelState(str, enum.Enum):
    """13 §5.4: CREATED → VALIDATED → AVAILABLE (+ consumption states).

    Models are immutable — there is no RETRAIN state on a model; retraining
    reads a model as source and produces a different model."""

    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    AVAILABLE = "AVAILABLE"
    EVALUATED = "EVALUATED"
    EXPORTED = "EXPORTED"
    DEPLOYED = "DEPLOYED"
    USED_AS_FINE_TUNE_BASE = "USED_AS_FINE_TUNE_BASE"


class RunState(str, enum.Enum):
    """13 §5.3 core run machine."""

    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    PREPARING = "PREPARING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    CHECKPOINTING = "CHECKPOINTING"
    INTERRUPTED = "INTERRUPTED"
    RECONCILING = "RECONCILING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class FailureRecovery(str, enum.Enum):
    """The recorded disposition of a FAILED run (13 §5.3)."""

    RESUME = "RESUME"
    FORK_ONLY = "FORK_ONLY"


def compute_recovery(*, has_valid_checkpoint: bool, cause_is_semantic: bool) -> FailureRecovery:
    """Decide failure.recovery once, at state entry (13 §5.3 rule):

        FORK_ONLY  if no valid checkpoint OR cause is semantic
        RESUME     otherwise
    """
    if not has_valid_checkpoint or cause_is_semantic:
        return FailureRecovery.FORK_ONLY
    return FailureRecovery.RESUME


#: States where a worker process may be alive (a crash can strike these).
#: A crash from any of these → INTERRUPTED, never PAUSED/RUNNING (13 §7:
#: "RUNNING (or any live state) | crash / power loss → INTERRUPTED").
#: Non-live: CREATED (no process yet), PAUSED/STOPPED/FAILED/COMPLETED
#: (no process by definition), INTERRUPTED (already crashed).
LIVE_STATES: frozenset[RunState] = frozenset(
    {
        RunState.VALIDATING,      # gate runs in a process during resume
        RunState.PREPARING,
        RunState.READY,           # preflight in flight
        RunState.RUNNING,
        RunState.PAUSING,
        RunState.STOPPING,
        RunState.CHECKPOINTING,
        RunState.RECONCILING,     # reconciliation scan runs in a process
    }
)

#: Automatic continuation is NO for every non-RUNNING state (13 §5.3
#: terminology note). Nothing in the system transitions a run on its own.
def has_automatic_continuation(state: RunState) -> bool:
    return state is RunState.RUNNING
