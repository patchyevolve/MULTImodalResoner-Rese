"""Training runtime: transactional checkpoints, heartbeat, reconciliation.

Normative: 12_training_system.md §11 (checkpoint transaction protocol),
§12.3 (reconciliation), §16.1 (CHECKPOINT STATE vs RUN STATE vs LIVE
STATE), 13 §10 (`state/heartbeat.json`).

Checkpoint write order (never torch.save directly):
    tmp → hashes → fsync → atomic rename → manifest → fsync → commit marker
Recovery selection: newest VALID by identity predicate (§11.2) — never
ordinal arithmetic (N-1).

Part of build step 6: transactional checkpoints, heartbeat, reconciliation,
the control channel, the pluggable trainer protocol, and the worker process
(`mlforge.runtime.worker`, spawned by the supervisor daemon — 12 §12.4).
"""

from mlforge.runtime.checkpoints import (
    REQUIRED_COMPONENTS,
    Candidate,
    CheckpointStore,
    Selection,
)
from mlforge.runtime.control import (
    clear_control,
    control_path,
    read_control,
    write_control,
)
from mlforge.runtime.heartbeat import DEFAULT_HEARTBEAT_INTERVAL, HeartbeatWriter
from mlforge.runtime.reconcile import ReconcileReport, scan_checkpoints
from mlforge.runtime.trainer import (
    HARNESS_ENV,
    HARNESS_TRAINER,
    ScaffoldTrainer,
    StepResult,
    TrainState,
    Trainer,
    harness_allowed,
    require_trainable,
    resolve_trainer,
)

__all__ = [
    "CheckpointStore",
    "Candidate",
    "Selection",
    "REQUIRED_COMPONENTS",
    "HeartbeatWriter",
    "DEFAULT_HEARTBEAT_INTERVAL",
    "ReconcileReport",
    "scan_checkpoints",
    "control_path",
    "read_control",
    "write_control",
    "clear_control",
    "Trainer",
    "ScaffoldTrainer",
    "TrainState",
    "StepResult",
    "HARNESS_TRAINER",
    "HARNESS_ENV",
    "harness_allowed",
    "require_trainable",
    "resolve_trainer",
]
