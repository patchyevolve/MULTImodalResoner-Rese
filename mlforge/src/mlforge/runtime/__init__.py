"""Training runtime: transactional checkpoints, heartbeat, reconciliation.

Normative: 12_training_system.md §11 (checkpoint transaction protocol),
§12.3 (reconciliation), §16.1 (CHECKPOINT STATE vs RUN STATE vs LIVE
STATE), 13 §10 (`state/heartbeat.json`).

Checkpoint write order (never torch.save directly):
    tmp → hashes → fsync → atomic rename → manifest → fsync → commit marker
Recovery selection: newest VALID by identity predicate (§11.2) — never
ordinal arithmetic (N-1).

Part of build step 6: checkpoints + heartbeat + reconciliation scan are
in; the worker process and train/resume/pause/stop CLI wiring arrive
next (12 §12.4 process ownership).
"""

from mlforge.runtime.checkpoints import (
    REQUIRED_COMPONENTS,
    Candidate,
    CheckpointStore,
    Selection,
)
from mlforge.runtime.heartbeat import DEFAULT_HEARTBEAT_INTERVAL, HeartbeatWriter
from mlforge.runtime.reconcile import ReconcileReport, scan_checkpoints

__all__ = [
    "CheckpointStore",
    "Candidate",
    "Selection",
    "REQUIRED_COMPONENTS",
    "HeartbeatWriter",
    "DEFAULT_HEARTBEAT_INTERVAL",
    "ReconcileReport",
    "scan_checkpoints",
]
