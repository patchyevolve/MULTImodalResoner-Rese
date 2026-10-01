"""Run / execution / artifact / command leases — build step 5 (pending).

Normative: 12_training_system.md §23.

Critical invariant: at most one live worker may write a run folder's
checkpoints at any time. A second resume on a leased run →
BLOCK RUN_ALREADY_EXECUTING (mlforge.errors.RunAlreadyExecuting).

Rules:
  * lease = atomic create (O_EXCL) + heartbeat renewal + owner token
  * stale ≠ free: expired lease is SUSPECT → reconciliation first (§12.3)
  * breaking a lease requires explicit user action and is always logged
"""
