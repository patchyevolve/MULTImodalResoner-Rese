"""Training runtime: transactional checkpoints, RNG hierarchy, heartbeat,
supervisor handoff — build step 6 (pending).

Normative: 12_training_system.md §11 (checkpoint transaction protocol),
§12.3–§12.4 (reconciliation, process ownership), §16.1 (five state
concepts: CHECKPOINT STATE vs RUN STATE vs LIVE STATE).

Checkpoint write order (never torch.save directly):
    tmp → hashes → fsync → atomic rename → manifest → fsync → commit marker
Recovery selection: newest VALID by identity predicate (§11.2) — never
ordinal arithmetic (N-1).
"""
