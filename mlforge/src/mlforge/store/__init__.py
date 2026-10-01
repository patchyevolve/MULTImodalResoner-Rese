"""Artifact registry + content-addressed store — build step 2 (pending).

Normative: 12_training_system.md §6.1 (CAS, registry transactions,
reachability-based GC), §5 (portable run folder).

Planned API:
    ContentStore.put(bytes) -> "sha256:..."      # staging → fsync → atomic rename
    ContentStore.get(hash) -> bytes
    ArtifactRegistry.register(kind, hash, meta)  # transactional commit
    ArtifactRegistry.gc(roots) -> None           # reachability only, never
                                                  # by age/size alone
"""
