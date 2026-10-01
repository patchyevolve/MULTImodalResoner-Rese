"""Artifact registry + content-addressed store — build step 2 (DONE).

Normative: 12_training_system.md §6.1 (CAS, registry transactions,
reachability-based GC), §5 (portable run folder references).

API:
    ContentStore.put_bytes/put_file  -> "sha256:..."   staging → fsync → atomic rename
    ContentStore.get_bytes/verify                      integrity checked, never assumed
    ContentStore.recover_staging()                     crash orphans swept
    ArtifactRegistry.register(kind, ...)               transactional commit
    ArtifactRegistry.recover()                         startup recovery scan
    ArtifactRegistry.link(run_id, hash)                run-folder references (GC roots)
    ArtifactRegistry.gc(grace, dry_run)                reachability only — never age/size;
                                                       blocked by active run leases
"""

from mlforge.store.content_store import ContentStore
from mlforge.store.registry import ArtifactRegistry, GCReport

__all__ = ["ContentStore", "ArtifactRegistry", "GCReport"]
