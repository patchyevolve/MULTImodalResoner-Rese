"""Run lease (single-writer) + gate providers — build step 5 (leases half).

Normative: 12_training_system.md §23.1 (lease types), §23.2 (protocol),
§23.3 (preflight → acquisition → revalidation), §17 (single-writer
invariant).

API:
    RunLeaseManager(root).acquire/renew/release/status/force_release
        FREE | HELD | SUSPECT — stale ≠ free; owner-checked by token.
    provide_run_lease(leases)       gate step 15 provider (12 §18)
    provide_revalidation(...)       gate step 16 provider (volatile subset)
"""

from mlforge.leases.providers import provide_revalidation, provide_run_lease
from mlforge.leases.run_lease import (
    DEFAULT_HEARTBEAT_TIMEOUT,
    LeaseInfo,
    LeaseState,
    RunLeaseManager,
)

__all__ = [
    "DEFAULT_HEARTBEAT_TIMEOUT",
    "LeaseInfo",
    "LeaseState",
    "RunLeaseManager",
    "provide_revalidation",
    "provide_run_lease",
]
