"""Command idempotency journal — build step 5 (idempotency half).

Normative: 13_product_specification.md §4.4, 12_training_system.md §23.4.

API:
    CommandJournal(root).begin/succeed/fail/latest
        append-only commands.jsonl; SUCCEEDED duplicate → original result;
        IN FLIGHT → exit 3; FAILED/stale-started → retry allowed.
    execute(journal, command_id, command, fn)   exactly-once wrapper
"""

from mlforge.commands.idempotency import (
    KIND_DUPLICATE,
    KIND_IN_FLIGHT,
    KIND_NEW,
    CommandJournal,
    DedupDecision,
    execute,
)

__all__ = [
    "CommandJournal",
    "DedupDecision",
    "execute",
    "KIND_NEW",
    "KIND_DUPLICATE",
    "KIND_IN_FLIGHT",
]
