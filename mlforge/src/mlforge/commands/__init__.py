"""Command idempotency journal — build step 5 (pending).

Normative: 12_training_system.md §23.4, 13_product_specification.md §4.4.

Planned API:
    CommandJournal.begin(command_id) -> InFlight | AlreadyDone(result)
    CommandJournal.succeed(command_id, result)
    CommandJournal.fail(command_id, error)     # failures remain retryable

Duplicates of SUCCESSES return the original result (exit 0,
COMMAND_DEDUPED event); failures may be retried with the same id.
"""
