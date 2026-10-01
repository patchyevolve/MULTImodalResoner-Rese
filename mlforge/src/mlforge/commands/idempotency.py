"""Command idempotency journal — safe retries for state-changing commands.

Normative: 13_product_specification.md §4.4 (Command Idempotency),
12_training_system.md §23.1 (command lease), §23.4 (Command Idempotency).

    ~/.mlforge/commands.jsonl     (workspace: <root>/commands.jsonl —
                                   append-only, fsynced, like events)

Rules enforced here:
  * duplicate `command_id` of a SUCCEEDED command → return the ORIGINAL
    result; never re-execute (13 §4.4: "never a second run").
  * duplicate while the command is still IN FLIGHT → PreconditionFailed
    (the command lease, 12 §23.1) — concurrent duplicates never race.
  * a FAILED command may be retried with the same `command_id`
    (failures are retryable; duplicates of successes are not).
  * a command that crashed before recording a terminal state leaves a
    stale `started` record — retryable after `stale_after` seconds so a
    dead client cannot brick the command id forever (the in-flight lease
    window is exactly the command duration).
  * duplicates are journaled as COMMAND_DEDUPED in the run's event
    journal when the result is a run (13 §4.4).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.journal import CorruptJournal

#: Command lease duration: an in-flight command older than this is a
#: crash orphan, not a live duplicate (retryable).
DEFAULT_STALE_AFTER = 600.0

KIND_NEW = "NEW"
KIND_DUPLICATE = "DUPLICATE"
KIND_IN_FLIGHT = "IN_FLIGHT"


@dataclass(frozen=True)
class DedupDecision:
    kind: str
    command_id: str
    command: str
    result: Any = None
    original_ts: float | None = None
    stale_recovery: bool = False

    @property
    def duplicate(self) -> bool:
        return self.kind == KIND_DUPLICATE


class CommandJournal:
    def __init__(
        self,
        root: str | Path,
        *,
        stale_after: float = DEFAULT_STALE_AFTER,
        clock: Any = time.time,
    ):
        self.root = Path(root)
        self.path = self.root / "commands.jsonl"
        self.stale_after = stale_after
        self.clock = clock

    # -- reads -----------------------------------------------------------

    def _records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    # Fail-closed: an unreadable journal must not silently
                    # allow double execution (12 §16 spirit).
                    raise CorruptJournal(
                        f"{self.path}:{lineno}: incomplete command record"
                    ) from exc
        return records

    def latest(self, command_id: str) -> dict[str, Any] | None:
        for rec in reversed(self._records()):
            if rec.get("command_id") == command_id:
                return rec
        return None

    def history(self, command_id: str) -> list[dict[str, Any]]:
        return [r for r in self._records() if r.get("command_id") == command_id]

    # -- append (fsynced, like the event journal) -------------------------

    def _append(self, record: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    # -- protocol ---------------------------------------------------------

    def begin(
        self, command_id: str, command: str, *, meta: dict[str, Any] | None = None
    ) -> DedupDecision:
        """Record `started` (or dedupe). The caller must NOT execute when
        the decision is not NEW."""
        latest = self.latest(command_id)
        now = self.clock()
        if latest is not None:
            status = latest.get("status")
            if status == "succeeded":
                return DedupDecision(
                    KIND_DUPLICATE, command_id, command,
                    result=latest.get("result"),
                    original_ts=float(latest.get("ts", 0.0)),
                )
            if status == "started" and now - float(latest.get("ts", 0.0)) < self.stale_after:
                return DedupDecision(KIND_IN_FLIGHT, command_id, command)
            # failed → retry allowed; stale started → crash orphan, retry
            # allowed (recorded so history shows the recovery).
            self._append({
                "ts": now,
                "command_id": command_id,
                "command": command,
                "status": "started",
                "retry_of": status,
                "meta": meta or {},
            })
            return DedupDecision(
                KIND_NEW, command_id, command,
                stale_recovery=(status == "started"),
            )

        self._append({
            "ts": now,
            "command_id": command_id,
            "command": command,
            "status": "started",
            "meta": meta or {},
        })
        return DedupDecision(KIND_NEW, command_id, command)

    def succeed(self, command_id: str, result: Any = None) -> None:
        # Result must be JSON-serializable — fail loudly rather than
        # write an unreadable journal record.
        try:
            json.dumps(result, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValidationBlock(
                f"command {command_id}: result is not JSON-serializable ({exc}); "
                "the command succeeded but its journal record would be unreadable",
                hint="store large/odd results as an artifact; journal only its identity",
            ) from exc
        latest = self.latest(command_id)
        if latest is None or latest.get("status") != "started":
            raise ValidationBlock(
                f"command {command_id}: cannot record success without a "
                f"started record (latest: {latest and latest.get('status')})"
            )
        self._append({
            "ts": self.clock(),
            "command_id": command_id,
            "command": latest.get("command"),
            "status": "succeeded",
            "result": result,
        })

    def fail(self, command_id: str, error: str) -> None:
        latest = self.latest(command_id)
        if latest is None or latest.get("status") != "started":
            return  # never clobber a terminal record
        self._append({
            "ts": self.clock(),
            "command_id": command_id,
            "command": latest.get("command"),
            "status": "failed",
            "error": error,
        })


def execute(
    journal: CommandJournal,
    command_id: str,
    command: str,
    fn: Callable[[], Any],
    *,
    meta: dict[str, Any] | None = None,
    on_dedup: Callable[[DedupDecision], None] | None = None,
) -> Any:
    """Run `fn` exactly once per successful command_id (13 §4.4).

    * NEW       → execute; record succeeded/failed
    * DUPLICATE → do not execute; return the original result
    * IN_FLIGHT → PreconditionFailed (command lease held)
    """
    decision = journal.begin(command_id, command, meta=meta)
    if decision.kind == KIND_DUPLICATE:
        if on_dedup is not None:
            on_dedup(decision)
        return decision.result
    if decision.kind == KIND_IN_FLIGHT:
        raise PreconditionFailed(
            f"command {command_id} is already in flight (command lease held)",
            hint="wait for it to finish or use a different --command-id",
        )
    try:
        result = fn()
    except BaseException as exc:
        journal.fail(command_id, f"{type(exc).__name__}: {exc}")
        raise
    journal.succeed(command_id, result)
    return result


__all__ = [
    "CommandJournal",
    "DedupDecision",
    "execute",
    "KIND_NEW",
    "KIND_DUPLICATE",
    "KIND_IN_FLIGHT",
    "DEFAULT_STALE_AFTER",
]
