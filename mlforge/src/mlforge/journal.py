"""Append-only event journal — the authority for history and lifecycle state.

Normative: 12_training_system.md §16.

    Five State Concepts (Never Conflated)          authority
    CHECKPOINT STATE   checkpoint manifest          transactional, survives crash
    RUN STATE          event journal                state.json = projection only
    LIVE STATE         supervisor heartbeat          dies with the process
    METRICS            metrics log                   observations only
    EVENTS             events.jsonl (append-only)    history / audit trail

Rules enforced here:
  * append-only — no mutation, no deletion API
  * every write is fsynced (a crash may lose nothing already acknowledged)
  * RUN STATE is *derived* by replay: status.json is a projection and is
    never authoritative on its own (12 §12: "never trust state.json alone")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

FSYNC_EVERY_WRITE = True


@dataclass(frozen=True)
class Event:
    ts: float
    event: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "event": self.event, **self.data}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        d = dict(d)
        ts = float(d.pop("ts", 0.0))
        name = str(d.pop("event"))
        return cls(ts=ts, event=name, data=d)


class EventJournal:
    """One journal per object (run/project/dataset/model)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, event: str, **data: Any) -> Event:
        ev = Event(ts=time.time(), event=event, data=data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(ev.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            if FSYNC_EVERY_WRITE:
                os.fsync(f.fileno())
        return ev

    def read(self) -> list[Event]:
        if not self.path.exists():
            return []
        events: list[Event] = []
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(Event.from_dict(json.loads(line)))
                except json.JSONDecodeError as exc:
                    # Truncated tail after power loss: keep all complete
                    # records, never trust a partial line (12 §11.1 spirit).
                    raise CorruptJournal(
                        f"{self.path}:{lineno}: incomplete event record"
                    ) from exc
        return events

    def __iter__(self) -> Iterator[Event]:
        return iter(self.read())

    def project_state(self, initial: str) -> str:
        """Derive current RUN STATE by replaying transition-bearing events.

        The journal is the authority; this projection may be recomputed at
        any time (12 §16.2: "Run lifecycle state → event journal
        (state.json = projection)"). Any event carrying `to` advances the
        state — event names are for humans, `to` is the machine field."""
        state = initial
        for ev in self.read():
            if "to" in ev.data and ev.data["to"] is not None:
                state = str(ev.data["to"])
        return state

    def last_transition(self) -> Event | None:
        for ev in reversed(self.read()):
            if "to" in ev.data:
                return ev
        return None


class CorruptJournal(RuntimeError):
    """A journal record failed to parse — caller decides fail-closed policy."""
