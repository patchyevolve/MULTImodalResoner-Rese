"""Heartbeat writer — live-state emission for the training worker.

Normative: 12_training_system.md §16 (LIVE STATE = supervisor heartbeat —
dies with the process; heartbeats gate liveness reporting only), §23.2
(renew interval 30s; the run lease is renewed WITH the heartbeat so a
live worker never becomes SUSPECT), 13 §10 (`state/heartbeat.json`).

The worker owns its heartbeat; the supervisor reads it. A stale or
missing heartbeat is what makes the run INTERRUPTED (never PAUSED).
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from mlforge.leases import RunLeaseManager

DEFAULT_HEARTBEAT_INTERVAL = 30.0  # 12 §23.2: renew (heartbeat, 30s)


class HeartbeatWriter:
    """Writes `state/heartbeat.json` atomically; optionally renews the
    run lease on every beat (owner-checked — losing the lease raises)."""

    def __init__(
        self,
        root: str | Path,
        run_id: str,
        session_token: str,
        *,
        lease: RunLeaseManager | None = None,
        clock: Any = time.time,
    ):
        self.root = Path(root)
        self.run_id = run_id
        self.session_token = session_token
        self.lease = lease
        self.clock = clock
        self.path = self.root / "runs" / run_id / "state" / "heartbeat.json"

    def beat(self, **extra: Any) -> dict[str, Any]:
        """One heartbeat: renew lease (if owned) THEN write the file.
        A lost lease raises — the worker must stop writing checkpoints
        (single-writer invariant, 12 §23.2).

        `extra` carries the fields the worker knows and status displays
        (13 §9.4 example: `global_step`, `state`)."""
        if self.lease is not None:
            self.lease.renew(self.run_id, self.session_token)  # owner check
        payload = {
            "run_id": self.run_id,
            "ts": self.clock(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "session_token": self.session_token,
            **extra,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)  # readers never see a partial heartbeat
        return payload

    def due(self, last_beat: float | None, interval: float = DEFAULT_HEARTBEAT_INTERVAL) -> bool:
        if last_beat is None:
            return True
        return self.clock() - last_beat >= interval

    def clear(self) -> None:
        """Remove the heartbeat file (graceful worker exit — the run is no
        longer live; the supervisor must not later read a stale beat)."""
        self.path.unlink(missing_ok=True)


__all__ = ["DEFAULT_HEARTBEAT_INTERVAL", "HeartbeatWriter"]
