"""Run lease — single-writer enforcement for a run folder (12 §23.1–23.2).

    Medium: runs/<id>/.lease  (atomic O_EXCL-style create + PID + host +
    session_token; content never partial: write tmp → fsync → hard-link)

Protocol (12 §23.2):
    acquire  .lease.tmp {run_id, pid, host, session_token, acquired_at,
             heartbeat_at} → fsync → atomic link → .lease (fails if exists)
    renew    update heartbeat_at ONLY if session_token matches (owner check)
    release  delete .lease ONLY if session_token matches (owner check)
    steal    never implicit — `mlforge lease break RUN --force` (+ --yes),
             always logged as LEASE_BROKEN by the caller

Rules enforced here:
  * **Stale ≠ free.** A lease whose heartbeat exceeds the timeout is
    SUSPECT, not reclaimable — acquisition fails too; only an explicit,
    logged break frees it. Reconciliation decides whether the process is
    actually dead (12 §23.2).
  * At most one live worker writes a run folder's checkpoints (hard
    invariant): a second acquire/resume → RunAlreadyExecuting (exit 3).
  * Every mutation is owner-checked by session_token — a pid alone is
    never trusted (pid reuse).
"""

from __future__ import annotations

import errno
import json
import os
import socket
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound, PreconditionFailed, RunAlreadyExecuting
from mlforge.ids import new_session_token

DEFAULT_HEARTBEAT_TIMEOUT = 120.0  # 12 §23.2: default 120s


class LeaseState(str, Enum):
    FREE = "FREE"
    HELD = "HELD"
    SUSPECT = "SUSPECT"


@dataclass(frozen=True)
class LeaseInfo:
    state: LeaseState
    run_id: str
    pid: int | None = None
    host: str | None = None
    session_token: str | None = None
    acquired_at: float | None = None
    heartbeat_at: float | None = None
    age_seconds: float | None = None

    @property
    def holder(self) -> str:
        if self.pid is None:
            return "nobody"
        return f"pid {self.pid} on {self.host}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "run_id": self.run_id,
            "pid": self.pid,
            "host": self.host,
            "session_token": self.session_token,
            "acquired_at": self.acquired_at,
            "heartbeat_at": self.heartbeat_at,
            "age_seconds": self.age_seconds,
        }


class RunLeaseManager:
    def __init__(
        self,
        root: str | Path,
        *,
        timeout: float = DEFAULT_HEARTBEAT_TIMEOUT,
        clock: Any = time.time,
    ):
        self.root = Path(root)
        self.timeout = timeout
        self.clock = clock

    # -- addressing -----------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        d = self.root / "runs" / run_id
        if not d.is_dir():
            raise NotFound(f"run {run_id!r} not found under {self.root}")
        return d

    def path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / ".lease"

    # -- reads -----------------------------------------------------------

    def status(self, run_id: str) -> LeaseInfo:
        """FREE | HELD | SUSPECT (heartbeat older than timeout — stale is
        never free, 12 §23.2)."""
        p = self.path(run_id)
        if not p.is_file():
            return LeaseInfo(LeaseState.FREE, run_id)
        data = self._read(p)
        now = self.clock()
        heartbeat = float(data.get("heartbeat_at", 0.0))
        age = now - heartbeat
        state = LeaseState.HELD if age <= self.timeout else LeaseState.SUSPECT
        return LeaseInfo(
            state=state,
            run_id=run_id,
            pid=int(data.get("pid", -1)),
            host=str(data.get("host", "")),
            session_token=str(data.get("session_token", "")),
            acquired_at=float(data.get("acquired_at", 0.0)),
            heartbeat_at=heartbeat,
            age_seconds=age,
        )

    # -- protocol (12 §23.2) ---------------------------------------------

    def acquire(self, run_id: str) -> LeaseInfo:
        """Claim single-writer ownership. Fails if any lease exists —
        even a SUSPECT one (stale ≠ free: break it explicitly first)."""
        d = self.run_dir(run_id)
        dest = d / ".lease"
        existing = self.status(run_id)
        if existing.state == LeaseState.HELD:
            raise RunAlreadyExecuting(run_id, holder=existing.holder)
        if existing.state == LeaseState.SUSPECT:
            raise PreconditionFailed(
                f"run {run_id}: lease is SUSPECT (held by {existing.holder}, "
                f"heartbeat {existing.age_seconds:.0f}s old > {self.timeout:.0f}s)",
                hint="stale ≠ free: confirm the worker is dead, then "
                     "`mlforge lease break RUN --force --yes` (12 §23.2)",
            )

        now = self.clock()
        token = new_session_token()
        payload = {
            "run_id": run_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "session_token": token,
            "acquired_at": now,
            "heartbeat_at": now,
        }
        tmp = d / ".lease.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            # Atomic create: hard-link fails if .lease already exists, and
            # the destination never holds partial content.
            os.link(tmp, dest)
        except FileExistsError:
            holder = self.status(run_id).holder
            raise RunAlreadyExecuting(run_id, holder=holder) from None
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                holder = self.status(run_id).holder
                raise RunAlreadyExecuting(run_id, holder=holder) from None
            raise
        finally:
            tmp.unlink(missing_ok=True)
        return self.status(run_id)

    def renew(self, run_id: str, session_token: str) -> LeaseInfo:
        """Heartbeat (default every 30s). Owner check: only the matching
        session_token may update heartbeat_at."""
        p = self.path(run_id)
        if not p.is_file():
            raise PreconditionFailed(
                f"run {run_id}: cannot renew — no lease exists",
                hint="the lease was broken or released; re-acquire before continuing",
            )
        data = self._read(p)
        self._require_owner(run_id, data, session_token, action="renew")
        data["heartbeat_at"] = self.clock()
        self._write_atomic(p, data)
        return self.status(run_id)

    def release(self, run_id: str, session_token: str) -> None:
        """Graceful release. Owner check: releasing someone else's lease
        is refused (never silently steal)."""
        p = self.path(run_id)
        if not p.is_file():
            raise PreconditionFailed(f"run {run_id}: no lease to release")
        data = self._read(p)
        self._require_owner(run_id, data, session_token, action="release")
        p.unlink()

    def force_release(self, run_id: str) -> dict[str, Any]:
        """BREAK — caller (workflow/CLI) must have enforced `--force` +
        `--yes` and must log LEASE_BROKEN (12 §23.2: always logged)."""
        p = self.path(run_id)
        if not p.is_file():
            raise PreconditionFailed(
                f"run {run_id}: no lease to break",
                hint="nothing is holding this run",
            )
        previous = self._read(p)
        p.unlink()
        return previous

    # -- plumbing ---------------------------------------------------------

    @staticmethod
    def _read(p: Path) -> dict[str, Any]:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PreconditionFailed(
                f"lease file {p} is corrupt: {exc}",
                hint="inspect manually, then `mlforge lease break RUN --force --yes`",
            ) from exc

    @staticmethod
    def _write_atomic(p: Path, data: dict[str, Any]) -> None:
        tmp = p.with_suffix(".lease.renew.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    @staticmethod
    def _require_owner(
        run_id: str, data: dict[str, Any], session_token: str, *, action: str
    ) -> None:
        if data.get("session_token") != session_token:
            raise PreconditionFailed(
                f"run {run_id}: cannot {action} — session_token does not match "
                f"the lease holder (pid {data.get('pid')} on {data.get('host')})",
                hint="only the holder may renew/release; break requires "
                     "`mlforge lease break RUN --force --yes`",
            )


__all__ = [
    "DEFAULT_HEARTBEAT_TIMEOUT",
    "LeaseInfo",
    "LeaseState",
    "RunLeaseManager",
]
