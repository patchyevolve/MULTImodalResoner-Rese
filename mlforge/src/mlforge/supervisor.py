"""Supervisor — live-state monitoring and crash detection.

Normative: 12_training_system.md §12.3 (reconciliation after unexpected
events), §16 (LIVE STATE = supervisor heartbeat — dies with the process;
heartbeats gate *liveness reporting* only, §25 rule 5), 13 §5.3
(crash → INTERRUPTED), §9.4 ("a stale status.json saying RUNNING after a
crash is INTERRUPTED the moment the heartbeat expires").

Scope:
  * crash DETECTION only — heartbeat expiry → WorkflowAPI.crash() →
    INTERRUPTED (never reported as PAUSED or RUNNING).
  * the supervisor NEVER starts, resumes, or reconciles anything:
    reconciliation is an explicit scan (12 §12.3) and `resume` is a user
    decision (13 §1 "Automatic continuation is NO"). It DOES spawn
    workers that the CLI queued (12 §12.4: the CLI never spawns the
    training process itself) — spawning a queued request is not
    "continuation", it is delivering the user's own command.
  * it owns no state — every run change goes through the same Workflow
    API path as the CLI (single write path: machine → journal →
    projection).

Heartbeat file: `runs/<id>/state/heartbeat.json` (13 §10), written by the
training worker (build step 6). A run that just entered RUNNING without a
heartbeat yet gets one full timeout of grace — only an *expired* heartbeat
(or missing file with no journal activity) counts as dead.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed
from mlforge.states import RunState
from mlforge.workflow import WorkflowAPI

DEFAULT_HEARTBEAT_TIMEOUT = 120.0  # 12 §23.2 default; live-state budget
DEFAULT_SCAN_INTERVAL = 30.0


class Supervisor:
    def __init__(
        self,
        root: str | Path,
        *,
        workflow: WorkflowAPI | None = None,
        timeout: float = DEFAULT_HEARTBEAT_TIMEOUT,
        interval: float = DEFAULT_SCAN_INTERVAL,
        clock: Any = time.time,
    ):
        self.root = Path(root)
        self.wf = workflow or WorkflowAPI(root)
        self.timeout = timeout
        self.interval = interval
        self.clock = clock

    # -- heartbeat --------------------------------------------------------

    def heartbeat_path(self, run_id: str) -> Path:
        return self.root / "runs" / run_id / "state" / "heartbeat.json"

    def read_heartbeat(self, run_id: str) -> dict[str, Any] | None:
        p = self.heartbeat_path(run_id)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None  # unreadable heartbeat = no live signal

    def last_activity_ts(self, run_id: str) -> float | None:
        events = self.wf.get_run_events(run_id)
        ts = [float(e.get("ts", 0.0)) for e in events]
        return max(ts) if ts else None

    # -- one deterministic scan (testable) --------------------------------

    def scan_once(self) -> dict[str, Any]:
        """Check every RUNNING run once. Returns a structured report:
        {"checked": [...], "crashed": [...], "errors": [...]}."""
        now = self.clock()
        checked: list[str] = []
        crashed: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        try:
            runs = self.wf.list_runs()
        except Exception as exc:  # noqa: BLE001 - one broken workspace must be reported, not fatal
            return {
                "checked": [], "crashed": [],
                "errors": [{"workspace": str(self.root),
                            "error": f"{type(exc).__name__}: {exc}"}],
            }

        for proj in runs:
            run_id = proj["id"]
            if proj.get("state") != RunState.RUNNING.value:
                continue
            checked.append(run_id)
            try:
                hb = self.read_heartbeat(run_id)
                if hb is not None:
                    age = now - float(hb.get("ts", 0.0))
                    if age > self.timeout:
                        self._crash(
                            crashed, run_id,
                            f"heartbeat expired ({age:.0f}s > {self.timeout:.0f}s) "
                            f"pid {hb.get('pid')} host {hb.get('host')}",
                        )
                else:
                    # No heartbeat file: crash only after the grace window
                    # (a run may have JUST entered RUNNING).
                    last = self.last_activity_ts(run_id)
                    silence = (now - last) if last is not None else float("inf")
                    if silence > self.timeout:
                        self._crash(
                            crashed, run_id,
                            f"heartbeat missing and no journal activity "
                            f"for {silence:.0f}s (live state is dead)",
                        )
            except Exception as exc:  # noqa: BLE001 - a broken run must not kill the scan
                errors.append(
                    {"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"}
                )
        return {"checked": checked, "crashed": crashed, "errors": errors}

    def _crash(self, crashed: list[dict[str, Any]], run_id: str, reason: str) -> None:
        # Explicit crash event through the single write path; state becomes
        # INTERRUPTED — never PAUSED/RUNNING (13 §7).
        self.wf.crash(run_id, reason)
        crashed.append({"run_id": run_id, "reason": reason})

    # -- daemon loop (library; started by train/runtime, build step 6) ----

    def run(self, *, max_scans: int | None = None) -> dict[str, Any]:
        """Scan → sleep → scan. `max_scans` bounds the loop (tests, tools)."""
        scans = 0
        aggregate: dict[str, Any] = {"checked": [], "crashed": [], "errors": []}
        while max_scans is None or scans < max_scans:
            report = self.scan_once()
            for key, bucket in aggregate.items():
                bucket.extend(report[key])
            scans += 1
            if max_scans is not None and scans >= max_scans:
                break
            time.sleep(self.interval)
        return aggregate

    # ------------------------------------------------------------------
    # daemon identity + spawn queue (12 §12.4: CLI → supervisor → worker)
    # ------------------------------------------------------------------

    def touch(self) -> Path:
        """Supervisor liveness file — `state/supervisor.json` (per-user,
        per-workspace). A dead supervisor never corrupts runs: workers
        keep running; the next command rebuilds what it needs from the
        run folders (13 §9.2)."""
        p = self.root / "state" / "supervisor.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "host": socket.gethostname(),
                   "last_seen": self.clock()}
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        return p

    def daemon_loop(self, *, poll_interval: float = 0.5) -> dict[str, Any]:
        """The per-user daemon: touch liveness → drain spawn queue → scan
        for expired heartbeats (on `interval`) → poll. NEVER restarts a
        dead worker — restart = resume = the user's decision (13 §1)."""
        aggregate: dict[str, Any] = {"checked": [], "crashed": [], "errors": [],
                                     "spawned": []}
        last_scan = 0.0
        while True:
            self.touch()
            aggregate["spawned"].extend(drain_pending(self.root))
            now = self.clock()
            if now - last_scan >= self.interval:
                report = self.scan_once()
                for key in ("checked", "crashed", "errors"):
                    aggregate[key].extend(report[key])
                last_scan = now
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# spawn queue — CLI writes intent, supervisor spawns (never the shell)
# ---------------------------------------------------------------------------

def _state_dir(root: str | Path) -> Path:
    return Path(root) / "state"


def _pending_dir(root: str | Path) -> Path:
    return _state_dir(root) / "pending"


def enqueue_spawn(root: str | Path, run_id: str, session_token: str) -> Path:
    """CLI side: record "please start this worker" (atomic write). The
    supervisor daemon picks it up — the worker is never a child of the
    interactive shell (12 §12.4)."""
    d = _pending_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{run_id}.json"
    payload = {"run_id": run_id, "session_token": session_token, "ts": time.time()}
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return p


def _default_spawner(root: Path, run_id: str, session_token: str) -> int:
    """Spawn the worker detached from every interactive session
    (start_new_session ⇒ survives terminal/SSH death). Logs go to the
    run folder's `state/worker.log`."""
    log = root / "runs" / run_id / "state" / "worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlforge.runtime.worker",
        "--root", str(root), "--run", run_id, "--token", session_token,
    ]
    with open(log, "a", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # 12 §12.4: not a child of any shell
            env=dict(os.environ),
        )
    return proc.pid


def drain_pending(root: str | Path, *, spawner: Any = None) -> list[dict[str, Any]]:
    """Supervisor side: spawn every queued worker. A failed spawn is
    RETAINED (retried next tick) and logged — never silently dropped.
    Each successful spawn is journaled `WORKER_SPAWNED` (no `to` — a
    spawn never changes run state by itself)."""
    root = Path(root)
    spawner = spawner or (lambda r, run_id, tok: _default_spawner(r, run_id, tok))
    spawned: list[dict[str, Any]] = []
    pending = _pending_dir(root)
    if not pending.is_dir():
        return spawned
    for p in sorted(pending.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            run_id = str(data["run_id"])
            token = str(data["session_token"])
        except (json.JSONDecodeError, KeyError) as exc:
            # unreadable request: quarantine, keep going (one bad file must
            # not wedge every other spawn)
            p.replace(p.with_name(p.name + ".corrupt"))
            _log(root, f"spawn request {p.name} quarantined: {exc}")
            continue
        try:
            pid = spawner(root, run_id, token)
        except Exception as exc:  # noqa: BLE001 - spawn failure is logged and retried, never fatal to the scan
            _log(root, f"spawn failed for {run_id} (will retry): {exc}")
            continue
        p.unlink(missing_ok=True)
        spawned.append({"run_id": run_id, "pid": pid})
        _log(root, f"spawned worker pid {pid} for {run_id}")
        events = root / "runs" / run_id / "events.jsonl"
        if events.parent.is_dir():
            with open(events, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event": "WORKER_SPAWNED",
                    "ts": time.time(),
                    "action": "spawn",
                    "pid": pid,
                }, sort_keys=True) + "\n")
                f.flush()
    return spawned


def _log(root: Path, message: str) -> None:
    d = _state_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "supervisor.log", "a", encoding="utf-8") as f:
        f.write(f"{time.time():.3f} {message}\n")


def supervisor_alive(root: str | Path, *, clock: Any = time.time,
                     stale: float = 60.0) -> dict[str, Any] | None:
    """Live supervisor info, or None (absent/stale/dead pid)."""
    p = _state_dir(root) / "supervisor.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if clock() - float(data.get("last_seen", 0.0)) > stale:
        return None
    pid = int(data.get("pid", -1))
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)  # existence probe only (signal 0)
    except (ProcessLookupError, PermissionError):
        return None
    return data


def ensure_supervisor(
    root: str | Path,
    *,
    launcher: Any = None,
    clock: Any = time.time,
    wait: float = 5.0,
    poll: float = 0.05,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Start the per-user supervisor daemon if it is not already alive.
    Delegation model (12 §12.4): the CLI never spawns the worker — it
    only guarantees the daemon is up and the request is queued."""
    root = Path(root)
    alive = supervisor_alive(root, clock=clock)
    if alive is not None:
        return {"started": False, "pid": int(alive["pid"])}

    launcher = launcher or _default_supervisor_launcher
    pid = launcher(root)
    deadline = clock() + wait
    while clock() < deadline:
        alive = supervisor_alive(root, clock=clock)
        if alive is not None:
            return {"started": True, "pid": int(alive["pid"])}
        sleep(poll)
    raise PreconditionFailed(
        f"supervisor daemon did not come up within {wait:.0f}s "
        f"(launcher pid {pid}); check state/supervisor.log",
        hint="the daemon only spawns workers — training never runs under "
             "your shell (12 §12.4)",
    )


def _default_supervisor_launcher(root: Path) -> int:
    log = _state_dir(root)
    log.mkdir(parents=True, exist_ok=True)
    with open(log / "supervisor.log", "a", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mlforge.supervisor", "--root", str(root)],
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ),
        )
    return proc.pid


__all__ = [
    "DEFAULT_HEARTBEAT_TIMEOUT",
    "DEFAULT_SCAN_INTERVAL",
    "Supervisor",
    "drain_pending",
    "enqueue_spawn",
    "ensure_supervisor",
    "supervisor_alive",
]


# ---------------------------------------------------------------------------
# process entry: `python -m mlforge.supervisor --root <workspace>`
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="mlforge.supervisor",
        description="MLForge per-user supervisor daemon (12 §12.4)",
    )
    p.add_argument("--root", required=True)
    p.add_argument("--poll", type=float, default=0.5)
    p.add_argument("--scan-interval", type=float, default=DEFAULT_SCAN_INTERVAL)
    args = p.parse_args(argv)
    sup = Supervisor(args.root, interval=args.scan_interval)
    try:
        sup.daemon_loop(poll_interval=args.poll)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry
    import sys as _sys

    _sys.exit(_main())


__all__ = ["DEFAULT_HEARTBEAT_TIMEOUT", "DEFAULT_SCAN_INTERVAL", "Supervisor"]
