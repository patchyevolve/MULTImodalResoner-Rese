"""Supervisor — live-state monitoring and crash detection.

Normative: 12_training_system.md §12.3 (reconciliation after unexpected
events), §16 (LIVE STATE = supervisor heartbeat — dies with the process;
heartbeats gate *liveness reporting* only, §25 rule 5), 13 §5.3
(crash → INTERRUPTED), §9.4 ("a stale status.json saying RUNNING after a
crash is INTERRUPTED the moment the heartbeat expires").

Scope (build step 5):
  * crash DETECTION only — heartbeat expiry → WorkflowAPI.crash() →
    INTERRUPTED (never reported as PAUSED or RUNNING).
  * the supervisor NEVER starts, resumes, or reconciles anything:
    reconciliation is an explicit scan (12 §12.3) and `resume` is a user
    decision (13 §1 "Automatic continuation is NO").
  * it owns no state — every change goes through the same Workflow API
    path as the CLI (single write path: machine → journal → projection).

Heartbeat file: `runs/<id>/state/heartbeat.json` (13 §10), written by the
training worker (build step 6). A run that just entered RUNNING without a
heartbeat yet gets one full timeout of grace — only an *expired* heartbeat
(or missing file with no journal activity) counts as dead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

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
        except Exception as exc:  # one broken workspace must be reported, not fatal
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
            except Exception as exc:  # a broken run must not kill the scan
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
            for key in aggregate:
                aggregate[key].extend(report[key])
            scans += 1
            if max_scans is not None and scans >= max_scans:
                break
            time.sleep(self.interval)
        return aggregate


__all__ = ["Supervisor", "DEFAULT_HEARTBEAT_TIMEOUT", "DEFAULT_SCAN_INTERVAL"]
