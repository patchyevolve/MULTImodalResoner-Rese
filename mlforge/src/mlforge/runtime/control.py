"""Worker control channel — pause/stop requests (13 §4.1, §12 §12.4).

The CLI (or TUI/GUI) never reaches into a running worker — it writes an
intent file; the worker reads it at its next control poll and performs
the graceful transition through the state machine. This keeps every
transition on the single write path and makes pause/stop survive a dead
CLI (`mlforge pause` returning does not mean the worker stopped — it
means the intent was recorded).

    runs/<id>/state/control.json   {"action": "pause"|"stop", ...}

An unreadable control file is quarantined (renamed `*.corrupt`), never
silently ignored and never able to crash the worker — the run must not
become INTERRUPTED because of a torn control write.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

CONTROL_ACTIONS = ("pause", "stop")


def control_path(root: str | Path, run_id: str) -> Path:
    return Path(root) / "runs" / run_id / "state" / "control.json"


def write_control(
    root: str | Path, run_id: str, action: str, *, requested_by: str = "cli"
) -> Path:
    if action not in CONTROL_ACTIONS:
        raise ValueError(f"unknown control action: {action!r}")
    p = control_path(root, run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"action": action, "requested_by": requested_by, "ts": time.time()}
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)  # readers never see a partial intent
    return p


def read_control(root: str | Path, run_id: str) -> dict[str, Any] | None:
    """Latest control intent, or None. Corrupt content is quarantined to
    `control.corrupt.json` and reported as an unknown action (the worker
    logs it via its own error path rather than crashing the run)."""
    p = control_path(root, run_id)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        p.replace(p.with_name("control.corrupt.json"))
        return {"action": "corrupt (quarantined)"}
    if not isinstance(data, dict) or data.get("action") not in CONTROL_ACTIONS:
        return {"action": "unknown", "raw": data}
    return data


def clear_control(root: str | Path, run_id: str) -> None:
    control_path(root, run_id).unlink(missing_ok=True)


def wait_for_state(
    root: str | Path,
    run_id: str,
    states: tuple[str, ...],
    *,
    timeout: float = 10.0,
    poll: float = 0.05,
    clock: Any = time.time,
    sleep: Any = time.sleep,
) -> str | None:
    """CLI-side wait after writing a control intent: returns the state when
    it reaches one of `states`, or None on timeout (the caller reports the
    current state honestly — "requested" ≠ "done")."""
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(root)
    deadline = clock() + timeout
    while True:
        current = wf.get_run_state(run_id)
        if current in states:
            return current
        if clock() >= deadline:
            return None
        sleep(poll)


__all__ = [
    "CONTROL_ACTIONS",
    "control_path",
    "write_control",
    "read_control",
    "clear_control",
    "wait_for_state",
]
