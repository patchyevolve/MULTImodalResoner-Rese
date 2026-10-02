"""Shared read-only view model — the single data source both UIs render.

13 §1 design rule 2: "Three interfaces (CLI, TUI, GUI) call one Workflow
API. Never implement business logic per interface." This module is where
the TUI and the GUI AGREE: it performs only reads and display shaping —
never a state transition, never a rule the Workflow API already owns.

Design rule 3 applies too: everything here is an observation layer,
never authoritative for correctness (13 §9.1).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound
from mlforge.status import collect_run

#: How many tail events the events screen shows (13 §9.7 tail -f style).
EVENT_TAIL = 20


def _run_model_name(root: Path, run_id: str) -> str | None:
    """Display-only: the run's model definition from run_spec.json.
    A missing/malformed spec never breaks the viewer (read-only layer)."""
    spec_path = root / "runs" / run_id / "run_spec.json"
    if not spec_path.is_file():
        return None
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model = data.get("model") if isinstance(data, dict) else None
    return str(model) if model else None


def run_row(root: Path, projection: dict[str, Any]) -> dict[str, Any]:
    """One dashboard row — projection fields + display-only extras.
    FAILED rows always carry their disposition (13 §9.5): the viewer
    never makes the user guess whether `resume` is allowed."""
    failure = projection.get("failure") or None
    return {
        "id": projection.get("id"),
        "state": projection.get("state"),
        "model": _run_model_name(root, str(projection.get("id"))),
        "updated_ts": projection.get("updated_ts"),
        "failure": (
            {
                "recovery": failure.get("recovery"),
                "cause": failure.get("cause"),
            }
            if isinstance(failure, dict) else None
        ),
    }


def dashboard_model(wf) -> dict[str, Any]:
    """The whole dashboard payload (runs + models + datasets), read-only.

    Ordering is the registry's own order (created → listed); selection
    defaulting (most recently active run) is done by the view, not here —
    this is data, not behavior.
    """
    root = Path(wf.root)
    runs = [run_row(root, r) for r in wf.list_runs()]
    models = [
        {
            "ref": (f"{m.get('name')}:{m.get('version')}"
                    if m.get("name") else m.get("model_id")),
            "model_id": m.get("model_id"),
            "state": m.get("state"),
            "run_id": m.get("run_id"),
            "origin": m.get("origin") or "run",
        }
        for m in wf.list_models()
    ]
    datasets = [
        {
            "id": d.get("dataset_id") or d.get("id"),
            "state": d.get("state"),
            "version": d.get("version"),
        }
        for d in wf.list_datasets()
    ]
    return {
        "kind": "dashboard",
        "generated_ts": time.time(),
        "runs": runs,
        "models": models,
        "datasets": datasets,
        "counts": {
            "runs": len(runs),
            "models": len(models),
            "datasets": len(datasets),
        },
    }


def default_selection(runs: list[dict[str, Any]]) -> int:
    """Default cursor: most recently updated run (active work first)."""
    if not runs:
        return 0
    return max(
        range(len(runs)),
        key=lambda i: (runs[i].get("state") == "RUNNING",
                       runs[i].get("updated_ts") or 0.0),
    )


def run_detail(wf, run_id: str) -> dict[str, Any]:
    """L1/L2 detail for one run — the same collector `watch RUN` uses
    (13 §9.6). NotFound ⇒ exit 2 for an unknown run."""
    detail = collect_run(Path(wf.root), run_id)
    if detail.get("run_id") != run_id:
        raise NotFound(f"run {run_id!r} not found")
    return detail


def run_events(wf, run_id: str, *, limit: int = EVENT_TAIL) -> list[dict[str, Any]]:
    """Tail of the append-only event journal (13 §9.7), oldest → newest
    within the tail so the feed reads chronologically."""
    wf._require("run", run_id)  # NotFound ⇒ exit 2 (same as status)
    events = wf.get_run_events(run_id)
    tail = events[-limit:] if limit and len(events) > limit else events
    return list(tail)


def control_intent(root: Path, run_id: str, action: str,
                   *, requested_by: str) -> dict[str, Any]:
    """Write a control INTENT the worker obeys (13 §9.6) — the viewer
    never transitions anything itself. Shared by TUI keys and GUI
    buttons so both follow the identical channel as `pause`/`stop`."""
    from mlforge.runtime.control import write_control

    return write_control(root, run_id, action, requested_by=requested_by)


__all__ = [
    "EVENT_TAIL",
    "control_intent",
    "dashboard_model",
    "default_selection",
    "run_detail",
    "run_events",
    "run_row",
]
