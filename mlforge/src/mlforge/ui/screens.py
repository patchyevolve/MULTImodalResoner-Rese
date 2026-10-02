"""Pure TUI screens + the key state machine (13 §4.1 Live dashboard,
§9.6 watch, §9.7 events).

Everything here is a pure function: frames are `list[str]` built from the
shared view model, and `handle_key` is a data-in/data-out state machine
(the only side effect is writing a control INTENT — same channel as the
CLI, never a transition). No terminal I/O lives in this module, so the
whole TUI is testable without a TTY.

Key model (normalized): single characters as-is, plus "up", "down",
"enter", "esc" (the app loop translates raw bytes / escape sequences).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from mlforge.status import render_watch
from mlforge.ui.viewmodel import control_intent

#: Dashboard content width inside the box (watch uses 60; the dashboard
#: carries run ids + states + dispositions and needs more room — 70
#: keeps a full row inside an 80-column terminal).
DASH_WIDTH = 70

CONTROL_KEYS = {"p": "pause", "s": "stop", "c": "checkpoint"}


# -- small display helpers --------------------------------------------------


def _age(ts: float | None, now: float | None = None) -> str:
    if not ts:
        return "—"
    delta = max(0.0, (now if now is not None else _now()) - ts)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _now() -> float:
    import time

    return time.time()


def _fit(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _box_row(text: str, width: int = DASH_WIDTH) -> str:
    return "║ " + _fit(text, width).ljust(width) + " ║"


def _section(title: str) -> str:
    return _box_row(title)


# -- frames -----------------------------------------------------------------


def render_dashboard(model: dict[str, Any], selection: int = 0,
                     *, now: float | None = None) -> list[str]:
    """Dashboard overview: runs (selectable) + models + datasets (13 §1
    primary objects). FAILED rows always show their disposition (§9.5)."""
    top = "╔" + "═" * (DASH_WIDTH + 2) + "╗"
    mid = "╠" + "═" * (DASH_WIDTH + 2) + "╣"
    bot = "╚" + "═" * (DASH_WIDTH + 2) + "╝"
    counts = model.get("counts") or {}
    lines = [
        top,
        _box_row(f"MLForge Dashboard            "
                 f"{counts.get('runs', 0)} runs · "
                 f"{counts.get('models', 0)} models · "
                 f"{counts.get('datasets', 0)} datasets"),
        mid,
        _section("RUNS"),
    ]
    runs = model.get("runs") or []
    if not runs:
        lines.append(_box_row("  no runs — `mlforge train --config F`"))
    for i, run in enumerate(runs):
        marker = "▸ " if i == selection else "  "
        row = f"{marker}{_fit(str(run.get('id')), 30)}  {str(run.get('state')):<12}"
        model_name = run.get("model")
        if model_name:
            row += f" {model_name:<12}"
        failure = run.get("failure")
        if failure:
            row += f" {failure.get('recovery')}"
            lines.append(_box_row(row))
            # 13 §9.5: the cause gets its own line — never truncated away
            if failure.get("cause"):
                lines.append(_box_row(f"      cause: {failure['cause']}"))
        else:
            row += f"  {_age(run.get('updated_ts'), now)}"
            lines.append(_box_row(row))
    lines.append(_section("MODELS"))
    models = model.get("models") or []
    if not models:
        lines.append(_box_row("  no models — published when a run COMPLETES"))
    for m in models:
        lines.append(_box_row(
            f"  {_fit(str(m.get('ref')), 24)}  {str(m.get('state')):<16} "
            f"{str(m.get('origin') or 'run')}"
        ))
    lines.append(_section("DATASETS"))
    datasets = model.get("datasets") or []
    if not datasets:
        lines.append(_box_row("  no datasets — `mlforge dataset add <ID> <PATH>`"))
    for d in datasets:
        version = d.get("version") or "v?"
        lines.append(_box_row(
            f"  {_fit(str(d.get('id')), 24)}  {str(d.get('state')):<16} {version}"
        ))
    lines.append(bot)
    lines.append(" [j/k] move  [Enter] run  [e] events  [r] refresh  "
                 "[?] help  [q]uit viewer")
    return lines


def render_run(detail: dict[str, Any]) -> list[str]:
    """Run screen: the §9.6 watch frame + a back hint (dashboard-only)."""
    lines = list(render_watch(detail))
    lines.append(" [b] back  [p/s/c] control intents  [e] events")
    return lines


def render_events(run_id: str, events: list[dict[str, Any]]) -> list[str]:
    """Event feed (13 §9.7): `HH:MM:SS  event  detail`, tail of the
    append-only journal — never a synthesized story."""
    top = "╔" + "═" * (DASH_WIDTH + 2) + "╗"
    bot = "╚" + "═" * (DASH_WIDTH + 2) + "╝"
    lines = [top, _box_row(f"Events — {run_id}")]
    if not events:
        lines.append(_box_row("  no events yet"))
    for ev in events:
        import datetime as _dt

        ts = ev.get("ts")
        stamp = (
            _dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            if isinstance(ts, (int, float)) else "--:--:--"
        )
        detail_bits = [
            f"{k}={v}" for k, v in ev.items()
            if k not in ("ts", "event", "action") and v is not None
        ]
        row = f"{stamp}  {ev.get('event', '?')}"
        if detail_bits:
            row += "  " + " ".join(detail_bits[:3])
        lines.append(_box_row(row))
    lines.append(bot)
    lines.append(" [b] back  [q] quit viewer")
    return lines


def render_help() -> list[str]:
    top = "╔" + "═" * (DASH_WIDTH + 2) + "╗"
    bot = "╚" + "═" * (DASH_WIDTH + 2) + "╝"
    return [
        top,
        _box_row("Help — MLForge dashboard (a VIEWER, 13 §9)"),
        _box_row(""),
        _box_row("  j/k or ↑/↓   move the run selection"),
        _box_row("  Enter / o     open the run (§9.6 frame)"),
        _box_row("  e            events feed for the selected run (§9.7)"),
        _box_row("  p/s/c        pause / stop / checkpoint INTENT"),
        _box_row("               (the worker obeys; the viewer never"),
        _box_row("                transitions anything itself)"),
        _box_row("  r            refresh now (reads are file-based)"),
        _box_row("  b / Esc      back one screen"),
        _box_row("  q            quit the viewer only — training continues"),
        bot,
        " [b] back",
    ]


# -- key state machine ------------------------------------------------------


@dataclass(frozen=True)
class TuiState:
    """Screen cursor — a value, mutated only by replacement. `stack` is
    the back-navigation path so [b] pops exactly one screen."""

    screen: str = "overview"          # overview | run | events | help
    run_ids: tuple[str, ...] = ()
    selection: int = 0
    run_id: str | None = None         # active run on run/events screens
    stack: tuple[str, ...] = ()
    notice: str | None = None         # transient last-action message

    @property
    def selected_run(self) -> str | None:
        if not self.run_ids:
            return None
        idx = min(max(self.selection, 0), len(self.run_ids) - 1)
        return self.run_ids[idx]


def _push(state: TuiState, screen: str) -> TuiState:
    return replace(state, screen=screen,
                   stack=state.stack + (state.screen,), notice=None)


def _back(state: TuiState) -> TuiState:
    if not state.stack:
        return replace(state, screen="overview", notice=None)
    return replace(state, screen=state.stack[-1],
                   stack=state.stack[:-1], notice=None)


def handle_key(wf, state: TuiState, key: str) -> tuple[TuiState, str | None]:
    """One keypress → (next state, outcome).

    outcome: "quit" · "intent:<action>" · None. p/s/c write a control
    INTENT (13 §9.6) and NOTHING else — no state transition ever happens
    in the viewer. `q` exits the viewer only; training continues (§9.1).
    """
    if key == "q":
        return state, "quit"

    if state.screen == "overview":
        if key in ("j", "down"):
            return replace(state, selection=min(
                state.selection + 1, max(0, len(state.run_ids) - 1)),
                notice=None), None
        if key in ("k", "up"):
            return replace(state, selection=max(state.selection - 1, 0),
                           notice=None), None
        if key in ("enter", "o"):
            run_id = state.selected_run
            if run_id is None:
                return replace(state, notice="no runs to open"), None
            nxt = _push(state, "run")
            return replace(nxt, run_id=run_id), None
        if key == "e":
            run_id = state.selected_run
            if run_id is None:
                return replace(state, notice="no runs selected"), None
            nxt = _push(state, "events")
            return replace(nxt, run_id=run_id), None
        if key in ("?", "h"):
            return _push(state, "help"), None
        if key == "r":
            return replace(state, notice="refreshed"), None
        if key in CONTROL_KEYS:
            return replace(
                state,
                notice="open a run first ([Enter]) — controls live on the "
                       "run screen",
            ), None
        return state, None

    if state.screen == "run":
        if key in CONTROL_KEYS:
            action = CONTROL_KEYS[key]
            assert state.run_id is not None  # run screen always has one
            control_intent(wf.root, state.run_id, action,
                           requested_by="cli:tui")
            return replace(
                state,
                notice=f"{action} requested for {state.run_id} "
                       "(worker obeys; viewer transitions nothing)",
            ), f"intent:{action}"
        if key == "e" and state.run_id:
            return _push(state, "events"), None
        if key in ("b", "esc"):
            return _back(state), None
        if key in ("?", "h"):
            return _push(state, "help"), None
        return state, None

    if state.screen == "events":
        if key in ("b", "esc"):
            return _back(state), None
        if key in ("?", "h"):
            return _push(state, "help"), None
        return state, None

    if state.screen == "help":
        if key in ("b", "esc", "enter", "?", "h"):
            return _back(state), None
        return state, None

    return state, None


__all__ = [
    "CONTROL_KEYS",
    "DASH_WIDTH",
    "TuiState",
    "handle_key",
    "render_dashboard",
    "render_events",
    "render_help",
    "render_run",
]
