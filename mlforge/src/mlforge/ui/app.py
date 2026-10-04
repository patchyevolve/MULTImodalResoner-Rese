"""The TUI loop shell — raw terminal I/O around the pure state machine.

I/O only: collect → render → read a key → `handle_key` → repeat. All
decisions live in `mlforge.ui.screens.handle_key` (pure) and all data in
`mlforge.ui.viewmodel` (reads). A viewer crash or a closed terminal
changes nothing about training (13 §9.1 STATUS DOWN → TRAINING
CONTINUES); terminal modes are always restored on the way out.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any

from mlforge.errors import NotFound
from mlforge.ui.screens import (
    TuiState,
    handle_key,
    render_dashboard,
    render_events,
    render_help,
    render_run,
)
from mlforge.ui.viewmodel import (
    dashboard_model,
    default_selection,
    run_detail,
    run_events,
)

_CLEAR = "\x1b[2J\x1b[H"


def _decode(raw: bytes) -> str | None:
    """Raw bytes → normalized key (see screens: up/down/enter/esc/char)."""
    text = raw.decode("utf-8", errors="ignore")
    if not text:
        return None
    if text.startswith("\x1b"):
        if text[1:3] == "[A":
            return "up"
        if text[1:3] == "[B":
            return "down"
        return "esc"
    ch = text[0]
    return "enter" if ch in ("\r", "\n") else ch


def _frame(wf, model: dict[str, Any], state: TuiState) -> tuple[list[str], TuiState]:
    """Screen lines for the current state. A run that vanished under the
    cursor (deleted elsewhere) drops the viewer back to the overview with
    a notice — read-only resilience, never a crash (13 §9.9: the view is
    reconstructed from persistence)."""
    if state.screen == "overview":
        lines = render_dashboard(model, state.selection)
    else:
        try:
            if state.screen == "run":
                lines = render_run(run_detail(wf, str(state.run_id)))
            elif state.screen == "events":
                lines = render_events(
                    str(state.run_id), run_events(wf, str(state.run_id))
                )
            else:
                lines = render_help()
        except NotFound as exc:
            state = replace(state, screen="overview", stack=(),
                            notice=f"{exc} — back to overview")
            lines = render_dashboard(model, state.selection)
    if state.notice:
        lines.append(f" ! {state.notice}")
    return lines, state


def run_dashboard(wf, *, interval: float = 0.5) -> int:
    """Interactive dashboard (13 §4.1 `mlforge watch` — Live dashboard).

    Non-TTY: exactly one frame, exit 0 (scripts and pipes get a report,
    never a hang). TTY: raw-mode loop until `q` / Ctrl+C — the viewer
    exits, training continues.
    """
    model = dashboard_model(wf)
    if not model["runs"]:
        raise NotFound(
            "no runs to watch",
            hint="`mlforge train --config F` first",
        )
    if not sys.stdin.isatty():
        selection = default_selection(model["runs"])
        print("\n".join(render_dashboard(model, selection)))
        return 0

    import os
    import select
    import termios
    import tty

    runs = model["runs"]
    state = TuiState(
        run_ids=tuple(str(r["id"]) for r in runs),
        selection=default_selection(runs),
    )
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            # Re-read persistence every tick (13 §9.9): the view is
            # rebuilt from files — state is never held by the viewer.
            model = dashboard_model(wf)
            ids = tuple(str(r["id"]) for r in model["runs"])
            if not ids:
                print("\nno runs remain — " + _CLEAR + "q exits the viewer")
                return 0
            state = replace(
                state,
                run_ids=ids,
                selection=min(state.selection, len(ids) - 1),
            )
            lines, state = _frame(wf, model, state)
            print(_CLEAR + "\n".join(lines), flush=True)
            ready, _, _ = select.select([fd], [], [], max(0.1, interval))
            if not ready:
                continue
            key = _decode(os.read(fd, 8))
            if key is None:
                continue
            state, outcome = handle_key(wf, state, key)
            if outcome == "quit":
                print("\nq exits the viewer only — training continues")
                return 0
    except KeyboardInterrupt:
        print("\ndetached — training continues")
        return 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


__all__ = ["_decode", "run_dashboard"]
