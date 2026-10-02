"""Thin-client UI layer (build step 12) — TUI + GUI over ONE Workflow API.

13 §1: "Three interfaces (CLI, TUI, GUI) call one Workflow API. Never
implement business logic per interface." Composition:

    viewmodel.py  shared reads + display shaping (no transitions)
    screens.py    pure TUI frames + the key state machine (no I/O)
    app.py        the raw-terminal loop shell (I/O only)
    web.py        the localhost GUI (pure route() + HTML renderer)

Core invariant (13 §9.1): STATUS DOWN → TRAINING CONTINUES. Every screen
here is an observation layer; keys/buttons write control INTENTS the
worker obeys — the viewer never transitions a run.
"""

from mlforge.ui.viewmodel import (
    EVENT_TAIL,
    control_intent,
    dashboard_model,
    default_selection,
    run_detail,
    run_events,
    run_row,
)

__all__ = [
    "EVENT_TAIL",
    "control_intent",
    "dashboard_model",
    "default_selection",
    "run_detail",
    "run_events",
    "run_row",
]
