"""Build step 12 tests — TUI/GUI over the same Workflow API (13 §1, §9).

Specs as executable checks:
  * §1 rule 2: three interfaces, ONE Workflow API — both UIs consume the
    shared view model; no business logic lives in a screen or a page
  * §1 rule 3 / §9.1: read-only observation — the viewer never
    transitions a run; keys/buttons write control INTENTS only;
    STATUS DOWN → TRAINING CONTINUES (closing a viewer changes nothing)
  * §4.1 `watch [RUN]`: without RUN = the Live dashboard; with RUN = the
    §9.6 frame (unchanged); no runs ⇒ exit 2; non-TTY ⇒ one frame
  * §9.5: FAILED rows always carry their disposition
  * §9.6: p/s/c are intents through the same channel as the CLI;
    `q` quits the viewer only
  * §9.7: events screen = tail of the append-only journal
  * §9.9: the view is rebuilt from persistence each tick (vanished run
    drops back to the overview with a notice, never a crash)
  * GUI: localhost only, HTML from the same view model, run page embeds
    the §9.6 frame, unknown run/action fails closed (404/400),
    intents carry `requested_by=cli:gui`
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from mlforge.cli.main import main
from mlforge.run_spec import RunSpec
from mlforge.runtime.control import read_control
from mlforge.ui.screens import (
    TuiState,
    handle_key,
    render_dashboard,
    render_events,
    render_help,
    render_run,
)
from mlforge.ui.viewmodel import (
    EVENT_TAIL,
    control_intent,
    dashboard_model,
    default_selection,
    run_detail,
    run_events,
    run_row,
)
from mlforge.ui.app import _decode, _frame
from mlforge.ui.web import (
    INTENT_ACTIONS,
    make_handler,
    render_dashboard_html,
    render_run_html,
    route,
)
from mlforge.workflow import WorkflowAPI


# -- helpers ---------------------------------------------------------------


def _spec() -> RunSpec:
    return RunSpec(
        model="rf_detr_s",
        train_datasets=("coco_2017:v1",),
        semantic={
            "optimizer": "adamw", "learning_rate": 1e-4, "scheduler": "cosine",
            "loss": "l1", "seed": 42, "global_batch": 32, "epochs": 50,
            "precision_policy": "bf16",
        },
    )


def _make_runs(wf, count: int = 2) -> list[str]:
    return [wf.create_run(_spec()).run_id for _ in range(count)]


def _to_running(wf, run_id: str) -> None:
    wf.begin_validation(run_id)
    wf.validation_pass(run_id)
    wf.preflight_pass(run_id)


def _state(**kw) -> TuiState:
    return TuiState(**kw)


# -- viewmodel (shared reads) ------------------------------------------------


def test_dashboard_model_shape_and_counts(wf):
    ids = _make_runs(wf, 3)
    model = dashboard_model(wf)
    assert model["kind"] == "dashboard"
    assert model["counts"] == {"runs": 3, "models": 0, "datasets": 0}
    assert [r["id"] for r in model["runs"]] == ids
    assert all(r["state"] == "CREATED" for r in model["runs"])
    assert "generated_ts" in model


def test_dashboard_model_is_read_only(wf):
    _make_runs(wf, 1)
    before = wf.get_run_state(wf.list_runs()[0]["id"])
    dashboard_model(wf)
    run_events_snapshot = wf.get_run_events(wf.list_runs()[0]["id"])
    dashboard_model(wf)
    assert wf.get_run_state(wf.list_runs()[0]["id"]) == before
    assert wf.get_run_events(wf.list_runs()[0]["id"]) == run_events_snapshot


def test_run_row_carries_failed_disposition(wf):
    """13 §9.5: FAILED rows always show recovery — no guessing."""
    run_id = _make_runs(wf, 1)[0]
    _to_running(wf, run_id)
    wf.runtime_error(run_id, "CUDA OOM at step 41200",
                     has_valid_checkpoint=False)
    projection = wf.get_run_status(run_id)
    row = run_row(Path(wf.root), projection)
    assert row["state"] == "FAILED"
    assert row["failure"]["recovery"] == "FORK_ONLY"
    assert "CUDA OOM" in row["failure"]["cause"]


def test_default_selection_prefers_running_then_recent(wf):
    stopped = _make_runs(wf, 1)[0]
    running = _make_runs(wf, 1)[0]
    _to_running(wf, running)
    runs = [run_row(Path(wf.root), r) for r in wf.list_runs()]
    assert runs[default_selection(runs)]["id"] == running
    # no RUNNING → most recently updated wins
    _to_running(wf, stopped)
    runs = [run_row(Path(wf.root), r) for r in wf.list_runs()]
    assert isinstance(default_selection(runs), int)
    assert default_selection([]) == 0


def test_run_events_is_tail_and_chronological(wf):
    run_id = _make_runs(wf, 1)[0]
    _to_running(wf, run_id)
    full = wf.get_run_events(run_id)
    assert len(full) >= 2
    # explicit limit slices the tail exactly
    assert run_events(wf, run_id, limit=2) == full[-2:]
    # default: everything up to EVENT_TAIL, chronological (13 §9.7)
    default = run_events(wf, run_id)
    assert default == (full if len(full) <= EVENT_TAIL else full[-EVENT_TAIL:])
    assert [e["ts"] for e in default] == sorted(e["ts"] for e in default)


def test_run_detail_and_events_unknown_run_exit_2(wf):
    with pytest.raises(Exception) as exc:
        run_detail(wf, "run_MISSING")
    from mlforge.errors import NotFound
    assert isinstance(exc.value, NotFound)
    with pytest.raises(NotFound):
        run_events(wf, "run_MISSING")


def test_control_intent_shared_channel_never_transitions(wf):
    run_id = _make_runs(wf, 1)[0]
    control_intent(Path(wf.root), run_id, "pause", requested_by="cli:tui")
    ctrl = read_control(Path(wf.root), run_id)
    assert ctrl["action"] == "pause"
    assert wf.get_run_state(run_id) == "CREATED"  # intent ≠ transition


# -- screens (pure frames) ---------------------------------------------------


def test_render_dashboard_frame_layout(wf):
    _make_runs(wf, 2)
    model = dashboard_model(wf)
    lines = render_dashboard(model, selection=1)
    joined = "\n".join(lines)
    assert "MLForge Dashboard" in joined
    assert "RUNS" in joined and "MODELS" in joined and "DATASETS" in joined
    assert "▸ " in joined                      # selection marker
    assert "[j/k] move" in joined and "[q]uit viewer" in joined
    assert "2 runs · 0 models · 0 datasets" in joined
    for line in lines:                          # fits an 80-column terminal
        assert len(line) <= 80, line


def test_render_dashboard_failed_row_shows_disposition(wf):
    run_id = _make_runs(wf, 1)[0]
    _to_running(wf, run_id)
    wf.runtime_error(run_id, "invariant violation (run_spec mismatch)",
                     cause_is_semantic=True)
    lines = render_dashboard(dashboard_model(wf), selection=0)
    joined = "\n".join(lines)
    assert "FAILED" in joined and "FORK_ONLY" in joined
    assert "invariant violation" in joined


def test_render_dashboard_empty_states(wf):
    lines = render_dashboard(dashboard_model(wf))
    joined = "\n".join(lines)
    assert "no runs — `mlforge train --config F`" in joined
    assert "no models" in joined and "no datasets" in joined


def test_render_run_embeds_watch_frame_plus_back_hint(wf):
    run_id = _make_runs(wf, 1)[0]
    detail = run_detail(wf, run_id)
    lines = render_run(detail)
    joined = "\n".join(lines)
    assert f"MLForge — {run_id}" in joined      # §9.6 frame, reused
    assert "[b] back" in joined
    assert "[q]uit viewer" in joined


def test_render_events_feed_format(wf):
    run_id = _make_runs(wf, 1)[0]
    lines = render_events(run_id, run_events(wf, run_id))
    joined = "\n".join(lines)
    assert f"Events — {run_id}" in joined
    assert "run_created" in joined
    assert "[b] back" in joined
    # HH:MM:SS stamp on the row
    assert any(len(line) > 20 and "  " in line for line in lines[2:-2])


def test_render_help_lists_every_key(wf):
    joined = "\n".join(render_help())
    for token in ("j/k", "Enter", "p/s/c", "b / Esc", "quit the viewer"):
        assert token in joined, token


# -- key state machine (pure) -------------------------------------------------


def test_handle_key_navigation_and_clamping():
    ids = ("run_a", "run_b")
    st = _state(run_ids=ids, selection=0)
    st, outcome = handle_key(None, st, "j")
    assert st.selection == 1 and outcome is None
    st, outcome = handle_key(None, st, "j")       # clamps at the end
    assert st.selection == 1
    st, outcome = handle_key(None, st, "k")
    assert st.selection == 0
    st, outcome = handle_key(None, st, "k")       # clamps at the start
    assert st.selection == 0
    st, outcome = handle_key(None, st, "up")
    assert st.selection == 0
    st, outcome = handle_key(None, st, "enter")
    assert st.screen == "run" and st.run_id == "run_a"
    assert st.stack == ("overview",)


def test_handle_key_run_controls_write_intents_only(wf):
    run_id = _make_runs(wf, 1)[0]
    st = _state(run_ids=(run_id,), run_id=run_id, screen="run")
    for key, action in (("p", "pause"), ("s", "stop"), ("c", "checkpoint")):
        st, outcome = handle_key(wf, st, key)
        assert outcome == f"intent:{action}"
        ctrl = read_control(Path(wf.root), run_id)
        assert ctrl["action"] == action
        assert ctrl["requested_by"] == "cli:tui"
        assert action in st.notice and "viewer transitions nothing" in st.notice
    assert wf.get_run_state(run_id) == "CREATED"   # never a transition


def test_handle_key_controls_need_an_open_run(wf):
    run_id = _make_runs(wf, 1)[0]
    st = _state(run_ids=(run_id,), screen="overview", selection=0)
    st, outcome = handle_key(wf, st, "p")
    assert outcome is None
    assert "open a run first" in st.notice
    assert not (Path(wf.root) / "runs" / run_id / "control.json").exists()


def test_handle_key_stack_back_and_help():
    ids = ("run_a",)
    st = _state(run_ids=ids, selection=0)
    st, _ = handle_key(None, st, "e")              # overview → events
    assert st.screen == "events" and st.stack == ("overview",)
    st, _ = handle_key(None, st, "?")              # events → help
    assert st.screen == "help" and st.stack == ("overview", "events")
    st, _ = handle_key(None, st, "b")              # help → events
    assert st.screen == "events" and st.stack == ("overview",)
    st, _ = handle_key(None, st, "esc")            # events → overview
    assert st.screen == "overview" and st.stack == ()
    st, _ = handle_key(None, st, "b")              # at root: stays overview
    assert st.screen == "overview"


def test_handle_key_quit_from_every_screen():
    st = _state(run_ids=("run_a",), screen="overview")
    for screen in ("overview", "run", "events", "help"):
        cur = _state(run_ids=("run_a",), screen=screen, run_id="run_a")
        _, outcome = handle_key(None, cur, "q")
        assert outcome == "quit", screen


def test_handle_key_refresh_notice():
    st = _state(run_ids=("run_a",))
    st, outcome = handle_key(None, st, "r")
    assert outcome is None and st.notice == "refreshed"
    st, outcome = handle_key(None, st, "j")        # next key clears it
    assert st.notice is None


def test_handle_key_no_runs_safe():
    st = _state(run_ids=(), selection=0)
    for key in ("j", "k", "enter", "e"):
        st, outcome = handle_key(None, st, key)
        assert outcome is None
    assert st.screen == "overview"


# -- app shell (decode + vanished-run resilience) -----------------------------


def test_decode_translates_escape_sequences():
    assert _decode(b"\x1b[A") == "up"
    assert _decode(b"\x1b[B") == "down"
    assert _decode(b"\x1b") == "esc"
    assert _decode(b"\x1b[") == "esc"
    assert _decode(b"\r") == "enter"
    assert _decode(b"\n") == "enter"
    assert _decode(b"q") == "q"
    assert _decode(b"") is None


def test_frame_vanished_run_drops_to_overview(wf):
    """13 §9.9: the view reconstructs from persistence — never crashes."""
    run_id = _make_runs(wf, 1)[0]
    model = dashboard_model(wf)
    st = _state(run_ids=(run_id,), screen="run", run_id="run_DELETED")
    lines, new_state = _frame(wf, model, st)
    assert new_state.screen == "overview"
    assert new_state.stack == ()
    assert "back to overview" in new_state.notice
    assert any("MLForge Dashboard" in line for line in lines)


# -- CLI: watch = Live dashboard ----------------------------------------------


def test_watch_without_run_renders_dashboard(tmp_path, capsys):
    wf = WorkflowAPI(tmp_path / "workspace")
    run_id = _make_runs(wf, 1)[0]
    # non-TTY in tests → exactly one dashboard frame, exit 0
    assert main(["--root", str(wf.root), "watch"]) == 0
    out = capsys.readouterr().out
    assert "MLForge Dashboard" in out
    assert run_id in out
    assert "[j/k] move" in out
    assert wf.get_run_state(run_id) == "CREATED"   # viewer never transitions


def test_watch_json_returns_dashboard_model(tmp_path, capsys):
    wf = WorkflowAPI(tmp_path / "workspace")
    _make_runs(wf, 2)
    assert main(["--root", str(wf.root), "watch", "--json"]) == 0
    model = json.loads(capsys.readouterr().out)
    assert model["kind"] == "dashboard"
    assert model["counts"]["runs"] == 2


def test_watch_json_without_runs_exits_2(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "watch", "--json"]) == 2
    assert "no runs to watch" in capsys.readouterr().err


def test_watch_with_run_stays_the_single_run_frame(tmp_path, capsys):
    wf = WorkflowAPI(tmp_path / "workspace")
    run_id = _make_runs(wf, 1)[0]
    assert main(["--root", str(wf.root), "watch", run_id]) == 0
    out = capsys.readouterr().out
    assert f"MLForge — {run_id}" in out          # §9.6 frame, not dashboard
    assert "MLForge Dashboard" not in out


# -- CLI: gui command ----------------------------------------------------------


def test_gui_is_implemented_and_validates_port(tmp_path, capsys):
    from mlforge.cli.main import _IMPLEMENTED, _PENDING

    assert "gui" in _IMPLEMENTED and "gui" not in _PENDING
    # invalid port → exit 1 (ValidationBlock), nothing bound
    assert main(["--root", str(tmp_path), "gui", "--port", "99999"]) == 1
    err = capsys.readouterr().err
    assert "invalid --port" in err


# -- GUI routes (pure) ----------------------------------------------------------


def test_route_dashboard_html_lists_objects(wf):
    run_id = _make_runs(wf, 1)[0]
    status, ctype, body = route("GET", "/", wf)
    assert status == 200
    assert ctype.startswith("text/html")
    assert "MLForge Dashboard" in body
    assert run_id in body
    assert "TRAINING CONTINUES" in body


def test_route_run_page_embeds_watch_frame_events_and_buttons(wf):
    run_id = _make_runs(wf, 1)[0]
    status, _, body = route("GET", f"/run/{run_id}", wf)
    assert status == 200
    assert f"MLForge — {run_id}" in body          # §9.6 frame embedded
    assert "run_created" in body                  # events feed
    for action in INTENT_ACTIONS:
        assert f"intent?action={action}" in body
    assert "viewer never transitions" in body


def test_route_unknown_run_404(wf):
    status, ctype, body = route("GET", "/run/run_MISSING", wf)
    assert status == 404 and "not found" in body
    status, _, _ = route("POST", "/run/run_MISSING/intent?action=pause", wf)
    assert status == 404


def test_route_bad_action_400_and_method_405(wf):
    run_id = _make_runs(wf, 1)[0]
    status, _, body = route("POST", f"/run/{run_id}/intent?action=rm", wf)
    assert status == 400 and "unknown action" in body
    status, _, _ = route("GET", f"/run/{run_id}/intent?action=pause", wf)
    assert status == 405
    status, _, _ = route("POST", "/", wf)
    assert status == 404


def test_route_intent_writes_control_redirect_no_transition(wf):
    run_id = _make_runs(wf, 1)[0]
    status, _, location = route(
        "POST", f"/run/{run_id}/intent?action=pause", wf
    )
    assert status == 303 and location == f"/run/{run_id}"
    ctrl = read_control(Path(wf.root), run_id)
    assert ctrl["action"] == "pause"
    assert ctrl["requested_by"] == "cli:gui"
    assert wf.get_run_state(run_id) == "CREATED"   # intent ≠ transition


def test_route_root_404_and_pages_escape_html(wf):
    run_id = _make_runs(wf, 1)[0]
    _to_running(wf, run_id)
    wf.runtime_error(run_id, "<script>alert('x')</script>",
                     has_valid_checkpoint=False)
    status, _, body = route("GET", "/", wf)
    assert status == 200
    assert "<script>alert" not in body             # disposition escaped
    assert "&lt;script&gt;" in body
    status, _, _ = route("GET", "/nowhere", wf)
    assert status == 404


# -- GUI real server round-trip -------------------------------------------------


def test_server_roundtrip_over_http(wf):
    import http.client
    from http.server import ThreadingHTTPServer

    run_id = _make_runs(wf, 1)[0]
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(wf))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert b"MLForge Dashboard" in resp.read()
        conn.request("GET", f"/run/{run_id}")
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
        conn.request("POST", f"/run/{run_id}/intent?action=stop")
        resp = conn.getresponse()
        assert resp.status == 303
        assert resp.getheader("Location") == f"/run/{run_id}"
        resp.read()
        conn.close()
        ctrl = read_control(Path(wf.root), run_id)
        assert ctrl["action"] == "stop"
        assert wf.get_run_state(run_id) == "CREATED"
    finally:
        server.shutdown()
        server.server_close()


def test_html_render_helpers_are_pure(wf):
    model = dashboard_model(wf)
    assert render_dashboard_html(model).startswith("<!doctype html>")
    detail = run_detail(wf, _make_runs(wf, 1)[0])
    page = render_run_html("run_x", detail, [])
    assert page.startswith("<!doctype html>")
    assert "no events" in page
