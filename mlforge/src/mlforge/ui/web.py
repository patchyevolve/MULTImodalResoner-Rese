"""Localhost GUI — a thin web client over the SAME Workflow API (13 §1).

The GUI is deliberately the smallest possible browser UI:
  * stdlib only (`http.server`), no JS dependencies, no external assets;
  * binds 127.0.0.1 — the local control plane (13 §9.2), never exposed;
  * pages are rendered from `mlforge.ui.viewmodel` (shared with the TUI)
    and the run screen embeds the exact §9.6 watch frame;
  * interaction = control INTENTS (pause/stop/checkpoint) through the
    same channel as the CLI and TUI keys — the page never transitions a
    run (13 §9.6), and closing the tab changes nothing (§9.1: STATUS
    DOWN → TRAINING CONTINUES).

`route()` is the testable core: (method, path) → (status, content_type,
body) with no sockets involved.
"""

from __future__ import annotations

import html as _html
from typing import Any
from urllib.parse import parse_qs, urlparse

from mlforge.errors import NotFound
from mlforge.status import render_watch
from mlforge.ui.viewmodel import (
    control_intent,
    dashboard_model,
    run_detail,
    run_events,
)

HTML = "text/html; charset=utf-8"
TEXT = "text/plain; charset=utf-8"
INTENT_ACTIONS = ("pause", "stop", "checkpoint")


def esc(value: Any) -> str:
    return _html.escape(str(value), quote=True)


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{esc(title)}</title>"
        "<style>"
        "body{font-family:ui-monospace,Menlo,Consolas,monospace;"
        "margin:2rem;background:#0b0e14;color:#d8dee9}"
        "a{color:#88c0d0}h1{font-size:1.3rem}h2{font-size:1rem;"
        "color:#88c0d0;margin-top:1.6rem}"
        "table{border-collapse:collapse}"
        "td,th{padding:.25rem .8rem .25rem 0;border-bottom:1px solid #2a2f3a;"
        "text-align:left;vertical-align:top}"
        "pre{background:#141922;padding:1rem;border:1px solid #2a2f3a;"
        "line-height:1.35}"
        ".meta{color:#8890a0;font-size:.85rem}"
        ".state{font-weight:bold}"
        ".fail{color:#bf616a}"
        "button{font:inherit;background:#2e3440;color:#d8dee9;"
        "border:1px solid #4c566a;padding:.3rem .9rem;margin-right:.5rem;"
        "cursor:pointer}"
        "</style></head><body>"
        + body
        + "<p class=\"meta\">read-only observation layer — STATUS DOWN → "
          "TRAINING CONTINUES (13 §9.1)</p>"
        "</body></html>"
    )


def render_dashboard_html(model: dict[str, Any]) -> str:
    counts = model.get("counts") or {}
    rows = []
    for run in model.get("runs") or []:
        failure = run.get("failure")
        note = ""
        if failure:
            note = (f"<span class=\"fail\">{esc(failure.get('recovery'))}"
                    + (f" — {esc(failure.get('cause'))}" if failure.get("cause")
                       else "")
                    + "</span>")
        rows.append(
            "<tr>"
            f"<td><a href=\"/run/{esc(run.get('id'))}\">{esc(run.get('id'))}</a></td>"
            f"<td class=\"state\">{esc(run.get('state'))}</td>"
            f"<td>{esc(run.get('model') or '—')}</td>"
            f"<td>{note}</td>"
            "</tr>"
        )
    model_rows = [
        "<tr>"
        f"<td>{esc(m.get('ref'))}</td>"
        f"<td class=\"state\">{esc(m.get('state'))}</td>"
        f"<td>{esc(m.get('origin') or 'run')}</td>"
        "</tr>"
        for m in model.get("models") or []
    ]
    dataset_rows = [
        "<tr>"
        f"<td>{esc(d.get('id'))}</td>"
        f"<td class=\"state\">{esc(d.get('state'))}</td>"
        f"<td>{esc(d.get('version') or '—')}</td>"
        "</tr>"
        for d in model.get("datasets") or []
    ]
    body = (
        "<h1>MLForge Dashboard</h1>"
        f"<p class=\"meta\">{counts.get('runs', 0)} runs · "
        f"{counts.get('models', 0)} models · "
        f"{counts.get('datasets', 0)} datasets</p>"
        "<h2>Runs</h2>"
        "<table><tr><th>Run</th><th>State</th><th>Model</th><th>Disposition</th></tr>"
        + ("".join(rows) or "<tr><td colspan=4>no runs</td></tr>")
        + "</table>"
        "<h2>Models</h2>"
        "<table><tr><th>Model</th><th>State</th><th>Origin</th></tr>"
        + ("".join(model_rows) or "<tr><td colspan=3>no models</td></tr>")
        + "</table>"
        "<h2>Datasets</h2>"
        "<table><tr><th>Dataset</th><th>State</th><th>Version</th></tr>"
        + ("".join(dataset_rows) or "<tr><td colspan=3>no datasets</td></tr>")
        + "</table>"
    )
    return _page("MLForge Dashboard", body)


def render_run_html(run_id: str, detail: dict[str, Any],
                    events: list[dict[str, Any]]) -> str:
    frame = "\n".join(render_watch(detail))
    buttons = "".join(
        f"<form action=\"/run/{esc(run_id)}/intent?action={action}\" "
        "method=\"post\" style=\"display:inline\">"
        f"<button>{esc(action)}</button></form>"
        for action in INTENT_ACTIONS
    )
    event_rows = []
    import datetime as _dt

    for ev in events:
        ts = ev.get("ts")
        stamp = (
            _dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            if isinstance(ts, (int, float)) else "—"
        )
        extras = " ".join(
            f"{esc(k)}={esc(v)}" for k, v in ev.items()
            if k not in ("ts", "event") and v is not None
        )
        event_rows.append(
            f"<tr><td>{stamp}</td><td>{esc(ev.get('event'))}</td>"
            f"<td>{extras}</td></tr>"
        )
    state = detail.get("state")
    body = (
        f"<h1>{esc(run_id)} — <span class=\"state\">{esc(state)}</span></h1>"
        "<p class=\"meta\"><a href=\"/\">← dashboard</a></p>"
        "<pre>" + esc(frame) + "</pre>"
        "<p class=\"meta\">buttons write control INTENTS the worker obeys — "
        "the viewer never transitions anything (13 §9.6)</p>"
        + buttons
        + "<h2>Events</h2>"
        "<table><tr><th>Time</th><th>Event</th><th>Detail</th></tr>"
        + ("".join(event_rows) or "<tr><td colspan=3>no events</td></tr>")
        + "</table>"
    )
    return _page(f"{run_id} — MLForge", body)


def route(method: str, path: str, wf) -> tuple[int, str, str]:
    """(status, content_type, body) — the whole GUI surface.

    For 303 the body carries the redirect Location (handler sends it).
    Unknown runs/actions fail closed: 404/400, never a best-effort page.
    """
    parsed = urlparse(path)
    query = parse_qs(parsed.query)
    route_path = parsed.path or "/"
    if route_path != "/":
        route_path = route_path.rstrip("/")

    if method == "GET" and route_path == "/":
        return 200, HTML, render_dashboard_html(dashboard_model(wf))

    if route_path.startswith("/run/") and route_path.endswith("/intent"):
        run_id = route_path[len("/run/"):-len("/intent")]
        if method != "POST":
            return 405, TEXT, "POST required\n"
        action = (query.get("action") or [""])[0]
        if action not in INTENT_ACTIONS:
            return 400, TEXT, (
                f"unknown action: {action!r} (one of "
                f"{', '.join(INTENT_ACTIONS)})\n"
            )
        try:
            run_detail(wf, run_id)  # unknown run ⇒ 404, nothing written
        except NotFound:
            return 404, TEXT, f"run not found: {run_id}\n"
        control_intent(wf.root, run_id, action, requested_by="cli:gui")
        return 303, TEXT, f"/run/{run_id}"

    if route_path.startswith("/run/"):
        run_id = route_path[len("/run/"):]
        if method != "GET":
            return 405, TEXT, "GET required\n"
        try:
            detail = run_detail(wf, run_id)
            events = run_events(wf, run_id)
        except NotFound:
            return 404, TEXT, f"run not found: {run_id}\n"
        return 200, HTML, render_run_html(run_id, detail, events)

    return 404, TEXT, f"not found: {route_path}\n"


def make_handler(wf):
    """BaseHTTPRequestHandler bound to one workspace (thin dispatch)."""
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        server_version = "mlforge-gui"

        def log_message(self, *_args) -> None:  # quiet: no access log spam
            return

        def _dispatch(self, method: str) -> None:
            length = min(int(self.headers.get("Content-Length") or 0), 4096)
            body = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
            path = self.path
            if body and "action=" not in path:
                sep = "&" if "?" in path else "?"
                path = f"{path}{sep}{body}"
            status, ctype, payload = route(method, path, wf)
            data = payload.encode("utf-8")
            self.send_response(status)
            if status == 303:
                self.send_header("Location", payload)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

    return Handler


def serve(wf, *, port: int = 8765, open_browser: bool = False,
          host: str = "127.0.0.1") -> int:
    """Block serving the GUI until Ctrl+C — viewer exit code 0; training
    is untouched by whatever happens here (13 §9.1)."""
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer((host, port), make_handler(wf))
    bound = server.server_address[1]
    url = f"http://{host}:{bound}/"
    print(f"MLForge GUI on {url}  (localhost only — Ctrl+C stops the "
          "viewer; training continues)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nGUI stopped — training continues")
        return 0
    finally:
        server.server_close()
    return 0


__all__ = [
    "HTML",
    "INTENT_ACTIONS",
    "TEXT",
    "make_handler",
    "render_dashboard_html",
    "render_run_html",
    "route",
    "serve",
]
