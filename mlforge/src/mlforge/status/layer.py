"""Read-only status layer — L1/L2/L3 telemetry presentation (13 §9.4–9.5).

Contract:
  * READ-ONLY (13 §4.1): this module never writes run state, never
    transitions a run, never breaks/makes leases. A stale RUNNING
    projection is REPORTED here and reconciled only by `resume` /
    supervisor crash detection (12 §12.3) — status merely shows it.
  * Never infer lifecycle from telemetry (13 §9.4): state comes from the
    orchestrator projection; GPU/CPU numbers are diagnostic only.
  * Instant: bounded reads (events tail, metrics tail, one checkpoint
    verification of the NEWEST committed checkpoint only).

Layers (13 §9.5):
  L1 `mlforge status [RUN]`           overview / single-run block
  L2 `mlforge status [RUN] --verbose`  training detail (config, loops)
  L3 `mlforge hardware`               host telemetry (diagnostic)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound
from mlforge.leases import RunLeaseManager
from mlforge.run_spec import RunSpec
from mlforge.runtime.checkpoints import CheckpointStore

#: 13 §9.4 example: "WARNING — no heartbeat for 120s."
HEARTBEAT_STALE_SECONDS = 120.0

#: 13 §9.5 stage-aware status.
STAGES = ("TRAINING", "VALIDATING", "CHECKPOINTING", "EVALUATING", "PREPARING_DATA")

_METRICS_TAIL = 200  # bounded read — status must stay instant


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _metrics_tail(path: Path, limit: int = _METRICS_TAIL) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn tail line never breaks status
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out[-limit:]


def _speed(records: list[dict[str, Any]]) -> float | None:
    """it/s from the tail's timestamps (diagnostic — never lifecycle)."""
    ts = [r["ts"] for r in records if isinstance(r.get("ts"), (int, float))]
    if len(ts) < 2 or ts[-1] <= ts[0]:
        return None
    return round((len(ts) - 1) / (ts[-1] - ts[0]), 3)


def _format_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def gpu_telemetry() -> dict[str, Any] | None:
    """L3 GPU numbers via `nvidia-smi` (13 §9.5). `None` = unavailable —
    telemetry absence never changes what status believes about state."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, ("--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                  "temperature.gpu,power.draw"),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    parts = [p.strip() for p in out[0].split(",")]
    if len(parts) < 6:
        return None

    def _num(v: str) -> float | None:
        try:
            return float(v)
        except ValueError:
            return None

    return {
        "name": parts[0],
        "util_pct": _num(parts[1]),
        "vram_used_mb": _num(parts[2]),
        "vram_total_mb": _num(parts[3]),
        "temp_c": _num(parts[4]),
        "power_w": _num(parts[5]),
    }


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def collect_overview(root: str | Path) -> list[dict[str, Any]]:
    """L1 all-runs list (projection-only; one row per run)."""
    from mlforge.workflow import WorkflowAPI  # late: avoids import cycle

    return WorkflowAPI(Path(root)).list_runs()


def collect_run(
    root: str | Path,
    run_id: str,
    *,
    verbose: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """L1/L2 detail for one run — read-only, all sources file-based."""
    from mlforge.workflow import WorkflowAPI  # late: avoids import cycle

    root = Path(root)
    wf = WorkflowAPI(root)
    projection = wf.get_run_status(run_id)  # NotFound ⇒ exit 2
    run_dir = root / "runs" / run_id
    if not run_dir.is_dir():
        raise NotFound(f"run {run_id!r} not found under {root}")

    spec: dict[str, Any] = {}
    spec_path = run_dir / "run_spec.json"
    if spec_path.is_file():
        try:
            spec = RunSpec.from_dict(json.loads(spec_path.read_text(encoding="utf-8"))).to_dict()
        except Exception:  # noqa: BLE001 - a malformed spec must not break status
            spec = {}

    events = wf.get_run_events(run_id)
    live = _read_json(run_dir / "state" / "live.json") or {}
    heartbeat = _read_json(run_dir / "state" / "heartbeat.json")
    runtime = _read_json(run_dir / "state" / "runtime.json") or {}
    records = _metrics_tail(run_dir / "metrics" / "metrics.jsonl")

    ts_now = now if now is not None else __import__("time").time()
    hb_age = (ts_now - float(heartbeat["ts"])) if heartbeat and "ts" in heartbeat else None
    stale_limit = max(
        HEARTBEAT_STALE_SECONDS, 2.0 * float(runtime.get("heartbeat_interval", 30))
    )
    state = projection.get("state")
    heartbeat_stale = (
        state == "RUNNING" and (heartbeat is None or (hb_age or 0.0) > stale_limit)
    )

    # Last committed checkpoint (event authority) + bounded verification.
    # NOTE: journal events FLATTEN their data (Event.to_dict: ts, event, **data).
    last_ordinal: int | None = None
    for ev in reversed(events):
        if ev.get("event") == "checkpoint_committed":
            last_ordinal = ev.get("ordinal")
            if last_ordinal is not None:
                break
    checkpoint: dict[str, Any] = {"ordinal": last_ordinal, "step": None,
                                  "integrity": "none"}
    store = CheckpointStore(run_dir)
    if last_ordinal is not None:
        manifest = _read_json(store.dir_for(int(last_ordinal)) / "manifest.json") or {}
        checkpoint["step"] = manifest.get("global_step")
        ok, detail = store.verify(int(last_ordinal))
        checkpoint["integrity"] = "VALID" if ok else "INVALID"
        checkpoint["integrity_detail"] = detail

    # Progress / speed / ETA (telemetry — NEVER lifecycle).
    last = records[-1] if records else {}
    step = live.get("global_step", last.get("global_step"))
    epoch = live.get("epoch", last.get("epoch"))
    total_steps = None
    if runtime.get("epochs") and runtime.get("steps_per_epoch"):
        total_steps = int(runtime["epochs"]) * int(runtime["steps_per_epoch"])
    progress_pct = (
        round(100.0 * int(step) / total_steps, 1)
        if isinstance(step, int) and total_steps
        else None
    )
    speed = _speed(records)
    eta_seconds = (
        round((total_steps - int(step)) / speed)
        if (speed and total_steps and isinstance(step, int) and total_steps > step)
        else None
    )

    try:
        lease = RunLeaseManager(root).status(run_id)
        lease_info = lease.to_dict() if hasattr(lease, "to_dict") else dict(vars(lease))
    except Exception:  # noqa: BLE001 - a broken lease must not break the status layer
        lease_info = None

    detail: dict[str, Any] = {
        "run_id": run_id,
        "state": state,
        "model": spec.get("model"),
        "failure": projection.get("failure"),
        "live": live or None,
        "stage": live.get("stage") if state == "RUNNING" else None,
        "step": step,
        "epoch": epoch,
        "total_steps": total_steps,
        "progress_pct": progress_pct,
        "loss": live.get("loss", last.get("loss")),
        "metrics": live.get("metrics") or (last.get("metrics") if isinstance(last.get("metrics"), dict) else None),
        "lr": last.get("lr", (live.get("metrics") or {}).get("lr")),
        "speed_it_s": speed,
        "eta_seconds": eta_seconds,
        "checkpoint": checkpoint,
        "heartbeat": heartbeat,
        "heartbeat_age_seconds": hb_age,
        "heartbeat_stale": heartbeat_stale,
        "heartbeat_stale_limit": stale_limit,
        "gpu": gpu_telemetry(),
        "last_event": events[-1] if events else None,
        "updated_ts": projection.get("updated_ts"),
        "lease": lease_info,
    }
    if verbose:
        detail["semantic"] = dict(spec.get("semantic") or {})
        detail["runtime"] = runtime
        detail["datasets"] = {
            "train": spec.get("train_datasets"),
            "val": spec.get("val_datasets") or spec.get("val_dataset"),
        }
        detail["identity"] = {"run_spec_hash": projection.get("run_spec_hash")}
        detail["recent_records"] = records[-10:]
        # best metric over history when a comparable series exists
        for key in ("mAP", "map", "accuracy"):
            series = [
                (r[key], r.get("epoch")) for r in records
                if isinstance(r.get(key), (int, float))
            ]
            if series:
                best_val, best_epoch = max(series)
                detail["best"] = {"metric": key, "value": best_val, "epoch": best_epoch}
                break
    return detail


def collect_hardware(root: str | Path) -> dict[str, Any]:
    """L3 — host telemetry, diagnostic only (13 §9.5). Never lifecycle."""
    out: dict[str, Any] = {"gpu": gpu_telemetry()}
    try:
        out["load_avg"] = list(os.getloadavg())
    except OSError:
        out["load_avg"] = None
    mem: dict[str, Any] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                mem[k.lower() + "_kb"] = int(v.strip().split()[0])
        if mem.get("memtotal_kb"):
            used = mem["memtotal_kb"] - mem.get("memavailable_kb", 0)
            mem["used_pct"] = round(100.0 * used / mem["memtotal_kb"], 1)
    except (OSError, ValueError, IndexError):
        mem = {}
    out["memory"] = mem or None
    try:
        usage = shutil.disk_usage(Path(root))
        out["disk"] = {
            "total_gb": round(usage.total / 1e9, 1),
            "free_gb": round(usage.free / 1e9, 1),
            "used_pct": round(100.0 * usage.used / usage.total, 1),
        }
    except OSError:
        out["disk"] = None
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_l1(detail: dict[str, Any]) -> list[str]:
    """13 §9.5 Level-1 block (+ stage-aware + FAILED disposition)."""
    lines: list[str] = [str(detail["state"])]

    if detail.get("heartbeat_stale"):
        age = detail.get("heartbeat_age_seconds")
        lines += [
            "",
            f"WARNING — no heartbeat for {age:.0f}s."
            if age is not None else "WARNING — heartbeat missing.",
            "Possible: stalled process / hung kernel / machine unreachable / crash.",
            ("Run will be INTERRUPTED (not failed) on reconciliation — "
            f"awaiting `mlforge resume {detail['run_id']}` (12 §12.3)."),
        ]

    head = f"Run {detail['run_id']}"
    if detail.get("model"):
        head = f"Model {detail['model']} · {head}"
    lines.append(head)

    if detail.get("stage"):
        lines.append(f"  Stage: {detail['stage']}")
        live = detail.get("live") or {}
        if live.get("saving"):
            lines.append(f"  Saving: {live['saving']}")

    progress_bits = []
    if detail.get("epoch") is not None:
        bit = f"Epoch {detail['epoch']}"
        if detail.get("progress_pct") is not None:
            bit += f" ({detail['progress_pct']:.0f}%)"
        progress_bits.append(bit)
    if detail.get("step") is not None:
        progress_bits.append(f"Step {detail['step']:,}")
    if progress_bits:
        lines.append(" · ".join(progress_bits))

    metric_bits = []
    if detail.get("loss") is not None:
        metric_bits.append(f"Loss {detail['loss']}")
    if detail.get("metrics"):
        for k, v in detail["metrics"].items():
            if k != "lr":
                metric_bits.append(f"{k} {v}")
    if detail.get("best"):
        b = detail["best"]
        metric_bits.append(f"Best {b['value']} @ {b['epoch']}")
    if metric_bits:
        lines.append(" · ".join(metric_bits))

    speed_bits = []
    if detail.get("speed_it_s") is not None:
        speed_bits.append(f"Speed {detail['speed_it_s']} it/s")
    if detail.get("eta_seconds") is not None:
        speed_bits.append(f"ETA {_format_eta(detail['eta_seconds'])}")
    if speed_bits:
        lines.append(" · ".join(speed_bits))

    gpu = detail.get("gpu")
    if gpu:
        vram = f"{(gpu.get('vram_used_mb') or 0) / 1024:.1f}/{(gpu.get('vram_total_mb') or 0) / 1024:.1f} GB"
        bits = [gpu.get("name") or "GPU", f"VRAM {vram}"]
        if gpu.get("temp_c") is not None:
            bits.append(f"{gpu['temp_c']:.0f}°C")
        if gpu.get("power_w") is not None:
            bits.append(f"{gpu['power_w']:.0f}W")
        lines.append("GPU " + " · ".join(bits))

    ck = detail.get("checkpoint") or {}
    if ck.get("ordinal") is not None:
        step = ck.get("step")
        shown = f"step {step:,}" if step is not None else f"checkpoint-{ck['ordinal']}"
        lines.append(f"Checkpoint {shown} · Integrity {ck.get('integrity', 'unverified')}")
    else:
        lines.append("Checkpoint none")

    failure = detail.get("failure")
    if detail.get("state") == "FAILED" and failure:
        lines += ["", f"Cause:      {failure.get('cause')}"]
        recovery = failure.get("recovery")
        ckpt_step = (detail.get("checkpoint") or {}).get("step")
        if recovery == "RESUME":
            suffix = f" (valid checkpoint: step {ckpt_step:,})" if ckpt_step is not None else ""
            lines.append(f"Recovery:   RESUME{suffix}")
            lines.append(f"Action:     mlforge resume {detail['run_id']}   (after resolving cause)")
        else:
            lines.append("Recovery:   FORK_ONLY — no valid continuation exists")
            lines.append(
                f"Action:     mlforge fork {detail['run_id']}  |  mlforge retrain "
                f"{detail.get('model') or '<model>'}"
            )
    return lines


def render_l2(detail: dict[str, Any]) -> list[str]:
    """13 §9.5 Level-2 additions (training detail) — appended after L1."""
    lines = ["", "— verbose —"]
    sem = detail.get("semantic") or {}
    if sem:
        lines.append("Config  " + " · ".join(f"{k} {v}" for k, v in sem.items()))
    runtime = detail.get("runtime") or {}
    if runtime:
        lines.append("Runtime " + " · ".join(f"{k} {v}" for k, v in runtime.items()))
    datasets = detail.get("datasets") or {}
    if datasets.get("train"):
        lines.append(f"Train data    {', '.join(datasets['train'])}")
    if datasets.get("val"):
        val = datasets["val"]
        lines.append(f"Val data      {', '.join(val) if isinstance(val, list) else val}")
    if detail.get("lr") is not None:
        lines.append(f"LR {detail['lr']}")
    ck = detail.get("checkpoint") or {}
    lines.append(
        f"Checkpoint last: {ck.get('ordinal')} "
        f"(step {ck.get('step')}, integrity {ck.get('integrity')})"
    )
    if detail.get("best"):
        b = detail["best"]
        lines.append(f"Best {b['metric']} {b['value']} @ epoch {b['epoch']}")
    lease = detail.get("lease")
    if lease:
        lines.append(
            f"Lease {lease.get('state')} — pid {lease.get('pid')} on {lease.get('host')}"
        )
    hb_age = detail.get("heartbeat_age_seconds")
    if hb_age is not None:
        lines.append(f"Heartbeat {hb_age:.0f}s ago")
    if detail.get("identity", {}).get("run_spec_hash"):
        lines.append(f"run_spec {detail['identity']['run_spec_hash']}")
    return lines


def render_hardware(info: dict[str, Any]) -> list[str]:
    lines = []
    gpu = info.get("gpu")
    if gpu:
        lines.append(
            f"GPU  {gpu.get('name')} · util {gpu.get('util_pct')}% · "
            f"VRAM {(gpu.get('vram_used_mb') or 0) / 1024:.1f}/"
            f"{(gpu.get('vram_total_mb') or 0) / 1024:.1f} GB · "
            f"{gpu.get('temp_c')}°C · {gpu.get('power_w')}W"
        )
    else:
        lines.append("GPU  unavailable (nvidia-smi not found) — diagnostic only")
    load = info.get("load_avg")
    if load:
        lines.append(f"CPU  load {load[0]:.2f} / {load[1]:.2f} / {load[2]:.2f}")
    mem = info.get("memory")
    if mem:
        total_gb = (mem.get("memtotal_kb") or 0) / 1e6
        avail_gb = (mem.get("memavailable_kb") or 0) / 1e6
        lines.append(
            f"RAM  {total_gb - avail_gb:.1f}/{total_gb:.1f} GB "
            f"({mem.get('used_pct')}%)"
        )
    disk = info.get("disk")
    if disk:
        lines.append(
            f"Disk {disk['used_pct']}% used · {disk['free_gb']} GB free"
        )
    return lines


def render_overview(runs: list[dict[str, Any]]) -> list[str]:
    if not runs:
        return ["No runs."]
    return [
        r["id"] + "  " + r["state"]
        + ("  recovery=" + r["failure"]["recovery"]
           + "  cause=" + str(r["failure"].get("cause"))
           if r.get("failure") else "")
        for r in runs
    ]


_WATCH_WIDTH = 60  # content columns inside the box (13 §9.6)


def _box_row(text: str, width: int = _WATCH_WIDTH) -> str:
    return "║ " + text.ljust(width) + " ║"


def render_watch(detail: dict[str, Any]) -> list[str]:
    """13 §9.6 live dashboard frame. A VIEWER: `q` quits the viewer only."""
    top = "╔" + "═" * (_WATCH_WIDTH + 2) + "╗"
    mid = "╠" + "═" * (_WATCH_WIDTH + 2) + "╣"
    bot = "╚" + "═" * (_WATCH_WIDTH + 2) + "╝"
    state = str(detail.get("state"))
    lines = [
        top,
        _box_row(f"MLForge — {detail['run_id']}" + " " * 8 + state),
        mid,
        _box_row(str(detail.get("model") or "")),
    ]
    progress_bits = []
    if detail.get("epoch") is not None:
        bit = f"Epoch {detail['epoch']}"
        if detail.get("progress_pct") is not None:
            bit += f"  {detail['progress_pct']:.1f}%"
        progress_bits.append(bit)
    if detail.get("step") is not None:
        total = detail.get("total_steps")
        progress_bits.append(
            f"Step {detail['step']:,}" + (f"/{total:,}" if total else "")
        )
    if progress_bits:
        lines.append(_box_row("  ".join(progress_bits)))
    metric_bits = []
    if detail.get("loss") is not None:
        metric_bits.append(f"Loss {detail['loss']}")
    if detail.get("speed_it_s") is not None:
        metric_bits.append(f"Speed {detail['speed_it_s']} it/s")
    if detail.get("eta_seconds") is not None:
        metric_bits.append(f"ETA {_format_eta(detail['eta_seconds'])}")
    if metric_bits:
        lines.append(_box_row("  ".join(metric_bits)))
    gpu = detail.get("gpu")
    if gpu:
        lines.append(_box_row(
            f"GPU0 {gpu.get('name')}  Util {gpu.get('util_pct')}%  "
            f"VRAM {(gpu.get('vram_used_mb') or 0) / 1024:.1f}/"
            f"{(gpu.get('vram_total_mb') or 0) / 1024:.0f} GB  "
            f"{gpu.get('temp_c')}°C  {gpu.get('power_w')}W"
        ))
    ck = detail.get("checkpoint") or {}
    if ck.get("ordinal") is not None:
        lines.append(_box_row(
            f"Checkpoint {ck.get('step') if ck.get('step') is not None else ck['ordinal']}"
            f"   Integrity {ck.get('integrity', 'unverified')}"
        ))
    last = detail.get("last_event") or {}
    if last:
        lines.append(_box_row(f"Last event: {last.get('event')}"))
    if detail.get("heartbeat_stale"):
        lines.append(_box_row("WARNING: heartbeat stale — reconcile on resume"))
    lines.append(bot)
    lines.append(" [p]ause  [s]top  [c]heckpoint  [e]vents  [q]uit viewer")
    return lines


__all__ = [
    "HEARTBEAT_STALE_SECONDS",
    "STAGES",
    "collect_hardware",
    "collect_overview",
    "collect_run",
    "gpu_telemetry",
    "render_hardware",
    "render_l1",
    "render_l2",
    "render_overview",
    "render_watch",
]
