"""Status & telemetry plane — build step 7 (13 §9).

Normative: 13_product_specification.md §9, §4.1 (OBSERVABILITY).

Core invariant: STATUS DOWN → TRAINING CONTINUES (read-only; this layer
never writes state, never transitions a run — 13 §4.1 "read-only,
always safe"). Liveness rule: state.json never proves a live process —
only a fresh heartbeat does (12 §16.1 LIVE STATE). Never infer lifecycle
from telemetry (13 §9.4).

Layered status: L1 overview / L2 training detail / L3 hardware telemetry
(implementation in `mlforge.status.layer`)."""

from mlforge.status.layer import (
    HEARTBEAT_STALE_SECONDS,
    STAGES,
    collect_hardware,
    collect_overview,
    collect_run,
    gpu_telemetry,
    render_hardware,
    render_l1,
    render_l2,
    render_overview,
    render_watch,
)

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
