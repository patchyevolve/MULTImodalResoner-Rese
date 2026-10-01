"""Capability-based execution planner — build step 9.

Normative: 12_training_system.md §13 (capability negotiation, feasibility
solver, hard namespace separation), §8.3 (global batch identity), §14
(precision policy → PORTABLE), §8.5 (topology), §6.5 (execution capability
profile), 13 §6.1 (SHOW PLAN), §11 step 9.

Feasibility solver: find micro_batch × grad_accum × world_size ==
frozen global_batch subject to VRAM/precision/model-min constraints.
Hard namespace separation: execution → semantic mutation is forbidden;
adapting micro/accum/workers is free, LR/optimizer/loss/dataset are not.

Modules:
    capabilities — hardware → measured capability profile (never tables)
    solver       — the constrained search + deterministic pick policy
    plan         — ExecutionPlan artifact + disk estimate for step 16
"""

from mlforge.planner.capabilities import (
    PRECISIONS,
    Capabilities,
    GPUInfo,
    detect_capabilities,
)
from mlforge.planner.plan import (
    PLAN_FILENAME,
    PLAN_SCHEMA_VERSION,
    ExecutionPlan,
    build_plan,
    estimate_required_disk_bytes,
    load_runtime,
    normalize_precision_policy,
    resolve_precision,
    topology_for,
)
from mlforge.planner.solver import (
    MemoryProfile,
    Solution,
    feasible_solutions,
    pick,
    solve_or_block,
)

__all__ = [
    "Capabilities",
    "ExecutionPlan",
    "GPUInfo",
    "MemoryProfile",
    "PLAN_FILENAME",
    "PLAN_SCHEMA_VERSION",
    "PRECISIONS",
    "Solution",
    "build_plan",
    "detect_capabilities",
    "estimate_required_disk_bytes",
    "feasible_solutions",
    "load_runtime",
    "normalize_precision_policy",
    "pick",
    "resolve_precision",
    "solve_or_block",
    "topology_for",
]
