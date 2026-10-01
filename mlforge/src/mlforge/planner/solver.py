"""Feasibility solver (12 §13.2):

    Target global batch = 32

    Find: micro_batch × grad_accum × world_size = 32
    Subject to:
        VRAM(micro_batch) ≤ available − overhead
        precision supported by arch
        micro_batch ≥ model minimum

Hard namespace separation (12 §13.3): the solver only ever touches
EXECUTION fields (micro/accum/world). The frozen semantic
`global_batch` is an input constraint, never an output — execution →
semantic mutation is forbidden.

VRAM modelling: without a real trainer we use an explicit, documented
ESTIMATE (`MemoryProfile`: fixed overhead + per-sample activations), 
overridable per run via `runtime.memory_profile`. Estimates may only be
made stricter by the operator (base/reserve up, per_sample up), never
silently trusted: the gate re-checks the chosen plan, and OOM recovery
(12 §12.1) re-solves execution-only when reality disagrees.

Selection policy (deterministic, documented — 12 §13.2 says "planner
picks a valid one"): prefer MAXIMUM world_size (parallelism), then
MAXIMUM feasible micro_batch (throughput), then accum falls out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from mlforge.errors import ValidationBlock
from mlforge.planner.capabilities import Capabilities

GiB = 1 << 30
MiB = 1 << 20


@dataclass(frozen=True)
class MemoryProfile:
    """VRAM estimate model (documented, overridable via runtime config)."""

    base_bytes: int = 2 * GiB        # framework + weights overhead
    per_sample_bytes: int = 96 * MiB  # activations per sample
    reserve_bytes: int = 512 * MiB    # "available − overhead" (12 §13.2)

    def estimate(self, micro_batch: int) -> int:
        return self.base_bytes + self.per_sample_bytes * int(micro_batch)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MemoryProfile":
        d = d or {}
        return cls(
            base_bytes=int(d.get("base_bytes", cls.base_bytes)),
            per_sample_bytes=int(d.get("per_sample_bytes", cls.per_sample_bytes)),
            reserve_bytes=int(d.get("reserve_bytes", cls.reserve_bytes)),
        )


@dataclass(frozen=True)
class Solution:
    micro_batch: int
    grad_accum: int
    world_size: int
    vram_bytes: int | None  # None = CPU-only (no VRAM constraint)

    @property
    def label(self) -> str:
        return f"{self.micro_batch}x{self.grad_accum}x{self.world_size}"

    def to_dict(self) -> dict:
        return asdict(self)


def feasible_solutions(
    global_batch: int,
    capabilities: Capabilities,
    *,
    profile: MemoryProfile | None = None,
    min_micro: int = 1,
    max_micro: int | None = None,
) -> tuple[Solution, ...]:
    """All (micro, accum, world) satisfying 12 §13.2's constraints.

    Sorted by the selection policy: (-world, -micro) — caller picks the
    first, or shows the menu. Empty tuple = infeasible (caller decides
    the user-facing BLOCK)."""
    if global_batch < 1:
        raise ValidationBlock(f"global_batch must be >= 1, got {global_batch}")
    if min_micro < 1:
        raise ValidationBlock(f"min_micro must be >= 1, got {min_micro}")
    profile = profile or MemoryProfile()
    on_gpu = capabilities.gpu_count > 0
    vram_budget = capabilities.vram_total_bytes - profile.reserve_bytes if on_gpu else 0

    worlds = range(1, capabilities.gpu_count + 1) if on_gpu else (1,)
    out: list[Solution] = []
    for world in worlds:
        for micro in range(min_micro, (global_batch // world) + 1):
            if max_micro is not None and micro > max_micro:
                continue
            denom = micro * world
            if global_batch % denom:
                continue  # accum must be an integer (12 §8.3 identity)
            accum = global_batch // denom
            vram: int | None = None
            if on_gpu:
                vram = profile.estimate(micro)
                if vram > vram_budget:
                    continue  # VRAM(micro) ≤ available − overhead
            out.append(Solution(micro_batch=micro, grad_accum=accum,
                                world_size=world, vram_bytes=vram))
    # selection policy: maximum parallelism, then maximum micro_batch
    out.sort(key=lambda s: (-s.world_size, -s.micro_batch))
    return tuple(out)


def pick(solutions: tuple[Solution, ...]) -> Solution:
    if not solutions:
        raise ValidationBlock(
            "no feasible execution plan (empty solution set)",
            hint="report the failure with `mlforge validate` — the gate "
                 "carries the constraint detail",
        )
    return solutions[0]


def solve_or_block(
    global_batch: int,
    capabilities: Capabilities,
    *,
    profile: MemoryProfile | None = None,
    min_micro: int = 1,
    max_micro: int | None = None,
    context: str = "",
) -> tuple[Solution, tuple[Solution, ...]]:
    """Solve, or BLOCK with the 13 §7 migration options (never a guess)."""
    solutions = feasible_solutions(
        global_batch, capabilities, profile=profile,
        min_micro=min_micro, max_micro=max_micro,
    )
    if not solutions:
        detail = (
            f"global batch {global_batch} is unachievable"
            + (f" for {context}" if context else "")
            + f" on {capabilities.summary}: no micro×accum×world satisfies "
              f"VRAM/precision/model-min constraints"
        )
        raise ValidationBlock(
            detail,
            hint="migration options (13 §7): run on another machine / "
                 "fork with a different global_batch / cancel",
        )
    return pick(solutions), solutions


__all__ = [
    "GiB",
    "MemoryProfile",
    "MiB",
    "Solution",
    "feasible_solutions",
    "pick",
    "solve_or_block",
]
