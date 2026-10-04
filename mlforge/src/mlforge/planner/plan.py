"""Execution plan — the solver's output, validated before start
(12 §13, 13 §6.1 "SHOW PLAN", 13 §17 "Execution plan" row).

    ExecutionPlan = capabilities + chosen (micro, accum, world)
                    + precision resolution + topology + all solutions

Invariants carried by every plan:

  * global_batch is INPUT (frozen semantic) and OUTPUT must equal it —
    micro × accum × world == semantic.global_batch (12 §8.3);
  * precision fallback (bf16 → fp32) or CPU-only forces PORTABLE
    (12 §14: fallback "must produce execution_mode = PORTABLE");
  * the plan never mutates semantic — execution → semantic is
    forbidden (12 §13.3); the plan is a pure function of
    (run_spec, capabilities, runtime overrides).

The plan artifact persists at `runs/<id>/execution_plan.json` when the
gate's step 10 runs (machine-local: a resume on different hardware
regenerates it — the invariant checked later is the global batch, not
the exact micro/accum split).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash
from mlforge.planner.capabilities import Capabilities, detect_capabilities
from mlforge.planner.solver import MemoryProfile, solve_or_block
from mlforge.run_spec import RunSpec

PLAN_SCHEMA_VERSION = 1
PLAN_FILENAME = "execution_plan.json"

GiB = 1 << 30


# ---------------------------------------------------------------------------
# precision policy (12 §14)
# ---------------------------------------------------------------------------


def normalize_precision_policy(value: Any) -> dict[str, str]:
    """`"bf16"` or `{preferred, fallback}` → one shape. Unknown ⇒ BLOCK
    (never silently reinterpret a policy — 12 §17)."""
    if isinstance(value, str) and value:
        return {"preferred": value, "fallback": "fp32"}
    if isinstance(value, dict) and value.get("preferred"):
        return {
            "preferred": str(value["preferred"]),
            "fallback": str(value.get("fallback", "fp32")),
        }
    raise ValidationBlock(
        f"precision_policy unsupported: {value!r} (expected 'bf16' or a "
        f"{{'preferred': ..., 'fallback': ...}} mapping — schema_version 1)"
    )


def resolve_precision(policy: dict[str, str], caps: Capabilities) -> tuple[str, bool]:
    """→ (effective precision, fallback_used). Neither supported ⇒ BLOCK
    with migration options (12 §17: "Precision policy supported → Block
    or migration")."""
    preferred, fallback = policy["preferred"], policy["fallback"]
    if caps.supports(preferred):
        return preferred, False
    if caps.supports(fallback):
        return fallback, True
    raise ValidationBlock(
        f"precision policy unsupported on {caps.summary}: preferred "
        f"{preferred!r} and fallback {fallback!r} are both unavailable "
        f"(capability features: {caps.features})",
        hint="migration options: run on a host whose architecture "
             "supports the policy / fork with a supported precision / cancel",
    )


# ---------------------------------------------------------------------------
# topology (12 §8.5)
# ---------------------------------------------------------------------------


def topology_for(world_size: int, caps: Capabilities) -> dict[str, Any]:
    """Single-node topology identity (multi-node scheduling arrives with
    the distributed runtime; today node_count=1 is measured, not assumed
    — world_size can never exceed the measured gpu_count)."""
    on_gpu = caps.gpu_count > 0
    return {
        "node_count": 1,
        "gpus_per_node": world_size if on_gpu else 0,
        "world_size": world_size,
        "rank_assignment": "static",
        "gpu_mapping": [str(i) for i in range(world_size)] if on_gpu else [],
        "collective_backend": "nccl" if on_gpu else "gloo",
        "network_fabric": caps.interconnect if on_gpu else "none",
        "sharding_strategy": "ddp",
    }


# ---------------------------------------------------------------------------
# runtime overrides (execution knobs — 12 §13.3 execution namespace)
# ---------------------------------------------------------------------------


def load_runtime(run_dir: Path) -> dict[str, Any]:
    """Execution-only overrides written by `train --config` (state/runtime.json)."""
    p = run_dir / "state" / "runtime.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreconditionFailed(f"runtime config unreadable: {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise PreconditionFailed(f"runtime config must be a JSON object: {p}")
    return data


def _runtime_memory(runtime: dict[str, Any]) -> tuple[MemoryProfile, int | None, int]:
    profile = MemoryProfile.from_dict(runtime.get("memory_profile"))
    max_micro = runtime.get("max_micro_batch")
    min_micro = int(runtime.get("min_micro_batch", 1))
    return profile, (int(max_micro) if max_micro is not None else None), min_micro


# ---------------------------------------------------------------------------
# plan artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionPlan:
    model: str
    global_batch: int
    micro_batch: int
    grad_accum: int
    world_size: int
    precision_preferred: str
    precision_effective: str
    precision_fallback_used: bool
    portable_required: bool
    capabilities: dict[str, Any]
    capabilities_identity: str
    topology: dict[str, Any]
    solutions: tuple[dict[str, Any], ...]
    estimated_vram_bytes: int | None
    memory_profile: dict[str, int]
    schema_version: int = PLAN_SCHEMA_VERSION

    @property
    def identity(self) -> str:
        """Plan identity — distinguishes a regenerated plan (new host)
        from the same plan (idempotent write, no journal spam)."""
        return content_hash(
            {k: v for k, v in self._fields().items() if k != "schema_version"}
        )

    @property
    def product(self) -> int:
        return self.micro_batch * self.grad_accum * self.world_size

    def _fields(self) -> dict[str, Any]:
        """Plan fields WITHOUT the derived identity (no recursion)."""
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "global_batch": self.global_batch,
            "micro_batch": self.micro_batch,
            "grad_accum": self.grad_accum,
            "world_size": self.world_size,
            "precision_preferred": self.precision_preferred,
            "precision_effective": self.precision_effective,
            "precision_fallback_used": self.precision_fallback_used,
            "portable_required": self.portable_required,
            "capabilities": self.capabilities,
            "capabilities_identity": self.capabilities_identity,
            "topology": self.topology,
            "solutions": list(self.solutions),
            "estimated_vram_bytes": self.estimated_vram_bytes,
            "memory_profile": self.memory_profile,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._fields(), "identity": self.identity}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionPlan:
        if d.get("schema_version") != PLAN_SCHEMA_VERSION:
            raise ValidationBlock(
                f"unsupported execution_plan schema_version: "
                f"{d.get('schema_version')!r} (supported: {PLAN_SCHEMA_VERSION})",
                hint="old schemas require explicit migration: v1 → migration → v2",
            )
        return cls(
            model=d["model"],
            global_batch=int(d["global_batch"]),
            micro_batch=int(d["micro_batch"]),
            grad_accum=int(d["grad_accum"]),
            world_size=int(d["world_size"]),
            precision_preferred=d.get("precision_preferred", ""),
            precision_effective=d.get("precision_effective", ""),
            precision_fallback_used=bool(d.get("precision_fallback_used")),
            portable_required=bool(d.get("portable_required")),
            capabilities=dict(d.get("capabilities") or {}),
            capabilities_identity=str(d.get("capabilities_identity", "")),
            topology=dict(d.get("topology") or {}),
            solutions=tuple(d.get("solutions") or ()),
            estimated_vram_bytes=d.get("estimated_vram_bytes"),
            memory_profile=dict(d.get("memory_profile") or {}),
            schema_version=int(d.get("schema_version", PLAN_SCHEMA_VERSION)),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(path)  # atomic (12 §11.1)

    @classmethod
    def read(cls, path: Path) -> ExecutionPlan:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def summary_line(self) -> str:
        """13 §6.1: `micro batch 2 · accumulation 16 · GPUs 1 · global batch 32`."""
        devices = (f"GPUs {self.world_size}" if self.capabilities.get("gpu_count")
                   else "CPU")
        return (f"micro batch {self.micro_batch} · accumulation {self.grad_accum} "
                f"· {devices} · global batch {self.global_batch}")


def build_plan(
    spec: RunSpec,
    capabilities: Capabilities | None = None,
    *,
    runtime: dict[str, Any] | None = None,
) -> ExecutionPlan:
    """Pure function of (semantic identity, measured capabilities,
    execution overrides). Raises ValidationBlock when infeasible — the
    gate turns that into a FAIL at step 10, the CLI into exit 1."""
    caps = capabilities if capabilities is not None else detect_capabilities()
    runtime = runtime or {}
    semantic = spec.semantic

    try:
        global_batch = int(semantic["global_batch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationBlock(
            f"semantic.global_batch missing or not an integer: "
            f"{semantic.get('global_batch')!r}"
        ) from exc

    policy = normalize_precision_policy(semantic.get("precision_policy"))
    effective, fallback_used = resolve_precision(policy, caps)
    profile, max_micro, min_micro = _runtime_memory(runtime)

    chosen, solutions = solve_or_block(
        global_batch, caps,
        profile=profile, min_micro=min_micro, max_micro=max_micro,
        context=f"model {spec.model}",
    )
    # 12 §14 + §21: precision fallback OR CPU-only ⇒ PORTABLE, never EXACT.
    portable = fallback_used or caps.gpu_count == 0
    return ExecutionPlan(
        model=spec.model,
        global_batch=global_batch,
        micro_batch=chosen.micro_batch,
        grad_accum=chosen.grad_accum,
        world_size=chosen.world_size,
        precision_preferred=policy["preferred"],
        precision_effective=effective,
        precision_fallback_used=fallback_used,
        portable_required=portable,
        capabilities=caps.to_dict(),
        capabilities_identity=caps.identity,
        topology=topology_for(chosen.world_size, caps),
        solutions=tuple(s.to_dict() for s in solutions),
        estimated_vram_bytes=chosen.vram_bytes,
        memory_profile=profile.to_dict(),
    )


# ---------------------------------------------------------------------------
# disk estimate (replaces the gate step-16 fact — 13 §11 step 9)
# ---------------------------------------------------------------------------


def _registered_dataset_bytes(root: Path, ref: str) -> int:
    """`total_bytes` from a dataset's registration — machine-independent
    (identity.json), available without touching local paths."""
    text = ref.split("://", 1)[-1]
    name = text.split(":", 1)[0]
    p = root / "datasets" / name / "identity.json"
    if not p.is_file():
        return 0
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("total_bytes") or 0)
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0


def disk_estimate_components(
    root: Path,
    spec: RunSpec,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Worst-case disk inputs (12 §7.3) — shared by the gate's step 16
    (facts) and the preflight context so both use the SAME numbers.
    `runtime` overrides (`train --config` → state/runtime.json) win."""
    runtime = runtime or {}
    dataset_bytes = sum(
        _registered_dataset_bytes(root, ref)
        for ref in (
            *spec.train_datasets,
            *([spec.val_dataset] if spec.val_dataset else []),
        )
    )
    return {
        "checkpoint_bytes": int(runtime.get("checkpoint_bytes", 4 * GiB)),
        "log_bytes": int(runtime.get("log_bytes", 2 * GiB)),
        "safety_margin_bytes": int(runtime.get("safety_margin_bytes", 2 * GiB)),
        "dataset_cache_bytes": int(runtime.get("dataset_cache_bytes",
                                               dataset_bytes)),
    }


def estimate_required_disk_bytes(
    root: Path,
    spec: RunSpec,
    *,
    runtime: dict[str, Any] | None = None,
) -> int:
    """Worst-case disk for gate step 16 (12 §7.3 formula) + the planner's
    dataset-cache term (registered dataset bytes) — the estimate the
    workflow comment promised when the planner arrived (13 §11 step 9)."""
    c = disk_estimate_components(root, spec, runtime=runtime)
    # checkpoint + temp checkpoint (transactional double-write) + dataset
    # cache + logs + safety margin — same shape preflight mandates
    return 2 * c["checkpoint_bytes"] + c["dataset_cache_bytes"] \
        + c["log_bytes"] + c["safety_margin_bytes"]


__all__ = [
    "PLAN_FILENAME",
    "PLAN_SCHEMA_VERSION",
    "ExecutionPlan",
    "build_plan",
    "disk_estimate_components",
    "estimate_required_disk_bytes",
    "load_runtime",
    "normalize_precision_policy",
    "resolve_precision",
    "topology_for",
]
