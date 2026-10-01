"""Preflight — host + identity checks before any expensive operation.

Normative: 12_training_system.md §7.3.

    Validates: OS, Python, container runtime, GPU, driver, VRAM, disk space,
    RAM, dataset hashes, code, environment, model, checkpoint, filesystem
    write permissions, atomic rename support, storage headroom.

    Only after PREFLIGHT PASSED does train/resume proceed.
    Disk-space validation is mandatory — worst case:
    checkpoint + temp checkpoint + dataset cache + logs + safety margin.
    If free < required → BLOCK. Never "train until disk fills."

Design:
  * Same fail-closed engine as the gate: a probe that cannot produce a
    verdict is a FAIL (no "probably fine").
  * Probes are injectable (`probes` mapping) so tests do not touch the
    real host, and later steps can add container-runtime/VRAM probes
    without changing this module.
  * Identity re-verification (dataset/code/env/checkpoint) uses the SAME
    provider keys as the resume gate — one vocabulary, two invocation
    points (12 §7.3 lists both host and identity concerns).
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.validation.gate import GateContext, Provider, scan_for_secrets
from mlforge.validation.report import FAIL, PASS, WARN, Check, ValidationReport

#: A probe answers one host check with (verdict, detail, data).
Probe = Callable[["PreflightContext"], tuple[str, str, dict[str, Any]]]

DEFAULT_MIN_RAM_BYTES = 8 * (1 << 30)          # 8 GiB floor
DEFAULT_SAFETY_MARGIN_BYTES = 2 * (1 << 30)    # 2 GiB margin


@dataclass
class PreflightContext(GateContext):
    """Preflight shares the gate's run context plus host expectations."""

    #: Worst-case sizing inputs (12 §7.3: disk-space validation mandatory).
    checkpoint_bytes: int = 4 * (1 << 30)          # committed ckpt
    dataset_cache_bytes: int = 0
    log_bytes: int = 2 * (1 << 30)
    safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES
    min_ram_bytes: int = DEFAULT_MIN_RAM_BYTES
    gpu_required: bool = True
    #: Overrides for injectable probes (tests / containers).
    probe_overrides: dict[str, Probe] = field(default_factory=dict)

    @property
    def required_disk_bytes(self) -> int:
        # checkpoint + temp checkpoint (transactional double-write) +
        # dataset cache + logs + safety margin
        return (
            2 * self.checkpoint_bytes
            + self.dataset_cache_bytes
            + self.log_bytes
            + self.safety_margin_bytes
        )


# ---------------------------------------------------------------------------
# Built-in host probes (stdlib only; injectable where the host may differ)
# ---------------------------------------------------------------------------

def probe_python(ctx: PreflightContext) -> tuple[str, str, dict[str, Any]]:
    v = sys.version_info
    detail = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 10):
        return FAIL, f"{detail} — requires >= 3.10", {}
    return PASS, detail, {"python": f"{v.major}.{v.minor}"}


def probe_platform(ctx: PreflightContext) -> tuple[str, str, dict[str, Any]]:
    return PASS, f"{platform.system()} {platform.release()} ({platform.machine()})", {
        "platform": platform.system().lower(),
        "machine": platform.machine(),
    }


def probe_disk(ctx: PreflightContext) -> tuple[str, str, dict[str, Any]]:
    """Mandatory worst-case disk check (12 §7.3)."""
    required = ctx.required_disk_bytes
    free = shutil.disk_usage(ctx.root).free
    data = {"required": required, "free": free}
    detail = (f"worst-case need {required / (1 << 30):.1f} GiB, "
              f"free {free / (1 << 30):.1f} GiB")
    if free < required:
        return FAIL, detail + " — insufficient", data
    return PASS, detail, data


def probe_atomic_rename(ctx: PreflightContext) -> tuple[str, str, dict[str, Any]]:
    """Actually perform the staging → rename dance on the workspace fs —
    capability proven, never assumed (12 §7.3, §11.1)."""
    probe_dir = ctx.root / ".mlforge_probe"
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        tmp = probe_dir / f"rename_probe_{os.getpid()}.tmp"
        dest = probe_dir / f"rename_probe_{os.getpid()}.ok"
        tmp.write_text("ok", encoding="utf-8")
        os.replace(tmp, dest)
        ok = dest.read_text(encoding="utf-8") == "ok"
        dest.unlink(missing_ok=True)
        if ok:
            return PASS, "atomic rename supported (filesystem write + rename verified)", {}
        return FAIL, "staging → rename did not produce committed content", {}
    except OSError as exc:
        return FAIL, f"atomic rename unavailable: {exc}", {}


def probe_write_permission(ctx: PreflightContext) -> tuple[str, str, dict[str, Any]]:
    try:
        ctx.root.mkdir(parents=True, exist_ok=True)
        p = ctx.root / f".mlforge_perm_{os.getpid()}"
        p.write_text("x", encoding="utf-8")
        p.unlink()
        return PASS, f"write+delete ok under {ctx.root}", {}
    except OSError as exc:
        return FAIL, f"no write permission under {ctx.root}: {exc}", {}


def probe_ram(ctx: PreflightContext) -> tuple[str, str, dict[str, Any]]:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return FAIL, "RAM total unverifiable on this platform (fail-closed)", {}
    detail = f"{total / (1 << 30):.1f} GiB total, need >= {ctx.min_ram_bytes / (1 << 30):.1f} GiB"
    if total < ctx.min_ram_bytes:
        return FAIL, detail, {"total": total}
    return PASS, detail, {"total": total}


def probe_gpu(ctx: PreflightContext) -> tuple[str, str, dict[str, Any]]:
    """GPU detection via `nvidia-smi` (no ML deps). Missing binary while
    gpu_required ⇒ FAIL — unverifiable is a failure, not a warning."""
    if not ctx.gpu_required:
        return WARN, "GPU not required for this operation", {}
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return FAIL, "nvidia-smi not found — GPU unverifiable (fail-closed)", {}
    import subprocess

    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError) as exc:
        return FAIL, f"nvidia-smi failed: {exc}", {}
    if not out:
        return FAIL, "nvidia-smi reported no GPUs", {}
    name, vram_mb, driver = (part.strip() for part in out[0].split(",")[:3])
    return PASS, f"{name}, {vram_mb} MiB VRAM, driver {driver}", {
        "gpu": name, "vram_mb": int(vram_mb), "driver": driver,
    }


def probe_driver(ctx: PreflightContext) -> tuple[str, str, dict[str, Any]]:
    if not ctx.gpu_required:
        return WARN, "driver not required (no GPU for this operation)", {}
    gpu = probe_gpu(ctx)
    if gpu[0] == FAIL:
        return FAIL, f"driver unverifiable: {gpu[1]}", {}
    return PASS, f"driver {gpu[2].get('driver')} present (compatibility vs CUDA runtime verified by gate step 8)", gpu[2]


def probe_secrets(ctx: PreflightContext) -> tuple[str, str, dict[str, Any]]:
    """12 §7.1: secrets present in artifacts → BLOCK."""
    findings = scan_for_secrets(ctx.run_dir)
    if findings:
        return FAIL, "secrets detected in run folder:\n" + "\n".join(findings), {
            "findings": findings,
        }
    return PASS, "no secret patterns found in run folder", {}


#: Default probe spine (identity probes use the gate's provider keys).
HOST_PROBES: tuple[tuple[str, str, Probe], ...] = (
    ("python", "Python runtime", probe_python),
    ("platform", "Operating system", probe_platform),
    ("disk", "Disk space (worst-case)", probe_disk),
    ("atomic_rename", "Atomic rename supported", probe_atomic_rename),
    ("write_permission", "Filesystem write permission", probe_write_permission),
    ("ram", "RAM", probe_ram),
    ("gpu", "GPU capability", probe_gpu),
    ("driver", "Driver compatibility", probe_driver),
    ("secrets", "No secrets in artifacts", probe_secrets),
)

#: Identity checks re-verified during preflight (12 §7.3 lists dataset
#: hashes, code, environment, model, checkpoint) — resolved through the
#: same provider keys as the resume gate.
IDENTITY_PROVIDER_KEYS: tuple[tuple[str, int, str], ...] = (
    ("source_code", 10, "Code artifact present + hash match"),
    ("dataset", 11, "Dataset identity re-verified"),
    ("environment", 12, "Environment image available"),
    ("model", 13, "Model architecture + base weights hash"),
    ("checkpoint", 14, "Checkpoint integrity"),
)


class Preflight:
    """Host + identity checks before any expensive operation (12 §7.3)."""

    def __init__(
        self,
        *,
        identity_providers: Mapping[str, Provider] | None = None,
        probes: Mapping[str, Probe] | None = None,
    ):
        self.identity_providers = dict(identity_providers or {})
        self.probes = dict(probes or {})

    def run(self, ctx: PreflightContext) -> ValidationReport:
        checks: list[Check] = []
        blocked = False

        # Host probes first (cheap, no identity work if the host is wrong).
        for step_no, (key, label, default_probe) in enumerate(HOST_PROBES, start=1):
            probe = ctx.probe_overrides.get(key) or self.probes.get(key) or default_probe
            try:
                verdict, detail, data = probe(ctx)
            except Exception as exc:
                verdict, detail, data = FAIL, f"probe raised: {type(exc).__name__}: {exc}", {}
            checks.append(Check(step_no, key, label, verdict, detail, data))
            if verdict == FAIL:
                blocked = True
                break  # fail-fast: do not run identity work on a broken host

        # Identity re-verification (same provider vocabulary as the gate).
        if not blocked:
            base = len(HOST_PROBES)
            for i, (key, _n, label) in enumerate(IDENTITY_PROVIDER_KEYS):
                provider = self.identity_providers.get(key)
                if provider is None:
                    checks.append(Check(
                        base + i + 1, key, label, FAIL,
                        f"unverifiable: no provider for '{key}' (fail-closed)",
                    ))
                    blocked = True
                    break
                try:
                    result = provider(ctx)
                except PreconditionFailed:
                    raise  # exit-3 preconditions propagate (13 §7)
                except ValidationBlock as exc:
                    checks.append(Check(base + i + 1, key, label, FAIL, exc.message))
                    blocked = True
                    break
                except Exception as exc:
                    checks.append(Check(base + i + 1, key, label, FAIL,
                                        f"check raised: {type(exc).__name__}: {exc}"))
                    blocked = True
                    break
                if result is None:
                    checks.append(Check(base + i + 1, key, label, FAIL,
                                        "unverifiable: provider returned no verdict"))
                    blocked = True
                    break
                checks.append(Check(base + i + 1, key, label,
                                    result.verdict, result.detail, dict(result.data)))
                if result.verdict == FAIL:
                    blocked = True
                    break

        return ValidationReport(
            run_id=ctx.run_id,
            checks=tuple(checks),
            mode=None,
            segment=None,
            title="PREFLIGHT REPORT",
            pass_result="PREFLIGHT PASSED",
            fail_result="PREFLIGHT FAILED",
        )


__all__ = [
    "Preflight",
    "PreflightContext",
    "Probe",
    "HOST_PROBES",
    "IDENTITY_PROVIDER_KEYS",
]
