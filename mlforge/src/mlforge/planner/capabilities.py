"""Capability negotiation — hardware → capabilities (12 §13.1).

    Hardware → Capabilities:
        memory · compute_capability · bf16 · fp16 · fp8 · tensor_cores
        gpu_count · interconnect · cpu · ram

Rules that keep this honest:

  * VRAM sizes are QUERIED from the device (`nvidia-smi`), never looked
    up in a table (12 §21: "VRAM lookup tables → capability negotiation").
  * Precision feature flags are DERIVED from the compute capability with
    fail-closed defaults: an unknown/unparseable arch supports NOTHING
    beyond fp32 (unknown == unsupported — never "probably bf16").
  * `nvidia-smi` absent ⇒ a determinate CPU-only capability set (fp32,
    world 1, PORTABLE — 12 §21: CPU runs the same gate, PORTABLE only).
  * `nvidia-smi` present but failing ⇒ capabilities are UNKNOWN ⇒
    ValidationBlock (fail-closed: we cannot plan what we cannot measure).
  * Conservative minima across GPUs (VRAM = slowest device, flags =
    intersection) — heterogeneous hosts plan for their weakest GPU.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.errors import ValidationBlock
from mlforge.hashing import content_hash

#: Precision names the planner understands. fp32 is always supported
#: (software fallback); everything else must be provable from arch.
PRECISIONS = ("fp32", "fp16", "bf16", "fp8")


@dataclass(frozen=True)
class GPUInfo:
    name: str
    vram_total_mb: int
    compute_capability: str | None  # "8.6" | None (unknown ⇒ fail-closed)

    @property
    def features(self) -> dict[str, bool]:
        return _features_from_arch(self.compute_capability)


def _features_from_arch(compute_capability: str | None) -> dict[str, bool]:
    """Arch → precision features. Unknown arch ⇒ nothing beyond fp32."""
    if not compute_capability or compute_capability.upper().startswith("N/A"):
        return {"fp32": True, "fp16": False, "bf16": False, "fp8": False,
                "tensor_cores": False}
    try:
        parts = compute_capability.strip().split(".")
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return {"fp32": True, "fp16": False, "bf16": False, "fp8": False,
                "tensor_cores": False}
    arch = (major, minor)
    return {
        "fp32": True,
        "tensor_cores": arch >= (7, 0),
        "fp16": arch >= (7, 0),   # fp16 tensor path from Volta
        "bf16": arch >= (8, 0),   # Ampere
        "fp8": arch >= (9, 0),    # Hopper
    }


@dataclass(frozen=True)
class Capabilities:
    """Execution capability profile (12 §6.5 triple, §13.1)."""

    gpu_count: int
    gpus: tuple[GPUInfo, ...]
    ram_bytes: int
    cpu_count: int
    interconnect: str          # "nvlink" | "pcie" | "none"
    source: str                # "nvidia-smi" | "cpu-only"

    @property
    def vram_total_bytes(self) -> int:
        """Per-device budget (min across GPUs — weakest device wins)."""
        if not self.gpus:
            return 0
        return min(g.vram_total_mb for g in self.gpus) * (1 << 20)

    @property
    def compute_capability(self) -> str | None:
        caps = {g.compute_capability for g in self.gpus}
        if len(caps) == 1:
            return next(iter(caps))
        return None  # heterogeneous — treat arch as unknown (fail-closed)

    @property
    def features(self) -> dict[str, bool]:
        if not self.gpus:
            return {"fp32": True, "fp16": False, "bf16": False, "fp8": False,
                    "tensor_cores": False}
        # intersection across devices (weakest GPU decides)
        feats = [g.features for g in self.gpus]
        return {k: all(f[k] for f in feats) for k in feats[0]}

    @property
    def identity(self) -> str:
        """Capability identity — part of every execution segment (12 §12.2)."""
        return content_hash(self.to_dict())

    def supports(self, precision: str) -> bool:
        return bool(self.features.get(precision, False))

    @property
    def summary(self) -> str:
        if not self.gpus:
            return f"CPU ({self.cpu_count} threads, {self.ram_bytes >> 30} GiB RAM)"
        g0 = self.gpus[0]
        arch = f"sm_{g0.compute_capability.replace('.', '')}" if g0.compute_capability else "arch-unknown"
        return (f"{self.gpu_count}× {g0.name} ({self.vram_total_bytes >> 30} GiB, "
                f"{arch}, {self.interconnect})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_count": self.gpu_count,
            "gpus": [
                {"name": g.name, "vram_total_mb": g.vram_total_mb,
                 "compute_capability": g.compute_capability}
                for g in self.gpus
            ],
            "ram_bytes": self.ram_bytes,
            "cpu_count": self.cpu_count,
            "interconnect": self.interconnect,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Capabilities:
        return cls(
            gpu_count=int(d["gpu_count"]),
            gpus=tuple(GPUInfo(str(g["name"]), int(g["vram_total_mb"]),
                                g.get("compute_capability"))
                       for g in d.get("gpus", [])),
            ram_bytes=int(d.get("ram_bytes", 0)),
            cpu_count=int(d.get("cpu_count", 0)),
            interconnect=str(d.get("interconnect", "none")),
            source=str(d.get("source", "unknown")),
        )


def _ram_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0  # unknown RAM never blocks planning (preflight owns RAM floors)


def _cpu_count() -> int:
    return os.cpu_count() or 1


def _query_gpus() -> list[GPUInfo]:
    """Query the device. Present-but-broken ⇒ fail-closed ValidationBlock."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise LookupError("nvidia-smi not found")  # caller → CPU-only set
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValidationBlock(
            f"capability detection failed: nvidia-smi present but query "
            f"error ({exc}) — cannot plan without measured capabilities "
            f"(fix: repair the driver install or remove the broken tool)"
        ) from exc
    gpus: list[GPUInfo] = []
    for line in out:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            raise ValidationBlock(
                "capability detection failed: unparseable nvidia-smi output "
                f"{line!r} (fail-closed: unknown capability is not a capability)"
            )
        try:
            vram = int(float(parts[1]))
        except ValueError as exc:
            raise ValidationBlock(
                f"capability detection failed: unparseable VRAM {parts[1]!r}"
            ) from exc
        cc = parts[2]
        gpus.append(GPUInfo(
            name=parts[0],
            vram_total_mb=vram,
            compute_capability=None if cc.upper().startswith(("N/A", "[N/A")) else cc,
        ))
    if not gpus:
        raise ValidationBlock("capability detection failed: no GPUs reported")
    return gpus


def _detect_interconnect() -> str:
    """Best-effort fabric detection; conservative default = pcie (claiming
    LESS interconnect than present can only make topology checks stricter)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return "none"
    try:
        out = subprocess.run([exe, "nvlink", "-s"], capture_output=True,
                             text=True, timeout=3, check=False).stdout
        if "active" in out:
            return "nvlink"
    except (subprocess.SubprocessError, OSError):
        pass
    return "pcie"


def detect_capabilities() -> Capabilities:
    """Measure this machine (12 §13.1). Never guesses, never fails soft
    on a GPU host: broken detection is a ValidationBlock, not a shrug."""
    gpus: list[GPUInfo]
    source: str
    interconnect: str
    try:
        gpus = _query_gpus()
        source = "nvidia-smi"
        interconnect = _detect_interconnect()
    except LookupError:
        gpus, source, interconnect = [], "cpu-only", "none"
    return Capabilities(
        gpu_count=len(gpus),
        gpus=tuple(gpus),
        ram_bytes=_ram_bytes(),
        cpu_count=_cpu_count(),
        interconnect=interconnect,
        source=source,
    )


__all__ = [
    "PRECISIONS",
    "Capabilities",
    "GPUInfo",
    "detect_capabilities",
]
