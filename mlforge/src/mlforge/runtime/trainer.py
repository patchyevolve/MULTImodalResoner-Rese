"""Trainer protocol + system scaffold — the pluggable training step.

Normative context: 12_training_system.md §12.4 (the worker OWNS the
process; what it *trains* is injected), §11.4 (checkpoint contents are
the trainer's state — model/optimizer/RNG/sampler/...).

The runtime is framework-agnostic on purpose: MLForge orchestrates state,
identity, and recovery; the actual gradient step belongs to a Trainer
implementation. The default `ScaffoldTrainer` is a deterministic,
dependency-free stand-in that exercises the FULL runtime contract
(steps, epochs, checkpoint payloads, completion) — it explicitly does
not claim to train a model. Real trainers (RF-DETR, OSNet, ...) plug in
at integration without touching MLForge orchestration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS


@dataclass(frozen=True)
class TrainState:
    """What the worker knows at the top of each step (§11.4 identity)."""

    global_step: int
    epoch: int
    #: newest-valid resume point the worker started from (§11.2), if any
    resume_from: str | None = None


@dataclass(frozen=True)
class StepResult:
    global_step: int
    epoch: int
    loss: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    done: bool = False


@runtime_checkable
class Trainer(Protocol):
    """One training unit + checkpoint payload. Implementations must be
    deterministic given (TrainState) so recovery is reproducible."""

    def step(self, state: TrainState) -> StepResult:
        ...

    def checkpoint_payload(self, state: TrainState) -> dict[str, bytes]:
        """Framework-specific state for §11.4 components. Keys should be
        covered by the manifest's component set — an incomplete payload is
        caught by recovery (missing components ⇒ not resumable)."""
        ...


class ScaffoldTrainer:
    """Deterministic stand-in trainer (system harness, not a model).

    loss decays as 1/(1+step) so metrics files are plausible; completion
    after `max_steps`; payload covers every §11.4 component name with a
    JSON identity of (component, step, epoch) — honest scaffold content,
    replaced wholesale by a real trainer."""

    def __init__(self, *, max_steps: int = 20, steps_per_epoch: int = 10,
                 start_step: int = 0, start_epoch: int = 0):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1")
        self.max_steps = max_steps
        self.steps_per_epoch = steps_per_epoch
        self.start_step = start_step
        self.start_epoch = start_epoch

    def step(self, state: TrainState) -> StepResult:
        nxt = state.global_step + 1
        epoch = nxt // self.steps_per_epoch
        return StepResult(
            global_step=nxt,
            epoch=epoch,
            loss=round(1.0 / (1.0 + nxt), 6),
            metrics={"lr": round(1e-4 * (1.0 - nxt / self.max_steps), 8)},
            done=nxt >= self.max_steps,
        )

    def checkpoint_payload(self, state: TrainState) -> dict[str, bytes]:
        return {
            name: json.dumps(
                {"component": name, "global_step": state.global_step,
                 "epoch": state.epoch},
                sort_keys=True,
            ).encode("utf-8")
            for name in REQUIRED_COMPONENTS
        }


__all__ = ["TrainState", "StepResult", "Trainer", "ScaffoldTrainer"]
