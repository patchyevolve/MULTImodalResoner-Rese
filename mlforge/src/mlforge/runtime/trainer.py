"""Trainer protocol + the HARNESS trainer — never a silent default.

Normative context: 12_training_system.md §12.4 (the worker OWNS the
process; what it *trains* is injected), §11.4 (checkpoint contents are
the trainer's state — model/optimizer/RNG/sampler/...).

CORRECTION (was shipped silently and mischecked as complete):
`ScaffoldTrainer` is a **test harness only** — a deterministic fake
loss curve and JSON stand-in payloads. It is NOT a deliverable, it does
NOT train anything, and production refuses to use it:

  * `state/runtime.json`: {"trainer": "harness-scaffold"}  — per-run
  * env `MLFORGE_HARNESS=1`                                — system tests

Anything else fails closed with `PreconditionFailed`: no real trainer
is integrated yet (the torch learning loop is not built) and MLForge
will never fake training. When a real trainer lands it becomes the
default here — no opt-in key required.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from mlforge.errors import PreconditionFailed
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS

#: The only accepted non-default trainer id (explicit opt-in, auditable).
HARNESS_TRAINER = "harness-scaffold"
#: Session-wide opt-in for system tests (never set in production).
HARNESS_ENV = "MLFORGE_HARNESS"


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
    """TEST HARNESS — deterministic fake trainer, never a deliverable.

    Emits `loss = 1/(1+step)` and JSON (component, step, epoch) payloads
    so the runtime contract (steps, epochs, checkpoints, completion) can
    be exercised without a model. Explicitly NOT training: production
    code reaches it only through `resolve_trainer`'s opt-in paths
    (`runtime.trainer=harness-scaffold` / `MLFORGE_HARNESS=1`)."""

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


def harness_allowed(runtime: Mapping[str, Any] | None = None) -> bool:
    """True only on an EXPLICIT harness opt-in (per-run key or test env)."""
    choice = str((runtime or {}).get("trainer", "")).strip().lower()
    if choice == HARNESS_TRAINER:
        return True
    if choice:
        return False  # an explicit unknown id never falls back to env
    return os.environ.get(HARNESS_ENV, "").strip().lower() in ("1", "true", "yes")


def resolve_trainer(
    runtime: Mapping[str, Any] | None = None,
    *,
    start_step: int = 0,
    start_epoch: int = 0,
) -> Trainer:
    """The ONLY way production code obtains a trainer (fail-closed).

    Order: explicit `trainer:` id → harness opt-in → refuse. No silent
    scaffold, ever — a run that cannot honestly train must not start."""
    choice = str((runtime or {}).get("trainer", "")).strip().lower()
    if choice and choice != HARNESS_TRAINER:
        raise PreconditionFailed(
            f"unknown trainer {choice!r}",
            hint=f'the only accepted value today is "{HARNESS_TRAINER}" '
                 "(system-test harness — fake loss, not a model); real "
                 "trainers register here when integrated",
        )
    if harness_allowed(runtime):
        cfg = runtime or {}
        return ScaffoldTrainer(
            max_steps=int(cfg.get("max_steps", 20)),
            steps_per_epoch=int(cfg.get("steps_per_epoch", 10)),
            start_step=start_step,
            start_epoch=start_epoch,
        )
    raise PreconditionFailed(
        "no real trainer integrated — MLForge's learning loop (real "
        "gradients/loss) is not built yet; refusing to start a run that "
        "cannot train",
        hint=f'set "runtime": {{"trainer": "{HARNESS_TRAINER}"}} in the '
             "train config (per-run) or MLFORGE_HARNESS=1 (system tests) "
             "ONLY — the harness is a fake loss curve, never a model",
    )


def require_trainable(runtime: Mapping[str, Any] | None = None) -> None:
    """Raise unless some trainer (real, or opted-in harness) can run."""
    resolve_trainer(runtime)


__all__ = [
    "HARNESS_ENV",
    "HARNESS_TRAINER",
    "StepResult",
    "TrainState",
    "Trainer",
    "ScaffoldTrainer",
    "harness_allowed",
    "require_trainable",
    "resolve_trainer",
]
