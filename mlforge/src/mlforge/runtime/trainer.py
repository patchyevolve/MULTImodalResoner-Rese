"""Trainer protocol + the HARNESS trainer — never a silent default.

Normative context: 12_training_system.md §12.4 (the worker OWNS the
process; what it *trains* is injected), §11.4 (checkpoint contents are
the trainer's state — model/optimizer/RNG/sampler/...).

Resolution order (`resolve_trainer` — the ONLY way production code
obtains a trainer):

  1. explicit `runtime.trainer` id  → "harness-scaffold" only (system
     tests); any other id ⇒ unknown-trainer refusal
  2. harness opt-in via `MLFORGE_HARNESS=1` (system tests)
  3. REAL trainer for `run_spec.model` via the mlforge.trainers
     registry — e.g. `reasoner_s` → the torch byte-level LM loop
     (needs torch installed; missing ⇒ honest refusal naming the
     install command)
  4. anything else ⇒ `PreconditionFailed`: unknown model, no model
     context, or missing framework. MLForge never falls back to a
     scaffold to make a run "work" — the old silent
     `ScaffoldTrainer` default is gone and stays gone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from mlforge.errors import PreconditionFailed
from mlforge.runtime.checkpoints import REQUIRED_COMPONENTS

#: The only accepted non-default trainer id (explicit opt-in, auditable).
HARNESS_TRAINER = "harness-scaffold"
#: Session-wide opt-in for system tests (never set in production).
HARNESS_ENV = "MLFORGE_HARNESS"

#: Refusal hint — the harness is named, never silently reached.
_HARNESS_HINT = (
    f'system-test harness only: "runtime": {{\"trainer\": '
    f'"{HARNESS_TRAINER}"}} or {HARNESS_ENV}=1 — the harness is a fake '
    "loss curve, never a model"
)


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
    model: str | None = None,
    semantic: Mapping[str, Any] | None = None,
    train_datasets: Sequence[str] = (),
    root: str | Path | None = None,
    payloads: Mapping[str, bytes] | None = None,
    plan: Any = None,
) -> Trainer:
    """The ONLY way production code obtains a trainer (fail-closed).

    Order: explicit `trainer:` id → harness opt-in → the REAL trainer
    registered for `model` → refuse. No silent scaffold, ever."""
    choice = str((runtime or {}).get("trainer", "")).strip().lower()
    if choice and choice != HARNESS_TRAINER:
        raise PreconditionFailed(
            f"unknown trainer {choice!r}",
            hint=f'the only accepted value today is "{HARNESS_TRAINER}" '
                 "(system-test harness — fake loss, not a model); real "
                 "trainers resolve by run_spec.model (mlforge.trainers)",
        )
    if harness_allowed(runtime):
        cfg = runtime or {}
        return ScaffoldTrainer(
            max_steps=int(cfg.get("max_steps", 20)),
            steps_per_epoch=int(cfg.get("steps_per_epoch", 10)),
            start_step=start_step,
            start_epoch=start_epoch,
        )
    # Real path — model-selected, framework-checked, built here so a
    # refusal lands BEFORE READY → RUNNING (the run stays untouched).
    from mlforge import trainers as _trainers

    if not model:
        raise PreconditionFailed(
            "no real trainer integrated — no model context "
            "(run_spec.model missing) to select a trainer",
            hint=_HARNESS_HINT,
        )
    err = _trainers.availability_error(model)
    if err:
        raise PreconditionFailed(
            f"no real trainer integrated for model {model!r} — {err}",
            hint=_HARNESS_HINT,
        )
    return _trainers.build_trainer(
        model,
        runtime=runtime or {},
        semantic=semantic or {},
        train_datasets=tuple(train_datasets),
        root=Path(root) if root is not None else Path("."),
        payloads=payloads,
        start_step=start_step,
        start_epoch=start_epoch,
        plan=plan,
    )


def require_trainable(
    runtime: Mapping[str, Any] | None = None,
    *,
    model: str | None = None,
) -> None:
    """Cheap preflight: raise unless a trainer CAN run for this model.

    Checks availability only (no corpus load, no model build) — the
    worker still constructs the real trainer before READY → RUNNING."""
    choice = str((runtime or {}).get("trainer", "")).strip().lower()
    if choice and choice != HARNESS_TRAINER:
        raise PreconditionFailed(
            f"unknown trainer {choice!r}",
            hint=f'the only accepted value today is "{HARNESS_TRAINER}" '
                 "(system-test harness — fake loss, not a model)",
        )
    if harness_allowed(runtime):
        return
    from mlforge import trainers as _trainers

    if not model:
        raise PreconditionFailed(
            "no real trainer integrated — no model context "
            "(run_spec.model missing) to select a trainer",
            hint=_HARNESS_HINT,
        )
    err = _trainers.availability_error(model)
    if err:
        raise PreconditionFailed(
            f"no real trainer integrated for model {model!r} — {err}",
            hint=_HARNESS_HINT,
        )


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
