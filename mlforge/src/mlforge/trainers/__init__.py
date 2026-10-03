"""Real trainers — the learning loops injected where 12 §12.4 says the
worker owns the process but what it *trains* is pluggable.

Core MLForge stays stdlib-only: a trainer module may depend on its
framework (torch / rfdetr) and is imported ONLY when selected. Machines
without the framework run everything except actual training — with an
honest error naming the install command, never a fake trainer (the old
silent `ScaffoldTrainer` default is gone; see mlforge.runtime.trainer).

Model → trainer resolution is a REGISTRY (fail-closed): an unknown model
or a missing framework raises `PreconditionFailed` with the concrete
detail — MLForge never substitutes a scaffold to make a run "work".

Adding a trainable model:
  1. define its architecture/config + MODEL_REGISTRY in a trainer module
     (text_lm = byte-level LM, reid = OSNet, rfdetr = RF-DETR family)
  2. expose `dependency_error()` in that module (None = runnable here)
  3. add the module to `_TRAINER_MODULES` below

Trainer construction channels (all trainers accept the same keywords):
  payloads      — newest-valid OWN checkpoint (resume: full state)
  init_weights  — parent run's `model` component (fine-tune: weights only,
                  fresh optimizer — 13 §6.4 "init from weights")
  run_dir       — this run's directory (framework scratch/output dirs)
  on_progress   — liveness callback (worker heartbeat) for steps that
                  outlive the heartbeat interval (12 §23.2: 30s/120s)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mlforge.errors import PreconditionFailed, ValidationBlock

#: Real trainer modules in registry order. Each module is import-safe
#: without its framework and exposes MODEL_REGISTRY + dependency_error().
_TRAINER_MODULES: tuple[str, ...] = (
    "text_lm", "reid", "rfdetr", "calibrator", "gbdt",
)


def _module(name: str):
    import importlib

    return importlib.import_module(f"mlforge.trainers.{name}")


def _all_modules() -> tuple[list[Any], dict[str, str]]:
    """Import every registry module — one broken module must not take the
    others down (its models then report honestly as unavailable)."""
    mods: list[Any] = []
    broken: dict[str, str] = {}
    for name in _TRAINER_MODULES:
        try:
            mods.append(_module(name))
        except Exception as exc:  # pragma: no cover — broken installation
            broken[name] = f"{type(exc).__name__}: {exc}"
    return mods, broken


def _find_module(model: str) -> Any | None:
    """Registry membership lookup — imports modules (framework-safe)."""
    mods, _broken = _all_modules()
    for mod in mods:
        if model in mod.MODEL_REGISTRY:
            return mod
    return None


def available_models() -> list[str]:
    """Models whose trainer can actually run in THIS environment."""
    mods, _broken = _all_modules()
    out: set[str] = set()
    for mod in mods:
        if mod.dependency_error() is None:
            out.update(mod.MODEL_REGISTRY)
    return sorted(out)


def registered_models() -> list[str]:
    """Every model with an integrated trainer, runnable or not — for
    honest messages ("registered but unavailable: ...")."""
    mods, _broken = _all_modules()
    out: set[str] = set()
    for mod in mods:
        out.update(mod.MODEL_REGISTRY)
    return sorted(out)


def availability_error(model: str | None) -> str | None:
    """None = trainable right now; otherwise the honest refusal detail."""
    if not model:
        return "no model context — run_spec.model is required to select a trainer"
    mods, broken = _all_modules()
    for mod in mods:
        if model in mod.MODEL_REGISTRY:
            # Known model — the module's own dependency verdict is the message.
            return mod.dependency_error()
    runnable: set[str] = set()
    unrunnable: list[str] = []
    for mod in mods:
        err = mod.dependency_error()
        if err is None:
            runnable.update(mod.MODEL_REGISTRY)
        elif mod.MODEL_REGISTRY:
            names = ", ".join(sorted(mod.MODEL_REGISTRY))
            unrunnable.append(f"{names} ({err})")
    known = ", ".join(sorted(runnable)) or "(none registered)"
    detail = (
        f"model {model!r} has no integrated trainer — trainable here: {known}"
    )
    if unrunnable:
        detail += "; registered but unavailable: " + "; ".join(unrunnable)
    if broken:
        detail += "; trainer module failed to import: " + "; ".join(
            f"{name}: {why}" for name, why in sorted(broken.items())
        )
    return detail


def build_trainer(
    model: str,
    *,
    runtime: Mapping[str, Any],
    semantic: Mapping[str, Any],
    train_datasets: Sequence[str],
    root: Path,
    payloads: Mapping[str, bytes] | None = None,
    start_step: int = 0,
    start_epoch: int = 0,
    plan: Any = None,
    run_dir: Path | None = None,
    on_progress: Callable[..., Any] | None = None,
    init_weights: bytes | None = None,
):
    """Construct the real trainer for `model` (fail-closed first)."""
    err = availability_error(model)
    if err:
        raise PreconditionFailed(
            f"no real trainer integrated for model {model!r} — {err}",
            hint='system-test harness only: "runtime": '
            '{"trainer": "harness-scaffold"} or MLFORGE_HARNESS=1 — '
            "the harness is a fake loss curve, never a model",
        )
    if payloads and init_weights:
        raise ValidationBlock(
            "resume payloads and fine-tune init weights are mutually "
            "exclusive — a run resumes from ITS OWN checkpoint or "
            "initializes from the parent's weights, never both",
        )
    mod = _find_module(model)
    assert mod is not None  # availability_error just returned None
    return mod.TRAINER_CLASS(
        model_name=model,
        semantic=semantic,
        runtime=runtime,
        train_datasets=tuple(train_datasets),
        root=Path(root),
        payloads=dict(payloads) if payloads else None,
        start_step=start_step,
        start_epoch=start_epoch,
        plan=plan,
        run_dir=Path(run_dir) if run_dir is not None else None,
        on_progress=on_progress,
        init_weights=init_weights,
    )


__all__ = [
    "availability_error",
    "available_models",
    "build_trainer",
    "registered_models",
]
