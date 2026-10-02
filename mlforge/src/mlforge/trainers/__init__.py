"""Real trainers — the learning loops injected where 12 §12.4 says the
worker owns the process but what it *trains* is pluggable.

Core MLForge stays stdlib-only: a trainer module may depend on its
framework (torch here) and is imported ONLY when selected. Machines
without the framework run everything except actual training — with an
honest error naming the install command, never a fake trainer (the old
silent `ScaffoldTrainer` default is gone; see mlforge.runtime.trainer).

Model → trainer resolution is a REGISTRY (fail-closed): an unknown model
or a missing framework raises `PreconditionFailed` with the concrete
detail — MLForge never substitutes a scaffold to make a run "work".

Adding a trainable model:
  1. define its architecture/config in a trainer module (see text_lm)
  2. add it to that module's MODEL_REGISTRY
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from mlforge.errors import PreconditionFailed, ValidationBlock

#: Real trainers, by the framework they need.
_TorchTextTrainer: Any = None  # imported lazily (torch is optional)


def _text_lm():
    from mlforge.trainers import text_lm
    return text_lm


def available_models() -> list[str]:
    """Models whose trainer can actually run in THIS environment."""
    lm = _text_lm()
    if not lm.torch_available():
        return []
    return sorted(lm.MODEL_REGISTRY)


def availability_error(model: str | None) -> str | None:
    """None = trainable right now; otherwise the honest refusal detail."""
    if not model:
        return "no model context — run_spec.model is required to select a trainer"
    try:
        lm = _text_lm()
    except ImportError as exc:  # pragma: no cover — broken installation
        return f"trainer module unavailable ({exc})"
    if not lm.torch_available():
        return (
            "torch is not installed — the real learning loop needs it; "
            "pip install torch (CPU wheel: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu)"
        )
    if model in lm.MODEL_REGISTRY:
        return None
    known = ", ".join(sorted(lm.MODEL_REGISTRY)) or "(none registered)"
    return (
        f"model {model!r} has no integrated trainer — trainable here: {known}; "
        "RF-DETR/object-detection training is NOT integrated (it needs a GPU "
        "runtime) and MLForge never fakes it"
    )


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
    from mlforge.trainers.text_lm import TorchTextTrainer

    return TorchTextTrainer(
        model_name=model,
        semantic=semantic,
        runtime=runtime,
        train_datasets=tuple(train_datasets),
        root=Path(root),
        payloads=dict(payloads) if payloads else None,
        start_step=start_step,
        start_epoch=start_epoch,
        plan=plan,
    )


__all__ = [
    "availability_error",
    "available_models",
    "build_trainer",
]
