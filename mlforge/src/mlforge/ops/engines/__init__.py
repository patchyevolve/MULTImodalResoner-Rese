"""Real execution engines — the evaluate / infer / export seams where
12 §12.4 says the harness is pluggable, wired exactly like
`mlforge.trainers` (the trainer is the model of this design).

Normative: 12_training_system.md §12.4 (pluggable harnesses — a real
engine plugs in by replacing the harness and the code hash follows),
§15.3 (weights alone ≠ model; the contract declares how the model is
called), 13_product_specification §6.7 (EVALUATE runs a harness over the
dataset), §6.8 (INFER: validate → contract check → execute), §6.9
(EXPORT: operator validation, real binary, numerical round-trip).

Design (fail-closed, mirroring `mlforge.trainers`):
  * default = REAL engines; the deterministic scaffolds live behind the
    same opt-in as `ScaffoldTrainer` — `MLFORGE_HARNESS=1` (system-test
    harness only). Without the harness, a model that cannot actually run
    here gets an honest `PreconditionFailed` naming why, never a fake
    number or a fake binary;
  * model → engine resolution is a REGISTRY (fail-closed): an unknown
    model or a missing framework reports exactly which engines exist and
    why any of them cannot run;
  * engine modules are import-safe without their frameworks and are
    imported ONLY when selected (stdlib core stays dependency-free).

Engine module contract (each module in `_ENGINE_MODULES` exposes):

    FAMILY            str   — harness label recorded in artifacts
    MODEL_NAMES       frozenset[str]
    dependency_error() -> str | None
    input_types() -> frozenset[str]        # contract input types it runs
    metric_names() -> tuple[str, ...]      # protocol metric_definitions
    contract(entry, fmt, *, semantic, precision) -> dict  # model_spec doc
    supports_format(fmt) -> str | None     # None = convertible here
    execute(*, entry, contract_doc, checked, weights) -> dict  # result
    compute_metrics(*, entry, weights, dataset, protocol, semantic) -> dict
    export_bytes(*, entry, weights, fmt) -> tuple[bytes, dict]

Weights resolution (`load_weights`): run-origin models serve the `model`
component of the run's newest VALID committed checkpoint (§11.2 recovery
selection — never "the newest directory"); import-origin models serve the
weights file the registry stores at import (12 §15.3 package file
family). No weights ⇒ `PreconditionFailed`, never a random-init model.

Adding an engine:
  1. implement the contract above in a new module (import-safe,
     framework imports inside functions)
  2. add it to `_ENGINE_MODULES`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.runtime.trainer import HARNESS_ENV

#: Real engine modules in registry order (each import-safe without its
#: framework; `MODEL_NAMES` membership is the lookup key).
_ENGINE_MODULES: tuple[str, ...] = (
    "calibrator",
    "detection",
    "ranker",
    "reid",
    "text_lm",
)


def harness_active() -> bool:
    """System-test harness opt-in — the SAME gate as `ScaffoldTrainer`
    (12 §12.4): default is the real engine, `MLFORGE_HARNESS=1` selects
    the deterministic scaffold for system tests."""
    return os.environ.get(HARNESS_ENV, "").strip().lower() in ("1", "true", "yes")


def _module(name: str):
    import importlib

    return importlib.import_module(f"mlforge.ops.engines.{name}")


def _all_modules() -> tuple[list[Any], dict[str, str]]:
    """Import every registry module — one broken module must not take
    the others down (its models then report honestly as unavailable)."""
    mods: list[Any] = []
    broken: dict[str, str] = {}
    for name in _ENGINE_MODULES:
        try:
            mods.append(_module(name))
        except Exception as exc:  # pragma: no cover — broken installation
            broken[name] = f"{type(exc).__name__}: {exc}"
    return mods, broken


def _find_module(model: str) -> Any | None:
    mods, _broken = _all_modules()
    for mod in mods:
        if model in mod.MODEL_NAMES:
            return mod
    return None


def registered_models() -> list[str]:
    """Every model with an integrated engine, runnable or not."""
    mods, _broken = _all_modules()
    out: set[str] = set()
    for mod in mods:
        out.update(mod.MODEL_NAMES)
    return sorted(out)


def availability_error(model: str | None) -> str | None:
    """None = the engine can run this model right now; otherwise the
    honest refusal detail (mirrors trainers.availability_error)."""
    if not model:
        return "no model context — the model entry names no family"
    mods, broken = _all_modules()
    for mod in mods:
        if model in mod.MODEL_NAMES:
            return mod.dependency_error()
    known: set[str] = set()
    unavailable: list[str] = []
    for mod in mods:
        err = mod.dependency_error()
        if err is None:
            known.update(mod.MODEL_NAMES)
        elif mod.MODEL_NAMES:
            unavailable.append(f"{', '.join(sorted(mod.MODEL_NAMES))} ({err})")
    detail = (
        f"model {model!r} has no integrated engine — runnable here: "
        + (", ".join(sorted(known)) or "(none registered)")
    )
    if unavailable:
        detail += "; registered but unavailable: " + "; ".join(unavailable)
    if broken:
        detail += "; engine module failed to import: " + "; ".join(
            f"{name}: {why}" for name, why in sorted(broken.items())
        )
    return detail


def build_engine(model: str | None) -> Any:
    """Resolve the engine module for `model` (fail-closed first)."""
    err = availability_error(model)
    if err:
        raise PreconditionFailed(
            f"no real engine for model {model!r} — {err}",
            hint="MLFORGE_HARNESS=1 selects the deterministic scaffold "
                 "harness (system tests only — 12 §12.4); the real "
                 "engines are listed above",
        )
    mod = _find_module(str(model))
    assert mod is not None  # availability_error just returned None
    return mod


def load_weights(root: Path, entry: Mapping[str, Any]) -> bytes:
    """The model's REAL weights bytes (fail-closed).

    run-origin  → `model` component of the run's newest VALID checkpoint
                  (CheckpointStore verification runs first — a corrupt
                  generation is never served), cross-checked against the
                  registry's `artifact_hash` (the COMMIT marker, 12 §15.3);
    import-origin → the weights file the registry stored at import.
    """
    from mlforge.ops.importing import WEIGHTS_FILE

    name = f"{entry.get('name')}:{entry.get('version')}"
    run_id = entry.get("run_id")
    if run_id:
        from mlforge.runtime.checkpoints import CheckpointStore

        store = CheckpointStore(Path(root) / "runs" / str(run_id))
        selection = store.newest_valid()
        if selection.selected is None:
            raise PreconditionFailed(
                f"model {name} has no valid committed checkpoint weights",
                hint="weights live in the run's committed checkpoints "
                     "(12 §11) — train this run to completion first; a "
                     "model without weights is never run from random "
                     "initialization",
            )
        marker = selection.selected.path / "COMMIT"
        expected = entry.get("artifact_hash")
        if marker.is_file() and expected:
            got = marker.read_text(encoding="utf-8").strip()
            if got != str(expected):
                raise PreconditionFailed(
                    f"model {name}: registry artifact_hash does not match "
                    "the run's newest committed checkpoint",
                    hint="the registry and the checkpoint store disagree "
                         "(12 §15.3) — re-publish from the run or "
                         "re-register the model; refusing to serve "
                         "unverifiable weights",
                )
        return store.load_payload(selection.selected.ordinal, "model")
    p = Path(root) / "models" / str(entry.get("model_id")) / WEIGHTS_FILE
    if p.is_file():
        return p.read_bytes()
    raise PreconditionFailed(
        f"model {name} has no stored weights file in the registry",
        hint="imported models keep their hashed weights under "
             f"models/{entry.get('model_id')}/{WEIGHTS_FILE} (12 §15.3) — "
             "re-import the package with `mlforge model import <PATH>`",
    )


def engine_input_types(model: str | None) -> frozenset[str]:
    """Contract input types the engine for `model` actually executes."""
    return frozenset(build_engine(model).input_types())


def unsupported_engine_result(action: str, model: str | None,
                              detail: str) -> PreconditionFailed:
    """One refusal shape for the three seams (evaluate/infer/export)."""
    return PreconditionFailed(
        f"cannot {action} model {model!r} — {detail}",
        hint="MLFORGE_HARNESS=1 selects the deterministic scaffold "
             "harness (system tests only — 12 §12.4)",
    )


__all__ = [
    "_ENGINE_MODULES",
    "availability_error",
    "build_engine",
    "engine_input_types",
    "harness_active",
    "load_weights",
    "registered_models",
    "unsupported_engine_result",
]
