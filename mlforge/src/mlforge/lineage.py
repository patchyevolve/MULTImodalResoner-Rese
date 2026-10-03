"""Lineage DAG — parent edges for runs and models.

Normative: 12_training_system.md §5 (`runs/<id>/lineage.json`), §4
(semantic change = NEW run with parent set, never mutate), §15.2
(lineage DAG), §13.3 (execution → semantic mutation forbidden);
13 §5.5 (RETRAIN/FINETUNE create new runs), §6.3/§6.4 (delta display).

    train      parent null        — a fresh experiment (13 §6.1)
    fork       parent_run set     — semantic change of an existing run
    retrain    parent_run set     — same config, fresh init (13 §6.3)
    finetune   parent_model +
               parent_checkpoint  — init from weights (13 §6.4)

Hard rules carried by this module:

  * `lineage.json` is written ONCE at run creation (immutable edges —
    "Parent always immutable. No accidental mutation." 12 §15.2);
  * deriving a child spec never mutates the parent's run_spec (the
    parent is read-only input — frozen identity, write-once file);
  * `--set` overrides accept ONLY semantic keys (12 §13.3: execution
    knobs live in runtime config, not here) — unknown key ⇒ BLOCK,
    never a silent typo'd identity;
  * a fork whose derived identity equals the parent is BLOCKed —
    semantic change is the point (resume continues; retrain repeats).

`NewRunPlan` is the display-then-create unit: the CLI SHOWs the delta
(13 §6.3/§6.4) BEFORE anything exists, then `create_from_plan` in the
workflow materializes it exactly once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from mlforge.errors import NotFound, ValidationBlock
from mlforge.ids import new_run_id
from mlforge.run_spec import REQUIRED_SEMANTIC_FIELDS, RunSpec

LINEAGE_SCHEMA = 1
LINEAGE_FILENAME = "lineage.json"

#: 13 §6.4 strategy menu (resolved into run_spec — user never edits
#: checkpoint internals).
FINETUNE_STRATEGIES = (
    "full", "freeze_backbone", "freeze_encoder", "lora", "adapter", "custom",
)

#: `--set` accepts semantic identity keys only (12 §13.3). The two spec
#: aliases (12 §14.2 shows `--set lr=2e-4`) map to the canonical names.
_SEMANTIC_ALIASES = {
    "lr": "learning_rate",
    "batch": "global_batch",
    "precision": "precision_policy",
}
_ALLOWED_OVERRIDE_KEYS = frozenset(REQUIRED_SEMANTIC_FIELDS) | frozenset(
    _SEMANTIC_ALIASES
) | {"finetune_strategy", "pretrained", "windows_per_epoch"}

#: Sentinel: "leave this field as the parent's" (None is a real value —
#: val_dataset=None means "no validation set").
_KEEP = object()


# ---------------------------------------------------------------------------
# lineage.json (12 §5)
# ---------------------------------------------------------------------------


def default_lineage(origin: str = "train") -> dict[str, Any]:
    return {
        "schema_version": LINEAGE_SCHEMA,
        "origin": origin,
        "parent_run": None,
        "parent_run_spec_hash": None,
        "parent_model": None,
        "parent_checkpoint": None,
    }


def build_lineage(
    origin: str,
    *,
    parent_run: str | None = None,
    parent_run_spec_hash: str | None = None,
    parent_model: str | None = None,
    parent_checkpoint: str | None = None,
) -> dict[str, Any]:
    lin = default_lineage(origin)
    lin.update(
        parent_run=parent_run,
        parent_run_spec_hash=parent_run_spec_hash,
        parent_model=parent_model,
        parent_checkpoint=parent_checkpoint,
    )
    return lin


def write_lineage(run_dir: Path, lineage: Mapping[str, Any]) -> None:
    """Write-once: edges are immutable once recorded (12 §15.2).
    Partial payloads are completed with the default (null) fields so the
    file on disk always carries the full edge set."""
    complete = default_lineage()
    complete.update({k: lineage.get(k, complete.get(k)) for k in complete})
    p = run_dir / LINEAGE_FILENAME
    if p.exists():
        current = read_lineage(run_dir)
        if current != complete:
            raise ValidationBlock(
                f"lineage at {p} already exists and differs — "
                "lineage edges are immutable",
                hint="a child run gets ITS own lineage.json; the parent's "
                     "is never rewritten (12 §15.2)",
            )
        return  # identical rewrite is a no-op (idempotent creation)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(complete, indent=2, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(p)  # atomic (12 §11.1)


def read_lineage(run_dir: Path) -> dict[str, Any]:
    """Missing file ⇒ a run from before the lineage build (train/nulls) —
    a corruption (bad JSON) is a BLOCK, never a shrug."""
    p = run_dir / LINEAGE_FILENAME
    if not p.is_file():
        return default_lineage()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationBlock(f"lineage unreadable: {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationBlock(f"lineage must be a JSON object: {p}")
    merged = default_lineage()
    merged.update({k: data.get(k, merged.get(k)) for k in merged})
    return merged


# ---------------------------------------------------------------------------
# overrides + derivation (fork/retrain/finetune)
# ---------------------------------------------------------------------------


def parse_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    """`["lr=2e-4", "epochs=10"]` → `{"learning_rate": 0.0002, ...}`.

    Values are JSON when parseable (numbers, booleans, arrays), raw
    strings otherwise. Unknown/mistyped keys BLOCK — a typo must never
    silently create a different experiment identity."""
    out: dict[str, Any] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValidationBlock(
                f"override {raw!r} must be KEY=VALUE",
                hint=f"allowed keys: {', '.join(sorted(_ALLOWED_OVERRIDE_KEYS))}",
            )
        key, _, value = raw.partition("=")
        key = key.strip()
        canon = _SEMANTIC_ALIASES.get(key, key)
        if canon not in _ALLOWED_OVERRIDE_KEYS:
            raise ValidationBlock(
                f"unknown semantic key {key!r} in override",
                hint=f"semantic identity keys only (12 §13.3): "
                     f"{', '.join(sorted(_ALLOWED_OVERRIDE_KEYS))} — "
                     f"execution knobs belong in runtime config",
            )
        try:
            out[canon] = json.loads(value)
        except json.JSONDecodeError:
            out[canon] = value  # plain string (e.g. precision_policy=bf16)
    return out


def canonicalize_overrides(semantic: Mapping[str, Any]) -> dict[str, Any]:
    """Mapping form of `parse_overrides` (used by `--config`'s semantic
    block and workflow `Mapping` overrides): aliases normalized, unknown
    keys BLOCKed — a typo'd `learnign_rate` must never become part of an
    experiment identity (fail-closed, 12 §4/§13.3)."""
    out: dict[str, Any] = {}
    for key, value in semantic.items():
        canon = _SEMANTIC_ALIASES.get(key, key)
        if canon not in _ALLOWED_OVERRIDE_KEYS:
            raise ValidationBlock(
                f"unknown semantic key {key!r} in override",
                hint=f"semantic identity keys only (12 §13.3): "
                     f"{', '.join(sorted(_ALLOWED_OVERRIDE_KEYS))} — "
                     f"execution knobs belong in runtime config",
            )
        out[canon] = value
    return out


def derive_spec(
    parent: RunSpec,
    *,
    semantic: Mapping[str, Any] | None = None,
    model: str | None = None,
    train_datasets: Iterable[str] | None = None,
    val_dataset: Any = _KEEP,
) -> RunSpec:
    """A NEW RunSpec from a parent + explicit changes (parent untouched).

    `semantic` here is OVERRIDES only — the parent's own keys are never
    re-validated (it may carry optional identity fields beyond the
    override allowlist), but every incoming key must be allowlisted."""
    merged = dict(parent.semantic)
    merged.update(canonicalize_overrides(semantic or {}))
    return RunSpec(
        model=model if model is not None else parent.model,
        train_datasets=tuple(train_datasets)
        if train_datasets is not None else parent.train_datasets,
        semantic=merged,
        val_dataset=parent.val_dataset if val_dataset is _KEEP else val_dataset,
    )


def spec_delta(old: RunSpec, new: RunSpec) -> dict[str, Any]:
    """What CHANGED between two specs (13 §6.3 SHOW DELTA).

    Flat keys: `model` / `train_datasets` / `val_dataset` for spec-level
    fields, canonical semantic names for semantic fields. Only actual
    differences appear — an empty delta == identical experiment."""
    delta: dict[str, Any] = {}
    if old.model != new.model:
        delta["model"] = {"from": old.model, "to": new.model}
    if old.train_datasets != new.train_datasets:
        delta["train_datasets"] = {
            "from": list(old.train_datasets), "to": list(new.train_datasets),
        }
    if old.val_dataset != new.val_dataset:
        delta["val_dataset"] = {"from": old.val_dataset, "to": new.val_dataset}
    for key in sorted(set(old.semantic) | set(new.semantic)):
        a, b = old.semantic.get(key), new.semantic.get(key)
        if a != b:
            delta[key] = {"from": a, "to": b}
    return delta


# ---------------------------------------------------------------------------
# model references (13 §4.3: model://rf_detr_s:v2 or rf_detr_s:v2)
# ---------------------------------------------------------------------------


def parse_model_ref(ref: str) -> tuple[str, str | None]:
    """→ (name, version | None). Bare name ⇒ None (resolver picks latest)."""
    text = ref.split("://", 1)[-1].strip()
    if not text:
        raise NotFound(f"empty model reference: {ref!r}",
                       hint="expected model://name:vN or name:vN")
    if ":" in text:
        name, _, version = text.rpartition(":")
        if not name or not version.startswith("v"):
            raise NotFound(
                f"malformed model reference {ref!r}",
                hint="expected model://rf_detr_s:v2 or rf_detr_s:v2",
            )
        return name, version
    return text, None


def format_model_ref(name: str, version: str) -> str:
    return f"{name}:{version}"


# ---------------------------------------------------------------------------
# the display-then-create unit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewRunPlan:
    """A proposed new run, computed BEFORE anything exists (13 §6.3/§6.4
    SHOW DELTA → confirm → CREATE)."""

    kind: str                      # fork | retrain | finetune
    run_id: str                    # pre-generated (shown in the delta)
    spec: RunSpec
    lineage: dict[str, Any]
    delta: dict[str, Any] = field(default_factory=dict)
    source_run_id: str | None = None      # parent run (fork/retrain/finetune)
    base_model: str | None = None         # normalized name:vN (finetune)
    base_model_id: str | None = None      # internal model id (finetune)
    starting_state: str | None = None     # display fact (retrain)
    warnings: tuple[str, ...] = ()        # display facts (finetune)

    @property
    def identity(self) -> str:
        return self.spec.identity


def new_plan(
    kind: str,
    spec: RunSpec,
    lineage: Mapping[str, Any],
    *,
    delta: Mapping[str, Any] | None = None,
    source_run_id: str | None = None,
    base_model: str | None = None,
    base_model_id: str | None = None,
    starting_state: str | None = None,
    warnings: Iterable[str] = (),
    run_id: str | None = None,
) -> NewRunPlan:
    return NewRunPlan(
        kind=kind,
        run_id=run_id or new_run_id(),
        spec=spec,
        lineage=dict(lineage),
        delta=dict(delta or {}),
        source_run_id=source_run_id,
        base_model=base_model,
        base_model_id=base_model_id,
        starting_state=starting_state,
        warnings=tuple(warnings),
    )


__all__ = [
    "FINETUNE_STRATEGIES",
    "LINEAGE_FILENAME",
    "LINEAGE_SCHEMA",
    "NewRunPlan",
    "build_lineage",
    "canonicalize_overrides",
    "default_lineage",
    "derive_spec",
    "format_model_ref",
    "new_plan",
    "parse_model_ref",
    "parse_overrides",
    "read_lineage",
    "spec_delta",
    "write_lineage",
]
