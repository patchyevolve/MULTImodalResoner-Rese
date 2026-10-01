"""Immutable run_spec — the experiment's SEMANTIC IDENTITY.

Normative: 12_training_system.md §4, §6 (identity model), §13.3 (hard
namespace separation), §25 (five-layer identity stack).

  * run_spec is written ONCE at run creation and never mutated.
    Semantic change = `fork` = new run. There is no --reconfigure.
  * Its canonical hash is the SEMANTIC IDENTITY of the run; hardware,
    lifecycle, and execution live in other layers and never touch it.
  * `effective_batch` invariant (12 §8.3):
        global_batch == micro_batch × grad_accum × world_size
    (execution may re-solve micro/accum/world; the product must equal
    the frozen value — enforced by the validation gate, step 3.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mlforge.errors import ValidationBlock
from mlforge.hashing import run_spec_hash

SCHEMA_VERSION = 1

#: Semantic fields that must be present (12 §17 invariant table rows:
#: optimizer semantics preserved, LR schedule state recoverable, global
#: batch preserved, seed). Missing semantic fields → BLOCK at creation.
REQUIRED_SEMANTIC_FIELDS = (
    "optimizer",
    "learning_rate",
    "scheduler",
    "loss",
    "seed",
    "global_batch",
    "epochs",
    "precision_policy",
)


@dataclass(frozen=True)
class RunSpec:
    model: str
    train_datasets: tuple[str, ...]
    semantic: Mapping[str, Any]
    val_dataset: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not self.model:
            raise ValidationBlock("run_spec.model is required")
        if not self.train_datasets:
            raise ValidationBlock("run_spec.train_datasets must not be empty")
        missing = [f for f in REQUIRED_SEMANTIC_FIELDS if f not in self.semantic]
        if missing:
            raise ValidationBlock(
                f"run_spec.semantic missing required fields: {', '.join(missing)}",
                hint="semantic identity is frozen at creation (12 §13.3)",
            )
        # Freeze: callers must not mutate semantic after construction.
        object.__setattr__(self, "semantic", dict(self.semantic))
        object.__setattr__(self, "train_datasets", tuple(self.train_datasets))

    @property
    def identity(self) -> str:
        """`sha256:<hex>` SEMANTIC IDENTITY of this experiment (12 §4)."""
        return run_spec_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "train_datasets": list(self.train_datasets),
            "val_dataset": self.val_dataset,
            "semantic": dict(self.semantic),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunSpec":
        if d.get("schema_version") != SCHEMA_VERSION:
            # Never silently reinterpret (12 §17): unknown schema → BLOCK.
            raise ValidationBlock(
                f"unsupported run_spec schema_version: {d.get('schema_version')!r} "
                f"(supported: {SCHEMA_VERSION})",
                hint="old schemas require explicit migration: v1 → migration → v2",
            )
        return cls(
            model=d["model"],
            train_datasets=tuple(d["train_datasets"]),
            val_dataset=d.get("val_dataset"),
            semantic=d["semantic"],
            schema_version=d["schema_version"],
        )

    def write_once(self, path: str | Path) -> None:
        """Persist immutability: refuse to overwrite an existing run_spec."""
        p = Path(path)
        if p.exists():
            raise ValidationBlock(
                f"run_spec already exists at {p} — run_spec is immutable",
                hint="semantic change = `mlforge fork` = new run (12 §13.3)",
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
        tmp.replace(p)  # atomic rename (12 §11.1)
