"""Ingestion DAG — models, sources, transforms (12 §10.2).

`<workspace>/ingestion.yaml` declares WHAT feeds WHAT:

    models:
      rf_detr_s:
        transform: coco_detection
        train_sources: [coco_2017:train, custom_clips:train]
        val_sources: [coco_2017:val]

Source forms (12 §10.2):
  * `name:split` — a registered dataset (split is SOURCE metadata; the
    dataset object key is `name`, identity comes from its registration)
  * `{generated_from: <model>, dataset: <ID>}` — output of another model
    (derived data): `generated_from` is provenance — the model must be
    AVAILABLE in the registry — and `dataset` names the registered bytes

Fail-closed rules: missing/unreadable `ingestion.yaml` (exit 3), unknown
keys/transforms (exit 1), unregistered sources (exit 2), hash mismatch
vs registration (exit 1), unmet `depends_on`/`generated_from` (exit 3 —
the upstream model must be AVAILABLE in the model registry, 13 §5.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound, PreconditionFailed, ValidationBlock
from mlforge.yamlmini import YamlError, load_file

INGESTION_FILE = "ingestion.yaml"

_ALLOWED_MODEL_KEYS = {"transform", "train_sources", "val_sources", "depends_on"}


@dataclass(frozen=True)
class Source:
    """One input of a transform (split semantics from 12 §10.2)."""

    dataset: str | None = None      # registered dataset name (static)
    split: str | None = None
    generated_from: str | None = None  # model name (derived)

    @property
    def is_generated(self) -> bool:
        return self.generated_from is not None

    @property
    def ref(self) -> str:
        if self.is_generated:
            base = f"generated_from:{self.generated_from}"
            # Include the backing dataset when declared: two outputs of the
            # same model must not look like the same source in the doc.
            return f"{base}@{self.dataset}" if self.dataset else base
        return f"{self.dataset}:{self.split}" if self.split else str(self.dataset)

    @classmethod
    def parse(cls, item: Any, where: str) -> Source:
        if isinstance(item, str):
            text = item.strip()
            if ":" in text:
                name, _, split = text.partition(":")
                if not name or not split:
                    raise PreconditionFailed(f"{where}: bad source {item!r}")
                return cls(dataset=name, split=split)
            if not text:
                raise PreconditionFailed(f"{where}: empty source")
            return cls(dataset=text, split=None)
        if isinstance(item, dict):
            extra = set(item) - {"generated_from", "dataset"}
            gen = item.get("generated_from")
            if gen and not extra:
                ds = item.get("dataset")
                if ds is not None and (not isinstance(ds, str) or not ds.strip()):
                    raise PreconditionFailed(
                        f"{where}: dataset must be a non-empty name in {item!r}"
                    )
                return cls(dataset=str(ds).strip() if ds else None,
                           generated_from=str(gen))
            raise PreconditionFailed(
                f"{where}: unsupported source form {item!r} "
                f"(expected 'name:split' or {{generated_from: model, "
                f"dataset: <id>}})"
            )
        raise PreconditionFailed(f"{where}: unsupported source {item!r}")


@dataclass(frozen=True)
class ModelPlan:
    name: str
    transform: str
    train_sources: tuple[Source, ...] = ()
    val_sources: tuple[Source, ...] = ()
    depends_on: tuple[str, ...] = ()

    @property
    def sources(self) -> tuple[Source, ...]:
        return self.train_sources + self.val_sources

    @staticmethod
    def _source_out(s: Source) -> Any:
        # Generated sources must round-trip as their dict form — a bare
        # `generated_from:m@dataset` string would re-parse as name:split.
        if s.is_generated:
            out: dict[str, Any] = {"generated_from": s.generated_from}
            if s.dataset:
                out["dataset"] = s.dataset
            return out
        return s.ref

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform,
            "train_sources": [self._source_out(s) for s in self.train_sources],
            "val_sources": [self._source_out(s) for s in self.val_sources],
            "depends_on": list(self.depends_on),
        }


def ingestion_path(root: str | Path) -> Path:
    return Path(root) / INGESTION_FILE


def load_ingestion(root: str | Path) -> dict[str, ModelPlan]:
    path = ingestion_path(root)
    if not path.is_file():
        raise PreconditionFailed(
            f"ingestion config not found: {path}",
            hint="create ingestion.yaml (12 §10.2): models: {<name>: "
                 "{transform, train_sources, val_sources, depends_on}}",
        )
    try:
        data = load_file(path)
    except YamlError as exc:
        raise PreconditionFailed(f"ingestion.yaml unreadable: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict) \
            or not data["models"]:
        raise PreconditionFailed(
            "ingestion.yaml must contain a non-empty `models:` mapping"
        )
    plans: dict[str, ModelPlan] = {}
    for name, entry in data["models"].items():
        where = f"ingestion.yaml models.{name}"
        if not isinstance(entry, dict):
            raise PreconditionFailed(f"{where} must be a mapping")
        unknown = set(entry) - _ALLOWED_MODEL_KEYS
        if unknown:
            raise PreconditionFailed(
                f"{where}: unknown keys {sorted(unknown)} "
                f"(allowed: {sorted(_ALLOWED_MODEL_KEYS)})"
            )
        transform = entry.get("transform")
        if not isinstance(transform, str) or not transform:
            raise PreconditionFailed(f"{where}: `transform:` is required")
        deps = entry.get("depends_on") or []
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise PreconditionFailed(f"{where}: depends_on must be a list of names")
        plans[str(name)] = ModelPlan(
            name=str(name),
            transform=transform,
            train_sources=tuple(
                Source.parse(s, f"{where}.train_sources")
                for s in (entry.get("train_sources") or [])
            ),
            val_sources=tuple(
                Source.parse(s, f"{where}.val_sources")
                for s in (entry.get("val_sources") or [])
            ),
            depends_on=tuple(str(d) for d in deps),
        )
    return plans


def resolve_order(plans: dict[str, ModelPlan], target: str) -> list[str]:
    """Topological order ending at `target` (12 §10.2 "formalized as DAG").
    Cycles are a config defect → BLOCK, never an infinite loop."""
    order: list[str] = []
    state: dict[str, int] = {}  # 0 visiting / 1 done

    def visit(name: str, chain: tuple[str, ...]) -> None:
        if state.get(name) == 1:
            return
        if name in chain:
            raise ValidationBlock(
                "ingestion DAG cycle: " + " → ".join(chain + (name,))
            )
        plan = plans.get(name)
        if plan is None:
            raise NotFound(f"ingestion model {name!r} not found")
        for dep in plan.depends_on:
            if dep not in plans:
                # Registry-only dependency: nothing to order here — prepare
                # checks availability against the model registry (12 §10.2).
                continue
            visit(dep, chain + (name,))
        state[name] = 1
        order.append(name)

    visit(target, ())
    return order


__all__ = [
    "INGESTION_FILE",
    "ModelPlan",
    "Source",
    "ingestion_path",
    "load_ingestion",
    "resolve_order",
]
