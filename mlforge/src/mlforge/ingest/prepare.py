"""`mlforge prepare` — resolve → transform → register the derived dataset.

Sequence (13 §6.6 / 12 §6.4):

  1. resolve every source of the model plan:
       registered? → path configured? → content identity matches?
  2. cache lookup on the four-field cache_key (inputs + transform + env)
  3. on a miss: run the deterministic transform, persist the artifact in
     the content store, record the cache index
  4. register the DERIVED dataset `<model>_prepared` and walk it through
     REGISTERED → VERIFIED → PREPARED (the normal dataset machine)

Fail-closed mapping (13 §4.2 exit codes):
  unknown model / missing ingestion.yaml / unregistered source → 2/3
  identity mismatch at a source path                      → 1 (+reregister)
  unknown transform / empty output                        → 1
  unmet depends_on / generated_from (model registry later) → 3

Idempotency (13 §4.4): the caller wraps `prepare()` in
`execute_idempotent(command_id, "prepare", ...)` — a retry returns the
original result and journals COMMAND_DEDUPED, never a second artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound, PreconditionFailed, ValidationBlock
from mlforge.hashing import canonical_json, content_hash_bytes
from mlforge.ingest import config as ingest_config
from mlforge.ingest.dag import ModelPlan, load_ingestion, resolve_order
from mlforge.ingest.identity import DEFAULT_VERSION, recompute_identity
from mlforge.ingest.transforms import (
    PREPARED_SCHEMA,
    ResolvedSource,
    cache_key,
    env_fingerprint,
    run_transform,
    transform_identity,
)
from mlforge.store import ContentStore

#: Namespace of the cache index inside the workspace (derived data, NOT
#: source registration — registration lives in datasets/<id>/).
CACHE_INDEX = Path("artifacts") / "prepare_cache.json"


@dataclass(frozen=True)
class PreparedResult:
    model: str
    dataset_ref: str            # dataset://<model>_prepared:v1
    dataset_id: str
    artifact_hash: str
    cache: str                  # "hit" | "miss"
    record_count: int
    cache_key: str
    sources: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dataset_id": self.dataset_id,
            "dataset": self.dataset_ref,
            "artifact_hash": self.artifact_hash,
            "cache": self.cache,
            "record_count": self.record_count,
            "cache_key": self.cache_key,
            "sources": list(self.sources),
        }


def derived_dataset_id(model: str) -> str:
    """Content-addressed derivation: same model ⇒ same derived key."""
    return f"{model}_prepared"


def _read_registration(root: Path, dataset_id: str) -> dict[str, Any]:
    p = root / "datasets" / dataset_id / "identity.json"
    if not p.is_file():
        raise NotFound(
            f"dataset {dataset_id!r} not registered",
            hint=f"register it first: mlforge dataset add {dataset_id} <PATH>",
        )
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreconditionFailed(
            f"dataset {dataset_id}: identity.json unreadable ({exc})"
        ) from exc


def _dataset_state(workflow: Any, root: Path, dataset_id: str) -> str:
    p = root / "datasets" / dataset_id / "status.json"
    if p.is_file():
        try:
            return str(json.loads(p.read_text(encoding="utf-8"))["state"])
        except (json.JSONDecodeError, KeyError):
            pass
    return str(workflow.get_dataset_status(dataset_id)["state"])


def resolve_sources(
    root: Path, plan: ModelPlan, workflow: Any
) -> list[ResolvedSource]:
    """Path config → recompute identity → compare with registration.

    Verify, never scan (13 §10): we resolve the CONFIGURED path and hash
    exactly what is there; a wrong path yields a mismatch, not a search.
    """
    paths = ingest_config.load_paths(root)
    resolved: list[ResolvedSource] = []
    for src in plan.sources:
        if src.is_generated:
            raise PreconditionFailed(
                f"source generated_from:{src.generated_from} — the model "
                f"registry cannot resolve it yet",
                hint="derived sources arrive with the model registry build step",
            )
        if not src.dataset:
            raise PreconditionFailed(f"source {src.ref!r} has no dataset name")
        name = src.dataset
        reg = _read_registration(root, name)
        state = _dataset_state(workflow, root, name)
        if state not in ("VERIFIED", "PREPARED"):
            raise PreconditionFailed(
                f"dataset {name!r} is {state} — sources must be VERIFIED "
                f"before prepare",
                hint=f"mlforge dataset verify {name}",
            )
        path = paths.get(name)
        if not path:
            raise PreconditionFailed(
                f"dataset {name!r}: no machine-local path configured",
                hint=f"mlforge dataset add {name} <PATH>",
            )
        identity, manifest = recompute_identity(
            path, name, reg.get("version", DEFAULT_VERSION), schema=reg.get("schema")
        )
        if identity != reg.get("identity"):
            raise ValidationBlock(
                f"dataset {name!r}: content at {path} no longer matches its "
                f"registration (found {identity}, registered "
                f"{reg.get('identity')})",
                hint=f"re-register if intended: mlforge dataset add {name} "
                     f"<PATH> --force",
            )
        resolved.append(ResolvedSource(
            ref=f"{name}:{src.split}" if src.split else name,
            split=src.split,
            dataset_id=name,
            path=Path(path),
            identity=identity,
            entries=tuple(manifest.files),
        ))
    return resolved


def _load_cache(root: Path) -> dict[str, Any]:
    p = root / CACHE_INDEX
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}  # corrupt index ⇒ cache miss (rebuild), never a crash
    return data if isinstance(data, dict) else {}


def _save_cache(root: Path, data: dict[str, Any]) -> None:
    p = root / CACHE_INDEX
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def _ensure_prepared(
    workflow: Any,
    root: Path,
    dataset_id: str,
    identity: str,
    *,
    version: str = DEFAULT_VERSION,
    file_count: int,
    total_bytes: int,
) -> str:
    """Register/verify/prepare the derived dataset idempotently.

    Returns "registered" | "reregistered" | "verified" | "prepared" |
    "unchanged" — the machine (13 §5.2) decides each step's legality."""
    meta: dict[str, Any] = dict(
        version=version, file_count=file_count, total_bytes=total_bytes
    )
    d = root / "datasets" / dataset_id
    if not d.is_dir():
        workflow.register_dataset(dataset_id, identity, **meta)
        workflow.verify_dataset(dataset_id, identity_matches=True,
                                found_identity=identity)
        workflow.prepare_dataset(dataset_id)
        return "registered"

    reg = _read_registration(root, dataset_id)
    state = _dataset_state(workflow, root, dataset_id)
    if reg.get("identity") != identity or state == "REJECTED":
        # Content changed (or was rejected earlier) ⇒ explicit reregister
        # (journal, no silent reinterpretation — 13 §5.2).
        workflow.reregister_dataset(dataset_id, identity, **meta)
        workflow.verify_dataset(dataset_id, identity_matches=True,
                                found_identity=identity)
        workflow.prepare_dataset(dataset_id)
        return "reregistered"
    if state == "REGISTERED":
        workflow.verify_dataset(dataset_id, identity_matches=True,
                                found_identity=identity)
        workflow.prepare_dataset(dataset_id)
        return "verified"
    if state == "VERIFIED":
        workflow.prepare_dataset(dataset_id)
        return "prepared"
    return "unchanged"  # already PREPARED with this identity


def prepare(root: str | Path, model: str, *, workflow: Any) -> PreparedResult:
    """Resolve, transform (cache-aware), and register `<model>_prepared`."""
    root = Path(root)
    plans = load_ingestion(root)  # missing/unreadable → exit 3
    if model not in plans:
        raise NotFound(
            f"model {model!r} not found in ingestion.yaml",
            hint="known models: " + (", ".join(sorted(plans)) or "(none)"),
        )
    # Explicit DAG walk — proves order is resolvable (12 §10.2) and
    # surfaces cycles as a BLOCK before any hashing work.
    order = resolve_order(plans, model)
    plan = plans[model]
    if plan.depends_on:
        raise PreconditionFailed(
            f"model {model!r} depends on {', '.join(plan.depends_on)} — "
            f"upstream models must be AVAILABLE before prepare",
            hint="the model registry lands in a later build step",
        )

    sources = resolve_sources(root, plan, workflow)
    t_ident = transform_identity(plan.transform)
    key = cache_key(
        inputs=[{"ref": s.ref, "identity": s.identity} for s in sources],
        transform_identity_data=t_ident,
        environment=env_fingerprint(),
    )

    derived = derived_dataset_id(model)
    store = ContentStore(root / "store")
    cache = _load_cache(root)
    hit = cache.get(key)
    artifact_hash: str | None = None
    doc: dict[str, Any] | None = None
    if hit and isinstance(hit, dict) and store.contains(str(hit.get("artifact_hash", ""))):
        artifact_hash = str(hit["artifact_hash"])
        doc = json.loads(store.get_bytes(artifact_hash).decode("utf-8"))
        cache_state = "hit"
    else:
        cache_state = "miss"

    if doc is None:
        out = run_transform(plan, sources)
        doc = {
            "artifact_schema": PREPARED_SCHEMA,
            "model": model,
            "transform": {**t_ident, "output_schema": out.get("output_schema")},
            "cache_key": key,
            "order": order,
            "sources": [
                {"ref": s.ref, "identity": s.identity, "files": s.file_count}
                for s in sources
            ],
            "record_count": len(out["records"]),
            "records": out["records"],
            "annotations": out.get("annotations", []),
        }
        blob = canonical_json(doc)
        artifact_hash = store.put_bytes(blob)
        if content_hash_bytes(blob) != artifact_hash:
            raise ValidationBlock("prepared artifact hash mismatch after write")
        cache[key] = {"artifact_hash": artifact_hash, "derived_id": derived}
        _save_cache(root, cache)

    assert artifact_hash is not None  # hit or just written
    _ensure_prepared(
        workflow, root, derived, artifact_hash,
        file_count=int(doc.get("record_count", 0)),
        total_bytes=len(canonical_json(doc)),
    )
    return PreparedResult(
        model=model,
        dataset_ref=f"dataset://{derived}:{DEFAULT_VERSION}",
        dataset_id=derived,
        artifact_hash=artifact_hash,
        cache=cache_state,
        record_count=int(doc.get("record_count", 0)),
        cache_key=key,
        sources=tuple(s.ref for s in sources),
    )


__all__ = [
    "CACHE_INDEX",
    "PreparedResult",
    "derived_dataset_id",
    "prepare",
    "resolve_sources",
]
