"""Materialize a prepared COCO detection artifact into RF-DETR's layout.

RF-DETR reads the roboflow directory convention: split folders each
holding an `_annotations.coco.json` plus the referenced images
(`rfdetr.datasets` / `dcoco.build_roboflow_from_coco` — the same reader
`is_valid_coco_dataset` gates on: `train/_annotations.coco.json`).

Prepared `coco_detection.v1` records carry `source/split/relative_path/
sha256` for every file — including annotation JSONs (the transform
re-hashes them, so their content is identity-covered). This module maps
those records onto RF-DETR's layout under `<run_dir>/rfdetr_data/`:

  roboflow layout   source already split/… — annotation JSONs are COPIED
                    (verified against the record sha256 first), images
                    referenced by the annotations are SYMLINKED from the
                    registered dataset path.
  coco2017 layout   `annotations/instances_{train,val}2017.json` +
                    `train2017/`, `val2017/` image dirs — JSONs copied
                    into `train/`, `valid/` (file_names are basenames,
                    so no content rewrite); images symlinked.

Fail-closed (never a silent guess):
  * not exactly one dataset ref        → ValidationBlock
  * neither layout recognized          → ValidationBlock naming both
  * missing train or valid annotations → ValidationBlock (RF-DETR
    validates EVERY epoch — `_ForceLastEpochValidationCallback`
    guarantees the final epoch validates too, so a run without a
    valid split cannot work; say so up front)
  * annotation references an image absent from the prepared records
    → ValidationBlock (source content drifted after prepare)
  * two sources claim the same split path with different bytes
    → ValidationBlock (joining datasets is never implied)

Only stdlib — unit-testable without rfdetr installed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash_bytes

#: Materialized dataset root inside the run directory.
DATA_DIRNAME = "rfdetr_data"

_LAYOUT_HINT = (
    "RF-DETR needs a COCO layout: roboflow (`train/_annotations.coco.json` "
    "+ `valid/…`) or COCO-2017 (`annotations/instances_train2017.json` + "
    "`train2017/` images) — `mlforge dataset types` lists the supported "
    "dataset transforms"
)


@dataclass(frozen=True)
class Materialized:
    """What the trainer needs to start (recorded into dataloader_state)."""

    dir: Path
    layout: str
    train_images: int
    valid_images: int
    classes: int
    categories: tuple[str, ...]


def _load_records(
    root: Path, train_datasets: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Prepared store artifact(s) → records + source-name → base path map.

    Mirrors gate semantics locally (registered, version-exact,
    store-backed, verified bytes) — same contract as text/reid loaders."""
    from mlforge.ingest import config as ingest_config
    from mlforge.ingest.identity import parse_ref
    from mlforge.store import ContentStore

    if len(train_datasets) != 1:
        raise ValidationBlock(
            f"RF-DETR trains from ONE prepared dataset — got "
            f"{len(train_datasets)} ({', '.join(train_datasets) or 'none'})",
            hint="declare a single dataset in run_spec.train_datasets "
                 "(joining COCO datasets would merge label spaces — "
                 "never implied)",
        )
    ref = train_datasets[0]
    name, version = parse_ref(ref)
    reg_path = root / "datasets" / name / "identity.json"
    if not reg_path.is_file():
        raise PreconditionFailed(
            f"dataset {name!r} not registered — mlforge dataset add "
            f"{name} <PATH>",
        )
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    reg_version = reg.get("version")
    if reg_version and str(reg_version) != version:
        raise ValidationBlock(
            f"{ref}: registered version {reg_version!r} != {version!r} "
            "(versions are identity, never reinterpreted)",
        )
    store = ContentStore(root / "store")
    identity = str(reg.get("identity") or "")
    if not identity or not store.contains(identity):
        raise PreconditionFailed(
            f"{ref} is not a prepared store artifact — "
            "run `mlforge prepare <model>` first",
        )
    doc = json.loads(store.get_bytes(identity, verify=True).decode("utf-8"))
    schema = str((doc.get("transform") or {}).get("name") or "")
    if schema and schema != "coco_detection":
        raise ValidationBlock(
            f"{ref} was prepared with transform {schema!r} — RF-DETR "
            "detection training needs `coco_detection` records",
        )
    records = [r for r in (doc.get("records") or []) if isinstance(r, dict)]
    if not records:
        raise ValidationBlock(f"{ref} contains no file records")

    paths = ingest_config.load_paths(root)
    bases: dict[str, str] = {}
    for rec in records:
        src = str(rec.get("source") or "")
        ds_name = src.split(":", 1)[0]
        base = paths.get(ds_name)
        if not base:
            raise PreconditionFailed(
                f"dataset {ds_name!r}: no machine-local path configured — "
                f"mlforge dataset add {ds_name} <PATH>",
            )
        bases[ds_name] = base
    return records, bases


def _find_ann(
    records: Sequence[Mapping[str, Any]], suffix: str
) -> Mapping[str, Any] | None:
    """First record whose relative_path ends with `suffix`."""
    for rec in records:
        rel = str(rec.get("relative_path") or "")
        if rel == suffix or rel.endswith("/" + suffix):
            return rec
    return None


def _detect_layout(records: Sequence[Mapping[str, Any]]) -> str:
    rels = {str(r.get("relative_path") or "") for r in records}
    if any(r == "train/_annotations.coco.json" or r.endswith("/train/_annotations.coco.json")
           for r in rels):
        return "roboflow"
    if any(r.startswith("annotations/instances_") and "_train" in r for r in rels):
        return "coco2017"
    raise ValidationBlock(
        f"dataset does not match a layout RF-DETR can read — {_LAYOUT_HINT}",
    )


def _verify_and_copy(src_path: Path, rec: Mapping[str, Any], dest: Path) -> None:
    """Annotation JSONs are small: verify identity, then copy (so later
    source edits cannot mutate a running training run's labels)."""
    if not src_path.is_file():
        raise ValidationBlock(
            f"annotation missing at registered path: {src_path}",
            hint="dataset content changed since registration — "
                 "mlforge dataset add <NAME> <PATH> --force",
        )
    data = src_path.read_bytes()
    expected = str(rec.get("sha256") or "")
    actual = content_hash_bytes(data)  # "sha256:<hex>" — same shape as records
    if expected and expected != actual:
        raise ValidationBlock(
            f"annotation {rec.get('relative_path')} no longer matches its "
            "prepared sha256 — content drift since prepare",
            hint="re-run `mlforge prepare <model>` (identity is verified, "
                 "never reinterpreted)",
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def _link_image(src_path: Path, dest: Path, rec: Mapping[str, Any]) -> None:
    """Symlink an identity-recorded image into the materialized layout."""
    if not src_path.is_file():
        raise ValidationBlock(
            f"image missing at registered path: {src_path}",
            hint="dataset content changed since registration — "
                 "mlforge dataset add <NAME> <PATH> --force",
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink() and Path(os.readlink(dest)) == src_path:
            return  # idempotent rebuild
        dest.unlink()
    dest.symlink_to(src_path)


def _resolve_image(
    base: Path, file_name: str, split_names: Sequence[str]
) -> tuple[Path, str]:
    """Locate an annotation-referenced image under the source tree.

    COCO file_names are basenames relative to the image dir; some
    exports embed a prefix. Try the name as-is, then each known split
    image dir. Returns (absolute source path, relative record key)."""
    candidates: list[str] = [file_name]
    for name in split_names:
        cand = f"{name}/{file_name}"
        if cand not in candidates:
            candidates.append(cand)
    for rel in candidates:
        p = base / rel
        if p.is_file():
            return p, rel
    tried = ", ".join(str(base / c) for c in candidates)
    raise ValidationBlock(
        f"annotation references image {file_name!r} not found under the "
        f"registered path (tried: {tried})",
        hint="annotations and images must match the registered dataset "
             "content — re-register with `mlforge dataset add --force`",
    )


def materialize(
    root: Path, run_dir: Path, train_datasets: Sequence[str]
) -> Materialized:
    """Build `<run_dir>/rfdetr_data/` from the prepared artifact.

    Rebuilt from scratch on every construction (cheap: symlinks + small
    JSON copies) so a resume always re-verifies identity — the scratch
    dir is run-scoped and never shared between runs."""
    records, bases = _load_records(root, train_datasets)
    layout = _detect_layout(records)
    by_rel: dict[str, dict[str, Any]] = {}
    for rec in records:
        rel = str(rec.get("relative_path") or "")
        if not rel:
            continue
        prior = by_rel.get(rel)
        if prior is not None and str(prior.get("sha256")) != str(rec.get("sha256")):
            raise ValidationBlock(
                f"two sources claim {rel!r} with different bytes — "
                "RF-DETR needs a single annotation/image tree (joining "
                "datasets is never implied)",
            )
        by_rel[rel] = dict(rec)

    dest = run_dir / DATA_DIRNAME
    if dest.exists():
        # Run-scoped scratch — rebuilt from scratch every construction so
        # a resume re-verifies identity (cheap: symlinks + small JSONs).
        import shutil

        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    if layout == "roboflow":
        return _materialize_roboflow(bases, by_rel, dest)
    return _materialize_coco2017(bases, by_rel, dest)


def _split_plan(
    bases: dict[str, str],
    by_rel: Mapping[str, dict[str, Any]],
) -> tuple[str, str, Path, Mapping[str, Any], Mapping[str, Any] | None]:
    """Resolve (train_rel, valid_rel, source base, train_rec, valid_rec)."""
    train_rec = _find_ann(by_rel.values(), "train/_annotations.coco.json")
    valid_rec = (_find_ann(by_rel.values(), "valid/_annotations.coco.json")
                 or _find_ann(by_rel.values(), "val/_annotations.coco.json"))
    if train_rec is None:
        raise ValidationBlock(
            f"dataset has no `train/_annotations.coco.json` — "
            f"{_LAYOUT_HINT}",
        )
    if valid_rec is None:
        raise ValidationBlock(
            "dataset has no valid-split annotations "
            "(`valid/_annotations.coco.json`) — RF-DETR validates every "
            "epoch (the final epoch is forced to validate), so training "
            "without a valid split cannot run",
            hint="add a valid/ (or val/) split with COCO annotations to "
                 "the dataset, then re-register and re-prepare",
        )
    train_src = str(train_rec["source"]).split(":", 1)[0]
    base = Path(bases[train_src])
    return (str(train_rec["relative_path"]), str(valid_rec["relative_path"]),
            base, train_rec, valid_rec)


def _materialize_roboflow(
    bases: dict[str, str],
    by_rel: dict[str, dict[str, Any]],
    dest: Path,
) -> Materialized:
    train_rel, valid_rel, _base, train_rec, valid_rec = _split_plan(
        bases, by_rel,
    )
    assert valid_rec is not None
    pairs = (("train", train_rel, train_rec), ("valid", valid_rel, valid_rec))
    return _emit_splits(bases, by_rel, dest, pairs, layout="roboflow")


def _materialize_coco2017(
    bases: dict[str, str],
    by_rel: dict[str, dict[str, Any]],
    dest: Path,
) -> Materialized:
    """Standard COCO: copy instances_{train,val}*.json into train/valid
    (file_names are basenames → no rewrite) + symlink the image dirs."""
    train_ann = None
    valid_ann = None
    for rel in sorted(by_rel):
        if not rel.startswith("annotations/instances_"):
            continue
        if "_train" in rel:
            train_ann = by_rel[rel]
        elif "_val" in rel:
            valid_ann = by_rel[rel]
    if train_ann is None:
        raise ValidationBlock(
            f"dataset has no `annotations/instances_*_train*.json` — "
            f"{_LAYOUT_HINT}",
        )
    if valid_ann is None:
        raise ValidationBlock(
            "dataset has no `annotations/instances_*_val*.json` — RF-DETR "
            "validates every epoch (the final epoch is forced to validate)",
            hint="add validation annotations to the dataset, re-register, "
                 "re-prepare",
        )
    pairs = (
        ("train", str(train_ann["relative_path"]), train_ann),
        ("valid", str(valid_ann["relative_path"]), valid_ann),
    )
    return _emit_splits(bases, by_rel, dest, pairs, layout="coco2017")


def _emit_splits(
    bases: dict[str, str],
    by_rel: dict[str, dict[str, Any]],
    dest: Path,
    pairs: tuple[tuple[str, str, Mapping[str, Any]], ...],
    *,
    layout: str,
) -> Materialized:
    """Copy each split's annotations + symlink every referenced image."""
    categories: tuple[str, ...] = ()
    counts: dict[str, int] = {}
    for dest_split, ann_rel, ann_rec in pairs:
        src_ds = str(ann_rec["source"]).split(":", 1)[0]
        base = Path(bases[src_ds])
        ann_path = base / ann_rel
        _verify_and_copy(ann_path, ann_rec, dest / dest_split / "_annotations.coco.json")
        doc = json.loads(ann_path.read_bytes())
        if dest_split == "train":
            categories = tuple(
                str(c.get("name") or c.get("id"))
                for c in doc.get("categories") or []
            )
            if not categories:
                raise ValidationBlock(
                    "train annotations declare no categories — nothing to "
                    "train (labels are identity, never inferred)",
                )
        # image dir for the SOURCE side of this split
        if layout == "roboflow":
            src_dir = str(Path(ann_rel).parent)  # "train" / "valid"
            split_names = [src_dir]
        else:
            # annotations/instances_train2017.json → train2017/
            stem = Path(ann_rel).stem  # instances_train2017
            tag = stem.split("_", 1)[1] if "_" in stem else "train"
            src_dir = tag
            split_names = [tag, tag.replace("2017", ""), "images/" + tag]
        n = 0
        for image in doc.get("images") or []:
            file_name = str(image.get("file_name") or "")
            if not file_name:
                continue
            if layout == "roboflow":
                rel = f"{src_dir}/{file_name}"
                src_path = base / rel
                if not src_path.is_file():
                    raise ValidationBlock(
                        f"annotation references {rel!r} which is not under "
                        f"the registered path {base}",
                    )
            else:
                src_path, rel = _resolve_image(base, file_name, split_names)
            rec = by_rel.get(rel)
            if rec is None:
                raise ValidationBlock(
                    f"image {rel!r} is referenced by the annotations but "
                    "absent from the prepared records — the dataset content "
                    "drifted after prepare",
                    hint="re-run `mlforge prepare <model>` after fixing "
                         "the dataset (identity is verified, never "
                         "reinterpreted)",
                )
            _link_image(src_path, dest / dest_split / file_name, rec)
            n += 1
        if n == 0:
            raise ValidationBlock(
                f"split {dest_split!r} references no images — nothing to "
                "train/validate",
            )
        counts[dest_split] = n

    # Prove is_valid_coco_dataset's exact predicate before rfdetr sees it.
    if not (dest / "train" / "_annotations.coco.json").is_file():
        raise ValidationBlock(
            "materialization produced no train annotations — internal "
            f"invariant broken ({_LAYOUT_HINT})",
        )
    return Materialized(
        dir=dest,
        layout=layout,
        train_images=counts.get("train", 0),
        valid_images=counts.get("valid", 0),
        classes=len(categories),
        categories=categories,
    )


__all__ = ["DATA_DIRNAME", "Materialized", "materialize"]
