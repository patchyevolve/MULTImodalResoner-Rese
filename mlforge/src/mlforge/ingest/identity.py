"""Dataset identity — cryptographic, never count-based (12 §6.2/§6.3).

    Dataset
       ├── dataset_id        # e.g. "coco_2017"  (object key, POSIX-safe)
       ├── version           # e.g. "v1"
       ├── schema            # annotation format (optional)
       ├── files[]
       │     ├── sha256
       │     ├── size
       │     └── relative_path      # NEVER absolute path (12 §6.3)
       └── identity = SHA256(canonical manifest)   # "sha256:..."

Hard rules enforced here:
  * sample counts are NOT identity — every file's bytes are hashed;
  * absolute paths never enter identity (machine-local, 12 §6.3);
  * symlinks are refused (an identity we cannot walk deterministically
    on every OS is not an identity — 12 §6.3 "verify, never guess");
  * an empty dataset has no identity → BLOCK (fail-closed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mlforge.errors import NotFound, ValidationBlock
from mlforge.hashing import content_hash, file_hash

DEFAULT_VERSION = "v1"


@dataclass(frozen=True)
class FileEntry:
    relative_path: str  # POSIX separators — portable across OSes
    sha256: str
    size: int

    def to_dict(self) -> dict:
        return {"relative_path": self.relative_path, "sha256": self.sha256,
                "size": self.size}

    @classmethod
    def from_dict(cls, d: dict) -> "FileEntry":
        return cls(str(d["relative_path"]), str(d["sha256"]), int(d["size"]))


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    version: str
    files: tuple[FileEntry, ...]
    schema: str | None = None

    @property
    def identity(self) -> str:
        """manifest_hash = SHA256(canonical manifest) (12 §6.2)."""
        return content_hash({
            "dataset_id": self.dataset_id,
            "version": self.version,
            "schema": self.schema,
            "files": [f.to_dict() for f in self.files],
        })

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "schema": self.schema,
            "identity": self.identity,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "files": [f.to_dict() for f in self.files],
        }


def parse_ref(ref: str) -> tuple[str, str]:
    """`coco_2017` | `coco_2017:v1` | `dataset://coco_2017:v1` → (id, version)."""
    text = ref.strip()
    if text.startswith("dataset://"):
        text = text[len("dataset://"):]
    if not text:
        raise ValidationBlock(f"empty dataset reference: {ref!r}")
    if ":" in text:
        name, _, version = text.partition(":")
        if not name or not version:
            raise ValidationBlock(f"invalid dataset reference: {ref!r}")
        return name, version
    return text, DEFAULT_VERSION


def full_ref(dataset_id: str, version: str = DEFAULT_VERSION) -> str:
    return f"{dataset_id}:{version}"


def scan_files(path: str | Path) -> list[FileEntry]:
    """Hash every regular file under `path` (deterministic, sorted).

    Refuses symlinks and empty trees — both would make the identity a
    guess (13 §1 fail-closed)."""
    root = Path(path)
    if not root.exists():
        raise NotFound(f"dataset path not found: {path}")
    if not root.is_dir():
        raise ValidationBlock(f"dataset path is not a directory: {path}")

    entries: list[FileEntry] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames):
            p = Path(dirpath) / name
            if p.is_symlink():
                raise ValidationBlock(
                    f"symlink in dataset tree refused (identity must be "
                    f"portable): {p.relative_to(root).as_posix()}"
                )
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                raise ValidationBlock(
                    f"symlink in dataset tree refused (identity must be "
                    f"portable): {p.relative_to(root).as_posix()}"
                )
            if not p.is_file():
                continue  # fifo/socket/etc. — not dataset bytes
            entries.append(FileEntry(
                relative_path=p.relative_to(root).as_posix(),
                sha256=file_hash(str(p)),
                size=p.stat().st_size,
            ))

    if not entries:
        raise ValidationBlock(
            f"no files under {path} — an empty dataset has no identity "
            f"(fail-closed)"
        )
    entries.sort(key=lambda e: e.relative_path)
    return entries


def build_manifest(
    path: str | Path,
    dataset_id: str,
    version: str = DEFAULT_VERSION,
    schema: str | None = None,
) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=dataset_id,
        version=version,
        schema=schema,
        files=tuple(scan_files(path)),
    )


def recompute_identity(
    path: str | Path,
    dataset_id: str,
    version: str = DEFAULT_VERSION,
    schema: str | None = None,
) -> tuple[str, DatasetManifest]:
    manifest = build_manifest(path, dataset_id, version, schema)
    return manifest.identity, manifest


__all__ = [
    "DEFAULT_VERSION",
    "DatasetManifest",
    "FileEntry",
    "build_manifest",
    "full_ref",
    "parse_ref",
    "recompute_identity",
    "scan_files",
]
