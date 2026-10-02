"""Transforms — deterministic, registry-gated, code-hashed (12 §6.4).

A transform turns VERIFIED source datasets into a prepared artifact:

    output = transform(input)                      # deterministic
    cache_key = SHA256(inputs + transform + config + environment)
                                                    # (12 §6.4)

Identity has four independent parts — a change in ANY of them invalidates
the cache (no stale reuse, no guessing):

  * input   — each source dataset's cryptographic identity
  * code    — hash of the transform implementation itself (`inspect`-based)
  * config  — hash of the transform configuration (empty today, wired for
              when per-model options arrive)
  * env     — python/platform fingerprint (placeholder until the OCI
              image digest lands: an image digest is a *better* env
              identity, same mechanism)

Unknown transforms are a ValidationBlock (exit 1) — the registry only
runs code that is REGISTERED (`register_transform` is the global,
open extension point for new dataset types); we never "run whatever
is there".
"""

from __future__ import annotations

import inspect
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash, content_hash_bytes
from mlforge.ingest.dag import ModelPlan
from mlforge.ingest.identity import FileEntry

#: Prepared artifact envelope schema (what `mlforge prepare` persists).
PREPARED_SCHEMA = "mlforge.prepared_dataset.v1"


@dataclass(frozen=True)
class ResolvedSource:
    """One VERIFIED source dataset, ready for a transform."""

    ref: str                    # "coco_2017:train"
    split: str | None
    dataset_id: str
    path: Path
    identity: str               # cryptographic identity (sha256:...)
    entries: tuple[FileEntry, ...]

    @property
    def file_count(self) -> int:
        return len(self.entries)


def coco_detection(sources: list[ResolvedSource]) -> dict[str, Any]:
    """COCO-style detection data: one record per file, canonical re-hash
    of every JSON annotation (proves annotations PARSE, not just exist)."""
    records: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for src in sources:
        for entry in src.entries:
            records.append({
                "source": src.ref,
                "split": src.split,
                "relative_path": entry.relative_path,
                "sha256": entry.sha256,
                "size": entry.size,
            })
            if entry.relative_path.endswith(".json"):
                raw = (src.path / entry.relative_path).read_text(encoding="utf-8")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValidationBlock(
                        f"annotation does not parse as JSON: "
                        f"{src.ref}:{entry.relative_path}: {exc}"
                    ) from exc
                annotations.append({
                    "source": src.ref,
                    "relative_path": entry.relative_path,
                    "canonical_sha256": content_hash(parsed),
                })
    return {
        "output_schema": "coco_detection.v1",
        "records": records,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# text_corpus — extract → normalize → chunk text documents (build step 8:
# ingestion/transform DAG for the reasoner's text data; 13 §10.2)
# ---------------------------------------------------------------------------

#: Target characters per training chunk (chunk boundaries follow paragraph
#: breaks when possible — models learn from complete passages, not cuts).
TEXT_CHUNK_CHARS = 2000
#: Below this, a fragment is noise — merged into neighbours, never emitted
#: as a record of its own.
TEXT_MIN_CHUNK_CHARS = 40

_TEXT_SUFFIXES = frozenset({".txt", ".md", ".rst"})
#: Extensions the transform KNOWS how to read — everything else in the
#: source folder is ignored (not an error: datasets carry stray files).
_READABLE_SUFFIXES = _TEXT_SUFFIXES | {".pdf", ".docx", ".zip"}
#: zip-in-zip depth guard — archives are data, not programs.
_MAX_ARCHIVE_DEPTH = 2


def _plain_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _pdf_text(data: bytes, label: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PreconditionFailed(
            "PDF text extraction needs pypdf (not installed)",
            hint="pip install pypdf",
        ) from exc
    import io

    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(
            (page.extract_text() or "") for page in reader.pages
        )
    except Exception as exc:  # corrupt/encrypted PDF ⇒ fail-closed with label
        raise ValidationBlock(f"PDF unreadable: {label}: {exc}") from exc


def _docx_text(data: bytes, label: str) -> str:
    """DOCX = a zip of XML — stdlib extraction (no dependency)."""
    import io
    import xml.etree.ElementTree as ET
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            xml = zf.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValidationBlock(f"DOCX unreadable: {label}: {exc}") from exc
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValidationBlock(f"DOCX XML unreadable: {label}: {exc}") from exc
    lines: list[str] = []
    for para in root.iter():
        if para.tag.rsplit("}", 1)[-1] == "p":
            text = "".join(
                node.text or ""
                for node in para.iter()
                if node.tag.rsplit("}", 1)[-1] == "t"
            )
            lines.append(text)
    return "\n\n".join(line for line in lines if line.strip())


def _archive_text(data: bytes, label: str, depth: int) -> str:
    import io
    import zipfile

    if depth >= _MAX_ARCHIVE_DEPTH:
        return ""  # deeper nesting is not data we promise to read
    parts: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                try:
                    member = zf.read(info.filename)
                except Exception:  # a torn member must not kill the source
                    continue
                text = _extract_unit(info.filename, member, depth + 1)
                if text.strip():
                    parts.append(text)
    except zipfile.BadZipFile as exc:
        raise ValidationBlock(f"archive unreadable: {label}: {exc}") from exc
    return "\n\n".join(parts)


def _extract_unit(name: str, data: bytes, depth: int = 0) -> str:
    """Bytes → text for one file/archive member. Unknown suffix ⇒ "". """
    suffix = Path(name).suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return _plain_text(data)
    if suffix == ".pdf":
        return _pdf_text(data, name)
    if suffix == ".docx":
        return _docx_text(data, name)
    if suffix == ".zip":
        return _archive_text(data, name, depth)
    return ""


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    # collapse runs of blank lines — chunk identity stays stable
    out: list[str] = []
    blank = False
    for line in text.split("\n"):
        stripped = line.rstrip()
        if not stripped:
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(stripped)
            blank = False
    return "\n".join(out).strip()


def _chunk_text(text: str) -> list[str]:
    """Deterministic paragraph-packing into ~TEXT_CHUNK_CHARS chunks."""
    text = _normalize_text(text)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + TEXT_CHUNK_CHARS, n)
        if end < n:
            window = text[start:end]
            cut = window.rfind("\n\n")
            if cut >= int(len(window) * 0.7):  # keep the paragraph whole
                end = start + cut + 2
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    # merge a tail fragment too small to stand alone
    if len(chunks) > 1 and len(chunks[-1]) < TEXT_MIN_CHUNK_CHARS:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()
    return chunks


def text_corpus(sources: list[ResolvedSource]) -> dict[str, Any]:
    """Books/papers/code → extractable text → chunked training records.

    Reads .txt/.md/.rst, .pdf (needs pypdf), .docx (stdlib zip+XML), and
    nested .zip archives. Fail-closed: a source that yields NO extractable
    text BLOCKs with a named reason (scanned PDFs never silently pass —
    fabricating content would corrupt the dataset identity)."""
    records: list[dict[str, Any]] = []
    for src in sources:
        chunks_for_source = 0
        readable_seen = 0
        for entry in src.entries:
            if Path(entry.relative_path).suffix.lower() not in _READABLE_SUFFIXES:
                continue
            readable_seen += 1
            data = (src.path / entry.relative_path).read_bytes()
            text = _extract_unit(entry.relative_path, data)
            for i, chunk in enumerate(_chunk_text(text)):
                records.append({
                    "source": src.ref,
                    "split": src.split,
                    "relative_path": f"{entry.relative_path}#{i}",
                    "chars": len(chunk),
                    "text": chunk,
                })
                chunks_for_source += 1
        if chunks_for_source == 0:
            raise ValidationBlock(
                f"text_corpus: no extractable text in {src.ref} "
                f"({readable_seen} candidate file"
                f"{'s' if readable_seen != 1 else ''}) — image-only/scanned "
                "PDFs yield nothing and MLForge never fabricates content",
                hint="add text-based documents, or convert the scan "
                     "(OCR) before `mlforge prepare`",
            )
    return {
        "output_schema": "text_corpus.v1",
        "records": records,
    }


_REGISTRY: dict[str, Callable[[list[ResolvedSource]], dict[str, Any]]] = {
    "coco_detection": coco_detection,
    "text_corpus": text_corpus,
}


def register_transform(
    name: str, fn: Callable[[list[ResolvedSource]], dict[str, Any]], *,
    replace: bool = False,
) -> Callable[[list[ResolvedSource]], dict[str, Any]]:
    """Register a transform in the GLOBAL registry (open extension point).

    The registry is global across the installation — datasets of any type
    register here; nothing runs that is not registered (fail-closed).
    Identity = the function's own source hash, so a re-registered
    implementation invalidates prepare caches automatically."""
    import re

    if not re.fullmatch(r"[a-z][a-z0-9_]*", name or ""):
        raise ValidationBlock(
            f"invalid transform name {name!r} — use snake_case ([a-z][a-z0-9_]*)"
        )
    if not callable(fn):
        raise ValidationBlock(f"transform {name!r} must be callable")
    if name in _REGISTRY and not replace:
        raise PreconditionFailed(
            f"transform {name!r} is already registered",
            hint="pass replace=True only to intentionally override a "
                 "registered transform (identity changes invalidate caches)",
        )
    _REGISTRY[name] = fn
    return fn


def registry_names() -> list[str]:
    return sorted(_REGISTRY)


def get_transform(name: str) -> Callable[[list[ResolvedSource]], dict[str, Any]]:
    fn = _REGISTRY.get(name)
    if fn is None:
        raise ValidationBlock(
            f"unknown transform {name!r} (registry: {', '.join(registry_names())})",
            hint="fix `transform:` in ingestion.yaml — unregistered "
                 "transforms never run (fail-closed)",
        )
    return fn


def transform_identity(name: str, config: dict[str, Any] | None = None) -> dict[str, str]:
    """Four-field identity: name + code hash + config hash (12 §6.4)."""
    fn = get_transform(name)
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:  # source unavailable ⇒ identity unknown
        raise ValidationBlock(
            f"transform {name!r}: implementation source unavailable — "
            f"cannot compute code identity ({exc})"
        ) from exc
    return {
        "name": name,
        "code_hash": content_hash_bytes(source.encode("utf-8")),
        "config_hash": content_hash(config or {}),
    }


def env_fingerprint() -> str:
    """Environment identity — python + platform until the OCI digest lands
    (12 §6.4: environment_hash is a first-class cache-key field)."""
    return content_hash({
        "python": sys.version,
        "platform": platform.platform(),
    })


def cache_key(
    inputs: list[dict[str, str]],
    transform_identity_data: dict[str, str],
    environment: str,
) -> str:
    """cache_key = SHA256(input hashes + transform + config + env) (12 §6.4)."""
    return content_hash({
        "inputs": sorted(inputs, key=lambda d: d["ref"]),
        "transform": transform_identity_data,
        "environment": environment,
    })


def run_transform(
    plan: ModelPlan, sources: list[ResolvedSource]
) -> dict[str, Any]:
    """Execute + validate the transform output (fail-closed on empties)."""
    fn = get_transform(plan.transform)
    out = fn(sources)
    if not isinstance(out, dict) or "records" not in out:
        raise ValidationBlock(
            f"transform {plan.transform!r} returned no records field "
            f"(fail-closed: unverifiable output == failed output)"
        )
    records = out["records"]
    if not records:
        raise ValidationBlock(
            f"transform {plan.transform!r} produced 0 records — an empty "
            f"prepared dataset has no identity (fail-closed)"
        )
    # Every source must actually contribute (a silently skipped source
    # would make the artifact a guess).
    covered = {r.get("source") for r in records}
    missing = [s.ref for s in sources if s.ref not in covered]
    if missing:
        raise ValidationBlock(
            f"transform {plan.transform!r} produced no records for: "
            f"{', '.join(missing)}"
        )
    return out


__all__ = [
    "PREPARED_SCHEMA",
    "ResolvedSource",
    "cache_key",
    "coco_detection",
    "env_fingerprint",
    "get_transform",
    "register_transform",
    "registry_names",
    "run_transform",
    "text_corpus",
    "transform_identity",
]
