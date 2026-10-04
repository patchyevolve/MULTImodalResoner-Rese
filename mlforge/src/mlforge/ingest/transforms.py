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

import configparser
import csv
import inspect
import io
import json
import math
import platform
import re
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

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
    generated_from: str | None = None  # producing model (derived data, 12 §10.2)

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


# ---------------------------------------------------------------------------
# mot_challenge — MOTChallenge tracking CSV (02_dataset_preparation §3:
# MOT17, MOT20, SportsMOT). Columns are validated against the spec's
# 9-column layout, never guessed; other CSV layouts belong to `tabular`.
# ---------------------------------------------------------------------------

#: The 9-column format from 02_dataset_preparation §3 (order matters).
_MOT_COLUMNS = (
    "frame_id", "track_id", "bbox_left", "bbox_top",
    "bbox_width", "bbox_height", "confidence", "class", "visibility",
)
#: Real MOTChallenge `gt.txt` files spell several columns differently.
_MOT_HEADER_ALIASES = {
    "frame": "frame_id", "frame_id": "frame_id",
    "id": "track_id", "track_id": "track_id",
    "bb_left": "bbox_left", "bbox_left": "bbox_left",
    "bb_top": "bbox_top", "bbox_top": "bbox_top",
    "bb_width": "bbox_width", "bbox_width": "bbox_width",
    "bb_height": "bbox_height", "bbox_height": "bbox_height",
    "conf": "confidence", "confidence": "confidence",
    "class": "class", "cls": "class",
    "vis": "visibility", "visibility": "visibility",
}
#: Annotations ship as gt.txt / *.csv next to the frame images.
_MOT_SUFFIXES = frozenset({".csv", ".txt"})


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _looks_like_mot(text: str) -> bool:
    """First non-empty line decides — .txt readmes are skipped, not errors."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cells = [c.strip().lower() for c in line.split(",")]
        if all(c in _MOT_HEADER_ALIASES for c in cells):
            return True
        return len(cells) == len(_MOT_COLUMNS) and all(_is_number(c) for c in cells)
    return False


def _parse_mot_rows(text: str, label: str) -> list[list[str]]:
    """Parse + validate one MOTChallenge file (headered or headerless)."""
    raw = [(n, r) for n, r in enumerate(csv.reader(io.StringIO(text)), 1)
           if r and any(c.strip() for c in r)]
    if not raw:
        raise ValidationBlock(f"{label}: empty file — no MOT annotations")
    first = [c.strip().lower() for c in raw[0][1]]
    if all(c in _MOT_HEADER_ALIASES for c in first):
        header = tuple(_MOT_HEADER_ALIASES[c] for c in first)
        if header != _MOT_COLUMNS:
            raise ValidationBlock(
                f"{label}: unexpected header {list(first)} — MOTChallenge "
                f"needs {list(_MOT_COLUMNS)}",
                hint="use `transform: tabular` for a different CSV layout",
            )
        data = raw[1:]
    elif len(raw[0][1]) == len(_MOT_COLUMNS) and all(
        _is_number(c.strip()) for c in raw[0][1]
    ):
        data = raw
    else:
        raise ValidationBlock(
            f"{label}: first line is neither a MOTChallenge header nor "
            f"{len(_MOT_COLUMNS)} numeric columns",
            hint="use `transform: tabular` for generic tables",
        )
    validated: list[list[str]] = []
    for lineno, row in data:
        if len(row) != len(_MOT_COLUMNS):
            raise ValidationBlock(
                f"{label}: line {lineno}: expected {len(_MOT_COLUMNS)} columns "
                f"{list(_MOT_COLUMNS)}, got {len(row)}"
            )
        for cell, name in zip(row, _MOT_COLUMNS):
            text_cell = cell.strip()
            if name in {"frame_id", "track_id", "class"}:
                try:
                    value: float = int(text_cell)
                except ValueError:
                    raise ValidationBlock(
                        f"{label}: line {lineno}: {name} must be an integer, "
                        f"got {text_cell!r}"
                    ) from None
                if name == "frame_id" and value < 1:
                    raise ValidationBlock(
                        f"{label}: line {lineno}: frame ids start at 1, "
                        f"got {value}"
                    )
            elif not _is_number(text_cell):
                raise ValidationBlock(
                    f"{label}: line {lineno}: {name} must be a number, "
                    f"got {text_cell!r}"
                )
            elif name in {"bbox_width", "bbox_height"} and float(text_cell) < 0:
                raise ValidationBlock(
                    f"{label}: line {lineno}: {name} must be >= 0, "
                    f"got {text_cell}"
                )
            elif name == "visibility" and not 0.0 <= float(text_cell) <= 1.0:
                raise ValidationBlock(
                    f"{label}: line {lineno}: visibility is a 0..1 ratio "
                    f"(02 §3), got {text_cell}"
                )
        validated.append([c.strip() for c in row])
    if not validated:
        raise ValidationBlock(f"{label}: header only — no annotation rows")
    return validated


def mot_challenge(sources: list[ResolvedSource]) -> dict[str, Any]:
    """MOTChallenge tracking data: validated 9-column boxes per frame
    (MOT17 / MOT20 / SportsMOT, 02_dataset_preparation §3). Frame images
    ride along as provenance records; anything that is not gt-format BLOCKs
    with a named reason instead of being reinterpreted."""
    records: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for src in sources:
        parsed = 0
        for entry in src.entries:
            suffix = Path(entry.relative_path).suffix.lower()
            if suffix not in _MOT_SUFFIXES:
                continue
            raw = (src.path / entry.relative_path).read_text(
                encoding="utf-8-sig", errors="replace")
            if suffix == ".txt" and not _looks_like_mot(raw):
                continue  # readme.txt etc. — datasets carry stray files
            label = f"{src.ref}:{entry.relative_path}"
            rows = _parse_mot_rows(raw, label)
            parsed += 1
            annotations.append({
                "source": src.ref,
                "relative_path": entry.relative_path,
                "boxes": len(rows),
                "frames": len({r[0] for r in rows}),
                "tracks": len({r[1] for r in rows}),
                "canonical_sha256": content_hash(
                    {"columns": list(_MOT_COLUMNS), "rows": rows}),
            })
        if parsed == 0:
            raise ValidationBlock(
                f"mot_challenge: no MOTChallenge CSV in {src.ref} — expected "
                f"9-column {', '.join(_MOT_COLUMNS)} files (gt.txt / gt.csv)",
                hint="provide the dataset's annotations (02 §3), or choose "
                     "another transform: mlforge dataset types",
            )
        for entry in src.entries:  # every file = provenance record
            records.append({
                "source": src.ref,
                "split": src.split,
                "relative_path": entry.relative_path,
                "sha256": entry.sha256,
                "size": entry.size,
            })
    return {
        "output_schema": "mot_challenge.v1",
        "records": records,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# reid_crops — person re-ID image folders, Market1501 naming (12 §10.2;
# 02_dataset_preparation §5)
# ---------------------------------------------------------------------------

#: Market1501-style filename: 0002_c1s1_000151_01.jpg (identity_camera...).
_REID_NAME = re.compile(r"^(?P<pid>-?\d+)_c(?P<cam>\d+)", re.IGNORECASE)
_REID_IMAGES = frozenset({".jpg", ".jpeg", ".png"})
#: Top-level folders that carry split meaning (Market1501 + generic).
_REID_SPLITS = {
    "bounding_box_train": "train", "bounding_box_test": "test",
    "query": "query", "train": "train", "val": "val", "valid": "val",
    "test": "test",
}


def reid_crops(sources: list[ResolvedSource]) -> dict[str, Any]:
    """Person re-ID images in Market1501 naming ({identity}_{camera}*.jpg)
    → identity/camera manifest — the input the OSNet plan expects
    (12 §10.2). Wrong or absent naming BLOCKs; identity is never guessed
    from folder layout alone."""
    records: list[dict[str, Any]] = []
    identities: dict[str, dict[str, Any]] = {}
    for src in sources:
        recognized = 0
        image_count = 0
        for entry in src.entries:
            path = Path(entry.relative_path)
            if path.suffix.lower() not in _REID_IMAGES:
                continue
            image_count += 1
            match = _REID_NAME.match(path.name)
            if not match:
                continue
            recognized += 1
            pid = match.group("pid")
            cam = int(match.group("cam"))
            split = src.split
            if len(path.parts) > 1 and path.parts[0].lower() in _REID_SPLITS:
                split = _REID_SPLITS[path.parts[0].lower()]
            info = identities.setdefault(
                pid, {"identity": pid, "images": 0, "cameras": set()})
            info["images"] += 1
            info["cameras"].add(cam)
            records.append({
                "source": src.ref,
                "split": split,
                "relative_path": entry.relative_path,
                "identity": pid,
                "camera": cam,
                "sha256": entry.sha256,
                "size": entry.size,
            })
        if recognized == 0:
            detail = (
                f"no image files ({', '.join(sorted(_REID_IMAGES))})"
                if image_count == 0 else
                f"{image_count} image(s), none named {{identity}}_c{{camera}}*"
            )
            raise ValidationBlock(
                f"reid_crops: {detail} in {src.ref} — expected "
                "Market1501-style names (e.g. 0002_c1s1_000151_01.jpg, 02 §5)",
                hint="choose another transform: mlforge dataset types",
            )
    return {
        "output_schema": "reid_crops.v1",
        "records": records,
        "identities": [
            {"identity": info["identity"], "images": info["images"],
             "cameras": sorted(info["cameras"])}
            for _, info in sorted(identities.items(),
                                  key=lambda kv: int(kv[0]))
        ],
    }


# ---------------------------------------------------------------------------
# tabular — generic tables: CSV/TSV/JSONL/XLSX → strict row records.
# (The formats the dataset inventory leans on beyond images: MOT-style
# exports, logs, spreadsheets. Zero-dependency xlsx reader: zip + XML.)
# ---------------------------------------------------------------------------

_TABULAR_SUFFIXES = frozenset({".csv", ".tsv", ".jsonl", ".ndjson", ".xlsx"})
_XLSX_SHEET_RE = re.compile(r"xl/worksheets/sheet(\d+)\.xml$")
_XML_TAG = re.compile(r"\{[^}]*\}")


def _tabular_header(cells: list[str], label: str) -> list[str]:
    header = [c.strip() for c in cells]
    if not header or any(not h for h in header):
        raise ValidationBlock(f"{label}: header row has an empty column name")
    if len(set(header)) != len(header):
        dupes = sorted({h for h in header if header.count(h) > 1})
        raise ValidationBlock(
            f"{label}: duplicate column names {dupes} — columns must be unique")
    return header


def _read_delimited(text: str, delim: str, label: str) -> tuple[list[str], list[dict[str, str]]]:
    raw = [(n, r) for n, r in enumerate(csv.reader(io.StringIO(text)), 1)
           if r and any(c.strip() for c in r)]
    if not raw:
        raise ValidationBlock(f"{label}: empty table")
    header = _tabular_header(raw[0][1], label)
    rows: list[dict[str, str]] = []
    for lineno, row in raw[1:]:
        if len(row) != len(header):
            raise ValidationBlock(
                f"{label}: line {lineno}: expected {len(header)} columns "
                f"{header}, got {len(row)} — rows are strict, never padded")
        rows.append(dict(zip(header, row)))
    if not rows:
        raise ValidationBlock(f"{label}: header only — no data rows")
    return header, rows


def _read_jsonl(text: str, label: str) -> tuple[list[str], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationBlock(
                f"{label}: line {lineno} is not valid JSON: {exc}") from None
        if not isinstance(obj, dict):
            raise ValidationBlock(
                f"{label}: line {lineno} must be a JSON object, "
                f"got {type(obj).__name__}")
        rows.append({k: v if isinstance(v, str) else json.dumps(v, sort_keys=True)
                     for k, v in obj.items()})
    if not rows:
        raise ValidationBlock(f"{label}: no JSON records")
    return sorted({k for row in rows for k in row}), rows


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValidationBlock(
            f"xlsx: sharedStrings.xml corrupt: {exc}") from None
    out: list[str] = []
    for si in root:
        if _XML_TAG.sub("", si.tag) != "si":
            continue
        out.append("".join(
            t.text or "" for t in si.iter()
            if _XML_TAG.sub("", t.tag) == "t"))
    return out


def _xlsx_col(ref: str, fallback: int) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    if not letters:
        return fallback
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _read_xlsx(path: Path, label: str) -> list[tuple[str, list[str], list[dict[str, str]]]]:
    """Minimal stdlib reader: sheets, shared/inline strings, numbers, bools.
    Sparse cells become '' (sheets are sparse by nature); a data row WIDER
    than the header BLOCKs — that means row 1 wasn't the header."""
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValidationBlock(
            f"{label}: not a readable .xlsx file: {exc}") from None
    with zf:
        sheets = sorted(
            (int(m.group(1)), name) for name in zf.namelist()
             if (m := _XLSX_SHEET_RE.match(name)))
        if not sheets:
            raise ValidationBlock(f"{label}: no worksheets in the workbook")
        shared = _xlsx_shared_strings(zf)
        parsed: list[tuple[str, list[str], list[dict[str, str]]]] = []
        for _, sheet_name in sheets:
            display = Path(sheet_name).stem
            sheet_label = f"{label}#{display}"
            try:
                root = ET.fromstring(zf.read(sheet_name))
            except ET.ParseError as exc:
                raise ValidationBlock(
                    f"{sheet_label}: corrupt sheet XML: {exc}") from None
            grid: list[list[str]] = []
            for row_el in root.iter():
                if _XML_TAG.sub("", row_el.tag) != "row":
                    continue
                cells: dict[int, str] = {}
                next_col = 0
                for cell in row_el:
                    if _XML_TAG.sub("", cell.tag) != "c":
                        continue
                    col = _xlsx_col(cell.get("r", ""), next_col)
                    next_col = col + 1
                    kind = cell.get("t")
                    if kind == "inlineStr":
                        value = "".join(
                            x.text or "" for x in cell.iter()
                            if _XML_TAG.sub("", x.tag) == "t")
                    else:
                        v_el = next(
                            (x for x in cell
                             if _XML_TAG.sub("", x.tag) == "v"), None)
                        raw = "" if v_el is None or v_el.text is None else v_el.text
                        if kind == "s":
                            try:
                                value = shared[int(raw)]
                            except (ValueError, IndexError):
                                raise ValidationBlock(
                                    f"{sheet_label}: shared-string index "
                                    f"{raw!r} out of range — file corrupt"
                                ) from None
                        elif kind == "b":
                            value = ("TRUE" if raw.strip() in
                                     {"1", "true", "TRUE"} else "FALSE")
                        else:
                            value = raw
                    cells[col] = value
                if cells:
                    width = max(cells) + 1
                    grid.append([cells.get(i, "") for i in range(width)])
            if not grid:
                raise ValidationBlock(f"{sheet_label}: sheet has no rows")
            header = _tabular_header(grid[0], sheet_label)
            rows: list[dict[str, str]] = []
            for n, row in enumerate(grid[1:], start=2):
                if len(row) > len(header):
                    raise ValidationBlock(
                        f"{sheet_label}: row {n}: {len(row)} cells but the "
                        f"first row declares {len(header)} columns {header} "
                        "— a wider data row means row 1 is not the header")
                padded = row + [""] * (len(header) - len(row))
                rows.append(dict(zip(header, padded)))
            if not rows:
                raise ValidationBlock(
                    f"{sheet_label}: header only — no data rows")
            parsed.append((display, header, rows))
        return parsed


def tabular(sources: list[ResolvedSource]) -> dict[str, Any]:
    """Generic tables → strict row records. CSV/TSV rows must match the
    header exactly (never silently padded), JSONL lines must be objects,
    XLSX is parsed with the stdlib (zip + XML). Every row keeps its
    source + line so the prepared artifact stays traceable."""
    records: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    for src in sources:
        candidates = 0
        rows_here = 0
        for entry in src.entries:
            suffix = Path(entry.relative_path).suffix.lower()
            if suffix not in _TABULAR_SUFFIXES:
                continue
            candidates += 1
            label = f"{src.ref}:{entry.relative_path}"
            if suffix == ".xlsx":
                for display, header, rows in _read_xlsx(
                        src.path / entry.relative_path, label):
                    rel = f"{entry.relative_path}#{display}"
                    tables.append({
                        "source": src.ref, "relative_path": rel,
                        "columns": header, "rows": len(rows),
                        "canonical_sha256": content_hash(rows),
                    })
                    for i, row in enumerate(rows):
                        records.append({
                            "source": src.ref, "split": src.split,
                            "relative_path": rel, "row": i, "data": row,
                        })
                        rows_here += 1
                continue
            text = (src.path / entry.relative_path).read_text(
                encoding="utf-8-sig", errors="replace")
            if suffix in {".jsonl", ".ndjson"}:
                header, rows = _read_jsonl(text, label)
            else:
                header, rows = _read_delimited(
                    text, "\t" if suffix == ".tsv" else ",", label)
            tables.append({
                "source": src.ref,
                "relative_path": entry.relative_path,
                "columns": header,
                "rows": len(rows),
                "canonical_sha256": content_hash(rows),
            })
            for i, row in enumerate(rows):
                records.append({
                    "source": src.ref, "split": src.split,
                    "relative_path": entry.relative_path, "row": i,
                    "data": row,
                })
                rows_here += 1
        if rows_here == 0:
            raise ValidationBlock(
                f"tabular: "
                f"{'table files found but no data rows' if candidates else 'no table files'}"
                f" in {src.ref} — expected "
                f"{', '.join(sorted(_TABULAR_SUFFIXES))}",
                hint="choose another transform: mlforge dataset types",
            )
    return {
        "output_schema": "tabular.v1",
        "records": records,
        "tables": tables,
    }


#: The five confidence-decomposition components (06 §1 formula names) —
#: a `components` object in a prediction record must be exactly this set.
_CAL_COMPONENTS: tuple[str, ...] = (
    "perception", "temporal", "motion", "cross_modal_agreement", "reasoning",
)


def calibration(sources: list[ResolvedSource]) -> dict[str, Any]:
    """Prediction records → calibrator training rows (C5, 13 §Model 5).

    Reads `.jsonl`/`.ndjson` (one prediction per line) or a `.json`
    array, each row `{logits|probs, label}` plus optional `components`
    (the five decomposition scores, 06 §1). `probs` must sum to ~1 and
    are converted to log-probabilities (softmax-invariant — temperature
    scaling on them is exact). Every file must agree on the class count:
    one calibration pool is one label space, never a mix."""
    import math

    records: list[dict[str, Any]] = []
    classes_seen: set[int] = set()
    for src in sources:
        for entry in src.entries:
            suffix = Path(entry.relative_path).suffix.lower()
            if suffix not in {".jsonl", ".ndjson", ".json"}:
                continue
            label = f"{src.ref}:{entry.relative_path}"
            text = (src.path / entry.relative_path).read_text(
                encoding="utf-8-sig", errors="replace")
            if suffix == ".json":
                try:
                    rows = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValidationBlock(
                        f"{label}: not valid JSON: {exc}") from None
                if not isinstance(rows, list):
                    raise ValidationBlock(
                        f"{label}: expected a JSON array of prediction "
                        f"objects, got {type(rows).__name__} — one "
                        "prediction per element",
                    )
                enumerated = list(enumerate(rows, 1))
            else:
                enumerated = []
                for lineno, line in enumerate(text.splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValidationBlock(
                            f"{label}: line {lineno} is not valid JSON: "
                            f"{exc}") from None
                    enumerated.append((lineno, obj))
            for lineno, obj in enumerated:
                where = f"{label}: {'line' if suffix != '.json' else 'element'} {lineno}"
                if not isinstance(obj, dict):
                    raise ValidationBlock(
                        f"{where} must be a JSON object with `logits` "
                        f"(or `probs`) and `label`, got "
                        f"{type(obj).__name__}")
                if "logits" in obj:
                    raw = obj["logits"]
                    if not isinstance(raw, list) or len(raw) < 2:
                        raise ValidationBlock(
                            f"{where}: `logits` must be a list of at "
                            "least 2 numbers (a 1-class score has "
                            "nothing to calibrate)")
                    try:
                        logits = [float(v) for v in raw]
                    except (TypeError, ValueError) as exc:
                        raise ValidationBlock(
                            f"{where}: `logits` must be numbers: {exc}"
                        ) from None
                elif "probs" in obj:
                    raw = obj["probs"]
                    if not isinstance(raw, list) or len(raw) < 2:
                        raise ValidationBlock(
                            f"{where}: `probs` must be a list of at "
                            "least 2 numbers")
                    try:
                        probs = [float(v) for v in raw]
                    except (TypeError, ValueError) as exc:
                        raise ValidationBlock(
                            f"{where}: `probs` must be numbers: {exc}"
                        ) from None
                    if any(not math.isfinite(p) or p < 0 for p in probs):
                        raise ValidationBlock(
                            f"{where}: `probs` must be finite and >= 0")
                    total = sum(probs)
                    if abs(total - 1.0) > 1e-3:
                        raise ValidationBlock(
                            f"{where}: `probs` sum to {total!r}, not 1 — "
                            "send a probability vector or raw `logits`",
                        )
                    # ln(p) with a floor for zeros — softmax-invariant.
                    logits = [math.log(max(p, 1e-12)) for p in probs]
                else:
                    raise ValidationBlock(
                        f"{where}: needs `logits` (preferred) or `probs` "
                        "— a prediction without scores cannot calibrate "
                        "confidence",
                    )
                if any(not math.isfinite(v) for v in logits):
                    raise ValidationBlock(
                        f"{where}: logits contain a non-finite value")
                lab = obj.get("label")
                if isinstance(lab, bool) or not isinstance(lab, (int, float)) \
                        or int(lab) != lab:
                    raise ValidationBlock(
                        f"{where}: `label` must be an integer class index, "
                        f"got {lab!r}")
                lab = int(lab)
                if not 0 <= lab < len(logits):
                    raise ValidationBlock(
                        f"{where}: label {lab} out of range for "
                        f"{len(logits)} classes")
                components = obj.get("components")
                if components is not None:
                    if not isinstance(components, dict):
                        raise ValidationBlock(
                            f"{where}: `components` must be an object "
                            f"with exactly {', '.join(_CAL_COMPONENTS)}")
                    missing = [c for c in _CAL_COMPONENTS
                               if c not in components]
                    extra = sorted(set(components) - set(_CAL_COMPONENTS))
                    if missing or extra:
                        raise ValidationBlock(
                            f"{where}: `components` mismatch — missing "
                            f"{missing or '[]'}, unknown {extra or '[]'} "
                            f"(exactly: {', '.join(_CAL_COMPONENTS)})",
                        )
                    try:
                        comp = {c: float(components[c])
                                for c in _CAL_COMPONENTS}
                    except (TypeError, ValueError) as exc:
                        raise ValidationBlock(
                            f"{where}: component scores must be numbers: "
                            f"{exc}") from None
                    if any(not math.isfinite(v) for v in comp.values()):
                        raise ValidationBlock(
                            f"{where}: component scores must be finite")
                else:
                    comp = None
                classes_seen.add(len(logits))
                rec: dict[str, Any] = {
                    "source": src.ref,
                    "split": src.split,
                    "relative_path": entry.relative_path,
                    "row": lineno,
                    "logits": logits,
                    "label": lab,
                }
                if comp is not None:
                    rec["components"] = comp
                records.append(rec)
    if not records:
        raise ValidationBlock(
            "calibration: no prediction records — expected `.jsonl` lines "
            "or a `.json` array of objects with `logits`|`probs` + "
            "`label` (optionally `components`)",
            hint="generate predictions on the held-out split with the "
                 "eval step, then register them: "
                 "`mlforge dataset add <ID> <PATH>` — derived data uses "
                 "{generated_from: <model>, dataset: <ID>} (12 §10.2)",
        )
    if len(classes_seen) > 1:
        raise ValidationBlock(
            f"calibration: class count differs across files "
            f"({sorted(classes_seen)}) — one calibration pool is one "
            "label space; split the mixed predictions into separate "
            "datasets",
        )
    return {"output_schema": "calibration.v1", "records": records}


# ---------------------------------------------------------------------------
# video_clips — bounded decode probe for video files + canonical re-hash of
# COCO sidecars (02_dataset_preparation §11 custom clips, §8 Celeb-DF++,
# §9 FaceForensics++; §15 quality checks: "All clips play without errors",
# "Annotations in COCO format")
# ---------------------------------------------------------------------------

#: Container suffixes that carry video (02 §11 clip collections; §8/§9
#: datasets ship .mp4/.mkv mixes). Module scope so the probe, the
#: validation hint and the suggestion table can never disagree.
_VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm",
                             ".mpeg", ".mpg"})
#: Bounded decode sample: enough frames to prove the file actually plays,
#: never a full decode (a 2-hour broadcast must not dominate `prepare`).
_PROBE_VIDEO_FRAMES = 8
#: One reason every decode failure blocks — 02 §15, verbatim.
_CLIP_PLAY_HINT = ("clips must play without errors "
                   "(02_dataset_preparation §15) — re-encode the file or "
                   "drop it from the source")


def _probe_video_av(path: Path, label: str) -> dict[str, Any]:
    """PyAV probe: container metadata + a bounded decode sample.

    PyAV is the sanctioned decoder (01_pre_training_preparation §2.2,
    line 75: `pip install av`); opencv-python-headless is the fallback."""
    import av

    try:
        with av.open(str(path)) as container:
            stream = next(
                (s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise ValidationBlock(
                    f"video_clips: {label} has no video stream",
                    hint=_CLIP_PLAY_HINT,
                )
            meta = {
                "codec": (stream.codec_context.name
                          if stream.codec_context is not None else None),
                "width": int(stream.width or 0),
                "height": int(stream.height or 0),
                "fps": (round(float(stream.average_rate), 3)
                        if stream.average_rate else None),
                "duration_s": (round(container.duration / av.time_base, 3)
                               if container.duration else None),
                "frames": int(stream.frames) if stream.frames else None,
                "decoder": "av",
            }
            decoded = 0
            # `streams=` takes absolute stream indices (never type hints)
            for _frame in container.decode(streams=[stream.index]):
                decoded += 1
                if decoded >= _PROBE_VIDEO_FRAMES:
                    break
    except (av.FFmpegError, OSError) as exc:  # corrupt/undecodable ⇒ block
        raise ValidationBlock(
            f"video_clips: {label} does not play — {exc}",
            hint=_CLIP_PLAY_HINT,
        ) from exc
    if decoded == 0:
        raise ValidationBlock(
            f"video_clips: {label} decoded 0 frames — nothing to train on",
            hint=_CLIP_PLAY_HINT,
        )
    return meta


def _probe_video_cv2(path: Path, label: str) -> dict[str, Any]:
    """OpenCV fallback probe — used only when PyAV is not importable
    (01 §2.2: opencv-python-headless is the sanctioned video I/O dep)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValidationBlock(
                f"video_clips: {label} does not play — cv2 cannot open it",
                hint=_CLIP_PLAY_HINT,
            )
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF)
                        for i in range(4)).strip() or None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        meta = {
            "codec": codec,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": round(float(fps_raw), 3) if fps_raw else None,
            # cv2 exposes no container duration — never invented
            "duration_s": None,
            "frames": total or None,
            "decoder": "cv2",
        }
        decoded = 0
        while decoded < _PROBE_VIDEO_FRAMES:
            ok, _frame = cap.read()
            if not ok:
                break
            decoded += 1
    except ValidationBlock:
        raise
    except Exception as exc:  # cv2.error / OS failures ⇒ fail-closed
        raise ValidationBlock(
            f"video_clips: {label} does not play — {exc}",
            hint=_CLIP_PLAY_HINT,
        ) from exc
    finally:
        cap.release()
    if decoded == 0:
        raise ValidationBlock(
            f"video_clips: {label} decoded 0 frames — nothing to train on",
            hint=_CLIP_PLAY_HINT,
        )
    return meta


def _video_probe(path: Path, label: str) -> dict[str, Any]:
    """Probe with PyAV first; opencv only if PyAV is NOT importable;
    neither ⇒ PreconditionFailed naming the sanctioned install (01 §75)."""
    try:
        import av  # noqa: F401  — availability check only
    except ImportError:
        pass
    else:
        return _probe_video_av(path, label)
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        raise PreconditionFailed(
            "video_clips needs a video decoder — neither PyAV nor opencv "
            "is importable",
            hint="pip install av — video decoding dependency "
                 "(01_pre_training_preparation §2.2, line 75)",
        ) from exc
    return _probe_video_cv2(path, label)


def video_clips(sources: list[ResolvedSource]) -> dict[str, Any]:
    """Videos → decode-proved clip probes; COCO `.json` sidecars →
    canonical re-hash (02_dataset_preparation §11 custom clips, §8
    Celeb-DF++, §9 FaceForensics++; §15: "All clips play without errors",
    "Annotations in COCO format").

    Every candidate video is probed for container metadata and a BOUNDED
    sample (≤8 frames) is really decoded — a file that opens but does not
    play BLOCKs with a named reason (never a silent skip). `.json`
    sidecars must parse (and, when they look like COCO, carry list-valued
    `images`/`annotations`) — parsing is proven by the canonical re-hash,
    exactly like `coco_detection`; referenced image files need not exist
    yet (materialization resolves them later). The split comes from the
    source ref — a clip tree never invents one. A source with neither a
    video nor a sidecar BLOCKs with the accepted extensions, so an empty
    or mispointed path can never pass as a prepared dataset."""
    records: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for src in sources:
        videos = 0
        sidecars = 0
        for entry in src.entries:
            suffix = Path(entry.relative_path).suffix.lower()
            label = f"{src.ref}:{entry.relative_path}"
            base: dict[str, Any] = {
                "source": src.ref,
                "split": src.split,
                "relative_path": entry.relative_path,
                "sha256": entry.sha256,
                "size": entry.size,
            }
            if suffix in _VIDEO_SUFFIXES:
                videos += 1
                records.append({
                    **base,
                    "video": _video_probe(
                        src.path / entry.relative_path, label),
                })
            elif suffix == ".json":
                sidecars += 1
                raw = (src.path / entry.relative_path).read_text(
                    encoding="utf-8")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValidationBlock(
                        f"annotation does not parse as JSON: {label}: {exc}",
                        hint="annotations must be COCO-format JSON "
                             "(02_dataset_preparation §11, §15)",
                    ) from exc
                if (isinstance(parsed, dict) and "images" in parsed
                        and "annotations" in parsed):
                    if (not isinstance(parsed["images"], list)
                            or not isinstance(parsed["annotations"], list)):
                        raise ValidationBlock(
                            f"{label}: COCO annotation shape invalid — "
                            "`images` and `annotations` must both be lists",
                            hint="annotations in COCO format "
                                 "(02_dataset_preparation §11, §15)",
                        )
                records.append(base)
                annotations.append({
                    "source": src.ref,
                    "relative_path": entry.relative_path,
                    "canonical_sha256": content_hash(parsed),
                })
        if videos == 0 and sidecars == 0:
            raise ValidationBlock(
                f"video_clips: no video files and no COCO annotation .json "
                f"in {src.ref}",
                hint=f"accepted video extensions: "
                     f"{', '.join(sorted(_VIDEO_SUFFIXES))}, or COCO "
                     "annotation .json — `mlforge dataset types` lists the "
                     "other transforms",
            )
    return {
        "output_schema": "video_clips.v1",
        "records": records,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# soccernet_events — SoccerNet v2 action spotting (02_dataset_preparation
# §7: league/season/game layout with Labels-v2.json, video.ini,
# Labels-cameras.json; §15: "Labels-v2.json parseable for all 500 games")
# ---------------------------------------------------------------------------

#: The 17 action classes, verbatim from 02_dataset_preparation §7 —
#: a label outside this set is a spec violation, never silently dropped.
_SOCCERNET_CLASSES: tuple[str, ...] = (
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside",
    "Shots on target", "Shots off target", "Clearance",
    "Ball out of play", "Throw-in", "Foul", "Indirect free-kick",
    "Direct free-kick", "Corner", "Yellow card", "Red card",
    "Yellow→red card",
)
_SOCCERNET_LABELS = "Labels-v2.json"
_SOCCERNET_CAMERAS = "Labels-cameras.json"
_SOCCERNET_INI = "video.ini"
#: The three artifacts that identify a SoccerNet tree (02 §7) — a source
#: holding none of them is not SoccerNet data and must not be "prepared".
_SOCCERNET_ARTIFACTS = frozenset(
    {_SOCCERNET_LABELS, _SOCCERNET_CAMERAS, _SOCCERNET_INI})


def _parse_labels_v2(raw: str, label: str) -> list[Any]:
    """Strict parse of one Labels-v2.json → its annotation objects
    (02 §15: parseable for all 500 games; §7 for the expected shape)."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationBlock(
            f"{label}: Labels-v2.json does not parse as JSON: {exc}",
            hint="02_dataset_preparation §15 — Labels-v2.json must be "
                 "parseable for all 500 games",
        ) from exc
    if not isinstance(parsed, dict) or not isinstance(
            parsed.get("annotations"), list):
        raise ValidationBlock(
            f"{label}: expected an object with an `annotations` list of "
            f'{{"position": <ms>, "label": <class>}} entries, got '
            f"{type(parsed).__name__}",
            hint='SoccerNet Labels-v2.json shape: {"annotations": '
                 '[{"position": <number ms>, "label": <class str>}, ...]} '
                 "(02_dataset_preparation §7)",
        )
    return parsed["annotations"]


def _validate_action(
    index: int, action: Any, label: str,
) -> tuple[float, str]:
    """One annotation → (position_ms, label) or a named ValidationBlock."""
    where = f"{label}: annotation {index}"
    if not isinstance(action, dict):
        raise ValidationBlock(
            f"{where} must be a JSON object, got {type(action).__name__}")
    position = action.get("position")
    if isinstance(position, bool) or not isinstance(position, (int, float)):
        raise ValidationBlock(
            f"{where}: `position` must be a number of milliseconds, "
            f"got {position!r}",
            hint="02_dataset_preparation §7 — action spots are "
                 "`position` timestamps in ms",
        )
    if not math.isfinite(position):
        raise ValidationBlock(
            f"{where}: `position` must be finite, got {position!r}")
    name = action.get("label")
    if not isinstance(name, str):
        raise ValidationBlock(
            f"{where}: `label` must be a string, got {name!r}")
    if name not in _SOCCERNET_CLASSES:
        raise ValidationBlock(
            f"{label}: unknown action label {name!r} — the 17 SoccerNet "
            f"classes are: {', '.join(_SOCCERNET_CLASSES)}",
            hint="02_dataset_preparation §7 lists the exact class set — "
                 "unknown labels never pass (fail-closed)",
        )
    return float(position), name


def _parse_video_ini(raw: str, label: str) -> dict[str, dict[str, str]]:
    """video.ini → {section: {key: value}} with raw string values
    (02 §7: start/duration per half must be readable)."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # keys stay raw — no case folding
    try:
        parser.read_string(raw)
    except configparser.Error as exc:
        raise ValidationBlock(
            f"{label}: video.ini does not parse: {exc}",
            hint="02_dataset_preparation §7 — video.ini carries "
                 "start/duration per half and must be readable",
        ) from exc
    return {name: dict(parser[name]) for name in parser.sections()}


def soccernet_events(sources: list[ResolvedSource]) -> dict[str, Any]:
    """SoccerNet v2 → action-spot events + one per-game summary record
    (02_dataset_preparation §7 directory layout, Labels-v2.json,
    video.ini, Labels-cameras.json; §15 "Labels-v2.json parseable for
    all 500 games").

    Only the three named artifacts are parsed: the 17-class check is
    exact (§7 — unknown labels BLOCK, never dropped), positions must be
    finite numbers of milliseconds, and video.ini must be readable.
    Labels-cameras.json is strictly parsed and canonically re-hashed but
    its INTERNAL camera schema is not validated — the spec names no
    schema for the 13 transition types (§7 table), so guessing one would
    be a guess. Videos and `.npy` features ride along as byte-provenance
    records only: decoding belongs to `video_clips`, and SoccerNet ships
    as "features + labels" (§13 budget) with videos behind an NDA — no
    numpy import here (stdlib only). A source with no recognized
    artifact at all BLOCKs with the §7 layout instead of silently
    "preparing" an unrelated tree."""
    records: list[dict[str, Any]] = []
    for src in sources:
        recognized = 0
        for entry in src.entries:
            relative = entry.relative_path
            name = Path(relative).name
            suffix = Path(relative).suffix.lower()
            base: dict[str, Any] = {
                "source": src.ref,
                "split": src.split,
                "relative_path": relative,
                "sha256": entry.sha256,
                "size": entry.size,
            }
            if name in _SOCCERNET_ARTIFACTS:
                recognized += 1
            if name == _SOCCERNET_LABELS:
                label = f"{src.ref}:{relative}"
                actions = _parse_labels_v2(
                    (src.path / relative).read_text(encoding="utf-8"), label)
                validated = [
                    _validate_action(i, action, label)
                    for i, action in enumerate(actions)
                ]
                game = relative[: -len(_SOCCERNET_LABELS)].rstrip("/")
                for position, action_label in validated:
                    records.append({
                        **base,
                        "kind": "action_spot",
                        "game": game,
                        "position_ms": position,
                        "label": action_label,
                    })
                records.append({
                    **base,
                    "kind": "game",
                    "game": game,
                    "events": len(validated),
                    "classes": sorted({a for _, a in validated}),
                })
            elif name == _SOCCERNET_INI:
                records.append({
                    **base,
                    "kind": "video_ini",
                    "sections": _parse_video_ini(
                        (src.path / relative).read_text(encoding="utf-8"),
                        f"{src.ref}:{relative}"),
                })
            elif name == _SOCCERNET_CAMERAS:
                label = f"{src.ref}:{relative}"
                raw = (src.path / relative).read_text(encoding="utf-8")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValidationBlock(
                        f"camera labels do not parse as JSON: {label}: {exc}",
                        hint="02_dataset_preparation §7 — "
                             "Labels-cameras.json ships with every game",
                    ) from exc
                camera: dict[str, Any] = {
                    **base,
                    "kind": "camera_labels",
                    "canonical_sha256": content_hash(parsed),
                }
                if (isinstance(parsed, dict)
                        and isinstance(parsed.get("annotations"), list)):
                    camera["annotations_count"] = len(parsed["annotations"])
                records.append(camera)
            elif suffix in _VIDEO_SUFFIXES:
                records.append({**base, "kind": "video_file"})
            elif suffix == ".npy":
                records.append({**base, "kind": "features"})
            else:
                records.append({**base, "kind": "file"})
        if recognized == 0:
            raise ValidationBlock(
                f"soccernet_events: no recognized SoccerNet artifact in "
                f"{src.ref} — no Labels-v2.json, video.ini or "
                "Labels-cameras.json found",
                hint="expected SoccerNet layout per 02_dataset_preparation "
                     "§7: <league>/<season>/<game>/Labels-v2.json "
                     "(+ video.ini, Labels-cameras.json, *_720p.mkv)",
            )
    return {"output_schema": "soccernet_events.v1", "records": records}


#: The user-facing catalog — what `mlforge dataset types` prints. Every
#: `name` MUST be a registered transform (invariant covered by tests).
DATASET_TYPES: tuple[dict[str, str], ...] = (
    {"name": "text_corpus",
     "title": "Documents",
     "inputs": ".pdf .docx .md .txt .rst .zip",
     "produces": "extracted, chunked text (books, papers, code, notes)"},
    {"name": "coco_detection",
     "title": "COCO detection",
     "inputs": "images + COCO annotation .json",
     "produces": "file records + re-hashed annotations (COCO 2017)"},
    {"name": "mot_challenge",
     "title": "MOT tracking",
     "inputs": "MOTChallenge .csv/.txt (frame,track,bbox,...) + frames",
     "produces": "validated per-frame boxes (MOT17, MOT20, SportsMOT)"},
    {"name": "reid_crops",
     "title": "Person re-ID",
     "inputs": "{identity}_{camera}*.jpg folders",
     "produces": "identity/camera manifest (Market1501)"},
    {"name": "tabular",
     "title": "Tables",
     "inputs": ".csv .tsv .jsonl .xlsx",
     "produces": "strict row records (exports, logs, spreadsheets)"},
    {"name": "calibration",
     "title": "Model predictions",
     "inputs": "prediction .jsonl/.json array {logits|probs, label}",
     "produces": "calibrator rows — temperature + conformal (C5)"},
    {"name": "video_clips",
     "title": "Video clips",
     "inputs": ".mp4 .mkv .avi .mov .webm + COCO .json",
     "produces": "decoded/validated clip probes + re-hashed COCO sidecars "
                 "(Celeb-DF++, FF++, custom clips)"},
    {"name": "soccernet_events",
     "title": "SoccerNet action spotting",
     "inputs": "SoccerNet Labels-v2.json / video.ini / videos",
     "produces": "17-class action-spot event records (SoccerNet v2)"},
)

#: Honest gaps — named so users see them, never faked (fail-closed).
PLANNED_DATASET_TYPES: tuple[dict[str, str], ...] = (
    {"name": "audio", "title": "Audio datasets",
     "reason": "AudioSet — needs feature extraction (later build step)"},
    {"name": "generated", "title": "Generated sources",
     "reason": "the {generated_from: <model>, dataset: <id>} source form "
               "resolves against the model registry and the `calibration` "
               "transform consumes prediction records today (12 §10.2); "
               "PRODUCING those predictions is the eval/infer step "
               "(later build step)"},
)


def dataset_types() -> dict[str, list[dict[str, str]]]:
    """Catalog for `mlforge dataset types`: what exists + honest gaps."""
    return {"supported": list(DATASET_TYPES),
            "planned": list(PLANNED_DATASET_TYPES)}


#: File extension → transforms that read it (for post-`add` suggestions).
_EXT_TRANSFORMS: dict[str, tuple[str, ...]] = {
    ".txt": ("text_corpus",), ".md": ("text_corpus",),
    ".rst": ("text_corpus",), ".pdf": ("text_corpus",),
    ".docx": ("text_corpus",),
    ".csv": ("mot_challenge", "tabular"),
    ".tsv": ("tabular",), ".jsonl": ("tabular", "calibration"),
    ".ndjson": ("tabular", "calibration"), ".xlsx": ("tabular",),
    ".mp4": ("video_clips",), ".mkv": ("video_clips", "soccernet_events"),
    ".avi": ("video_clips",), ".mov": ("video_clips",),
    ".webm": ("video_clips",), ".ini": ("soccernet_events",),
}
_IMG_SUGGEST = frozenset({".jpg", ".jpeg", ".png"})


def suggest_transforms(relative_paths: Any) -> list[str]:
    """Best-effort transform candidates from file names — a HINT printed
    after `dataset add`, never a decision (the model plan in ingestion.yaml
    stays the user's explicit declaration)."""
    found: set[str] = set()
    have_json = False
    have_reid_image = False
    have_image = False
    for rel in relative_paths:
        path = Path(str(rel))
        suffix = path.suffix.lower()
        found.update(_EXT_TRANSFORMS.get(suffix, ()))
        if suffix == ".json":
            have_json = True
        if suffix in _IMG_SUGGEST:
            have_image = True
            if _REID_NAME.match(path.name):
                have_reid_image = True
    if have_reid_image:
        found.add("reid_crops")
    elif have_json and have_image:
        found.add("coco_detection")
    order = [t["name"] for t in DATASET_TYPES]
    return sorted(found, key=order.index)


_REGISTRY: dict[str, Callable[[list[ResolvedSource]], dict[str, Any]]] = {
    "coco_detection": coco_detection,
    "text_corpus": text_corpus,
    "mot_challenge": mot_challenge,
    "reid_crops": reid_crops,
    "tabular": tabular,
    "calibration": calibration,
    "video_clips": video_clips,
    "soccernet_events": soccernet_events,
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
                 "transforms never run (fail-closed); `mlforge dataset "
                 "types` lists every supported type",
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
    "DATASET_TYPES",
    "PLANNED_DATASET_TYPES",
    "PREPARED_SCHEMA",
    "ResolvedSource",
    "cache_key",
    "calibration",
    "coco_detection",
    "dataset_types",
    "env_fingerprint",
    "get_transform",
    "mot_challenge",
    "register_transform",
    "registry_names",
    "reid_crops",
    "run_transform",
    "soccernet_events",
    "suggest_transforms",
    "tabular",
    "text_corpus",
    "transform_identity",
    "video_clips",
]
