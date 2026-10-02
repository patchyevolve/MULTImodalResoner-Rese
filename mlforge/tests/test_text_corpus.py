"""Text ingestion: `text_corpus` transform + the open transform registry
(build step 8 — ingestion/transform DAG, 12 §6.4, 13 §10.2)."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from mlforge.errors import PreconditionFailed, ValidationBlock
from mlforge.ingest.dag import ModelPlan, Source
from mlforge.ingest.identity import FileEntry
from mlforge.ingest.transforms import (
    ResolvedSource,
    get_transform,
    register_transform,
    registry_names,
    run_transform,
    text_corpus,
)


def make_source(tmp_path, files: dict[str, bytes], ref: str = "books:train",
                split: str | None = "train") -> ResolvedSource:
    base = tmp_path / "src"
    base.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, data in files.items():
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        entries.append(FileEntry(
            relative_path=name, sha256="x" * 64, size=len(data),
        ))
    return ResolvedSource(
        ref=ref, split=split, dataset_id="books", path=base,
        identity="sha256:y", entries=tuple(entries),
    )


def test_text_corpus_chunks_plain_text(tmp_path):
    para = "MLForge trains models honestly. " * 30  # ~960 chars
    src = make_source(tmp_path, {
        "a.txt": (para + "\n\n" + para).encode(),
        "b.md": ("# Heading\n\n" + para).encode(),
    })
    out = text_corpus([src])
    assert out["output_schema"] == "text_corpus.v1"
    records = out["records"]
    assert records, "must produce records"
    assert all(r["source"] == "books:train" for r in records)
    assert all(r["text"].strip() for r in records)
    assert all(r["chars"] == len(r["text"]) for r in records)
    # deterministic: same input → same output
    assert text_corpus([src]) == out


def test_text_corpus_reads_docx(tmp_path):
    doc_xml = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<w:document xmlns:w='w'><w:body>"
        "<w:p><w:r><w:t>Docx paragraph one.</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Docx paragraph two.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)
    src = make_source(tmp_path, {"book.docx": buf.getvalue()})
    out = text_corpus([src])
    text = "\n".join(r["text"] for r in out["records"])
    assert "Docx paragraph one." in text
    assert "Docx paragraph two." in text


def test_text_corpus_reads_zip_archive(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("notes/inside.txt", "Zipped chapter text. " * 30)
    src = make_source(tmp_path, {"bundle.zip": inner.getvalue()})
    out = text_corpus([src])
    text = "\n".join(r["text"] for r in out["records"])
    assert "Zipped chapter text." in text


def test_text_corpus_reads_pdf(tmp_path):
    # minimal single-page PDF with an extractable text run
    stream = b"BT /F1 12 Tf 72 720 Td (Hello corpus from pdf) Tj ET"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(body))
        body += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(body)
    body += f"xref\n0 {len(objs) + 1}\n".encode()
    body += b"0000000000 65535 f \n"
    for off in offsets:
        body += f"{off:010d} 00000 n \n".encode()
    body += (
        b"trailer\n<< /Size " + str(len(objs) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode()
        + b"\n%%EOF"
    )
    src = make_source(tmp_path, {"paper.pdf": body})
    out = text_corpus([src])
    text = "\n".join(r["text"] for r in out["records"])
    assert "Hello corpus from pdf" in text


def test_text_corpus_blocks_when_source_yields_nothing(tmp_path):
    src = make_source(tmp_path, {"photo.png": b"\x89PNG\r\n\x1a\n" * 10})
    with pytest.raises(ValidationBlock) as exc:
        text_corpus([src])
    assert "no extractable text in books:train" in str(exc.value)


def test_text_corpus_blocks_corrupt_pdf(tmp_path):
    src = make_source(tmp_path, {"scan.pdf": b"not a pdf at all"})
    with pytest.raises(ValidationBlock) as exc:
        text_corpus([src])
    assert "PDF unreadable" in str(exc.value)


# -- run_transform's fail-closed envelope (shared machinery) ---------------

def test_run_transform_covers_every_source(tmp_path):
    good = make_source(
        tmp_path, {"a.txt": ("word " * 200).encode()}, ref="books:train",
    )
    plan = ModelPlan(name="reasoner_s", transform="text_corpus",
                     train_sources=(Source(dataset="books", split="train"),))
    # second source with NO contribution (mixed content)
    empty = make_source(tmp_path, {"photo.png": b"png"}, ref="more:train",
                        split=None)
    with pytest.raises(ValidationBlock) as exc:
        run_transform(plan, [good, empty])
    # the transform's own fail-closed message fires first (named source)
    assert "no extractable text in more:train" in str(exc.value)


def test_unknown_transform_still_blocks():
    with pytest.raises(ValidationBlock) as exc:
        get_transform("definitely_not_registered")
    assert "unknown transform" in str(exc.value)


# -- the OPEN, global registry ---------------------------------------------

def test_register_transform_rejects_bad_names():
    with pytest.raises(ValidationBlock):
        register_transform("Bad Name", lambda s: {"records": []})
    with pytest.raises(ValidationBlock):
        register_transform("ok_name", "not-callable")


def test_register_transform_duplicate_needs_replace():
    fn = lambda s: {"records": [{"source": s[0].ref, "text": "x"}]}
    register_transform("unit_probe_tf", fn)
    assert "unit_probe_tf" in registry_names()
    with pytest.raises(PreconditionFailed):
        register_transform("unit_probe_tf", fn)
    register_transform("unit_probe_tf", fn, replace=True)  # explicit override
    assert get_transform("unit_probe_tf") is fn


def test_custom_transform_runs_through_run_transform(tmp_path):
    register_transform(
        "unit_upper_tf",
        lambda sources: {
            "output_schema": "unit.v1",
            "records": [
                {"source": s.ref, "text": e.relative_path.upper()}
                for s in sources for e in s.entries
            ],
        },
    )
    src = make_source(tmp_path, {"a.txt": b"hi"})
    plan = ModelPlan(name="m", transform="unit_upper_tf",
                     train_sources=(Source(dataset="books", split="train"),))
    out = run_transform(plan, [src])
    assert out["records"][0]["text"] == "A.TXT"


def test_transform_identity_uses_registered_source():
    from mlforge.ingest.transforms import transform_identity

    ident = transform_identity("text_corpus")
    assert ident["name"] == "text_corpus"
    assert ident["code_hash"].startswith("sha256:")
