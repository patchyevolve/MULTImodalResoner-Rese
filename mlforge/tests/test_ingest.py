"""Build step 8 — ingestion: identity, config, DAG, transforms, prepare,
and the gate's step-4 dataset provider (12 §6.2–§6.4, §10; 13 §4.1, §5.2,
§6.6).

Fail-closed rules under test: symlinks/empty trees refused, unknown
transforms/models/keys rejected with the SPECIFIC exit code, content
mismatch never silently reinterpreted, cache keys change with any of the
four identity fields.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from test_train_cli import _spec

from mlforge.cli.main import main
from mlforge.errors import NotFound, PreconditionFailed, ValidationBlock
from mlforge.ingest import config as ingest_config
from mlforge.ingest.dag import ModelPlan, Source, load_ingestion, resolve_order
from mlforge.ingest.identity import (
    build_manifest,
    parse_ref,
    recompute_identity,
    scan_files,
)
from mlforge.ingest.prepare import derived_dataset_id, prepare
from mlforge.ingest.transforms import (
    ResolvedSource,
    cache_key,
    calibration,
    dataset_types,
    env_fingerprint,
    get_transform,
    mot_challenge,
    registry_names,
    reid_crops,
    run_transform,
    soccernet_events,
    suggest_transforms,
    tabular,
    transform_identity,
    video_clips,
)
from mlforge.validation import RESUME_GATE_STEPS, provide_pass
from mlforge.workflow import WorkflowAPI
from mlforge.yamlmini import YamlError, dump, loads

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """Machine-local config lives outside the workspace (13 §10)."""
    h = tmp_path / "mlforge-home"
    monkeypatch.setenv("MLFORGE_HOME", str(h))
    return h


@pytest.fixture
def ws(tmp_path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


def _tree(base: Path, name: str, files: dict[str, bytes]) -> Path:
    d = base / "data" / name
    d.mkdir(parents=True)
    for rel, content in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    return d


def _add(ws: Path, name: str, path: Path, *extra: str) -> int:
    return main(["--root", str(ws), "dataset", "add", name, str(path),
                 "--yes", *extra])


def _verify(ws: Path, name: str, *extra: str) -> int:
    return main(["--root", str(ws), "dataset", "verify", name, *extra])


def _add_verified(ws: Path, name: str, path: Path) -> None:
    assert _add(ws, name, path) == 0
    assert _verify(ws, name) == 0


def _ingestion(ws: Path, text: str) -> None:
    (ws / "ingestion.yaml").write_text(text, encoding="utf-8")


SIMPLE_INGESTION = """
# 12 §10.2
models:
  rf_detr_s:
    transform: coco_detection
    train_sources: [coco_2017:train, custom_clips:train]
    val_sources: [coco_2017:val]
  calibrator:
    transform: calibration
    train_sources: [{generated_from: rf_detr_s}]
    depends_on: [rf_detr_s]
"""

COCO_FILES = {
    "annotations.json": json.dumps({"images": [{"id": 1}]}).encode(),
    "imgs/0001.jpg": b"jpegbytes-one",
    "imgs/0002.jpg": b"jpegbytes-two",
}
CLIP_FILES = {"frame1.jpg": b"clip-one"}


# ---------------------------------------------------------------------------
# 12 §6.2 — dataset identity
# ---------------------------------------------------------------------------


def test_identity_deterministic_across_locations(tmp_path):
    a = _tree(tmp_path / "a", "d", COCO_FILES)
    b = _tree(tmp_path / "b", "d", COCO_FILES)
    ma, mb = build_manifest(a, "coco_2017"), build_manifest(b, "coco_2017")
    assert ma.identity == mb.identity
    assert ma.file_count == 3
    # relative POSIX paths only — absolute paths never enter identity (§6.3)
    for entry in ma.files:
        assert not entry.relative_path.startswith("/")
        assert str(a) not in entry.relative_path
        assert entry.relative_path in ("annotations.json", "imgs/0001.jpg",
                                       "imgs/0002.jpg")


def test_identity_changes_with_content(tmp_path):
    d = _tree(tmp_path, "d", COCO_FILES)
    before = build_manifest(d, "coco_2017").identity
    (d / "imgs" / "0001.jpg").write_bytes(b"changed")
    after = build_manifest(d, "coco_2017").identity
    assert before != after
    # counts are NOT identity: same count, different bytes ⇒ different hash
    assert build_manifest(d, "coco_2017").file_count == 3


def test_symlink_refused(tmp_path):
    d = _tree(tmp_path, "d", COCO_FILES)
    (d / "link.jpg").symlink_to(d / "imgs" / "0001.jpg")
    with pytest.raises(ValidationBlock, match="symlink"):
        build_manifest(d, "coco_2017")


def test_empty_tree_refused(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(ValidationBlock, match="empty dataset"):
        build_manifest(d, "coco_2017")


def test_missing_path_not_found(tmp_path):
    with pytest.raises(NotFound):
        scan_files(tmp_path / "nope")


def test_parse_ref_forms():
    assert parse_ref("coco_2017") == ("coco_2017", "v1")
    assert parse_ref("coco_2017:v2") == ("coco_2017", "v2")
    assert parse_ref("dataset://coco_2017:v3") == ("coco_2017", "v3")
    with pytest.raises(ValidationBlock):
        parse_ref("dataset://")


# ---------------------------------------------------------------------------
# yamlmini — the strict YAML subset both config files use
# ---------------------------------------------------------------------------


def test_yamlmini_ingestion_example_roundtrip():
    doc = loads(SIMPLE_INGESTION)
    assert doc["models"]["rf_detr_s"]["train_sources"] == [
        "coco_2017:train", "custom_clips:train"
    ]
    assert doc["models"]["calibrator"]["train_sources"] == [
        {"generated_from": "rf_detr_s"}
    ]
    assert doc["models"]["calibrator"]["depends_on"] == ["rf_detr_s"]
    assert loads(dump(doc)) == doc  # round-trip


def test_yamlmini_fail_closed():
    for bad in ["a:\n\tb: 1",  # tab indentation
                "---\na: 1",  # multi-document
                "a: |\n  text",  # block scalar
                "a: 1\na: 2",  # duplicate key
                "a:\n b: 1\n  c: 2"]:  # bad indent
        with pytest.raises(YamlError):
            loads(bad)


# ---------------------------------------------------------------------------
# 12 §6.3 / §10.1 — machine-local paths vs portable registry
# ---------------------------------------------------------------------------


def test_paths_live_outside_workspace(ws, home, tmp_path):
    d = _tree(tmp_path, "d", COCO_FILES)
    ingest_config.set_path(ws, "coco_2017", d)
    pf = ingest_config.paths_file(ws)
    assert pf.is_file()
    assert pf.parent == home  # $MLFORGE_HOME, not the workspace
    assert ingest_config.load_paths(ws) == {"coco_2017": str(d)}
    # registry (identity index) stays in the workspace — portable
    ingest_config.write_registry(ws, "coco_2017", {"identity": "sha256:x"})
    assert (ws / "datasets.yaml").is_file()


def test_default_home_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFORGE_HOME", raising=False)
    assert ingest_config.mlforge_home() == Path.home() / ".mlforge"


# ---------------------------------------------------------------------------
# 12 §10.2 — ingestion DAG
# ---------------------------------------------------------------------------


def test_load_ingestion_full(ws):
    _ingestion(ws, SIMPLE_INGESTION)
    plans = load_ingestion(ws)
    assert set(plans) == {"rf_detr_s", "calibrator"}
    plan = plans["rf_detr_s"]
    assert plan.transform == "coco_detection"
    assert [s.ref for s in plan.train_sources] == [
        "coco_2017:train", "custom_clips:train"
    ]
    assert plan.val_sources[0].split == "val"
    cal = plans["calibrator"]
    assert cal.depends_on == ("rf_detr_s",)
    assert cal.sources[0].generated_from == "rf_detr_s"


def test_load_ingestion_errors(ws):
    with pytest.raises(PreconditionFailed, match="not found"):
        load_ingestion(ws)  # no file ⇒ exit 3
    _ingestion(ws, "models:\n  m:\n    transform: t\n    typo: 1\n")
    with pytest.raises(PreconditionFailed, match="unknown keys"):
        load_ingestion(ws)
    _ingestion(ws, "models:\n  m:\n    train_sources: [a:train]\n")
    with pytest.raises(PreconditionFailed, match="transform"):
        load_ingestion(ws)


def test_source_parse_forms():
    assert Source.parse("coco_2017:train", "x") == Source(
        dataset="coco_2017", split="train"
    )
    assert Source.parse("coco_2017", "x").split is None
    assert Source.parse({"generated_from": "m1"}, "x").is_generated
    with pytest.raises(PreconditionFailed):
        Source.parse({"generated_from": "m1", "extra": 1}, "x")
    with pytest.raises(PreconditionFailed):
        Source.parse(42, "x")


def test_source_parse_generated_with_dataset():
    # {generated_from, dataset} — provenance + located bytes (12 §10.2)
    s = Source.parse({"generated_from": "m1", "dataset": "d1"}, "x")
    assert s.is_generated
    assert s.dataset == "d1"
    assert s.ref == "generated_from:m1@d1"
    # bare form still parses — prepare's guidance names the missing key
    assert Source.parse({"generated_from": "m1"}, "x").dataset is None
    # dataset must be a non-empty name; dict form requires provenance
    with pytest.raises(PreconditionFailed):
        Source.parse({"generated_from": "m1", "dataset": 5}, "x")
    with pytest.raises(PreconditionFailed):
        Source.parse({"generated_from": "m1", "dataset": "  "}, "x")
    with pytest.raises(PreconditionFailed):
        Source.parse({"dataset": "d1"}, "x")


def test_model_plan_to_dict_roundtrip_generated(ws):
    plan = ModelPlan(
        name="cal", transform="text_corpus",
        train_sources=(
            Source(dataset="d1", generated_from="m1"),
            Source(dataset="coco_2017", split="train"),
        ),
        val_sources=(Source(dataset="d1", generated_from="m1"),),
        depends_on=("rf_detr_s",),
    )
    out = plan.to_dict()
    # generated sources emit their DICT form (a bare ref string would
    # re-parse as name:split and lose provenance)
    assert out["train_sources"][0] == {"generated_from": "m1", "dataset": "d1"}
    assert out["train_sources"][1] == "coco_2017:train"
    reparsed = tuple(Source.parse(s, "x") for s in out["train_sources"])
    assert reparsed == plan.train_sources
    # through the real loader too (to_dict → YAML → load_ingestion)
    _ingestion(ws, dump({"models": {"cal": out}}))
    assert load_ingestion(ws)["cal"] == plan


def test_resolve_order_and_cycle(ws):
    _ingestion(ws, SIMPLE_INGESTION)
    plans = load_ingestion(ws)
    assert resolve_order(plans, "calibrator") == ["rf_detr_s", "calibrator"]
    with pytest.raises(NotFound, match="not found"):
        resolve_order(plans, "ghost")
    # cycle
    _ingestion(ws, """
models:
  a:
    transform: t
    depends_on: [b]
  b:
    transform: t
    depends_on: [a]
""")
    plans = load_ingestion(ws)
    with pytest.raises(ValidationBlock, match="cycle"):
        resolve_order(plans, "a")


# ---------------------------------------------------------------------------
# 12 §6.4 — transforms (closed registry, code hash, deterministic)
# ---------------------------------------------------------------------------


def test_transform_registry_closed():
    assert "coco_detection" in registry_names()
    ident = transform_identity("coco_detection")
    assert ident["name"] == "coco_detection"
    assert ident["code_hash"].startswith("sha256:")
    assert ident["config_hash"].startswith("sha256:")
    with pytest.raises(ValidationBlock, match="unknown transform"):
        get_transform("nope")


def test_coco_detection_records(tmp_path):
    d = _tree(tmp_path, "d", COCO_FILES)
    manifest = build_manifest(d, "coco_2017")
    src = ResolvedSource(ref="coco_2017:train", split="train",
                         dataset_id="coco_2017", path=d,
                         identity=manifest.identity, entries=manifest.files)
    out = get_transform("coco_detection")([src])
    assert len(out["records"]) == 3
    assert all(r["source"] == "coco_2017:train" for r in out["records"])
    assert all(r["relative_path"] and r["sha256"].startswith("sha256:")
               for r in out["records"])
    (ann,) = out["annotations"]
    assert ann["relative_path"] == "annotations.json"
    # canonical re-hash proves annotations PARSE, not just exist
    from mlforge.hashing import content_hash
    assert ann["canonical_sha256"] == content_hash(
        json.loads((d / "annotations.json").read_text())
    )


def test_coco_detection_rejects_broken_annotation(tmp_path):
    d = _tree(tmp_path, "d", {"annotations.json": b"{not json"})
    manifest = build_manifest(d, "coco_2017")
    src = ResolvedSource(ref="coco_2017:train", split="train",
                         dataset_id="coco_2017", path=d,
                         identity=manifest.identity, entries=manifest.files)
    with pytest.raises(ValidationBlock, match="does not parse"):
        get_transform("coco_detection")([src])


def test_run_transform_fail_closed(monkeypatch, tmp_path):
    from mlforge.ingest import transforms as tf

    d = _tree(tmp_path, "d", COCO_FILES)
    manifest = build_manifest(d, "coco_2017")
    src = ResolvedSource(ref="coco_2017:train", split="train",
                         dataset_id="coco_2017", path=d,
                         identity=manifest.identity, entries=manifest.files)
    plan = ModelPlan(name="m", transform="empty_out")
    monkeypatch.setitem(tf._REGISTRY, "empty_out",
                        lambda s: {"output_schema": "x", "records": []})
    with pytest.raises(ValidationBlock, match="0 records"):
        run_transform(plan, [src])
    # a transform that silently skips a source is a guess → BLOCK
    monkeypatch.setitem(tf._REGISTRY, "empty_out",
                        lambda s: {"output_schema": "x", "records": [
                            {"source": "other:x"}]})
    with pytest.raises(ValidationBlock, match="no records for"):
        run_transform(plan, [src])


def test_cache_key_covers_four_fields():
    base = ([{"ref": "a:train", "identity": "sha256:1"}],
            {"name": "t", "code_hash": "sha256:2", "config_hash": "sha256:3"},
            "sha256:4")
    k = cache_key(*base)
    assert cache_key(*base) == k  # deterministic
    assert cache_key([{"ref": "a:train", "identity": "sha256:9"}], base[1],
                     base[2]) != k  # input
    assert cache_key(base[0], {**base[1], "code_hash": "sha256:9"},
                     base[2]) != k  # transform code
    assert cache_key(base[0], base[1], "sha256:9") != k  # environment
    assert env_fingerprint().startswith("sha256:")


# ---------------------------------------------------------------------------
# 13 §4.1 DATA — dataset add / list / verify
# ---------------------------------------------------------------------------


def test_dataset_add_list_verify_flow(ws, home, tmp_path):
    d = _tree(tmp_path, "d", COCO_FILES)
    assert _add(ws, "coco_2017", d) == 0
    wf = WorkflowAPI(ws)
    proj = wf.get_dataset_status("coco_2017")
    assert proj["state"] == "REGISTERED"
    reg = json.loads((ws / "datasets" / "coco_2017" / "identity.json")
                     .read_text())
    assert reg["file_count"] == 3 and reg["version"] == "v1"
    assert (ws / "datasets.yaml").is_file()  # portable identity index

    assert _verify(ws, "coco_2017") == 0
    assert wf.get_dataset_status("coco_2017")["state"] == "VERIFIED"

    # journal: registered + verified + path audit, in order
    events = [e.event for e in wf._journal("dataset", "coco_2017").read()]
    assert "dataset_registered" in events
    assert "dataset_path_configured" in events
    assert "dataset_verified" in events


def test_dataset_list_output(ws, home, tmp_path, capsys):
    d = _tree(tmp_path, "d", COCO_FILES)
    _add_verified(ws, "coco_2017", d)
    capsys.readouterr()  # drop add/verify chatter — parse clean JSON
    code = main(["--root", str(ws), "dataset", "list", "--json"])
    assert code == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == [{
        "dataset": "coco_2017", "state": "VERIFIED", "version": "v1",
        "identity": rows[0]["identity"], "file_count": 3, "path": str(d),
    }]
    assert rows[0]["identity"].startswith("sha256:")
    # human table
    assert main(["--root", str(ws), "dataset", "list"]) == 0
    out = capsys.readouterr().out
    assert "coco_2017" in out and "VERIFIED" in out


def test_verify_tamper_rejects(ws, home, tmp_path):
    d = _tree(tmp_path, "d", COCO_FILES)
    _add_verified(ws, "coco_2017", d)
    (d / "imgs" / "0001.jpg").write_bytes(b"tampered")
    assert _verify(ws, "coco_2017") == 1  # mismatch ⇒ REJECTED, exit 1
    wf = WorkflowAPI(ws)
    assert wf.get_dataset_status("coco_2017")["state"] == "REJECTED"
    # REJECTED requires explicit re-registration (exit 3)
    assert _verify(ws, "coco_2017") == 3


def test_add_conflict_requires_force(ws, home, tmp_path):
    d = _tree(tmp_path, "d", COCO_FILES)
    _add_verified(ws, "coco_2017", d)
    (d / "imgs" / "0001.jpg").write_bytes(b"changed")
    assert _add(ws, "coco_2017", d) == 1  # different identity, no --force
    wf = WorkflowAPI(ws)
    assert wf.get_dataset_status("coco_2017")["state"] == "VERIFIED"  # unchanged
    assert _add(ws, "coco_2017", d, "--force") == 0  # explicit reinterpret
    assert wf.get_dataset_status("coco_2017")["state"] == "REGISTERED"
    events = wf._journal("dataset", "coco_2017").read()
    assert events[-1].event == "dataset_path_configured"
    reregs = [e for e in events if e.event == "dataset_reregistered"]
    assert len(reregs) == 1 and reregs[0].data["action"] == "reregister"
    assert _verify(ws, "coco_2017") == 0  # usable again


def test_add_same_identity_updates_path_only(ws, home, tmp_path):
    d1 = _tree(tmp_path / "one", "d", COCO_FILES)
    d2 = _tree(tmp_path / "two", "d", COCO_FILES)  # same bytes elsewhere
    _add_verified(ws, "coco_2017", d1)
    assert _add(ws, "coco_2017", d2) == 0
    wf = WorkflowAPI(ws)
    # no transition: identity is content, path is machine-local (12 §6.3)
    assert wf.get_dataset_status("coco_2017")["state"] == "VERIFIED"
    assert ingest_config.load_paths(ws)["coco_2017"] == str(d2)
    events = wf._journal("dataset", "coco_2017").read()
    assert sum(1 for e in events
               if e.event == "dataset_reregistered") == 0


def test_add_missing_path_exit_2(ws, home, tmp_path):
    assert _add(ws, "ghost", tmp_path / "nope") == 2
    assert not (ws / "datasets" / "ghost").exists()


def test_add_dedupe_journal(ws, home, tmp_path):
    d = _tree(tmp_path, "d", COCO_FILES)
    args = ["--root", str(ws), "dataset", "add", "coco_2017", str(d),
            "--yes", "--command-id", "cid-add-1"]
    assert main(args) == 0
    assert main(args) == 0  # duplicate returns the original result
    wf = WorkflowAPI(ws)
    events = wf._journal("dataset", "coco_2017").read()
    dedupes = [e for e in events if e.event == "COMMAND_DEDUPED"]
    assert len(dedupes) == 1 and dedupes[0].data["command"] == "dataset_add"
    assert wf.get_dataset_status("coco_2017")["state"] == "REGISTERED"


# ---------------------------------------------------------------------------
# 13 §6.6 — prepare (transform → derived dataset)
# ---------------------------------------------------------------------------


def _prepared_ws(ws, home, tmp_path) -> tuple[Path, Path]:
    coco = _tree(tmp_path / "src1", "coco", COCO_FILES)
    clips = _tree(tmp_path / "src2", "clips", CLIP_FILES)
    _add_verified(ws, "coco_2017", coco)
    _add_verified(ws, "custom_clips", clips)
    _ingestion(ws, """
models:
  rf_detr_s:
    transform: coco_detection
    train_sources: [coco_2017:train, custom_clips:train]
""")
    return coco, clips


def test_prepare_miss_then_hit(ws, home, tmp_path):
    _prepared_ws(ws, home, tmp_path)
    assert main(["--root", str(ws), "prepare", "rf_detr_s", "--json"]) == 0
    wf = WorkflowAPI(ws)
    derived = derived_dataset_id("rf_detr_s")
    assert derived == "rf_detr_s_prepared"
    assert wf.get_dataset_status(derived)["state"] == "PREPARED"
    cache_index = json.loads((ws / "artifacts" / "prepare_cache.json")
                             .read_text())
    assert len(cache_index) == 1
    # second run: same artifact from cache, no re-transform
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["--root", str(ws), "prepare", "rf_detr_s", "--json"])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["cache"] == "hit"
    assert payload["artifact_hash"] in {
        v["artifact_hash"] for v in cache_index.values()
    }


def test_prepare_invalidates_on_source_change(ws, home, tmp_path):
    coco, _ = _prepared_ws(ws, home, tmp_path)
    assert main(["--root", str(ws), "prepare", "rf_detr_s"]) == 0
    # change source content without re-verifying ⇒ prepare's own identity
    # check blocks (exit 1) — never transforms mismatched bytes
    (coco / "imgs" / "0001.jpg").write_bytes(b"changed")
    assert main(["--root", str(ws), "prepare", "rf_detr_s"]) == 1


def test_prepare_error_exit_codes(ws, home, tmp_path):
    # missing ingestion.yaml ⇒ 3
    assert main(["--root", str(ws), "prepare", "rf_detr_s"]) == 3
    _prepared_ws(ws, home, tmp_path)
    # unknown model ⇒ 2
    assert main(["--root", str(ws), "prepare", "ghost"]) == 2
    # unknown transform ⇒ 1
    _ingestion(ws, "models:\n  m:\n    transform: nope\n    "
                   "train_sources: [coco_2017:train]\n")
    assert main(["--root", str(ws), "prepare", "m"]) == 1
    # unmet depends_on / generated_from ⇒ 3 (upstream model not in registry)
    _ingestion(ws, SIMPLE_INGESTION)
    assert main(["--root", str(ws), "prepare", "calibrator"]) == 3
    # source not VERIFIED yet ⇒ 3 (add a fresh, unverified dataset)
    fresh = _tree(tmp_path / "fresh", "fresh", {"a.txt": b"a"})
    assert _add(ws, "fresh_ds", fresh) == 0  # REGISTERED, not verified
    _ingestion(ws, "models:\n  m:\n    transform: coco_detection\n    "
                   "train_sources: [fresh_ds:train]\n")
    assert main(["--root", str(ws), "prepare", "m"]) == 3


def test_prepare_generated_source(ws, home, tmp_path):
    """{generated_from, dataset} end-to-end: registry gate → located
    bytes → provenance in the artifact doc and the CLI result (12 §10.2)."""
    # producing model published (create → validate → publish, 13 §5.4)
    wf = WorkflowAPI(ws)
    mid = wf.create_model(artifact_hash="sha256:det1", name="rf_detr_s",
                          version="v1")
    wf.validate_model(mid)
    wf.publish_model(mid)
    # its output registered as an ordinary dataset (verify, never scan)
    preds = _tree(tmp_path / "preds", "preds",
                  {"p0.txt": b"0.9 car 10 20 30 40"})
    _add_verified(ws, "det_out", preds)
    _ingestion(ws, """
models:
  rf_detr_s:
    transform: coco_detection
    train_sources: [coco_2017:train]   # declared, not prepared in this test
  calibrator:
    transform: text_corpus
    train_sources: [{generated_from: rf_detr_s, dataset: det_out}]
    depends_on: [rf_detr_s]
""")
    plans = load_ingestion(ws)
    assert resolve_order(plans, "calibrator") == ["rf_detr_s", "calibrator"]
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["--root", str(ws), "prepare", "calibrator", "--json"]) == 0
    payload = json.loads(buf.getvalue())
    assert payload["sources"] == ["det_out (generated from rf_detr_s)"]
    assert wf.get_dataset_status(derived_dataset_id("calibrator"))["state"] \
        == "PREPARED"
    # provenance is IN the artifact (doc sources + every record)
    from mlforge.store import ContentStore
    idx = json.loads((ws / "artifacts" / "prepare_cache.json").read_text())
    doc = json.loads(ContentStore(ws / "store").get_bytes(
        next(iter(idx.values()))["artifact_hash"]))
    entry = doc["sources"][0]
    assert entry["generated_from"] == "rf_detr_s"
    assert entry["ref"] == "det_out"
    assert doc["records"] and all(
        r["source"] == "det_out" for r in doc["records"])


def test_prepare_calibration_generated_source(ws, home, tmp_path):
    """{generated_from} + the REAL `calibration` transform: detector
    prediction rows become the calibrator's prepared artifact — logits
    normalized, provenance kept (12 §10.2)."""
    wf = WorkflowAPI(ws)
    mid = wf.create_model(artifact_hash="sha256:det1", name="rf_detr_s",
                          version="v1")
    wf.validate_model(mid)
    wf.publish_model(mid)
    preds = _tree(tmp_path / "preds", "preds", {
        "p0.jsonl": b'{"logits": [2.0, -1.5], "label": 0}\n'
                    b'{"logits": [-0.5, 1.1], "label": 1}\n',
        "p1.json": b'[{"probs": [0.7, 0.3], "label": 0}]',
    })
    _add_verified(ws, "det_out", preds)
    _ingestion(ws, """
models:
  rf_detr_s:
    transform: coco_detection
    train_sources: [coco_2017:train]   # declared, not prepared in this test
  calibrator:
    transform: calibration
    train_sources: [{generated_from: rf_detr_s, dataset: det_out}]
    depends_on: [rf_detr_s]
""")
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["--root", str(ws), "prepare", "calibrator", "--json"]) == 0
    payload = json.loads(buf.getvalue())
    assert payload["sources"] == ["det_out (generated from rf_detr_s)"]
    assert wf.get_dataset_status(derived_dataset_id("calibrator"))["state"] \
        == "PREPARED"
    from mlforge.store import ContentStore
    idx = json.loads((ws / "artifacts" / "prepare_cache.json").read_text())
    doc = json.loads(ContentStore(ws / "store").get_bytes(
        next(iter(idx.values()))["artifact_hash"]))
    assert doc["transform"]["name"] == "calibration"
    assert len(doc["records"]) == 3           # 2 jsonl rows + 1 json row
    for rec in doc["records"]:
        assert len(rec["logits"]) == 2        # probs normalized to logits
        assert rec["label"] in (0, 1)
        assert rec["source"] == "det_out"     # provenance in every record


def test_prepare_generated_gate(ws, home, tmp_path):
    """Provenance must be AVAILABLE; the generated bytes must be named."""
    preds = _tree(tmp_path / "preds", "preds", {"p0.txt": b"0.9 car"})
    _add_verified(ws, "det_out", preds)
    _ingestion(ws, """
models:
  calibrator:
    transform: text_corpus
    train_sources: [{generated_from: rf_detr_s, dataset: det_out}]
""")
    # producing model absent from the registry ⇒ 3 (never exit 2 — the
    # config is well-formed, the registry just cannot serve it yet)
    assert main(["--root", str(ws), "prepare", "calibrator"]) == 3
    wf = WorkflowAPI(ws)
    mid = wf.create_model(artifact_hash="sha256:det1", name="rf_detr_s",
                          version="v1")
    # CREATED then VALIDATED: not consumable until published ⇒ 3
    assert main(["--root", str(ws), "prepare", "calibrator"]) == 3
    wf.validate_model(mid)
    assert main(["--root", str(ws), "prepare", "calibrator"]) == 3
    wf.publish_model(mid)
    # AVAILABLE — now the missing `dataset:` key is the blocker, and the
    # error must carry the exact registration guidance
    _ingestion(ws, """
models:
  calibrator:
    transform: text_corpus
    train_sources: [{generated_from: rf_detr_s}]
""")
    with pytest.raises(PreconditionFailed, match="names no dataset") as ei:
        prepare(ws, "calibrator", workflow=WorkflowAPI(ws))
    hint = ei.value.hint or ""
    assert "mlforge dataset add" in hint
    assert "dataset: <ID>" in hint


def test_prepare_depends_on_registry_only_model(ws, home, tmp_path):
    """A depends_on entry outside ingestion.yaml is legal: resolve_order
    skips it and the registry gate decides availability (12 §10.2)."""
    preds = _tree(tmp_path / "preds", "preds", {"p0.txt": b"0.9 car"})
    _add_verified(ws, "det_out", preds)
    _ingestion(ws, """
models:
  calibrator:
    transform: text_corpus
    train_sources: [{generated_from: rf_detr_s, dataset: det_out}]
    depends_on: [rf_detr_s]
""")
    plans = load_ingestion(ws)
    # registry-only dep: nothing to order locally, no NotFound
    assert resolve_order(plans, "calibrator") == ["calibrator"]
    # …but prepare still fails closed until the model is AVAILABLE
    assert main(["--root", str(ws), "prepare", "calibrator"]) == 3
    wf = WorkflowAPI(ws)
    mid = wf.create_model(artifact_hash="sha256:det1", name="rf_detr_s",
                          version="v1")
    wf.validate_model(mid)
    wf.publish_model(mid)
    assert main(["--root", str(ws), "prepare", "calibrator"]) == 0
    assert wf.get_dataset_status(derived_dataset_id("calibrator"))["state"] \
        == "PREPARED"


def test_prepare_dedupe(ws, home, tmp_path):
    _prepared_ws(ws, home, tmp_path)
    args = ["--root", str(ws), "prepare", "rf_detr_s", "--json",
            "--command-id", "cid-prep-1"]
    assert main(args) == 0
    assert main(args) == 0  # original result returned
    wf = WorkflowAPI(ws)
    events = wf._journal("dataset", "rf_detr_s_prepared").read()
    dedupes = [e for e in events if e.event == "COMMAND_DEDUPED"]
    assert len(dedupes) == 1 and dedupes[0].data["command"] == "prepare"
    # exactly one artifact registered in the cache index
    cache_index = json.loads((ws / "artifacts" / "prepare_cache.json")
                             .read_text())
    assert len(cache_index) == 1


def test_prepare_api_returns_result(ws, home, tmp_path):
    _coco, _clips = _prepared_ws(ws, home, tmp_path)
    wf = WorkflowAPI(ws)
    res = prepare(ws, "rf_detr_s", workflow=wf)
    assert res.cache == "miss"
    assert res.record_count == 4  # 3 coco files + 1 clip
    assert set(res.sources) == {"coco_2017:train", "custom_clips:train"}
    assert res.dataset_ref == "dataset://rf_detr_s_prepared:v1"
    # artifact stored content-addressed in the store
    from mlforge.store import ContentStore
    store = ContentStore(ws / "store")
    assert store.contains(res.artifact_hash)
    doc = json.loads(store.get_bytes(res.artifact_hash))
    assert doc["artifact_schema"] == "mlforge.prepared_dataset.v1"
    assert doc["transform"]["code_hash"].startswith("sha256:")


# ---------------------------------------------------------------------------
# 12 §18 step 4 — builtin dataset identity provider
# ---------------------------------------------------------------------------


def _gate_without_dataset(root: Path) -> WorkflowAPI:
    ids = [s.id for s in RESUME_GATE_STEPS if s.id not in ("manifest", "schema")]
    providers = {k: provide_pass(f"{k} ok") for k in ids}
    providers.pop("dataset", None)  # builtin must fill step 4
    return WorkflowAPI(root, gate_providers=providers)


def test_gate_step4_builtin_pass(ws, home, tmp_path):
    d = _tree(tmp_path, "d", COCO_FILES)
    _add_verified(ws, "coco_2017", d)
    wf = _gate_without_dataset(ws)
    run_id = wf.create_run(_spec()).run_id
    report = wf.validate_run(run_id)
    assert not report.blocked, report.render()
    step4 = next(c for c in report.checks if c.id == "dataset")
    assert "3 files hashed" in step4.detail


def test_gate_step4_unregistered_blocks(ws, home):
    wf = _gate_without_dataset(ws)
    run_id = wf.create_run(_spec()).run_id  # coco_2017 never registered
    report = wf.validate_run(run_id)
    assert report.blocked and report.failed_step == 4
    assert "not registered" in report.first_failure.detail
    assert wf.get_run_state(run_id) == "FAILED"  # train path ⇒ FORK_ONLY


def test_gate_step4_no_path_blocks(ws, home):
    wf = WorkflowAPI(ws)
    # registered directly (API) but no machine-local path configured
    _, manifest = recompute_identity(
        _tree(Path(home) / "d", "d", COCO_FILES), "coco_2017"
    )
    wf.register_dataset("coco_2017", manifest.identity, file_count=3)
    gate_wf = _gate_without_dataset(ws)
    run_id = gate_wf.create_run(_spec()).run_id
    report = gate_wf.validate_run(run_id)
    assert report.blocked and report.failed_step == 4
    assert "no machine-local path" in report.first_failure.detail


def test_gate_step4_mismatch_blocks(ws, home, tmp_path):
    d = _tree(tmp_path, "d", COCO_FILES)
    _add_verified(ws, "coco_2017", d)
    (d / "imgs" / "0001.jpg").write_bytes(b"tampered")
    wf = _gate_without_dataset(ws)
    run_id = wf.create_run(_spec()).run_id
    report = wf.validate_run(run_id)
    assert report.blocked and report.failed_step == 4
    assert "identity mismatch" in report.first_failure.detail


def test_explicit_dataset_provider_wins(ws, home, tmp_path):
    """Callers that verify more honestly keep their wiring (setdefault)."""
    ids = [s.id for s in RESUME_GATE_STEPS if s.id not in ("manifest", "schema")]
    providers = {k: provide_pass(f"{k} ok") for k in ids}
    # dataset provider present in the caller's set → builtin must NOT run
    wf = WorkflowAPI(ws, gate_providers=providers)
    run_id = wf.create_run(_spec()).run_id  # nothing registered
    report = wf.validate_run(run_id)
    assert not report.blocked, report.render()


# ---------------------------------------------------------------------------
# dataset types catalog + new transforms (02_dataset_preparation; 12 §10.2)
# ---------------------------------------------------------------------------

#: 9-column MOTChallenge layout from 02_dataset_preparation §3 — MOT17,
#: MOT20 and SportsMOT all ship this (with the `frame,id,bb_left,...`
#: header spelling real gt.txt files use).
MOT_CSV = (
    "frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility\n"
    "1,1,10,20,30,40,1,1,0.85\n"
    "2,1,11,21,31,41,1,1,0.90\n"
)


def _src(tmp_path: Path, name: str, files: dict[str, bytes]) -> ResolvedSource:
    """A ResolvedSource over a fresh file tree (identity computed for real)."""
    base = _tree(tmp_path / "type-sources", name, files)
    manifest = build_manifest(base, name)
    return ResolvedSource(ref=f"{name}:train", split="train",
                          dataset_id=name, path=base,
                          identity=manifest.identity, entries=manifest.files)


# -- mot_challenge (MOT17 / MOT20 / SportsMOT, 02 §3) ----------------------

def test_mot_challenge_valid_gt(tmp_path):
    src = _src(tmp_path, "mot17",
               {"gt/gt.txt": MOT_CSV.encode(),
                "img1/000001.jpg": b"jpgframe"})
    out = mot_challenge([src])
    assert out["output_schema"] == "mot_challenge.v1"
    # every file (frames included) is a provenance record
    assert {r["relative_path"] for r in out["records"]} == {
        "gt/gt.txt", "img1/000001.jpg"}
    ann = out["annotations"][0]
    assert (ann["boxes"], ann["frames"], ann["tracks"]) == (2, 2, 1)
    assert ann["canonical_sha256"]


def test_mot_challenge_headerless_ok(tmp_path):
    src = _src(tmp_path, "mot20",
               {"gt.csv": b"1,1,10,20,30,40,1,1,0.5\n"})
    out = mot_challenge([src])
    assert out["annotations"][0]["boxes"] == 1


def test_mot_challenge_wrong_layout_blocks(tmp_path):
    src = _src(tmp_path, "badmot", {"gt.csv": b"a,b,c\n1,2,3\n"})
    with pytest.raises(ValidationBlock, match="MOTChallenge"):
        mot_challenge([src])


def test_mot_challenge_bad_value_blocks(tmp_path):
    src = _src(tmp_path, "badval",
               {"gt.csv": b"1,1,10,20,30,40,1,1,42\n"})  # visibility 0..1
    with pytest.raises(ValidationBlock, match="0..1"):
        mot_challenge([src])


def test_mot_challenge_stray_txt_skipped_no_annotations_blocks(tmp_path):
    src = _src(tmp_path, "readmeonly", {"readme.txt": b"see motchallenge.net"})
    with pytest.raises(ValidationBlock, match="no MOTChallenge CSV"):
        mot_challenge([src])


# -- reid_crops (Market1501, 02 §5 / 12 §10.2) -----------------------------

def test_reid_crops_manifest(tmp_path):
    src = _src(tmp_path, "market", {
        "bounding_box_train/0001_c1s1_000001_01.jpg": b"a",
        "bounding_box_train/0001_c2s1_000002_01.jpg": b"b",
        "bounding_box_train/0002_c1s1_000003_01.jpg": b"c",
        "query/0001_c3s1_000004_01.jpg": b"d",
        "readme.txt": b"x",
    })
    out = reid_crops([src])
    assert out["output_schema"] == "reid_crops.v1"
    assert {r["identity"] for r in out["records"]} == {"0001", "0002"}
    # split comes from the folder, not from the caller's assumption
    assert next(r for r in out["records"] if r["split"] == "query")["identity"] == "0001"
    by_id = {i["identity"]: i for i in out["identities"]}
    assert by_id["0001"]["images"] == 3
    assert by_id["0001"]["cameras"] == [1, 2, 3]


def test_reid_crops_wrong_naming_blocks(tmp_path):
    src = _src(tmp_path, "plainimgs", {"imgs/cat.jpg": b"x"})
    with pytest.raises(ValidationBlock, match="Market1501"):
        reid_crops([src])


# -- tabular (csv / tsv / jsonl / xlsx) ------------------------------------

def test_tabular_csv_strict_rows(tmp_path):
    src = _src(tmp_path, "tables",
               {"data.csv": b"name,age\nalice,30\nbob,25\n"})
    out = tabular([src])
    assert out["output_schema"] == "tabular.v1"
    assert [r["data"] for r in out["records"]] == [
        {"name": "alice", "age": "30"}, {"name": "bob", "age": "25"}]
    assert out["tables"][0]["columns"] == ["name", "age"]
    assert out["tables"][0]["rows"] == 2


def test_tabular_ragged_rows_block(tmp_path):
    src = _src(tmp_path, "ragged", {"data.csv": b"a,b\n1,2\n3\n"})
    with pytest.raises(ValidationBlock, match="never padded"):
        tabular([src])


def test_tabular_duplicate_header_blocks(tmp_path):
    src = _src(tmp_path, "dupe", {"data.csv": b"a,a\n1,2\n"})
    with pytest.raises(ValidationBlock, match="duplicate"):
        tabular([src])


def test_tabular_jsonl(tmp_path):
    src = _src(tmp_path, "jl", {"rows.jsonl": b'{"x": 1}\n{"y": true}\n'})
    out = tabular([src])
    assert len(out["records"]) == 2
    # non-string JSON values stay canonical JSON scalars — never str()'d
    assert out["records"][1]["data"]["y"] == "true"


def test_tabular_jsonl_non_object_blocks(tmp_path):
    src = _src(tmp_path, "jl2", {"rows.jsonl": b"[1,2]\n"})
    with pytest.raises(ValidationBlock, match="JSON object"):
        tabular([src])


def _xlsx_bytes() -> bytes:
    """Minimal workbook: sharedStrings + typed + inline + sparse cells."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "xl/sharedStrings.xml",
            f'<?xml version="1.0"?><sst xmlns="{ns}" count="2" uniqueCount="2">'
            "<si><t>name</t></si><si><t>score</t></si></sst>")
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f'<?xml version="1.0"?><worksheet xmlns="{ns}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c>'
            '<c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A1" t="inlineStr"><is><t>ada</t></is></c>'
            '<c r="B1"><v>9.5</v></c></row>'
            '<row r="3"><c r="A1" t="inlineStr"><is><t>bo</t></is></c></row>'
            "</sheetData></worksheet>")
    return buf.getvalue()


def test_tabular_xlsx_reader(tmp_path):
    src = _src(tmp_path, "xlsx", {"book.xlsx": _xlsx_bytes()})
    out = tabular([src])
    assert out["tables"][0]["columns"] == ["name", "score"]
    rows = [r["data"] for r in out["records"]]
    assert rows[0] == {"name": "ada", "score": "9.5"}
    # sparse cell → empty string (sheets are sparse by nature)
    assert rows[1] == {"name": "bo", "score": ""}
    assert out["records"][0]["relative_path"] == "book.xlsx#sheet1"


def test_tabular_xlsx_not_a_zip_blocks(tmp_path):
    src = _src(tmp_path, "badxlsx", {"book.xlsx": b"not a zip"})
    with pytest.raises(ValidationBlock, match="readable .xlsx"):
        tabular([src])


def test_tabular_no_table_files_blocks(tmp_path):
    src = _src(tmp_path, "notabs", {"img.jpg": b"x"})
    with pytest.raises(ValidationBlock, match="no table files"):
        tabular([src])


# -- calibration (C5 prediction records, 13 §Model 5 / 06 §1) ---------------

def test_calibration_jsonl_logits_probs_and_components(tmp_path):
    src = _src(tmp_path, "preds", {
        "p.jsonl": (
            b'{"logits": [1.0, -1.0], "label": 0}\n'
            b'{"probs": [0.7, 0.3], "label": 1, "components": '
            b'{"perception": 0.8, "temporal": 0.6, "motion": 0.5, '
            b'"cross_modal_agreement": 0.7, "reasoning": 0.4}}\n'
        )})
    out = calibration([src])
    assert out["output_schema"] == "calibration.v1"
    assert len(out["records"]) == 2
    first, second = out["records"]
    assert first["logits"] == [1.0, -1.0] and first["label"] == 0
    assert first["source"] == src.ref and first["split"] == "train"
    # probs → log-probabilities (softmax-invariant for temperature scaling)
    import math
    assert second["logits"] == pytest.approx(
        [math.log(0.7), math.log(0.3)])
    assert second["components"]["cross_modal_agreement"] == 0.7


def test_calibration_json_array(tmp_path):
    src = _src(tmp_path, "predsarr", {
        "p.json": json.dumps(
            [{"logits": [0.0, 0.0], "label": 1}]).encode()})
    out = calibration([src])
    assert out["records"][0]["label"] == 1


def test_calibration_needs_scores_blocks(tmp_path):
    src = _src(tmp_path, "noscores", {"p.jsonl": b'{"label": 0}\n'})
    with pytest.raises(ValidationBlock, match="`logits`"):
        calibration([src])


def test_calibration_probs_must_sum_to_one(tmp_path):
    src = _src(tmp_path, "badprobs",
               {"p.jsonl": b'{"probs": [0.9, 0.9], "label": 0}\n'})
    with pytest.raises(ValidationBlock, match="sum to"):
        calibration([src])


def test_calibration_label_out_of_range_blocks(tmp_path):
    src = _src(tmp_path, "badlabel",
               {"p.jsonl": b'{"logits": [0.1, 0.2], "label": 5}\n'})
    with pytest.raises(ValidationBlock, match="out of range"):
        calibration([src])


def test_calibration_mixed_class_counts_block(tmp_path):
    src = _src(tmp_path, "mixed", {
        "a.jsonl": b'{"logits": [0.1, 0.2], "label": 0}\n',
        "b.jsonl": b'{"logits": [0.1, 0.2, 0.3], "label": 1}\n'})
    with pytest.raises(ValidationBlock, match="class count differs"):
        calibration([src])


def test_calibration_components_must_be_exact_set(tmp_path):
    src = _src(tmp_path, "badcomp", {
        "p.jsonl": b'{"logits": [0.1, 0.2], "label": 0, '
                   b'"components": {"perception": 0.5}}\n'})
    with pytest.raises(ValidationBlock, match="components.*mismatch"):
        calibration([src])


def test_calibration_no_records_blocks_with_guidance(tmp_path):
    src = _src(tmp_path, "emptycal", {"img.jpg": b"x"})
    with pytest.raises(ValidationBlock, match="no prediction records") as ei:
        calibration([src])
    assert "mlforge dataset add" in (ei.value.hint or "")


def test_calibration_json_must_be_array(tmp_path):
    src = _src(tmp_path, "objjson", {
        "p.json": b'{"logits": [0.1, 0.2], "label": 0}'})
    with pytest.raises(ValidationBlock, match="JSON array"):
        calibration([src])


# -- video_clips (02 §11 custom clips, §8 Celeb-DF++, §9 FaceForensics++;
#    §15 "All clips play without errors" / "Annotations in COCO format") ---

def _ffmpeg_clip(tmp_path: Path) -> bytes:
    """A REAL 64x64 @ 8 fps, 1 s clip — ffmpeg's `testsrc` is the simplest
    deterministic video source (no numpy). Skips if ffmpeg is absent."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not installed — cannot build a real video fixture")
    out = tmp_path / "clip-fixture.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=8",
         "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out.read_bytes()


def test_video_clips_probes_real_mp4(tmp_path):
    src = _src(tmp_path, "clips", {"clip.mp4": _ffmpeg_clip(tmp_path)})
    out = get_transform("video_clips")([src])  # registered, not raw-called
    assert out["output_schema"] == "video_clips.v1"
    (rec,) = out["records"]
    assert rec["source"] == src.ref and rec["split"] == "train"
    assert rec["relative_path"] == "clip.mp4"
    assert rec["sha256"].startswith("sha256:") and rec["size"] > 0
    video = rec["video"]
    assert (video["width"], video["height"]) == (64, 64)
    assert video["fps"] > 0 and video["duration_s"] > 0
    assert video["decoder"] in {"av", "cv2"}
    assert video["codec"]                      # h264 for the testsrc clip
    assert video["frames"] is None or video["frames"] >= 1
    # bounded decode really happened — the probe proves the clip PLAYS
    try:
        import av  # noqa: F401
    except ImportError:
        pass
    else:
        assert video["decoder"] == "av"        # PyAV is primary (01 §75)


def test_video_clips_coco_sidecar_rehashes(tmp_path):
    annotation = {"images": [{"id": 1, "file_name": "a.jpg"}],
                  "annotations": [{"id": 1, "image_id": 1,
                                   "bbox": [0, 0, 10, 10],
                                   "category_id": 1}]}
    src = _src(tmp_path, "clipann",
               {"annotations.json": json.dumps(annotation).encode(),
                "readme.txt": b"clip notes"})          # stray file: ignored
    out = get_transform("video_clips")([src])
    # a sidecar-only source still yields records (run_transform coverage)
    (rec,) = out["records"]
    assert rec["relative_path"] == "annotations.json"
    assert rec["source"] == src.ref
    (ann,) = out["annotations"]
    from mlforge.hashing import content_hash
    assert ann["canonical_sha256"] == content_hash(annotation)


def test_video_clips_corrupt_sidecar_blocks(tmp_path):
    src = _src(tmp_path, "badclipann", {"annotations.json": b"{not json"})
    with pytest.raises(ValidationBlock, match="does not parse"):
        video_clips([src])


def test_video_clips_coco_shape_blocks(tmp_path):
    src = _src(tmp_path, "badshape", {"annotations.json": json.dumps(
        {"images": {"id": 1}, "annotations": {"bad": True}}).encode()})
    with pytest.raises(ValidationBlock, match="must both be lists"):
        video_clips([src])


def test_video_clips_no_videos_blocks_with_extensions_hint(tmp_path):
    src = _src(tmp_path, "noclips",
               {"readme.txt": b"see 02_dataset_preparation section 11"})
    with pytest.raises(ValidationBlock, match="no video files") as ei:
        video_clips([src])
    hint = ei.value.hint or ""
    assert ".mp4" in hint and ".mkv" in hint and ".webm" in hint
    assert "COCO annotation .json" in hint
    assert "mlforge dataset types" in hint


def test_video_clips_undecodable_file_blocks(tmp_path):
    # a "video" that is really just bytes must never pass as playable
    src = _src(tmp_path, "brokenclip",
               {"broken.mp4": b"this is definitely not a video file\n" * 8})
    with pytest.raises(ValidationBlock, match="play") as ei:
        video_clips([src])
    assert "02_dataset_preparation §15" in (ei.value.hint or "")


# -- soccernet_events (02 §7 SoccerNet v2 layout, §15 "Labels-v2.json
#    parseable for all 500 games") ------------------------------------------

#: league/season/game prefix straight from the §7 directory tree.
_SN_GAME = "england_epl/2016-2017/2017-02-04 - 12-30 Chelsea 1 - 1 Liverpool"


def _soccernet_files() -> dict[str, bytes]:
    labels = {"annotations": [
        {"gameTime": "1 - 00:12:00", "position": 12000, "label": "Goal"},
        {"position": 45000.5, "label": "Throw-in"},
        {"position": 90000, "label": "Foul"},
    ]}
    return {
        f"{_SN_GAME}/Labels-v2.json": json.dumps(labels).encode(),
        f"{_SN_GAME}/video.ini": (
            b"[First Half]\nstart=00:00:00.000\nduration=00:47:30.000\n"
            b"[Second Half]\nstart=00:47:30.000\nduration=00:47:30.000\n"),
        f"{_SN_GAME}/Labels-cameras.json": json.dumps(
            {"annotations": [{"position": 0, "type": 1},
                             {"position": 240, "type": 2}]}).encode(),
        f"{_SN_GAME}/1_720p.mkv": b"mkv-bytes-never-decoded",
        f"{_SN_GAME}/1_ResNET_TF2_PCA512.npy": b"npy-bytes-never-loaded",
    }


def test_soccernet_events_full_layout(tmp_path):
    src = _src(tmp_path, "soccernet", _soccernet_files())
    out = get_transform("soccernet_events")([src])
    assert out["output_schema"] == "soccernet_events.v1"
    records = out["records"]
    assert all(r["source"] == src.ref and r["split"] == "train"
               for r in records)
    kinds = [r["kind"] for r in records]
    assert kinds.count("action_spot") == 3
    spots = [r for r in records if r["kind"] == "action_spot"]
    assert [s["label"] for s in spots] == ["Goal", "Throw-in", "Foul"]
    assert [s["position_ms"] for s in spots] == [12000.0, 45000.5, 90000.0]
    assert all(s["game"] == _SN_GAME for s in spots)
    (game,) = [r for r in records if r["kind"] == "game"]
    assert (game["events"], game["classes"]) == (
        3, ["Foul", "Goal", "Throw-in"])
    assert game["sha256"].startswith("sha256:")
    (ini,) = [r for r in records if r["kind"] == "video_ini"]
    assert ini["sections"]["First Half"]["start"] == "00:00:00.000"
    assert ini["sections"]["Second Half"]["duration"] == "00:47:30.000"
    (cam,) = [r for r in records if r["kind"] == "camera_labels"]
    assert cam["canonical_sha256"].startswith("sha256:")
    assert cam["annotations_count"] == 2
    # videos + features are provenance only — no decode, no numpy import
    (vid,) = [r for r in records if r["kind"] == "video_file"]
    assert vid["relative_path"].endswith("1_720p.mkv")
    (feat,) = [r for r in records if r["kind"] == "features"]
    assert feat["relative_path"].endswith(".npy")


def test_soccernet_events_unknown_label_blocks(tmp_path):
    src = _src(tmp_path, "snbadlabel", {f"{_SN_GAME}/Labels-v2.json":
               json.dumps({"annotations": [
                   {"position": 1000, "label": "Goal celebration"}]}).encode()})
    with pytest.raises(ValidationBlock, match="Yellow→red card") as ei:
        soccernet_events([src])
    # the offending label is named, not swallowed
    assert "Goal celebration" in str(ei.value)
    assert "17 SoccerNet classes" in str(ei.value)


def test_soccernet_events_unparseable_labels_block(tmp_path):
    src = _src(tmp_path, "snbadjson",
               {f"{_SN_GAME}/Labels-v2.json": b'{"annotations": ['})
    with pytest.raises(ValidationBlock, match="does not parse") as ei:
        soccernet_events([src])
    assert "§15" in (ei.value.hint or "")


def test_soccernet_events_annotations_shape_blocks(tmp_path):
    src = _src(tmp_path, "snshape", {f"{_SN_GAME}/Labels-v2.json":
               json.dumps({"nope": 1}).encode()})
    with pytest.raises(ValidationBlock, match="`annotations` list"):
        soccernet_events([src])


def test_soccernet_events_no_artifacts_blocks(tmp_path):
    src = _src(tmp_path, "sngarbage",
               {"readme.txt": b"not a SoccerNet tree",
                "stats.csv": b"a,b\n1,2\n"})
    with pytest.raises(ValidationBlock, match="Labels-v2.json") as ei:
        soccernet_events([src])
    hint = ei.value.hint or ""
    assert "02_dataset_preparation §7" in hint
    assert "Labels-cameras.json" in hint and "video.ini" in hint


# -- catalog invariant + post-add suggestions ------------------------------

def test_dataset_types_catalog_matches_registry():
    """Every advertised transform must actually be registered — the
    catalog can never drift from what `prepare` can run (fail-closed)."""
    catalog = dataset_types()
    registered = set(registry_names())
    names = [t["name"] for t in catalog["supported"]]
    assert len(names) == len(set(names))
    for name in names:
        assert name in registered, f"catalog advertises unregistered {name!r}"
    assert {p["name"] for p in catalog["planned"]} >= {"audio"}


def test_dataset_types_video_transforms_registered():
    """Video is supported now, not planned: both transforms are cataloged
    AND registered, and the honest-gap list keeps what is still missing
    (audio) — the gap list never shrinks by faking support."""
    catalog = dataset_types()
    supported = {t["name"] for t in catalog["supported"]}
    assert {"video_clips", "soccernet_events"} <= supported
    registered = set(registry_names())
    for name in supported:
        assert name in registered, f"catalog advertises unregistered {name!r}"
    assert {"video_clips", "soccernet_events"} <= registered
    planned = {p["name"] for p in catalog["planned"]}
    assert "video" not in planned
    assert "audio" in planned


def test_suggest_transforms_priority_and_hints():
    assert suggest_transforms(["gt.csv", "readme.md"]) == [
        "text_corpus", "mot_challenge", "tabular"]
    assert suggest_transforms(["a.xlsx"]) == ["tabular"]
    # .jsonl feeds both the generic table reader and the calibrator
    assert suggest_transforms(["preds.jsonl"]) == ["tabular", "calibration"]
    assert suggest_transforms(["imgs/0001_c1s1_000001_01.jpg"]) == ["reid_crops"]
    assert suggest_transforms(["annotations.json", "imgs/a.jpg"]) == ["coco_detection"]
    assert suggest_transforms(["raw.zip"]) == []   # ambiguous — no bad advice


def test_suggest_transforms_video_files():
    """Video extensions must route to the video transforms (.mkv feeds
    both SoccerNet and the generic clip probe — the user picks)."""
    assert suggest_transforms(["clips/match.mp4"]) == ["video_clips"]
    assert suggest_transforms(["clip.avi"]) == ["video_clips"]
    assert suggest_transforms(["clip.mov", "clip.webm"]) == ["video_clips"]
    assert suggest_transforms(["league/game/1_720p.mkv"]) == [
        "video_clips", "soccernet_events"]
    assert suggest_transforms(["league/game/video.ini"]) == ["soccernet_events"]


# -- discovery surfaces point at the catalog -------------------------------

def test_prepare_unknown_model_hint_points_to_types(ws, home, capsys):
    _ingestion(ws, "models:\n  m:\n    transform: text_corpus\n    "
                   "train_sources: []\n")
    assert main(["--root", str(ws), "prepare", "ghost"]) == 2
    err = capsys.readouterr().err
    assert "mlforge dataset types" in err


def test_add_prints_transform_suggestion(ws, home, tmp_path, capsys):
    d = _tree(tmp_path, "motsrc", {"gt.csv": MOT_CSV.encode(),
                                   "readme.md": b"hi"})
    assert _add(ws, "mtrack", d) == 0
    out = capsys.readouterr().out
    assert "likely transform: text_corpus / mot_challenge / tabular" in out
    assert "mlforge dataset types" in out
