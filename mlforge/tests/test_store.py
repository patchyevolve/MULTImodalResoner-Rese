"""Build step 2 tests — content store, registry transactions, GC.

Specs as executable checks (12_training_system.md §5, §6.1):
  * identity is content hash — dedup, corrupt blob never verifies
  * writes are atomic: staging → fsync → rename; partial blobs impossible
  * registry registration is transactional; uncommitted tmp swept by recover()
  * GC: reachability ONLY (never age/size), grace period keeps fresh
    orphans, active run leases block execution-mode GC
  * run refs live in the run folder — portability survives registry deletion
"""

from __future__ import annotations

import json
import os
import time

import pytest

from mlforge.errors import NotFound, PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash_bytes
from mlforge.store import ArtifactRegistry, ContentStore


@pytest.fixture
def store(tmp_path):
    return ContentStore(tmp_path / "store")


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "runs").mkdir(parents=True)
    return root


@pytest.fixture
def registry(workspace, store):
    return ArtifactRegistry(workspace, store)


def _mk_run(root, run_id):
    (root / "runs" / run_id).mkdir()


# -- ContentStore: identity + dedup -------------------------------------

def test_put_bytes_identity_is_content_hash(store):
    h = store.put_bytes(b"hello world")
    assert h == content_hash_bytes(b"hello world")
    assert h.startswith("sha256:")
    assert store.get_bytes(h) == b"hello world"


def test_identical_bytes_dedup_to_one_blob(store):
    h1 = store.put_bytes(b"same bytes")
    h2 = store.put_bytes(b"same bytes")
    assert h1 == h2
    assert len(store.list_hashes()) == 1


def test_different_bytes_different_hash(store):
    assert store.put_bytes(b"a") != store.put_bytes(b"b")


def test_malformed_hash_rejected(store):
    with pytest.raises(ValidationBlock):
        store.path_for("sha256:zz")  # wrong length
    with pytest.raises(ValidationBlock):
        store.path_for("md5:abcd")  # wrong algorithm


def test_get_missing_raises_not_found(store):
    with pytest.raises(NotFound):
        store.get_bytes("sha256:" + "0" * 64)


def test_verify_detects_corruption(store):
    h = store.put_bytes(b"checkpoint weights v17")
    assert store.verify(h)
    # corrupt the blob on disk (simulate bit rot / tampering)
    store.path_for(h).write_bytes(b"CORRUPTED!")
    assert not store.verify(h)
    with pytest.raises(ValidationBlock):
        store.get_bytes(h, verify=True)


def test_staging_files_never_visible_at_final_path(store):
    """A staging file exists only until the atomic rename — readers of the
    final path see nothing or complete bytes, never a partial write."""
    store.staging_dir.mkdir(parents=True)
    orphan = store.staging_dir / "partial.tmp"
    orphan.write_bytes(b"half writ")
    h = store.put_bytes(b"complete payload")
    assert store.path_for(h).is_file()
    # orphan did not affect identity; recover sweeps it
    removed = store.recover_staging()
    assert orphan.name in removed
    assert not orphan.exists()
    assert store.get_bytes(h) == b"complete payload"


def test_put_file_with_wrong_expected_hash_blocks(store, tmp_path):
    src = tmp_path / "weights.bin"
    src.write_bytes(b"actual content")
    wrong = content_hash_bytes(b"claimed content")
    with pytest.raises(ValidationBlock):
        store.put_file(src, chunk_hash=wrong)
    # nothing was committed
    assert store.list_hashes() == []


def test_put_file_with_correct_hash_commits(store, tmp_path):
    src = tmp_path / "weights.bin"
    src.write_bytes(b"actual content")
    h = store.put_file(src, chunk_hash=content_hash_bytes(b"actual content"))
    assert store.get_bytes(h) == b"actual content"


def test_put_file_missing_source(store, tmp_path):
    with pytest.raises(NotFound):
        store.put_file(tmp_path / "nope.bin")


def test_delete_is_available_for_gc(store):
    h = store.put_bytes(b"garbage")
    assert store.delete(h) is True
    assert not store.contains(h)
    assert store.delete(h) is False


# -- Registry: transactional registration --------------------------------

def test_register_requires_blob_in_store(registry, store):
    with pytest.raises(NotFound):
        registry.register("sha256:" + "0" * 64, kind="checkpoint")
    h = store.put_bytes(b"model blob")
    registry.register(h, kind="checkpoint", meta={"epoch": 17})
    assert registry.is_registered(h)
    entry = registry.get_entry(h)
    assert entry["kind"] == "checkpoint"
    assert entry["committed"] is True
    assert entry["meta"] == {"epoch": 17}


def test_register_is_atomic_no_tmp_left_behind(registry, store):
    h = store.put_bytes(b"payload")
    registry.register(h, kind="dataset_index")
    assert list(registry.index_dir.glob("*.tmp")) == []


def test_register_missing_blob_leaves_no_entry(registry, store):
    with pytest.raises(NotFound):
        registry.register("sha256:" + "1" * 64, kind="checkpoint")
    assert list(registry.index_dir.glob("*")) == []


def test_recover_sweeps_uncommitted_tmp_and_completes_entries(registry, store, workspace):
    """Crash windows: (a) registry tmp before commit → swept;
    (b) blob committed but index write never happened → entry completed."""
    h = store.put_bytes(b"orphan blob")           # (b): blob exists, no entry
    registry.index_dir.mkdir(parents=True, exist_ok=True)
    stale_tmp = registry.index_dir / ".deadbeef.tmp"
    stale_tmp.write_text("{partial")               # (a): uncommitted garbage
    store.staging_dir.mkdir(parents=True, exist_ok=True)
    staging_tmp = store.staging_dir / "crash.tmp"
    staging_tmp.write_bytes(b"half")

    report = registry.recover()

    assert not stale_tmp.exists()
    assert not staging_tmp.exists()
    assert h in report["entries_completed"]
    assert registry.is_registered(h)


def test_recover_is_idempotent(registry, store):
    h = store.put_bytes(b"stable")
    registry.recover()
    registry.recover()
    assert registry.is_registered(h)
    assert registry.get_entry(h)["kind"] in ("recovered",)  # completed once


# -- Run references (GC roots live with runs) ----------------------------

def test_link_requires_existing_run(registry):
    with pytest.raises(NotFound):
        registry.link("run_MISSING", "sha256:" + "a" * 64)


def test_link_is_a_set_atomic_rewrite(registry, workspace):
    _mk_run(workspace, "run_A")
    h = "sha256:" + "b" * 64
    registry.link("run_A", h)
    registry.link("run_A", h)  # idempotent
    refs = json.loads(registry.refs_path("run_A").read_text())
    assert refs["refs"] == [h]


def test_all_run_refs_survive_corrupt_one(registry, workspace):
    _mk_run(workspace, "run_A")
    _mk_run(workspace, "run_B")
    registry.link("run_A", "sha256:" + "c" * 64)
    registry.refs_path("run_B").write_text("{corrupt", encoding="utf-8")
    refs = registry.all_run_refs()
    assert "run_A" in refs
    assert "run_B" not in refs  # skipped, GC does not crash on one bad run


# -- GC: reachability only ------------------------------------------------

def test_gc_keeps_reachable_blob(workspace, store, registry):
    h = store.put_bytes(b"needed by run")
    _mk_run(workspace, "run_A")
    registry.link("run_A", h)
    store.path_for(h).touch()  # age it past grace? not needed — reachable wins
    report = registry.gc(grace_seconds=0, dry_run=False)
    assert h not in report.swept
    assert store.contains(h)


def test_gc_sweeps_unreachable_older_than_grace(workspace, store, registry):
    h = store.put_bytes(b"nobody references this")
    blob = store.path_for(h)
    old = time.time() - 30 * 24 * 3600
    os.utime(blob, (old, old))
    report = registry.gc(grace_seconds=7 * 24 * 3600, dry_run=False)
    assert h in report.swept
    assert not store.contains(h)
    assert not registry.entry_path(h).exists()


def test_gc_grace_period_protects_fresh_orphans(workspace, store, registry):
    h = store.put_bytes(b"fresh, unreferenced")
    report = registry.gc(grace_seconds=7 * 24 * 3600, dry_run=False)
    assert h in report.kept_grace
    assert store.contains(h)  # NOT swept by age — only unreachable AND old


def test_gc_dry_run_is_default_and_deletes_nothing(workspace, store, registry):
    h = store.put_bytes(b"unreferenced")
    old = time.time() - 90 * 24 * 3600
    os.utime(store.path_for(h), (old, old))
    report = registry.gc()  # dry_run default True
    assert report.dry_run is True
    assert h in report.swept
    assert store.contains(h)  # reported but not deleted


def test_gc_never_uses_age_alone(workspace, store, registry):
    """The ONE criterion is unreachability: an old blob that IS referenced
    is kept (age alone must never sweep)."""
    h = store.put_bytes(b"old but referenced")
    _mk_run(workspace, "run_A")
    registry.link("run_A", h)
    old = time.time() - 365 * 24 * 3600
    os.utime(store.path_for(h), (old, old))
    report = registry.gc(grace_seconds=0, dry_run=False)
    assert h not in report.swept
    assert store.contains(h)


def test_active_lease_blocks_gc_execution(workspace, store, registry):
    h = store.put_bytes(b"referenced by live run")
    _mk_run(workspace, "run_A")
    registry.link("run_A", h)
    orphan = store.put_bytes(b"old orphan")
    old = time.time() - 90 * 24 * 3600
    os.utime(store.path_for(orphan), (old, old))
    (workspace / "runs" / "run_A" / ".lease").write_text("sess", encoding="utf-8")

    with pytest.raises(PreconditionFailed):
        registry.gc(grace_seconds=0, dry_run=False)
    # dry-run still allowed (reports only)
    report = registry.gc(grace_seconds=0, dry_run=True)
    assert report.skipped_lease == ["run_A"]


def test_registry_deletion_does_not_break_reachability(workspace, store, registry):
    """Portable run folder: refs live with the run, so deleting the whole
    registry index does not make artifacts unreachable."""
    h = store.put_bytes(b"portable")
    _mk_run(workspace, "run_A")
    registry.link("run_A", h)
    for entry in registry.index_dir.glob("*.json"):
        entry.unlink()
    assert h in registry.reachable()


def test_gc_report_shape(workspace, store, registry):
    report = registry.gc()
    keys = {"dry_run", "reachable", "swept", "kept_grace", "skipped_lease"}
    assert set(report.to_dict()) == keys
