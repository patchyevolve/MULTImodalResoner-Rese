"""Direct unit tests for ops/bundle.py — the packaging module.

Workflow tests exercise the happy path of `mlforge package`; these target
the refusal paths the mutation baseline scored weakest for this file
(bundle kill rate 50.6 % of tested mutants, 79/156 — an earlier draft
misattributed exporting's 37.6 % here): write immutability,
load_bundle's NotFound/corrupt/non-object triple, and list_bundles'
absent-dir/skip-partial/corrupt/filter edges.
"""

from __future__ import annotations

import json

import pytest

from mlforge.errors import NotFound, ValidationBlock
from mlforge.ops.bundle import (
    BUNDLES_DIR,
    component_integrity,
    list_bundles,
    load_bundle,
    write_bundle,
)


def _write_bundle(root, bundle_id: str = "bdl_1") -> None:
    return write_bundle(
        root,
        record={"bundle_id": bundle_id, "model_id": "m1"},
        model_spec={"inference_contract": {}},
        provenance={"by": "test"},
        integrity={"model_spec": "h"},
        weights=b"weights-payload",
    )


def _seed_bundle_json(root, payload, bundle_id: str = "bdl_1"):
    d = root / BUNDLES_DIR / bundle_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "bundle.json").write_text(payload, encoding="utf-8")
    return d


def test_write_bundle_weights_land_then_second_write_refused(tmp_path):
    d = _write_bundle(tmp_path)
    assert (d / "model.safetensors").read_bytes() == b"weights-payload"
    assert (d / "bundle.json").is_file()
    assert not list(d.glob("*.tmp"))
    with pytest.raises(ValidationBlock, match="already exists"):
        _write_bundle(tmp_path)


def test_load_bundle_missing_is_not_found(tmp_path):
    with pytest.raises(NotFound, match="not found"):
        load_bundle(tmp_path, "bdl_nope")


def test_load_bundle_corrupt_json_is_validation_block(tmp_path):
    _seed_bundle_json(tmp_path, "{not json", "bdl_bad")
    with pytest.raises(ValidationBlock, match="bundle unreadable"):
        load_bundle(tmp_path, "bdl_bad")


def test_load_bundle_non_object_is_validation_block(tmp_path):
    _seed_bundle_json(tmp_path, "[]", "bdl_list")
    with pytest.raises(ValidationBlock, match="must be a JSON object"):
        load_bundle(tmp_path, "bdl_list")


def test_load_bundle_happy_returns_dict(tmp_path):
    _seed_bundle_json(tmp_path, json.dumps({"bundle_id": "bdl_1"}))
    assert load_bundle(tmp_path, "bdl_1") == {"bundle_id": "bdl_1"}


def test_list_bundles_absent_dir_is_empty(tmp_path):
    assert list_bundles(tmp_path) == []


def test_list_bundles_skips_directory_without_manifest(tmp_path):
    (tmp_path / BUNDLES_DIR / "bdl_partial").mkdir(parents=True)
    assert list_bundles(tmp_path) == []


def test_list_bundles_corrupt_manifest_raises(tmp_path):
    _seed_bundle_json(tmp_path, "{oops", "bdl_bad")
    with pytest.raises(ValidationBlock, match="bundle unreadable"):
        list_bundles(tmp_path)


def test_list_bundles_filters_by_model_id(tmp_path):
    _seed_bundle_json(tmp_path, json.dumps({"bundle_id": "a", "model_id": "m1"}), "a")
    _seed_bundle_json(tmp_path, json.dumps({"bundle_id": "b", "model_id": "m2"}), "b")
    all_bundles = list_bundles(tmp_path)
    assert [b["bundle_id"] for b in all_bundles] == ["a", "b"]
    assert [b["bundle_id"] for b in list_bundles(tmp_path, model_id="m1")] == ["a"]
    assert list_bundles(tmp_path, model_id="m9") == []


def test_component_integrity_without_artifact_hash_has_no_weights():
    integ = component_integrity({"a": 1}, {"by": "t"}, None)
    assert set(integ) == {"model_spec", "provenance"}


def test_component_integrity_records_weights_hash():
    integ = component_integrity({"a": 1}, {"by": "t"}, "deadbeef")
    assert integ["weights"] == "deadbeef"
