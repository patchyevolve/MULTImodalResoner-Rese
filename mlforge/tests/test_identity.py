import pytest

from mlforge.errors import ValidationBlock
from mlforge.hashing import canonical_json, content_hash, run_spec_hash
from mlforge.ids import new_command_id, new_run_id, new_ulid
from mlforge.run_spec import RunSpec


def test_canonical_json_is_key_order_independent():
    a = {"b": 1, "a": {"z": True, "y": None}}
    b = {"a": {"y": None, "z": True}, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert content_hash(a) == content_hash(b)


def test_content_hash_prefixed():
    assert content_hash({"x": 1}).startswith("sha256:")


def test_run_spec_identity_stable_and_sensitive(spec):
    same = RunSpec(
        model=spec.model,
        train_datasets=spec.train_datasets,
        val_dataset=spec.val_dataset,
        semantic=dict(spec.semantic),
    )
    assert spec.identity == same.identity
    changed = dict(spec.semantic)
    changed["learning_rate"] = 2e-4
    other = RunSpec(
        model=spec.model,
        train_datasets=spec.train_datasets,
        val_dataset=spec.val_dataset,
        semantic=changed,
    )
    assert spec.identity != other.identity


def test_run_spec_requires_semantic_fields():
    with pytest.raises(ValidationBlock) as e:
        RunSpec(model="m", train_datasets=("d",), semantic={"seed": 1})
    assert "global_batch" in str(e.value)


def test_run_spec_unknown_schema_blocks():
    with pytest.raises(ValidationBlock):
        RunSpec.from_dict(
            {
                "schema_version": 99,
                "model": "m",
                "train_datasets": ["d"],
                "semantic": {},
            }
        )


def test_run_spec_hash_function_matches_property(spec):
    assert run_spec_hash(spec.to_dict()) == spec.identity


def test_ids_are_unique_and_prefixed():
    ids = [new_run_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert all(i.startswith("run_") for i in ids)
    assert all(len(new_ulid()) == 26 for _ in range(10))
    assert new_command_id().startswith("cmd_")
