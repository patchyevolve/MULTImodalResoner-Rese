"""Build step 10 tests — resume/retrain/finetune flows + lineage DAG
(12 §5, §15; 13 §4.3/§5.5, §6.3/§6.4).

Specs as executable checks:
  * lineage.json is written once at run creation (train = null parents;
    fork/retrain/finetune = parent set) and never rewritten (12 §15.2)
  * lineage_chain walks root-first with cycle/missing-parent guards
  * --set / --config semantic overrides accept ONLY allowlisted keys —
    a typo BLOCKs, never a silent identity (12 §4 fail-closed)
  * fork with no identity change BLOCKs; the parent's spec never mutates
  * retrain sources from the LATEST run of that model (13 §6.3)
  * completion publishes the model (name:vN, provenance, artifact) and
    version bumps per name (13 §5.5)
  * finetune: NEW run + NEW model, base model → USED_AS_FINE_TUNE_BASE,
    never mutated; repeated finetunes from one base are tolerated
  * CLI: SHOW DELTA → confirm → CREATE; --command-id dedupe; exit codes
    (1 block, 2 not found, 4 pending `model import`)
"""

from __future__ import annotations

import json

import pytest

from mlforge.cli.main import main
from mlforge.errors import NotFound, ValidationBlock
from mlforge.leases import RunLeaseManager, provide_run_lease
from mlforge.lineage import (
    canonicalize_overrides,
    parse_model_ref,
    parse_overrides,
    read_lineage,
    spec_delta,
    write_lineage,
)
from mlforge.run_spec import RunSpec
from mlforge.validation import RESUME_GATE_STEPS, provide_pass
from mlforge.workflow import WorkflowAPI

GATE_PASS_KEYS = [s.id for s in RESUME_GATE_STEPS if s.id not in ("manifest", "schema")]


def _spec(**semantic_overrides) -> RunSpec:
    semantic = {
        "optimizer": "adamw", "learning_rate": 1e-4, "scheduler": "cosine",
        "loss": "l1", "seed": 42, "global_batch": 32, "epochs": 50,
        "precision_policy": "bf16",
    }
    semantic.update(semantic_overrides)
    return RunSpec(
        model="rf_detr_s", train_datasets=("coco_2017:v1",), semantic=semantic,
    )


class _TestWorkflow(WorkflowAPI):
    """Test-scale preflight floors (same env adjustment as 6b tests)."""

    def preflight_run(self, run_id, preflight=None, **kw):
        kw["gpu_required"] = False
        kw.setdefault("checkpoint_bytes", 1 << 20)
        kw.setdefault("log_bytes", 1 << 20)
        kw.setdefault("safety_margin_bytes", 1 << 20)
        kw.setdefault("min_ram_bytes", 1 << 20)
        return super().preflight_run(run_id, preflight, **kw)


def wf_factory(root):
    providers = {k: provide_pass(f"{k} verified") for k in GATE_PASS_KEYS}
    providers["lease"] = provide_run_lease(RunLeaseManager(root))
    return _TestWorkflow(root, gate_providers=providers)


@pytest.fixture
def no_supervisor(monkeypatch):
    import mlforge.cli.main as cli

    monkeypatch.setattr(
        cli, "ensure_supervisor",
        lambda root, **kw: {"started": False, "pid": 99999},
    )


def _complete(wf: WorkflowAPI, run_id: str) -> str:
    """Drive a run to COMPLETED so its model gets published."""
    wf.begin_validation(run_id)
    wf.validation_pass(run_id)
    wf.preflight_pass(run_id)
    wf.complete(run_id)
    return run_id


# -- lineage.json / chain ---------------------------------------------------

def test_train_run_writes_lineage_with_null_parents(tmp_path):
    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    lin = wf.get_lineage(h.run_id)
    assert lin["origin"] == "train"
    assert lin["parent_run"] is None and lin["parent_model"] is None
    assert (tmp_path / "runs" / h.run_id / "lineage.json").is_file()
    # full edge set is always on disk (write_lineage completes defaults)
    on_disk = json.loads(
        (tmp_path / "runs" / h.run_id / "lineage.json").read_text())
    assert set(on_disk) == {
        "schema_version", "origin", "parent_run", "parent_run_spec_hash",
        "parent_model", "parent_checkpoint",
    }


def test_run_created_event_carries_origin_and_parent(tmp_path):
    wf = WorkflowAPI(tmp_path)
    parent = wf.create_run(_spec())
    plan = wf.preview_fork(parent.run_id, overrides=["lr=2e-4"])
    child = wf.create_from_plan(plan)
    events = wf.get_run_events(child.run_id)
    created = [e for e in events if e["event"] == "run_created"]
    assert created and created[0]["origin"] == "fork"
    assert created[0]["parent_run"] == parent.run_id


def test_lineage_chain_root_first(tmp_path):
    wf = WorkflowAPI(tmp_path)
    root_run = wf.create_run(_spec())
    fork = wf.create_from_plan(
        wf.preview_fork(root_run.run_id, overrides=["lr=2e-4"]))
    grandchild = wf.create_from_plan(
        wf.preview_fork(fork.run_id, overrides=["epochs=60"]))
    chain = wf.lineage_chain(grandchild.run_id)
    assert [c["run_id"] for c in chain] == [
        root_run.run_id, fork.run_id, grandchild.run_id]
    assert [c["origin"] for c in chain] == ["train", "fork", "fork"]


def test_lineage_chain_cycle_blocks(tmp_path):
    wf = WorkflowAPI(tmp_path)
    a = wf.create_run(_spec())
    b = wf.create_from_plan(wf.preview_fork(a.run_id, overrides=["lr=2e-4"]))
    # tamper: corrupt the edges into a loop (guards are on READ paths)
    for run_id, parent in ((a.run_id, b.run_id), (b.run_id, a.run_id)):
        p = tmp_path / "runs" / run_id / "lineage.json"
        d = json.loads(p.read_text())
        d["parent_run"] = parent
        p.write_text(json.dumps(d))
    with pytest.raises(ValidationBlock) as exc:
        wf.lineage_chain(b.run_id)
    assert "cycle" in str(exc.value)


def test_lineage_chain_missing_parent_blocks(tmp_path):
    wf = WorkflowAPI(tmp_path)
    a = wf.create_run(_spec())
    child = wf.create_from_plan(
        wf.preview_fork(a.run_id, overrides=["lr=2e-4"]))
    import shutil
    shutil.rmtree(tmp_path / "runs" / a.run_id)
    with pytest.raises(ValidationBlock) as exc:
        wf.lineage_chain(child.run_id)
    assert "missing" in str(exc.value)


def test_get_lineage_unknown_run_not_found(tmp_path):
    wf = WorkflowAPI(tmp_path)
    with pytest.raises(NotFound):
        wf.get_lineage("run_NOPE")


def test_write_lineage_immutable_on_differing_rewrite(tmp_path):
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    write_lineage(run_dir, {"origin": "fork", "parent_run": "run_A"})
    # identical / defaulted rewrite is a no-op...
    write_lineage(run_dir, {"origin": "fork", "parent_run": "run_A"})
    # ...but a DIFFERING rewrite is an immutable-edge violation
    with pytest.raises(ValidationBlock) as exc:
        write_lineage(run_dir, {"origin": "fork", "parent_run": "run_B"})
    assert "immutable" in str(exc.value)


def test_corrupt_lineage_blocks(tmp_path):
    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    (tmp_path / "runs" / h.run_id / "lineage.json").write_text("{not json")
    with pytest.raises(ValidationBlock):
        wf.get_lineage(h.run_id)


def test_missing_lineage_file_reads_as_train_defaults(tmp_path):
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    assert read_lineage(run_dir)["origin"] == "train"  # pre-step-10 runs


# -- overrides / derivation -------------------------------------------------

def test_parse_overrides_alias_and_json_number():
    out = parse_overrides(["lr=2e-4", "batch=64", "epochs=10"])
    assert out == {"learning_rate": 0.0002, "global_batch": 64, "epochs": 10}


def test_parse_overrides_string_fallback():
    out = parse_overrides(["precision=bf16"])
    assert out == {"precision_policy": "bf16"}


def test_parse_overrides_unknown_key_blocks():
    with pytest.raises(ValidationBlock) as exc:
        parse_overrides(["learnign_rate=1e-3"])
    # the hint lists the allowlist (asserted on render — str() has no hint)
    assert "learning_rate" in exc.value.render()
    assert "runtime config" in exc.value.render()


def test_parse_overrides_requires_key_value():
    with pytest.raises(ValidationBlock):
        parse_overrides(["lr"])


def test_config_semantic_typo_blocks_before_anything_exists(tmp_path):
    """--config semantic goes through the same allowlist as --set."""
    wf = WorkflowAPI(tmp_path)
    parent = wf.create_run(_spec())
    with pytest.raises(ValidationBlock):
        wf.preview_fork(parent.run_id, overrides={"weight_decwy": 0.01})
    assert canonicalize_overrides({"lr": 1e-3}) == {"learning_rate": 1e-3}


def test_spec_delta_reports_only_changes():
    old = _spec()
    new = _spec(epochs=10)
    assert spec_delta(old, new) == {"epochs": {"from": 50, "to": 10}}
    assert spec_delta(old, _spec()) == {}


def test_model_ref_parsing_variants():
    assert parse_model_ref("model://rf_detr_s:v2") == ("rf_detr_s", "v2")
    assert parse_model_ref("rf_detr_s:v2") == ("rf_detr_s", "v2")
    assert parse_model_ref("rf_detr_s") == ("rf_detr_s", None)
    with pytest.raises(NotFound):
        parse_model_ref("rf_detr_s:2")  # version must be vN


# -- fork -------------------------------------------------------------------

def test_fork_identity_unchanged_blocks(tmp_path):
    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    with pytest.raises(ValidationBlock) as exc:
        wf.preview_fork(h.run_id)
    assert "fork" in str(exc.value)
    assert "resume" in exc.value.render()


def test_fork_does_not_mutate_parent_spec(tmp_path):
    wf = WorkflowAPI(tmp_path)
    parent = wf.create_run(_spec())
    before = (tmp_path / "runs" / parent.run_id / "run_spec.json").read_text()
    plan = wf.preview_fork(parent.run_id, overrides=["lr=2e-4", "epochs=10"])
    wf.create_from_plan(plan)
    after = (tmp_path / "runs" / parent.run_id / "run_spec.json").read_text()
    assert before == after  # parent identity frozen (12 §4)


def test_fork_lineage_edges_and_delta(tmp_path):
    wf = WorkflowAPI(tmp_path)
    parent = wf.create_run(_spec())
    plan = wf.preview_fork(parent.run_id, overrides=["lr=2e-4"])
    assert plan.lineage["origin"] == "fork"
    assert plan.lineage["parent_run"] == parent.run_id
    assert plan.lineage["parent_run_spec_hash"] == parent.run_spec_hash
    assert plan.delta == {"learning_rate": {"from": 0.0001, "to": 0.0002}}
    child = wf.create_from_plan(plan)
    assert wf.get_lineage(child.run_id)["parent_run"] == parent.run_id


# -- retrain ----------------------------------------------------------------

def test_retrain_source_picks_latest_run(tmp_path):
    wf = WorkflowAPI(tmp_path)
    first = wf.create_run(_spec(epochs=50))
    second = wf.create_run(_spec(epochs=90))  # newer ULID (time-sorted)
    source_id, source = wf.retrain_source("rf_detr_s")
    assert source_id == second.run_id
    assert source.semantic["epochs"] == 90
    assert source_id != first.run_id


def test_retrain_source_missing_model_not_found(tmp_path):
    wf = WorkflowAPI(tmp_path)
    with pytest.raises(NotFound) as exc:
        wf.retrain_source("ghost_net")
    assert "train one first" in exc.value.render()


def test_retrain_preview_starting_state_and_empty_delta(tmp_path):
    wf = WorkflowAPI(tmp_path)
    parent = wf.create_run(_spec())
    plan = wf.preview_retrain("rf_detr_s")
    assert plan.kind == "retrain"
    assert plan.source_run_id == parent.run_id
    assert plan.lineage["origin"] == "retrain"
    assert plan.delta == {}  # same config unless changed (13 §6.3)
    assert plan.starting_state == "pretrained base weights (fresh init)"


def test_retrain_with_changes_delta(tmp_path):
    wf = WorkflowAPI(tmp_path)
    wf.create_run(_spec())
    plan = wf.preview_retrain(
        "rf_detr_s", overrides=["epochs=10"], dataset="custom_v2")
    assert plan.delta["epochs"] == {"from": 50, "to": 10}
    assert plan.delta["train_datasets"]["to"] == ["custom_v2"]


# -- completion publishes the model (13 §5.5) -------------------------------

def test_complete_publishes_model_v1(tmp_path):
    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    _complete(wf, h.run_id)
    (model,) = wf.list_models()
    assert model["name"] == "rf_detr_s"
    assert model["version"] == "v1"
    assert model["state"] == "AVAILABLE"
    assert model["run_id"] == h.run_id
    assert model["run_spec_hash"] == h.run_spec_hash
    sidecar = json.loads(
        (tmp_path / "models" / model["model_id"] / "model.json").read_text())
    assert sidecar["version"] == "v1" and sidecar["artifact_hash"] is None


def test_second_run_same_model_publishes_v2(tmp_path):
    wf = WorkflowAPI(tmp_path)
    _complete(wf, wf.create_run(_spec()).run_id)
    _complete(wf, wf.create_run(_spec()).run_id)
    versions = [m["version"] for m in wf.list_models()]
    assert versions == ["v1", "v2"]


def test_model_artifact_hash_from_commit_marker(tmp_path):
    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    ckpt = tmp_path / "runs" / h.run_id / "checkpoints" / "ckpt-000001"
    ckpt.mkdir(parents=True)
    (ckpt / "COMMIT").write_text("sha256:abc123manifest")
    _complete(wf, h.run_id)
    (model,) = wf.list_models()
    assert model["artifact_hash"] == "sha256:abc123manifest"


def test_resolve_model_ref_variants(tmp_path):
    wf = WorkflowAPI(tmp_path)
    _complete(wf, wf.create_run(_spec()).run_id)
    _complete(wf, wf.create_run(_spec()).run_id)  # v2
    assert wf.resolve_model("rf_detr_s:v1")["version"] == "v1"
    assert wf.resolve_model("model://rf_detr_s:v2")["version"] == "v2"
    assert wf.resolve_model("rf_detr_s")["version"] == "v2"  # bare = latest
    with pytest.raises(NotFound) as exc:
        wf.resolve_model("rf_detr_s:v7")
    assert "available versions: rf_detr_s:v1, rf_detr_s:v2" in exc.value.render()


def test_list_models_empty(tmp_path):
    assert WorkflowAPI(tmp_path).list_models() == []


# -- finetune (13 §6.4) -----------------------------------------------------

def _publish_base(tmp_path, wf: WorkflowAPI | None = None) -> tuple[str, str]:
    """Complete a run and return (model_id, name:vN) of the NEWEST model."""
    wf = wf or WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    _complete(wf, h.run_id)
    model = wf.list_models()[-1]  # sorted by name, version — newest last
    return model["model_id"], f"{model['name']}:{model['version']}"


def test_finetune_lineage_and_strategy(tmp_path):
    wf = WorkflowAPI(tmp_path)
    _, ref = _publish_base(tmp_path, wf)
    plan = wf.preview_finetune(ref, strategy="lora", dataset="custom_v2")
    assert plan.kind == "finetune"
    assert plan.lineage["origin"] == "finetune"
    assert plan.lineage["parent_model"] == ref
    assert plan.lineage["parent_run"] is not None
    assert plan.spec.semantic["finetune_strategy"] == "lora"
    assert "NOT modified" in plan.warnings[0]
    child = wf.create_from_plan(plan)
    lin = wf.get_lineage(child.run_id)
    assert lin["parent_model"] == ref and lin["parent_run"]


def test_finetune_records_base_model_state(tmp_path):
    wf = WorkflowAPI(tmp_path)
    _base_id, ref = _publish_base(tmp_path, wf)
    plan = wf.preview_finetune(ref, strategy="freeze_backbone")
    wf.create_from_plan(plan)
    base = wf.resolve_model(ref)
    assert base["state"] == "USED_AS_FINE_TUNE_BASE"


def test_double_finetune_from_same_base_tolerated(tmp_path):
    wf = WorkflowAPI(tmp_path)
    _, ref = _publish_base(tmp_path, wf)
    wf.create_from_plan(wf.preview_finetune(ref, strategy="lora"))
    # second use of an already-used base: state cannot re-enter the
    # transition — the lineage edge is the record, creation must succeed
    child2 = wf.create_from_plan(wf.preview_finetune(ref, strategy="adapter"))
    assert wf.get_lineage(child2.run_id)["parent_model"] == ref


def test_finetune_invalid_strategy_blocks(tmp_path):
    wf = WorkflowAPI(tmp_path)
    _, ref = _publish_base(tmp_path, wf)
    with pytest.raises(ValidationBlock):
        wf.preview_finetune(ref, strategy="magic")


def test_finetune_from_unversioned_bare_ref_takes_latest(tmp_path):
    rf = WorkflowAPI(tmp_path)
    _publish_base(tmp_path, rf)
    _publish_base(tmp_path, rf)  # v2
    plan = rf.preview_finetune("rf_detr_s")  # bare → v2
    assert plan.base_model == "rf_detr_s:v2"


# -- CLI --------------------------------------------------------------------

def test_cli_fork_display_and_cancel(tmp_path, capsys, monkeypatch):
    wf = WorkflowAPI(tmp_path)
    parent = wf.create_run(_spec())
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    code = main(["--root", str(tmp_path), "fork", parent.run_id,
                 "--set", "lr=2e-4"])
    assert code == 0
    out = capsys.readouterr().out
    assert f"Previous run:  {parent.run_id}" in out
    assert "New run:" in out
    assert "learning_rate: 0.0001 → 0.0002" in out
    assert "Cancelled" in out
    assert [r["id"] for r in wf.list_runs()] == [parent.run_id]  # nothing made


def test_cli_fork_no_change_blocks(tmp_path, capsys):
    wf = WorkflowAPI(tmp_path)
    parent = wf.create_run(_spec())
    code = main(["--root", str(tmp_path), "fork", parent.run_id])
    assert code == 1
    assert "fork" in capsys.readouterr().err


def test_cli_fork_unknown_run_exit_2(tmp_path, capsys):
    code = main(["--root", str(tmp_path), "fork", "run_NOPE",
                 "--set", "lr=2e-4"])
    assert code == 2
    assert "[NOT_FOUND]" in capsys.readouterr().err


def test_cli_fork_yes_creates_child(tmp_path, capsys, no_supervisor):
    wf = wf_factory(tmp_path)
    parent = wf.create_run(_spec())
    code = main(["--root", str(tmp_path), "fork", parent.run_id,
                 "--set", "lr=2e-4", "--yes", "--json"],
                wf_factory=wf_factory)
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "started"
    child_id = result["run_id"]
    assert wf.get_lineage(child_id)["parent_run"] == parent.run_id
    assert wf.get_run_state(child_id) == "READY"


def test_cli_fork_command_id_dedupe(tmp_path, capsys, no_supervisor):
    wf = wf_factory(tmp_path)
    parent = wf.create_run(_spec())
    argv = ["--root", str(tmp_path), "fork", parent.run_id,
            "--set", "lr=2e-4", "--yes", "--json", "--command-id", "cmd-77"]
    assert main(argv, wf_factory=wf_factory) == 0
    first = json.loads(capsys.readouterr().out)
    # retry with the SAME id (even a different --set) returns the ORIGINAL
    assert main(argv, wf_factory=wf_factory) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["run_id"] == first["run_id"]
    runs = [r["id"] for r in wf.list_runs()]
    assert runs == [parent.run_id, first["run_id"]]  # no second child
    events = wf.get_run_events(first["run_id"])
    assert any(e["event"] == "COMMAND_DEDUPED" for e in events)


def test_cli_retrain_display_lines(tmp_path, capsys, monkeypatch):
    wf = WorkflowAPI(tmp_path)
    parent = wf.create_run(_spec())
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    code = main(["--root", str(tmp_path), "retrain", "rf_detr_s"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Retraining rf_detr_s" in out
    assert f"Previous run:  {parent.run_id}" in out
    assert "Starting state: pretrained base weights (fresh init)" in out
    assert "Everything else: same as previous configuration" in out
    assert "This will create a NEW training run." in out


def test_cli_retrain_dataset_change_display(tmp_path, capsys, monkeypatch):
    wf = WorkflowAPI(tmp_path)
    wf.create_run(_spec())
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    code = main(["--root", str(tmp_path), "retrain", "rf_detr_s",
                 "--dataset", "custom_v2", "--set", "epochs=10"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Dataset changed: coco_2017:v1 → custom_v2" in out
    assert "This is a new experiment." in out
    assert "epochs: 50 → 10" in out


def test_cli_retrain_not_found_exit_2(tmp_path, capsys):
    code = main(["--root", str(tmp_path), "retrain", "rf_detr_s"])
    assert code == 2
    assert "[NOT_FOUND]" in capsys.readouterr().err


def test_cli_retrain_yes_creates_child(tmp_path, capsys, no_supervisor):
    wf = wf_factory(tmp_path)
    parent = wf.create_run(_spec())
    code = main(["--root", str(tmp_path), "retrain", "rf_detr_s",
                 "--yes", "--json"], wf_factory=wf_factory)
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "started"
    lin = wf.get_lineage(result["run_id"])
    assert lin["origin"] == "retrain" and lin["parent_run"] == parent.run_id


def test_cli_finetune_display_and_yes(tmp_path, capsys, no_supervisor):
    wf = wf_factory(tmp_path)
    _, ref = _publish_base(tmp_path, wf)
    # display first: SHOW (no launch) via a declined confirm
    import builtins
    answers = iter(["n"])
    real_input = builtins.input
    builtins.input = lambda _p: next(answers)
    try:
        code = main(["--root", str(tmp_path), "finetune", ref,
                     "--strategy", "lora"], wf_factory=wf_factory)
    finally:
        builtins.input = real_input
    assert code == 0
    out = capsys.readouterr().out
    assert f"Fine-tuning {ref}" in out
    assert "NEW run + NEW model" in out
    assert "Strategy:      lora" in out
    assert "NOT modified" in out

    # now the real create
    code = main(["--root", str(tmp_path), "finetune", ref,
                 "--strategy", "lora", "--yes", "--json"],
                wf_factory=wf_factory)
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "started"
    lin = wf.get_lineage(result["run_id"])
    assert lin["origin"] == "finetune" and lin["parent_model"] == ref
    assert wf.resolve_model(ref)["state"] == "USED_AS_FINE_TUNE_BASE"


def test_cli_finetune_no_base_model_exit_2(tmp_path, capsys):
    code = main(["--root", str(tmp_path), "finetune", "ghost:v1"])
    assert code == 2
    assert "[NOT_FOUND]" in capsys.readouterr().err


def test_cli_bad_strategy_argparse_exit_2(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--root", str(tmp_path), "finetune", "m", "--strategy", "nope"])
    assert exc.value.code == 2


def test_cli_unknown_set_key_blocks_exit_1(tmp_path, capsys):
    wf = WorkflowAPI(tmp_path)
    parent = wf.create_run(_spec())
    code = main(["--root", str(tmp_path), "fork", parent.run_id,
                 "--set", "learnign_rate=1e-3", "--yes"])
    assert code == 1
    err = capsys.readouterr().err
    assert "unknown semantic key" in err and "learning_rate" in err
    assert len(wf.list_runs()) == 1  # nothing created


def test_cli_model_list_empty(tmp_path, capsys):
    code = main(["--root", str(tmp_path), "model", "list"])
    assert code == 0
    assert "no models" in capsys.readouterr().out


def test_cli_model_list_and_inspect(tmp_path, capsys):
    wf = WorkflowAPI(tmp_path)
    run_id = _complete(wf, wf.create_run(_spec()).run_id)
    assert main(["--root", str(tmp_path), "model", "list"]) == 0
    out = capsys.readouterr().out
    assert "rf_detr_s:v1" in out and "AVAILABLE" in out

    assert main(["--root", str(tmp_path), "model", "inspect",
                 "model://rf_detr_s:v1"]) == 0
    out = capsys.readouterr().out
    assert "MODEL                   rf_detr_s:v1" in out
    assert run_id in out
    assert "origin" in out and "train" in out

    assert main(["--root", str(tmp_path), "model", "inspect",
                 "rf_detr_s:v1", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["version"] == "v1"
    assert data["lineage"]["origin"] == "train"


def test_cli_model_import_missing_path_exit_2(tmp_path, capsys):
    # build step 11: `model import` is implemented — a path that is not a
    # package directory is a clean NotFound (13 §7), never exit 4
    code = main(["--root", str(tmp_path), "model", "import", "x_missing"])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_cli_inspect_run_includes_lineage(tmp_path, capsys):
    wf = WorkflowAPI(tmp_path)
    parent = wf.create_run(_spec())
    child = wf.create_from_plan(
        wf.preview_fork(parent.run_id, overrides=["lr=2e-4"]))
    assert main(["--root", str(tmp_path), "inspect", child.run_id]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["lineage"]["origin"] == "fork"
    assert data["lineage"]["parent_run"] == parent.run_id


def test_cli_inspect_unknown_id_exit_2(tmp_path, capsys):
    code = main(["--root", str(tmp_path), "inspect", "run_NOPE"])
    assert code == 2
    assert "[NOT_FOUND]" in capsys.readouterr().err
