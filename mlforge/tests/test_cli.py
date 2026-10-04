import json

from mlforge.cli.main import main
from mlforge.run_spec import RunSpec


def _spec() -> RunSpec:
    return RunSpec(
        model="rf_detr_s",
        train_datasets=("coco_2017:v1",),
        semantic={
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "scheduler": "cosine",
            "loss": "l1",
            "seed": 42,
            "global_batch": 32,
            "epochs": 50,
            "precision_policy": "bf16",
        },
    )


def test_version(capsys):
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0
    assert "mlforge" in capsys.readouterr().out


def test_status_empty(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "status"]) == 0
    assert "No runs." in capsys.readouterr().out


def test_status_json_after_run(tmp_path, capsys):
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    assert main(["--root", str(tmp_path), "status", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["id"] == h.run_id
    assert out[0]["state"] == "CREATED"


def test_inspect_and_events(tmp_path, capsys):
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    assert main(["--root", str(tmp_path), "inspect", h.run_id]) == 0
    capsys.readouterr()
    assert main(["--root", str(tmp_path), "events", h.run_id]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert json.loads(lines[0])["event"] == "run_created"


def test_step_1_init_creates_project(tmp_path, capsys):
    # build step 1 finally landed: init creates the §10 project workspace
    # (serve remains the only honest exit-4 command — see test_ops).
    assert main(["--root", str(tmp_path), "init", "proj"]) == 0
    assert (tmp_path / "proj" / "project.yaml").is_file()
    assert (tmp_path / "proj" / "configs" / "train.example.json").is_file()
    assert "NOT_IMPLEMENTED" not in capsys.readouterr().err


def test_missing_run_exits_2(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "inspect", "run_NOPE"]) == 2


# -- store gc CLI (12 §6.1: a command, never automatic) -------------------


def test_store_gc_dry_run_is_default(tmp_path, capsys):
    from mlforge.store import ContentStore

    h = ContentStore(tmp_path / "store").put_bytes(b"unreferenced")
    # fresh blob → inside grace; age it past the default 7-day window
    import os
    import time

    old = time.time() - 30 * 24 * 3600
    os.utime(ContentStore(tmp_path / "store").path_for(h), (old, old))
    assert main(["--root", str(tmp_path), "store", "gc", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert out["swept"] == [h]
    assert (tmp_path / "store").exists()  # nothing actually deleted


def test_store_gc_execute_deletes(tmp_path, capsys):
    from mlforge.store import ContentStore

    store = ContentStore(tmp_path / "store")
    h = store.put_bytes(b"unreferenced old")
    import os
    import time

    old = time.time() - 30 * 24 * 3600
    os.utime(store.path_for(h), (old, old))
    assert main(["--root", str(tmp_path), "store", "gc", "--execute"]) == 0
    assert not store.contains(h)
    assert "EXECUTED" in capsys.readouterr().out


def test_store_gc_blocked_by_lease_exits_3(tmp_path, capsys):
    from mlforge.store import ContentStore

    store = ContentStore(tmp_path / "store")
    store.put_bytes(b"blob")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "run_A").mkdir()
    (tmp_path / "runs" / "run_A" / ".lease").write_text("sess")
    assert main(["--root", str(tmp_path), "store", "gc", "--execute"]) == 3
    err = capsys.readouterr().err
    assert "run_A" in err


def test_store_without_gc_subcommand_exits_4(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "store"]) == 4
    assert "NOT_IMPLEMENTED" in capsys.readouterr().err


# -- validate / preflight CLI (12 §7, step 3) -----------------------------


def test_validate_fails_closed_on_fresh_run(tmp_path, capsys):
    """A fresh run has source+environment captures (12 §6.4) but no
    configured dataset identity ⇒ unverifiable ⇒ BLOCK. Fail-closed
    demo: exit 1, report printed, run left FAILED[FORK_ONLY]."""
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    assert main(["--root", str(tmp_path), "validate", h.run_id]) == 1
    out = capsys.readouterr().out
    assert "VALIDATION REPORT" in out
    assert "[PASS] Run manifest integrity" in out
    assert "[PASS] Source artifact + code hash" in out
    assert "[FAIL] Dataset identity (cryptographic)" in out
    assert "TRAINING BLOCKED" in out          # flow=TRAIN for a CREATED run
    # train path: gate BLOCK → FAILED[FORK_ONLY] (13 §7)
    assert wf.get_run_state(h.run_id) == "FAILED"
    assert wf.get_run_status(h.run_id)["failure"]["recovery"] == "FORK_ONLY"


def test_validate_json_report(tmp_path, capsys):
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    assert main(["--root", str(tmp_path), "validate", h.run_id, "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["blocked"] is True
    assert out["failed_step"] == 4  # dataset identity unverifiable
    assert out["run_id"] == h.run_id


def test_validate_missing_run_exits_2(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "validate", "run_MISSING"]) == 2


def test_preflight_report_only_and_fails_closed(tmp_path, capsys):
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    code = main(["--root", str(tmp_path), "preflight", h.run_id, "--no-gpu"])
    assert code == 1  # identity providers unverifiable in this build step
    out = capsys.readouterr().out
    assert "PREFLIGHT REPORT" in out
    # report-only: no transition, no events (runtime wires READY→RUNNING)
    assert wf.get_run_state(h.run_id) == "CREATED"
    assert [e["event"] for e in wf.get_run_events(h.run_id)] == ["run_created"]


def test_preflight_cli_defaults_are_host_plan_and_runtime(tmp_path, capsys):
    """CPU-laptop usability: no flags ⇒ GPU follows host/plan (never a
    blind FAIL), disk/RAM honor state/runtime.json knobs."""
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    state = tmp_path / "runs" / h.run_id / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "runtime.json").write_text(json.dumps({
        "checkpoint_bytes": 1 << 20, "log_bytes": 1 << 20,
        "safety_margin_bytes": 1 << 20, "min_ram_bytes": 1,
    }), encoding="utf-8")
    code = main(["--root", str(tmp_path), "preflight", h.run_id, "--json"])
    report = json.loads(capsys.readouterr().out)
    checks = {c["id"]: c for c in report["checks"]}
    assert checks["disk"]["verdict"] == "PASS"    # tiny runtime budget
    assert checks["ram"]["verdict"] == "PASS"     # runtime.min_ram_bytes=1
    assert checks["gpu"]["verdict"] != "FAIL"     # host/plan-derived, no flags
    # still fail-closed where it must be: unregistered dataset ⇒ blocked
    assert code == 1 and report["blocked"] is True


# -- lease CLI (12 §23, 13 §4.1) ------------------------------------------


def test_lease_status_free(tmp_path, capsys):
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    assert main(["--root", str(tmp_path), "lease", "status", h.run_id]) == 0
    assert "no lease (FREE)" in capsys.readouterr().out


def test_lease_status_held_json(tmp_path, capsys):
    from mlforge.leases import RunLeaseManager
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    RunLeaseManager(tmp_path).acquire(h.run_id)
    assert main(["--root", str(tmp_path), "lease", "status", h.run_id, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "HELD" and out["pid"]


def test_lease_break_requires_force_and_yes(tmp_path, capsys):
    from mlforge.leases import RunLeaseManager
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    RunLeaseManager(tmp_path).acquire(h.run_id)
    assert main(["--root", str(tmp_path), "lease", "break", h.run_id]) == 3
    assert main(["--root", str(tmp_path), "lease", "break", h.run_id, "--force"]) == 3
    assert "requires --force and --yes" in capsys.readouterr().err
    assert wf.lease_status(h.run_id)["state"] == "HELD"  # no silent escalation


def test_lease_break_executed_and_logged(tmp_path, capsys):
    from mlforge.leases import RunLeaseManager
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    RunLeaseManager(tmp_path).acquire(h.run_id)
    assert main([
        "--root", str(tmp_path), "lease", "break", h.run_id,
        "--force", "--yes", "--reason", "worker confirmed dead",
    ]) == 0
    assert "LEASE BROKEN" in capsys.readouterr().out
    events = wf.get_run_events(h.run_id)
    assert events[-1]["event"] == "LEASE_BROKEN"
    assert wf.lease_status(h.run_id)["state"] == "FREE"


def test_lease_break_without_lease_exits_3(tmp_path, capsys):
    from mlforge.workflow import WorkflowAPI

    wf = WorkflowAPI(tmp_path)
    h = wf.create_run(_spec())
    code = main([
        "--root", str(tmp_path), "lease", "break", h.run_id, "--force", "--yes",
    ])
    assert code == 3


def test_lease_without_subcommand_exits_4(tmp_path, capsys):
    # argparse rejects unknown subchoices itself (usage error, exit 2);
    # a bare `mlforge lease` reaches our dispatch → NOT_IMPLEMENTED (4).
    assert main(["--root", str(tmp_path), "lease"]) == 4
    assert "NOT_IMPLEMENTED" in capsys.readouterr().err


def test_dataset_types_lists_catalog(capsys):
    """The 'what options are there' command — discoverable, no project needed."""
    assert main(["dataset", "types"]) == 0
    out = capsys.readouterr().out
    for name in ("text_corpus", "coco_detection", "mot_challenge",
                 "reid_crops", "tabular"):
        assert name in out
    assert "Not supported yet" in out
    assert "ingestion.yaml" in out
    # the YAML example nests correctly under the model name
    assert "      transform: text_corpus" in out


def test_dataset_types_json(capsys):
    assert main(["dataset", "types", "--json"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    names = {t["name"] for t in catalog["supported"]}
    assert {"text_corpus", "coco_detection", "mot_challenge",
            "reid_crops", "tabular"} <= names
    assert any(p["name"] == "audio" for p in catalog["planned"])
