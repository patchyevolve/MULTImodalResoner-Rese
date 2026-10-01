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


def test_pending_command_exits_4(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "train", "--config", "x.yaml"]) == 4
    assert "NOT_IMPLEMENTED" in capsys.readouterr().err


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
    from mlforge.store import ArtifactRegistry, ContentStore

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
