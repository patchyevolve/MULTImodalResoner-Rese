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
