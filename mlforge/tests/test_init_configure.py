"""Build step 1 (delivered last) — `init` + `configure datasets`.

Specs as executable checks:
  * 13 §4.1 `mlforge init <name>` — Create project: scaffolds the §10
    product view (project.yaml, datasets.yaml, ingestion.yaml template,
    configs/ with a schema-valid example, models/, runs/, artifacts/)
    and journals the project genesis (13 §5.1 CREATED); never touches a
    foreign directory; re-run = natural-key no-op exit 0; `--command-id`
    replays the ORIGINAL result (13 §4.4)
  * 12 §10.1 `mlforge configure datasets` — one-time-per-machine paths:
    every candidate path is HASHED against its registration BEFORE any
    write; mismatch → explicit error with fix options; non-TTY without
    --set fails closed; journal-only audit (no state transition, 12 §6.3)
  * `project.yaml` pins the project name → the machine-local file
    `datasets_<project>.yaml` follows it (12 §6.3)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlforge.cli.main import main
from mlforge.ingest import recompute_identity
from mlforge.ingest.config import load_paths, paths_file, write_registry
from mlforge.journal import EventJournal
from mlforge.run_spec import RunSpec
from mlforge.workflow import WorkflowAPI

# -- helpers ---------------------------------------------------------------


def _raw(tmp_path: Path, name: str = "raw", files: dict | None = None) -> Path:
    p = tmp_path / name
    p.mkdir(parents=True, exist_ok=True)
    for fn, content in (files or {"a.jpg": "one", "b.jpg": "two"}).items():
        (p / fn).write_text(content, encoding="utf-8")
    return p


def _register(wf: WorkflowAPI, ds_id: str, path: Path) -> str:
    """Register WITHOUT a machine-local path — `configure` is the SUT."""
    identity, manifest = recompute_identity(path, ds_id, "v1")
    wf.register_dataset(
        ds_id, identity, version="v1",
        file_count=manifest.file_count, total_bytes=manifest.total_bytes,
    )
    write_registry(wf.root, ds_id, {"version": "v1", "identity": identity})
    return identity


def _ds_events(root: Path, ds_id: str) -> list:
    return EventJournal(root / "datasets" / ds_id / "events.jsonl").read()


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """Machine-local config lands in tmp — never the real ~/.mlforge."""
    h = tmp_path / "home"
    monkeypatch.setenv("MLFORGE_HOME", str(h))
    return h


# -- init: layout + identity -------------------------------------------------


def test_init_creates_product_view_layout(tmp_path, capsys, home):
    root = tmp_path / "root"
    assert main(["--root", str(root), "init", "myproj"]) == 0
    proj = root / "myproj"
    assert proj.is_dir()
    for d in ("configs", "models", "runs", "artifacts"):
        assert (proj / d).is_dir(), d
    from mlforge.yamlmini import load_file
    data = load_file(proj / "project.yaml")
    assert data["schema_version"] == 1
    assert data["name"] == "myproj"
    assert isinstance(data["created_ts"], float)
    assert data["mlforge"]
    assert load_file(proj / "datasets.yaml") == {}
    # ingestion plan scaffolded + parseable (12 §10.2) — matches the
    # starter train config so prepare works without hand-written YAML
    assert (proj / "ingestion.yaml").is_file()
    from mlforge.ingest.dag import load_ingestion

    plans = load_ingestion(proj)
    assert "rf_detr_s" in plans
    assert plans["rf_detr_s"].transform == "coco_detection"
    out = capsys.readouterr().out
    assert "Initialized project 'myproj'" in out
    assert "mlforge train --config configs/train.example.json" in out
    assert "ingestion.yaml" in out


def test_init_example_config_passes_live_schema(tmp_path, home):
    root = tmp_path / "root"
    assert main(["--root", str(root), "init", "p", "--json"]) == 0
    cfg = json.loads(
        (root / "p" / "configs" / "train.example.json").read_text()
    )
    spec = RunSpec.from_dict(cfg)  # raises on any schema drift
    assert spec.model and spec.train_datasets and spec.semantic


def test_init_journals_project_genesis(tmp_path, home):
    root = tmp_path / "root"
    assert main(["--root", str(root), "init", "p"]) == 0
    status = json.loads((root / "p" / "projects" / "p" / "status.json")
                        .read_text())
    assert status["state"] == "CREATED"          # 13 §5.1 first step
    events = EventJournal(root / "p" / "projects" / "p" / "events.jsonl").read()
    assert [e.event for e in events] == ["project_created"]


def test_init_json_shape(tmp_path, capsys, home):
    root = tmp_path / "root"
    assert main(["--root", str(root), "init", "p", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["name"] == "p"
    assert result["status"] == "created"
    assert result["example_config"] == "configs/train.example.json"
    assert result["ingestion"] == "ingestion.yaml"
    assert Path(result["path"]).is_dir()


# -- init: fail-closed guards ------------------------------------------------


@pytest.mark.parametrize("bad", ["../evil", ".hidden", "a b", "x" * 65, ""])
def test_init_rejects_invalid_names(tmp_path, bad, home):
    root = tmp_path / "root"
    root.mkdir()
    assert main(["--root", str(root), "init", bad]) == 1
    # rejected BEFORE any target computation — the root stays empty
    assert not any(root.iterdir())


def test_init_never_touches_foreign_directory(tmp_path, capsys, home):
    root = tmp_path / "root"
    foreign = root / "other"
    foreign.mkdir(parents=True)
    (foreign / "keep.txt").write_text("important")
    assert main(["--root", str(root), "init", "other"]) == 1
    assert (foreign / "keep.txt").read_text() == "important"
    assert not (foreign / "project.yaml").exists()
    assert "[VALIDATION_BLOCK]" in capsys.readouterr().err


def test_init_reserves_empty_directory(tmp_path, home):
    root = tmp_path / "root"
    (root / "proj").mkdir(parents=True)  # operator pre-created it
    assert main(["--root", str(root), "init", "proj"]) == 0
    assert (root / "proj" / "project.yaml").is_file()


def test_init_corrupt_project_yaml_fails_closed(tmp_path, home):
    root = tmp_path / "root"
    proj = root / "proj"
    proj.mkdir(parents=True)
    (proj / "project.yaml").write_text("{{{::: not yaml")
    assert main(["--root", str(root), "init", "proj"]) == 3
    assert not (proj / "configs").exists()  # untouched


def test_init_name_mismatch_is_a_different_project(tmp_path, home):
    root = tmp_path / "root"
    assert main(["--root", str(root), "init", "alpha"]) == 0
    assert main(["--root", str(root), "init", "beta", "--path",
                 str(root / "alpha")]) == 1


# -- init: idempotency --------------------------------------------------------


def test_init_rerun_is_natural_key_noop(tmp_path, capsys, home):
    root = tmp_path / "root"
    assert main(["--root", str(root), "init", "p"]) == 0
    capsys.readouterr()  # drain the first (created) output
    from mlforge.yamlmini import load_file
    before = load_file(root / "p" / "project.yaml")["created_ts"]
    events_before = len(EventJournal(root / "p" / "projects" / "p"
                                     / "events.jsonl").read())
    assert main(["--root", str(root), "init", "p"]) == 0
    out = capsys.readouterr().out
    assert "already initialized" in out and "no changes" in out
    after = load_file(root / "p" / "project.yaml")["created_ts"]
    assert after == before
    assert len(EventJournal(root / "p" / "projects" / "p"
                            / "events.jsonl").read()) == events_before


def test_init_command_id_replays_original_result(tmp_path, capsys, home):
    root = tmp_path / "root"
    assert main(["--root", str(root), "init", "p",
                 "--command-id", "01CID", "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "created"
    assert main(["--root", str(root), "init", "p",
                 "--command-id", "01CID", "--json"]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay == first  # original result, not a re-computation
    events = EventJournal(root / "p" / "projects" / "p" / "events.jsonl").read()
    assert "COMMAND_DEDUPED" in [e.event for e in events]  # 13 §4.4


def test_init_path_flag_and_machine_slug(tmp_path, home):
    """project.yaml pins the name → datasets_<project>.yaml follows it."""
    from mlforge.ingest.config import project_slug
    root = tmp_path / "root"
    target = tmp_path / "elsewhere"
    assert main(["--root", str(root), "init", "pinalias",
                 "--path", str(target)]) == 0
    assert (target / "project.yaml").is_file()
    assert not (root / "pinalias").exists()
    assert project_slug(target) == "pinalias"
    # slug of a workspace without project.yaml keeps the directory name
    plain = tmp_path / "plainws"
    plain.mkdir()
    assert project_slug(plain) == "plainws"
    # and the machine-local file name follows the pinned name
    assert paths_file(target).name == "datasets_pinalias.yaml"


# -- configure datasets --------------------------------------------------------


def test_configure_without_datasets_exits_2(tmp_path, capsys, home):
    root = tmp_path / "root"
    root.mkdir()
    assert main(["--root", str(root), "configure", "datasets"]) == 2
    err = capsys.readouterr().err
    assert "no registered datasets" in err
    assert "dataset add" in err


def test_configure_set_verifies_and_writes(tmp_path, capsys, home):
    wf = WorkflowAPI(tmp_path / "workspace")
    raw = _raw(tmp_path)
    identity = _register(wf, "coco_2017", raw)
    assert main(["--root", str(wf.root), "configure", "datasets",
                 "--set", f"coco_2017={raw}"]) == 0
    out = capsys.readouterr().out
    assert "✅ matches" in out and identity in out
    assert "Saved:" in out and str(paths_file(wf.root)) in out
    assert load_paths(wf.root) == {"coco_2017": str(raw)}
    # machine-local file under $MLFORGE_HOME (13 §10)
    assert paths_file(wf.root).parent == home
    # journal-only audit — state never transitions (12 §6.3)
    events = [e.event for e in _ds_events(wf.root, "coco_2017")]
    assert "dataset_path_configured" in events
    assert wf.get_dataset_status("coco_2017")["state"] == "REGISTERED"


def test_configure_mismatch_blocks_before_any_write(tmp_path, capsys, home):
    wf = WorkflowAPI(tmp_path / "workspace")
    reg_dir = _raw(tmp_path, "registered", {"a.jpg": "one"})
    other_dir = _raw(tmp_path, "other", {"a.jpg": "DIFFERENT"})
    _register(wf, "coco_2017", reg_dir)
    assert main(["--root", str(wf.root), "configure", "datasets",
                 "--set", f"coco_2017={other_dir}"]) == 1
    err = capsys.readouterr().err
    assert "identity mismatch" in err
    assert "--force" in err  # explicit fix options (12 §10.1)
    assert load_paths(wf.root) == {}  # nothing written
    assert not paths_file(wf.root).exists()
    assert "dataset_path_configured" not in [e.event for e in
                                             _ds_events(wf.root, "coco_2017")]


def test_configure_unknown_set_id_exits_2(tmp_path, capsys, home):
    wf = WorkflowAPI(tmp_path / "workspace")
    raw = _raw(tmp_path)
    _register(wf, "coco_2017", raw)
    assert main(["--root", str(wf.root), "configure", "datasets",
                 "--set", f"nope={raw}"]) == 2
    assert "not registered" in capsys.readouterr().err


def test_configure_bad_set_format_exits_1(tmp_path, capsys, home):
    wf = WorkflowAPI(tmp_path / "workspace")
    raw = _raw(tmp_path)
    _register(wf, "coco_2017", raw)
    assert main(["--root", str(wf.root), "configure", "datasets",
                 "--set", "just-an-id"]) == 1
    assert "--set expects ID=PATH" in capsys.readouterr().err


def test_configure_non_tty_unconfigured_exits_3(tmp_path, capsys, home,
                                                monkeypatch):
    wf = WorkflowAPI(tmp_path / "workspace")
    raw = _raw(tmp_path)
    _register(wf, "coco_2017", raw)
    monkeypatch.setattr("mlforge.cli.main._stdin_is_tty", lambda: False)
    assert main(["--root", str(wf.root), "configure", "datasets"]) == 3
    err = capsys.readouterr().err
    assert "not a TTY" in err and "--set" in err
    assert load_paths(wf.root) == {}


def test_configure_interactive_prompt(tmp_path, capsys, home, monkeypatch):
    wf = WorkflowAPI(tmp_path / "workspace")
    raw = _raw(tmp_path)
    _register(wf, "coco_2017", raw)
    monkeypatch.setattr("mlforge.cli.main._stdin_is_tty", lambda: True)
    answers = iter([str(raw)])
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": (print(prompt, end="", flush=True),
                           next(answers))[1],
    )
    assert main(["--root", str(wf.root), "configure", "datasets"]) == 0
    out = capsys.readouterr().out
    assert "coco_2017 path?" in out and "✅ matches" in out
    assert load_paths(wf.root) == {"coco_2017": str(raw)}


def test_configure_interactive_empty_answer_blocks(tmp_path, capsys, home,
                                                   monkeypatch):
    wf = WorkflowAPI(tmp_path / "workspace")
    raw = _raw(tmp_path)
    _register(wf, "coco_2017", raw)
    monkeypatch.setattr("mlforge.cli.main._stdin_is_tty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert main(["--root", str(wf.root), "configure", "datasets"]) == 1
    assert "path is required" in capsys.readouterr().err
    assert load_paths(wf.root) == {}


def test_configure_reaudit_is_noop_no_duplicate_journal(tmp_path, capsys,
                                                        home):
    wf = WorkflowAPI(tmp_path / "workspace")
    raw = _raw(tmp_path)
    identity = _register(wf, "coco_2017", raw)
    assert main(["--root", str(wf.root), "configure", "datasets",
                 "--set", f"coco_2017={raw}"]) == 0
    capsys.readouterr()
    events_before = len(_ds_events(wf.root, "coco_2017"))
    # re-audit without --set, non-TTY: uses the configured path, re-hashes
    assert main(["--root", str(wf.root), "configure", "datasets",
                 "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["datasets"][0]["status"] == "verified"
    assert result["datasets"][0]["identity"] == identity
    assert result["file"] == str(paths_file(wf.root))
    assert len(_ds_events(wf.root, "coco_2017")) == events_before  # no spam


def test_configure_missing_path_exits_3(tmp_path, capsys, home):
    wf = WorkflowAPI(tmp_path / "workspace")
    raw = _raw(tmp_path)
    _register(wf, "coco_2017", raw)
    assert main(["--root", str(wf.root), "configure", "datasets",
                 "--set", "coco_2017=/nonexistent/dataset/dir"]) == 3
    err = capsys.readouterr().err
    assert "cannot read dataset" in err and "check the path" in err
    assert load_paths(wf.root) == {}


def test_configure_json_shape(tmp_path, capsys, home):
    wf = WorkflowAPI(tmp_path / "workspace")
    raw = _raw(tmp_path)
    _register(wf, "coco_2017", raw)
    assert main(["--root", str(wf.root), "configure", "datasets",
                 "--set", f"coco_2017={raw}", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    entry = payload["datasets"][0]
    assert set(entry) == {"dataset", "path", "identity", "status",
                          "file_count", "total_bytes"}
    assert entry["status"] == "path_updated"
    assert entry["identity"].startswith("sha256:")
