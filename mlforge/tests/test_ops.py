"""Build step 11 tests — evaluate / compare / infer / export / package /
`model import` (13 §4.1, §5.4, §6.7–§6.10, §7; 12 §15).

Specs as executable checks:
  * §6.7 five-component evaluation identity (model+dataset+code+
    environment+protocol); SHOW PROTOCOL preview equals the recorded
    artifact; write-once immutability
  * §6.10 comparison is descriptive — same protocol ⇒ comparable, a
    different protocol ⇒ NOT_COMPARABLE (shown, never averaged), never a
    winner
  * §6.9 export: operator/format validation BLOCKs with a readable list;
    never a partial export; contract is retargeted, never invented twice
  * 12 §15.3 package: contract required, secrets scan, immutable bundle,
    no state change
  * §6.8 infer: contract check SAFE|BLOCK (never weaken the schema),
    missing input ⇒ exit 2, deterministic output identity
  * §5.4 first-consumption marker: AVAILABLE fans out to exactly one
    state; later consumptions record artifacts without flipping state
  * §4.4 idempotency: evaluate/export/package `--command-id` returns the
    ORIGINAL result, never recomputes
  * model import: declared identity, hashed weights, duplicate `name:vN`
    blocks, missing package ⇒ exit 2
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlforge.errors import NotFound, PreconditionFailed, ValidationBlock
from mlforge.hashing import file_hash
from mlforge.ingest.config import set_path as ingest_set_path
from mlforge.ingest.identity import recompute_identity
from mlforge.run_spec import RunSpec
from mlforge.workflow import WorkflowAPI

# -- helpers ---------------------------------------------------------------


def _spec() -> RunSpec:
    return RunSpec(
        model="rf_detr_s",
        train_datasets=("coco_2017:v1",),
        semantic={
            "optimizer": "adamw", "learning_rate": 1e-4, "scheduler": "cosine",
            "loss": "l1", "seed": 42, "global_batch": 32, "epochs": 50,
            "precision_policy": "bf16",
        },
    )


def _publish(wf: WorkflowAPI) -> dict:
    """Drive a run to COMPLETED → its model gets published AVAILABLE.

    The run trains through a REAL worker session (scaffold trainer under
    `MLFORGE_HARNESS`): completed runs therefore carry committed
    checkpoint weights, which package/export/infer load (12 §15.3 — a
    model whose weights never existed is never packaged).
    """
    from mlforge.leases import RunLeaseManager
    from mlforge.runtime.worker import Worker

    h = wf.create_run(_spec())
    wf.begin_validation(h.run_id)
    wf.validation_pass(h.run_id)
    root = Path(wf.root)
    info = RunLeaseManager(root).acquire(h.run_id)
    rc = Worker(
        root,
        h.run_id,
        info.session_token,
        heartbeat_interval=60.0,
        poll_interval=0.0,
    ).run()
    assert rc == 0, f"worker exited {rc} — training never wedges mid-publish"
    entry = [m for m in wf.list_models() if m.get("run_id") == h.run_id]
    assert entry, "completion must publish a model"
    return entry[0]


def _ref(entry: dict) -> str:
    return f"{entry['name']}:{entry['version']}"


def _register_dataset(wf: WorkflowAPI, name: str = "custom_v2") -> str:
    """Register + configure a machine path so evaluation can re-hash it."""
    data = Path(wf.root) / "ds_data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "a.txt").write_text("hello", encoding="utf-8")
    identity, _ = recompute_identity(str(data), name, "v1")
    wf.register_dataset(name, identity, version="v1", file_count=1)
    ingest_set_path(Path(wf.root), name, str(data))
    return f"{name}:v1"


def _model_events(wf: WorkflowAPI, model_id: str) -> list[dict]:
    path = Path(wf.root) / "models" / model_id / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _external_package(root: Path, *, name: str = "extnet",
                      version: str = "v1", weights: bytes = b"w-v1",
                      with_contract: bool = True) -> str:
    pkg = root / f"pkg_{name}_{version}"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "model.safetensors").write_bytes(weights)
    spec: dict = {"schema_version": 1, "name": name, "version": version}
    if with_contract:
        spec["inference_contract"] = {
            "input_schema": [{"name": "frame", "type": "image"}],
            "output_schema": [{"name": "detections", "type": "structured"}],
            "operators": ["conv", "gemm", "relu"],
            "runtime": {"framework": "onnx", "opset": 17},
        }
    (pkg / "model_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return str(pkg)


def _cli(monkeypatch, capsys, root, *argv, expect: int = 0, answer=None):
    from mlforge.cli.main import main as cli_main

    if answer is not None:
        # Emulate input() faithfully: the prompt lands on stdout.
        monkeypatch.setattr(
            "builtins.input",
            lambda prompt="": (print(prompt, end="", flush=True), answer)[1],
        )
    rc = cli_main(["--root", str(root), *(str(a) for a in argv)])
    out, err = capsys.readouterr()
    assert rc == expect, (
        f"exit {rc}, want {expect}\nstdout:\n{out}\nstderr:\n{err}"
    )
    return out, err


# -- evaluate (13 §6.7, 12 §15.4) ------------------------------------------


def test_evaluate_writes_five_component_identity(wf):
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    rec = wf.evaluate_model(ref, ds)
    for key in ("model_hash", "dataset_hash", "code_hash",
                "environment_hash", "evaluation_protocol_hash"):
        assert str(rec[key]).startswith("sha256:"), key
    identity = rec["identity"]
    assert identity.startswith("sha256:") and len(identity) == 71
    # artifact identity is derived from all five components
    from mlforge.hashing import content_hash
    assert identity == content_hash({
        "model_hash": rec["model_hash"],
        "dataset_hash": rec["dataset_hash"],
        "code_hash": rec["code_hash"],
        "environment_hash": rec["environment_hash"],
        "evaluation_protocol_hash": rec["evaluation_protocol_hash"],
    })
    assert rec["harness"] == "scaffold"
    assert Path(wf.root, "evaluations", rec["eval_id"],
                "evaluation.json").is_file()


def test_evaluate_metrics_deterministic_across_repeats(wf):
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    first = wf.evaluate_model(ref, ds)
    second = wf.evaluate_model(ref, ds)
    assert first["metrics"] == second["metrics"]
    assert first["evaluation_protocol_hash"] == second["evaluation_protocol_hash"]
    # different artifacts, identical values (12 §15.4: repeat = same
    # protocol = same identity inputs, never an "update")
    assert first["eval_id"] != second["eval_id"]


def test_preview_equals_recorded_artifact(wf):
    """SHOW PROTOCOL shows exactly what gets written (same build)."""
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    preview = wf.preview_evaluation(ref, ds, protocol_overrides={"seed": 7})
    rec = wf.evaluate_model(ref, ds, protocol_overrides={"seed": 7})
    assert preview["identity"] == rec["identity"]
    assert preview["evaluation_protocol"] == rec["evaluation_protocol"]
    assert preview["metrics"] == rec["metrics"]
    assert preview["eval_id"] == ""  # nothing written yet


def test_evaluate_first_consumption_then_state_stable(wf):
    entry = _publish(wf)
    ref = _ref(entry)
    ds = _register_dataset(wf)
    assert wf.resolve_model(ref)["state"] == "AVAILABLE"
    first = wf.evaluate_model(ref, ds)
    assert wf.resolve_model(ref)["state"] == "EVALUATED"
    # second evaluation (different protocol) — artifact yes, state no flip
    wf.evaluate_model(ref, ds, protocol_overrides={"seed": 999})
    assert wf.resolve_model(ref)["state"] == "EVALUATED"
    events = _model_events(wf, entry["model_id"])
    eval_events = [e for e in events if e["event"] == "evaluation_recorded"]
    assert len(eval_events) == 2
    assert eval_events[0]["to"] == "EVALUATED"       # first fires the marker
    assert "to" not in eval_events[1]                # later ones never flip


def test_evaluate_unknown_dataset_exit_2(wf):
    ref = _ref(_publish(wf))
    with pytest.raises(NotFound):
        wf.evaluate_model(ref, "ghost_ds:v1")


def test_evaluate_unknown_model_exit_2(wf):
    _register_dataset(wf)
    with pytest.raises(NotFound):
        wf.evaluate_model("ghost:v1", "custom_v2:v1")


def test_protocol_unknown_key_blocks(wf):
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    with pytest.raises(ValidationBlock) as exc:
        wf.evaluate_model(ref, ds, protocol_overrides={"bogus": 1})
    assert "unknown evaluation protocol keys" in str(exc.value)


def test_derived_protocol_fields_not_overridable(wf):
    """Faking the harness identity would fake comparability (12 §15.4)."""
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    for field in ("harness_code_hash", "metric_definitions"):
        with pytest.raises(ValidationBlock) as exc:
            wf.evaluate_model(ref, ds, protocol_overrides={field: "sha256:x"})
        assert "derived protocol fields cannot be overridden" in str(exc.value)


def test_custom_seed_changes_protocol_hash_and_metrics(wf):
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    base = wf.evaluate_model(ref, ds)
    seeded = wf.evaluate_model(ref, ds, protocol_overrides={"seed": 999})
    assert base["evaluation_protocol_hash"] != seeded["evaluation_protocol_hash"]
    assert base["metrics"] != seeded["metrics"]


def test_evaluation_artifact_immutable(wf):
    from mlforge.ops.evaluation import write_evaluation

    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    rec = wf.evaluate_model(ref, ds)
    forged = dict(rec, metrics={"mAP": 1.0, "AP50": 1.0})
    with pytest.raises(ValidationBlock) as exc:
        write_evaluation(Path(wf.root), forged)
    assert "immutable" in str(exc.value)


def test_cli_evaluate_cancel_writes_nothing(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    out, _ = _cli(monkeypatch, capsys, wf.root, "evaluate", ref,
                  "--dataset", ds, answer="n")
    assert "Cancelled" in out
    assert not list(Path(wf.root).glob("evaluations/eval_*"))
    assert wf.resolve_model(ref)["state"] == "AVAILABLE"


def test_cli_evaluate_confirm_prints_uri(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    out, _ = _cli(monkeypatch, capsys, wf.root, "evaluate", ref,
                  "--dataset", ds, answer="y")
    assert "Produce evaluation artifact?" in out
    assert "evaluation://eval_" in out
    assert "evaluation identity = model + dataset + code + environment" in out
    assert wf.resolve_model(ref)["state"] == "EVALUATED"


def test_cli_evaluate_command_id_dedupes(monkeypatch, capsys, tmp_path):
    """§4.4: never recompute — the retry returns the original eval_id."""
    wf = WorkflowAPI(tmp_path / "workspace")
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    out1, _ = _cli(monkeypatch, capsys, wf.root, "evaluate", ref,
                   "--dataset", ds, "--yes", "--command-id", "cmd_e1",
                   "--json")
    out2, _ = _cli(monkeypatch, capsys, wf.root, "evaluate", ref,
                   "--dataset", ds, "--yes", "--command-id", "cmd_e1",
                   "--json")
    assert json.loads(out1)["eval_id"] == json.loads(out2)["eval_id"]
    assert len(list(Path(wf.root).glob("evaluations/eval_*"))) == 1


def test_cli_evaluate_json_and_error_codes(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    out, _ = _cli(monkeypatch, capsys, wf.root, "evaluate", ref,
                  "--dataset", ds, "--yes", "--json")
    data = json.loads(out)
    assert data["status"] == "evaluated"
    assert set(data["metrics"]) == {"mAP", "AP50"}
    assert data["evaluation_protocol_hash"].startswith("sha256:")
    # unknown dataset / model → exit 2 (failure matrix)
    _cli(monkeypatch, capsys, wf.root, "evaluate", ref,
         "--dataset", "ghost:v1", "--yes", expect=2)
    _cli(monkeypatch, capsys, wf.root, "evaluate", "ghost:v1",
         "--dataset", ds, "--yes", expect=2)
    # unknown protocol key → exit 1
    bad = tmp_path / "bad_protocol.json"
    bad.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    _cli(monkeypatch, capsys, wf.root, "evaluate", ref, "--dataset", ds,
         "--protocol", str(bad), "--yes", expect=1)


# -- compare (13 §6.10) ----------------------------------------------------


def test_compare_requires_two_refs(wf):
    ref = _ref(_publish(wf))
    with pytest.raises(ValidationBlock):
        wf.compare_models([ref])


def test_compare_same_protocol_is_comparable(wf):
    a = _ref(_publish(wf))
    b = _ref(_publish(wf))
    ds = _register_dataset(wf)
    wf.evaluate_model(a, ds)
    wf.evaluate_model(b, ds)  # default protocol on both
    data = wf.compare_models([a, b])
    assert len(data["columns"]) == 2
    assert all(c["comparable"] for c in data["columns"])
    assert len(data["comparable_protocols"]) == 1


def test_compare_different_protocol_not_comparable(wf):
    a = _ref(_publish(wf))
    b = _ref(_publish(wf))
    ds = _register_dataset(wf)
    wf.evaluate_model(a, ds)                              # default protocol
    wf.evaluate_model(b, ds, protocol_overrides={"seed": 7})  # different
    data = wf.compare_models([a, b])
    flags = {c["ref"]: c["comparable"] for c in data["columns"]}
    assert flags[a] is False and flags[b] is False
    assert data["not_comparable_protocols"]  # reported, never merged


def test_compare_without_evaluation_shows_gap(wf):
    a = _ref(_publish(wf))
    b = _ref(_publish(wf))
    data = wf.compare_models([a, b])
    assert all(c["evaluation"] is None for c in data["columns"])
    assert all(c["comparable"] is None for c in data["columns"])


def test_compare_is_read_only(wf):
    a = _ref(_publish(wf))
    b = _ref(_publish(wf))
    ds = _register_dataset(wf)
    wf.evaluate_model(a, ds)
    before = sorted(p.name for p in Path(wf.root).iterdir())
    wf.compare_models([a, b])
    assert sorted(p.name for p in Path(wf.root).iterdir()) == before
    assert wf.resolve_model(a)["state"] == "EVALUATED"
    assert wf.resolve_model(b)["state"] == "AVAILABLE"


def test_cli_compare_display_comparable(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    a = _ref(_publish(wf))
    b = _ref(_publish(wf))
    ds = _register_dataset(wf)
    wf.evaluate_model(a, ds)
    wf.evaluate_model(b, ds)
    out, _ = _cli(monkeypatch, capsys, wf.root, "compare", a, b)
    assert "MODEL COMPARISON" in out
    assert "Base model" in out and "Protocol" in out
    assert "Comparable" in out
    assert out.count("yes") >= 2
    assert "does not choose a winner" in out
    assert "NOT_COMPARABLE" not in out


def test_cli_compare_display_not_comparable_legend(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    a = _ref(_publish(wf))
    b = _ref(_publish(wf))
    ds = _register_dataset(wf)
    wf.evaluate_model(a, ds)
    wf.evaluate_model(b, ds, protocol_overrides={"seed": 7})
    out, _ = _cli(monkeypatch, capsys, wf.root, "compare", a, b)
    assert "NOT_COMPARABLE" in out
    assert "never averaged" in out
    # descriptive only — never a winner
    assert "does not choose a winner" in out
    _cli(monkeypatch, capsys, wf.root, "compare", "ghost:v1", a, expect=2)


# -- export (13 §6.9) ------------------------------------------------------


def test_export_onnx_writes_artifact_and_contract(wf):
    ref = _ref(_publish(wf))
    rec = wf.export_model(ref, "onnx")
    base = Path(wf.root, "exports", rec["export_id"])
    assert (base / "export.json").is_file()
    assert (base / "model_spec.json").is_file()
    spec = json.loads((base / "model_spec.json").read_text())
    assert spec["inference_contract"]["runtime"]["framework"] == "onnx"
    assert rec["identity"].startswith("sha256:")
    # export consumes a fresh model exactly once
    assert wf.resolve_model(ref)["state"] == "EXPORTED"


def test_export_validations_all_pass(wf):
    ref = _ref(_publish(wf))
    rec = wf.export_model(ref, "onnx")
    assert set(rec["validations"]) == {"architecture", "operators",
                                       "dynamic_shapes"}
    assert all(v == "PASS" for v in rec["validations"].values())
    num = rec["numerical_validation"]
    assert num["result"] == "PASS" and num["max_error"] < num["tolerance"]
    assert num["harness"] == "scaffold"


def test_export_tflite_blocks_unsupported_operator(wf):
    """The scaffold graph uses non_max_suppression; TFLite cannot run it
    in this harness → BLOCK with the operator list + a capable format."""
    ref = _ref(_publish(wf))
    with pytest.raises(ValidationBlock) as exc:
        wf.export_model(ref, "tflite")
    msg = exc.value.render()
    assert "non_max_suppression" in msg
    assert "onnx" in msg  # capable-format hint
    assert not list(Path(wf.root).glob("exports/exp_*"))  # never partial


def test_export_unknown_format_blocks_with_registry(wf):
    ref = _ref(_publish(wf))
    with pytest.raises(ValidationBlock) as exc:
        wf.export_model(ref, "flux")
    msg = exc.value.render()
    assert "unsupported export format" in msg
    assert "onnx" in msg and "tflite" in msg  # closed registry in the hint
    assert not list(Path(wf.root).glob("exports/exp_*"))


def test_export_after_evaluate_does_not_flip_state(wf):
    ref = _ref(_publish(wf))
    ds = _register_dataset(wf)
    wf.evaluate_model(ref, ds)
    assert wf.resolve_model(ref)["state"] == "EVALUATED"
    wf.export_model(ref, "onnx")
    assert wf.resolve_model(ref)["state"] == "EVALUATED"  # marker stays


def test_export_reuses_contract_and_retargets_runtime(wf):
    ref = _ref(_publish(wf))
    onnx = wf.export_model(ref, "onnx")
    vino = wf.export_model(ref, "openvino")
    spec_one = json.loads(
        (Path(wf.root, "exports", onnx["export_id"], "model_spec.json"))
        .read_text())
    spec_two = json.loads(
        (Path(wf.root, "exports", vino["export_id"], "model_spec.json"))
        .read_text())
    ops_one = spec_one["inference_contract"]["operators"]
    ops_two = spec_two["inference_contract"]["operators"]
    assert ops_one == ops_two  # operator set is the model's, preserved
    assert spec_two["inference_contract"]["runtime"]["framework"] == "openvino"


def test_cli_export_display(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    ref = _ref(_publish(wf))
    out, _ = _cli(monkeypatch, capsys, wf.root, "export", ref,
                  "--format", "onnx")
    assert "Source:" in out and "Target: ONNX" in out
    assert "Validating:" in out and "Operators PASS" in out
    assert "Numerical validation:" in out and "PASS" in out
    assert "Export artifact: exp_" in out
    # failure paths: tflite operator BLOCK → exit 1, unknown → exit 1
    _cli(monkeypatch, capsys, wf.root, "export", ref, "--format", "tflite",
         expect=1)
    _cli(monkeypatch, capsys, wf.root, "export", ref, "--format", "flux",
         expect=1)


def test_cli_export_command_id_dedupes(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    ref = _ref(_publish(wf))
    out1, _ = _cli(monkeypatch, capsys, wf.root, "export", ref,
                   "--format", "onnx", "--command-id", "cmd_x1", "--json")
    out2, _ = _cli(monkeypatch, capsys, wf.root, "export", ref,
                   "--format", "onnx", "--command-id", "cmd_x1", "--json")
    assert json.loads(out1)["export_id"] == json.loads(out2)["export_id"]
    assert len(list(Path(wf.root).glob("exports/exp_*"))) == 1


# -- package (12 §15.3, 13 §4.1) -------------------------------------------


def test_package_without_contract_blocks(wf):
    ref = _ref(_publish(wf))  # no export yet → no contract
    with pytest.raises(ValidationBlock) as exc:
        wf.package_model(ref)
    assert "export" in exc.value.render()  # points at the fix (§6.9)
    assert not list(Path(wf.root).glob("bundles/bdl_*"))


def test_package_happy_path_bundle_files(wf):
    ref = _ref(_publish(wf))
    wf.export_model(ref, "onnx")
    rec = wf.package_model(ref)
    base = Path(wf.root, "bundles", rec["bundle_id"])
    for name in ("model.safetensors", "bundle.json", "model_spec.json",
                 "provenance.json", "integrity.json"):
        assert (base / name).is_file(), name
    assert (base / "model.safetensors").stat().st_size > 0
    assert set(rec["integrity"]) >= {"model_spec", "provenance"}
    spec = json.loads((base / "model_spec.json").read_text())
    assert "inference_contract" in spec
    # the bundle must be IMPORTABLE — read_package hashes the weights
    # file the bundle actually carries (13 §4.1 `model import`)
    from mlforge.ops.importing import read_package

    pkg_spec, weights = read_package(base)
    assert pkg_spec["inference_contract"] == spec["inference_contract"]
    assert weights["bytes"] == (base / "model.safetensors").stat().st_size


def test_package_blocks_on_planted_secret(wf):
    ref = _ref(_publish(wf))
    entry = wf.resolve_model(ref)
    wf.export_model(ref, "onnx")
    leak = Path(wf.root) / "models" / entry["model_id"] / "leak.txt"
    leak.write_text("sk-abcdefghijklmnop1234567890", encoding="utf-8")
    with pytest.raises(ValidationBlock) as exc:
        wf.package_model(ref)
    assert "secrets detected" in str(exc.value)
    assert not list(Path(wf.root).glob("bundles/bdl_*"))


def test_package_changes_no_state(wf):
    ref = _ref(_publish(wf))
    entry = wf.resolve_model(ref)
    wf.export_model(ref, "onnx")
    state_before = wf.resolve_model(ref)["state"]
    rec = wf.package_model(ref)
    assert wf.resolve_model(ref)["state"] == state_before
    events = [e for e in _model_events(wf, entry["model_id"])
              if e["event"] == "bundle_created"]
    assert len(events) == 1 and "to" not in events[0]
    assert rec["bundle_id"].startswith("bdl_")


def test_cli_package_display(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    ref = _ref(_publish(wf))
    _cli(monkeypatch, capsys, wf.root, "export", ref, "--format", "onnx")
    out, _ = _cli(monkeypatch, capsys, wf.root, "package", ref)
    assert "Secrets:   PASS" in out
    assert "Bundle:    bdl_" in out
    assert "contract sha256:" in out
    # no contract → exit 1
    fresh = _ref(_publish(wf))
    _cli(monkeypatch, capsys, wf.root, "package", fresh, expect=1)


# -- infer (13 §6.8, 12 §15.3) ---------------------------------------------


def test_infer_without_contract_blocks(wf):
    ref = _ref(_publish(wf))
    img = Path(wf.root) / "frame.png"
    img.write_bytes(b"\x89PNG data")
    with pytest.raises(ValidationBlock):
        wf.infer_model(ref, str(img))


def test_infer_schema_violation_blocks_with_entry(wf):
    ref = _ref(_publish(wf))
    wf.export_model(ref, "onnx")  # contract: image input
    txt = Path(wf.root) / "notes.txt"
    txt.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValidationBlock) as exc:
        wf.infer_model(ref, str(txt))
    msg = str(exc.value)
    assert "frame" in msg and ".png" in msg  # names the offending entry


def test_infer_missing_input_exit_2(wf):
    ref = _ref(_publish(wf))
    wf.export_model(ref, "onnx")
    with pytest.raises(NotFound):
        wf.infer_model(ref, str(Path(wf.root) / "missing.png"))


def test_infer_happy_deterministic_identity(wf):
    ref = _ref(_publish(wf))
    wf.export_model(ref, "onnx")
    img = Path(wf.root) / "frame.png"
    img.write_bytes(b"\x89PNG frame-1")
    first = wf.infer_model(ref, str(img))
    second = wf.infer_model(ref, str(img))
    # same contract + same input bytes ⇒ same output identity; each
    # invocation still gets its own artifact id
    assert first["identity"] == second["identity"]
    assert first["output_id"] != second["output_id"]
    assert first["harness"] == "scaffold"
    assert "no model was run" in first["result"]["note"]
    assert Path(wf.root, "outputs", f"{first['output_id']}.json").is_file()


def test_cli_infer_display_and_codes(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    ref = _ref(_publish(wf))
    wf.export_model(ref, "onnx")
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG data")
    out, _ = _cli(monkeypatch, capsys, wf.root, "infer", ref, str(img))
    assert "contract check: SAFE" in out
    assert "outputs/out_" in out
    txt = tmp_path / "notes.txt"
    txt.write_text("x", encoding="utf-8")
    _cli(monkeypatch, capsys, wf.root, "infer", ref, str(txt), expect=1)
    _cli(monkeypatch, capsys, wf.root, "infer", ref,
         str(tmp_path / "missing.png"), expect=2)


# -- model import (13 §4.1, 12 §15.3) --------------------------------------


def test_import_happy_available_with_contract(wf, tmp_path):
    pkg = _external_package(tmp_path, weights=b"weights-v1")
    entry = wf.import_external_model(pkg)
    assert entry["name"] == "extnet" and entry["version"] == "v1"
    assert entry["state"] == "AVAILABLE"
    assert entry["origin"] == "import"
    assert entry["artifact_hash"] == file_hash(Path(pkg) / "model.safetensors")
    from mlforge.ops.exporting import contract_source_dir
    assert contract_source_dir(Path(wf.root), entry) is not None
    # imported model is consumable like any other
    wf.export_model("extnet:v1", "tflite")  # its op list is tflite-safe


def test_import_missing_contract_blocks(wf, tmp_path):
    pkg = _external_package(tmp_path, with_contract=False)
    with pytest.raises(ValidationBlock) as exc:
        wf.import_external_model(pkg)
    assert "inference_contract" in str(exc.value)


def test_import_missing_weights_blocks(wf, tmp_path):
    pkg = _external_package(tmp_path)
    (Path(pkg) / "model.safetensors").unlink()
    with pytest.raises(ValidationBlock) as exc:
        wf.import_external_model(pkg)
    assert "weights" in str(exc.value)


def test_import_duplicate_version_blocks(wf, tmp_path):
    pkg = _external_package(tmp_path, weights=b"w1")
    wf.import_external_model(pkg)
    with pytest.raises(ValidationBlock) as exc:
        wf.import_external_model(pkg)
    assert "already registered" in str(exc.value)
    # a NEW version may still be imported (immutable versions, not names)
    pkg2 = _external_package(tmp_path, version="v2", weights=b"w2")
    entry = wf.import_external_model(pkg2)
    assert entry["version"] == "v2"


def test_import_missing_directory_exit_2(wf, tmp_path):
    with pytest.raises(NotFound):
        wf.import_external_model(str(tmp_path / "nope"))


def test_cli_model_import_display(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    pkg = _external_package(tmp_path, weights=b"w-cli")
    out, _ = _cli(monkeypatch, capsys, wf.root, "model", "import", pkg,
                  "--json")
    data = json.loads(out)
    assert data["status"] == "imported"
    assert data["model_ref"] == "extnet:v1"
    assert data["state"] == "AVAILABLE"
    # duplicate → exit 1, missing dir → exit 2
    _cli(monkeypatch, capsys, wf.root, "model", "import", pkg, expect=1)
    _cli(monkeypatch, capsys, wf.root, "model", "import",
         str(tmp_path / "nope"), expect=2)


def test_cli_model_import_command_id_dedupes(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    pkg = _external_package(tmp_path, name="deduped", weights=b"w-d")
    out1, _ = _cli(monkeypatch, capsys, wf.root, "model", "import", pkg,
                   "--command-id", "cmd_i1", "--json")
    out2, _ = _cli(monkeypatch, capsys, wf.root, "model", "import", pkg,
                   "--command-id", "cmd_i1", "--json")
    assert json.loads(out1)["model_id"] == json.loads(out2)["model_id"]
    assert json.loads(out2)["status"] == "imported"


# -- validate <MODEL>, serve, gates ----------------------------------------


def test_cli_validate_model_report(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    ref = _ref(_publish(wf))
    out, _ = _cli(monkeypatch, capsys, wf.root, "validate", ref)
    assert "MODEL VALIDATION" in out
    assert "[PASS]" in out and "MODEL OK" in out
    _cli(monkeypatch, capsys, wf.root, "validate", "nothing_here", expect=2)


def test_serve_is_honest_not_implemented(monkeypatch, capsys, tmp_path):
    out, err = _cli(monkeypatch, capsys, tmp_path, "serve", expect=4)
    assert "[NOT_IMPLEMENTED]" in err
    assert "no build step yet" in err
    assert "infer" in err  # points at today's real path


def test_step_11_gates_flipped():
    from mlforge.cli.main import _IMPLEMENTED, _PENDING
    for cmd in ("evaluate", "compare", "infer", "export", "package"):
        assert cmd in _IMPLEMENTED, cmd
        assert cmd not in _PENDING, cmd
    # build step 1 delivered init/configure — nothing is pending anymore
    # (`serve` has no build step and keeps its special case instead).
    for cmd in ("init", "configure", "status", "train", "watch"):
        assert cmd in _IMPLEMENTED, cmd
    assert not _PENDING


def test_model_inspect_shows_contract_presence(monkeypatch, capsys, tmp_path):
    wf = WorkflowAPI(tmp_path / "workspace")
    ref = _ref(_publish(wf))
    out, _ = _cli(monkeypatch, capsys, wf.root, "model", "inspect", ref)
    assert "none yet" in out  # before any export
    wf.export_model(ref, "onnx")
    out, _ = _cli(monkeypatch, capsys, wf.root, "model", "inspect", ref)
    assert "present (12 §15.3)" in out
