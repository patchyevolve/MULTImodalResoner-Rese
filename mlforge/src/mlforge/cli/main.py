"""`mlforge` command-line interface.

Normative contract: 13_product_specification.md §4.
Thin layer over the Workflow API (§1: three interfaces, one API). Any
command whose machinery is not built yet exits 4 with `NOT_IMPLEMENTED`
rather than pretending to work — fail loudly, never fake success.

Exit codes (§4.2): 0 ok · 1 validation BLOCK · 2 not found ·
3 precondition failed · 4 runtime error.

Process ownership (12 §12.4): `train`/`resume` never spawn the worker
themselves — they queue a spawn request for the supervisor daemon.
`pause`/`stop` write a control intent the running worker obeys (the CLI
is not the parent, and not the trainer).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import mlforge
import mlforge.status as status_layer
from mlforge.errors import (
    MlforgeError,
    NoValidContinuation,
    NotFound,
    PreconditionFailed,
    ValidationBlock,
)
from mlforge.ingest import (
    load_paths as ingest_load_paths,
    prepare as ingest_prepare,
    recompute_identity as ingest_recompute_identity,
)
from mlforge.ingest.config import (
    paths_file as ingest_paths_file,
    set_path as ingest_set_path,
    write_registry as ingest_write_registry,
)
from mlforge.ingest.transforms import (
    dataset_types as ingest_dataset_types,
    suggest_transforms,
)
from mlforge.leases import LeaseState, RunLeaseManager
from mlforge.ops import DEFAULT_METRIC_NAMES, contract_source_dir
from mlforge.planner import (
    PLAN_FILENAME,
    ExecutionPlan,
    build_plan,
    detect_capabilities,
    load_runtime,
)
from mlforge.run_spec import RunSpec
from mlforge.runtime.control import wait_for_state, write_control
from mlforge.runtime.trainer import require_trainable
from mlforge.store import ArtifactRegistry, ContentStore
from mlforge.supervisor import enqueue_spawn, ensure_supervisor
from mlforge.validation import Preflight, ValidationGate
from mlforge.workflow import WorkflowAPI
from mlforge.yamlmini import YamlError, dump as yaml_dump, load_file as yaml_load_file

# 13 §11 build-order gates: implemented vs pending.
# Every §4.1 command is implemented now; `serve` (not in the build
# order) keeps its honest exit-4 special case below.
_IMPLEMENTED = {
    "status", "inspect", "events", "store", "validate", "preflight", "lease",
    "train", "resume", "pause", "stop", "watch", "hardware", "dataset",
    "prepare", "fork", "retrain", "finetune", "model",
    "evaluate", "compare", "infer", "export", "package",
    "gui", "init", "configure",
}
_PENDING: dict[str, int] = {}

#: States after which a worker is gone and `--attach` may stop waiting.
_TERMINAL = {"COMPLETED", "FAILED", "STOPPED", "INTERRUPTED", "PAUSED"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlforge",
        description="MLForge — hardware-agnostic ML training system "
        "(spec: 10_training_plan/12 + 13)",
    )
    p.add_argument("--version", action="version", version=f"mlforge {mlforge.__version__}")
    p.add_argument(
        "--root",
        default=".",
        help="workspace root (default: current directory)",
    )
    sub = p.add_subparsers(dest="command")

    # PROJECT (13 §4.1)
    ini = sub.add_parser("init", help="create a project workspace (13 §4.1)")
    ini.add_argument("name",
                     help="project name: letters, digits, '.', '_', '-'")
    ini.add_argument("--path", default=None,
                     help="place the project here instead of <root>/<name>")
    ini.add_argument("--command-id", default=None,
                     help="idempotency key for safe retries (13 §4.4)")
    ini.add_argument("--json", action="store_true", help="machine-readable output")

    cfg = sub.add_parser("configure",
                         help="machine-local configuration (13 §10.1)")
    cfg_sub = cfg.add_subparsers(dest="configure_command", required=True)
    cds = cfg_sub.add_parser(
        "datasets",
        help="re-point/verify machine-local dataset paths (12 §10.1)",
        description=(
            "Machine-local dataset paths (12 §10.1): identity lives in "
            "the project (portable), paths live per machine. You normally "
            "don't need this right after `mlforge dataset add` — add "
            "already records the path on the machine where it ran. Use "
            "configure when (a) you cloned/synced this project to a new "
            "machine, (b) the data folder moved, or (c) several datasets "
            "need paths at once. Every candidate path is hashed against "
            "its registration before being written — wrong path ⇒ "
            "explicit mismatch, never silent acceptance."
        ),
    )
    cds.add_argument("--set", action="append", default=[], metavar="ID=PATH",
                     help="non-interactive path assignment (repeatable)")
    cds.add_argument("--json", action="store_true", help="machine-readable output")

    st = sub.add_parser(
        "status",
        help="overview of all runs, or one run's block (read-only, L1/L2)",
    )
    st.add_argument("run_id", nargs="?", default=None,
                    help="show one run's L1 block (default: all-runs overview)")
    st.add_argument("-v", "--verbose", action="store_true",
                    help="L2 training detail (config, loops, checkpoints)")
    st.add_argument("--json", action="store_true", help="machine-readable output")

    insp = sub.add_parser("inspect", help="detailed report for one object")
    insp.add_argument("object_id")
    insp.add_argument("--json", action="store_true")

    ev = sub.add_parser("events", help="structured event stream for a run")
    ev.add_argument("run_id")
    ev.add_argument("--follow", action="store_true",
                    help="tail new events until Ctrl+C (read-only)")

    wch = sub.add_parser("watch", help="live dashboard for a run (viewer only)")
    wch.add_argument("run_id", nargs="?", default=None,
                     help="run to watch (default: the live dashboard)")
    wch.add_argument("--interval", type=float, default=1.0,
                     help="refresh interval in seconds (default: 1)")
    wch.add_argument("--json", action="store_true",
                     help="one-shot machine-readable frame (no loop)")

    hw = sub.add_parser("hardware", help="L3 host telemetry (diagnostic only)")
    hw.add_argument("--json", action="store_true", help="machine-readable output")

    val = sub.add_parser(
        "validate", help="run the 19-step validation gate for a run (fail-closed)"
    )
    val.add_argument("object_id", help="run id or model ref (13 §4.1: "
                                       "validate <RUN|MODEL>)")
    val.add_argument("--json", action="store_true")

    pf = sub.add_parser(
        "preflight",
        help="host + identity preflight before any expensive operation (report-only)",
    )
    pf.add_argument("run_id")
    pf.add_argument("--json", action="store_true")
    gpu_group = pf.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu",
        action="store_true",
        help="force the GPU/driver checks (default: follow the run's plan "
        "or live host detection)",
    )
    gpu_group.add_argument(
        "--no-gpu",
        action="store_true",
        help="this operation does not require a GPU (GPU/driver checks become WARN)",
    )

    lease_parent = sub.add_parser("lease", help="run lease operations (12 §23)")
    lease_sub = lease_parent.add_subparsers(dest="lease_command")
    lease_status = lease_sub.add_parser("status", help="who holds the run lease")
    lease_status.add_argument("run_id")
    lease_status.add_argument("--json", action="store_true")
    lease_break = lease_sub.add_parser(
        "break", help="break a lease (requires --force AND --yes; logged)"
    )
    lease_break.add_argument("run_id")
    lease_break.add_argument("--force", action="store_true")
    lease_break.add_argument("--yes", action="store_true")
    lease_break.add_argument("--reason", default="unspecified")
    lease_break.add_argument("--json", action="store_true")

    tr = sub.add_parser("train", help="start a new run (async — 13 §4.2)")
    tr.add_argument("--config", required=True,
                    help="JSON run config: model, train_datasets, semantic, "
                         "(optional) runtime knobs")
    tr.add_argument("--command-id", default=None,
                    help="dedupe key: a retry returns the original run id (§4.4)")
    tr.add_argument("--yes", action="store_true", help="skip the start confirmation")
    tr.add_argument("--attach", action="store_true",
                    help="tail events until the run reaches a terminal state "
                         "(Ctrl+C detaches the viewer; training continues)")
    tr.add_argument("--json", action="store_true")

    rs = sub.add_parser("resume", help="continue a PAUSED/INTERRUPTED/FAILED(RESUME) run")
    rs.add_argument("run_id")
    rs.add_argument("--command-id", default=None)
    rs.add_argument("--attach", action="store_true")
    rs.add_argument("--json", action="store_true")

    pz = sub.add_parser("pause", help="graceful pause → PAUSED (resumable)")
    pz.add_argument("run_id")
    pz.add_argument("--timeout", type=float, default=15.0,
                    help="seconds to wait for the worker to commit the pause")
    pz.add_argument("--json", action="store_true")

    sp = sub.add_parser("stop", help="graceful stop → STOPPED (no resume)")
    sp.add_argument("run_id")
    sp.add_argument("--timeout", type=float, default=15.0)
    sp.add_argument("--json", action="store_true")

    # fork / retrain / finetune (13 §6.3/§6.4, build step 10): all three
    # create a NEW run — semantic identity is frozen (12 §4 hard rule).
    def _add_flow_flags(sp_: argparse.ArgumentParser) -> None:
        sp_.add_argument("--set", dest="overrides", action="append",
                         default=[], metavar="KEY=VALUE",
                         help="semantic override (repeatable; JSON values, "
                              "e.g. --set lr=2e-4 --set 'epochs=10')")
        sp_.add_argument("--dataset", default=None,
                         help="replace the training dataset (fork/retrain/"
                              "finetune → new experiment identity)")
        sp_.add_argument("--config", default=None,
                         help='partial config: {"semantic": {...}, '
                              '"runtime": {...}} (base; --set wins)')
        sp_.add_argument("--command-id", default=None,
                         help="dedupe key: a retry returns the original run id (§4.4)")
        sp_.add_argument("--yes", action="store_true",
                         help="skip the SHOW DELTA confirmation")
        sp_.add_argument("--attach", action="store_true",
                         help="tail events until the run reaches a terminal state")
        sp_.add_argument("--json", action="store_true")

    fk = sub.add_parser("fork", help="semantic change → NEW run, parent "
                        "recorded (12 §4: never reconfigure in place)")
    fk.add_argument("run_id", help="parent run to fork from")
    _add_flow_flags(fk)

    rt = sub.add_parser("retrain", help="repeat the latest run of MODEL "
                        "with fresh init (13 §6.3)")
    rt.add_argument("model", help="model name (e.g. rf_detr_s)")
    _add_flow_flags(rt)

    ft = sub.add_parser("finetune", help="init from model://name:vN "
                        "weights → NEW run + NEW model (13 §6.4)")
    ft.add_argument("model", help="base model ref: name | name:vN | "
                    "model://name:vN")
    ft.add_argument("--strategy", default=None,
                    choices=["full", "freeze_backbone", "freeze_encoder",
                             "lora", "adapter", "custom"],
                    help="fine-tuning strategy (default: config's "
                         "finetune_strategy, else full)")
    _add_flow_flags(ft)

    md = sub.add_parser("model", help="model registry (13 §4.1)")
    md_sub = md.add_subparsers(dest="model_command", required=True)
    md_list = md_sub.add_parser("list", help="all registered models")
    md_list.add_argument("--json", action="store_true")
    md_ins = md_sub.add_parser("inspect", help="model detail + lineage")
    md_ins.add_argument("ref", help="name | name:vN | model://name:vN")
    md_ins.add_argument("--json", action="store_true")
    md_imp = md_sub.add_parser("import", help="import an external model "
                                "package (13 §4.1)")
    md_imp.add_argument("path", help="package directory: model_spec.json "
                                     "+ model.safetensors (12 §15.3)")
    md_imp.add_argument("--command-id", default=None,
                        help="dedupe key: a retry returns the original result (§4.4)")
    md_imp.add_argument("--json", action="store_true")

    # EVALUATION / INFERENCE / ARTIFACT (13 §4.1, §6.7–§6.10; build step 11)
    ev = sub.add_parser("evaluate", help="produce an evaluation artifact "
                        "(13 §6.7)")
    ev.add_argument("model", help="model ref: name | name:vN | model://name:vN")
    ev.add_argument("--dataset", required=True,
                    help="dataset ref: name | name:vN | dataset://name:vN")
    ev.add_argument("--protocol", default=None, metavar="FILE",
                    help="JSON protocol overrides (split/seed/thresholds/... "
                         "— derived fields are not overridable)")
    ev.add_argument("--command-id", default=None,
                    help="dedupe key: never recompute a finished evaluation (§4.4)")
    ev.add_argument("--yes", action="store_true",
                    help="skip the SHOW PROTOCOL confirmation")
    ev.add_argument("--json", action="store_true")

    cp = sub.add_parser("compare", help="side-by-side evaluation comparison "
                        "(13 §6.10 — descriptive, never a winner)")
    cp.add_argument("models", nargs="+", metavar="MODEL")
    cp.add_argument("--json", action="store_true")

    inf = sub.add_parser("infer", help="one-shot inference (13 §6.8)")
    inf.add_argument("model", help="model ref")
    inf.add_argument("input", help="input file (validated against the "
                                   "contract's input_schema)")
    inf.add_argument("--json", action="store_true")

    ex = sub.add_parser("export", help="export artifact + inference contract "
                        "(13 §6.9)")
    ex.add_argument("model", help="model ref")
    ex.add_argument("--format", required=True, dest="fmt",
                    help="onnx | openvino | tensorrt | coreml | tflite")
    ex.add_argument("--command-id", default=None,
                    help="dedupe key: never recompute a finished export (§4.4)")
    ex.add_argument("--json", action="store_true")

    pk = sub.add_parser("package", help="build the inference bundle "
                         "(12 §15.3)")
    pk.add_argument("model", help="model ref")
    pk.add_argument("--command-id", default=None,
                    help="dedupe key: never rebuild a finished bundle (§4.4)")
    pk.add_argument("--json", action="store_true")

    sv = sub.add_parser("serve",
                        help="(long-running model server — no build step yet)")
    sv.add_argument("extra", nargs=argparse.REMAINDER)

    # GUI (13 §9.2 control plane; build step 12): localhost web viewer
    # over the SAME Workflow API as the CLI and the TUI (§1 rule 2).
    gui = sub.add_parser("gui", help="localhost web dashboard (read-only "
                         "viewer over the same Workflow API)")
    gui.add_argument("--port", type=int, default=8765,
                     help="localhost port (default 8765)")
    gui.add_argument("--open", action="store_true", dest="open_browser",
                     help="open the browser after binding")

    st_gc_parent = sub.add_parser("store", help="artifact store operations")
    st_gc = st_gc_parent.add_subparsers(dest="store_command").add_parser(
        "gc",
        help="garbage-collect unreachable artifacts (dry-run by default)",
    )
    st_gc.add_argument(
        "--execute",
        action="store_true",
        help="actually delete (default: report only — deletion is an explicit decision)",
    )
    st_gc.add_argument(
        "--grace-days",
        type=float,
        default=7.0,
        help="keep unreachable artifacts younger than this many days (default: 7)",
    )
    st_gc.add_argument("--json", action="store_true")

    # 13 §4.1 DATA: dataset registration is explicit, content-addressed,
    # and always verifies a CONFIGURED path (never discovers one).
    ds = sub.add_parser("dataset", help="dataset registration commands (13 §4.1)")
    ds_sub = ds.add_subparsers(dest="dataset_command", required=True)

    ds_add = ds_sub.add_parser("add", help="register a dataset identity at PATH")
    ds_add.add_argument("dataset_id")
    ds_add.add_argument("path", help="machine-local directory to hash")
    ds_add.add_argument("--version", default="v1", help="dataset version (default v1)")
    ds_add.add_argument("--schema", default=None,
                        help="annotation format label (part of the identity)")
    ds_add.add_argument("--force", action="store_true",
                        help="re-register when content changed since last add")
    ds_add.add_argument("--yes", action="store_true", help="skip the confirmation")
    ds_add.add_argument("--command-id", default=None,
                        help="dedupe key: a retry returns the original result (§4.4)")
    ds_add.add_argument("--json", action="store_true")

    ds_list = ds_sub.add_parser("list", help="registered datasets + local paths")
    ds_list.add_argument("--json", action="store_true")

    ds_ver = ds_sub.add_parser("verify", help="re-hash PATH against registration")
    ds_ver.add_argument("dataset_id")
    ds_ver.add_argument("--command-id", default=None)
    ds_ver.add_argument("--json", action="store_true")

    ds_types = ds_sub.add_parser(
        "types",
        help="supported dataset types & transforms — what fits what",
    )
    ds_types.add_argument("--json", action="store_true",
                          help="machine-readable catalog")

    pr = sub.add_parser(
        "prepare",
        help="transform VERIFIED sources into <model>_prepared (13 §6.6)",
    )
    pr.add_argument("model", help="model name from ingestion.yaml")
    pr.add_argument("--command-id", default=None,
                    help="dedupe key: a retry returns the original result (§4.4)")
    pr.add_argument("--json", action="store_true")

    # Pending build-order commands (13 §11): registered so they answer
    # NOT_IMPLEMENTED (exit 4) instead of an argparse "invalid choice".
    for _cmd, _step in sorted(_PENDING.items()):
        _p = sub.add_parser(_cmd, help=f"(pending build step {_step})")
        _p.add_argument("extra", nargs=argparse.REMAINDER)
    return p


# ---------------------------------------------------------------------------
# train / resume / pause / stop (13 §4.1 TRAINING, §6.1–6.2)
# ---------------------------------------------------------------------------


def _load_config(path: str) -> tuple[RunSpec, dict]:
    """JSON run config (stdlib-only core; YAML arrives with ingestion)."""
    p = Path(path)
    if not p.is_file():
        raise PreconditionFailed(
            f"config not found: {path}",
            hint="expected JSON: {model, train_datasets, semantic{...}}",
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreconditionFailed(f"config {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PreconditionFailed(f"config {path} must be a JSON object")
    runtime = dict(data.get("runtime") or {})
    spec = RunSpec.from_dict(data)  # raises ValidationBlock on bad identity fields
    return spec, runtime


def _load_partial_config(path: str | None) -> tuple[dict, dict]:
    """Partial config for fork/retrain/finetune (13 §6.4 `--config`):
    `{"semantic": {...}, "runtime": {...}}`, both optional — semantic
    entries are BASE overrides, `--set` applies on top (explicit wins)."""
    if not path:
        return {}, {}
    p = Path(path)
    if not p.is_file():
        raise PreconditionFailed(f"config not found: {path}",
                                 hint='expected JSON: {"semantic": {...}, '
                                      '"runtime": {...}}')
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreconditionFailed(f"config {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PreconditionFailed(f"config {path} must be a JSON object")
    unknown = set(data) - {"schema_version", "semantic", "runtime"}
    if unknown:
        raise PreconditionFailed(
            f"config {path} has unknown keys: {', '.join(sorted(unknown))}",
            hint='expected {"semantic": {...}, "runtime": {...}}',
        )
    semantic = data.get("semantic") or {}
    runtime = data.get("runtime") or {}
    if not isinstance(semantic, dict) or not isinstance(runtime, dict):
        raise PreconditionFailed(
            f"config {path}: semantic/runtime must be JSON objects")
    return dict(semantic), dict(runtime)


def _print_plan_section(spec: RunSpec, runtime: dict) -> None:
    """13 §6.1 SHOW PLAN — the feasibility solver's answer. Infeasible
    (ValidationBlock) propagates to exit 1 BEFORE anything is created."""
    plan = build_plan(spec, runtime=runtime)
    print("\nExecution plan")
    print("  " + plan.summary_line())
    bits = [f"precision {plan.precision_effective}"]
    if plan.precision_fallback_used:
        bits.append(f"fallback from {plan.precision_preferred} — PORTABLE")
    elif plan.portable_required:
        bits.append("CPU-only — PORTABLE")
    else:
        bits.append("PORTABLE-capable")
    print("  " + " · ".join(bits))


def _print_plan(spec: RunSpec, runtime: dict) -> None:
    sem = dict(spec.semantic)
    print("MLForge Training Setup\n")
    print(f"Model            {spec.model}")
    print(f"Training data    {', '.join(spec.train_datasets)}")
    if spec.val_dataset:
        print(f"Validation       {spec.val_dataset}")
    print("\nConfiguration")
    print("  " + " · ".join(f"{k} {v}" for k, v in sem.items()))
    if runtime:
        print("\nExecution (runtime overrides)")
        print("  " + " · ".join(f"{k} {v}" for k, v in runtime.items()))
    _print_plan_section(spec, runtime)
    print()


def _print_changed_lines(delta: dict) -> None:
    """SHOW DELTA (13 §6.3): every changed field, `from → to`."""
    for key in sorted(delta):
        change = delta[key]
        print(f"  {key}: {change['from']} → {change['to']}")


def _confirm(prompt: str) -> bool:
    """No interactive menu — just the y/n every mutating flow owes the
    user (13 §4.2). EOF (piped stdin) counts as "no" — never a yes."""
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().lower() in ("", "y", "yes")


def _write_runtime_config(run_id: str, root: Path, runtime: dict) -> None:
    if not runtime:
        return
    state = root / "runs" / run_id / "state"
    state.mkdir(parents=True, exist_ok=True)
    p = state / "runtime.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def _launch(wf: WorkflowAPI, run_id: str) -> dict:
    """Lease → queue spawn → ensure the daemon is up. The CLI stops here;
    the worker (fired from READY) is a daemon child, never ours (12 §12.4).

    Fail-closed: refuse to queue a spawn that cannot honestly train (no
    real trainer for this run's model + no explicit harness opt-in ⇒
    exit 3) — never a doomed "started" run."""
    spec_path = wf.root / "runs" / run_id / "run_spec.json"
    model: str | None = None
    if spec_path.is_file():
        try:
            model = str(json.loads(spec_path.read_text(encoding="utf-8"))
                        .get("model") or "") or None
        except (json.JSONDecodeError, AttributeError):  # gate re-validates
            model = None
    require_trainable(load_runtime(wf.root / "runs" / run_id), model=model)
    lease = RunLeaseManager(wf.root)
    info = lease.status(run_id)
    if info.state == LeaseState.HELD:
        token = info.session_token  # gate step 15 already owns it
    elif info.state == LeaseState.FREE:
        token = lease.acquire(run_id).session_token
    else:
        raise PreconditionFailed(
            f"run {run_id}: lease is SUSPECT (held by {info.holder})",
            hint="stale ≠ free: confirm no worker is alive, then "
                 "`mlforge lease break RUN --force --yes`",
        )
    enqueue_spawn(wf.root, run_id, token)
    sup = ensure_supervisor(wf.root)
    return {"supervisor_pid": sup["pid"]}


# ---------------------------------------------------------------------------
# status / watch / hardware — read-only observation layer (13 §9)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# init / configure (13 §4.1, §10.1 — build step 1, delivered last)
# ---------------------------------------------------------------------------

#: 13 §4.1 `init <name>` — POSIX-safe project token, max 64 chars.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: 13 §10 product-view layout under the project root.
_PROJECT_DIRS = ("configs", "models", "runs", "artifacts")

#: 13 §10 `configs/` starter — every field RunSpec requires (12 §17);
#: tests parse it against the live schema so it cannot silently rot.
_EXAMPLE_TRAIN_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "model": "rf_detr_s",
    "train_datasets": ["coco_2017:v1"],
    "semantic": {
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "scheduler": "cosine",
        "loss": "l1",
        "seed": 42,
        "global_batch": 32,
        "epochs": 50,
        "precision_policy": "bf16",
    },
}


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def _init_classify(target: Path, name: str) -> str:
    """`new` (create here) | `ours` (already initialized) — anything
    else RAISES before a single byte is written (never touch foreign)."""
    if not target.exists():
        return "new"
    proj = target / "project.yaml"
    if proj.is_file():
        data = _load_project_yaml(proj)
        if data.get("name") == name:
            return "ours"
        raise ValidationBlock(
            f"target exists with a different project: {target}",
            hint="it was initialized with another name — pick a different "
                 "name/path, or remove it first",
        )
    try:
        if not any(target.iterdir()):
            return "new"  # an empty directory is a legitimate reservation
    except OSError:
        raise ValidationBlock(
            f"target exists but is unreadable: {target}",
            hint="check permissions, then retry",
        ) from None
    raise ValidationBlock(
        f"target exists and is not an MLForge project: {target}",
        hint="remove or rename it, choose another name, or place the "
             "project elsewhere with --path",
    )


def _load_project_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml_load_file(path)
    except YamlError as exc:
        raise PreconditionFailed(
            f"project.yaml unreadable ({path}): {exc}",
            hint="fix the YAML (supported subset: mappings, lists, flow "
                 "collections) — the project will not be modified",
        ) from exc
    if not isinstance(data, dict):
        raise PreconditionFailed(f"project.yaml must be a mapping: {path}")
    return data


#: Scaffolding template for `ingestion.yaml` (12 §10.2) — matches the
#: starter train config (model `rf_detr_s` on `coco_2017:train`) so the
#: init → dataset → prepare flow works without hand-writing YAML first.
_INGESTION_TEMPLATE = """\
# MLForge ingestion plan (12 §10.2) — one entry per model.
#   transform:     name from the registry (unknown ⇒ prepare BLOCKs;
#                  add types with mlforge.ingest.register_transform)
#   train_sources: <dataset>:<split> — register with `mlforge dataset add`
#                  and verify BEFORE `mlforge prepare`
#                  derived data: {generated_from: <model>, dataset: <id>}
#                  (the model must be AVAILABLE; dataset holds its output)
#   depends_on:    upstream models that must be AVAILABLE first (registry)
models:
  rf_detr_s:
    transform: coco_detection
    train_sources: [coco_2017:train]
    # val_sources: [coco_2017:val]
    # depends_on: []
  # Text models (e.g. your books/papers corpus):
  # reasoner_s:
  #   transform: text_corpus          # pdf/docx/md/txt/zip → text chunks
  #   train_sources: [books:train]
  # Calibrator (12 §10.2) — trains on the detector's GENERATED
  # predictions (logits/labels, `{generated_from:}`), not raw images:
  # calibrator:
  #   transform: calibration          # prediction .jsonl/.json rows
  #   train_sources: [{generated_from: rf_detr_s}]
  #   depends_on: [rf_detr_s]         # detector must be AVAILABLE first
  # Hypothesis ranker (03 Model 4) — LambdaMART over the labeled
  # (event, hypothesis, relevance) triples as tabular rows:
  # hypothesis_ranker:
  #   transform: tabular              # label + group + feature columns
  #   train_sources: [ranker_features:train]
"""


def _init_scaffold(target: Path, name: str, *, status: str) -> dict[str, Any]:
    """Create the 13 §10 project view. Every write is existence-guarded,
    so a re-run (natural key or --command-id replay) is a no-op."""
    target.mkdir(parents=True, exist_ok=True)
    proj_path = target / "project.yaml"
    if not proj_path.exists():
        proj_path.write_text(yaml_dump({
            "schema_version": 1,
            "name": name,
            "created_ts": time.time(),
            "mlforge": mlforge.__version__,
        }), encoding="utf-8")
    for d in _PROJECT_DIRS:
        (target / d).mkdir(exist_ok=True)
    registry = target / "datasets.yaml"
    if not registry.exists():
        registry.write_text(yaml_dump({}), encoding="utf-8")
    example = target / "configs" / "train.example.json"
    if not example.exists():
        example.write_text(
            json.dumps(_EXAMPLE_TRAIN_CONFIG, indent=2) + "\n",
            encoding="utf-8",
        )
    ingestion = target / "ingestion.yaml"
    if not ingestion.exists():
        ingestion.write_text(_INGESTION_TEMPLATE, encoding="utf-8")
    if not (target / "projects" / name).is_dir():
        WorkflowAPI(target).create_project(name)  # journaled genesis (§5.1)
    return {
        "name": name,
        "path": str(target),
        "status": status,
        "created_ts": _load_project_yaml(proj_path).get("created_ts"),
        "example_config": "configs/train.example.json",
        "ingestion": "ingestion.yaml",
        "dirs": list(_PROJECT_DIRS),
    }


def _print_init_next_steps(name: str, target: Path) -> None:
    print(f"Initialized project {name!r} at {target}\n")
    print("  project.yaml                    project identity (13 §5.1)")
    print("  datasets.yaml                   dataset identity registry")
    print("  configs/train.example.json      starter training config — edit it")
    print("  ingestion.yaml                  model → transform plan (12 §10.2)")
    print("  models/  runs/  artifacts/      §10 layout")
    print("\nNext steps:")
    print(f"  cd {target}")
    print("  mlforge dataset add <ID> <PATH>     "
          "# register + verify identity; sets this machine's path")
    print("  mlforge dataset verify <ID>         "
          "# prove the bytes still match (prepare gate)")
    print("  mlforge dataset types               "
          "# supported formats & their transforms")
    print("  $EDITOR ingestion.yaml              "
          "# say which transform feeds which model")
    print("  mlforge prepare <MODEL>             # derived <MODEL>_prepared (§6.6)")
    print("  mlforge train --config configs/train.example.json")


def _do_init(wf: WorkflowAPI, args) -> int:
    name = args.name
    if not _NAME_RE.match(name):
        raise ValidationBlock(
            f"invalid project name {name!r}",
            hint="1-64 characters of letters, digits, '.', '_', '-'; "
                 "must start with a letter or digit",
        )
    target = Path(args.path).expanduser() if args.path else Path(args.root) / name
    kind = _init_classify(target, name)  # foreign ⇒ raises, nothing written

    def _scaffold() -> dict[str, Any]:
        return _init_scaffold(
            target, name,
            status="already_present" if kind == "ours" else "created",
        )

    if args.command_id:
        # The command journal lives INSIDE the target — WorkflowAPI's
        # constructor is pure, so building it before the directory exists
        # is safe (reads return [], appends mkdir as needed, 13 §4.4).
        result = WorkflowAPI(target).execute_idempotent(
            args.command_id, "init_project", _scaffold,
            kind="project", run_id=name,
        )
    else:
        result = _scaffold()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    elif result.get("status") == "already_present":
        print(f"Project {name!r} already initialized at {result['path']} "
              "— no changes.")
    else:
        _print_init_next_steps(name, Path(result["path"]))
    return 0


def _prompt_dataset_path(dataset_id: str) -> str:
    """12 §10.1: ask for the path — explicit input, never a scan."""
    try:
        answer = input(f"  {dataset_id} path? ").strip()
    except EOFError:
        answer = ""
    if not answer:
        raise ValidationBlock(
            f"{dataset_id}: a path is required",
            hint=f"pass --set {dataset_id}=<PATH> for non-interactive "
                 "configuration",
        )
    return answer


def _do_configure(wf: WorkflowAPI, args) -> int:
    if args.configure_command == "datasets":
        return _do_configure_datasets(wf, args)
    # argparse `required=True` on the subparsers keeps this unreachable.
    raise NotFound(f"unknown configure subcommand {args.configure_command!r}")


def _do_configure_datasets(wf: WorkflowAPI, args) -> int:
    """12 §10.1 one-time-per-machine path configuration.

    Every candidate path is HASHED against its registration BEFORE any
    write — wrong path ⇒ mismatch ⇒ explicit error with fix options;
    the system verifies identity, it never scans or guesses.
    """
    datasets = wf.list_datasets()
    if not datasets:
        raise NotFound(
            "no registered datasets to configure",
            hint="register one first: mlforge dataset add <ID> <PATH>",
        )
    sets: dict[str, str] = {}
    for entry in args.set or []:
        did, sep, path = str(entry).partition("=")
        if not sep or not did or not path:
            raise ValidationBlock(
                f"--set expects ID=PATH, got {entry!r}",
                hint="example: --set coco_2017=/data/coco",
            )
        sets[did] = path
    known = {d["id"] for d in datasets}
    for did in sets:
        if did not in known:
            raise NotFound(
                f"dataset {did!r} is not registered",
                hint=f"registered datasets: {', '.join(sorted(known))}",
            )
    configured = ingest_load_paths(wf.root)
    results: list[dict[str, Any]] = []
    if not args.json:
        print("Configure dataset paths (machine-local — never identity, 12 §6.3)")
    for d in sorted(datasets, key=lambda x: x["id"]):
        did = d["id"]
        if did in sets:
            path = sets[did]
        elif did in configured:
            path = configured[did]
        elif _stdin_is_tty():
            path = _prompt_dataset_path(did)
        else:
            raise PreconditionFailed(
                f"no machine-local path for {did!r} and stdin is not a TTY",
                hint=f"pass --set {did}=<PATH> (paths are explicit, never "
                     "discovered — 12 §6.3)",
            )
        reg = json.loads(
            (Path(wf.root) / "datasets" / did / "identity.json")
            .read_text(encoding="utf-8")
        )
        try:
            found, manifest = ingest_recompute_identity(
                path, did, str(reg.get("version") or "v1"),
                schema=reg.get("schema"),
            )
        except (NotFound, OSError) as exc:
            raise PreconditionFailed(
                f"{did}: cannot read dataset at {path}: {exc}",
                hint=f"check the path — {path} must be a directory of "
                     "dataset files",
            ) from exc
        registered = reg.get("identity")
        if registered and found != registered:
            raise ValidationBlock(
                f"{did}: identity mismatch at {path} "
                f"(registered {registered}, found {found})",
                hint=f"fix the path, or re-register changed content: "
                     f"mlforge dataset add {did} {path} --force "
                     "(13 §5.2 — never silently reinterpret identity)",
            )
        changed = configured.get(did) != str(Path(path))
        ingest_set_path(wf.root, did, path)
        if changed:
            wf.note_dataset_path(did, path)  # journal-only audit (12 §6.3)
        results.append({
            "dataset": did,
            "path": str(Path(path)),
            "identity": found,
            "status": "path_updated" if changed else "verified",
            "file_count": manifest.file_count,
            "total_bytes": manifest.total_bytes,
        })
        if not args.json:
            print(f"  {did:<20} {path}")
            print(f"    → verifying content hash... ✅ matches {found}")
    if args.json:
        print(json.dumps({
            "file": str(ingest_paths_file(wf.root)),
            "datasets": results,
        }, indent=2, sort_keys=True, default=str))
    else:
        print(f"\nSaved: {ingest_paths_file(wf.root)}")
    return 0


def _do_status(wf: WorkflowAPI, args) -> int:
    if args.run_id:
        detail = status_layer.collect_run(wf.root, args.run_id, verbose=args.verbose)
        if args.json:
            print(json.dumps(detail, indent=2, sort_keys=True, default=str))
        else:
            for line in status_layer.render_l1(detail):
                print(line)
            if args.verbose:
                for line in status_layer.render_l2(detail):
                    print(line)
        return 0
    runs = status_layer.collect_overview(wf.root)
    if args.json:
        print(json.dumps(runs, indent=2, sort_keys=True))
    else:
        for line in status_layer.render_overview(runs):
            print(line)
    return 0


def _watch_keys(wf: WorkflowAPI, run_id: str, key: str) -> str | None:
    """Map a viewer key to a control intent (13 §9.6). Returns 'quit' for q.
    The VIEWER never transitions anything itself — it only writes intents
    the worker obeys (same channel as `pause`/`stop`), or 'checkpoint'."""
    if key == "q":
        return "quit"
    if key in ("p", "s", "c"):
        write_control(wf.root, run_id, {"p": "pause", "s": "stop", "c": "checkpoint"}[key],
                      requested_by="cli:watch")
        return key
    return None


def _do_watch(wf: WorkflowAPI, args) -> int:
    if args.run_id:
        return _watch_run(wf, args, args.run_id)
    # `mlforge watch` without RUN = the Live dashboard (13 §4.1), a
    # multi-screen viewer over the same Workflow API (build step 12).
    if args.json:
        from mlforge.ui.viewmodel import dashboard_model

        model = dashboard_model(wf)
        if not model["runs"]:
            raise NotFound(
                "no runs to watch", hint="`mlforge train --config F` first"
            )
        print(json.dumps(model, indent=2, sort_keys=True, default=str))
        return 0
    from mlforge.ui.app import run_dashboard

    return run_dashboard(wf, interval=args.interval)


def _watch_run(wf: WorkflowAPI, args, run_id: str) -> int:
    """`watch RUN` — the §9.6 single-run frame (unchanged since step 7)."""
    detail = status_layer.collect_run(wf.root, run_id)  # NotFound ⇒ exit 2
    if args.json:
        print(json.dumps(detail, indent=2, sort_keys=True, default=str))
        return 0

    frame = "\n".join(status_layer.render_watch(detail))
    if not sys.stdin.isatty():
        # Piped/non-interactive: one frame, no loop (read-only either way).
        print(frame)
        return 0

    import os as _os
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    print(frame, flush=True)
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([fd], [], [], max(0.1, args.interval))
            if ready:
                key = _os.read(fd, 1).decode(errors="ignore")
                action = _watch_keys(wf, run_id, key)
                if action == "quit":
                    print("\nq exits the viewer only — training continues")
                    return 0
                if action is not None:
                    print(f"\n{action} requested for {run_id} (worker obeys; "
                          "viewer does not transition anything)")
            detail = status_layer.collect_run(wf.root, run_id)
            print("\x1b[2J\x1b[H" + "\n".join(status_layer.render_watch(detail)),
                  flush=True)
    except KeyboardInterrupt:
        print("\ndetached — training continues")
        return 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _do_hardware(wf: WorkflowAPI, args) -> int:
    """L3 (13 §9.5): diagnostic telemetry only — never training state."""
    info = status_layer.collect_hardware(wf.root)
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        for line in status_layer.render_hardware(info):
            print(line)
    return 0


def _attach(wf: WorkflowAPI, run_id: str) -> int:
    """--attach: a VIEWER, never the parent. Terminal death or Ctrl+C
    detaches the viewer only (13 §4.2, §9.2)."""
    print(f"attaching to {run_id} (Ctrl+C detaches the viewer; training continues)")
    seen = 0
    try:
        while True:
            events = wf.get_run_events(run_id)
            for ev in events[seen:]:
                print(json.dumps(ev, sort_keys=True))
            seen = len(events)
            state = wf.get_run_state(run_id)
            if state in _TERMINAL:
                print(f"run {run_id}: {state}")
                if state == "FAILED":
                    fail = wf.get_run_status(run_id).get("failure") or {}
                    print(f"  cause:    {fail.get('cause')}")
                    print(f"  recovery: {fail.get('recovery')}")
                return 0 if state in ("COMPLETED", "PAUSED", "STOPPED") else 4
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\ndetached — {run_id} keeps training")
        return 0


def _start_pipeline(
    wf: WorkflowAPI, args, runtime: dict, *, create: Any
) -> dict:
    """CREATE RUN → runtime config → VALIDATE → PREFLIGHT → LAUNCH.

    Shared by train/fork/retrain/finetune (13 §6.1): gate BLOCK → run
    FAILED[FORK_ONLY] exit 1, preflight BLOCK → exit 1, else queue the
    supervisor spawn — the CLI never becomes the training process."""
    run_id = create()
    _write_runtime_config(run_id, wf.root, runtime)
    # 13 §6.1: validated before anything expensive; gate BLOCK →
    # FAILED[FORK_ONLY] (failure matrix "VALIDATING → READY").
    report = wf.validate_run(run_id)
    if report.blocked:
        if not args.json:
            print(report.render())
        return {"status": "blocked", "run_id": run_id,
                "failed_step": report.failed_step, "exit": 1}
    pre = wf.preflight_run(
        run_id, gpu_required=_gpu_required_for(wf.root, run_id, runtime)
    )
    if pre.blocked:
        if not args.json:
            print(pre.render())
        f = pre.first_failure
        wf.preflight_fail(
            run_id, f"preflight: {f.label}: {f.detail}", pre.to_dict()
        )
        return {"status": "preflight_blocked", "run_id": run_id, "exit": 1}
    return {"status": "started", "run_id": run_id,
            **_launch(wf, run_id), "exit": 0}


def _finish_start(wf: WorkflowAPI, args, result: dict) -> int:
    run_id = result.get("run_id")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["status"] == "started":
        print(f"started {run_id} (supervisor pid {result.get('supervisor_pid')})")
    else:
        print(f"run {run_id}: {result['status'].upper()} (see report above)")
    if result["status"] == "started" and args.attach:
        return _attach(wf, run_id)
    return int(result["exit"])


def _execute_new_run(
    wf: WorkflowAPI, args, *, command: str, plan: Any, runtime: dict
) -> int:
    """SHOW DELTA → confirm → CREATE (once, via --command-id) → pipeline."""

    def _start() -> dict:
        return _start_pipeline(
            wf, args, runtime,
            create=lambda: wf.create_from_plan(plan).run_id,
        )

    result = (
        # A retry re-previews with a FRESH plan.run_id — the dedupe event
        # must land on the ORIGINAL run, so derive it from the stored
        # result (same rule as `train`, 13 §4.4).
        wf.execute_idempotent(args.command_id, command, _start,
                              run_id=lambda r: (r or {}).get("run_id")
                              if isinstance(r, dict) else None)
        if args.command_id else _start()
    )
    return _finish_start(wf, args, result)


def _do_train(wf: WorkflowAPI, args) -> int:
    spec, runtime = _load_config(args.config)
    # Fail-closed BEFORE anything exists: no real trainer for THIS model
    # and no explicit harness opt-in ⇒ exit 3, zero runs created (13 §7 —
    # never start what you cannot honestly run).
    require_trainable(runtime, model=spec.model)
    if not args.yes:
        _print_plan(spec, runtime)
        if not _confirm("Start training? [Y/n] "):
            print("Cancelled — nothing was created.")
            return 0

    def _start() -> dict:
        return _start_pipeline(
            wf, args, runtime, create=lambda: wf.create_run(spec).run_id
        )

    result = (
        wf.execute_idempotent(args.command_id, "train", _start,
                              run_id=lambda r: (r or {}).get("run_id")
                              if isinstance(r, dict) else None)
        if args.command_id else _start()
    )
    return _finish_start(wf, args, result)


# ---------------------------------------------------------------------------
# fork / retrain / finetune (13 §6.3/§6.4, build step 10)
# ---------------------------------------------------------------------------


def _merged_overrides(config_semantic: dict, pairs: list[str]) -> dict | list:
    """`--config` semantic is the BASE, `--set` applies on top (explicit
    wins). Returns a Mapping when config is present, else the raw pairs
    (workflow parses `KEY=VALUE` fail-closed)."""
    if not config_semantic:
        return pairs or None
    from mlforge.lineage import parse_overrides

    return {**config_semantic, **parse_overrides(pairs)}


def _do_fork(wf: WorkflowAPI, args) -> int:
    """Semantic change → NEW run with parent set (12 §4 hard rule:
    `mlforge fork`, never `--reconfigure`). Parent never mutates."""
    config_sem, runtime = _load_partial_config(args.config)
    plan = wf.preview_fork(
        args.run_id, overrides=_merged_overrides(config_sem, args.overrides),
        dataset=args.dataset,
    )
    if not args.yes:
        print(f"Forking {args.run_id}\n")
        print("This will create a NEW training run "
              "(semantic change = new experiment).\n")
        print(f"Previous run:  {plan.source_run_id}")
        print(f"New run:       {plan.run_id}")
        print("\nChanged:")
        _print_changed_lines(plan.delta)
        _print_plan_section(plan.spec, runtime)
        print()
        if not _confirm("Create new run? [Y/n] "):
            print("Cancelled — nothing was created.")
            return 0
    return _execute_new_run(wf, args, command="fork", plan=plan,
                            runtime=runtime)


def _do_retrain(wf: WorkflowAPI, args) -> int:
    """13 §6.3: load previous run_spec → SHOW DELTA → confirm → new run
    (fresh init, parent set)."""
    config_sem, runtime = _load_partial_config(args.config)
    plan = wf.preview_retrain(
        args.model, overrides=_merged_overrides(config_sem, args.overrides),
        dataset=args.dataset,
    )
    if not args.yes:
        print(f"Retraining {args.model}\n")
        print("This will create a NEW training run.\n")
        print(f"Previous run:  {plan.source_run_id}")
        print(f"New run:       {plan.run_id}")
        print(f"Starting state: {plan.starting_state}")
        datasets = plan.delta.get("train_datasets")
        if datasets:
            print(f"Dataset changed: {', '.join(datasets['from'])} → "
                  f"{', '.join(datasets['to'])}")
            print("\nThis is a new experiment.")
        else:
            print(f"Dataset:        {', '.join(plan.spec.train_datasets)}")
            print("Everything else: same as previous configuration")
        other = {k: v for k, v in plan.delta.items() if k != "train_datasets"}
        if other:
            print("\nChanged:")
            _print_changed_lines(other)
        _print_plan_section(plan.spec, runtime)
        print()
        if not _confirm("Create new run? [Y/n] "):
            print("Cancelled — nothing was created.")
            return 0
    return _execute_new_run(wf, args, command="retrain", plan=plan,
                            runtime=runtime)


def _do_finetune(wf: WorkflowAPI, args) -> int:
    """13 §6.4: resolve base model + hash → strategy → SHOW → confirm →
    NEW run (parent=model); base model is NEVER modified."""
    config_sem, runtime = _load_partial_config(args.config)
    plan = wf.preview_finetune(
        args.model,
        strategy=args.strategy,
        overrides=_merged_overrides(config_sem, args.overrides),
        dataset=args.dataset,
    )
    if not args.yes:
        print(f"Fine-tuning {plan.base_model}\n")
        print("This will create a NEW run + NEW model.\n")
        for warning in plan.warnings:
            print(f"  ! {warning}")
        print(f"\nBase model:    {plan.base_model}")
        checkpoint = plan.lineage.get("parent_checkpoint")
        if checkpoint:
            print(f"Base weights:  {checkpoint}")
        print(f"Base run:      {plan.source_run_id}")
        print(f"New run:       {plan.run_id}")
        print(f"Strategy:      {plan.spec.semantic['finetune_strategy']}")
        print(f"Dataset:       {', '.join(plan.spec.train_datasets)}")
        other = {k: v for k, v in plan.delta.items()
                 if k != "finetune_strategy"}
        if other:
            print("\nChanged:")
            _print_changed_lines(other)
        _print_plan_section(plan.spec, runtime)
        print()
        if not _confirm("Create new run? [Y/n] "):
            print("Cancelled — nothing was created.")
            return 0
    return _execute_new_run(wf, args, command="finetune", plan=plan,
                            runtime=runtime)


def _do_model(wf: WorkflowAPI, args) -> int:
    """`model list` / `model inspect` (+ lineage, contract) / `model
    import` — registry surfaces (13 §4.1)."""
    if args.model_command == "import":
        def _start() -> dict:
            entry = wf.import_external_model(args.path)
            return {
                "status": "imported",
                "exit": 0,
                "model_id": entry["model_id"],
                "model_ref": (f"{entry['name']}:{entry['version']}"
                              if entry.get("name") else entry["model_id"]),
                "state": entry["state"],
                "origin": entry.get("origin"),
                "artifact_hash": entry.get("artifact_hash"),
            }

        # §4.4: duplicate name:version blocks inside import; a retry with
        # the same command_id returns the original registration.
        result = (
            wf.execute_idempotent(
                args.command_id, "model_import", _start,
                kind="model",
                run_id=lambda r: (r or {}).get("model_id"),
            )
            if args.command_id else _start()
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"imported {result['model_ref']} ({result['model_id']}) — "
                  f"{result['state']}")
            print(f"  origin {result.get('origin')} · weights "
                  f"{_short_hash(result.get('artifact_hash'))} · "
                  f"contract created from the package (12 §15.3)")
        return 0
    if args.model_command == "list":
        models = wf.list_models()
        if args.json:
            print(json.dumps(models, indent=2, sort_keys=True))
        elif not models:
            print("no models — a model is published when a run COMPLETES "
                  "(13 §5.5)")
        else:
            print(f"{'REF':<24} {'STATE':<24} {'RUN':<30} ARTIFACT")
            for m in models:
                ref = (f"{m['name']}:{m['version']}"
                       if m.get("name") else m.get("model_id"))
                print(f"{ref:<24} {str(m.get('state')):<24} "
                      f"{str(m.get('run_id') or '—'):<30} "
                      f"{m.get('artifact_hash') or '—'}")
        return 0
    if args.model_command == "inspect":
        entry = wf.resolve_model(args.ref)
        lineage = None
        chain = None
        if entry.get("run_id"):
            try:
                lineage = wf.get_lineage(entry["run_id"])
                chain = [c["run_id"] for c in wf.lineage_chain(entry["run_id"])]
            except NotFound:
                lineage = None  # provenance run not in this workspace
        contract_dir = contract_source_dir(wf.root, entry)
        out = {**entry, "lineage": lineage, "chain": chain,
               "inference_contract_present": contract_dir is not None}
        if args.json:
            print(json.dumps(out, indent=2, sort_keys=True, default=str))
        else:
            ref = (f"{entry['name']}:{entry['version']}"
                   if entry.get("name") else entry["model_id"])
            print(f"MODEL                   {ref}")
            print(f"State                   {entry.get('state')}")
            print(f"Run                     {entry.get('run_id') or '—'}")
            print(f"Run spec                {entry.get('run_spec_hash') or '—'}")
            print(f"Artifact                {entry.get('artifact_hash') or '—'}")
            print(f"Inference contract      "
                  f"{'present (12 §15.3)' if contract_dir else 'none yet — export or import creates one'}")
            if lineage:
                print("\nLineage")
                print(f"  origin                {lineage.get('origin')}")
                print(f"  parent run            {lineage.get('parent_run') or '—'}")
                print(f"  parent model          {lineage.get('parent_model') or '—'}")
                print(f"  parent checkpoint     {lineage.get('parent_checkpoint') or '—'}")
                if chain:
                    print(f"  run chain             {' → '.join(chain)}")
        return 0
    return 4


def _gpu_required_for(root: Path, run_id: str, runtime: dict | None = None) -> bool:
    """`runtime["gpu"]` (train config) is the explicit override; then
    `runtime["device"]` naming cuda demands the probe (the trainer would
    refuse to place the run anywhere else); otherwise the validated
    execution plan decides — a plan measured on a GPU host requires
    `nvidia-smi` at preflight, a CPU/PORTABLE plan does not (12 §14).

    No plan yet (manual `preflight` before validation): fall back to live
    host detection — a CPU host never demands a GPU it cannot have; broken
    detection is a ValidationBlock ⇒ fail-closed True."""

    def _verdict(cfg: Any) -> bool | None:
        if not isinstance(cfg, dict):
            return None
        if "gpu" in cfg:
            return bool(cfg["gpu"])
        if str(cfg.get("device") or "").strip().lower().startswith("cuda"):
            return True
        return None

    verdict: bool | None = None
    if runtime is not None:
        verdict = _verdict(runtime)
    else:  # resume path: read the stored runtime config
        p = root / "runs" / run_id / "state" / "runtime.json"
        if p.is_file():
            try:
                verdict = _verdict(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
    if verdict is not None:
        return verdict
    try:
        plan = ExecutionPlan.read(root / "runs" / run_id / PLAN_FILENAME)
        return int(plan.capabilities.get("gpu_count") or 0) > 0
    except Exception:
        pass
    try:
        return detect_capabilities().gpu_count > 0
    except Exception:
        return True  # detection itself failed ⇒ fail-closed (12 §13.1)


def _do_resume(wf: WorkflowAPI, args) -> int:
    run_id = args.run_id
    state = wf.get_run_state(run_id)  # NotFound → exit 2
    if state == "FAILED":
        recovery = (wf.get_run_status(run_id).get("failure") or {}).get("recovery")
        if recovery != "RESUME":
            raise NoValidContinuation(run_id, cause=f"disposition {recovery}")

    def _continue() -> dict:
        st = wf.get_run_state(run_id)
        if st == "INTERRUPTED":
            # 13 §6.2: reconciliation BEFORE validation (12 §12.3 step 7
            # "emit report to user before any resume").
            rep = wf.reconcile_from_disk(run_id)
            if not args.json:
                print(f"RECONCILIATION of {run_id}: recorded "
                      f"{rep['prior_recorded_state']} → {rep['state_after']}")
                for s in rep.get("skips", []):
                    print(f"  skipped checkpoint-{s['ordinal']:06d}: {s['reason']}")
                print(f"  resume point: {rep.get('resume_point') or 'NONE'}")
            st = wf.get_run_state(run_id)
            if st == "FAILED":
                raise NoValidContinuation(run_id, cause="no valid checkpoint")

        if st == "READY":
            # validated earlier (e.g., launch was interrupted) — go straight
            # to launch; re-running the gate would re-acquire our own lease.
            pre = wf.preflight_run(
                run_id, gpu_required=_gpu_required_for(wf.root, run_id)
            )
            if pre.blocked:
                if not args.json:
                    print(pre.render())
                return {"status": "preflight_blocked", "run_id": run_id, "exit": 1}
            return {"status": "started", "run_id": run_id, **_launch(wf, run_id),
                    "exit": 0}

        # Preflight BEFORE the gate on the resume path: both failure modes
        # then leave the run completely untouched ("No changes were made").
        pre = wf.preflight_run(
                run_id, gpu_required=_gpu_required_for(wf.root, run_id)
            )
        if pre.blocked:
            if not args.json:
                print(pre.render())
            return {"status": "preflight_blocked", "run_id": run_id, "exit": 1}
        report = wf.validate_run(run_id)
        if report.blocked:
            if not args.json:
                print(report.render())
            return {"status": "blocked", "run_id": run_id,
                    "failed_step": report.failed_step, "exit": 1}
        if wf.get_run_state(run_id) != "READY":
            raise PreconditionFailed(
                f"run {run_id}: resume validation did not reach READY "
                f"(state {wf.get_run_state(run_id)})"
            )
        return {"status": "started", "run_id": run_id, **_launch(wf, run_id),
                "exit": 0}

    result = (
        wf.execute_idempotent(args.command_id, "resume", _continue,
                              run_id=run_id)
        if args.command_id else _continue()
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["status"] == "started":
        print(f"resuming {run_id} (supervisor pid {result.get('supervisor_pid')})")
    else:
        print(f"RESUME BLOCKED — run {run_id} unchanged "
              f"({result['status']}); see report above")
    if result["status"] == "started" and args.attach:
        return _attach(wf, run_id)
    return int(result["exit"])


def _do_pause(wf: WorkflowAPI, args) -> int:
    run_id = args.run_id
    state = wf.get_run_state(run_id)
    if state == "PAUSED":
        if args.json:
            print(json.dumps({"run_id": run_id, "state": state, "status": "already"}))
        else:
            print(f"{run_id} is already PAUSED")
        return 0
    if state not in ("RUNNING", "PAUSING"):
        raise PreconditionFailed(
            f"cannot pause {run_id} from state {state}",
            hint="pause applies to a RUNNING worker (13 §4.1)",
        )
    write_control(wf.root, run_id, "pause", requested_by="cli:pause")
    reached = wait_for_state(
        wf.root, run_id, ("PAUSED", "FAILED", "INTERRUPTED", "STOPPED", "COMPLETED"),
        timeout=args.timeout,
    )
    current = wf.get_run_state(run_id)
    if args.json:
        print(json.dumps({"run_id": run_id, "state": current,
                          "acknowledged": reached is not None}, sort_keys=True))
    if reached == "PAUSED":
        print(f"{run_id} → PAUSED (checkpoint committed; resumable)")
        return 0
    if reached == "FAILED":
        print(f"{run_id} → FAILED while pausing (see events)")
        return 4
    if reached is None:
        print(f"pause requested but not acknowledged within {args.timeout:.0f}s "
              f"(state {current}) — the worker may be dead; check "
              f"`mlforge status` / `mlforge lease status {run_id}`")
        return 3
    print(f"{run_id} → {current}")
    return 0


def _do_stop(wf: WorkflowAPI, args) -> int:
    run_id = args.run_id
    state = wf.get_run_state(run_id)
    if state == "STOPPED":
        if args.json:
            print(json.dumps({"run_id": run_id, "state": state, "status": "already"}))
        else:
            print(f"{run_id} is already STOPPED")
        return 0
    if state == "PAUSED":
        # no worker is alive: the explicit stop transitions are synchronous
        wf.stop(run_id)
        wf.stop_committed(run_id, "none")
        state = "STOPPED"
    elif state in ("RUNNING", "STOPPING"):
        write_control(wf.root, run_id, "stop", requested_by="cli:stop")
        reached = wait_for_state(
            wf.root, run_id, ("STOPPED", "FAILED", "INTERRUPTED", "COMPLETED", "PAUSED"),
            timeout=args.timeout,
        )
        if reached is None:
            current = wf.get_run_state(run_id)
            if args.json:
                print(json.dumps({"run_id": run_id, "state": current,
                                  "acknowledged": False}, sort_keys=True))
            print(f"stop requested but not acknowledged within {args.timeout:.0f}s "
                  f"(state {current}) — check `mlforge status`")
            return 3
        state = reached
    else:
        raise PreconditionFailed(
            f"cannot stop {run_id} from state {state}",
            hint="stop applies to RUNNING or PAUSED runs (13 §5.3)",
        )
    if args.json:
        print(json.dumps({"run_id": run_id, "state": state, "status": "stopped"},
                         sort_keys=True))
    else:
        print(f"{run_id} → {state} (no resume — stop is terminal)")
    return 0 if state in ("STOPPED", "PAUSED", "COMPLETED") else 4


# ---------------------------------------------------------------------------
# dataset / prepare (13 §4.1 DATA, §6.6; 12 §6.2–§6.4)
# ---------------------------------------------------------------------------


def _dataset_add(wf: WorkflowAPI, args) -> int:
    # Hash FIRST — a missing/unreadable path must fail before any
    # registration exists (verify, never discover: 13 §10).
    identity, manifest = ingest_recompute_identity(
        args.path, args.dataset_id, args.version, schema=args.schema
    )
    root = Path(wf.root)
    reg_path = root / "datasets" / args.dataset_id / "identity.json"
    existed = reg_path.is_file()
    same_identity = False
    if existed:
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        same_identity = reg.get("identity") == identity
        if not same_identity and not args.force:
            raise ValidationBlock(
                f"dataset {args.dataset_id!r} already registered with a "
                f"different identity (registered {reg.get('identity')}, "
                f"found {identity})",
                hint="use --force to re-register the changed content "
                     "(13 §5.2: never silently reinterpret)",
            )

    if not args.yes and not args.json:
        action = "Re-register" if existed and not same_identity else "Register"
        print(f"{action} dataset {args.dataset_id} "
              f"({manifest.file_count} files, {manifest.total_bytes} bytes)")
        print(f"  path:     {args.path}")
        print(f"  identity: {identity}")
        if not _confirm("Proceed? [Y/n] "):
            print("Cancelled — nothing was registered.")
            return 0

    def _add() -> dict:
        if not existed:
            wf.register_dataset(
                args.dataset_id, identity, version=args.version,
                schema=args.schema, file_count=manifest.file_count,
                total_bytes=manifest.total_bytes,
            )
            status = "registered"
        elif same_identity:
            # Same bytes: only the machine-local path may move (12 §6.3) —
            # identity untouched, no state transition.
            status = "path_updated"
        else:
            wf.reregister_dataset(
                args.dataset_id, identity, version=args.version,
                schema=args.schema, file_count=manifest.file_count,
                total_bytes=manifest.total_bytes,
            )
            status = "reregistered"
        ingest_set_path(root, args.dataset_id, args.path)
        ingest_write_registry(root, args.dataset_id, {
            "version": args.version,
            "identity": identity,
            "schema": args.schema,
            "file_count": manifest.file_count,
            "total_bytes": manifest.total_bytes,
        })
        wf.note_dataset_path(args.dataset_id, args.path)
        return {
            "status": status,
            "dataset": args.dataset_id,
            "version": args.version,
            "identity": identity,
            "file_count": manifest.file_count,
            "total_bytes": manifest.total_bytes,
            "path": str(args.path),
        }

    result = (
        wf.execute_idempotent(args.command_id, "dataset_add", _add,
                              kind="dataset", run_id=args.dataset_id)
        if args.command_id else _add()
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{args.dataset_id}: {result['status'].upper()} — "
              f"{result['identity']}")
        candidates = suggest_transforms(
            [e.relative_path for e in manifest.files])
        if candidates:
            print(f"  likely transform: {' / '.join(candidates)}")
            print(f"  next: mlforge dataset verify {args.dataset_id} → set "
                  "`transform:` (mlforge dataset types) in ingestion.yaml "
                  "→ mlforge prepare <model>")
    return 0


def _dataset_list(wf: WorkflowAPI, args) -> int:
    paths = ingest_load_paths(wf.root)
    rows = []
    for proj in wf.list_datasets():
        did = proj["id"]
        reg: dict[str, Any] = {}
        p = Path(wf.root) / "datasets" / did / "identity.json"
        if p.is_file():
            try:
                reg = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                reg = {}
        rows.append({
            "dataset": did,
            "state": proj["state"],
            "version": reg.get("version", "v1"),
            "identity": reg.get("identity"),
            "file_count": reg.get("file_count"),
            "path": paths.get(did),
        })
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("no datasets registered — mlforge dataset add <ID> <PATH>")
        return 0
    w_name = max(len("NAME"), max(len(r["dataset"]) for r in rows))
    w_state = max(len("STATE"), max(len(r["state"]) for r in rows))
    print(f"{'NAME':<{w_name}}  {'STATE':<{w_state}}  VERSION  FILES  PATH")
    for r in rows:
        print(f"{r['dataset']:<{w_name}}  {r['state']:<{w_state}}  "
              f"{r['version']:<7}  {str(r['file_count'] or '-'):>5}  "
              f"{r['path'] or '(not configured)'}")
    return 0


def _dataset_verify(wf: WorkflowAPI, args) -> int:
    root = Path(wf.root)
    reg_path = root / "datasets" / args.dataset_id / "identity.json"
    if not reg_path.is_file():
        raise NotFound(
            f"dataset {args.dataset_id!r} not registered",
            hint=f"mlforge dataset add {args.dataset_id} <PATH>",
        )
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    state = wf.get_dataset_status(args.dataset_id)["state"]
    if state in ("PREPARED", "REJECTED"):
        raise PreconditionFailed(
            f"dataset {args.dataset_id!r} is {state} — verify applies before "
            f"prepare; re-register to re-verify",
            hint=f"mlforge dataset add {args.dataset_id} <PATH> --force",
        )
    path = ingest_load_paths(root).get(args.dataset_id)
    if not path:
        raise PreconditionFailed(
            f"dataset {args.dataset_id!r}: no machine-local path configured",
            hint=f"mlforge dataset add {args.dataset_id} <PATH>",
        )

    def _verify() -> dict:
        found, manifest = ingest_recompute_identity(
            path, args.dataset_id, str(reg.get("version", "v1")),
            schema=reg.get("schema"),
        )
        matches = found == reg.get("identity")
        if matches:
            if state == "REGISTERED":
                wf.verify_dataset(args.dataset_id, identity_matches=True,
                                  found_identity=found)
            return {
                "status": "verified",
                "dataset": args.dataset_id,
                "state": "VERIFIED",
                "identity": found,
                "file_count": manifest.file_count,
                "path": str(path),
            }
        # Mismatch → REJECTED (13 §5.2), recorded, exit 1 (never a guess).
        wf.verify_dataset(args.dataset_id, identity_matches=False,
                          found_identity=found)
        return {
            "status": "rejected",
            "dataset": args.dataset_id,
            "state": "REJECTED",
            "registered_identity": reg.get("identity"),
            "identity": found,
            "file_count": manifest.file_count,
            "path": str(path),
        }

    result = (
        wf.execute_idempotent(args.command_id, "dataset_verify", _verify,
                              kind="dataset", run_id=args.dataset_id)
        if args.command_id else _verify()
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["status"] == "verified":
        print(f"{args.dataset_id}: VERIFIED — {result['file_count']} files "
              f"hashed ({result['identity']})")
    else:
        print(f"{args.dataset_id}: REJECTED — content at {path} does not "
              f"match its registration (found {result['identity']}, "
              f"registered {result['registered_identity']})")
        print("  nothing was silently reinterpreted — re-register with "
              "dataset add --force if the change is intended")
    return 0 if result["status"] == "verified" else 1


def _do_dataset(wf: WorkflowAPI, args) -> int:
    if args.dataset_command == "add":
        return _dataset_add(wf, args)
    if args.dataset_command == "list":
        return _dataset_list(wf, args)
    if args.dataset_command == "verify":
        return _dataset_verify(wf, args)
    if args.dataset_command == "types":
        return _dataset_types(args)
    return 4  # argparse required=True keeps this unreachable


def _dataset_types(args) -> int:
    """The answer to 'what options are there, which does what' — a stable,
    readable catalog of input type → transform → output, plus the honest
    gaps (never faked, fail-closed)."""
    catalog = ingest_dataset_types()
    if args.json:
        print(json.dumps(catalog, indent=2, sort_keys=True))
        return 0
    print("MLForge dataset types — set one as `transform:` in ingestion.yaml")
    print()
    for entry in catalog["supported"]:
        print(f"  {entry['name']}   [{entry['title']}]")
        print(f"    input:    {entry['inputs']}")
        print(f"    produces: {entry['produces']}")
        print()
    print("Not supported yet — fail-closed, never faked:")
    for entry in catalog["planned"]:
        print(f"  {entry['name']} ({entry['title']}): {entry['reason']}")
    print()
    print("Then, in ingestion.yaml (12 §10.2):")
    print("  models:")
    print("    <model_name>:")
    print(f"      transform: {catalog['supported'][0]['name']}   "
          "# one of the names above")
    print("      train_sources: [<dataset_id>:train]")
    print("  and run:  mlforge prepare <model_name>")
    return 0


def _do_prepare(wf: WorkflowAPI, args) -> int:
    def _prep() -> dict:
        return ingest_prepare(wf.root, args.model, workflow=wf).to_dict()

    result = (
        wf.execute_idempotent(args.command_id, "prepare", _prep,
                              kind="dataset",
                              run_id=lambda r: (r or {}).get("dataset_id"))
        if args.command_id else _prep()
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result.get("cache") == "hit":
        print(f"{result['dataset']}: cache hit — artifact unchanged "
              f"({result['record_count']} records)")
    else:
        print(f"prepared {result['dataset']} — {result['record_count']} "
              f"records ({', '.join(result['sources'])})")
    return 0


# ---------------------------------------------------------------------------
# evaluate / compare / infer / export / package (13 §6.7–§6.10, 12 §15;
# build step 11)
# ---------------------------------------------------------------------------


def _short_hash(value: Any) -> str:
    """Readable hash for tables — full values always stay in --json."""
    if value is None:
        return "—"
    text = str(value)
    return text if len(text) <= 20 else text[:19] + "…"


def _load_protocol(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise PreconditionFailed(
            f"protocol file not found: {path}",
            hint='expected JSON, e.g. {"seed": 7, "split": "val"}',
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationBlock(
            f"protocol file {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValidationBlock(
            "protocol file must be a JSON object",
            hint='e.g. {"seed": 7, "thresholds": {"iou": 0.5}}',
        )
    return data


def _print_eval_preview(model_ref: str, rec: dict) -> None:
    """13 §6.7 SHOW PROTOCOL — built by the same deterministic code that
    writes the artifact: what is shown is exactly what gets recorded."""
    proto = rec["evaluation_protocol"]
    thr = proto["thresholds"]
    nms = proto["nms"]
    print(f"Evaluating {model_ref}\n")
    print(f"  Dataset       {rec['dataset_ref']}  identity "
          f"{_short_hash(rec['dataset_hash'])}")
    print(f"  Model hash    {_short_hash(rec['model_hash'])}")
    print(f"  Code hash     {_short_hash(rec['code_hash'])}")
    print(f"  Environment   {_short_hash(rec['environment_hash'])}")
    print("  Protocol")
    print(f"    split        {proto['split']}")
    metric_names = ", ".join(rec.get("metrics") or DEFAULT_METRIC_NAMES)
    print(f"    metrics      {metric_names} "
          f"(definitions {_short_hash(proto['metric_definitions'])})")
    print(f"    aggregation  {proto['aggregation']}")
    print(f"    thresholds   iou={thr['iou']} conf={thr['conf']}")
    print(f"    nms          iou_thresh={nms['iou_thresh']} "
          f"max_det={nms['max_det']}")
    print(f"    seed         {proto['seed']}")
    print(f"    harness      {_short_hash(proto['harness_code_hash'])}")
    print("\n  evaluation identity = model + dataset + code + environment "
          "+ protocol (five components, 12 §15.4)")
    print(f"  harness: {rec['harness']} — {rec['note']}")
    print()


def _do_evaluate(wf: WorkflowAPI, args) -> int:
    """13 §6.7: SHOW PROTOCOL → confirm → immutable evaluation artifact.
    Resolution/validation run during the preview (unknown dataset/model →
    exit 2, unknown protocol keys → exit 1) before anything is written."""
    protocol = _load_protocol(args.protocol) if args.protocol else None
    if not args.yes and not args.json:
        preview = wf.preview_evaluation(
            args.model, args.dataset, protocol_overrides=protocol
        )
        _print_eval_preview(args.model, preview)
        if not _confirm("Produce evaluation artifact? [Y/n] "):
            print("Cancelled — no evaluation artifact was written.")
            return 0

    def _start() -> dict:
        rec = wf.evaluate_model(
            args.model, args.dataset, protocol_overrides=protocol
        )
        return {
            "status": "evaluated",
            "exit": 0,
            "eval_id": rec["eval_id"],
            "uri": f"evaluation://{rec['eval_id']}",
            "model_ref": rec["model_ref"],
            "model_id": rec["model_id"],
            "dataset_ref": rec["dataset_ref"],
            "metrics": rec["metrics"],
            "evaluation_protocol_hash": rec["evaluation_protocol_hash"],
            "identity": rec["identity"],
            "harness": rec["harness"],
        }

    # §4.4: a finished evaluation is never recomputed — the retry returns
    # the original eval_id.
    result = (
        wf.execute_idempotent(
            args.command_id, "evaluate", _start,
            kind="model",
            run_id=lambda r: (r or {}).get("model_id"),
        )
        if args.command_id else _start()
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        metrics = " ".join(
            f"{name} {value}" for name, value in result["metrics"].items()
        )
        print(f"{result['uri']}   {metrics}")
    return 0


def _print_comparison(data: dict) -> None:
    """13 §6.10 — descriptive side-by-side; never averaged across
    protocols, never a winner for the user to hide behind."""
    cols: list[dict] = data["columns"]
    refs = [c["ref"] for c in cols]

    def cell(col: dict, pick, missing: str = "—") -> str:
        ev = col.get("evaluation")
        return pick(ev) if ev is not None else missing

    # Metric rows: the protocols' own metric names (a real engine's
    # family metrics show up here; scaffold columns keep mAP/AP50 first).
    ordered: list[str] = [
        name for name in DEFAULT_METRIC_NAMES
        if any(name in (c["evaluation"] or {}).get("metrics", {})
               for c in cols if c.get("evaluation"))
    ]
    if not ordered and not any(c.get("evaluation") for c in cols):
        ordered = list(DEFAULT_METRIC_NAMES)  # no evaluations yet
    for col in cols:
        for name in (col.get("evaluation") or {}).get("metrics", {}):
            if name not in ordered:
                ordered.append(name)

    rows: list[tuple[str, list[str]]] = [
        ("Dataset", [c["dataset"] for c in cols]),
        ("Base model", [c["base_model"] for c in cols]),
    ]
    for name in ordered:
        rows.append((
            name,
            [cell(c, lambda ev, n=name: str(ev["metrics"].get(n, "—")))
             for c in cols],
        ))
    rows.append((
        "Evaluation",
        [cell(c, lambda ev: ev.get("eval_id") or "—") for c in cols],
    ))
    rows.append((
        "Protocol",
        [cell(c, lambda ev: _short_hash(ev["evaluation_protocol_hash"]),
              missing="no evaluation")
         for c in cols],
    ))

    def comparable_cell(col: dict) -> str:
        if col.get("evaluation") is None:
            return "no evaluation"
        return "yes" if col.get("comparable") else "NOT_COMPARABLE"

    rows.append(("Comparable", [comparable_cell(c) for c in cols]))

    label_w = max([len("Comparable")] + [len(lbl) for lbl, _ in rows])
    widths = [
        max([len(refs[i])] + [len(cells[i]) for _, cells in rows])
        for i in range(len(cols))
    ]

    print("MODEL COMPARISON\n")
    header = " " * label_w
    for ref, w in zip(refs, widths):
        header += f"  {ref:<{w}}"
    print(header)
    for label, cells in rows:
        line = f"{label:<{label_w}}"
        for text, w in zip(cells, widths):
            line += f"  {text:<{w}}"
        print(line)

    has_not_comparable = False
    for col in cols:
        ev = col.get("evaluation")
        if ev is None:
            print(f"\n{col['ref']}: no evaluation yet — `mlforge evaluate "
                  f"{col['ref']} --dataset <DATASET>`")
        elif not col.get("comparable"):
            has_not_comparable = True
            print(f"\nNOT_COMPARABLE: {col['ref']} used evaluation protocol "
                  f"{_short_hash(col.get('protocol_hash'))} — shown, never "
                  f"averaged (13 §6.10)")
    if has_not_comparable:
        print("\nNOT_COMPARABLE = a different evaluation protocol — rows stay "
              "side by side and are never averaged into one number.")
    print("\n(descriptive only — comparison does not choose a winner for "
          "the user)")


def _do_compare(wf: WorkflowAPI, args) -> int:
    """13 §6.10: read-only comparison — exit 2 for unknown refs, no
    confirmation (nothing is written)."""
    data = wf.compare_models(args.models)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        _print_comparison(data)
    return 0


def _do_infer(wf: WorkflowAPI, args) -> int:
    """13 §6.8: contract check → SAFE | BLOCK (exit 1) → output envelope.
    The contract itself is never weakened to accept an input — BLOCK."""
    rec = wf.infer_model(args.model, args.input)
    if args.json:
        print(json.dumps(rec, indent=2, sort_keys=True))
        return 0
    runtime = rec.get("runtime") or {}
    print(f"infer {rec['model_ref']} ← {args.input}")
    print(f"  contract check: SAFE ({_short_hash(rec.get('contract_hash'))} "
          f"· {runtime.get('framework', '—')})")
    print(f"  output:   outputs/{rec['output_id']}.json")
    print(f"  identity: {_short_hash(rec.get('identity'))}")
    note = (rec.get("result") or {}).get("note")
    if note:
        print(f"  note: {note}")
    return 0


def _do_export(wf: WorkflowAPI, args) -> int:
    """13 §6.9: validation report → immutable export + inference contract.
    A BLOCK (unknown format / unsupported operator) exits 1 and leaves
    nothing behind — never a partial export."""

    def _start() -> dict:
        rec = wf.export_model(args.model, args.fmt)
        return {**rec, "status": "exported", "exit": 0}

    result = (
        wf.execute_idempotent(
            args.command_id, "export", _start,
            kind="model",
            run_id=lambda r: (r or {}).get("model_id"),
        )
        if args.command_id else _start()
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    labels = {"architecture": "Architecture", "operators": "Operators",
              "dynamic_shapes": "Dynamic shapes"}
    validations = result.get("validations") or {}
    num = result.get("numerical_validation") or {}
    print("Export")
    print(f"  Source: {result['model_ref']} → "
          f"Target: {str(result.get('format', '')).upper()}")
    print("  Validating: " + " · ".join(
        f"{labels.get(key, key)} {value}"
        for key, value in validations.items()
    ))
    print(f"  Numerical validation: max error {num.get('max_error')} "
          f"(tolerance {num.get('tolerance')}) {num.get('result')}  "
          f"[{num.get('harness')} harness]")
    print(f"  Export artifact: {result['export_id']} "
          f"(own artifact identity {_short_hash(result.get('identity'))})")
    return 0


def _do_package(wf: WorkflowAPI, args) -> int:
    """12 §15.3: contract required, secrets scan, immutable bundle.
    Packaging changes NO state — it only records bundle_created."""

    def _start() -> dict:
        rec = wf.package_model(args.model)
        return {**rec, "status": "packaged", "exit": 0}

    result = (
        wf.execute_idempotent(
            args.command_id, "package", _start,
            kind="model",
            run_id=lambda r: (r or {}).get("model_id"),
        )
        if args.command_id else _start()
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    integrity = result.get("integrity") or {}
    print("Package")
    print(f"  Model:     {result['model_ref']} "
          f"(contract {_short_hash(result.get('contract_hash'))})")
    print("  Secrets:   PASS (model, run, and contract sources scanned)")
    print(f"  Integrity: {' · '.join(integrity)}")
    print(f"  Bundle:    {result['bundle_id']} → "
          f"bundles/{result['bundle_id']}/")
    print(f"  Files:     {' '.join(result.get('files') or [])}")
    return 0


def _do_gui(wf: WorkflowAPI, args) -> int:
    """`mlforge gui` — the localhost web viewer (13 §9.2 control plane).
    Same Workflow API as CLI/TUI (§1 rule 2); pages are observations and
    buttons write control intents only. Killing this process changes
    nothing about any run (§9.1 STATUS DOWN → TRAINING CONTINUES)."""
    from mlforge.ui.web import serve

    if not (0 <= args.port <= 65535):
        raise ValidationBlock(
            f"invalid --port: {args.port}",
            hint="valid TCP ports are 1-65535 (0 = ephemeral, for tests)",
        )
    try:
        return serve(wf, port=args.port, open_browser=args.open_browser)
    except OSError as exc:
        raise PreconditionFailed(
            f"cannot bind 127.0.0.1:{args.port}: {exc.strerror or exc}",
            hint="pick another --port (the GUI binds localhost only)",
        ) from exc


def _do_hello(args) -> int:
    """Bare `mlforge` — a contextual landing screen, not a usage dump.

    Outside a project: the shortest working recipe (init → add →
    prepare → train). Inside: the workspace's real facts (datasets,
    runs) and what to do next. Exit 0 either way; never crashes on a
    broken workspace (the landing must always render)."""
    root = Path(args.root).expanduser().resolve()
    print("MLForge — hardware-agnostic ML training system "
          "(spec: 10_training_plan/12 + 13)")
    print("usage: mlforge <command> [options]        "
          "mlforge --help · mlforge <command> --help · mlforge --version")
    print()

    proj_file = root / "project.yaml"
    if not proj_file.is_file():
        print("No MLForge project in this folder yet.")
        print()
        print("Start here:")
        print("  mlforge init myproj && cd myproj   # create a workspace")
        print("  mlforge dataset types              "
              "# what formats are supported, which transform fits")
        print("  mlforge dataset add books /path/to/books --yes"
              "   # registers identity + this machine's path")
        print("  mlforge dataset verify books         "
              "# prove the bytes still match (prepare gate)")
        print("  $EDITOR ingestion.yaml             "
              "# say which transform feeds which model")
        print("  mlforge prepare <model>            # extract → chunk")
        print("  mlforge train --config configs/train.example.json --yes")
        print("  mlforge watch                      # live loss dashboard")
        return 0

    try:
        from mlforge.ingest.config import load_paths, load_registry

        meta = yaml_load_file(proj_file)
        name = str((meta or {}).get("name") or root.name)
        registered = sorted(load_registry(root))
        configured = sorted(load_paths(root))
        runs = WorkflowAPI(root).list_runs()

        print(f"Project: {name}  ({root})")
        if registered:
            configured_note = (
                f", {len(configured)} configured here"
                if configured else ", path not configured on this machine"
            )
            print(f"  datasets: {', '.join(registered)} "
                  f"({len(registered)} registered{configured_note})")
        else:
            print("  datasets: none registered yet")
        if runs:
            states: dict[str, int] = {}
            for run in runs:
                state = str(run.get("state") or "?")
                states[state] = states.get(state, 0) + 1
            summary = ", ".join(f"{s} ×{n}" for s, n in sorted(states.items()))
            print(f"  runs: {len(runs)} total ({summary})")
            latest = runs[-1]
            print(f"  latest: {latest.get('id')} — {latest.get('state')}")
        else:
            print("  runs: none yet")
        print()
        print("Try next:")
        if not runs:
            if not registered:
                print("  mlforge dataset add <ID> /path/to/data --yes "
                      "# a folder can live anywhere")
            else:
                print("  mlforge prepare <model>            # if not done")
            print("  mlforge train --config configs/train.example.json")
            print("  mlforge watch                      # live dashboard")
        else:
            print("  mlforge status                     # all runs")
            print("  mlforge watch                      # live dashboard")
            print("  mlforge dataset add <ID> /path/to/data --yes  "
                  "# add more data any time")
        print("  mlforge --help                     # every command")
    except Exception:
        # A broken workspace must never break the landing screen —
        # point at the diagnostic that owns the detail instead.
        print("(workspace partially unreadable — run `mlforge status` "
              "for the diagnostic)")
    return 0


def main(argv: list[str] | None = None, *, wf_factory=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        return _do_hello(args)

    if args.command in _PENDING:
        step = _PENDING[args.command]
        print(
            f"[NOT_IMPLEMENTED] `mlforge {args.command}` arrives in build step {step} "
            f"(13 §11). Nothing was executed.",
            file=sys.stderr,
        )
        return 4

    if args.command == "serve":
        # Deliberately NOT in _PENDING: no build step (13 §11 runs through
        # 12) will ever deliver it — honest message, still fail-closed.
        print(
            "[NOT_IMPLEMENTED] `mlforge serve` (long-running model server, "
            "13 §4.1) has no build step yet (13 §11 covers through 12) — "
            "`mlforge infer` is the contract-checked one-shot path today. "
            "Nothing was executed.",
            file=sys.stderr,
        )
        return 4

    try:
        wf = (wf_factory or WorkflowAPI)(args.root)
        if args.command == "init":
            return _do_init(wf, args)
        if args.command == "configure":
            return _do_configure(wf, args)
        if args.command == "train":
            return _do_train(wf, args)
        if args.command == "fork":
            return _do_fork(wf, args)
        if args.command == "retrain":
            return _do_retrain(wf, args)
        if args.command == "finetune":
            return _do_finetune(wf, args)
        if args.command == "model":
            return _do_model(wf, args)
        if args.command == "resume":
            return _do_resume(wf, args)
        if args.command == "pause":
            return _do_pause(wf, args)
        if args.command == "stop":
            return _do_stop(wf, args)
        if args.command == "dataset":
            return _do_dataset(wf, args)
        if args.command == "prepare":
            return _do_prepare(wf, args)
        if args.command == "evaluate":
            return _do_evaluate(wf, args)
        if args.command == "compare":
            return _do_compare(wf, args)
        if args.command == "infer":
            return _do_infer(wf, args)
        if args.command == "export":
            return _do_export(wf, args)
        if args.command == "package":
            return _do_package(wf, args)
        if args.command == "gui":
            return _do_gui(wf, args)
        if args.command == "status":
            return _do_status(wf, args)

        if args.command == "watch":
            return _do_watch(wf, args)

        if args.command == "hardware":
            return _do_hardware(wf, args)

        if args.command == "inspect":
            try:
                out = dict(wf.get_run_status(args.object_id))
            except NotFound as run_err:
                # `inspect` also resolves model refs (13 §4.3): a ref that
                # is not a run falls through to the registry; a genuine
                # unknown id keeps exit 2 with the run's message.
                try:
                    entry = wf.resolve_model(args.object_id)
                except NotFound:
                    raise run_err from None
                lineage = None
                if entry.get("run_id"):
                    try:
                        lineage = wf.get_lineage(entry["run_id"])
                    except NotFound:
                        lineage = None
                print(json.dumps({**entry, "lineage": lineage}, indent=2,
                                 sort_keys=True, default=str))
                return 0
            out["lineage"] = wf.get_lineage(args.object_id)
            print(json.dumps(out, indent=2, sort_keys=True))
            return 0

        if args.command == "events":
            events = wf.get_run_events(args.run_id)
            for e in events:
                print(json.dumps(e, sort_keys=True), flush=bool(args.follow))
            if args.follow:
                seen = len(events)
                try:
                    while True:
                        time.sleep(0.5)
                        more = wf.get_run_events(args.run_id)
                        for e in more[seen:]:
                            print(json.dumps(e, sort_keys=True), flush=True)
                        seen = len(more)
                except KeyboardInterrupt:
                    print()  # clean detach — the stream itself never ends
            return 0

        if args.command == "validate":
            obj = args.object_id
            try:
                report = wf.validate_run(obj)
            except NotFound as run_err:
                # 13 §4.1: `validate <RUN|MODEL>` — a model ref falls
                # through to registry/contract integrity checks; a genuine
                # unknown id keeps exit 2 with the run's message.
                try:
                    mreport = wf.model_integrity_report(obj)
                except NotFound:
                    raise run_err from None
                if args.json:
                    print(json.dumps(mreport, indent=2, sort_keys=True))
                else:
                    print(f"MODEL VALIDATION  {mreport['model_ref']}  "
                          f"[{mreport['state']}]")
                    for check in mreport["checks"]:
                        print(f"[{check['status']}] {check['name']}: "
                              f"{check['detail']}")
                    print("RESULT: " + ("MODEL BLOCKED" if mreport["blocked"]
                                        else "MODEL OK"))
                return 1 if mreport["blocked"] else 0
            if args.json:
                print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            else:
                print(report.render())
            # 12 §7.2: any FAIL → training does not start (exit 1).
            return 1 if report.blocked else 0

        if args.command == "preflight":
            # Plan/host-derived GPU expectation by default (same story as
            # train/resume); --gpu/--no-gpu are explicit user overrides.
            gpu = (
                True
                if args.gpu
                else False if args.no_gpu else _gpu_required_for(wf.root, args.run_id)
            )
            report = wf.preflight_run(args.run_id, gpu_required=gpu)
            if args.json:
                print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            else:
                print(report.render())
            return 1 if report.blocked else 0

        if args.command == "lease":
            sub_cmd = getattr(args, "lease_command", None)
            if sub_cmd == "status":
                info = wf.lease_status(args.run_id)
                if args.json:
                    print(json.dumps(info, indent=2, sort_keys=True))
                elif info["state"] == "FREE":
                    print(f"{args.run_id}: no lease (FREE)")
                else:
                    print(
                        f"{args.run_id}: {info['state']} — held by "
                        f"pid {info['pid']} on {info['host']} "
                        f"(heartbeat {info['age_seconds']:.0f}s ago)"
                    )
                return 0
            if sub_cmd == "break":
                result = wf.lease_break(
                    args.run_id,
                    force=args.force,
                    yes=args.yes,
                    reason=args.reason,
                )
                if args.json:
                    print(json.dumps(result, indent=2, sort_keys=True, default=str))
                else:
                    prev = result["previous_owner"]
                    print(
                        f"LEASE BROKEN — was pid {prev.get('pid')} on "
                        f"{prev.get('host')} (reason: {result['reason']}, "
                        f"operator: {result['operator']}) — LEASE_BROKEN logged"
                    )
                return 0
            print(
                "[NOT_IMPLEMENTED] `mlforge lease` supports `status` and "
                "`break` in this build step. Nothing was executed.",
                file=sys.stderr,
            )
            return 4

        if args.command == "store":
            if getattr(args, "store_command", None) != "gc":
                print(
                    "[NOT_IMPLEMENTED] `mlforge store` has only the `gc` "
                    "subcommand in this build step. Nothing was executed.",
                    file=sys.stderr,
                )
                return 4
            # 12 §6.1: GC is a command, never automatic during training.
            root = Path(args.root)
            store = ContentStore(root / "store")
            registry = ArtifactRegistry(root, store)
            report = registry.gc(
                grace_seconds=args.grace_days * 86400.0,
                dry_run=not args.execute,
            )
            if args.json:
                print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            else:
                mode = "DRY-RUN (use --execute to delete)" if report.dry_run else "EXECUTED"
                print(f"GC {mode}")
                print(f"  reachable:     {report.reachable}")
                print(f"  swept:         {len(report.swept)}")
                for h in report.swept:
                    print(f"    - {h}")
                print(f"  kept (grace):  {len(report.kept_grace)}")
                if report.skipped_lease:
                    print(
                        "  blocked by active lease: "
                        + ", ".join(report.skipped_lease)
                    )
            return 0

        return 4
    except MlforgeError as exc:
        print(exc.render(), file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
