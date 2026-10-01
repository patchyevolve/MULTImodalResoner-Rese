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
import sys
import time
from pathlib import Path

import mlforge
from mlforge.errors import MlforgeError, NoValidContinuation, PreconditionFailed
from mlforge.leases import LeaseState, RunLeaseManager
from mlforge.run_spec import RunSpec
from mlforge.runtime.control import wait_for_state, write_control
from mlforge.store import ArtifactRegistry, ContentStore
from mlforge.supervisor import enqueue_spawn, ensure_supervisor
from mlforge.validation import Preflight, ValidationGate
from mlforge.workflow import WorkflowAPI

# 13 §11 build-order gates: implemented vs pending.
_IMPLEMENTED = {
    "status", "inspect", "events", "store", "validate", "preflight", "lease",
    "train", "resume", "pause", "stop",
}
_PENDING = {
    "init": 1,
    "configure": 1,
    "prepare": 8,
    "retrain": 10,
    "finetune": 10,
    "evaluate": 11,
    "compare": 11,
    "infer": 11,
    "export": 11,
    "package": 11,
    "watch": 7,
    "hardware": 7,
}

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

    st = sub.add_parser("status", help="overview of all runs (read-only, L1)")
    st.add_argument("--json", action="store_true", help="machine-readable output")

    insp = sub.add_parser("inspect", help="detailed report for one object")
    insp.add_argument("object_id")
    insp.add_argument("--json", action="store_true")

    ev = sub.add_parser("events", help="structured event stream for a run")
    ev.add_argument("run_id")

    val = sub.add_parser(
        "validate", help="run the 19-step validation gate for a run (fail-closed)"
    )
    val.add_argument("run_id")
    val.add_argument("--json", action="store_true")

    pf = sub.add_parser(
        "preflight",
        help="host + identity preflight before any expensive operation (report-only)",
    )
    pf.add_argument("run_id")
    pf.add_argument("--json", action="store_true")
    pf.add_argument(
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
    print()


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
    the worker (fired from READY) is a daemon child, never ours (12 §12.4)."""
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


def _do_train(wf: WorkflowAPI, args) -> int:
    spec, runtime = _load_config(args.config)
    if not args.yes:
        _print_plan(spec, runtime)
        if not _confirm("Start training? [Y/n] "):
            print("Cancelled — nothing was created.")
            return 0

    def _start() -> dict:
        h = wf.create_run(spec)
        run_id = h.run_id
        _write_runtime_config(run_id, wf.root, runtime)
        # 13 §6.1: validated before anything expensive; gate BLOCK →
        # FAILED[FORK_ONLY] (failure matrix "VALIDATING → READY").
        report = wf.validate_run(run_id)
        if report.blocked:
            if not args.json:
                print(report.render())
            return {"status": "blocked", "run_id": run_id,
                    "failed_step": report.failed_step, "exit": 1}
        pre = wf.preflight_run(run_id, gpu_required=bool(runtime.get("gpu", True)))
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

    result = (
        wf.execute_idempotent(args.command_id, "train", _start,
                              run_id=lambda r: (r or {}).get("run_id")
                              if isinstance(r, dict) else None)
        if args.command_id else _start()
    )
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


def _resume_gpu_required(root: Path, run_id: str) -> bool:
    """GPU expectation travels with the run's runtime config (written at
    train time); default = GPU required (fail-closed for real training)."""
    p = root / "runs" / run_id / "state" / "runtime.json"
    if p.is_file():
        try:
            return bool(json.loads(p.read_text(encoding="utf-8")).get("gpu", True))
        except json.JSONDecodeError:
            pass
    return True


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
                run_id, gpu_required=_resume_gpu_required(wf.root, run_id)
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
                run_id, gpu_required=_resume_gpu_required(wf.root, run_id)
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


def main(argv: list[str] | None = None, *, wf_factory=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command in _PENDING:
        step = _PENDING[args.command]
        print(
            f"[NOT_IMPLEMENTED] `mlforge {args.command}` arrives in build step {step} "
            f"(13 §11). Nothing was executed.",
            file=sys.stderr,
        )
        return 4

    try:
        wf = (wf_factory or WorkflowAPI)(args.root)
        if args.command == "train":
            return _do_train(wf, args)
        if args.command == "resume":
            return _do_resume(wf, args)
        if args.command == "pause":
            return _do_pause(wf, args)
        if args.command == "stop":
            return _do_stop(wf, args)
        if args.command == "status":
            runs = wf.list_runs()
            if args.json:
                print(json.dumps(runs, indent=2, sort_keys=True))
            elif not runs:
                print("No runs.")
            else:
                for r in runs:
                    fail = r.get("failure")
                    line = f"{r['id']}  {r['state']}"
                    if fail:
                        line += f"  recovery={fail.get('recovery')}  cause={fail.get('cause')}"
                    print(line)
            return 0

        if args.command == "inspect":
            print(json.dumps(wf.get_run_status(args.object_id), indent=2, sort_keys=True))
            return 0

        if args.command == "events":
            for e in wf.get_run_events(args.run_id):
                print(json.dumps(e, sort_keys=True))
            return 0

        if args.command == "validate":
            report = wf.validate_run(args.run_id)
            if args.json:
                print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            else:
                print(report.render())
            # 12 §7.2: any FAIL → training does not start (exit 1).
            return 1 if report.blocked else 0

        if args.command == "preflight":
            report = wf.preflight_run(args.run_id, gpu_required=not args.no_gpu)
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
