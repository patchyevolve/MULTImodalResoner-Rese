"""`mlforge` command-line interface.

Normative contract: 13_product_specification.md §4.
This is the minimal step-4 slice: commands that the Workflow API already
supports. The rest exit 4 (runtime error: not implemented in this build
step) rather than pretending to work — fail loudly, never fake success.

Exit codes (§4.2): 0 ok · 1 validation BLOCK · 2 not found ·
3 precondition failed · 4 runtime error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlforge
from mlforge.errors import MlforgeError
from mlforge.store import ArtifactRegistry, ContentStore
from mlforge.validation import Preflight, ValidationGate
from mlforge.workflow import WorkflowAPI

# 13 §11 build-order gates: implemented vs pending.
_IMPLEMENTED = {"status", "inspect", "events", "store", "validate", "preflight"}
_PENDING = {
    "init": 1,
    "configure": 1,
    "prepare": 8,
    "train": 6,
    "resume": 6,
    "pause": 6,
    "stop": 6,
    "retrain": 10,
    "finetune": 10,
    "evaluate": 11,
    "compare": 11,
    "infer": 11,
    "export": 11,
    "package": 11,
    "lease": 5,
    "watch": 7,
    "hardware": 7,
}


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

    tr = sub.add_parser("train", help="(pending build step 6)")
    tr.add_argument("--config")
    tr.add_argument("--command-id", default=None)

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
    return p


def main(argv: list[str] | None = None) -> int:
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
        wf = WorkflowAPI(args.root)
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
