"""The Workflow API — one facade for CLI, TUI, and GUI.

Normative: 13_product_specification.md §1:

    USER WORKFLOW → MLForge command/UI → WORKFLOW ORCHESTRATOR
        → validation → planning → execution → artifact

Design rules enforced here:
  * No business logic per interface — everything the CLI/TUI/GUI need goes
    through this class (13 §1 rule 2).
  * Every state change = fire the normative state machine (mlforge.machine)
    → append an authoritative event (journal) → rewrite status.json as a
    *projection*. status.json is never authority (12 §16.1).
  * Automatic continuation is NO: this API never transitions anything on
    its own; each method call *is* the explicit event (user command, crash
    detection, gate result, worker report).
  * FAILED always records its disposition (`failure.recovery`) in the same
    event that enters the state (13 §5.3).

Storage layout (per object):

    <root>/<kind>/<id>/
        run_spec.json      # runs only — immutable, written once
        events.jsonl       # AUTHORITY (append-only, fsynced)
        status.json        # PROJECTION only — rebuildable, may be stale
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.errors import NotFound, ValidationBlock
from mlforge.hashing import content_hash
from mlforge.ids import new_model_id, new_run_id
from mlforge.journal import EventJournal
from mlforge.machine import (
    DATASET_MACHINE,
    MODEL_MACHINE,
    PROJECT_MACHINE,
    RUN_MACHINE,
    StateMachine,
)
from mlforge.run_spec import RunSpec
from mlforge.states import (
    DatasetState,
    FailureRecovery,
    ModelState,
    ProjectState,
    RunState,
    compute_recovery,
)
from mlforge.validation import (
    GateContext,
    Preflight,
    PreflightContext,
    ValidationGate,
    ValidationReport,
)

_KINDS = {
    "run": ("runs", RunState.CREATED.value, RUN_MACHINE),
    "project": ("projects", ProjectState.CREATED.value, PROJECT_MACHINE),
    "dataset": ("datasets", DatasetState.REGISTERED.value, DATASET_MACHINE),
    "model": ("models", ModelState.CREATED.value, MODEL_MACHINE),
}


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    state: str
    run_spec_hash: str


class WorkflowAPI:
    """Project-agnostic orchestrator facade rooted at a workspace directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    def _dir(self, kind: str, obj_id: str) -> Path:
        plural, _, _ = _KINDS[kind]
        return self.root / plural / obj_id

    def _journal(self, kind: str, obj_id: str) -> EventJournal:
        return EventJournal(self._dir(kind, obj_id) / "events.jsonl")

    def _status_path(self, kind: str, obj_id: str) -> Path:
        return self._dir(kind, obj_id) / "status.json"

    def _require(self, kind: str, obj_id: str) -> Path:
        d = self._dir(kind, obj_id)
        if not d.is_dir():
            raise NotFound(f"{kind} {obj_id!r} not found under {self.root}")
        return d

    def _load_projection(self, kind: str, obj_id: str) -> dict[str, Any]:
        p = self._status_path(kind, obj_id)
        if not p.exists():
            # Projection missing/corrupt → rebuild from authority (12 §16.1):
            # status.json is never the source of truth.
            return self._rebuild_projection(kind, obj_id)
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._rebuild_projection(kind, obj_id)

    def _rebuild_projection(self, kind: str, obj_id: str) -> dict[str, Any]:
        _, initial, _ = _KINDS[kind]
        j = self._journal(kind, obj_id)
        if not j.path.exists():
            raise NotFound(f"{kind} {obj_id!r} not found under {self.root}")
        state = j.project_state(initial)
        failure = None
        run_spec_hash = None
        for ev in j.read():
            if "failure" in ev.data:
                failure = ev.data["failure"]
            if "run_spec_hash" in ev.data:
                run_spec_hash = ev.data["run_spec_hash"]
        return self._write_projection(
            kind, obj_id, state, failure=failure, run_spec_hash=run_spec_hash
        )

    def _write_projection(
        self,
        kind: str,
        obj_id: str,
        state: str,
        *,
        failure: dict[str, Any] | None = None,
        run_spec_hash: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        proj = {
            "kind": kind,
            "id": obj_id,
            "state": state,
            "failure": failure,
            "run_spec_hash": run_spec_hash,
            "updated_ts": time.time(),
            **extra,
        }
        p = self._status_path(kind, obj_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(proj, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)  # atomic — readers never see a half-written projection
        return proj

    def _current_failure(self, kind: str, obj_id: str) -> dict[str, Any] | None:
        for ev in reversed(self._journal(kind, obj_id).read()):
            if "failure" in ev.data:
                return ev.data["failure"]
        return None

    def _fire(
        self,
        kind: str,
        obj_id: str,
        action: str,
        event: str,
        *,
        failure: dict[str, Any] | None = None,
        ctx_extra: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Single write path: machine → journal (authority) → projection."""
        _, initial, machine = _KINDS[kind]
        state = self._load_projection(kind, obj_id)["state"]
        ctx: dict[str, Any] = {
            f"{kind}_id": obj_id,
            "run_id": obj_id if kind == "run" else None,
            "failure_recovery": (failure or self._current_failure(kind, obj_id) or {}).get(
                "recovery"
            ),
            "failure_cause": (failure or self._current_failure(kind, obj_id) or {}).get("cause"),
        }
        if ctx_extra:
            ctx.update(ctx_extra)

        target = machine.fire(state, action, ctx)

        payload: dict[str, Any] = {"frm": state, "to": target, "action": action}
        if data:
            payload.update(data)
        if failure is not None:
            payload["failure"] = failure
        if kind == "run":
            spec_path = self._dir(kind, obj_id) / "run_spec.json"
            if spec_path.exists():
                payload.setdefault(
                    "run_spec_hash",
                    RunSpec.from_dict(json.loads(spec_path.read_text(encoding="utf-8"))).identity,
                )
        self._journal(kind, obj_id).append(event, **payload)
        self._write_projection(
            kind,
            obj_id,
            target,
            failure=failure if failure is not None else self._current_failure(kind, obj_id),
            run_spec_hash=payload.get("run_spec_hash"),
        )
        return target

    # ------------------------------------------------------------------
    # RUN lifecycle (13 §5.3 / §6)
    # ------------------------------------------------------------------

    def create_run(self, spec: RunSpec, run_id: str | None = None) -> RunHandle:
        """RUN flow: create immutable run_spec, journal `run_created` (CREATED).

        13 §6.1: CREATE RUN (immutable run_spec) — semantic identity frozen;
        the 19-step gate runs on resume/preflight, not at creation.
        """
        run_id = run_id or new_run_id()
        d = self._dir("run", run_id)
        if d.exists():
            raise ValidationBlock(f"run {run_id} already exists")
        d.mkdir(parents=True)
        spec.write_once(d / "run_spec.json")
        self._journal("run", run_id).append(
            "run_created",
            frm=None,
            to=RunState.CREATED.value,
            action="create",
            run_spec_hash=spec.identity,
        )
        self._write_projection("run", run_id, RunState.CREATED.value, run_spec_hash=spec.identity)
        return RunHandle(run_id, RunState.CREATED.value, spec.identity)

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        self._require("run", run_id)
        return self._load_projection("run", run_id)

    def get_run_state(self, run_id: str) -> str:
        return self.get_run_status(run_id)["state"]

    def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        self._require("run", run_id)
        return [ev.to_dict() for ev in self._journal("run", run_id).read()]

    # -- validation gate ------------------------------------------------

    def begin_validation(self, run_id: str) -> str:
        return self._fire("run", run_id, "begin_validation", "validation_started")

    def validation_pass(self, run_id: str, report: dict[str, Any] | None = None) -> str:
        """All checks passed → READY.

        The event explicitly records `failure: null` so a successful gate
        clears any stale FAILED disposition from the projection (the
        disposition's job — gating resume — is finished). Journal history
        keeps every prior failure event."""
        data: dict[str, Any] = {"failure": None}
        if report is not None:
            data["report"] = report
        return self._fire(
            "run", run_id, "validation_pass", "validation_passed", data=data
        )

    def validation_fail(
        self, run_id: str, cause: str, report: dict[str, Any] | None = None
    ) -> str:
        """Gate BLOCK while in VALIDATING → FAILED (13 §5.3: "gate BLOCK →
        FAILED[FORK_ONLY]" — the train path has no checkpoint yet, and the
        cause is semantic: an invariant could not be verified."""
        failure = self._make_failure(cause, has_valid_checkpoint=False, cause_is_semantic=True)
        data = {"report": report} if report is not None else None
        return self._fire(
            "run", run_id, "validation_fail", "validation_failed",
            failure=failure, data=data,
        )

    def validation_blocked(self, run_id: str, report: dict[str, Any]) -> str:
        """Gate BLOCK on the resume path — NO state change (13 §7 failure
        matrix: "resume, dataset hash mismatch → BLOCK. Stays PAUSED.
        Exit 1. No changes." / "resume, cause unresolved → BLOCK, stays
        FAILED").

        The journal records the attempt (visible in `mlforge events`);
        the event carries no `to`, so the projected state is untouched."""
        f = next(
            (c for c in report.get("checks", []) if c.get("verdict") == "FAIL"), None
        )
        self._journal("run", run_id).append(
            "validation_blocked",
            action="validate",
            failed_step=f.get("step") if f else None,
            failed_check=f.get("label") if f else None,
            report=report,
        )
        return self.get_run_state(run_id)

    def preflight_pass(self, run_id: str) -> str:
        return self._fire("run", run_id, "preflight_pass", "preflight_passed",
                          data={"failure": None})

    def preflight_fail(self, run_id: str, cause: str, report: dict[str, Any] | None = None) -> str:
        failure = self._make_failure(cause, has_valid_checkpoint=False, cause_is_semantic=False)
        data = {"report": report} if report is not None else None
        return self._fire(
            "run", run_id, "preflight_fail", "preflight_failed",
            failure=failure, data=data,
        )

    # -- execution ------------------------------------------------------

    def checkpoint_begin(self, run_id: str) -> str:
        return self._fire("run", run_id, "checkpoint", "checkpoint_started")

    def checkpoint_commit(self, run_id: str, ordinal: int) -> str:
        return self._fire(
            "run", run_id, "checkpoint_done", "checkpoint_committed", data={"ordinal": ordinal}
        )

    def checkpoint_fail(self, run_id: str, cause: str) -> str:
        failure = self._make_failure(cause, has_valid_checkpoint=True, cause_is_semantic=False)
        return self._fire("run", run_id, "checkpoint_failed", "checkpoint_failed", failure=failure)

    def complete(self, run_id: str) -> str:
        return self._fire("run", run_id, "complete", "completed")

    def runtime_error(
        self,
        run_id: str,
        cause: str,
        *,
        has_valid_checkpoint: bool = True,
        cause_is_semantic: bool = False,
    ) -> str:
        failure = self._make_failure(
            cause, has_valid_checkpoint=has_valid_checkpoint, cause_is_semantic=cause_is_semantic
        )
        return self._fire("run", run_id, "runtime_error", "runtime_error", failure=failure)

    # -- pause / stop (user decisions — 13 §1) --------------------------

    def pause(self, run_id: str) -> str:
        return self._fire(
            "run", run_id, "pause", "pause_requested", ctx_extra={"explicit": True}
        )

    def pause_committed(self, run_id: str, checkpoint: str) -> str:
        return self._fire(
            "run", run_id, "pause_committed", "pause_checkpoint_committed", data={"checkpoint": checkpoint}
        )

    def pause_checkpoint_failed(self, run_id: str, cause: str) -> str:
        failure = self._make_failure(cause, has_valid_checkpoint=True, cause_is_semantic=False)
        return self._fire(
            "run", run_id, "pause_checkpoint_failed", "pause_checkpoint_failed", failure=failure
        )

    def stop(self, run_id: str) -> str:
        return self._fire("run", run_id, "stop", "stop_requested", ctx_extra={"explicit": True})

    def stop_committed(self, run_id: str, checkpoint: str) -> str:
        return self._fire(
            "run", run_id, "stop_committed", "stop_committed", data={"checkpoint": checkpoint}
        )

    # -- crash / reconciliation (12 §12.3) ------------------------------

    def crash(self, run_id: str, reason: str) -> str:
        """Heartbeat expiry / power loss / kill -9 → INTERRUPTED.

        Never reported as PAUSED or RUNNING (13 §7)."""
        return self._fire(
            "run",
            run_id,
            "crash",
            "crash_detected",
            ctx_extra={"explicit": True},
            data={"reason": reason},
        )

    def begin_reconciliation(self, run_id: str) -> str:
        return self._fire("run", run_id, "begin_reconciliation", "reconciliation_started")

    def reconcile(
        self, run_id: str, *, has_valid_checkpoint: bool, resume_point: str | None = None
    ) -> dict[str, Any]:
        """Deterministic reconciliation scan (12 §12.3).

        INTERRUPTED → RECONCILING, record the scan result, and:
          * valid checkpoint  → wait in RECONCILING for an explicit `resume`
            (reconciliation never starts training)
          * no valid checkpoint → FAILED(recovery: FORK_ONLY)
        """
        state = self.get_run_state(run_id)
        if state == RunState.INTERRUPTED.value:
            self.begin_reconciliation(run_id)
        elif state != RunState.RECONCILING.value:
            raise ValidationBlock(
                f"reconciliation only applies to INTERRUPTED runs (run is {state})"
            )
        self._journal("run", run_id).append(
            "reconciliation_scan",
            has_valid_checkpoint=has_valid_checkpoint,
            resume_point=resume_point,
        )
        if not has_valid_checkpoint:
            failure = self._make_failure(
                "no valid checkpoint after crash",
                has_valid_checkpoint=False,
                cause_is_semantic=False,
            )
            self._fire(
                "run", run_id, "reconcile_no_checkpoint", "reconciliation_failed", failure=failure
            )
        return self.get_run_status(run_id)

    def resume(self, run_id: str, *, has_valid_checkpoint: bool | None = None) -> str:
        """Explicit resume (13 §6.2). Guards (machine):
        FAILED(FORK_ONLY) → exit 3 NO VALID CONTINUATION;
        RECONCILING without a valid checkpoint → same block;
        PAUSED → ordinary path. Lease checks arrive with step 5."""
        proj = self._load_projection("run", run_id)
        if has_valid_checkpoint is None:
            has_valid_checkpoint = self._last_reconciliation(run_id)
        return self._fire(
            "run",
            run_id,
            "resume",
            "resume_requested",
            ctx_extra={
                "has_valid_checkpoint": bool(has_valid_checkpoint),
                "failure_recovery": (proj.get("failure") or {}).get("recovery"),
                "failure_cause": (proj.get("failure") or {}).get("cause"),
            },
        )

    def _last_reconciliation(self, run_id: str) -> bool | None:
        for ev in reversed(self._journal("run", run_id).read()):
            if ev.event == "reconciliation_scan":
                return bool(ev.data.get("has_valid_checkpoint"))
        return None

    @staticmethod
    def _make_failure(
        cause: str, *, has_valid_checkpoint: bool, cause_is_semantic: bool
    ) -> dict[str, Any]:
        recovery = compute_recovery(
            has_valid_checkpoint=has_valid_checkpoint, cause_is_semantic=cause_is_semantic
        )
        return {
            "cause": cause,
            "recovery": recovery.value,
            "valid_checkpoint": has_valid_checkpoint,
        }

    # -- gate orchestration (12 §18, 13 §6.1–6.2) -----------------------

    def _run_spec(self, run_id: str) -> RunSpec:
        p = self._dir("run", run_id) / "run_spec.json"
        if not p.is_file():
            raise ValidationBlock(
                f"run {run_id}: run_spec.json missing — cannot validate",
                hint="the run folder is incomplete; recreate the run",
            )
        return RunSpec.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def validate_run(self, run_id: str, gate: ValidationGate | None = None) -> ValidationReport:
        """Run the 19-step validation gate (12 §18) for this run's state.

        State-dependent semantics (13 §5.3, §7):

        * CREATED / VALIDATING (train path): enter VALIDATING (if not
          already), run the gate; PASS → READY; FAIL → FAILED[FORK_ONLY]
          and the run never reaches READY.
        * PAUSED / RECONCILING / FAILED(resume: RESUME) (resume path):
          run the gate BEFORE any transition; FAIL → **no state change**
          (recorded `validation_blocked`, exit 1, "No changes were made");
          PASS → `resume` → VALIDATING → validation_pass → READY.
        * any other state (report-only): the gate runs and the report is
          returned; the run is never moved by `validate`.

        The report is always returned (caller decides the exit code);
        the run's state follows the rules above — never anything else.
        """
        gate = gate or ValidationGate()
        proj = self.get_run_status(run_id)
        state = proj["state"]
        spec = self._run_spec(run_id)
        flow = "TRAIN" if state in (
            RunState.CREATED.value, RunState.VALIDATING.value
        ) else "RESUME"
        report = gate.run(
            GateContext(run_id=run_id, root=self.root, run_spec=spec, flow=flow)
        )
        rep = report.to_dict()

        if state == RunState.CREATED.value:
            self.begin_validation(run_id)
            state = RunState.VALIDATING.value

        if state == RunState.VALIDATING.value:
            # Train path (incl. crash-recovery of a gate interrupted mid-run).
            if report.blocked:
                self.validation_blocked(run_id, rep)  # journal attempt (still VALIDATING)
                self.validation_fail(
                    run_id,
                    f"gate BLOCK at step {report.failed_step}: "
                    f"{report.first_failure.label}",
                    report=rep,
                )
            else:
                self.validation_pass(run_id, report=rep)
            return report

        resumable = state in (RunState.PAUSED.value, RunState.RECONCILING.value) or (
            state == RunState.FAILED.value
            and (proj.get("failure") or {}).get("recovery")
            == FailureRecovery.RESUME.value
        )
        if resumable:
            if report.blocked:
                self.validation_blocked(run_id, rep)  # NO state change (13 §7)
            else:
                # Gate passed → the resume transition itself is now legal.
                self.resume(run_id)
                self.validation_pass(run_id, report=rep)
            return report

        # Report-only states (RUNNING/READY/STOPPED/COMPLETED/INTERRUPTED/
        # FAILED[FORK_ONLY]): `validate` never moves them.
        if report.blocked:
            self.validation_blocked(run_id, rep)
        return report

    def preflight_run(
        self, run_id: str, preflight: Preflight | None = None, **ctx_kwargs: Any
    ) -> ValidationReport:
        """Host + identity preflight (12 §7.3) — REPORT ONLY.

        The READY → RUNNING transition stays with the runtime (build step
        6): firing `preflight_pass` here would put a run in RUNNING with
        no process behind it. Only after PREFLIGHT PASSED does train/
        resume launch (12 §7.3)."""
        self._require("run", run_id)
        spec = self._run_spec(run_id)
        ctx = PreflightContext(
            run_id=run_id, root=self.root, run_spec=spec, **ctx_kwargs
        )
        preflight = preflight or Preflight()
        return preflight.run(ctx)

    # ------------------------------------------------------------------
    # PROJECT (13 §5.1)
    # ------------------------------------------------------------------

    def create_project(self, name: str) -> str:
        d = self._dir("project", name)
        if d.exists():
            raise ValidationBlock(f"project {name!r} already exists")
        d.mkdir(parents=True)
        self._journal("project", name).append(
            "project_created", frm=None, to=ProjectState.CREATED.value, action="create"
        )
        self._write_projection("project", name, ProjectState.CREATED.value)
        return name

    def configure_project(self, name: str) -> str:
        return self._fire("project", name, "configure", "project_configured")

    def mark_project_ready(self, name: str) -> str:
        return self._fire("project", name, "mark_ready", "project_ready")

    # ------------------------------------------------------------------
    # DATASET (13 §5.2)
    # ------------------------------------------------------------------

    def register_dataset(self, dataset_id: str, identity: str) -> str:
        d = self._dir("dataset", dataset_id)
        if d.exists():
            raise ValidationBlock(f"dataset {dataset_id!r} already registered")
        d.mkdir(parents=True)
        (d / "identity.json").write_text(
            json.dumps({"dataset_id": dataset_id, "identity": identity}, indent=2),
            encoding="utf-8",
        )
        self._journal("dataset", dataset_id).append(
            "dataset_registered",
            frm=None,
            to=DatasetState.REGISTERED.value,
            action="create",
            identity=identity,
        )
        self._write_projection("dataset", dataset_id, DatasetState.REGISTERED.value)
        return dataset_id

    def verify_dataset(self, dataset_id: str, *, identity_matches: bool, found_identity: str) -> str:
        """Verify content hash (10.1: system verifies identity, never scans)."""
        if not identity_matches:
            return self.reject_dataset(
                dataset_id, f"content hash mismatch: found {found_identity}"
            )
        return self._fire("dataset", dataset_id, "verify", "dataset_verified")

    def reject_dataset(self, dataset_id: str, reason: str) -> str:
        return self._fire(
            "dataset", dataset_id, "reject", "dataset_rejected", data={"reason": reason}
        )

    def prepare_dataset(self, dataset_id: str) -> str:
        return self._fire("dataset", dataset_id, "prepare", "dataset_prepared")

    # ------------------------------------------------------------------
    # MODEL (13 §5.4)
    # ------------------------------------------------------------------

    def create_model(self, *, artifact_hash: str | None = None) -> str:
        model_id = new_model_id()
        d = self._dir("model", model_id)
        d.mkdir(parents=True)
        self._journal("model", model_id).append(
            "model_created",
            frm=None,
            to=ModelState.CREATED.value,
            action="create",
            artifact_hash=artifact_hash,
        )
        self._write_projection("model", model_id, ModelState.CREATED.value)
        return model_id

    def validate_model(self, model_id: str) -> str:
        return self._fire("model", model_id, "validate", "model_validated")

    def publish_model(self, model_id: str) -> str:
        return self._fire("model", model_id, "publish", "model_available")

    def record_model_event(self, model_id: str, kind: str) -> str:
        """kind ∈ evaluation | export | deployment | finetune_base (13 §5.4)."""
        action = f"record_{kind}"
        return self._fire("model", model_id, action, f"model_{kind}_recorded")

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def list_runs(self) -> list[dict[str, Any]]:
        runs_dir = self.root / "runs"
        if not runs_dir.is_dir():
            return []
        out = []
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir() and (d / "events.jsonl").exists():
                out.append(self.get_run_status(d.name))
        return out


__all__ = ["WorkflowAPI", "RunHandle", "content_hash"]
