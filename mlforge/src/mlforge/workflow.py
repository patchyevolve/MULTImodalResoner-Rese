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
from typing import Any, Callable, Mapping

from mlforge.commands import CommandJournal
from mlforge.commands.idempotency import DedupDecision, execute as _execute_once
from mlforge.errors import (
    InvalidTransition,
    MlforgeError,
    NotFound,
    PreconditionFailed,
    RunAlreadyExecuting,
    ValidationBlock,
)
from mlforge.hashing import content_hash
from mlforge.ids import (
    new_bundle_id,
    new_eval_id,
    new_export_id,
    new_model_id,
    new_output_id,
    new_run_id,
)
from mlforge.journal import EventJournal
from mlforge.leases import LeaseState, RunLeaseManager
from mlforge.lineage import (
    FINETUNE_STRATEGIES,
    NewRunPlan,
    build_lineage,
    derive_spec,
    format_model_ref,
    new_plan,
    parse_model_ref,
    parse_overrides,
    read_lineage,
    spec_delta,
    write_lineage,
)
from mlforge.machine import (
    DATASET_MACHINE,
    MODEL_MACHINE,
    PROJECT_MACHINE,
    RUN_MACHINE,
    StateMachine,
)
from mlforge.ops import (
    FORMATS as EXPORT_FORMATS,
    build_bundle,
    build_evaluation,
    build_export,
    build_protocol,
    check_input,
    comparability_groups as _comparability_groups,
    component_integrity,
    contract_source_dir,
    execute_scaffold,
    latest_evaluation,
    latest_export as _latest_export,
    list_evaluations,
    list_exports as _list_exports,
    load_model_spec,
    model_entry_hash,
    numerical_validation,
    read_package,
    required_operators as _required_operators,
    scaffold_contract,
    source_tree_hash,
    validate_export,
    write_bundle,
    write_evaluation,
    write_export,
    write_output,
)
from mlforge.planner import (
    PLAN_FILENAME,
    Capabilities,
    ExecutionPlan,
    detect_capabilities,
    estimate_required_disk_bytes,
    load_runtime,
)
from mlforge.run_spec import RunSpec
from mlforge.states import (
    LIVE_STATES,
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
    provide_checkpoint,
    provide_dataset_identity,
    provide_driver,
    provide_global_batch,
    provide_hardware,
    provide_model,
    provide_plan,
    provide_precision,
    provide_topology,
    provide_transform,
)

_KINDS = {
    "run": ("runs", RunState.CREATED.value, RUN_MACHINE),
    "project": ("projects", ProjectState.CREATED.value, PROJECT_MACHINE),
    "dataset": ("datasets", DatasetState.REGISTERED.value, DATASET_MACHINE),
    "model": ("models", ModelState.CREATED.value, MODEL_MACHINE),
}


def _version_number(version: Any) -> int:
    """`v3` → 3; anything else → 0 (sorts unversioned/odd first)."""
    if isinstance(version, str) and version.startswith("v"):
        try:
            return int(version[1:])
        except ValueError:
            return 0
    return 0


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    state: str
    run_spec_hash: str


class WorkflowAPI:
    """Project-agnostic orchestrator facade rooted at a workspace directory.

    `gate_providers` wires the validation gate's step providers (12 §18).
    Unset ⇒ fail-closed default gate: any step without a built-in provider
    reports "unverifiable" ⇒ BLOCK — never a guess. Operators/tests wire
    the provider set their environment can honestly verify."""

    def __init__(
        self,
        root: str | Path,
        *,
        gate_providers: Mapping[str, Any] | None = None,
    ):
        self.root = Path(root)
        self.gate_providers = dict(gate_providers) if gate_providers else None

    def _providers(self) -> dict[str, Any]:
        """Provider set used when the caller supplies none.

        Step 4 (`dataset`) has a real builtin from build step 8: it
        re-hashes every configured source against its registration
        (12 §6.2). Steps 10–13 (`plan`, `global_batch`, `precision`,
        `topology`) come from build step 9: measured capabilities +
        the feasibility solver (12 §13). Steps 3/7 (`source_code`,
        `environment`) verify the captures written at creation
        (12 §6.4); step 5 (`transform`) verifies prepared artifacts;
        step 6 (`model`) verifies base-weights identity for finetune;
        steps 8/9/14 (`driver`, `hardware`, `checkpoint`) verify against
        live detection and the checkpoint store; steps 15/16 (`lease`,
        `revalidate`) close the TOCTOU window (12 §18, §23.3) — the
        launch path reuses the session token the gate acquired.
        Explicit providers always win (`setdefault`) — callers that can
        verify more honestly keep their wiring."""
        from mlforge.capture import provide_environment, provide_source_code
        from mlforge.leases import RunLeaseManager, provide_revalidation, provide_run_lease

        dataset_provider = provide_dataset_identity(self.root)
        providers = dict(self.gate_providers or {})
        providers.setdefault("source_code", provide_source_code(self.root))
        providers.setdefault("dataset", dataset_provider)
        providers.setdefault("transform", provide_transform(self.root))
        providers.setdefault("model", provide_model(self.root))
        providers.setdefault("environment", provide_environment(self.root))
        providers.setdefault("plan", provide_plan())
        providers.setdefault("global_batch", provide_global_batch())
        providers.setdefault("precision", provide_precision())
        providers.setdefault("topology", provide_topology())
        providers.setdefault("driver", provide_driver())
        providers.setdefault("hardware", provide_hardware())
        providers.setdefault("checkpoint", provide_checkpoint())
        leases = RunLeaseManager(self.root)
        providers.setdefault("lease", provide_run_lease(leases))
        providers.setdefault(
            "revalidate",
            provide_revalidation(leases, dataset_provider=dataset_provider),
        )
        return providers

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

    def create_run(
        self,
        spec: RunSpec,
        run_id: str | None = None,
        *,
        lineage: Mapping[str, Any] | None = None,
    ) -> RunHandle:
        """RUN flow: create immutable run_spec + lineage.json, journal
        `run_created` (CREATED).

        13 §6.1: CREATE RUN (immutable run_spec) — semantic identity frozen;
        the 19-step gate runs on resume/preflight, not at creation.
        12 §5/§15.2: every run folder carries its lineage edges (train =
        parent nulls; fork/retrain/finetune = parent set), written once.
        """
        run_id = run_id or new_run_id()
        d = self._dir("run", run_id)
        if d.exists():
            raise ValidationBlock(f"run {run_id} already exists")
        d.mkdir(parents=True)
        spec.write_once(d / "run_spec.json")
        lin = dict(lineage) if lineage else None
        write_lineage(d, lin or {})
        # 12 §6.4: every run records source + environment identity at
        # creation — the snapshot is the recovery authority if the
        # working tree later disappears. Fail-closed: a run that cannot
        # record its identity must not exist.
        from mlforge.capture import capture_run_identity
        try:
            capture_run_identity(self.root, d)
        except ValidationBlock:
            raise
        except Exception as exc:
            raise ValidationBlock(
                f"run {run_id}: identity capture failed: {exc}",
                hint="runs must record source + environment identity "
                     "(12 §6.4) — resolve the capture error first",
            ) from exc
        self._journal("run", run_id).append(
            "run_created",
            frm=None,
            to=RunState.CREATED.value,
            action="create",
            run_spec_hash=spec.identity,
            origin=(lin or {}).get("origin", "train"),
            parent_run=(lin or {}).get("parent_run"),
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
        """RUNNING → COMPLETED, then the run produces its model (13 line
        50 "A run produces a model", §5.5 TRAIN → RUN A → MODEL A): the
        registry entry is published CREATED → VALIDATED → AVAILABLE with
        provenance (name:vN, run, artifact). Recording an artifact is not
        a lifecycle continuation — nothing about the RUN auto-transitions
        beyond the explicit completion itself."""
        state = self._fire("run", run_id, "complete", "completed")
        self.publish_model_from_run(run_id)
        return state

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
                f"reconciliation only applies to INTERRUPTED runs (run is {state})",
                hint="if the worker is dead, heartbeat expiry (supervisor) "
                     "marks the run INTERRUPTED first — 12 §12.3",
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

    def resume(
        self,
        run_id: str,
        *,
        has_valid_checkpoint: bool | None = None,
        session_token: str | None = None,
    ) -> str:
        """Explicit resume (13 §6.2). Guards (machine):
        FAILED(FORK_ONLY) → exit 3 NO VALID CONTINUATION;
        RECONCILING without a valid checkpoint → same block;
        PAUSED → ordinary path.

        Lease guard (12 §23.2 hard invariant, 13 §7): a second resume of a
        leased run → BLOCK RUN_ALREADY_EXECUTING (exit 3); a SUSPECT
        (stale) lease → block until broken explicitly — stale ≠ free.
        `session_token` proves THIS caller owns the lease (the validation
        gate's step 15 acquired it).

        INTERRUPTED runs are reconciled from disk FIRST (13 §6.2,
        12 §12.3): derive resume_point + skipped checkpoints, journal them,
        then the normal flow continues (RECONCILING → resume). Idempotent
        — a prior full reconciliation is replayed from its RECONCILED
        event, never re-run."""
        state = self.get_run_state(run_id)
        if state == RunState.INTERRUPTED.value:
            # Step 7 of §12.3: reconcile before any resume promise. This
            # moves the run to RECONCILING (valid checkpoint) or
            # FAILED[FORK_ONLY] (none) — the resume guard below decides.
            self.reconcile_from_disk(run_id)
        lease = RunLeaseManager(self.root)
        info = lease.status(run_id)
        if info.state == LeaseState.HELD and info.session_token != session_token:
            raise RunAlreadyExecuting(run_id, holder=info.holder)
        if info.state == LeaseState.SUSPECT:
            raise PreconditionFailed(
                f"run {run_id}: lease is SUSPECT (held by {info.holder}, "
                f"heartbeat {info.age_seconds:.0f}s old)",
                hint="stale ≠ free: confirm the worker is dead, then "
                     "`mlforge lease break RUN --force --yes` (12 §23.2)",
            )
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
            if ev.event in ("reconciliation_scan", "RECONCILED"):
                if "has_valid_checkpoint" in ev.data:
                    return bool(ev.data.get("has_valid_checkpoint"))
                if ev.event == "RECONCILED":
                    # RECONCILED after a full scan: valid ⇔ a resume point
                    return ev.data.get("resume_point") is not None
        return None

    # -- full §12.3 reconciliation (deterministic, idempotent) -------------

    def reconcile_from_disk(self, run_id: str) -> dict[str, Any]:
        """Deterministic crash recovery (12 §12.3) — derive everything from
        the run folder on disk:

            1–3. checkpoint scan → §11.2 newest-valid predicate → resume_point
            4–5. projection vs journal authority → consistent | stale
            6.   CHECKPOINT_SKIPPED events + RECONCILED event
            7.   report (never starts training — `resume` stays explicit)

        Idempotent: if the journal's last event is already RECONCILED,
        nothing is appended and the stored report is returned. Stale
        projections are rebuilt from the journal (authority wins, §16).
        """
        from mlforge.runtime.reconcile import ReconcileReport, scan_checkpoints

        j = self._journal("run", run_id)
        events = j.read()  # raises NotFound via _require below if missing
        if events and events[-1].event == "RECONCILED":
            data = dict(events[-1].data)
            data.setdefault("stored", True)
            data["skips"] = data.get("skipped", [])
            return data

        self._require("run", run_id)
        prior = self.get_run_state(run_id)
        initial = _KINDS["run"][1]
        authority = j.project_state(initial)
        consistent = prior == authority

        if consistent:
            derived = authority
        else:
            # §12.3 step 5: inconsistent → INTERRUPTED (when the authority
            # is a live state — a live-state divergence means the recorded
            # picture is broken); otherwise the journal's truth wins.
            # Rebuild the projection FIRST so the machine fires from the
            # authority state, not the stale one (13 §9.4).
            failure = self._current_failure("run", run_id)
            self._write_projection(
                "run", run_id, authority,
                failure=failure,
                run_spec_hash=self.get_run_status(run_id).get("run_spec_hash"),
            )
            if authority in {s.value for s in LIVE_STATES}:
                # Record the divergence as an explicit crash — the journal
                # must stay the single authority for state.
                self.crash(
                    run_id,
                    "reconciliation: recorded state stale vs journal authority",
                )
                derived = RunState.INTERRUPTED.value
            else:
                derived = authority

        selection = scan_checkpoints(self._dir("run", run_id))
        reconciliation_id = content_hash(
            {"run": run_id, "authority_state": authority}
        )

        # §11.2 rule 4: every skip is an event, written BEFORE resume.
        for s in selection.skips:
            j.append(
                "CHECKPOINT_SKIPPED",
                action="reconcile",
                checkpoint=f"ckpt-{s.ordinal:06d}",
                ordinal=s.ordinal,
                reason=s.reason,
            )

        # Reconcile transitions (INTERRUPTED → RECONCILING, or → FAILED
        # when no valid checkpoint exists) — via the normative machine.
        self.reconcile(
            run_id,
            has_valid_checkpoint=not selection.no_valid_checkpoint,
            resume_point=selection.resume_point,
        )

        state_after = self.get_run_state(run_id)
        j.append(
            "RECONCILED",
            action="reconcile",
            reconciliation_id=reconciliation_id,
            prior_recorded_state=prior,
            authority_state=authority,
            derived_state=derived,
            consistent=consistent,
            resume_point=selection.resume_point,
            skipped=[s.to_dict() for s in selection.skips],
            has_valid_checkpoint=not selection.no_valid_checkpoint,
            state_after=state_after,
        )
        report = ReconcileReport(
            run_id=run_id,
            prior_recorded_state=prior,
            authority_state=authority,
            derived_state=state_after,
            selection=selection,
            consistent=consistent,
            reconciliation_id=reconciliation_id,
        )
        out = report.to_dict()
        out["state_after"] = state_after
        return out

    # -- run lease (12 §23, 13 §4.1 LEASES) ------------------------------

    def lease_status(self, run_id: str) -> dict[str, Any]:
        """`mlforge lease status RUN` — who holds the lease (read-only)."""
        return RunLeaseManager(self.root).status(run_id).to_dict()

    def lease_break(
        self,
        run_id: str,
        *,
        force: bool = False,
        yes: bool = False,
        reason: str = "unspecified",
        operator: str | None = None,
    ) -> dict[str, Any]:
        """Break a lease — ALWAYS requires --force AND --yes and is ALWAYS
        logged as LEASE_BROKEN (12 §23.2; 12 §24: no silent escalation)."""
        if not force or not yes:
            raise PreconditionFailed(
                "`mlforge lease break` requires --force and --yes",
                hint="breaking a lease overrides single-writer safety — "
                     "no silent escalation (12 §24)",
            )
        if operator is None:
            import getpass

            operator = getpass.getuser()
        mgr = RunLeaseManager(self.root)
        previous = mgr.force_release(run_id)  # raises exit 3 if FREE
        self._journal("run", run_id).append(
            "LEASE_BROKEN",
            action="lease_break",
            previous_owner={
                "pid": previous.get("pid"),
                "host": previous.get("host"),
                "session_token": previous.get("session_token"),
            },
            reason=reason,
            operator=operator,
        )
        return {"previous_owner": previous, "reason": reason, "operator": operator}

    # -- command idempotency (13 §4.4, 12 §23.4) -------------------------

    def execute_idempotent(
        self,
        command_id: str,
        command: str,
        fn: Callable[[], Any],
        *,
        run_id: Any = None,
        kind: str = "run",
        meta: dict[str, Any] | None = None,
    ) -> Any:
        """Run `fn` exactly once per successful command_id.

        Duplicate success → returns the ORIGINAL result and journals
        COMMAND_DEDUPED into the object's events (13 §4.4). Failure → the
        same command_id may be retried. In flight → exit 3.

        `run_id` may be a callable — for `train` the run id only exists
        INSIDE the deduplicated result (the event goes to the ORIGINAL
        run, never to a second run). `kind` selects whose journal the
        dedupe lands in (`run` | `dataset` | ...)."""

        def _on_dedup(d: DedupDecision) -> None:
            rid = run_id(d.result) if callable(run_id) else run_id
            if rid and (self.root / _KINDS[kind][0] / rid).is_dir():
                # Visible in `mlforge events` as COMMAND_DEDUPED (13 §4.4).
                # Carries no `to` → never changes object state.
                self._journal(kind, rid).append(
                    "COMMAND_DEDUPED",
                    action="dedupe",
                    command_id=d.command_id,
                    command=d.command,
                    original_ts=d.original_ts,
                )

        return _execute_once(
            CommandJournal(self.root),
            command_id,
            command,
            fn,
            meta=meta,
            on_dedup=_on_dedup,
        )

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
        if gate is None:
            gate = ValidationGate(self._providers())
        proj = self.get_run_status(run_id)
        state = proj["state"]
        spec = self._run_spec(run_id)
        flow = "TRAIN" if state in (
            RunState.CREATED.value, RunState.VALIDATING.value
        ) else "RESUME"
        ctx = GateContext(run_id=run_id, root=self.root, run_spec=spec, flow=flow)
        # Step 16 (revalidate) needs the worst-case disk requirement — the
        # 12 §7.3 formula, now extended by the planner with registered
        # dataset bytes (13 §11 step 9, `estimate_required_disk_bytes`).
        ctx.facts["required_disk_bytes"] = estimate_required_disk_bytes(
            self.root, spec, runtime=load_runtime(ctx.run_dir)
        )
        report = gate.run(ctx)
        rep = report.to_dict()
        plan = ctx.facts.get("execution_plan")
        if plan is not None and not report.blocked:
            # 13 §6.1: the plan artifact lands with the validated run;
            # unchanged plans never journal twice (identity-gated).
            self._persist_execution_plan(run_id, plan)

        if state == RunState.CREATED.value:
            self.begin_validation(run_id)
            state = RunState.VALIDATING.value

        if state == RunState.VALIDATING.value:
            # Train path (incl. crash-recovery of a gate interrupted mid-run).
            if report.blocked:
                self._release_gate_lease(ctx)  # "no changes" includes the lease
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
                self._release_gate_lease(ctx)  # step 15 may have run before the FAIL
                self.validation_blocked(run_id, rep)  # NO state change (13 §7)
            else:
                # Gate passed → the resume transition itself is now legal.
                # If the gate acquired the lease (step 15), present its
                # session token so the lease guard knows we own it.
                self.resume(
                    run_id, session_token=ctx.facts.get("session_token")
                )
                self.validation_pass(run_id, report=rep)
            return report

        # Report-only states (RUNNING/READY/STOPPED/COMPLETED/INTERRUPTED/
        # FAILED[FORK_ONLY]): `validate` never moves them.
        if report.blocked:
            self.validation_blocked(run_id, rep)
        return report

    def _release_gate_lease(self, ctx: GateContext) -> None:
        """A gate BLOCK must not strand the lease step 15 acquired —
        "No changes were made" includes single-writer ownership."""
        token = ctx.facts.get("session_token")
        if not token:
            return
        try:
            RunLeaseManager(self.root).release(ctx.run_id, str(token))
        except MlforgeError:
            pass  # never owned / already gone

    # ------------------------------------------------------------------
    # execution plan + segments (build step 9, 12 §12.2, §13)
    # ------------------------------------------------------------------

    def _persist_execution_plan(self, run_id: str, plan: Any) -> None:
        """Write `runs/<id>/execution_plan.json` (atomic) — only when the
        identity changed, so a re-validate on the same host is quiet and
        a resume on new hardware rewrites with one `plan_generated`."""
        path = self._dir("run", run_id) / PLAN_FILENAME
        if path.is_file():
            try:
                if ExecutionPlan.read(path).identity == plan.identity:
                    return  # identical plan already on disk
            except MlforgeError:
                pass  # unreadable/corrupt → rewrite (repair, fail-closed)
            except (KeyError, ValueError, TypeError):
                pass
        plan.write(path)
        self._journal("run", run_id).append(
            "plan_generated",
            action="plan",
            plan_hash=plan.identity,
            micro_batch=plan.micro_batch,
            grad_accum=plan.grad_accum,
            world_size=plan.world_size,
            precision=plan.precision_effective,
            portable=plan.portable_required,
        )

    def get_execution_plan(self, run_id: str) -> dict[str, Any] | None:
        """The persisted plan artifact, or None if never validated."""
        path = self._dir("run", run_id) / PLAN_FILENAME
        if not path.is_file():
            return None
        return ExecutionPlan.read(path).to_dict()

    def create_execution_segment(
        self,
        run_id: str,
        *,
        capabilities: Capabilities | None = None,
        plan: Any = None,
    ) -> int:
        """Create `runs/<id>/segments/segment_<NNNN>.json` (12 §12.2).

        Called by the worker AFTER `preflight_pass` succeeds: a failed
        preflight must leave the run in READY with no segment. Resume on
        new hardware appends a new ordinal — every execution context is
        preserved, never overwritten.
        """
        self._require("run", run_id)
        if capabilities is None:
            capabilities = detect_capabilities()
        if plan is None:
            plan_path = self._dir("run", run_id) / PLAN_FILENAME
            if plan_path.is_file():
                plan = ExecutionPlan.read(plan_path)
        seg_dir = self._dir("run", run_id) / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(seg_dir.glob("segment_*.json"))
        ordinal = len(existing) + 1
        payload = {
            "schema_version": 1,
            "ordinal": ordinal,
            "created_ts": time.time(),
            "capabilities": capabilities.to_dict(),
            "capabilities_identity": capabilities.identity,
            "plan_hash": plan.identity if plan is not None else None,
            "micro_batch": plan.micro_batch if plan is not None else None,
            "grad_accum": plan.grad_accum if plan is not None else None,
            "world_size": plan.world_size if plan is not None else None,
            "precision_effective": (
                plan.precision_effective if plan is not None else None
            ),
        }
        path = seg_dir / f"segment_{ordinal:04d}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                        encoding="utf-8")
        self._journal("run", run_id).append(
            "segment_created",
            action="create_segment",
            ordinal=ordinal,
            capabilities_identity=capabilities.identity,
        )
        return ordinal

    def preflight_run(
        self, run_id: str, preflight: Preflight | None = None, **ctx_kwargs: Any
    ) -> ValidationReport:
        """Host + identity preflight (12 §7.3) — REPORT ONLY.

        The READY → RUNNING transition stays with the runtime (build step
        6): firing `preflight_pass` here would put a run in RUNNING with
        no process behind it. Only after PREFLIGHT PASSED does train/
        resume launch (12 §7.3).

        Identity re-verification uses the SAME provider wiring as the gate
        (one vocabulary, two invocations); unwired ⇒ fail-closed FAIL."""
        self._require("run", run_id)
        spec = self._run_spec(run_id)
        # Worst-case disk inputs: the SAME runtime-aware components the
        # gate's step 16 uses (12 §7.3) — one formula, two invocations.
        from mlforge.planner import disk_estimate_components

        for key, value in disk_estimate_components(
            self.root, spec, runtime=load_runtime(self._dir("run", run_id))
        ).items():
            ctx_kwargs.setdefault(key, value)
        ctx = PreflightContext(
            run_id=run_id, root=self.root, run_spec=spec, **ctx_kwargs
        )
        preflight = preflight or Preflight(identity_providers=self._providers())
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

    def register_dataset(
        self,
        dataset_id: str,
        identity: str,
        *,
        version: str = "v1",
        schema: str | None = None,
        file_count: int | None = None,
        total_bytes: int | None = None,
    ) -> str:
        d = self._dir("dataset", dataset_id)
        if d.exists():
            raise ValidationBlock(f"dataset {dataset_id!r} already registered")
        d.mkdir(parents=True)
        self._write_identity(
            d, dataset_id, identity, version=version, schema=schema,
            file_count=file_count, total_bytes=total_bytes,
        )
        self._journal("dataset", dataset_id).append(
            "dataset_registered",
            frm=None,
            to=DatasetState.REGISTERED.value,
            action="create",
            identity=identity,
            version=version,
        )
        self._write_projection("dataset", dataset_id, DatasetState.REGISTERED.value)
        return dataset_id

    def reregister_dataset(
        self,
        dataset_id: str,
        identity: str,
        *,
        version: str = "v1",
        schema: str | None = None,
        file_count: int | None = None,
        total_bytes: int | None = None,
    ) -> str:
        """Explicit re-registration after content change / rejection.

        13 §5.2 + §4.1: never silently reinterpret an existing
        registration — the operator (or prepare for derived data) states
        the new identity and the journal records who/what changed."""
        d = self._require("dataset", dataset_id)
        state = self._load_projection("dataset", dataset_id)["state"]
        self._write_identity(
            d, dataset_id, identity, version=version, schema=schema,
            file_count=file_count, total_bytes=total_bytes,
        )
        self._journal("dataset", dataset_id).append(
            "dataset_reregistered",
            frm=state,
            to=DatasetState.REGISTERED.value,
            action="reregister",
            identity=identity,
            version=version,
        )
        self._write_projection("dataset", dataset_id, DatasetState.REGISTERED.value)
        return dataset_id

    @staticmethod
    def _write_identity(
        d: Path,
        dataset_id: str,
        identity: str,
        *,
        version: str,
        schema: str | None,
        file_count: int | None,
        total_bytes: int | None,
    ) -> None:
        payload: dict[str, Any] = {
            "dataset_id": dataset_id,
            "identity": identity,
            "version": version,
        }
        if schema is not None:
            payload["schema"] = schema
        if file_count is not None:
            payload["file_count"] = file_count
        if total_bytes is not None:
            payload["total_bytes"] = total_bytes
        p = d / "identity.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)  # atomic — readers never see a half-written identity

    def get_dataset_status(self, dataset_id: str) -> dict[str, Any]:
        self._require("dataset", dataset_id)
        return self._load_projection("dataset", dataset_id)

    def note_dataset_path(self, dataset_id: str, path: str | Path) -> str:
        """Audit the machine-local path config for a dataset.

        Journal-only: NO state transition (12 §6.3 — paths are
        machine-local and never part of identity)."""
        state = self.get_dataset_status(dataset_id)["state"]
        self._journal("dataset", dataset_id).append(
            "dataset_path_configured",
            frm=state,
            to=state,
            action="set_path",
            path=str(path),
        )
        return dataset_id

    def list_datasets(self) -> list[dict[str, Any]]:
        ds = self.root / "datasets"
        if not ds.is_dir():
            return []
        return [
            self.get_dataset_status(d.name)
            for d in sorted(ds.iterdir())
            if d.is_dir()
        ]

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

    def create_model(
        self,
        *,
        artifact_hash: str | None = None,
        name: str | None = None,
        version: str | None = None,
        run_id: str | None = None,
        run_spec_hash: str | None = None,
        origin: str | None = None,
    ) -> str:
        """Registry entry: the model.json sidecar carries the addressable
        `name:vN` identity (13 §4.3) + provenance (13 §5.5: every model
        traces to the run that produced it — or `import` for external
        packages, 13 §4.1)."""
        model_id = new_model_id()
        d = self._dir("model", model_id)
        d.mkdir(parents=True)
        sidecar = {
            "schema_version": 1,
            "model_id": model_id,
            "name": name,
            "version": version,
            "artifact_hash": artifact_hash,
            "run_id": run_id,
            "run_spec_hash": run_spec_hash,
            "origin": origin or "run",
            "created_ts": time.time(),
        }
        (d / "model.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._journal("model", model_id).append(
            "model_created",
            frm=None,
            to=ModelState.CREATED.value,
            action="create",
            artifact_hash=artifact_hash,
            name=name,
            version=version,
            run_id=run_id,
            origin=sidecar["origin"],
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
    # lineage DAG + new-run flows (build step 10; 12 §5, §15; 13 §5.5,
    # §6.3/§6.4)
    # ------------------------------------------------------------------

    def get_lineage(self, run_id: str) -> dict[str, Any]:
        self._require("run", run_id)
        return read_lineage(self._dir("run", run_id))

    def lineage_chain(self, run_id: str) -> list[dict[str, Any]]:
        """Ancestors root-first, SELF included (the full path a model
        traces back through — 12 invariant D). Cycle guard: corrupt edges
        BLOCK rather than loop forever."""
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        current: str | None = run_id
        while current is not None:
            if current in seen:
                raise ValidationBlock(
                    f"lineage cycle detected at {current} "
                    f"({' → '.join(sorted(seen))})",
                    hint="lineage edges are immutable — repair requires "
                         "forking from a healthy run (12 §15.2)",
                )
            seen.add(current)
            lin = self.get_lineage(current)
            chain.append({"run_id": current, **lin})
            parent = lin.get("parent_run")
            if parent is not None and not (self._dir("run", parent)).is_dir():
                raise ValidationBlock(
                    f"lineage parent {parent} of {current} is missing "
                    "from this workspace",
                    hint="the workspace is incomplete — restore the parent "
                         "run or fork from a healthy run",
                )
            current = parent
        chain.reverse()  # root first
        return chain

    def retrain_source(self, model: str) -> tuple[str, RunSpec]:
        """Most recent run of `model` — retrain's config source (13 §6.3
        "load previous run_spec"). Newest by run id (ULID time-sortable)."""
        runs_dir = self.root / "runs"
        if runs_dir.is_dir():
            names = sorted(
                (p.name for p in runs_dir.iterdir() if p.is_dir()), reverse=True
            )
            for name in names:
                if not (runs_dir / name / "run_spec.json").is_file():
                    continue
                spec = self._run_spec(name)
                if spec.model == model:
                    return name, spec
        raise NotFound(
            f"no previous run found for model {model!r}",
            hint="retrain continues a previous configuration (13 §6.3) — "
                 "train one first: `mlforge train --config ...`",
        )

    @staticmethod
    def _overrides(overrides: Any) -> dict[str, Any]:
        if overrides is None:
            return {}
        if isinstance(overrides, Mapping):
            return dict(overrides)
        return parse_overrides(overrides)  # iterable of "KEY=VALUE"

    def preview_fork(
        self,
        run_id: str,
        *,
        overrides: Any = None,
        dataset: str | None = None,
        model: str | None = None,
    ) -> NewRunPlan:
        """Semantic change → a NEW run with parent set (12 §4 hard rule:
        `mlforge fork`, never `--reconfigure`). The parent is read-only."""
        self._require("run", run_id)  # absent → NotFound (exit 2), not a
        # "folder incomplete" block — distinguish never-existed from broken.
        parent = self._run_spec(run_id)
        new = derive_spec(
            parent,
            semantic=self._overrides(overrides),
            train_datasets=(dataset,) if dataset else None,
            model=model,
        )
        if new.identity == parent.identity:
            raise ValidationBlock(
                f"fork of {run_id} would not change the experiment identity",
                hint="nothing to fork: `resume` continues this run, "
                     "`retrain` repeats it with fresh init (13 §5.5)",
            )
        lineage = build_lineage(
            "fork", parent_run=run_id, parent_run_spec_hash=parent.identity
        )
        return new_plan(
            "fork", new, lineage,
            delta=spec_delta(parent, new), source_run_id=run_id,
        )

    def preview_retrain(
        self,
        model: str,
        *,
        overrides: Any = None,
        dataset: str | None = None,
    ) -> NewRunPlan:
        """13 §6.3: same config unless changed, fresh init, parent set."""
        src_id, src = self.retrain_source(model)
        new = derive_spec(
            src,
            semantic=self._overrides(overrides),
            train_datasets=(dataset,) if dataset else None,
        )
        lineage = build_lineage(
            "retrain", parent_run=src_id, parent_run_spec_hash=src.identity
        )
        return new_plan(
            "retrain", new, lineage,
            delta=spec_delta(src, new),
            source_run_id=src_id,
            starting_state="pretrained base weights (fresh init)",
        )

    def resolve_model(self, ref: str) -> dict[str, Any]:
        """`model://rf_detr_s:v2` | `rf_detr_s:v2` | `rf_detr_s` (latest)
        → registry entry (13 §4.3 reference syntax)."""
        name, version = parse_model_ref(ref)
        candidates = [m for m in self.list_models() if m.get("name") == name]
        if version is None:
            if not candidates:
                raise NotFound(
                    f"model {name!r} not found",
                    hint="models are published when a run COMPLETES "
                         "(13 §5.5); `mlforge model list` shows what exists",
                )
            return max(
                candidates, key=lambda m: _version_number(m.get("version"))
            )
        for m in candidates:
            if m.get("version") == version:
                return m
        available = ", ".join(
            format_model_ref(str(m["name"]), str(m["version"]))
            for m in candidates
        )
        raise NotFound(
            f"model {format_model_ref(name, version)} not found",
            hint=f"available versions: {available}" if available
                 else f"model {name!r} not found — `mlforge model list`",
        )

    def preview_finetune(
        self,
        ref: str,
        *,
        strategy: str | None = None,
        overrides: Any = None,
        dataset: str | None = None,
    ) -> NewRunPlan:
        """13 §6.4: init from weights of `model://name:vN` → NEW run +
        NEW model; the base model is never modified."""
        base_entry = self.resolve_model(ref)
        ov = self._overrides(overrides)
        effective = strategy if strategy is not None else ov.get(
            "finetune_strategy", "full"
        )
        if effective not in FINETUNE_STRATEGIES:
            raise ValidationBlock(
                f"fine-tuning strategy {effective!r} is not supported",
                hint=f"allowed: {', '.join(FINETUNE_STRATEGIES)} (13 §6.4)",
            )
        ov["finetune_strategy"] = effective
        base_run = base_entry.get("run_id")
        if not base_run:
            raise NotFound(
                f"model {format_model_ref(str(base_entry['name']), str(base_entry['version']))} "
                "has no provenance run",
                hint="fine-tune sources are produced models (13 §6.4); "
                     "imported externals arrive with `model import` (step 11)",
            )
        base = self._run_spec(base_run)
        new = derive_spec(
            base,
            semantic=ov,
            train_datasets=(dataset,) if dataset else None,
            model=str(base_entry["name"]),
        )
        lineage = build_lineage(
            "finetune",
            parent_run=base_run,
            parent_run_spec_hash=base.identity,
            parent_model=format_model_ref(
                str(base_entry["name"]), str(base_entry["version"])
            ),
            parent_checkpoint=base_entry.get("artifact_hash"),
        )
        return new_plan(
            "finetune", new, lineage,
            delta=spec_delta(base, new),
            source_run_id=base_run,
            base_model=format_model_ref(
                str(base_entry["name"]), str(base_entry["version"])
            ),
            base_model_id=str(base_entry["model_id"]),
            warnings=(
                "base model NOT modified — this creates a NEW run + NEW model",
            ),
        )

    def create_from_plan(self, plan: NewRunPlan) -> RunHandle:
        """Materialize a previewed plan exactly once (SHOW DELTA →
        confirm → CREATE — 13 §6.3/§6.4)."""
        handle = self.create_run(
            plan.spec, plan.run_id, lineage=plan.lineage
        )
        if plan.kind == "finetune" and plan.base_model_id:
            try:
                self.record_model_event(plan.base_model_id, "finetune_base")
            except InvalidTransition:
                # already USED_AS_FINE_TUNE_BASE — the lineage edge in the
                # child is the record; the state just is not re-enterable.
                pass
        return handle

    # -- model registry (13 §5.4/§5.5, §4.3 references) -----------------

    def list_models(self) -> list[dict[str, Any]]:
        mdir = self.root / "models"
        if not mdir.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for d in sorted(mdir.iterdir()):
            p = d / "model.json"
            if not p.is_file():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValidationBlock(f"model sidecar unreadable: {p}: {exc}")
            if not isinstance(data, dict):
                raise ValidationBlock(f"model sidecar must be an object: {p}")
            data = dict(data)
            data["state"] = self._load_projection(
                "model", d.name
            ).get("state")
            out.append(data)
        out.sort(
            key=lambda m: (
                str(m.get("name") or ""), _version_number(m.get("version")),
                str(m.get("model_id") or ""),
            )
        )
        return out

    def publish_model_from_run(self, run_id: str) -> str:
        """The run produces its model on completion (13 §5.5): version
        bumps per name (`rf_detr_s:v1`, `:v2`, ...), artifact = newest
        COMMITTED checkpoint's manifest hash (or null — weights alone are
        not the model, 12 §15.3)."""
        self._require("run", run_id)
        spec = self._run_spec(run_id)
        name = spec.model
        version = f"v{self._next_version(name)}"
        model_id = self.create_model(
            artifact_hash=self._newest_checkpoint_hash(run_id),
            name=name,
            version=version,
            run_id=run_id,
            run_spec_hash=spec.identity,
        )
        self.validate_model(model_id)
        self.publish_model(model_id)
        return model_id

    def _next_version(self, name: str) -> int:
        versions = [
            _version_number(m.get("version"))
            for m in self.list_models()
            if m.get("name") == name
        ]
        return (max(versions) if versions else 0) + 1

    def _newest_checkpoint_hash(self, run_id: str) -> str | None:
        ckpt_dir = self._dir("run", run_id) / "checkpoints"
        if not ckpt_dir.is_dir():
            return None
        dirs = sorted(
            (p for p in ckpt_dir.iterdir()
             if p.is_dir() and p.name.startswith("ckpt-")),
            reverse=True,
        )
        for d in dirs:  # COMMIT marker = the manifest hash (12 §11.1)
            marker = d / "COMMIT"
            if marker.is_file():
                digest = marker.read_text(encoding="utf-8").strip()
                if digest:
                    return digest
        return None

    # ------------------------------------------------------------------
    # downstream operations (build step 11; 13 §6.7–§6.10, 12 §15.3–§15.4)
    # ------------------------------------------------------------------

    def _model_for_ops(self, ref: str) -> dict[str, Any]:
        """Model resolved + published (13 §5.4). CREATED/VALIDATED are
        not consumable — they never finished publishing."""
        entry = self.resolve_model(ref)
        state = entry.get("state")
        if state in (ModelState.CREATED.value, ModelState.VALIDATED.value):
            raise PreconditionFailed(
                f"model {ref} is {state} — not published for use",
                hint="published models are AVAILABLE (13 §5.4); a partially "
                     "registered model is repaired by re-import or "
                     "`validate` (13 §4.1)",
            )
        return entry

    def _record_consumption(
        self, model_id: str, action: str, event: str, data: dict[str, Any]
    ) -> str:
        """First-consumption marker semantics (13 §5.4: AVAILABLE fans out
        to exactly one of EVALUATED/EXPORTED/DEPLOYED/USED_AS_...).

        The FIRST consumption fires the normative transition. Later
        consumptions (second evaluation with a different protocol, export
        after evaluate, ...) still record their artifact + a no-`to`
        journal event — the marker never flips back and forth, and the
        artifacts are the full record (12 §15.4: a different protocol is
        a DIFFERENT artifact, not an update)."""
        state = self._load_projection("model", model_id)["state"]
        if state == ModelState.AVAILABLE.value:
            return self._fire("model", model_id, action, event, data=data)
        self._journal("model", model_id).append(event, action=action, **data)
        return state

    def _resolve_eval_dataset(self, ref: str) -> dict[str, Any]:
        """Registered + re-hashed dataset for evaluation (13 §7: not
        resolved → exit 2; content changed since registration → BLOCK)."""
        from mlforge.ingest import config as ingest_config
        from mlforge.ingest.identity import full_ref, parse_ref, recompute_identity

        name, version = parse_ref(ref)
        reg_path = self.root / "datasets" / name / "identity.json"
        if not reg_path.is_file():
            raise NotFound(
                f"dataset {ref} not registered",
                hint=f"register it first: mlforge dataset add {name} <PATH>",
            )
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationBlock(
                f"dataset {name}: registration unreadable ({exc})") from exc
        reg_version = str(reg.get("version") or "")
        if reg_version and reg_version != version:
            raise ValidationBlock(
                f"{ref}: registered version {reg_version!r} != {version!r} "
                "(versions are identity, never reinterpreted)",
            )
        paths = ingest_config.load_paths(self.root)
        path = paths.get(name)
        if not path:
            raise ValidationBlock(
                f"{name}: no machine-local path configured",
                hint=f"mlforge dataset add {name} <PATH> (paths are explicit, "
                     "never discovered — 12 §6.3)",
            )
        identity, _manifest = recompute_identity(
            path, name, version, schema=reg.get("schema")
        )
        registered = reg.get("identity")
        if registered and identity != registered:
            raise ValidationBlock(
                f"{name}: dataset content changed since registration "
                f"(registered {registered}, found {identity})",
                hint="re-register explicitly with `mlforge dataset add "
                     "--force` — never silently reinterpret identity (13 §5.2)",
            )
        return {
            "name": name,
            "version": version,
            "ref": full_ref(name, version),
            "identity": identity,
            "path": path,
        }

    def _scan_model_sources(self, entry: dict[str, Any]) -> list[str]:
        """Secrets scan over the model's source directories (13 §7:
        secrets in artifacts → BLOCK packaging/export, no partial run)."""
        from mlforge.validation.gate import scan_for_secrets

        findings: list[str] = []
        dirs: list[Path] = [self.root / "models" / str(entry.get("model_id"))]
        run_id = entry.get("run_id")
        if run_id:
            dirs.append(self.root / "runs" / str(run_id))
        contract_dir = contract_source_dir(self.root, entry)
        if contract_dir is not None:
            dirs.append(contract_dir)
        seen: set[Path] = set()
        for d in dirs:
            d = d.resolve()
            if d in seen or not d.is_dir():
                continue
            seen.add(d)
            findings.extend(scan_for_secrets(d))
        return findings

    def _prepare_evaluation(
        self,
        model_ref: str,
        dataset_ref: str,
        protocol_overrides: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve + build the record WITHOUT writing (deterministic —
        the previewed numbers are exactly what gets recorded)."""
        from mlforge.ingest.transforms import env_fingerprint

        entry = self._model_for_ops(model_ref)
        dataset = self._resolve_eval_dataset(dataset_ref)
        protocol = build_protocol(
            dict(protocol_overrides) if protocol_overrides else None
        )
        record = build_evaluation(
            eval_id="",
            model_entry=entry,
            dataset_hash=dataset["identity"],
            dataset_ref=dataset["ref"],
            code_hash=source_tree_hash(),
            environment_hash=env_fingerprint(),
            protocol=protocol,
        )
        return entry, record

    def preview_evaluation(
        self,
        model_ref: str,
        dataset_ref: str,
        *,
        protocol_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """SHOW PROTOCOL before confirmation (13 §6.7) — resolution and
        validation run now; nothing is written, no state changes."""
        _, record = self._prepare_evaluation(
            model_ref, dataset_ref, protocol_overrides
        )
        return record

    def evaluate_model(
        self,
        model_ref: str,
        dataset_ref: str,
        *,
        protocol_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """EVALUATE (13 §6.7): resolve hashes → run harness → immutable
        five-component evaluation artifact."""
        entry, record = self._prepare_evaluation(
            model_ref, dataset_ref, protocol_overrides
        )
        record["eval_id"] = new_eval_id()
        record["created_ts"] = time.time()
        write_evaluation(self.root, record)
        self._record_consumption(
            str(entry["model_id"]),
            "record_evaluation",
            "evaluation_recorded",
            {
                "eval_id": record["eval_id"],
                "dataset_ref": record["dataset_ref"],
                "evaluation_protocol_hash": record["evaluation_protocol_hash"],
            },
        )
        return record

    def list_model_evaluations(self, model_ref: str) -> list[dict[str, Any]]:
        entry = self.resolve_model(model_ref)
        return list_evaluations(self.root, model_id=str(entry["model_id"]))

    def compare_models(self, refs: list[str]) -> dict[str, Any]:
        """COMPARE (13 §6.10): descriptive columns; NOT_COMPARABLE when
        evaluation protocol hashes differ — shown, never averaged."""
        if len(refs) < 2:
            raise ValidationBlock(
                "compare needs at least two model references",
                hint="mlforge compare model://a:v1 model://b:v1 (13 §6.10)",
            )
        columns: list[dict[str, Any]] = []
        for ref in refs:
            entry = self.resolve_model(ref)  # unknown → NotFound (exit 2)
            record = latest_evaluation(self.root, str(entry["model_id"]))
            dataset, base_model = "—", "—"
            run_id = entry.get("run_id")
            if run_id:
                try:
                    spec = self._run_spec(str(run_id))
                    dataset = ", ".join(spec.train_datasets)
                except MlforgeError:
                    pass
                try:
                    lin = self.get_lineage(str(run_id))
                    base_model = lin.get("parent_model") or "—"
                except MlforgeError:
                    pass
            columns.append(
                {
                    "ref": (f"{entry.get('name')}:{entry.get('version')}"
                            if entry.get("name") else entry["model_id"]),
                    "model_id": entry.get("model_id"),
                    "dataset": dataset,
                    "base_model": base_model,
                    "evaluation": record,
                }
            )
        records = [c["evaluation"] for c in columns if c["evaluation"] is not None]
        comparable, not_comparable = _comparability_groups(records)
        comparable_hashes = set(comparable)
        for col in columns:
            ev = col["evaluation"]
            if ev is None:
                col["comparable"] = None
            else:
                col["comparable"] = bool(comparable) and (
                    ev["evaluation_protocol_hash"] in comparable_hashes
                )
                col["protocol_hash"] = ev["evaluation_protocol_hash"]
        return {
            "columns": columns,
            "comparable_protocols": sorted(comparable),
            "not_comparable_protocols": sorted(not_comparable),
        }

    def export_model(self, model_ref: str, fmt: str) -> dict[str, Any]:
        """EXPORT (13 §6.9): contract → operator validation → immutable
        export artifact with its own identity."""
        entry = self._model_for_ops(model_ref)
        if fmt not in EXPORT_FORMATS:
            # Guard BEFORE any contract work: an unknown format must give
            # the registry, never a KeyError (13 §7 failure matrix).
            raise ValidationBlock(
                f"unsupported export format {fmt!r}",
                hint=f"formats: {', '.join(sorted(EXPORT_FORMATS))}",
            )
        findings = self._scan_model_sources(entry)
        if findings:
            raise ValidationBlock(
                "secrets detected in artifacts: " + "; ".join(findings[:5]),
                hint="BLOCK export (13 §7) — remove the secret material and "
                     "re-register; secrets never enter artifacts (12 §16)",
            )
        # contract: reuse the existing one (operators/schema are the
        # model's), retarget the runtime; else build the scaffold contract
        existing_dir = contract_source_dir(self.root, entry)
        if existing_dir is not None:
            model_spec = load_model_spec(self.root, entry)
            contract = dict(model_spec["inference_contract"])
            contract["runtime"] = {
                "framework": fmt,
                "opset": EXPORT_FORMATS[fmt]["opset"]
                if fmt in EXPORT_FORMATS else contract.get("runtime", {}).get("opset"),
            }
            model_spec = {**model_spec, "inference_contract": contract}
            _required_operators(model_spec)  # fail-closed: list must exist
        else:
            model_spec = scaffold_contract(entry, fmt)
        validations = validate_export(model_spec, fmt)  # BLOCKs unknown fmt/ops
        record = build_export(
            export_id=new_export_id(),
            model_entry=entry,
            model_hash=model_entry_hash(entry),
            model_spec=model_spec,
            fmt=fmt,
            validations=validations,
            numerical=numerical_validation(
                model_spec, model_entry_hash(entry), fmt
            ),
        )
        record["created_ts"] = time.time()
        write_export(self.root, record)
        self._record_consumption(
            str(entry["model_id"]),
            "record_export",
            "export_recorded",
            {
                "export_id": record["export_id"],
                "format": fmt,
                "contract_hash": record["contract_hash"],
            },
        )
        return record

    def package_model(self, model_ref: str) -> dict[str, Any]:
        """PACKAGE (13 §4.1, 12 §15.3): contract required; secrets scan;
        immutable bundle artifact."""
        entry = self._model_for_ops(model_ref)
        model_spec = load_model_spec(self.root, entry)  # no contract ⇒ BLOCK
        findings = self._scan_model_sources(entry)
        if findings:
            raise ValidationBlock(
                "secrets detected in artifacts: " + "; ".join(findings[:5]),
                hint="BLOCK packaging (13 §7) — remove the secret material "
                     "first; secrets never enter artifacts (12 §16)",
            )
        provenance: dict[str, Any] = {
            "model": {k: entry.get(k) for k in
                      ("model_id", "name", "version", "artifact_hash",
                       "run_id", "run_spec_hash", "origin")},
        }
        if entry.get("run_id"):
            try:
                provenance["lineage"] = self.get_lineage(str(entry["run_id"]))
                provenance["run_chain"] = [
                    c["run_id"] for c in self.lineage_chain(str(entry["run_id"]))
                ]
            except MlforgeError:
                provenance["lineage"] = None  # provenance run not in workspace
        integrity = component_integrity(
            model_spec, provenance, entry.get("artifact_hash")
        )
        record = build_bundle(
            bundle_id=new_bundle_id(),
            model_entry=entry,
            model_hash=model_entry_hash(entry),
            model_spec=model_spec,
            provenance=provenance,
            integrity=integrity,
        )
        record["created_ts"] = time.time()
        write_bundle(
            self.root,
            record=record,
            model_spec=model_spec,
            provenance=provenance,
            integrity=integrity,
        )
        # packaging consumes no lifecycle state (13 §5.4 has no BUNDLED
        # state) — journal-only, artifacts carry the record.
        self._journal("model", str(entry["model_id"])).append(
            "bundle_created",
            action="package",
            bundle_id=record["bundle_id"],
            contract_hash=record["contract_hash"],
        )
        return record

    def infer_model(self, model_ref: str, input_path: str) -> dict[str, Any]:
        """ONE-SHOT INFER (13 §6.8): contract check → input schema check
        → scaffold execution → output artifact with its own identity."""
        entry = self._model_for_ops(model_ref)
        model_spec = load_model_spec(self.root, entry)  # no contract ⇒ BLOCK
        checked = check_input(model_spec, input_path)  # violations ⇒ BLOCK
        record = execute_scaffold(
            output_id=new_output_id(),
            model_entry=entry,
            contract_doc=model_spec,
            checked=checked,
        )
        record["created_ts"] = time.time()
        write_output(self.root, record)
        self._journal("model", str(entry["model_id"])).append(
            "inference_executed",
            action="infer",
            output_id=record["output_id"],
            input_identity=record["input"]["identity"],
        )
        return record

    def import_external_model(self, path: str) -> dict[str, Any]:
        """`model import <PATH>` (13 §4.1): validate the package, hash
        the weights, register + publish with its real contract."""
        spec, weights = read_package(path)
        name, version = str(spec["name"]), str(spec["version"])
        for existing in self.list_models():
            if existing.get("name") == name and existing.get("version") == version:
                raise ValidationBlock(
                    f"model {name}:{version} is already registered",
                    hint="imported versions are immutable — declare a new "
                         "version in model_spec.json or use the existing one "
                         "(`mlforge model list`)",
                )
        model_id = self.create_model(
            artifact_hash=str(weights["artifact_hash"]),
            name=name,
            version=version,
            run_id=None,
            run_spec_hash=None,
            origin="import",
        )
        d = self._dir("model", model_id)
        spec_path = d / "model_spec.json"
        tmp = spec_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(spec_path)
        self._journal("model", model_id).append(
            "model_imported",
            action="import",
            package=str(path),
            weights_bytes=weights["bytes"],
            artifact_hash=weights["artifact_hash"],
        )
        self.validate_model(model_id)
        self.publish_model(model_id)
        return self.resolve_model(f"{name}:{version}")

    def model_integrity_report(self, model_ref: str) -> dict[str, Any]:
        """`validate <MODEL>` (13 §4.1): registry + contract integrity
        checks — read-only, fail-closed on unreadable artifacts."""
        entry = self.resolve_model(model_ref)
        checks: list[dict[str, str]] = []

        def add(name: str, status: str, detail: str) -> None:
            checks.append({"name": name, "status": status, "detail": detail})

        add("registry sidecar", "PASS",
            f"{entry.get('name')}:{entry.get('version')} state "
            f"{entry.get('state')}")
        add("provenance", "PASS",
            f"origin {entry.get('origin') or 'run'}"
            + (f" · run {entry.get('run_id')}" if entry.get("run_id")
               else " · external package"))
        if entry.get("artifact_hash"):
            add("weights reference", "PASS", str(entry["artifact_hash"]))
        else:
            add("weights reference", "PASS",
                "none recorded (scaffold publish without checkpoints)")
        source = contract_source_dir(self.root, entry)
        if source is None:
            add("inference contract", "PASS",
                "none yet — export or import creates one (12 §15.3)")
        else:
            try:
                spec = load_model_spec(self.root, entry)
                ops = _required_operators(spec)
                add("inference contract", "PASS",
                    f"{len(ops)} operators · runtime "
                    f"{spec['inference_contract'].get('runtime')}")
            except ValidationBlock as exc:
                add("inference contract", "FAIL", str(exc))
        blocked = any(c["status"] == "FAIL" for c in checks)
        return {
            "model_ref": f"{entry.get('name')}:{entry.get('version')}",
            "model_id": entry.get("model_id"),
            "state": entry.get("state"),
            "checks": checks,
            "blocked": blocked,
        }

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
