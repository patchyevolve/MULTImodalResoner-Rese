"""State machine engine + the four normative transition tables.

Normative: 13_product_specification.md §5.1–§5.4.

Design:
  * transitions are data (tables), not scattered ifs — the spec's tables are
    directly auditable against these dicts
  * an illegal action raises InvalidTransition (exit 3) listing the legal
    actions — nothing changes, no state is left indeterminate (13 §7)
  * guards may raise a domain error (gate BLOCK, FORK_ONLY, ...) — they
    never silently pick another state
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from mlforge.errors import InvalidTransition, NoValidContinuation
from mlforge.states import (
    LIVE_STATES,
    DatasetState,
    FailureRecovery,
    ModelState,
    ProjectState,
    RunState,
)

Ctx = Mapping[str, Any]
#: Guard receives the caller context and the source state; raises to block.
Guard = Callable[[Ctx, str], None]


@dataclass(frozen=True)
class Rule:
    action: str
    sources: frozenset[str]
    target: str
    guard: Guard | None = None


class StateMachine:
    def __init__(self, name: str, initial: str, rules: tuple[Rule, ...]):
        self.name = name
        self.initial = initial
        self.rules = rules

    def _rules_from(self, state: str, action: str) -> list[Rule]:
        return [r for r in self.rules if r.action == action and state in r.sources]

    def legal_actions(self, state: str, ctx: Ctx | None = None) -> list[str]:
        """Actions legal from `state` (guards included when ctx provided)."""
        ctx = ctx or {}
        legal: list[str] = []
        for r in self.rules:
            if state not in r.sources:
                continue
            if r.guard is not None:
                try:
                    r.guard(ctx, state)
                except Exception:  # noqa: BLE001, S112 - a throwing guard is not-legal: fail-closed action filtering
                    continue
            legal.append(r.action)
        return sorted(set(legal))

    def fire(self, state: str, action: str, ctx: Ctx | None = None) -> str:
        """Apply `action` in `state`; returns the target state or raises."""
        ctx = ctx or {}
        candidates = self._rules_from(state, action)
        if not candidates:
            raise InvalidTransition(self.name, state, action, self.legal_actions(state, ctx))
        # Guards first: a blocked action must change nothing.
        for r in candidates:
            if r.guard is not None:
                r.guard(ctx, state)
        target = candidates[0].target
        return target


# ---------------------------------------------------------------------------
# 13 §5.3 — Training Run (core)
# ---------------------------------------------------------------------------

def _guard_resume(ctx: Ctx, source: str) -> None:
    """`resume` guards (13 §5.3; 12 §12 "Recover" flow).

    FAILED(recovery: RESUME)    → resume re-runs the full validation gate
    FAILED(recovery: FORK_ONLY) → NO VALID CONTINUATION (exit 3),
        fork/retrain only — never silently start from step 0.
    RECONCILING                 → requires a valid checkpoint (12 §11.2);
        reconciliation itself never starts training — `resume` does.
    """
    if source == RunState.FAILED.value:
        recovery = ctx.get("failure_recovery")
        if recovery != FailureRecovery.RESUME.value:
            raise NoValidContinuation(
                str(ctx.get("run_id", "<run>")),
                cause=ctx.get("failure_cause"),
            )
    if source == RunState.RECONCILING.value and not ctx.get("has_valid_checkpoint"):
        raise NoValidContinuation(
            str(ctx.get("run_id", "<run>")),
            cause="reconciliation found no valid checkpoint",
        )


def _guard_intentional(ctx: Ctx, source: str) -> None:
    """pause/stop/crash are never system-initiated (13 §1 core decision
    boundary: these are explicit events — user command or crash detection —
    never an internal automatic transition)."""
    if not (ctx.get("explicit") or ctx.get("user_initiated")):
        from mlforge.errors import PreconditionFailed

        raise PreconditionFailed(
            "pause/stop/crash must be explicitly initiated",
            hint="only an explicit user command or crash event may fire this",
        )


def _guard_no_checkpoint(ctx: Ctx, source: str) -> None:
    """reconcile_no_checkpoint is only legal when the scan found nothing."""
    if ctx.get("has_valid_checkpoint"):
        from mlforge.errors import PreconditionFailed

        raise PreconditionFailed(
            "a valid checkpoint exists — reconciliation must not mark it failed",
            hint="use `resume` (12 §12.3)",
        )


_LIVE = frozenset(s.value for s in LIVE_STATES)

RUN_RULES: tuple[Rule, ...] = (
    # creation / validation gate
    Rule("begin_validation", frozenset({RunState.CREATED.value}), RunState.VALIDATING.value),
    Rule("validation_pass", frozenset({RunState.VALIDATING.value}), RunState.READY.value),
    Rule("validation_fail", frozenset({RunState.VALIDATING.value}), RunState.FAILED.value),
    Rule("preflight_pass", frozenset({RunState.READY.value}), RunState.RUNNING.value),
    Rule("preflight_fail", frozenset({RunState.READY.value}), RunState.FAILED.value),
    # pause path (user, resumable)
    Rule("pause", frozenset({RunState.RUNNING.value}), RunState.PAUSING.value, _guard_intentional),
    Rule("pause_committed", frozenset({RunState.PAUSING.value}), RunState.PAUSED.value),
    Rule("pause_checkpoint_failed", frozenset({RunState.PAUSING.value}), RunState.FAILED.value),
    # stop path (user, no resume)
    Rule("stop", frozenset({RunState.RUNNING.value, RunState.PAUSED.value}), RunState.STOPPING.value, _guard_intentional),
    Rule("stop_committed", frozenset({RunState.STOPPING.value}), RunState.STOPPED.value),
    # checkpoint cycle
    Rule("checkpoint", frozenset({RunState.RUNNING.value}), RunState.CHECKPOINTING.value),
    Rule("checkpoint_done", frozenset({RunState.CHECKPOINTING.value}), RunState.RUNNING.value),
    Rule("checkpoint_failed", frozenset({RunState.CHECKPOINTING.value}), RunState.FAILED.value),
    # failures
    Rule("runtime_error", frozenset({RunState.RUNNING.value}), RunState.FAILED.value),
    Rule("complete", frozenset({RunState.RUNNING.value}), RunState.COMPLETED.value),
    # crash / power loss: any live state → INTERRUPTED (never PAUSED/RUNNING)
    Rule("crash", _LIVE, RunState.INTERRUPTED.value, _guard_intentional),
    # reconciliation (12 §12.3) — never starts training by itself;
    # after a successful scan the run waits in RECONCILING for `resume`
    Rule("begin_reconciliation", frozenset({RunState.INTERRUPTED.value}), RunState.RECONCILING.value),
    Rule(
        "reconcile_no_checkpoint",
        frozenset({RunState.RECONCILING.value}),
        RunState.FAILED.value,
        _guard_no_checkpoint,
    ),
    # resume: PAUSED ordinary; RECONCILING needs a valid checkpoint;
    # FAILED only with disposition RESUME
    Rule(
        "resume",
        frozenset(
            {
                RunState.PAUSED.value,
                RunState.RECONCILING.value,
                RunState.FAILED.value,
            }
        ),
        RunState.VALIDATING.value,
        _guard_resume,
    ),
)

RUN_MACHINE = StateMachine("run", RunState.CREATED.value, RUN_RULES)


# ---------------------------------------------------------------------------
# 13 §5.1–§5.2, §5.4 — Project / Dataset / Model
# ---------------------------------------------------------------------------

PROJECT_RULES: tuple[Rule, ...] = (
    Rule("configure", frozenset({ProjectState.CREATED.value}), ProjectState.CONFIGURED.value),
    Rule("mark_ready", frozenset({ProjectState.CONFIGURED.value}), ProjectState.READY.value),
)
PROJECT_MACHINE = StateMachine("project", ProjectState.CREATED.value, PROJECT_RULES)

DATASET_RULES: tuple[Rule, ...] = (
    Rule("verify", frozenset({DatasetState.REGISTERED.value}), DatasetState.VERIFIED.value),
    Rule("reject", frozenset({DatasetState.REGISTERED.value, DatasetState.VERIFIED.value}), DatasetState.REJECTED.value),
    Rule("prepare", frozenset({DatasetState.VERIFIED.value}), DatasetState.PREPARED.value),
)
DATASET_MACHINE = StateMachine("dataset", DatasetState.REGISTERED.value, DATASET_RULES)

_AVAILABLE = frozenset({ModelState.AVAILABLE.value})
MODEL_RULES: tuple[Rule, ...] = (
    Rule("validate", frozenset({ModelState.CREATED.value}), ModelState.VALIDATED.value),
    Rule("publish", frozenset({ModelState.VALIDATED.value}), ModelState.AVAILABLE.value),
    Rule("record_evaluation", _AVAILABLE, ModelState.EVALUATED.value),
    Rule("record_export", _AVAILABLE, ModelState.EXPORTED.value),
    Rule("record_deployment", _AVAILABLE, ModelState.DEPLOYED.value),
    Rule("record_finetune_base", _AVAILABLE, ModelState.USED_AS_FINE_TUNE_BASE.value),
)
MODEL_MACHINE = StateMachine("model", ModelState.CREATED.value, MODEL_RULES)
