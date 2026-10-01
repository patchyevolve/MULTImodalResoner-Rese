"""Error hierarchy and CLI exit codes.

Normative contract: 13_product_specification.md §4.2.

    0  success
    1  validation BLOCK
    2  not found
    3  precondition failed
    4  runtime error

"Exit codes are contract; scripts depend on them." — 13 §7 universal rules.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_VALIDATION_BLOCK = 1
EXIT_NOT_FOUND = 2
EXIT_PRECONDITION = 3
EXIT_RUNTIME = 4


class MlforgeError(Exception):
    """Base for all MLForge errors."""

    exit_code = EXIT_RUNTIME
    code = "MLFORGE_ERROR"

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def render(self) -> str:
        out = f"[{self.code}] {self.message}"
        if self.hint:
            out += f"\n  hint: {self.hint}"
        return out


class ValidationBlock(MlforgeError):
    """The validation gate refused to proceed (exit 1). Never trains."""

    exit_code = EXIT_VALIDATION_BLOCK
    code = "VALIDATION_BLOCK"


class NotFound(MlforgeError):
    """Referenced object does not exist (exit 2)."""

    exit_code = EXIT_NOT_FOUND
    code = "NOT_FOUND"


class PreconditionFailed(MlforgeError):
    """The command is well-formed but the current state forbids it (exit 3)."""

    exit_code = EXIT_PRECONDITION
    code = "PRECONDITION_FAILED"


class InvalidTransition(PreconditionFailed):
    """Illegal state transition — lists the legal actions from the current state.

    Per 13 §5: "No command leaves a run in an indeterminate state — every
    failure maps to a defined state." An unknown action changes nothing.
    """

    code = "INVALID_TRANSITION"

    def __init__(self, obj_type: str, state: str, action: str, legal_actions: list[str]):
        legal = ", ".join(sorted(legal_actions)) or "(none)"
        super().__init__(
            f"action {action!r} is not legal from {obj_type} state {state}",
            hint=f"legal actions from {state}: {legal}",
        )
        self.state = state
        self.action = action
        self.legal_actions = sorted(legal_actions)


class NoValidContinuation(PreconditionFailed):
    """FAILED(recovery: FORK_ONLY) — resume is forbidden (exit 3).

    13 §5.3: `FAILED + recovery=FORK_ONLY` → resume exits 3:
    NO VALID CONTINUATION — use mlforge fork or retrain (new run).
    Never silently starts from step 0.
    """

    code = "NO_VALID_CONTINUATION"

    def __init__(self, run_id: str, cause: str | None = None):
        msg = f"run {run_id}: no valid continuation exists"
        if cause:
            msg += f" (cause: {cause})"
        super().__init__(msg, hint="use `mlforge fork` or `mlforge retrain` (new run)")
        self.run_id = run_id


class RunAlreadyExecuting(PreconditionFailed):
    """Resume requested while the execution/run lease is held (exit 3).

    13 §7: RESUME (any state) | execution lease held →
    BLOCK RUN_ALREADY_EXECUTING.
    """

    code = "RUN_ALREADY_EXECUTING"

    def __init__(self, run_id: str, holder: str):
        super().__init__(
            f"run {run_id} is already executing (lease held by {holder})",
            hint="`mlforge status` shows the holder; `mlforge lease break --force` to override",
        )
        self.run_id = run_id
        self.holder = holder
