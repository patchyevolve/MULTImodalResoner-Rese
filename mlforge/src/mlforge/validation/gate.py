"""The validation gate — 19-step resume flow, fail-closed core.

Normative: 12_training_system.md §18 (the 19 steps), §7 (BLOCK vs WARN,
report, preflight), §8 (EXACT/PORTABLE), §17 (invariant table), §23.3
(revalidation), 13 §5.3/§7 (gate BLOCK behavior).

Step spine (12 §18 — steps 1–16 are checks; 17–19 are execution actions
owned by the runtime, build step 6):

     1 manifest      run manifest integrity          (built-in)
     2 schema        schema version supported        (built-in)
     3 source_code   source artifact + code hash     provider
     4 dataset       dataset identity (crypto)       provider
     5 transform     transform identity              provider
     6 model         arch + base weights hash        provider
     7 environment   environment image digest        provider
     8 driver        driver compatibility            provider
     9 hardware      hardware capabilities           provider
    10 plan          execution plan (solver)         provider
    11 global_batch  global batch preserved          provider (or derived)
    12 precision     precision policy supported      provider
    13 topology      distributed topology (§8.5)     provider
    14 checkpoint    newest-valid predicate (§11.2)  provider
    15 lease         ACQUIRE run lease (§23.2)       provider (step 5)
    16 revalidate    volatile subset (§23.3)         provider (step 5)

Fail-closed rules:
  * missing provider ⇒ FAIL "unverifiable" (never a skip, never a guess)
  * first FAIL stops the run — steps after it do not execute, so the
    side-effect step (15, lease acquisition) can never run after an
    earlier failure (12 §18: "Any failure in 1–16 → training does not
    start")
  * EXACT mode only when checks explicitly report exact_compatible=True;
    otherwise PORTABLE (the safe default, 12 §8)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from mlforge.errors import ValidationBlock
from mlforge.hashing import content_hash
from mlforge.run_spec import SCHEMA_VERSION, RunSpec
from mlforge.validation.report import FAIL, PASS, WARN, Check, ValidationReport

#: A provider answers one gate step. Returns a Check (verdict decided by
#: the provider) or None ⇒ the engine records FAIL "unverifiable".
Provider = Callable[["GateContext"], "Check | None"]


@dataclass
class GateContext:
    run_id: str
    root: Path
    run_spec: RunSpec
    #: TRAIN | RESUME — decides how the workflow applies the result.
    flow: str = "TRAIN"
    #: Execution segment label recorded on success (12 §8/§12.2).
    segment: str | None = None
    #: Extra facts providers may need (checkpoint id, planner output, ...).
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.run_id


@dataclass(frozen=True)
class GateStep:
    number: int
    id: str
    label: str
    provider: Provider | None = None  # None ⇒ built-in defaults below


def _builtin_manifest(ctx: GateContext) -> Check:
    """Step 1: run_spec.json exists, parses, and its identity matches the
    identity recorded when the run was created (manifest integrity)."""
    spec_path = ctx.run_dir / "run_spec.json"
    if not spec_path.is_file():
        return Check(1, "manifest", "Run manifest integrity", FAIL,
                     f"run_spec.json missing at {spec_path}")
    try:
        on_disk = RunSpec.from_dict(json.loads(spec_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationBlock, KeyError) as exc:
        return Check(1, "manifest", "Run manifest integrity", FAIL,
                     f"run_spec.json unreadable: {exc}")
    if on_disk.identity != ctx.run_spec.identity:
        return Check(1, "manifest", "Run manifest integrity", FAIL,
                     f"identity mismatch: manifest {on_disk.identity} != "
                     f"context {ctx.run_spec.identity}")
    # Cross-check against the identity recorded at creation (journal authority).
    events = ctx.run_dir / "events.jsonl"
    if events.is_file():
        recorded = None
        for line in events.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if "run_spec_hash" in ev:
                recorded = ev["run_spec_hash"]
        if recorded is not None and recorded != on_disk.identity:
            return Check(1, "manifest", "Run manifest integrity", FAIL,
                         f"journal recorded {recorded}, manifest is {on_disk.identity}")
    return Check(1, "manifest", "Run manifest integrity", PASS,
                 f"immutable identity {on_disk.identity}")


def _builtin_schema(ctx: GateContext) -> Check:
    """Step 2: schema version supported — never silently reinterpret."""
    v = ctx.run_spec.schema_version
    if v != SCHEMA_VERSION:
        return Check(2, "schema", "Schema version", FAIL,
                     f"unsupported schema_version {v} (supported: {SCHEMA_VERSION})",
                     {"supported": [SCHEMA_VERSION]})
    return Check(2, "schema", "Schema version", PASS, f"v{v} supported")


#: 12 §8.4 semantic-invariant keys that must be verbatim-equal for EXACT
#: continuation (checked by providers via ctx.run_spec; recorded here so
#: the classification exists in exactly one place).
SEMANTIC_INVARIANTS = (
    "optimizer", "learning_rate", "scheduler", "loss",
    "seed", "global_batch", "epochs", "precision_policy",
)

#: Secret patterns (12 §7.1 "secrets present in artifacts" → BLOCK).
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password)\s*[:=]\s*[\"']?[A-Za-z0-9/+_-]{16,}"),
)


def scan_for_secrets(directory: Path) -> list[str]:
    """Heuristic scan of a run folder for secret material (fail-closed
    when found; absence of findings is a PASS, not a proof)."""
    findings: list[str] = []
    if not directory.is_dir():
        return findings
    for p in sorted(directory.rglob("*")):
        if not p.is_file() or p.stat().st_size > 10 * (1 << 20):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in _SECRET_PATTERNS:
            if pat.search(text):
                findings.append(f"{p.relative_to(directory)}: matches {pat.pattern!r}")
                break
    return findings


#: Normative step spine (12 §18). `provider=None` ⇒ built-in above.
RESUME_GATE_STEPS: tuple[GateStep, ...] = (
    GateStep(1, "manifest", "Run manifest integrity", _builtin_manifest),
    GateStep(2, "schema", "Schema version", _builtin_schema),
    GateStep(3, "source_code", "Source artifact + code hash"),
    GateStep(4, "dataset", "Dataset identity (cryptographic)"),
    GateStep(5, "transform", "Transform artifact identity"),
    GateStep(6, "model", "Model architecture + base weights hash"),
    GateStep(7, "environment", "Environment image available"),
    GateStep(8, "driver", "Driver compatibility"),
    GateStep(9, "hardware", "Hardware capability detection"),
    GateStep(10, "plan", "Execution plan (feasibility solver)"),
    GateStep(11, "global_batch", "Global batch preserved"),
    GateStep(12, "precision", "Precision policy supported"),
    GateStep(13, "topology", "Distributed topology compatible"),
    GateStep(14, "checkpoint", "Checkpoint integrity (newest-valid predicate)"),
    GateStep(15, "lease", "Acquire run lease (single-writer)"),
    GateStep(16, "revalidate", "Revalidate volatile subset (TOCTOU close)"),
)

#: Steps 17–19 (restore state, create execution segment, resume) are
#: runtime actions after the gate passes — documented here so the
#: numbering of all 19 is visible in one place (12 §18).
POST_GATE_ACTIONS = ("restore_full_state", "create_execution_segment", "resume")

#: Report titles/results differ per flow (12 §7.2).
_FLOW_TITLES = {
    "TRAIN": ("VALIDATION REPORT", "SAFE TO TRAIN", "TRAINING BLOCKED"),
    "RESUME": ("VALIDATION REPORT", "SAFE TO RESUME", "RESUME BLOCKED"),
}


class ValidationGate:
    """Runs the normative step spine against a provider set.

    Providers are keyed by step id (`source_code`, `dataset`, ...). Missing
    key or a provider returning None ⇒ FAIL "unverifiable" — fail-closed.
    """

    def __init__(
        self,
        providers: Mapping[str, Provider] | None = None,
        *,
        steps: tuple[GateStep, ...] = RESUME_GATE_STEPS,
        fail_fast: bool = True,
    ):
        self.providers = dict(providers or {})
        self.steps = steps
        self.fail_fast = fail_fast

    def run(self, ctx: GateContext) -> ValidationReport:
        checks: list[Check] = []
        blocked_at: Check | None = None
        for step in self.steps:
            check = self._run_step(step, ctx)
            checks.append(check)
            if check.verdict == FAIL:
                blocked_at = check
                if self.fail_fast:
                    break
        mode = None if blocked_at else self._mode(checks, ctx)
        title, pass_result, fail_result = _FLOW_TITLES.get(
            ctx.flow, _FLOW_TITLES["RESUME"]
        )
        return ValidationReport(
            run_id=ctx.run_id,
            checks=tuple(checks),
            mode=mode,
            segment=None if blocked_at else ctx.segment,
            title=title,
            pass_result=pass_result,
            fail_result=fail_result,
        )

    def _run_step(self, step: GateStep, ctx: GateContext) -> Check:
        provider = step.provider or self.providers.get(step.id)
        if provider is None:
            return Check(
                step.number, step.id, step.label, FAIL,
                f"unverifiable: no provider for '{step.id}' "
                "(fail-closed: unverifiable == failed verification)",
            )
        try:
            result = provider(ctx)
        except Exception as exc:  # a throwing check is a failed check
            return Check(step.number, step.id, step.label, FAIL,
                         f"check raised: {type(exc).__name__}: {exc}")
        if result is None:
            return Check(step.number, step.id, step.label, FAIL,
                         "unverifiable: provider returned no verdict (fail-closed)")
        # Providers answer their own step; a mismatch is a bug → trust the
        # engine's numbering/label (spec's spine is normative).
        return Check(step.number, step.id, step.label, result.verdict,
                     result.detail, dict(result.data))

    @staticmethod
    def _mode(checks: list[Check], ctx: GateContext) -> str:
        """12 §8: EXACT only when checks explicitly say exact_compatible;
        otherwise PORTABLE (safe default)."""
        signals = [c.data.get("exact_compatible") for c in checks
                   if "exact_compatible" in c.data]
        if signals and all(s is True for s in signals):
            return "EXACT"
        return "PORTABLE"


# ---------------------------------------------------------------------------
# Ready-to-use providers (no external deps); richer providers arrive with
# their build steps (datasets 8, env 6, planner 9, leases 5).
# ---------------------------------------------------------------------------

def provide_unverifiable(label_hint: str) -> Provider:
    """Explicit 'I know this is missing' provider — useful in tests and
    for callers that must acknowledge a gap without silently passing."""
    def _p(ctx: GateContext) -> Check | None:
        return None  # engine records FAIL unverifiable
    return _p


def provide_pass(detail: str = "verified", **data: Any) -> Provider:
    """Test/helper provider: always PASS with given detail/data."""
    def _p(ctx: GateContext) -> Check:
        return Check(0, "", "", PASS, detail, dict(data))
    return _p


def provide_fail(detail: str, **data: Any) -> Provider:
    def _p(ctx: GateContext) -> Check:
        return Check(0, "", "", FAIL, detail, dict(data))
    return _p


def make_gate(providers: Mapping[str, Provider], **kwargs: Any) -> ValidationGate:
    """Convenience: providers for the listed steps, unverifiable elsewhere."""
    merged = dict(providers)
    return ValidationGate(merged, **kwargs)
