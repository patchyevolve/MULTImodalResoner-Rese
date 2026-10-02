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

from mlforge.errors import NotFound, PreconditionFailed, ValidationBlock
from mlforge.hashing import content_hash
from mlforge.planner.plan import PLAN_FILENAME
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
        except PreconditionFailed:
            # Concurrency/state preconditions (RUN_ALREADY_EXECUTING, ...)
            # are exit-3 product contracts (13 §7), not gate verdicts —
            # they propagate instead of becoming an exit-1 FAIL.
            raise
        except ValidationBlock as exc:
            # A provider that refuses via the validation contract becomes a
            # FAIL at its step (the gate's own block semantics).
            return Check(step.number, step.id, step.label, FAIL, exc.message)
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


def provide_dataset_identity(root: str | Path) -> Provider:
    """Step 4 (and preflight dataset re-verify): cryptographic identity
    for every dataset in the run spec (12 §18 step 4, §6.2).

    Check sequence per dataset — each failure mode has ONE message:
      not registered  → FAIL (register first, `mlforge dataset add`)
      version differs → FAIL (never reinterpret a version)
      store-backed    → content store must hold the exact registered
                        bytes (prepared datasets, 12 §6.4 — no
                        machine-local path exists for a derived dataset;
                        the transform step then verifies its identity)
      no path config  → FAIL (paths are explicit, never discovered)
      hash mismatch   → FAIL (content changed since registration)
      unreadable      → FAIL (path gone / symlink / empty — fail-closed)
    Passes only when EVERY dataset re-hashes to its registration.
    """
    from mlforge.ingest import config as ingest_config
    from mlforge.ingest.identity import parse_ref, recompute_identity
    from mlforge.store import ContentStore

    def _p(ctx: GateContext) -> Check:
        spec = ctx.run_spec
        refs: list[str] = list(spec.train_datasets)
        if spec.val_dataset:
            refs.append(spec.val_dataset)
        try:
            paths = ingest_config.load_paths(ctx.root)
        except PreconditionFailed as exc:
            return Check(0, "", "", FAIL, f"dataset path config unreadable: {exc}")
        store = ContentStore(ctx.root / "store")  # prepared datasets live here
        verified: list[str] = []
        files_hashed = 0
        for ref in refs:
            name, version = parse_ref(ref)
            reg_path = ctx.root / "datasets" / name / "identity.json"
            if not reg_path.is_file():
                return Check(
                    0, "", "", FAIL,
                    f"{ref}: not registered — "
                    f"mlforge dataset add {name} <PATH>",
                )
            try:
                reg = json.loads(reg_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return Check(0, "", "", FAIL,
                             f"{name}: registration unreadable ({exc})")
            reg_version = reg.get("version")
            if reg_version and str(reg_version) != version:
                return Check(
                    0, "", "", FAIL,
                    f"{ref}: registered version {reg_version!r} != {version!r} "
                    "(versions are identity, never reinterpreted)",
                )
            identity = reg.get("identity")
            # Store-backed (prepared) datasets: identity == artifact digest.
            # The content store must still hold those exact bytes.
            if identity and store.contains(identity):
                try:
                    store.get_bytes(identity, verify=True)
                except (ValidationBlock, NotFound, OSError) as exc:
                    detail = (exc.message if isinstance(exc, ValidationBlock)
                              else str(exc))
                    return Check(
                        0, "", "", FAIL,
                        f"{ref}: content store artifact failed integrity "
                        f"verification — {detail}",
                    )
                verified.append(ref)
                continue
            path = paths.get(name)
            if not path:
                return Check(
                    0, "", "", FAIL,
                    f"{name}: no machine-local path configured — "
                    f"mlforge dataset add {name} <PATH>",
                )
            try:
                identity, manifest = recompute_identity(
                    path, name, version, schema=reg.get("schema")
                )
            except (ValidationBlock, NotFound, OSError) as exc:
                detail = exc.message if isinstance(exc, ValidationBlock) else str(exc)
                return Check(0, "", "", FAIL, f"{ref}: cannot verify: {detail}")
            if identity != reg.get("identity"):
                return Check(
                    0, "", "", FAIL,
                    f"{ref}: identity mismatch — content at {path} hashes to "
                    f"{identity}, registration says {reg.get('identity')} "
                    f"(re-register with --force if intended)",
                )
            files_hashed += manifest.file_count
            verified.append(ref)
        return Check(
            0, "", "", PASS,
            f"{len(verified)} dataset(s) verified — {files_hashed} files hashed",
            {"datasets": verified, "files_hashed": files_hashed},
        )

    return _p


def make_gate(providers: Mapping[str, Provider], **kwargs: Any) -> ValidationGate:
    """Convenience: providers for the listed steps, unverifiable elsewhere."""
    merged = dict(providers)
    return ValidationGate(merged, **kwargs)


# ---------------------------------------------------------------------------
# Execution planner providers — gate steps 10–13 (12 §13, §18; build 9)
# ---------------------------------------------------------------------------


def _plan_for(ctx: GateContext) -> Any:
    """Plan from ctx.facts, building it once per gate run (capabilities
    are measured once and shared — 12 §13.1 negotiation happens here)."""
    plan = ctx.facts.get("execution_plan")
    if plan is None:
        from mlforge.planner import build_plan, detect_capabilities, load_runtime

        caps = ctx.facts.get("capabilities")
        if caps is None:
            caps = detect_capabilities()  # ValidationBlock ⇒ engine FAIL
            ctx.facts["capabilities"] = caps
        plan = build_plan(ctx.run_spec, caps, runtime=load_runtime(ctx.run_dir))
        ctx.facts["execution_plan"] = plan
    return plan


def provide_plan() -> Provider:
    """Step 10: generate the execution plan (feasibility solver)."""
    def _p(ctx: GateContext) -> Check:
        plan = _plan_for(ctx)
        detail = plan.summary_line() + f" · precision {plan.precision_effective}"
        if plan.precision_fallback_used:
            detail += (f" (from {plan.precision_preferred} — PORTABLE, "
                       f"12 §14)")
        elif plan.portable_required:
            detail += " (CPU-only — PORTABLE, 12 §21)"
        detail += f" · {len(plan.solutions)} feasible solution(s)"
        # portable_required is an explicit PORTABLE signal (never EXACT)
        data = {"exact_compatible": False} if plan.portable_required else {}
        return Check(0, "", "", PASS, detail, data)
    return _p


def provide_global_batch() -> Provider:
    """Step 11: micro × accum × world == frozen global_batch (12 §8.3)."""
    def _p(ctx: GateContext) -> Check:
        plan = _plan_for(ctx)
        frozen = int(ctx.run_spec.semantic["global_batch"])
        if plan.product != frozen or plan.global_batch != frozen:
            return Check(
                0, "", "", FAIL,
                f"plan {plan.micro_batch}×{plan.grad_accum}×"
                f"{plan.world_size} = {plan.product} != frozen "
                f"global_batch {frozen}",
            )
        # persisted artifact (resume path) must state the same invariant
        artifact = ctx.run_dir / PLAN_FILENAME
        if artifact.is_file():
            from mlforge.planner import ExecutionPlan

            on_disk = ExecutionPlan.read(artifact)  # schema check inside
            if on_disk.global_batch != frozen or on_disk.product != frozen:
                return Check(
                    0, "", "", FAIL,
                    f"persisted plan artifact violates the invariant: "
                    f"{on_disk.product} != frozen global_batch {frozen}",
                )
        return Check(
            0, "", "", PASS,
            f"{plan.micro_batch} × {plan.grad_accum} × "
            f"{plan.world_size} = {frozen} preserved",
            {"global_batch": frozen},
        )
    return _p


def provide_precision() -> Provider:
    """Step 12: precision policy supported (12 §14 — fallback ⇒ PORTABLE)."""
    def _p(ctx: GateContext) -> Check:
        plan = _plan_for(ctx)
        if not plan.precision_fallback_used \
                and plan.precision_effective != plan.precision_preferred:
            return Check(0, "", "", FAIL,
                         f"effective precision {plan.precision_effective!r} "
                         f"is not the preferred "
                         f"{plan.precision_preferred!r} without a recorded "
                         f"fallback")
        if plan.precision_fallback_used:
            return Check(
                0, "", "", PASS,
                f"{plan.precision_preferred} → {plan.precision_effective} "
                f"fallback — execution_mode PORTABLE (12 §14)",
                {"exact_compatible": False},
            )
        return Check(0, "", "", PASS,
                     f"{plan.precision_effective} supported by capabilities")
    return _p


def provide_topology() -> Provider:
    """Step 13: distributed topology compatible (12 §8.5) — the plan's
    world_size must fit the measured device set, backend must match."""
    def _p(ctx: GateContext) -> Check:
        from mlforge.planner import Capabilities

        plan = _plan_for(ctx)
        topo = plan.topology
        caps = Capabilities.from_dict(plan.capabilities)
        if topo.get("world_size") != plan.world_size:
            return Check(0, "", "", FAIL,
                         f"topology world_size {topo.get('world_size')} != "
                         f"plan world_size {plan.world_size}")
        if caps.gpu_count == 0:
            if plan.world_size != 1 or topo.get("collective_backend") != "gloo":
                return Check(0, "", "", FAIL,
                             "CPU-only plan must use world_size 1 + gloo")
        elif plan.world_size > caps.gpu_count:
            return Check(0, "", "", FAIL,
                         f"world_size {plan.world_size} exceeds measured "
                         f"gpu_count {caps.gpu_count}")
        elif topo.get("collective_backend") != "nccl":
            return Check(0, "", "", FAIL,
                         "GPU plan must use the nccl collective backend")
        devices = (f"1 node × {topo['gpus_per_node']} GPU"
                   if caps.gpu_count else "CPU-only")
        return Check(
            0, "", "", PASS,
            f"{devices} (world {plan.world_size}, "
            f"{topo['collective_backend']}/{topo['network_fabric']})",
            {"topology": topo},
        )
    return _p


# ---------------------------------------------------------------------------
# Host/runtime providers — gate steps 8/9/14 (12 §13, §11.2; build 3)
# ---------------------------------------------------------------------------


def provide_hardware() -> Provider:
    """Step 9: hardware capabilities measured live (12 §13.1 — never
    guessed; broken detection is ValidationBlock ⇒ engine FAIL)."""
    def _p(ctx: GateContext) -> Check:
        from mlforge.planner import detect_capabilities

        caps = detect_capabilities()
        parts = [f"{caps.cpu_count} CPU", f"{caps.ram_bytes // (1 << 30)} GiB RAM"]
        if caps.gpus:
            vram = max(g.vram_total_mb for g in caps.gpus)
            parts.append(
                f"{caps.gpu_count} GPU ({caps.gpus[0].name}, {vram} MiB)"
            )
        parts.append(f"source {caps.source}")
        return Check(
            0, "", "", PASS, "detected: " + " · ".join(parts),
            {"gpu_count": caps.gpu_count, "cpu_count": caps.cpu_count,
             "interconnect": caps.interconnect, "source": caps.source},
        )
    return _p


def provide_driver() -> Provider:
    """Step 8: driver compatibility (12 §18).

    GPUs are only ever measured *through* a responding driver
    (detect_capabilities shells out to nvidia-smi and raises when it is
    present but broken), so a successful GPU detection is itself the
    driver-responds evidence; a CPU-only host has nothing to drive."""
    def _p(ctx: GateContext) -> Check:
        from mlforge.planner import detect_capabilities

        caps = detect_capabilities()
        if not caps.gpus:
            return Check(
                0, "", "", PASS,
                "no CUDA device — driver compatibility not applicable "
                "(CPU execution)",
                {"gpu_count": 0},
            )
        return Check(
            0, "", "", PASS,
            f"driver responding — nvidia-smi measured {caps.gpu_count} "
            "GPU(s) (12 §13.1)",
            {"gpu_count": caps.gpu_count, "source": caps.source},
        )
    return _p


def provide_checkpoint() -> Provider:
    """Step 14: newest-valid predicate (12 §11.2).

    TRAIN (fresh): no checkpoint to restore — PASS with that fact.
    RESUME: the store must yield a verified newest-valid checkpoint or
    the run does not resume (fail-closed)."""
    def _p(ctx: GateContext) -> Check:
        from mlforge.runtime.checkpoints import CheckpointStore

        store = CheckpointStore(ctx.run_dir)
        selection = store.newest_valid()
        if getattr(ctx, "flow", "TRAIN") == "TRAIN":
            return Check(
                0, "", "", PASS,
                "fresh run — no checkpoint to restore (12 §11.2)",
                {"flow": "TRAIN"},
            )
        if selection.selected is None:
            skips = len(selection.skips)
            detail = "no newest-valid checkpoint — resume cannot proceed"
            if skips:
                detail += f" ({skips} candidate(s) failed verification)"
            return Check(0, "", "", FAIL, detail,
                         {"attempted_newest": selection.attempted_newest})
        cand = selection.selected
        step = (f", global_step {cand.global_step}"
                if cand.global_step is not None else "")
        return Check(
            0, "", "", PASS,
            f"ckpt-{cand.ordinal:06d} verified (newest-valid{step})",
            {"resume_ordinal": cand.ordinal, "skips": len(selection.skips)},
        )
    return _p


# ---------------------------------------------------------------------------
# Artifact providers — gate steps 5/6 (12 §6.4 EXACT-required identity)
# ---------------------------------------------------------------------------


def provide_transform(root: str | Path) -> Provider:
    """Step 5: transform identity for every dataset in the run (12 §18).

    The prepared artifact's registered content hash IS the transform
    identity anchor: the content store must still hold the exact bytes
    (verify=True re-hashes) and the artifact must carry a registered
    transform name. A raw, un-prepared dataset has no transform artifact
    ⇒ FAIL with the prepare hint — EXACT-required, never a guess
    (12 §8.4: "artifact hash must match → else BLOCK")."""

    def _p(ctx: GateContext) -> Check:
        from mlforge.ingest.identity import parse_ref
        from mlforge.ingest.transforms import registry_names
        from mlforge.store import ContentStore

        spec = ctx.run_spec
        refs: list[str] = list(spec.train_datasets)
        if spec.val_dataset:
            refs.append(spec.val_dataset)
        store = ContentStore(Path(root) / "store")
        known = set(registry_names())
        verified: list[str] = []
        for ref in refs:
            name, _version = parse_ref(ref)
            reg_path = Path(root) / "datasets" / name / "identity.json"
            if not reg_path.is_file():
                return Check(0, "", "", FAIL,
                             f"{ref}: not registered — "
                             f"mlforge dataset add {name} <PATH>")
            try:
                reg = json.loads(reg_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return Check(0, "", "", FAIL,
                             f"{name}: registration unreadable ({exc})")
            identity = reg.get("identity")
            if not identity:
                return Check(0, "", "", FAIL,
                             f"{name}: no recorded identity — re-register")
            if not store.contains(identity):
                return Check(
                    0, "", "", FAIL,
                    f"{name}: raw dataset — no transform artifact recorded "
                    "(transform identity is EXACT-required, 12 §6.4/§8.4); "
                    "derive it first: mlforge prepare <MODEL>",
                )
            try:
                doc = json.loads(store.get_bytes(identity, verify=True))
            except ValidationBlock as exc:
                return Check(0, "", "", FAIL,
                             f"{name}: transform artifact failed integrity "
                             f"verification — {exc.message}")
            except Exception as exc:
                return Check(0, "", "", FAIL,
                             f"{name}: transform artifact unreadable: {exc}")
            transform = doc.get("transform")
            if not isinstance(transform, dict) or not transform.get("name"):
                return Check(
                    0, "", "", FAIL,
                    f"{name}: artifact carries no transform identity — "
                    "re-run `mlforge prepare <MODEL>`",
                )
            t_name = str(transform["name"])
            if t_name not in known:
                return Check(
                    0, "", "", FAIL,
                    f"{name}: transform {t_name!r} not in the closed "
                    f"registry ({', '.join(sorted(known)) or 'empty'}) — "
                    "unverifiable transform code",
                )
            verified.append(f"{name}→{t_name}")
        return Check(
            0, "", "", PASS,
            f"{len(verified)} transformed dataset(s) verified — "
            + ", ".join(verified),
            {"datasets": verified},
        )

    return _p


def provide_model(root: str | Path) -> Provider:
    """Step 6: model architecture + base weights hash (12 §18).

    * from scratch (train/fork): nothing external to verify — the
      architecture is defined by the source captured at step 3 and the
      frozen `run_spec.model`; no base weights exist to hash.
    * finetune (`lineage.origin == "finetune"`, `parent_model` = base
      ref): the base registry entry must be published AND its weights
      artifact must still verify — for run-published models the weights
      digest is the producing run's newest COMMIT marker (re-read +
      match); for imported packages the recorded digest is the authority.
    * retrain (`parent_model`, other origins): the parent model must
      exist and be published (fresh init needs the model identity, not
      weights)."""

    def _p(ctx: GateContext) -> Check:
        from mlforge.lineage import read_lineage

        try:
            lineage = read_lineage(ctx.run_dir)
        except ValidationBlock as exc:
            return Check(0, "", "", FAIL, f"lineage unreadable: {exc.message}")
        origin = str(lineage.get("origin") or "train")
        parent = lineage.get("parent_model")
        # Production writes finetune's base as parent_model (+ origin);
        # base_model is honored too if a writer ever persists it.
        needs_weights = origin == "finetune" or bool(lineage.get("base_model"))
        ref = lineage.get("base_model") or parent
        name = ctx.run_spec.model

        if not ref:
            return Check(
                0, "", "", PASS,
                f"train from scratch — model '{name}'; architecture from "
                "captured source (step 3), no base weights to hash",
                {"model": name},
            )

        ref = str(ref)
        # Lazy: workflow is fully loaded by the time the gate runs.
        from mlforge.workflow import WorkflowAPI

        wf = WorkflowAPI(Path(root))
        try:
            entry = wf.resolve_model(ref)
        except (NotFound, PreconditionFailed, ValidationBlock) as exc:
            return Check(
                0, "", "", FAIL,
                f"base model {ref} not in the registry — {exc}"
                if not isinstance(exc, ValidationBlock)
                else f"base model {ref}: {exc.message}",
            )
        state = entry.get("state")
        if state not in {"AVAILABLE", "EVALUATED", "EXPORTED", "DEPLOYED",
                         "USED_AS_FINE_TUNE_BASE"}:
            return Check(
                0, "", "", FAIL,
                f"base model {ref} is {state or 'unpublished'} — a base "
                "must be published (models appear when a run completes, "
                "13 §5.5)",
            )
        canonical = f"{entry.get('name')}:{entry.get('version')}"
        artifact = entry.get("artifact_hash")

        if not needs_weights:  # retrain — fresh init, identity is enough
            detail = f"parent model {canonical} available (state {state})"
            if artifact:
                detail += f" · recorded artifact {str(artifact)[:19]}…"
            return Check(0, "", "", PASS, detail,
                         {"model": canonical, "state": state})

        if not artifact:
            return Check(
                0, "", "", FAIL,
                f"base model {canonical} has no weights artifact — "
                "nothing to initialize from (12 §15.3)",
            )
        if (entry.get("origin") or "run") != "run":
            # Imported package: the weight bytes live outside the
            # workspace — the registry digest is the authority (12 §15.3).
            return Check(
                0, "", "", PASS,
                f"base weights digest recorded — imported {canonical}, "
                f"{str(artifact)[:19]}…",
                {"model": canonical, "artifact_hash": artifact},
            )
        src_run = entry.get("run_id")
        marker_found = False
        ckpt_root = Path(root) / "runs" / str(src_run) / "checkpoints"
        if ckpt_root.is_dir():
            for d in sorted(ckpt_root.iterdir()):
                marker = d / "COMMIT" if d.is_dir() else None
                if marker is not None and marker.is_file():
                    try:
                        if marker.read_text(encoding="utf-8").strip() == artifact:
                            marker_found = True
                            break
                    except OSError:
                        continue
        if not marker_found:
            return Check(
                0, "", "", FAIL,
                f"base weights {str(artifact)[:19]}… not found under run "
                f"{src_run} — the producing run's checkpoint COMMIT marker "
                "is missing or differs; finetune cannot initialize "
                "fail-closed",
            )
        return Check(
            0, "", "", PASS,
            f"base weights verified — COMMIT {str(artifact)[:19]}… "
            f"({canonical}, run {src_run})",
            {"model": canonical, "artifact_hash": artifact},
        )

    return _p
