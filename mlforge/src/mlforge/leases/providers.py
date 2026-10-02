"""Gate providers for steps 15–16 (12 §18, §23.3).

    15. ACQUIRE RUN LEASE — fail → BLOCK, no start (RunAlreadyExecuting
        propagates as exit 3, not as a gate FAIL: it is a concurrency
        precondition, not an invariant violation — 13 §7 matrix row
        "RESUME | execution lease held → BLOCK RUN_ALREADY_EXECUTING").
    16. REVALIDATE volatile subset — dataset identity · disk headroom ·
        lease ownership (things that can change between validation and
        start; immutable hashes are NOT re-checked, 12 §23.3).
"""

from __future__ import annotations

import shutil

from mlforge.leases.run_lease import LeaseState, RunLeaseManager
from mlforge.validation.gate import Check, GateContext, Provider
from mlforge.validation.report import FAIL, PASS


def provide_run_lease(leases: RunLeaseManager) -> Provider:
    """Step 15: acquire the single-writer run lease for this gate run.

    The session token lands in `ctx.facts["session_token"]` — later steps
    (resume, renew, release) must present it to prove ownership."""

    def _p(ctx: GateContext) -> Check:
        info = leases.acquire(ctx.run_id)  # raises RunAlreadyExecuting (exit 3)
        ctx.facts["session_token"] = info.session_token
        ctx.facts["lease"] = info.to_dict()
        return Check(
            15, "lease", "Acquire run lease (single-writer)", PASS,
            f"held by {info.holder}",
            {"session_token": info.session_token},
        )

    return _p


def provide_revalidation(
    leases: RunLeaseManager,
    *,
    dataset_provider: Provider | None = None,
) -> Provider:
    """Step 16: revalidate the volatile subset (12 §23.3).

    All three sub-checks are fail-closed:
      * lease ownership — the gate's token must match the live lease
      * disk headroom   — `ctx.facts["required_disk_bytes"]` (set by
        preflight/planner) vs current free space; unknown requirement ⇒ FAIL
      * dataset identity — machine-local paths may change out-of-band;
        a missing re-verification provider ⇒ FAIL (unverifiable)
    """

    def _p(ctx: GateContext) -> Check:
        # 1) lease ownership (still ours? could have been broken mid-gate)
        token = ctx.facts.get("session_token")
        if not token:
            return Check(16, "revalidate", "Revalidate volatile subset", FAIL,
                         "no session token in gate context — step 15 (lease) "
                         "must run before revalidation (fail-closed)")
        info = leases.status(ctx.run_id)
        if info.state != LeaseState.HELD or info.session_token != token:
            return Check(16, "revalidate", "Revalidate volatile subset", FAIL,
                         f"lease ownership lost: now {info.state.value} "
                         f"held by {info.holder} (fail-closed)")

        # 2) disk headroom (volatile — another job may have filled the disk)
        required = ctx.facts.get("required_disk_bytes")
        if not isinstance(required, (int, float)) or required <= 0:
            return Check(16, "revalidate", "Revalidate volatile subset", FAIL,
                         "required_disk_bytes unknown in gate facts — preflight/"
                         "planner must record it (fail-closed)")
        free = shutil.disk_usage(ctx.root).free
        if free < required:
            return Check(
                16, "revalidate", "Revalidate volatile subset", FAIL,
                f"disk headroom shrank: need {required} B, free {free} B "
                "(free disk, or lower runtime.checkpoint_bytes / "
                "runtime.log_bytes / runtime.safety_margin_bytes in the "
                "train config — 12 §7.3)",
                {"required": required, "free": free},
            )

        # 3) dataset identity (machine-local path is mutable by design)
        if dataset_provider is None:
            return Check(16, "revalidate", "Revalidate volatile subset", FAIL,
                         "unverifiable: no dataset re-verification provider "
                         "(fail-closed)")
        result = dataset_provider(ctx)
        if result is None or result.verdict == FAIL:
            detail = result.detail if result else "no verdict (fail-closed)"
            return Check(16, "revalidate", "Revalidate volatile subset", FAIL,
                         f"dataset identity changed since validation: {detail}")

        return Check(16, "revalidate", "Revalidate volatile subset", PASS,
                     f"lease owned; disk headroom {free} B >= {required} B; "
                     "dataset identity unchanged",
                     {"free": free, "required": required})

    return _p


__all__ = ["provide_run_lease", "provide_revalidation"]
