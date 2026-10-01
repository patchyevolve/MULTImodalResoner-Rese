"""Validation gate + preflight — build step 3 (fail-closed core).

Normative: 12_training_system.md §7 (BLOCK vs WARN, report, preflight),
§18 (19-step resume flow), §8 (EXACT/PORTABLE), §23.3 (revalidation).

API:
    ValidationGate(providers).run(GateContext)   -> ValidationReport
        16 normative steps (12 §18); missing provider = FAIL (fail-closed);
        first FAIL stops the run; EXACT only on explicit signals.
    Preflight(identity_providers, probes).run(PreflightContext) -> report
        host probes (python/platform/disk/atomic-rename/write/ram/gpu/
        driver/secrets) + identity re-verify via the gate's provider keys.
    scan_for_secrets(run_dir)                    -> findings (12 §7.1)
"""

from mlforge.validation.gate import (
    POST_GATE_ACTIONS,
    RESUME_GATE_STEPS,
    SEMANTIC_INVARIANTS,
    GateContext,
    GateStep,
    ValidationGate,
    make_gate,
    provide_dataset_identity,
    provide_fail,
    provide_global_batch,
    provide_pass,
    provide_plan,
    provide_precision,
    provide_topology,
    provide_unverifiable,
    scan_for_secrets,
)
from mlforge.validation.preflight import (
    HOST_PROBES,
    Preflight,
    PreflightContext,
    Probe,
)
from mlforge.validation.report import FAIL, PASS, WARN, Check, ValidationReport

__all__ = [
    "ValidationGate",
    "GateContext",
    "GateStep",
    "RESUME_GATE_STEPS",
    "POST_GATE_ACTIONS",
    "SEMANTIC_INVARIANTS",
    "Preflight",
    "PreflightContext",
    "Probe",
    "HOST_PROBES",
    "Check",
    "ValidationReport",
    "PASS",
    "WARN",
    "FAIL",
    "make_gate",
    "provide_pass",
    "provide_fail",
    "provide_unverifiable",
    "provide_dataset_identity",
    "provide_plan",
    "provide_global_batch",
    "provide_precision",
    "provide_topology",
    "scan_for_secrets",
]
