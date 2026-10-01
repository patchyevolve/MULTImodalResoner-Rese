"""Deterministic reconciliation scan — 12_training_system.md §12.3.

    reconcile(run_folder):
        1. read all checkpoint directories
        2. filter to committed + hash-valid candidates (§11.2 predicate)
        3. select newest valid → resume_point
        4. read state.json + events.jsonl tail (projection, reporting only)
        5. compare: recorded state vs authority → consistent | INTERRUPTED
        6. write RECONCILED event: prior recorded state, derived state,
           resume_point, skipped checkpoints (with reasons)
        7. emit report to user before any resume

Invariants:
  * a deterministic function of on-disk state — any operator on any
    machine derives the same answer from the same run folder
  * idempotent — running twice produces identical results and no
    duplicate events
  * never mutates checkpoints or run_spec; never starts training
  * INTERRUPTED + no valid checkpoint → FAILED (FORK_ONLY), never a
    silent restart from step 0

This module owns STEPS 1–3 and the report shape; the Workflow API
(`reconcile_from_disk`) owns events/transitions (single write path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlforge.runtime.checkpoints import CheckpointStore, Selection


@dataclass(frozen=True)
class ReconcileReport:
    run_id: str
    prior_recorded_state: str
    authority_state: str
    derived_state: str
    selection: Selection
    consistent: bool
    reconciliation_id: str
    stored: bool = False  # True when replayed from an existing RECONCILED

    @property
    def resume_point(self) -> str | None:
        return self.selection.resume_point

    @property
    def skipped(self) -> tuple[dict[str, Any], ...]:
        return tuple(s.to_dict() for s in self.selection.skips)

    def lines(self) -> list[str]:
        out = [f"RECONCILIATION of {self.run_id}"]
        out.append(f"  recorded (projection): {self.prior_recorded_state}")
        out.append(f"  authority (journal):   {self.authority_state}")
        out.append(f"  derived:               {self.derived_state}"
                   + ("" if self.consistent else "  (projection stale — journal wins)"))
        out.extend(f"  {line}" for line in self.selection.report_lines())
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "prior_recorded_state": self.prior_recorded_state,
            "authority_state": self.authority_state,
            "derived_state": self.derived_state,
            "consistent": self.consistent,
            "reconciliation_id": self.reconciliation_id,
            "resume_point": self.resume_point,
            "skips": list(self.skipped),
            "no_valid_checkpoint": self.selection.no_valid_checkpoint,
            "stored": self.stored,
        }


def scan_checkpoints(run_dir: str | Path) -> Selection:
    """§12.3 steps 1–3: read generations, filter by the §11.2 predicate,
    select newest valid → resume_point."""
    return CheckpointStore(run_dir).newest_valid()


__all__ = ["ReconcileReport", "scan_checkpoints"]
