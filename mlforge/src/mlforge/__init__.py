"""MLForge — a hardware-agnostic, project-agnostic ML training system.

Ground truth (normative specifications, in reading order):

    10_training_plan/00_master_plan.md             what/why
    10_training_plan/12_training_system.md         the machinery (architecture v1.0)
    10_training_plan/13_product_specification.md    how the human drives it (product v1.0)

Build order follows 13_product_specification.md §11:

    1. Workflow API + state machines          ← this package core (done)
    2. Artifact registry + content store      (mlforge.store — pending)
    3. Validation gate + preflight            (mlforge.validation — pending)
    4. CLI contract                           (mlforge.cli — minimal)
    5. Supervisor + leases + idempotency      (mlforge.leases / mlforge.commands — pending)
    6. Training runtime + checkpoints         (mlforge.runtime — pending)
    7. Status layer                           (mlforge.status — pending)
    8. Ingestion/transform DAG                (pending)
    9. Execution planner                      (mlforge.planner — pending)
   10. Resume/retrain/finetune flows + lineage (partial: via workflow)
   11. Evaluate/compare/infer/export/package  (pending)
   12. TUI/GUI                                (pending)
"""

__version__ = "0.1.0"

from mlforge.errors import (  # noqa: F401
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_PRECONDITION,
    EXIT_RUNTIME,
    EXIT_VALIDATION_BLOCK,
)
