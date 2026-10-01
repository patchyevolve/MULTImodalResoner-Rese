# MLForge

A hardware-agnostic, project-agnostic ML training system: immutable experiment
identity, cryptographic artifact identities, fail-closed validation, explicit
resume/fork semantics, crash reconciliation, and portable run folders.

## Ground truth (normative)

Implementation follows these two documents exactly; where code and docs
disagree, the docs win until amended:

| Doc | Role |
|---|---|
| `../10_training_plan/12_training_system.md` | **Architecture v1.0** — the machinery (identity model, validation gate, checkpoints, leases, reconciliation, planner) |
| `../10_training_plan/13_product_specification.md` | **Product v1.0** — CLI contract, state machines, sequence diagrams, failure matrix, status plane |
| `../10_training_plan/00_master_plan.md` | What/why, training strategy |

## Layout

```
mlforge/
├── pyproject.toml            # zero runtime deps (stdlib-only core)
├── src/mlforge/
│   ├── errors.py             # exit codes 0-4 + error hierarchy   ✅
│   ├── ids.py                # run_/model_/cmd_ ULID ids          ✅
│   ├── hashing.py            # canonical JSON + sha256 identity   ✅
│   ├── journal.py            # append-only events (authority)     ✅
│   ├── states.py             # lifecycle states + FAILED disposition ✅
│   ├── machine.py            # 4 transition tables (13 §5)        ✅
│   ├── run_spec.py           # immutable semantic identity        ✅
│   ├── workflow.py           # Workflow API facade (13 §1)        ✅
│   ├── cli/                  # status/inspect/events (+stubs)     ✅ minimal
│   ├── store/                # CAS + registry + GC                ⛔ step 2
│   ├── validation/           # 19-step gate + preflight           ⛔ step 3
│   ├── leases/               # run/execution leases               ⛔ step 5
│   ├── commands/             # idempotency journal                ⛔ step 5
│   ├── runtime/              # checkpoints, heartbeat, supervisor  ⛔ step 6
│   ├── status/               # L1/L2/L3, watch                    ⛔ step 7
│   └── planner/              # capability feasibility solver      ⛔ step 9
└── tests/                    # specs as executable tests
```

## Build order (13 §11) — status

1. ✅ Workflow API + state machines (project/dataset/run/model)
2. ⛔ Artifact registry + content store + run_spec canonical hashing *(hashing done, store pending)*
3. ⛔ Validation gate + preflight (fail-closed core)
4. 🟡 CLI contract — `status` / `inspect` / `events` work; other commands exit 4 with `NOT_IMPLEMENTED` (never fake success)
5. ⛔ Supervisor daemon + run leases + idempotency journal
6. ⛔ Training runtime + transactional checkpoints + heartbeat + reconciliation scan
7. ⛔ Status layer (read-only L1/L2/L3)
8. ⛔ Ingestion/transform DAG
9. ⛔ Execution planner
10. ⛔ Resume/retrain/finetune flows + lineage DAG
11. ⛔ Evaluate/compare/infer/export/package
12. ⛔ TUI/GUI over the same Workflow API

## Invariants already enforced in code

* **FAILED has one meaning** — a recorded `failure.recovery` disposition
  (`RESUME` | `FORK_ONLY`); resume on `FORK_ONLY` exits 3
  (`NO_VALID_CONTINUATION`), never silently restarts (13 §5.3).
* **Automatic continuation is NO** for every non-`RUNNING` state; the system
  never transitions a run on its own.
* **status.json is a projection** — delete it and the state rebuilds from the
  event journal (12 §16.1).
* **run_spec is immutable** — written once (atomic rename); overwrite attempt
  → `VALIDATION_BLOCK`. Semantic change = fork = new run.
* **Crash ≠ pause** — crash from any live state → `INTERRUPTED`;
  reconciliation never starts training (12 §12.3).
* **Illegal transitions change nothing** and list the legal actions (exit 3).

## Develop

```bash
cd mlforge
pip install -e ".[dev]"
pytest
mlforge --version
mlforge --root /path/to/workspace status
```
