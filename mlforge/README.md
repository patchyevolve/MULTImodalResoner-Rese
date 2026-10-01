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
 │   ├── cli/                  # status/inspect/events/store/validate/preflight/lease/train/resume/pause/stop ✅
│   ├── store/                # CAS + registry + GC                ✅
│   ├── validation/           # 19-step gate + preflight           ✅
│   ├── leases/               # run lease + gate providers 15–16   ✅
│   ├── commands/             # idempotency journal                ✅
 │   ├── supervisor.py         # crash detection + spawn queue daemon  ✅
 │   ├── runtime/              # checkpoints, worker, control, trainer ✅
│   ├── status/               # L1/L2/L3, watch                    ⛔ step 7
│   └── planner/              # capability feasibility solver      ⛔ step 9
└── tests/                    # specs as executable tests
```

## Build order (13 §11) — status

1. ✅ Workflow API + state machines (project/dataset/run/model)
2. ✅ Artifact registry + content store + run_spec canonical hashing
3. ✅ Validation gate + preflight (fail-closed core)
4. 🟡 CLI contract — `status` / `inspect` / `events` / `store gc` /
   `validate` / `preflight` / `lease status` / `lease break` work; other
   commands exit 4 with `NOT_IMPLEMENTED` (never fake success)
5. ✅ Supervisor daemon + run leases + idempotency journal
6. ✅ Training runtime — transactional checkpoints (12 §11 write
   protocol / newest-valid predicate / verify / components), heartbeat
   writer, deterministic reconciliation (`reconcile_from_disk`, wired
   into `resume`), worker process (start contract, control-channel
   pause/stop, checkpoint loop), supervisor spawn queue + daemon
   (`state/pending/` → detached worker, 12 §12.4 delegation), and
   `train`/`resume`/`pause`/`stop` CLI (gate BLOCK train →
   FAILED[FORK_ONLY], resume → state unchanged; `--command-id` dedupe)
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
* **GC is reachability-only** — never age/size; run-folder `refs.json` are
  the roots (portable), grace protects fresh orphans, active run leases
  block execution-mode GC outright (12 §6.1).
* **Blobs are atomic and verified** — staging → fsync → rename; identity
  claimed at `put_file` must match (mismatch → BLOCK); corruption detected
  by `verify()`, never trusted.
* **Registry registration is transactional** — entry written to temp,
  renamed to commit; crash orphans swept by `recover()`.
* **Unverifiable == failed verification** — a missing provider, a throwing
  check, or a missing probe records FAIL, never SKIP (12 §7.1: "probably
  compatible does not exist"). First FAIL stops the run, so the
  side-effect lease step (15) never executes after an earlier failure.
* **Gate BLOCK has two distinct behaviors** — train path:
  CREATED → VALIDATING → FAILED[FORK_ONLY], never READY; resume path:
  **no state change** ("No changes were made" — 13 §7 failure matrix:
  Stays PAUSED / stays FAILED), journaled as `validation_blocked`.
* **EXACT continuation is opt-in, PORTABLE is the default** — EXACT only
  when every check reports `exact_compatible: True` (12 §8).
* **Disk-space check is worst-case and mandatory** —
  `2×checkpoint + dataset_cache + logs + safety_margin`; free < required
  ⇒ BLOCK, never "train until disk fills" (12 §7.3).
* **Preflight is report-only until the runtime exists** — it never moves
  a run to RUNNING (only after PREFLIGHT PASSED does training proceed).
* **Single-writer run lease** — `runs/<id>/.lease` acquired atomically
  (tmp → fsync → link); second resume → exit 3 `RUN_ALREADY_EXECUTING`;
  stale heartbeat → **SUSPECT, never auto-free** (stale ≠ free); break
  requires `--force --yes` and always journals `LEASE_BROKEN` (12 §23).
* **Command idempotency** — duplicate `command_id` of a success returns
  the original result (never a second run) and journals
  `COMMAND_DEDUPED`; in flight → exit 3; failures are retryable (13 §4.4).
* **Supervisor detects crashes, never continues runs** — expired/missing
  heartbeat → INTERRUPTED; it never resumes, reconciles, or starts
  anything (13 §1 automatic continuation = NO).

## Develop

```bash
cd mlforge
pip install -e ".[dev]"
pytest
mlforge --version
mlforge --root /path/to/workspace status
```
