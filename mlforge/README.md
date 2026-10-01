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
│   ├── cli/                  # status/inspect/events/store/validate/preflight/lease/train/resume/pause/stop/watch/hardware/dataset/prepare ✅
│   ├── yamlmini.py           # strict YAML subset (fail-closed)   ✅
│   ├── ingest/               # identity, paths, DAG, transforms, prepare ✅
│   ├── store/                # CAS + registry + GC                ✅
│   ├── validation/           # 19-step gate + preflight           ✅
│   ├── leases/               # run lease + gate providers 15–16   ✅
│   ├── commands/             # idempotency journal                ✅
│   ├── supervisor.py         # crash detection + spawn queue daemon  ✅
│   ├── runtime/              # checkpoints, worker, control, trainer ✅
│   ├── status/               # L1/L2/L3, watch, events --follow   ✅
│   └── planner/              # capability feasibility solver      ✅
└── tests/                    # specs as executable tests
```

## Build order (13 §11) — status

1. ✅ Workflow API + state machines (project/dataset/run/model)
2. ✅ Artifact registry + content store + run_spec canonical hashing
3. ✅ Validation gate + preflight (fail-closed core)
4. 🟡 CLI contract — `status` / `inspect` / `events` / `store gc` /
   `validate` / `preflight` / `lease status` / `lease break` /
   `dataset add|list|verify` / `prepare` work; other commands exit 4
   with `NOT_IMPLEMENTED` (never fake success)
5. ✅ Supervisor daemon + run leases + idempotency journal
6. ✅ Training runtime — transactional checkpoints (12 §11 write
   protocol / newest-valid predicate / verify / components), heartbeat
   writer, deterministic reconciliation (`reconcile_from_disk`, wired
   into `resume`), worker process (start contract, control-channel
   pause/stop, checkpoint loop), supervisor spawn queue + daemon
   (`state/pending/` → detached worker, 12 §12.4 delegation), and
   `train`/`resume`/`pause`/`stop` CLI (gate BLOCK train →
   FAILED[FORK_ONLY], resume → state unchanged; `--command-id` dedupe)
7. ✅ Status layer — read-only L1/L2 (`status [RUN] [-v]` incl. stale-
   heartbeat WARNING + FAILED disposition + stage-aware block), L3
   `hardware` (diagnostic telemetry), `watch [RUN]` (viewer only — keys
   write control intents, `q` quits the viewer), `events --follow`;
   worker publishes `state/heartbeat.json` (global_step/state) +
   `state/live.json` (transient metrics) per 13 §9.4
8. ✅ Ingestion/transform DAG — cryptographic dataset identity
   (every file hashed; counts are never identity, 12 §6.2), strict YAML
   config subset (`yamlmini`), machine-local paths at `$MLFORGE_HOME`
   vs portable `datasets.yaml` (12 §6.3), `ingestion.yaml` DAG
   (sources/depends_on, cycle detection), closed transform registry
   with code-hash identity + four-field cache key (12 §6.4),
   `dataset add|list|verify` (tamper → REJECTED, `--force`
   reregister), `prepare` (resolve → cache → transform → derived
   `<model>_prepared` REGISTERED→VERIFIED→PREPARED, `--command-id`
   dedupe), and the gate's step-4 builtin dataset provider
9. ✅ Execution planner — measured capabilities (`nvidia-smi` query,
   never VRAM tables; absent ⇒ determinate CPU-only set, broken ⇒
   BLOCK; unknown arch ⇒ fp32 only, 12 §13.1), feasibility solver
   (micro × accum × world == frozen global_batch s.t. VRAM/precision/
   model-min, deterministic pick: max world then max micro, 12 §13.2),
   `ExecutionPlan` artifact persisted on validation pass (identity-
   gated — unchanged plan never journals twice), gate builtins for
   steps 10–13 (`plan`/`global_batch`/`precision`/`topology`), disk
   estimate for step 16 now includes registered dataset bytes,
   execution segments (`segments/segment_<NNNN>.json` — hardware
   context recorded by the worker only AFTER preflight passes,
   12 §12.2), and `SHOW PLAN` in `train` before confirmation (13 §6.1)
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
* **Status is read-only** — it never writes state or transitions a run
  (13 §4.1); a stale RUNNING heartbeat is a WARNING, reconciled only by
  `resume` / supervisor crash detection. Never infer lifecycle from
  telemetry (13 §9.4).
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
* **Dataset identity is cryptographic** — every file's sha256 (sorted,
  POSIX-relative paths only; absolute paths never enter identity); sample
  counts are never identity. Symlinks and empty trees are refused —
  an identity we cannot walk deterministically is a guess (12 §6.2–§6.3).
* **Paths are configuration, never identity** — dataset paths live in
  `$MLFORGE_HOME` (default `~/.mlforge`), verified against registration
  on every gate/preflight/prepare; wrong path = mismatch, never a
  discovery scan (13 §10).
* **Transforms run only from a closed registry** — unknown transform
  ⇒ exit 1; code-hash + config-hash + input identities + env fingerprint
  form the cache key (12 §6.4); a transform that skips a source or
  emits 0 records is BLOCKed, never published.
* **Re-registration is explicit** — changed content requires
  `dataset add --force` (journaled `dataset_reregistered`); a tampered
  verified dataset becomes REJECTED, never silently reinterpreted
  (13 §5.2).
* **global_batch is input AND output** — the solver may only touch
  execution fields (micro/accum/world); `micro × accum × world` must
  equal the frozen semantic `global_batch` or gate step 11 FAILs
  (12 §8.3, §13.3: execution → semantic mutation is forbidden).
* **Capabilities are measured, never assumed** — VRAM comes from the
  device query; unknown compute capability supports nothing beyond
  fp32 (fail-closed); `nvidia-smi` present but broken ⇒ BLOCK, absent
  ⇒ a determinate CPU-only set (12 §13.1, §21).
* **Precision fallback and CPU runs are PORTABLE, never EXACT** —
  bf16→fp32 fallback or a world_size of 1 CPU records
  `exact_compatible: False` (12 §14, §21).
* **An infeasible plan BLOCKs with migration options** — "run on
  another machine / fork with a different global_batch / cancel"
  (13 §7), never a silent shrink of the batch; execution segments are
  created only after preflight passes, one per start (12 §12.2).

## Develop

```bash
cd mlforge
pip install -e ".[dev]"
pytest
mlforge --version
mlforge --root /path/to/workspace status
```
