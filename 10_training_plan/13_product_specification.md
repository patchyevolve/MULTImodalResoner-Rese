# MLForge — Product Specification (v1.0)

> The product-level blueprint for MLForge: user mental model, exact CLI contract, state machines, sequence diagrams for every operation, failure behavior for every transition, and the status/telemetry plane. This is the document to implement against. Architecture (invariants, artifacts, validation) lives in `12_training_system.md`; this defines **what the product does and how the user drives it.**

---

## 1. Two-Layer Architecture

```text
USER WORKFLOW        ← what the human intends (train, resume, fine-tune...)
    ↓
MLForge command / UI ← CLI, TUI, or GUI (one shared Workflow API)
    ↓
WORKFLOW ORCHESTRATOR ← translates intent into controlled state transitions
    ↓
validation → planning → execution → artifact   ← internal machinery (v1.0 architecture)
    ↓
immutable lineage
```

**Design rules:**

1. The user interacts with **projects, datasets, models, runs, and actions** — never with checkpoints, environments, CUDA, dataset hashes, optimizer state, or RNG.
2. Three interfaces (CLI, TUI, GUI) call **one Workflow API**. Never implement business logic per interface.
3. `status`/`watch` is a **read-only observation layer**, never authoritative for correctness.

---

## 2. User Mental Model

### 2.1 Primary Objects

```text
PROJECT
   ├── DATASETS
   ├── MODEL DEFINITIONS
   ├── RUNS
   └── MODELS
```

### 2.2 The Critical Distinction

```text
RUN ≠ MODEL

RUN    = an experiment / training process (transient, stateful)
MODEL  = a resulting immutable artifact (permanent, content-addressed)
```

A run produces a model. A model never mutates — retrain/fine-tune create new runs, which create new models.

### 2.3 Actions

```text
PREPARE · TRAIN · RESUME · PAUSE · STOP · RETRAIN · FINE-TUNE · EVALUATE · INFER · EXPORT · COMPARE
```

Each maps to exactly one command. **The command expresses intent** — no generic "what do you want to do?" menu (safer for automation and scripting).

---

## 3. Master Lifecycle

```text
                    ┌──────────────┐
                    │    PROJECT   │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ CONFIGURE    │
                    │ DATA + MODEL │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │   PREPARE    │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │    TRAIN     │
                    └──────┬───────┘
                           ▼
                       RUN #1
              ┌────────┼───────────┬────────────┐
              ▼        ▼           ▼            ▼
           RESUME   EVALUATE      PAUSE        STOP
              │        ▼        (parked,      (resume
              │     REPORT      resumable)     spent)
               ▼        │           │            │
            COMPLETE ◄──┘           └──resume──► ▼
               ▼                          (new run via RETRAIN)
           MODEL #1
      ┌──────┼───────────────┬────────────┐
      ▼      ▼               ▼            ▼
    INFER  EXPORT         RETRAIN      FINE-TUNE
                            │            │
                            ▼            ▼
                         RUN #2       RUN #3
                            │            │
                            ▼            ▼
                         MODEL #2     MODEL #3
                            └──────┬─────┘
                                   ▼
                                EVALUATE → COMPARE
```

**Retrain is deliberately NOT continuation:**

```text
OLD MODEL ──X── (not continued)

PROJECT + DATA + CONFIG → NEW TRAIN RUN → NEW MODEL
```

**Underneath every user action, the same machinery:**

```text
USER INTENT → WORKFLOW → RESOLVE ARTIFACTS → VALIDATE EVERYTHING
    → PLAN EXECUTION → CONFIRM IF SEMANTIC CHANGE → CREATE/RESUME RUN
    → EXECUTE → ATOMIC CHECKPOINTS → MODEL ARTIFACT → LINEAGE
```

---

## 4. CLI Contract

### 4.1 Command Reference

```text
PROJECT
────────────────────────────────────────────────────────────
mlforge init <name>                     Create project
mlforge status [RUN]                    Overview (instant, read-only)
mlforge inspect <OBJECT>                Detailed report
mlforge watch [RUN]                     Live dashboard (TUI)

DATA
────────────────────────────────────────────────────────────
mlforge dataset add <ID> <PATH>         Register + verify identity
mlforge dataset list                    All registered datasets
mlforge dataset verify <ID>             Re-verify content hash
mlforge prepare <MODEL_DEF>             Transform raw → model-ready

TRAINING
────────────────────────────────────────────────────────────
mlforge train [--config F] [--attach]   Start new run (from scratch)
mlforge resume <RUN>                    Continue paused/interrupted/FAILED(RESUME) run
mlforge pause <RUN>                     Graceful pause → PAUSED (resumable)
mlforge stop <RUN>                      Graceful stop → STOPPED (no resume)
mlforge retrain <MODEL> [--dataset D]   New run, fresh init
mlforge finetune <MODEL> [--dataset D]  New run, from model weights

LEASES (concurrency — architecture §23)
────────────────────────────────────────────────────────────
mlforge lease status <RUN>              Who holds the run lease
mlforge lease break <RUN> --force       Break stale lease (--yes, logged)

MODEL
────────────────────────────────────────────────────────────
mlforge model list                      All models
mlforge model inspect <MODEL>           Model detail + lineage
mlforge model import <PATH>             Import external model

EVALUATION
────────────────────────────────────────────────────────────
mlforge evaluate <MODEL> --dataset D    Produce evaluation artifact
mlforge compare <MODEL>...              Side-by-side comparison

INFERENCE
────────────────────────────────────────────────────────────
mlforge infer <MODEL> <INPUT>           One-shot inference
mlforge serve <MODEL>                   Start model server

ARTIFACT
────────────────────────────────────────────────────────────
mlforge export <MODEL> --format F       ONNX / TensorRT / etc.
mlforge validate <RUN|MODEL>            Run validation gate
mlforge package <MODEL>                 Build inference bundle
mlforge store gc [--execute]            Collect unreachable artifacts (dry-run
                                         default; blocked by active leases)

OBSERVABILITY
────────────────────────────────────────────────────────────
mlforge status [RUN] [--verbose]        Layered status (L1/L2)
mlforge hardware                        Layer 3 telemetry
mlforge events <RUN> [--follow]         Structured event stream
```

### 4.2 Command Behavior Rules

| Rule | Behavior |
|---|---|
| **Async by default** | `mlforge train` returns immediately with run ID; training runs as background process |
| **`--attach`** | Start + attach live viewer; terminal death does NOT kill training |
| **Exit codes** | `0` success · `1` validation BLOCK · `2` not found · `3` precondition failed · `4` runtime error |
| **`--json`** | Every command supports machine-readable output for scripting |
| **`--yes`** | Skip confirmation prompts (for automation) |
| **Semantic changes** | Always require confirmation unless `--yes` |
| **No interactive menu** | Intent is the command name, never a "what do you want?" prompt |
| **Idempotent by `--command-id`** | State-changing commands dedupe on a client-supplied id (§4.4) |
| **No auto-anything** | Nothing in this contract starts, restarts, or reconfigures training implicitly |

### 4.3 Object Reference Syntax

```text
RUN       run_01JABC...          or ./runs/run_01JABC...
MODEL     model://rf_detr_s:v2    or rf_detr_s:v2
DATASET   dataset://coco_2017:v1  or coco_2017:v1
EVAL      evaluation://eval_001
```

### 4.4 Command Idempotency (Safe Retries)

Networks and shells drop. Retrying a command must never duplicate work or create a second run:

```bash
$ mlforge train --config X.yaml --command-id 01JABC...
→ started run_01JDEF...

# same command re-submitted (script retry, double-Enter, flaky SSH):
$ mlforge train --config X.yaml --command-id 01JABC...
→ command 01JABC... already completed → returning run_01JDEF...   (exit 0)
```

| Command | Dedupe key | Duplicate behavior |
|---|---|---|
| `train` / `retrain` / `finetune` | `command_id` | return original run id — **never a second run** |
| `evaluate` / `export` / `package` | `command_id` | return original result id — never recompute |
| `prepare` | `command_id` (+ content-addressed result) | return cached artifact identity |
| `pause` / `stop` / `resume` / `fork` | run state (natural key) | idempotent: already in target state → success, no-op, exit 0 |
| `status` / `inspect` / `list` / `events` | n/a | read-only, always safe |

Rules:
- No `--command-id` → CLI generates one per invocation (dedupe covers internal retries only).
- A `command_id` whose command **failed** may be retried (failures are retryable); a `command_id` that **succeeded** returns its original result and never re-executes.
- Duplicates are recorded in the command journal — visible in `mlforge events` as `COMMAND_DEDUPED`.

---

## 5. State Machines

### 5.1 Project

```text
CREATED → CONFIGURED → READY
```

### 5.2 Dataset

```text
REGISTERED → VERIFIED → PREPARED
                │
                └──(hash mismatch)→ REJECTED
```

### 5.3 Training Run (core)

```text
CREATED → VALIDATING → READY → RUNNING ──epochs complete──► COMPLETED
                │                     │
                │                     ├── pause ──► PAUSING ──► PAUSED ──resume──► VALIDATING
                │                     │                              │
                │                     │                              └── stop ──► STOPPING ──► STOPPED
                │                     │
                │                     ├── checkpoint cycle ──► CHECKPOINTING ──► RUNNING
                │                     │
                │                     ├── crash / power loss ──► INTERRUPTED ──► RECONCILING
                │                     │                              (reconciliation scan)
                │                     │                         ├─ valid ckpt ─► (resume) VALIDATING
                │                     │                         └─ none ───────► FAILED[FORK_ONLY]
                │                     │
                │                     └── runtime error ──► FAILED[RESUME | FORK_ONLY]
                │
                └── gate BLOCK ──► FAILED[FORK_ONLY]

FAILED[RESUME]     ── resume + cause resolved + 19-step gate pass ──► VALIDATING
FAILED[RESUME]     ── resume, cause unresolved ──► BLOCK (exit 1), stays FAILED
FAILED[FORK_ONLY]  ── resume ──► BLOCK (exit 3): fork / retrain only (new run)
FAILED | STOPPED | COMPLETED ── retrain / finetune ──► new run (this run never mutates)
```

#### Lifecycle Terms (Never Interchangeable)

| State | Cause | Automatic continuation | Explicit recovery action | Meaning to user |
|---|---|---|---|---|
| `PAUSED` | you ran `mlforge pause` | **NO** | `resume` — ordinary gate path | "your run, safely parked" |
| `INTERRUPTED` | crash / power loss / kill -9 | **NO** | reconciliation scan, then `resume` | "something died; here's what I found" |
| `FAILED` | error stopped the run | **NO** | **depends on `failure.recovery`** (below) | "stopped; cause is in `mlforge events`" |
| `STOPPED` | you ran `mlforge stop` | **NO** | `retrain` / `finetune` — new run only | "you ended this run" |
| `COMPLETED` | reached end condition | **NO** | `retrain` / `finetune` — new run only | "done" |

*Automatic continuation* is `NO` for **every** non-`RUNNING` state: nothing in the system transitions a run on its own. The *Explicit recovery action* column is the only legal way out of each state — for `FAILED`, that action is determined by the recorded `failure.recovery` disposition (never by the state name alone). Do not read `FAILED` as "no legal transition out"; read it as "the transition requires the recorded disposition + an explicit command + a fresh validation gate."

#### FAILED Carries a Recovery Disposition (Not Resumability-by-Guesswork)

`FAILED` is one state with an explicit, recorded disposition — never two contradictory meanings:

```yaml
failure:
    cause: "CUDA OOM at step 41200"        # always recorded
    recovery: RESUME | FORK_ONLY           # recorded when the state is entered
    valid_checkpoint: ckpt-000021           # or null
```

```text
failure.recovery =
    FORK_ONLY   if NO valid checkpoint exists
             OR the cause is semantic (run_spec / invariant violation —
                 fixing it would change the experiment)
    RESUME      otherwise (valid checkpoint + cause resolvable without
                 semantic change: OOM, disk full, env unavailable, transient I/O)
```

- `FAILED + recovery=RESUME` → `mlforge resume` re-runs the full 19-step gate; if the cause is not actually fixed, the gate blocks again. Entry to `resume` never bypasses validation.
- `FAILED + recovery=FORK_ONLY` → `mlforge resume` exits `3` (precondition failed): `NO VALID CONTINUATION — use mlforge fork or retrain (new run).` Never silently starts from step 0.

**Confusable pairs:**
- `PAUSED` vs `STOPPED` — both user-caused; pause keeps resume, stop spends it.
- `FAILED` vs `INTERRUPTED` — both unexpected; interrupted *always* recovers through reconciliation (resume if any valid checkpoint), failed carries an explicit disposition set at failure time.
- `INTERRUPTED` vs "still running" — a stale `status.json` saying RUNNING after a crash is `INTERRUPTED` the moment the heartbeat expires; the system reconciles before showing you anything (§9.4).

#### Transitions

| From | Event | To |
|---|---|---|
| CREATED | start validation | VALIDATING |
| VALIDATING | all checks pass | READY |
| VALIDATING | any check fails | FAILED (blocked, never RUNNING; `recovery` disposition per §5.3) |
| READY | preflight pass | RUNNING |
| RUNNING | `pause` command | PAUSING |
| PAUSING | checkpoint committed | PAUSED |
| RUNNING | `stop` command | STOPPING |
| STOPPING | final checkpoint committed | STOPPED |
| RUNNING | checkpoint cycle | CHECKPOINTING → RUNNING |
| RUNNING | unrecoverable error | FAILED |
| RUNNING | epochs complete | COMPLETED |
| PAUSED | `resume` + validation pass | VALIDATING → READY → RUNNING |
| RUNNING (or any live state) | crash / power loss detected | **INTERRUPTED** (via heartbeat expiry) |
| INTERRUPTED | reconciliation scan | RECONCILING |
| RECONCILING | valid checkpoint found | VALIDATING → READY → RUNNING (after `resume`) |
| RECONCILING | no valid checkpoint | FAILED (`recovery: FORK_ONLY`) |
| VALIDATING / READY / RUNNING | failure recorded | FAILED with `recovery` disposition (§5.3) |
| FAILED (`recovery: RESUME`) | `resume` + cause resolved + gate pass | VALIDATING → READY → RUNNING |
| FAILED (`recovery: RESUME`) | `resume`, cause unresolved | BLOCK — stays FAILED, exit 1 |
| FAILED (`recovery: FORK_ONLY`) | `resume` attempted | BLOCK — exit 3, `NO VALID CONTINUATION` |
| FAILED / STOPPED / COMPLETED | `retrain` / `finetune` | **new run** (never mutates this one) |

**Resume IS a state transition. Retrain/fine-tune are NOT** — they create new runs.
**`stop` and `pause` are different commands** — the old behavior of `stop → PAUSED` was ambiguous and is removed.

### 5.4 Model

```text
CREATED → VALIDATED → AVAILABLE
                          ├── EVALUATED
                          ├── EXPORTED
                          ├── DEPLOYED
                          └── USED_AS_FINE_TUNE_BASE
```

Models are immutable. There is no "RETRAIN" state on a model — retraining reads a model as `source` and produces a different model.

### 5.5 Retrain vs Fine-tune vs Resume (never ambiguous)

```text
RESUME     RUN ──────────────────► SAME RUN      (continue trajectory)
RETRAIN    MODEL ──source──► NEW RUN ──► NEW MODEL (fresh init, same config)
FINE-TUNE  MODEL ──parent──► NEW RUN ──► NEW MODEL (init from weights, new data/config)
```

---

## 6. Operation Sequence Diagrams

### 6.1 TRAIN

```text
User                CLI               Orchestrator           Validation        Execution
 │                   │                      │                     │                │
 ├──mlforge train──►│                      │                     │                │
 │                   ├──TRAIN REQUEST──────►│                     │                │
 │                   │                      ├──RESOLVE────────────►                │
 │                   │                      │◄──artifacts─────────┤                │
 │                   │                      ├──VALIDATE───────────►                │
 │                   │                      │◄──pass/fail─────────┤                │
 │                   │                      ├──PLAN───────────────►                │
 │                   │                      │◄──execution plan────┤                │
 │                   │◄──SHOW PLAN──────────┤                     │                │
 │◄──plan + confirm──┤                      │                     │                │
 ├──[Y]─────────────►│                      │                     │                │
 │                   ├──CONFIRM─────────────►│                     │                │
 │                   │                      ├──CREATE RUN (immutable run_spec)      │
 │                   │                      ├──LOCK code/data/env  │                │
 │                   │                      ├──INIT CHECKPOINT STORE                │
 │                   │                      ├──TRAIN──────────────────────────────►│
 │                   │◄──RUN ID─────────────┤                     │                │
 │◄──run_01J...──────┤                      │                     │                │
```

**Output (validated plan shown before start):**

```text
MLForge Training Setup

Model            rf_detr_s
Training data    coco_2017:v1, custom_clips:v1
Validation       coco_2017:v1/val

Configuration
  epochs 50 · global batch 32 · AdamW · lr 1e-4 · cosine · seed 42 · bf16

Execution plan
  micro batch 2 · accumulation 16 · GPUs 1 · global batch 32

Data             118,487 train / 5,000 val
Storage          1.4 GB estimated
Runtime          ~48 h estimated

Everything required has been validated.
Start training? [Y/n]
```

### 6.2 RESUME

`resume` accepts a run in `PAUSED`, `INTERRUPTED`, or `FAILED (recovery: RESUME)` state — `FAILED (FORK_ONLY)` is blocked with exit 3 (§5.3). For `INTERRUPTED`, reconciliation (12 §12.3) runs *before* validation: it derives the true state from checkpoint manifests, reports any skipped checkpoints, then the 19-step gate proceeds.

```text
User              CLI            Orchestrator         Validation Gate (19 steps)
 │                 │                   │                      │
 ├──mlforge resume─┤                   │                      │
 │                 ├──RESUME──────────►│                      │
 │                 │                   ├──VALIDATE────────────►
 │                 │                   │◄──PASS (mode=EXACT    │
 │                 │                   │    or PORTABLE)───────┤
 │                 │                   │  (any FAIL → STOP, no execution)
 │                 │                   ├──ACQUIRE RUN LEASE    │
 │                 │                   ├──REVALIDATE volatile  │
 │                 │                   ├──CREATE SEGMENT       │
 │                 │                   ├──RESTORE full state   │
 │                 │                   ├──RUNNING              │
 │                 │◄──mode + plan─────┤                      │
 │◄──report────────┤                   │                      │
```

**Blocked output:**

```text
RESUME BLOCKED

[FAIL] Dataset identity
  Expected: sha256:abc... (coco_2017 v1, 118,287 files)
  Found:    sha256:def... at /home/daksh/data/coco
  Reason:   different content (same count, different files)

No changes were made. Run state unchanged.
```

**Migration-impossible output:**

```text
RESUME BLOCKED

Required global batch: 32
Maximum achievable on this machine: 24

No valid hardware configuration can preserve
the training semantics of this run.

No changes were made.

Options:
  1. Use another machine
  2. Change training semantics → creates NEW run (mlforge fork)
  3. Cancel
```

### 6.3 RETRAIN

```text
User                 CLI                Orchestrator
 │                    │                       │
 ├──mlforge retrain──►│                       │
 │   rf_detr_s        ├──RETRAIN─────────────►│
 │                    │                       ├──load previous run_spec
 │                    │◄──SHOW DELTA──────────┤  (new run ID, fresh init,
 │◄──confirm──────────┤                       │   same config unless changed)
 ├──[Y]──────────────►│                       │
 │                    ├──CREATE NEW RUN───────┤  parent_run set
 │                    │                       ├──TRAIN (fresh weights)
 │                    │◄──run_01KXYZ──────────┤
```

```text
Retraining model: rf_detr_s

This will create a NEW training run.

Previous run:  run_01JABC
New run:       run_01KXYZ
Starting state: pretrained base weights (fresh init)
Dataset:        coco_2017:v1
Everything else: same as previous configuration

Create new run? [Y/n]
```

With changed data — `mlforge retrain rf_detr_s --dataset custom_v2` — the delta is shown (`Dataset changed: custom_v1 → custom_v2`), explicitly labeled **"This is a new experiment."**

### 6.4 FINE-TUNE

```text
User                    CLI                 Orchestrator
 │                       │                        │
 ├──mlforge finetune────►│                        │
 │   model://rf_detr_s:v1│                        │
 │   --dataset custom_v2 ├──FINETUNE─────────────►│
 │                       │                        ├──resolve base model + hash
 │                       │◄──WIZARD───────────────┤  (strategy, LR, epochs)
 │◄──config + warning────┤                        │
 │  "base model NOT modified, new run+model"      │
 ├──[Y]─────────────────►│                        │
 │                       ├──CREATE RUN (parent=model) 
 │                       ├──TRAIN                 │
 │                       │◄──model://rf_detr_s:v2─┤
```

**Strategy selection** (resolved into run_spec, user never edits checkpoint internals):

```text
Fine-tuning strategy:  1. Full  2. Freeze backbone  3. Freeze encoder
                       4. LoRA  5. Adapter          6. Custom
```

**Default source:** `model://rf_detr_s:v1` (model artifact). Specific checkpoint (`--from run_001/checkpoint/37`) is supported but non-default.

### 6.5 PAUSE / STOP

```text
User          CLI          Orchestrator      Runtime
 │             │                 │               │
 ├──pause RUN──┤                 │               │
 │             ├──PAUSE─────────►│               │
 │             │                 ├──STOP REQUEST─►
 │             │                 │               ├──finish safe boundary
 │             │                 │               ├──save checkpoint.tmp
 │             │                 │               ├──verify + fsync + atomic rename
 │             │                 │◄──COMMITTED────┤
 │             │◄──PAUSED────────┤               │
 │◄──checkpoint VALID, PAUSED────┤               │
```

`mlforge stop` runs the same sequence but ends in **STOPPED (no resume)** — resume capability is spent; continue only via `retrain`/`finetune`.

**Only graceful signals use this protocol:**

```text
pause / stop / Ctrl+C / SIGTERM / SLURM preemption signal:
    → safe boundary → checkpoint → commit → PAUSED or STOPPED

kill -9 / power loss / host crash (NO graceful path):
    → nothing is saved → heartbeat expires → INTERRUPTED
    → reconciliation scan → resume from last COMMITTED checkpoint (12 §11.2)
```

A crash is never reported as `PAUSED` — it did not stop gracefully, so the system must reconcile before it can promise a resume point.

### 6.6 PREPARE

```text
User              CLI            Orchestrator      Transform DAG
 │                 │                   │                │
 ├──prepare MODEL──┤                   │                │
 │                 ├──PREPARE─────────►│                │
 │                 │                   ├──resolve sources►
 │                 │                   ├──check cache────► (hit = skip)
 │                 │                   ├──transform──────►
 │                 │                   ├──validate───────►
 │                 │                   ├──write manifest─►
 │                 │◄──dataset://ID────┤                │
```

### 6.7 EVALUATE

```text
User                    CLI              Orchestrator
 │                       │                     │
 ├──evaluate MODEL──────►│                     │
 │   --dataset coco:test ├──EVALUATE───────────►│
 │                       │                     ├──resolve model+dataset hashes
 │                       │◄──show protocol─────┤
 │◄──confirm─────────────┤                     │
 ├──[Y]─────────────────►│                     │
 │                       ├──RUN INFERENCE      │
 │                       ├──COMPUTE METRICS    │
 │                       │◄──evaluation://eval_001
```

Evaluation artifact records `model_hash + dataset_hash + code_hash + environment_hash + evaluation_protocol_hash` → reproducible, **and comparable only when the protocol hash matches** (`12` §15.4). The protocol shown at confirmation includes split, metric definitions, thresholds, NMS settings, and seed — if the user later evaluates the same model with a different protocol, that is a *different* evaluation artifact, not an update.

### 6.8 INFER

```text
User                 CLI              Orchestrator        Runtime
 │                    │                     │                │
 ├──infer MODEL INPUT─┤                     │                │
 │                    ├──INFER─────────────►│                │
 │                    │                     ├──validate model │
 │                    │                     ├──validate input │
 │                    │                     │   schema       │
 │                    │                     ├──EXECUTE──────────────────────►
 │                    │                     │◄──results──────────────────────┤
 │                    │◄──output path───────┤                │
```

**No training-environment assumptions.** Inference validates its own runtime. Two modes: one-shot (`infer`) and long-running (`serve` → own environment validation).

### 6.9 EXPORT

```text
mlforge export model://rf_detr_s:v2 --format onnx

Export
  Source: rf_detr_s:v2 → Target: ONNX
  Validating: Architecture PASS · Operators PASS · Dynamic shapes PASS
  Exporting... ██████████ 100%
  Numerical validation: max error 0.003 (tolerance 0.01) PASS
  Export artifact: rf_detr_s_v2.onnx (own artifact identity)
```

### 6.10 COMPARE

```text
mlforge compare model://rf_detr_s:v1 model://rf_detr_s:v2 model://rf_detr_s:v3

MODEL COMPARISON
              v1      v2      v3
Dataset       D1      D2      D2
mAP           49.1    52.7    53.4
AP50          67.2    71.1    72.0
Base model    —       v1      v1
Training data ...

(descriptive only — does not choose a winner for the user)
```

**Comparability gate:** if any two evaluated models used different `evaluation_protocol_hash` values (different split, metrics, thresholds, or harness), their metric columns are marked `NOT_COMPARABLE` — shown, never averaged together. Same protocol = comparable numbers; different protocol = different measurement.

---

## 7. Failure Behavior Matrix

Every transition has defined failure behavior. **Fail-closed: no partial execution, no silent degradation.**

> **The invariants these failures enforce are defined in `12_training_system.md` §17; the 19-step resume validation gate is §18; checkpoint transactionality is §11; leases are §23.** This matrix defines the *product-visible behavior* when those mechanisms fail.

| Transition | Failure | Behavior |
|---|---|---|
| CREATED → VALIDATING | artifact resolution fails | → FAILED (`recovery: FORK_ONLY` — no checkpoint exists). Report missing artifact. Nothing written beyond run folder creation (cleaned up). |
| VALIDATING → READY | any invariant fails | → FAILED (`recovery: FORK_ONLY` — semantic cause), `exit 1`. Run stays inspectable. **Never reaches READY.** |
| VALIDATING → READY | disk insufficient | → FAILED before execution (`recovery: FORK_ONLY` — no checkpoint yet). Show required vs available. |
| READY → RUNNING | preflight fails | → FAILED (`recovery: FORK_ONLY` — no checkpoint yet). No training started. |
| RUNNING → CHECKPOINTING | write fails / disk full | → attempt rollback to last committed checkpoint; if repeated → FAILED (`recovery: RESUME`, last-good commit recorded). |
| RUNNING → CHECKPOINTING | power loss mid-write | On restart: incomplete ckpt detected (no commit marker) → newest-valid identity predicate selects resume point, all skips reported (`12` §11.2). |
| RUNNING | crash / power loss / kill -9 | Heartbeat expires → **INTERRUPTED** (never reported as PAUSED or RUNNING). Reconciliation scan → valid checkpoint → user may `resume`; no valid checkpoint → **FAILED** (`recovery: FORK_ONLY`). |
| RUNNING → PAUSING | pause during checkpoint | Wait for in-flight checkpoint to commit, then PAUSED. |
| PAUSING → PAUSED | checkpoint fails | Retry once; if fails → FAILED (`recovery: RESUME` — last-good checkpoint recorded). |
| PAUSED → VALIDATING | resume, dataset hash mismatch | → **BLOCK**. Stays PAUSED. Exit 1. No changes. |
| PAUSED → VALIDATING | resume, environment unavailable | → **BLOCK** through compatibility resolver. No "probably compatible." |
| PAUSED → VALIDATING | global batch unachievable | → **BLOCK** with migration options (other machine / fork / cancel). |
| PAUSED → VALIDATING | checkpoint corrupt | → newest-valid predicate skips corrupt generations (with per-skip report); if none valid → **BLOCK** with `NO VALID CHECKPOINT` (fork or restart). |
| RESUME (any state) | execution lease held | → **BLOCK** `RUN_ALREADY_EXECUTING`: `run lease held by pid N on host H`. Suggest `mlforge status`. Explicit `mlforge lease break` required to override. |
| Any | duplicate `--command-id` | → return original result; no second run created. Exit 0 with `COMMAND_DEDUPED` event. |
| RESUME on INTERRUPTED | state.json stale vs checkpoint manifest | Manifest wins; state.json rebuilt from journal during reconciliation. User sees reconciled state, never the stale one. |
| RUNNING (OOM) | GPU OOM | Capture failure → rollback to last committed checkpoint → planner adjusts **execution only** (micro_batch/accum) → validate global batch preserved → new execution segment. Never touches LR/optimizer/loss/dataset. |
| RUNNING → FAILED | runtime error, valid checkpoint intact | → FAILED (`recovery: RESUME`); cause recorded; `resume` re-runs the full gate after cause is resolved. |
| RUNNING → FAILED | unrecoverable / no valid checkpoint | → FAILED (`recovery: FORK_ONLY`); `resume` exits 3 with `NO VALID CONTINUATION — use mlforge fork or retrain`. |
| RUNNING → COMPLETED | epochs done | → MODEL CREATED (validate → AVAILABLE). |
| TRAIN (semantic) | user changes LR/batch mid-flow | Not possible in-place — `fork` only → new run with parent. |
| RETRAIN | dataset changed | Delta shown, confirmation required, labeled "new experiment." |
| FINE-TUNE | base model hash mismatch | BLOCK. Base must match recorded digest. |
| EVALUATE | model/dataset not resolved | exit 2, nothing run. |
| INFER | input schema invalid | BLOCK before execution; list violations. |
| EXPORT | operator unsupported | BLOCK with specific operator list; no partial export. |
| Any | secrets detected in artifacts | BLOCK packaging/export. |
| Any | schema_version unsupported | BLOCK + require explicit migration (`v1 → v2`). |

**Universal rules:**

- No command leaves a run in an indeterminate state — every failure maps to a defined state.
- `No changes were made` is printed whenever a BLOCK occurs after any mutation was attempted but rolled back.
- Exit codes are contract; scripts depend on them.

---

## 8. Decision Boundaries

### 8.1 System Handles Automatically (user never decides)

```text
Which CUDA library / PyTorch wheel        GPU micro-batch size
How many DataLoader workers               Which checkpoint format
How to restore optimizer / RNG / sampler  How to verify dataset hashes
How to compute global batch               How to create execution segments
How to validate checkpoint integrity      How to construct lineage
How to package the environment            Atomic write protocol
Heartbeat / telemetry collection          Status layer assembly
```

### 8.2 User Must Explicitly Decide (semantic — always confirmed)

```text
New dataset?              New architecture?         Retrain?
Fine-tune?                Change learning rate?     Change global batch?
Change loss?              Change augmentation?      Change training objective?
```

The system explains consequences; it never silently chooses.

---

## 9. Status & Telemetry Plane

> **Authoritative source-of-truth rules (state from checkpoint manifests, metrics as observations, append-only event journal) are defined in `12_training_system.md` §16.** This section specifies how that state is *presented and observed* without ever becoming authoritative itself.

### 9.1 Separation of Concerns

```text
                    TRAINING RUNTIME
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       AUTHORITATIVE STATE      LIVE TELEMETRY
              │                       │
              │                       ├── loss / LR / metrics
              │                       ├── GPU util / VRAM / temp / power
              │                       ├── throughput / ETA
              │                       └── current batch
              ▼
     checkpoint + state + events        (transient)
```

| Term | Meaning | Authority |
|---|---|---|
| **State** | What's required to recover training | checkpoint manifest (+ commit marker); `state.json` is a projection (`12` §16.1) |
| **Liveness** | Is a process running *right now* | supervisor heartbeat (not `state.json`) |
| **Status** | What is happening right now | live projection (never authoritative) |
| **Metrics** | Historical observations | `metrics/metrics.jsonl` |
| **Events** | What happened and when | `events/events.jsonl` (append-only) |

**Core invariant: `STATUS DOWN → TRAINING CONTINUES`.** GUI crash, closed terminal, lost network — none can corrupt or alter the run.

**Liveness rule: `state.json` never proves a live process.** Only a fresh heartbeat does; expired heartbeat + `RUNNING` on disk → reconciled to `INTERRUPTED` before display (`12` §12.3).

### 9.2 Control Plane vs Training Plane

```text
                 MLFORGE
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    CONTROL PLANE        TRAINING PLANE
          │                   │
          ├── CLI             ├── Model
          ├── TUI / GUI       ├── Optimizer
          ├── status/watch    ├── DataLoader
          ├── events          ├── GPU
          ├── commands        └── Checkpoint
          └── monitoring
```

The training plane continues if the control plane disappears. **Commands are asynchronous** — `mlforge train` returns a run ID and releases the terminal; `--attach` opts into the viewer.

**Process ownership:** the training worker is never a child of your shell. `mlforge train` submits to a per-user **supervisor daemon**, which spawns and owns the worker and emits heartbeats (`12` §12.4). Consequences:

- closing the terminal / dropping SSH does not kill training (`--attach` viewer detaches; worker lives on);
- the supervisor **never auto-restarts** a dead worker — restart = `resume` = your decision;
- if the supervisor itself dies, workers keep running and status rebuilds from run folders on next command.

### 9.3 Local State Server (instant, decoupled status)

```text
TRAINING PROCESS ──heartbeat, state, metrics, telemetry──► LOCAL STATE SERVER
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                             CLI                             GUI
```

`mlforge status` reads the state server — it never blocks on the training process. Works from another terminal; eventually from another machine (authenticated, optional).

### 9.4 Run Directory: Status Files

```text
run/
├── state/
│   ├── state.json         # authoritative lifecycle state
│   ├── heartbeat.json     # is the process alive?
│   └── live.json          # current transient metrics
├── events/events.jsonl    # historical events (append-only)
└── metrics/metrics.jsonl  # training metrics (observations only)
```

Different purposes, different files. **Metrics are never used to reconstruct state.**

**Heartbeat:**

```json
{"run_id":"run_01JABC","pid":18392,"host_id":"...","started_at":"...",
 "last_seen":"...","global_step":18492,"state":"RUNNING"}
```

Stale heartbeat → `INTERRUPTED`, **not** `FAILED` (unexpected ≠ unrecoverable):

```text
WARNING — no heartbeat for 120s.
Possible: stalled process / hung kernel / machine unreachable / crash.
Run marked INTERRUPTED (not failed) — awaiting reconciliation.
Last committed checkpoint: step 18,492 (integrity verified).

$ mlforge resume run_01JABC
  reconciliation: state.json said RUNNING, no live worker → INTERRUPTED
  checkpoint 18,492: VERIFIED → resume point
```

**Never infer state from telemetry** — `GPU=0%` may mean checkpointing, validation, data loading, or CPU preprocessing. Lifecycle state comes only from the orchestrator.

### 9.5 Layered Status

**Level 1 — `mlforge status` (overview):**

```text
RUNNING
Model RF-DETR-S · Run run_01JABC
Epoch 23/50 (46%) · Step 18,492
Loss 0.382 · mAP 49.1 · Best 49.1 @ 23
Speed 2.14 it/s · ETA 11h 42m
GPU RTX 3070 · VRAM 7.3/8.0 GB · 72°C · 198W
Checkpoint 18,492 · Integrity VALID
```

**Level 2 — `mlforge status --verbose` (training detail):** epoch/step breakdown, optimizer + scheduler + batch config, precision/AMP, gradient norm/clipping, samples/sec, loader wait %, checkpoint last/best/next.

**Level 3 — `mlforge hardware` (telemetry):** GPU util/VRAM/temp/power, compute + bandwidth %, CPU, RAM, disk I/O, free space. Diagnostic only — not training state.

**Stage-aware status** (lifecycle stage within RUNNING):

```text
RUNNING
  Stage: CHECKPOINTING
  Saving: checkpoint-18492
  Progress: 87%
```

Stages: `TRAINING · VALIDATING · CHECKPOINTING · EVALUATING · PREPARING_DATA`.

**FAILED runs always display their disposition** — no guessing whether `resume` is allowed:

```text
FAILED
Model RF-DETR-S · Run run_01JABC
Cause:      CUDA OOM at step 41200
Recovery:   RESUME (valid checkpoint: step 41,200)
Action:     mlforge resume run_01JABC   (after resolving cause)
```

```text
FAILED
Cause:      invariant violation (run_spec mismatch)
Recovery:   FORK_ONLY — no valid continuation exists
Action:     mlforge fork run_01JABC  |  mlforge retrain rf_detr_s
```

### 9.6 Watch (primary interactive view)

```bash
mlforge watch run_01JABC
```

```text
╔══════════════════════════════════════════════════════════╗
║ MLForge — run_01JABC                    RUNNING          ║
╠══════════════════════════════════════════════════════════╣
║ RF-DETR-S                                                ║
║ Epoch 23/50  46.0%    Step 18,492/40,000                 ║
║ Loss 0.382   mAP 49.1  Best 49.1 @ 23                   ║
║ LR 7.50e-5   Speed 2.14 it/s   ETA 11h 42m              ║
║ GPU0 RTX 3070  Util 97%  VRAM 7.3/8 GB  72°C  198W      ║
║ Checkpoint 18,492   Integrity VALID                      ║
║ Last event: New best mAP = 49.1                          ║
╚══════════════════════════════════════════════════════════╝
 [p]ause  [s]top  [c]heckpoint  [e]vents  [q]uit viewer
```

`q` exits the **viewer only** — training continues.

### 9.7 Events

```bash
mlforge events run_01JABC --follow      # structured tail -f
```

```text
21:31:02  Epoch 23 started
21:31:19  Validation started
21:31:51  Validation completed
21:31:52  New best mAP: 49.1
21:32:04  Checkpoint started
21:32:11  Checkpoint committed
21:32:12  Epoch 24 started
```

### 9.8 Status For Every Workflow

Same infrastructure for `PREPARING` (transform progress + cache %), `EVALUATING` (samples done, running metrics), `INFERENCE` (throughput + ETA), `EXPORT` (conversion + validation bars).

### 9.9 Event-Driven, State-Reconstructing

Internally: runtime → state/metric/telemetry/checkpoint events → event bus / local IPC → CLI + GUI (polling as fallback).

On (re)connect, the viewer: `read authoritative state → read latest telemetry → subscribe to events`. Close `watch` at epoch 12, reopen at 23 — state is never lost, the view is reconstructed from persistence.

### 9.10 Remote (Optional)

```text
Machine A (training) → mlforge daemon → authenticated telemetry → Machine B: mlforge watch
```

Optional. Core system works fully local/offline.

---

## 10. Directory & Artifact Schema (Product View)

```text
my-project/
├── project.yaml              # project config
├── datasets.yaml             # dataset registrations (identity hashes)
├── configs/                  # model training configs
├── models/                   # model definitions (code wrappers)
├── runs/
│   └── run_01JABC.../        # portable run (see architecture §5)
│       ├── run_spec.json     # immutable identity
│       ├── .lease             # single-writer run lease (architecture §23)
│       ├── state/            # state.json · heartbeat.json · live.json
│       ├── events/ · metrics/
│       ├── checkpoints/ · segments/ · code/ · environment/
│       ├── datasets/ · evaluations/ · exports/
│       └── integrity.json
├── artifacts/                # model artifacts (content-addressed refs)
└── ~/.mlforge/
    ├── datasets_<project>.yaml   # machine-local paths (never in identity)
    ├── commands.jsonl            # idempotency journal (§4.4)
    └── store/sha256/...          # content-addressed artifact store
```

Machine-local path config differs per machine; artifact identities never do.

---

## 11. Implementation Order

Given this spec + the architecture spec (`12_training_system.md`), build order:

```text
1. Workflow API + state machines (project/dataset/run/model)
2. Artifact registry + content store + run_spec canonical hashing
3. Validation gate + preflight (fail-closed core)
4. CLI contract implementation (thin layer over Workflow API)
5. Supervisor daemon + run leases + idempotency journal (§4.4, 12 §23)
6. Training runtime + transactional checkpoints + heartbeat/state server
   + reconciliation scan (12 §12.3)
7. Status layer (L1/L2/L3, watch, events) — read-only
8. Ingestion/transform DAG
9. Execution planner (capability negotiation)
10. Resume/retrain/finetune flows + lineage DAG
11. Evaluate/compare/infer/export/package (incl. evaluation protocol identity)
12. TUI/GUI over the same Workflow API
```

The invariant machinery (architecture spec) is the foundation; this product spec defines how the human drives it. Neither exists without the other.
