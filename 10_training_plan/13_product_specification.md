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
PREPARE · TRAIN · RESUME · RETRAIN · FINE-TUNE · EVALUATE · INFER · EXPORT · COMPARE
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
              ┌────────────┼───────────────┐
              ▼            ▼               ▼
           RESUME       EVALUATE         STOP
              │            ▼               │
              │         REPORT             │
              ▼                            ▼
           COMPLETE ◄─────────────────────(paused→resume)
              ▼
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
mlforge resume <RUN>                    Continue paused run
mlforge stop <RUN>                      Graceful stop → PAUSED
mlforge retrain <MODEL> [--dataset D]   New run, fresh init
mlforge finetune <MODEL> [--dataset D]  New run, from model weights

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

### 4.3 Object Reference Syntax

```text
RUN       run_01JABC...          or ./runs/run_01JABC...
MODEL     model://rf_detr_s:v2    or rf_detr_s:v2
DATASET   dataset://coco_2017:v1  or coco_2017:v1
EVAL      evaluation://eval_001
```

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
CREATED → VALIDATING → READY → RUNNING
                              │  │  │
              ┌───────────────┘  │  └───────────────┐
              ▼                  ▼                  ▼
          PAUSING            CHECKPOINTING        FAILED
              │                  │                  │
              ▼                  │                  ▼
          PAUSED ───resume──►VALIDATING        (recover)→ VALIDATING
              │
              ▼
          STOPPED (terminal)
                             ┌────────────────────┐
                             │                    ▼
                             │               COMPLETED
                             │                    │
                             └────────────► MODEL CREATED
```

Transitions:

| From | Event | To |
|---|---|---|
| CREATED | start validation | VALIDATING |
| VALIDATING | all checks pass | READY |
| VALIDATING | any check fails | FAILED (blocked, never RUNNING) |
| READY | preflight pass | RUNNING |
| RUNNING | `stop` command | PAUSING |
| PAUSING | checkpoint committed | PAUSED |
| RUNNING | checkpoint cycle | CHECKPOINTING → RUNNING |
| RUNNING | unrecoverable error | FAILED |
| RUNNING | epochs complete | COMPLETED |
| PAUSED | `resume` + validation pass | VALIDATING → READY → RUNNING |
| PAUSED | `stop` (final) | STOPPED |
| FAILED | `resume` + cause resolved | VALIDATING |

**Resume IS a state transition. Retrain/fine-tune are NOT** — they create new runs.

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

```text
User              CLI            Orchestrator         Validation Gate (17 steps)
 │                 │                   │                      │
 ├──mlforge resume─┤                   │                      │
 │                 ├──RESUME──────────►│                      │
 │                 │                   ├──VALIDATE────────────►
 │                 │                   │◄──PASS (mode=EXACT    │
 │                 │                   │    or PORTABLE)───────┤
 │                 │                   │  (any FAIL → STOP, no execution)
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

No changes were made. Run remains PAUSED.
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

### 6.5 STOP

```text
User          CLI          Orchestrator      Runtime
 │             │                 │               │
 ├──stop RUN──►│                 │               │
 │             ├──STOP──────────►│               │
 │             │                 ├──STOP REQUEST─►
 │             │                 │               ├──finish safe boundary
 │             │                 │               ├──save checkpoint.tmp
 │             │                 │               ├──verify + fsync + atomic rename
 │             │                 │◄──COMMITTED────┤
 │             │◄──PAUSED────────┤               │
 │◄──checkpoint VALID, PAUSED────┤               │
```

Same protocol for Ctrl+C, SIGTERM, SLURM preemption, power loss — the only difference is whether the last checkpoint committed.

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

Evaluation artifact records `model_hash + dataset_hash + code_hash + environment_hash` → reproducible metrics.

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

---

## 7. Failure Behavior Matrix

Every transition has defined failure behavior. **Fail-closed: no partial execution, no silent degradation.**

| Transition | Failure | Behavior |
|---|---|---|
| CREATED → VALIDATING | artifact resolution fails | → FAILED. Report missing artifact. Nothing written beyond run folder creation (cleaned up). |
| VALIDATING → READY | any invariant fails | → FAILED, `exit 1`. Run stays inspectable. **Never reaches READY.** |
| VALIDATING → READY | disk insufficient | → FAILED before execution. Show required vs available. |
| READY → RUNNING | preflight fails | → FAILED. No training started. |
| RUNNING → CHECKPOINTING | write fails / disk full | → attempt rollback to last committed checkpoint; if repeated → FAILED. Run remains resumable from last commit. |
| RUNNING → CHECKPOINTING | power loss mid-write | On restart: incomplete ckpt detected (no commit marker) → auto-report `resume point: N-1`. |
| RUNNING → PAUSING | stop during checkpoint | Wait for in-flight checkpoint to commit, then PAUSED. |
| PAUSING → PAUSED | checkpoint fails | Retry once; if fails → FAILED with last-good checkpoint recorded. |
| PAUSED → VALIDATING | resume, dataset hash mismatch | → **BLOCK**. Stays PAUSED. Exit 1. No changes. |
| PAUSED → VALIDATING | resume, environment unavailable | → **BLOCK** through compatibility resolver. No "probably compatible." |
| PAUSED → VALIDATING | global batch unachievable | → **BLOCK** with migration options (other machine / fork / cancel). |
| PAUSED → VALIDATING | checkpoint corrupt | → fall back to N-1, N-2; record `recovery_from_checkpoint`. If all corrupt → BLOCK. |
| RUNNING (OOM) | GPU OOM | Capture failure → rollback to last committed checkpoint → planner adjusts **execution only** (micro_batch/accum) → validate global batch preserved → new execution segment. Never touches LR/optimizer/loss/dataset. |
| RUNNING → FAILED | unrecoverable error | State FAILED; last committed checkpoint intact; `resume` allowed after cause resolved. |
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
| **State** | What's required to recover training | checkpoint manifest + `state.json` |
| **Status** | What is happening right now | live projection (never authoritative) |
| **Metrics** | Historical observations | `metrics/metrics.jsonl` |
| **Events** | What happened and when | `events/events.jsonl` (append-only) |

**Core invariant: `STATUS DOWN → TRAINING CONTINUES`.** GUI crash, closed terminal, lost network — none can corrupt or alter the run.

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

Stale heartbeat ≠ failure:

```text
WARNING — no heartbeat for 120s.
Possible: stalled process / hung kernel / machine unreachable / crash.
Training state has NOT been marked failed.
Last committed checkpoint: step 18,492.
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
│       ├── state/            # state.json · heartbeat.json · live.json
│       ├── events/ · metrics/
│       ├── checkpoints/ · segments/ · code/ · environment/
│       ├── datasets/ · evaluations/ · exports/
│       └── integrity.json
├── artifacts/                # model artifacts (content-addressed refs)
└── ~/.mlforge/
    ├── datasets_<project>.yaml   # machine-local paths (never in identity)
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
5. Training runtime + transactional checkpoints + heartbeat/state server
6. Status layer (L1/L2/L3, watch, events) — read-only
7. Ingestion/transform DAG
8. Execution planner (capability negotiation)
9. Resume/retrain/finetune flows + lineage DAG
10. Evaluate/compare/infer/export/package
11. TUI/GUI over the same Workflow API
```

The invariant machinery (architecture spec) is the foundation; this product spec defines how the human drives it. Neither exists without the other.
