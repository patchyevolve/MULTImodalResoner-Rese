# MLForge — Training System Architecture (v1.0)

> A hardware-agnostic, project-agnostic training platform built on **immutable artifacts, cryptographic identities, explicit execution modes, and fail-closed validation**. Install on any OS, train on any GPU, move between machines via portable run folders — with a hard guarantee: **if the system cannot prove a continuation is valid, it refuses to run.**

---

## 1. Guarantees

The system advertises exactly these — no more:

```
MLFORGE GUARANTEES
────────────────────────────────────────────
 1. No silent semantic changes.
 2. No resume with unidentified data.
 3. No resume with unidentified code.
 4. No resume with unidentified model architecture.
 5. No resume from corrupt checkpoints.
 6. No execution in an unverified environment.
 7. Hardware may change only through validated execution adaptation.
 8. Semantic configuration is immutable.
 9. Fine-tuning always creates a new lineage node.
10. Retraining always creates a new run.
11. Every checkpoint is independently recoverable.
12. Every model is traceable to code, data, transforms,
    environment, configuration, checkpoint.
13. Every execution segment records its hardware.
14. Every artifact is content-addressed.
15. Unknown compatibility = BLOCK, never GUESS.
```

**What we explicitly do NOT promise:**

> ~~"Train on RTX 3070 and resume identically on H100."~~

What we promise:

> **MLForge preserves experiment semantics across supported hardware migrations and explicitly distinguishes exact deterministic continuation from portable continuation. If MLForge cannot prove the continuation is valid, MLForge refuses to continue.**

### 1.1 What "never break" means precisely

We **cannot** guarantee identical floating-point bits across GPU architectures, drivers, CUDA kernels, reduction orderings, or compiler versions. Anyone claiming that is lying.

We **can** guarantee:

| Guarantee | Meaning |
|---|---|
| **A: No silent semantic change** | If continuation validity cannot be proven → `DO NOT TRAIN`. No guessing, no downgrading, no substituting. |
| **B: Reproducible experiment identity** | Every run has immutable identities for code, config, data, transforms, base model, environment, randomness, training state. |
| **C: Controlled hardware migration** | GPU/CPU/count/Precision changes never silently alter semantic parameters. |
| **D: Lineage-traceable continuation** | Every model traces back through checkpoints → datasets → code → environment image. |

---

## 2. Core Architecture

```
                        ┌───────────────────────┐
                        │      mlforge CLI       │
                        └───────────┬───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   Run Orchestrator     │
                        │                        │
                        │  State Machine          │
                        │  Validation Gate        │
                        │  Lineage Manager        │
                        │  Policy Engine          │
                        └───────────┬───────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       │                            │                            │
       ▼                            ▼                            ▼
┌───────────────┐          ┌──────────────────┐          ┌─────────────────┐
│  Artifact     │          │  Execution       │          │  Dataset        │
│  Registry     │          │  Environment     │          │  Registry       │
│               │          │                  │          │                 │
│ code          │          │ container image  │          │ immutable data  │
│ models        │          │ dependencies     │          │ manifests       │
│ checkpoints   │          │ runtime          │          │ transforms      │
│ configs       │          │ hardware adapter │          │ cache           │
└───────┬───────┘          └────────┬─────────┘          └────────┬────────┘
        │                           │                             │
        └───────────────────────────┼─────────────────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │  Training Runtime      │
                        │                        │
                        │  Model                  │
                        │  Optimizer + Scheduler  │
                        │  AMP / scaler           │
                        │  RNG hierarchy          │
                        │  Sampler + Dataloader   │
                        │  Distributed runtime    │
                        └───────────┬───────────┘
                                    │
                         ┌──────────▼──────────┐
                         │ Atomic Checkpoint    │
                         │ + Event Journal      │
                         └──────────┬──────────┘
                                     │
                   ┌─────────────────┼─────────────────┐
                   ▼                 ▼                 ▼
                RESUME           EVALUATE           EXPORT
```

**The key conceptual change:**

> ~~"A run is a folder that contains enough things to resume."~~
>
> **"A run is an immutable, content-addressed scientific state whose execution can occur across multiple machines. The folder is a portable materialization of that state."**

---

## 3. The Four Separated Concepts

Do not conflate these:

| Concept | Question it answers |
|---|---|
| **Dataset** | What data exists in the world? |
| **Prepared dataset** | What exact representation enters the model? |
| **Run** | What experiment was performed? |
| **Model** | What learned parameters were produced? |

Each has its own identity, its own artifact, its own lifecycle.

---

## 4. Immutable Run Specification

Every run is defined by a `run_spec.json` — created once, **never mutated**:

```yaml
# run_spec.json
schema_version: 1

run:
    id: "run_01JABC..."
    created_at: "2026-09-30T14:30:00Z"
    parent: null                    # set when forked

code:
    source_hash: "sha256:..."       # hash of full source tree
    repository_commit: "abc123"
    dirty: false                    # uncommitted changes?
    training_runtime_version: "1.0.0"

model:
    architecture: "rf_detr_s"
    architecture_hash: "sha256:..." # hash of model implementation code
    base_weights:
        artifact: "pretrained_rf_detr_s"
        digest: "sha256:..."        # exact pretrained checkpoint hash

data:
    train_manifest: "sha256:..."    # dataset identity (NOT path)
    validation_manifest: "sha256:..."
    transform_manifest: "sha256:..." # transform artifact identity

training:
    optimizer: "adamw"
    learning_rate: 1e-4
    scheduler: "cosine_with_restarts"
    global_batch_size: 32           # THE semantic invariant
    seed: 12345
    target_epochs: 50
    early_stopping: {patience: 10, metric: "mAP", mode: "max"}
    losses: [...]
    augmentations: [...]
    precision_policy:
        preferred: "bf16"
        fallback: "fp32"            # fallback requires PORTABLE mode

environment:
    image_digest: "sha256:..."      # OCI container image — primary identity
    lockfile_sha256: "sha256:..."

reproducibility:
    level: "R2"                     # see §14
    deterministic_algorithms: true
    cudnn_benchmark: false
    data_order: "deterministic"
    allow_nondeterministic_kernels: false
```

**Canonicalization before hashing:**

```
YAML → schema validation → normalized representation → canonical JSON → SHA256
```

So `lr: 0.0001` and `lr: 1e-4` produce the **same** `run_spec_hash`.

**Hard rule:** `run_spec_hash` is immutable. Changing LR, batch, optimizer, augmentation, loss, dataset, or model creates a **NEW RUN** with `parent_run = <old hash>`. Never mutate. (`mlforge fork`, not `--reconfigure`.)

---

## 5. Portable Run Folder

The folder is a **materialization** of the immutable run state:

```
runs/run_01JABC.../
│
├── manifest.json              # folder-level integrity (hashes of everything below)
├── run_spec.json              # 🔒 immutable experiment identity
├── lineage.json               # parent runs/models (DAG edges)
├── status.json                # CREATED/PAUSED/COMPLETED/FAILED
│
├── code/
│   └── source.snapshot.tar.gz # full source tree snapshot (gap: code capture)
│       + artifact.manifest     # sha256 of snapshot
│
├── environment/
│   ├── image.json             # OCI image digest + requirements
│   ├── lock.json              # pip freeze / lockfile
│   └── hardware_requirements.json  # min driver, min VRAM, arch
│
├── datasets/
│   ├── train.manifest         # dataset identity: hashes, counts, schemas
│   ├── val.manifest
│   └── transforms.manifest    # transform artifact identities
│
├── checkpoints/
│   ├── ckpt-000001/
│   │   ├── manifest.json      # hashes of every file below + commit marker
│   │   ├── model.safetensors
│   │   ├── optimizer.pt
│   │   ├── scheduler.pt
│   │   ├── amp_scaler.pt
│   │   ├── rng_state.bin      # full RNG hierarchy (§11)
│   │   ├── sampler_state.bin
│   │   ├── dataloader_state.bin
│   │   └── state.json         # epoch, global_step, batch_position, metrics
│   ├── ckpt-000002/
│   └── ckpt-000003/
│       └── (keep top-K generations for recovery)
│
├── segments/                  # one per hardware execution context
│   ├── 0001/
│   │   ├── execution.json     # RTX 3070, 1 GPU, bf16, micro=2, accum=16
│   │   └── metrics.jsonl
│   └── 0002/
│       ├── execution.json     # H100, 1 GPU, bf16, micro=32, accum=1
│       └── metrics.jsonl
│
├── evaluations/
│   └── eval_001/              # model_hash + dataset_hash + code_hash + metrics
│
├── exports/
│   └── onnx/model.onnx
│
├── events.jsonl               # append-only audit journal
│
└── integrity.json             # merkle root of all artifacts in this folder
```

**Size:** ~400MB-1GB for RF-DETR-S scale models. Transferable via git-LFS, USB, network share.

---

## 6. Artifact Identity Model

### 6.1 Content-Addressed Store

```text
~/.mlforge/store/
    sha256/
        ab/abcdef...
        72/72f91...
```

All artifacts — datasets, transforms, source, environments, models, checkpoints, evaluations — are immutable objects identified by content hash. The run folder contains **references**:

```json
{"model": "sha256:abc...", "dataset": "sha256:def...", "environment": "sha256:ghi..."}
```

Deduplication and integrity come for free.

### 6.2 Dataset Identity Is Cryptographic (Not Count-Based)

```
Dataset
   ├── dataset_id        # e.g. "coco_2017"
   ├── version           # e.g. "v1"
   ├── schema            # annotation format
   ├── files[]
   │     ├── sha256
   │     ├── size
   │     └── relative_path      # NEVER absolute path
   └── manifest_hash = SHA256(canonical_manifest)
```

**Sample counts are NOT identity.** Two 100,000-image datasets can be completely different data. Hashes catch that.

### 6.3 Paths Are Machine-Local, Never Part of Identity

```yaml
# run_spec.json (immutable)
dataset:
    id: "coco_2017"
    version: "v1"
    identity: "sha256:abc..."     # ← this is the identity
```

```yaml
# ~/.mlforge/datasets_<project>.yaml (machine-local, mutable)
coco_2017:
    path: "/home/daksh/data/coco"     # Machine A
    # path: "/mnt/datasets/coco"      # Machine B
    # path: "D:\datasets\coco"        # Machine C
```

Same dataset identity, different physical location per machine. User configures paths via `mlforge configure datasets` (§10.2).

### 6.4 Transform Identity Is an Artifact

```
Transform Artifact
   ├── transform_id
   ├── code_hash            # hash of transform implementation
   ├── config_hash
   ├── input_schema
   ├── output_schema
   └── implementation_version
```

Chain:

```
RAW DATA (dataset_hash)
    ↓
TRANSFORM (transform_hash)
    ↓
MODEL DATA (output_manifest_hash)
    ↓
TRAINING
```

**Cache key becomes:**

```
cache_key = SHA256(input_artifact_hash + transform_hash + transform_config_hash + environment_hash)
```

Same inputs + same transform + same environment = same artifact. Change any one → new artifact. No stale-cache guessing.

### 6.5 Environment Identity Is an Image Digest

```yaml
environment:
    schema_version: 1
    image:
        type: oci
        digest: "sha256:..."      # ← THE identity, not package versions
    python: {implementation: CPython, version: "3.11.9"}
    framework: {pytorch: "2.5.1"}
    cuda: {runtime: "12.4"}
    dependencies: {lockfile_sha256: "sha256:..."}
```

`same package versions ≠ same executable environment`. The image digest is authoritative.

**Host vs training environment separation:**

```
HOST MACHINE provides:      CPU, RAM, GPU, driver, filesystem, container runtime
TRAINING ENV provides:      Python, PyTorch, Lightning, CUDA runtime, model code, all deps
```

Same image runs on 3070 machine and H100 machine. Eliminates pip drift, system Python drift, dependency drift, compiler drift.

### 6.6 Code Is Captured

Every run records:

- source tree hash
- git commit + submodules + dirty flag
- training runtime version
- model implementation version
- **full source snapshot** in `code/source.snapshot.tar.gz`

If the git repo disappears, the experiment is still recoverable.

---

## 7. Validation Gate (Fail-Closed)

### 7.1 BLOCK vs WARN

**WARN** — information that does not invalidate the operation:

```
H100 is 15× faster than original GPU
Disk has 1.8TB free (plenty)
```

**BLOCK** — anything that could change the experiment:

```
dataset hash mismatch
code artifact missing
architecture hash mismatch
base model hash mismatch
environment unavailable / unverifiable
checkpoint integrity failure
global batch cannot be preserved
driver incompatible
checkpoint atomicity unsupported
schema version unsupported
secrets present in artifacts
```

Environment incompatibility is **NOT a warning path** — it routes through a compatibility resolver:

```
environment mismatch
       ↓
compatibility resolver
       ↓
  ┌────┴────┐
SAFE      UNSAFE
  ↓          ↓
continue    BLOCK
```

"Probably compatible" does not exist.

### 7.2 Validation Report

```text
$ mlforge validate runs/run_01JABC...

VALIDATION REPORT
─────────────────────────────────────────
[PASS] Run manifest integrity
[PASS] Schema version (v1 supported)
[PASS] Code artifact present + hash match
[PASS] Source snapshot hash match
[PASS] Dataset identity: coco_2017 v1 (sha256:abc...)
[PASS] Dataset sample verification (118,287 files hashed)
[PASS] Transform artifact identity
[PASS] Model architecture hash
[PASS] Base pretrained weights digest
[PASS] Checkpoint integrity (ckpt-000023, all hashes valid)
[PASS] Environment image available (sha256:ghi...)
[PASS] GPU capability (H100, bf16 supported)
[PASS] Driver compatibility (CUDA 12.4 runtime ≤ driver 550.x)
[PASS] Global batch preserved (32 = 32)
[PASS] Precision policy supported (bf16 → bf16)
[PASS] Optimizer state recoverable
[PASS] LR scheduler state recoverable
[PASS] RNG hierarchy recoverable
[PASS] Sampler state recoverable
[PASS] Disk space sufficient (need 8GB, have 1.8TB)
[PASS] Atomic rename supported (filesystem: ext4)
[PASS] No secrets in artifacts

RESULT:
    SAFE TO RESUME
    MODE: PORTABLE CONTINUATION
    SEGMENT: 0002 (new execution context will be recorded)
```

Or:

```text
[FAIL] Dataset identity
    Expected: sha256:abc... (coco_2017 v1, 118,287 files)
    Found:    sha256:def... at /home/daksh/data/coco
    Reason:   different content (same count, different files)

RESULT:
    RESUME BLOCKED
```

**If steps 1–N fail: TRAINING DOES NOT START.**

### 7.3 Preflight

Before any expensive operation:

```bash
$ mlforge preflight runs/run_01JABC...
```

Validates: OS, Python, container runtime, GPU, driver, VRAM, disk space, RAM, dataset hashes, code, environment, model, checkpoint, filesystem write permissions, atomic rename support, storage headroom.

Only after `PREFLIGHT PASSED` does `mlforge train/resume` proceed.

**Disk-space validation is mandatory** — computes worst-case: checkpoint + temp checkpoint + dataset cache + logs + safety margin. If free < required → BLOCK. Never "train until disk fills."

---

## 8. Two Resume Modes

### 8.1 Mode EXACT

Guarantees continuation of the same trajectory:

```
same execution environment  ·  same software  ·  same world size
same precision  ·  same deterministic settings  ·  same dataset
same sampler  ·  same code  ·  same config
```

Requires: R3+ reproducibility level, identical execution topology, recoverable RNG/sampler/dataloader state.

```
CHECKPOINT → EXACT CONTINUATION
```

### 8.2 Mode PORTABLE

Hardware may change (3070→H100, 1→4 GPU, GPU→CPU):

**Preserved:** model, optimizer semantics, global batch, LR semantics, dataset, transform, code, configuration.

**Explicitly recorded:**

```json
{
    "trajectory_continuity": "semantic",
    "bitwise_reproducibility": false
}
```

This is honest and technically defensible.

### 8.3 Why global batch is the invariant (and why even that isn't bitwise)

```
global_effective_batch = micro_batch × grad_accum × world_size
```

| Machine | micro | accum | world | global |
|---|---|---|---|---|
| RTX 3070 | 2 | 16 | 1 | **32** |
| H100 | 32 | 1 | 1 | **32** |
| 2× H100 | 8 | 2 | 2 | **32** |

**But even preserved global batch ≠ identical training**, because:

- **BatchNorm**: micro_batch 2→32 changes batch statistics
- **FP reduction order**: `(a+b)+c ≠ a+(b+c)` in floating point
- **Distributed all-reduce**: world size changes reduction structure
- **AMP kernels**: different GPU architectures execute different kernels
- **Data ordering**: worker counts affect loading order unless controlled

Therefore: EXACT mode requires identical topology; everything else is PORTABLE mode with `bitwise_reproducibility: false`.

---

## 9. Reproducibility Levels

```text
R0 — no reproducibility guarantee
R1 — experiment reproducibility (same config/data/code)
R2 — execution reproducibility (same environment + hardware class)
R3 — deterministic continuation (same topology + deterministic runtime)
R4 — verified bitwise reproducibility (only where framework/hardware supports it)
```

Recorded in `run_spec.reproducibility.level`. **Never claim R4 when you only have R1.**

```yaml
reproducibility:
    level: "R2"
    seed: 123456
    deterministic_algorithms: true
    cudnn_benchmark: false
    data_order: deterministic
    worker_seed_policy: deterministic
    allow_nondeterministic_kernels: false
```

If a model requires a nondeterministic kernel: STRICT mode → BLOCK; RELAXED mode → allow + record exception.

---

## 10. Dataset Configuration & Ingestion

### 10.1 User-Configured Paths (No Discovery)

One-time per machine:

```bash
$ mlforge configure datasets
  coco_2017 path? /home/daksh/data/coco
    → verifying content hash... ✅ matches sha256:abc (coco_2017 v1)
  market1501 path? /home/daksh/data/market1501
    → verifying... ✅ matches sha256:def
Saved: ~/.mlforge/datasets_multimodal_reasoner.yaml
```

Wrong path → hash mismatch → explicit error with fix options. The system **verifies identity**, it does not scan or guess.

### 10.2 Ingestion Is a Content-Addressed DAG

```yaml
# ingestion.yaml
models:
  rf_detr_s:
    transform: coco_detection
    train_sources: [coco_2017:train, custom_clips:train]
    val_sources: [coco_2017:val]
  osnet:
    transform: reid_crops
    train_sources: [market1501:train, custom_reid:train]
    depends_on: [rf_detr_s]           # crops need trained detector
  calibrator:
    transform: calibration
    train_sources: [{generated_from: rf_detr_s}]
    depends_on: [rf_detr_s]
```

Formalized as DAG:

```
Dataset → Transform → Artifact → Model → Derived Dataset → Next Model
```

Every node gets `artifact_hash`. Invalidation is deterministic.

---

## 11. Checkpoint Integrity & Recovery

### 11.1 Transactional Writes (Never `torch.save` Directly)

```text
write checkpoint.tmp files
        ↓
calculate hashes
        ↓
fsync
        ↓
atomic rename → final names
        ↓
write manifest.json
        ↓
fsync
        ↓
write commit marker
```

Power loss mid-write → `ckpt-23` has no commit marker → treated as incomplete. System reports:

```text
Latest valid checkpoint: 22
Latest attempted checkpoint: 23 (incomplete — ignored)
Resume point: 22
```

### 11.2 Multiple Generations (Recovery Quorum)

```
checkpoints/
    ckpt-000017/   ← committed
    ckpt-000018/   ← committed
    ckpt-000019/   ← committed (keep top-K)
```

If N corrupt → fall back to N-1 → N-2. Record `recovery_from_checkpoint`.

### 11.3 Integrity Verification on Load

```text
load manifest → verify per-file sha256 → verify commit marker → load
```

Corrupt → **BLOCK**, not "try anyway."

### 11.4 Complete Checkpoint Contents

```text
MODEL · OPTIMIZER · LR SCHEDULER · AMP SCALER · EMA STATE
GLOBAL STEP · EPOCH · BATCH POSITION
SAMPLER STATE · DATALOADER STATE · DISTRIBUTED STATE
GRADIENT ACCUMULATION STATE · EARLY STOPPING STATE · BEST MODEL STATE
RNG HIERARCHY:
    python_rng · numpy_rng · torch_cpu_rng · torch_cuda_rng[per-device]
    dataloader_worker_seeds · sampler_state
```

Without RNG/sampler/dataloader state, "resume" means *same weights + different future data* — not continuation.

---

## 12. Run State Machine

CLI commands never manipulate training directly — they drive the state machine:

```text
CREATED → VALIDATING → PREPARING → READY → RUNNING
                                        ├── PAUSING → PAUSED
                                        ├── CHECKPOINTING
                                        ├── FAILED
                                        └── COMPLETED
```

```text
Resume:    PAUSED → VALIDATING → READY → RUNNING
Fork:      CHECKPOINT → FORK → NEW RUN → VALIDATE → TRAIN
Inference: MODEL ARTIFACT → ENV VALIDATION → SCHEMA VALIDATION → INFERENCE
```

**SIGINT / preemption / OOM / power loss all use the same checkpoint protocol:**

```text
RUNNING → STOP REQUESTED → finish safe boundary → checkpoint → commit → STOPPED
```

### 12.1 OOM Recovery (Constrained)

```text
OOM → capture failure → rollback to last committed checkpoint
    → planner proposes hardware execution adjustment
    → validate semantic invariants
    → new execution segment
```

May auto-adapt: `micro_batch`, `grad_accum`, `checkpointing`, `num_workers`.
Never: `LR`, `optimizer`, `loss`, `dataset`, `augmentation`, `global batch`, `model` — those require a fork.

### 12.2 Execution Segments (Hardware Migration Record)

Hardware changes create a new segment, never overwrite history:

```text
Segment 0001: RTX 3070, 1 GPU, bf16, micro=2, accum=16  → epoch 0→23
Segment 0002: H100, 1 GPU, bf16, micro=32, accum=1       → epoch 23→50
```

Each segment's `execution.json` preserves the historical execution environment forever.

---

## 13. Capability-Based Execution Planner

### 13.1 Capability Negotiation (Not Hardcoded VRAM Tables)

```text
Hardware → Capabilities:
    memory · compute_capability · bf16 · fp16 · fp8 · tensor_cores
    gpu_count · interconnect · cpu · ram
```

### 13.2 Feasibility Solver

```text
Target global batch = 32

Find: micro_batch × grad_accum × world_size = 32
Subject to:
    VRAM(micro_batch) ≤ available − overhead
    precision supported by arch
    micro_batch ≥ model minimum
```

H100 solutions: `32×1×1`, `8×2×2`, `4×4×2` — planner picks a valid one.

### 13.3 Hard Namespace Separation

```text
SEMANTIC (immutable)                EXECUTION (adaptable)
────────────────────                ────────────────────
optimizer                           micro_batch
learning_rate                       workers
scheduler                           gpu_count
loss                                checkpointing
augmentation                        distributed strategy
global_batch                        kernel selection
seed                                prefetch
epochs                              cpu threads
model · dataset · precision policy  precision fallback (triggers PORTABLE)
```

**Enforcement: `execution → semantic` is forbidden.** Execution code cannot mutate semantic configuration. There is no `--reconfigure` that edits a run in place — semantic change = `mlforge fork` = new run.

---

## 14. Reproducibility & Precision Policy

`precision_policy` declared at run creation:

```yaml
precision_policy:
    preferred: bf16
    fallback: fp32
```

`bf16 → fp32` (e.g., on CPU) is allowed but **must produce `execution_mode = PORTABLE`**, never EXACT.

---

## 15. Lineage & Operations

### 15.1 Three Distinct Operations (Never Ambiguous CLI)

```bash
mlforge resume RUN       # continue an existing trajectory (same run)
mlforge finetune MODEL   # new optimization from existing model (new run, parent set)
mlforge train CONFIG     # from scratch (new run, parent null)
```

### 15.2 Lineage DAG

```bash
mlforge finetune --from model://rf_detr_s:v4 --dataset custom_v2 --config finetune.yaml
```

```text
creates run_005:
    parent_model: rf_detr_s:v4
    parent_checkpoint: sha256:...
    dataset: custom_v2
```

```text
                    ┌── finetune B
base run ── ckpt ───┼── finetune C
                    └── evaluation
```

Parent always immutable. No accidental mutation.

### 15.3 Model Artifact (Weights Alone ≠ Model)

```text
model/
├── model.safetensors
├── model_spec.json
├── architecture.json
├── preprocessing.json
├── postprocessing.json
├── label_map.json
├── normalization.json
├── environment.json
├── provenance.json        # full lineage
└── integrity.json
```

Inference is independent of training: `mlforge package` → `mlforge validate` → `mlforge infer`.

### 15.4 Evaluation References Immutable Artifacts

```bash
mlforge eval --model model://abc --dataset dataset://xyz
```

Result records `model_hash + dataset_hash + code_hash + environment_hash` → metrics are reproducible.

---

## 16. Event Journal & Source of Truth

```text
events.jsonl (append-only):
{"step":0,"event":"run_created"}
{"step":0,"event":"validation_passed"}
{"step":18492,"event":"checkpoint_committed"}
{"step":18492,"event":"hardware_migration","segment":"0002"}
```

**Metrics are observations, never state.** Training state comes only from checkpoint manifests. Secrets (API keys, tokens) never enter run_spec/environment/logs/checkpoints — only references, injected at runtime.

### 16.1 Single Authority Per Fact

| Information | Authority |
|---|---|
| Experiment semantics | `run_spec` |
| Code | source artifact |
| Dataset | dataset manifest |
| Transform | transform artifact |
| Environment | environment artifact |
| Model | model artifact |
| Training state | checkpoint |
| Hardware | execution segment |
| History | event journal |
| Metrics | metrics log |
| Lineage | lineage graph |

No duplicate sources of truth.

---

## 17. Invariant Table (Implement as Spec)

| Invariant | If violated |
|---|---|
| Run spec immutable | Block |
| Dataset hash unchanged | Block resume |
| Transform hash unchanged | Block |
| Model architecture hash unchanged | Block |
| Base model hash unchanged | Block |
| Code artifact available | Block |
| Environment reproducible | Block or explicit migration |
| Checkpoint integrity valid | Block |
| Global batch preserved | Block |
| Optimizer semantics preserved | Block |
| LR schedule state recoverable | Block |
| RNG state recoverable | Exact mode only |
| Sampler state recoverable | Exact mode only |
| Precision policy supported | Block or migration |
| GPU driver compatible | Block |
| Disk space sufficient | Block |
| Checkpoint atomicity supported | Block |
| Schema version supported | Block/migrate |
| Secrets absent from artifacts | Block packaging |

Schema versions everywhere (`schema_version: 1`). Old schemas require explicit migration: `v1 → migration → v2`. Never silently reinterpret.

---

## 18. Resume Flow (17 Steps)

```text
mlforge resume RUN
        │
        ├──  1. Verify run manifest integrity
        ├──  2. Verify schema version
        ├──  3. Verify source artifact + code hash
        ├──  4. Verify dataset identity (cryptographic)
        ├──  5. Verify transform identity
        ├──  6. Verify model architecture + base weights hash
        ├──  7. Verify environment availability (image digest)
        ├──  8. Verify driver compatibility (CUDA runtime → min driver)
        ├──  9. Detect hardware + capabilities
        ├── 10. Generate execution plan (feasibility solver)
        ├── 11. Verify global batch preserved
        ├── 12. Verify precision policy
        ├── 13. Verify distributed semantics
        ├── 14. Verify checkpoint integrity (hashes + commit marker)
        ├── 15. Restore full state (optimizer/scheduler/RNG/sampler)
        ├── 16. Create execution segment
        └── 17. Resume
```

**Any failure in 1–15 → TRAINING DOES NOT START.**

---

## 19. Full Lifecycle

```text
PROJECT SPEC → DATA REGISTER → TRANSFORM DAG → CODE ARTIFACT
    → ENV ARTIFACT → RUN SPEC (IMMUTABLE) → PREFLIGHT
        ├── FAIL → STOP
        └── PASS → EXECUTION → CHECKPOINT (atomic, hashed)
                        ├── CONTINUE → CHECKPOINT...
                        └── STOP → COMPLETE
                                          │
                                     NEW MACHINE
                                          │
                                       PREFLIGHT
                                    ┌─────┴─────┐
                                 EXACT        PORTABLE
                                    │             │
                                 RESUME     NEW SEGMENT
```

Model lifecycle:

```text
TRAIN → RUN A → MODEL A → {EVALUATE | FINETUNE → RUN B → MODEL B | EXPORT}
```

Every edge recorded in lineage.

---

## 20. CLI Surface

```bash
# Setup
mlforge init                          # create project
mlforge configure datasets            # machine-local paths (hash-verified)
mlforge preflight RUN                 # full validation before expense

# Data
mlforge prepare --model rf_detr_s     # transform DAG, cached, content-addressed

# Operations (three distinct, never ambiguous)
mlforge train --config X.yaml         # from scratch (new run)
mlforge resume RUN                    # continue trajectory (same run)
mlforge fork RUN --set lr=2e-4        # semantic change (new run, parent set)

# Inspection
mlforge inspect RUN                   # full human-readable report
mlforge validate RUN                  # the gate — PASS/FAIL per invariant
mlforge status                        # all runs, explicit actions, no auto-anything

# Downstream
mlforge eval --model model://X --dataset dataset://Y
mlforge export --run RUN --format onnx|tensorrt
mlforge package --run RUN             # inference bundle
mlforge infer --model model://X
```

`mlforge inspect RUN` output:

```text
RUN                          CODE
──────────────────────       ──────────────────────
ID:       run_01J...         Commit:   abc123
Status:   INTERRUPTED        Source:   sha256:...
Epoch:    23 / 50            Dirty:    NO

DATA                         ENVIRONMENT
──────────────────────       ──────────────────────
Train:  dataset://coco:v1    Image:    sha256:...
Hash:   sha256:...           Python:   3.11.9
Val:    dataset://cust:v2    PyTorch:  2.5.1
                              CUDA:     12.4 runtime

TRAINING                     CHECKPOINT
──────────────────────       ──────────────────────
Global batch: 32             Epoch:    23
Optimizer:     AdamW         Step:     18492
LR:            1e-4          Integrity: VALID
Scheduler:     cosine        Mode:     PORTABLE-capable
Seed:          12345
```

---

## 21. What Changed From v0.1

| v0.1 (old) | v1.0 (this spec) |
|---|---|
| Run = folder with files | Run = immutable content-addressed state; folder is materialization |
| `environment.json` + warnings | Environment artifact = OCI image digest; unknown → BLOCK |
| Dataset validated by sample count | Dataset validated by cryptographic manifest hash |
| Absolute paths in configs | Paths machine-local; identity is hash, never path |
| Transform cache by name | Transform = artifact; cache key = hash(input+transform+config+env) |
| `--reconfigure` edits semantics | Semantic change = `fork` = new run with parent (DAG) |
| One checkpoint file | Transactional multi-generation checkpoints + integrity hashes |
| Hardware config overwritten | Execution segments preserve every hardware context |
| VRAM lookup tables | Capability negotiation + feasibility solver |
| `effective_batch = micro × accum` | `global_batch = micro × accum × world_size` |
| Resume = "compatible → continue" | 17-step validation gate; any fail → no start |
| CPU: warn + estimate | CPU: same gate; PORTABLE mode only; explicit user choice |
| No code capture | Source snapshot + git commit + dirty flag in run folder |
| No RNG preservation | Full RNG hierarchy in checkpoint; Exact mode requires it |
| Training state from metrics | Metrics = observations; state = checkpoint manifest only |
| No lineage | Lineage DAG: runs → models → evaluations → exports |
| No schema versioning | `schema_version` on all artifacts; migration required |
| Warning-heavy | Formal invariant table; BLOCK vs WARN distinction |
| Single "resume" command | `train` / `resume` / `fork` — three distinct semantics |

---

## 22. Guarantees Restated

```
If MLForge cannot prove the continuation is valid,
MLForge refuses to continue.
```

That is the design principle that prevents the system from silently
corrupting a training run. Everything above — immutable specs,
cryptographic identities, transactional checkpoints, execution
segments, fail-closed validation — exists to make that refusal
correct, informative, and rare.
