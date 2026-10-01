# MLForge — Training System Architecture (v1.0)

> A hardware-agnostic, project-agnostic training platform built on **immutable artifacts, cryptographic identities, explicit execution modes, and fail-closed validation**. Install on any OS, train on any GPU, move between machines via portable run folders — with a hard guarantee: **if the system cannot prove a continuation is valid, it refuses to run.**

**Companion document:** `13_product_specification.md` — the product layer for this architecture: user mental model, exact CLI contract, full state machines, sequence diagrams for every operation, failure behavior matrix, and the status/telemetry plane. This document (12) defines **the machinery**; the product spec (13) defines **how the human drives it**. Neither exists without the other.

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
 16. At most one writer per run (leases), always.
 17. Crash recovery is deterministic: newest VALID
     checkpoint, verified before selection, skips reported.
 18. Duplicated commands never duplicate work
     (idempotent by command_id).
 19. No state is inferred from observation — metrics
     and telemetry never control execution.
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

> **The two-layer view — USER WORKFLOW → CLI/TUI/GUI (one Workflow API) → Workflow Orchestrator → the machinery below — is specified in `13_product_specification.md` §1.** CLI, TUI, and GUI are thin clients over this orchestrator; no business logic per interface.

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
├── status.json                # PROJECTION of event journal (§16.1) — not authority
├── .lease                     # run lease: single-writer guard (§23) — absent = unleased
├── heartbeat.json             # LIVE STATE: {worker_pid, host, heartbeat_at} (§16.1)
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

#### Registry Transactions (Crash-Safe Registration)

Registration is multi-step: `upload blob → calculate hash → store blob → update metadata → commit reference`. A crash between blob storage and reference commit must not leave dangling state:

```text
1. write blob to staging area (content-addressed name)
2. fsync blob
3. atomic rename into store/sha256/...
4. write registry reference to temp file
5. fsync + atomic rename into registry index
6. commit marker written
```

Recovery scan on startup: blobs without committed references are **orphans** — either completed (index rebuilt from run folders) or deleted (no referencing run). Never half-registered.

#### Garbage Collection (Reachability-Based Only)

An artifact may be collected **only if unreachable from every root**:

```text
ROOTS: projects → runs → run_spec → checkpoints → models → evaluations → exports
                                ↓
                        artifact references
                                ↓
                    store/sha256/... blobs
```

```text
GC algorithm:
1. mark phase: traverse all roots, mark reachable artifact hashes
2. sweep phase: unmarked blobs older than grace period → candidates
3. verify: no run folder references candidate (re-scan, not cached)
4. delete
```

Rules:
- **Never GC by age or size alone.** Only unreachability.
- **Never GC inside an active run's grace period** (default: 7 days after run COMPLETED).
- Checkpoints of PAUSED/INTERRUPTED/FAILED runs are always roots.
- GC is a `mlforge store gc` command — never automatic during training.

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

#### Iteration / Sampling Policy Has Its Own Identity

Same dataset bytes + same model + same code + same global batch can STILL produce a different trajectory if iteration semantics change. Therefore the training identity is a triple:

```text
TRAINING DATA IDENTITY = dataset_identity
                       + sampling_policy_identity
                       + iteration_policy_identity
```

```yaml
sampling_policy:                  # semantic — FROZEN in run_spec
    shuffle: true
    seed_policy: deterministic    # seed: 12345
    replacement: false
    drop_last: true
    class_balancing: null         # or strategy name
    ordering: deterministic
iteration_policy:                 # semantic for EXACT; recorded for PORTABLE
    shard_policy: distributed     # how dataset splits across ranks
    worker_partitioning: deterministic
    distributed_sampler: true|false
```

- **EXACT continuation** requires sampling + iteration policy identical, and sampler/dataloader state restored from checkpoint.
- **PORTABLE continuation** preserves the policy *definition* but records that worker partitioning / sharding may be recomputed for the new topology (`bitwise_reproducibility: false`).
- Changing shuffle seed, balancing, replacement, or drop_last = **semantic change → FORK**.

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

#### Host Compatibility Profile (Beyond the Image)

An OCI image does **not** fully define host execution. The complete identity is a triple:

```text
ENVIRONMENT IDENTITY = image_digest
                     + HOST COMPATIBILITY PROFILE
                     + EXECUTION CAPABILITY PROFILE
```

```yaml
host_compatibility:               # detected on each machine, compared at resume
    kernel: ">=5.15"
    gpu_driver: ">=535"           # derived from image's CUDA runtime requirement
    container_runtime: "docker|podman|apptainer"
    filesystem: "supports_atomic_rename"     # required for transactional checkpoints
    cpu_instruction_set: "x86_64 + avx2"     # or as recorded
    gpu_arch: "sm_86|sm_90"
    interconnect: "pcie|nvlink|none"         # required by distributed topology
    device_topology: "1xGPU"                 # vs "2x2"
```

Compatibility resolver chain — only at the end is execution SAFE:

```text
run requirements → environment identity → host compatibility
    → hardware capabilities → filesystem guarantees
    → distributed topology → execution plan → SAFE | BLOCK
```

Driver rule (never "probably compatible"):

```text
image CUDA runtime 12.4 → requires driver >= 535.xx
host driver 525.xx → BLOCK (with exact fix: upgrade driver to >= 535)
```

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
[PASS] Host compatibility profile (kernel 6.8, driver 550.x ≥ 535, atomic rename, x86_64)
[PASS] Distributed topology compatible (1 node × 1 GPU — unchanged)
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

### 8.4 Semantic-Invariant Matrix (Formal Classification)

"Semantic preservation" must not remain implicit. Every semantic dimension is classified:

| Dimension | Classification | Rule |
|---|---|---|
| model architecture (`architecture_hash`) | **EXACT-required** | hash must match → else BLOCK |
| model weights (checkpoint) | **EXACT-required** | integrity hash must match → else BLOCK |
| optimizer algorithm + hyperparameters | **PORTABLE-preservable** | must be identical value |
| LR + LR schedule state | **PORTABLE-preservable** | must be identical value + position |
| global batch | **PORTABLE-preservable** | `micro × accum × world_size` must equal frozen value |
| dataset identity | **EXACT-required** | manifest hash must match → else BLOCK |
| transform identity | **EXACT-required** | artifact hash must match → else BLOCK |
| loss | **PORTABLE-preservable** | identical definition |
| augmentation | **PORTABLE-preservable** | identical definition |
| sampling policy | **PORTABLE-preservable** | identical policy (see §6.2) |
| iteration/sharding policy | **PORTABLE-but-changed-and-recorded** | policy defined same; worker/shard recomputation recorded |
| precision (bf16→fp32 fallback) | **PORTABLE-but-changed-and-recorded** | triggers PORTABLE mode; recorded in segment |
| precision policy declaration | **PORTABLE-preservable** | policy itself unchanged |
| micro_batch / grad_accum | **execution — free to adapt** | never semantic |
| world_size / topology change | **PORTABLE-but-changed-and-recorded** | requires distributed topology check (§8.5); recorded as new segment |
| data ordering policy (definition) | **PORTABLE-preservable** | seed + policy frozen |
| data ordering (actual bytes/order) | **PORTABLE-but-changed-and-recorded** | unless EXACT mode with sampler state restore |
| gradient accumulation semantics | **PORTABLE-preservable** | accumulation math definition unchanged |
| training objective | **FORK-required** | any change → new run |
| seed | **PORTABLE-preservable** | value frozen |
| any other semantic field | **FORK-required** (default) | unknown dimension → FORK, never assume |

**Default rule:** a dimension not classified → treated as **FORK-required**. Unknown ≠ safe.

### 8.5 Distributed Topology Identity

`world_size = 4` does NOT make `1 node × 4 GPU` equivalent to `2 nodes × 2 GPU`. Distributed topology has its own identity:

```yaml
distributed_topology:
    node_count: 1
    gpus_per_node: 4
    world_size: 4
    rank_assignment: "static"
    gpu_mapping: ["0","1","2","3"]
    collective_backend: "nccl"
    network_fabric: "nvlink|ib|ethernet"
    nccl_config_hash: "sha256:..."
    sharding_strategy: "ddp|fsdp|zero3"
    checkpoint_sharding_format: "unsharded|fsdp_resharded"
```

**Compatibility rules:**

| Change | EXACT | PORTABLE |
|---|---|---|
| Same node count, same GPU mapping | required | allowed |
| 1×4 → 2×2 (same world_size) | **BLOCK** | allowed only if: collective backend compatible, fabric sufficient, FSDP reshard format supported, global batch preserved; recorded as new segment |
| world_size change (4 → 2) | **BLOCK** | allowed only if global batch preserved via accum; new segment |
| Backend change (nccl → gloo) | **BLOCK** | BLOCK (numerics differ) unless declared in run_spec as acceptable fallback |
| FSDP reshard incompatible | **BLOCK** | BLOCK |

Checkpoint sharding format must be loadable by the target topology or migration is BLOCKED — never silently unsharded/resharded without validation.

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

### 11.2 Multiple Generations (Identity-Based Recovery Selection)

```
checkpoints/
    ckpt-000017/   ← committed
    ckpt-000018/   ← committed
    ckpt-000019/   ← committed (keep top-K)
    ckpt-000020/   ← no commit marker (incomplete)
```

**Recovery selection is a predicate over identities — never ordinal arithmetic.** The buggy mental model `resume = N − 1` assumes ordinals are contiguous, complete, and ordered by recency. None of those hold after a crash: the checkpoint with the highest ordinal may be uncommitted, a stale directory may linger, or generations may be non-contiguous across a fork.

```text
recovery_candidates = { c : c ∈ checkpoints
                       AND c.commit_marker == true
                       AND c.manifest_hash verifies
                       AND c.checkpoint_hash verifies (all files)
                       AND c.state == COMMITTED }

resume_point = argmax over recovery_candidates of (global_step, ordinal)
             where selection = newest VALID, not newest ATTEMPTED
```

Rules:

1. **Commit marker is the sole authority for existence.** A directory without a valid commit marker does not exist for recovery purposes (regardless of ordinal).
2. **Verification before selection.** Hash verification (§11.3) runs *during* selection, not after: a corrupt candidate is skipped and reported, the next-newest valid candidate is considered.
3. **Report the recovery path explicitly:**

```text
[RECOVERY] checkpoint-23: NO COMMIT MARKER (incomplete)
[RECOVERY] checkpoint-22: MANIFEST HASH MISMATCH (corrupt — skipped)
[RECOVERY] checkpoint-21: VERIFIED (global_step=41200, valid)
RESUME POINT: checkpoint-21
recovery_from: 23 → 22 → 21 (2 skips, both recorded in event log)
```

4. **Every skip is an event** — `CHECKPOINT_SKIPPED {ordinal, hash, reason}` written to the event journal before training resumes.
5. **If NO candidate is valid** → run enters `INTERRUPTED` (§12), user is told `NO VALID CHECKPOINT — resume impossible; fork from last known state or restart`. Never silently start from step 0 while claiming "resumed".

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

> **Full state machines for project, dataset, run, and model — plus transition tables and the status/telemetry plane — are specified in `13_product_specification.md` §5 and §9.** This section covers the run machine as it relates to the execution machinery.

```text
CREATED → VALIDATING → PREPARING → READY → RUNNING
                                        ├── PAUSING → PAUSED        (user, resumable)
                                        ├── STOPPING → STOPPED      (user; no resume)
                                        ├── CHECKPOINTING
                                        ├── INTERRUPTED             (unexpected, needs reconciliation)
                                        ├── FAILED                  (recovery disposition recorded)
                                        └── COMPLETED
```

```text
Resume:    PAUSED → VALIDATING → READY → RUNNING
Recover:   INTERRUPTED → RECONCILING → (explicit resume) → VALIDATING → READY → RUNNING
Retry:     FAILED (recovery: RESUME) → VALIDATING → READY → RUNNING
Fork:      CHECKPOINT → FORK → NEW RUN → VALIDATE → TRAIN
Inference: MODEL ARTIFACT → ENV VALIDATION → SCHEMA VALIDATION → INFERENCE
```

#### Lifecycle Terms Are Never Interchangeable

| Term | Cause | Automatic continuation | Explicit recovery action |
|---|---|---|---|
| `PAUSED` | user ran `mlforge pause` | **NO** | `resume` → 19-step gate → RUNNING |
| `INTERRUPTED` | crash, power loss, kill -9, host reboot | **NO** | reconciliation scan (§12.3), then `resume` |
| `FAILED` | error stopped the run | **NO** | **per recorded `failure.recovery`**: `resume` (RESUME) or `fork`/`retrain` (FORK_ONLY) |
| `STOPPED` | user ran `mlforge stop` (graceful, final checkpoint committed) | **NO** | `retrain` / `finetune` — new run only; optional `evaluate` / `export` first |
| `COMPLETED` | training reached its end condition | **NO** | `retrain` — new run only; optional `evaluate` / `export` first |

*Automatic continuation* is `NO` for every non-`RUNNING` state — the system never transitions a run on its own. The *Explicit recovery action* column is the only legal exit. Do not read `FAILED` as "no transition out": its exit is the recorded disposition plus an explicit command plus a fresh gate (§5.3 disposition rule below).

#### FAILED Carries `failure.recovery` (One Meaning, Recorded Disposition)

`FAILED` must not carry two contradictory meanings ("no exit" vs "freely resumable"). It is a single state whose **disposition is recorded when the state is entered**:

```yaml
failure:
    cause: "CUDA OOM at step 41200"      # always recorded
    recovery: RESUME | FORK_ONLY         # decided once, at entry
    valid_checkpoint: ckpt-000021         # or null
```

```text
failure.recovery =
    FORK_ONLY   if NO valid checkpoint exists OR the cause is semantic
                (fixing it would change the experiment → must fork)
    RESUME      otherwise (valid checkpoint + cause resolvable without
                semantic change)
```

- `RESUME` → `mlforge resume` re-runs the full 19-step gate (§18). An unfixed cause blocks again — disposition never bypasses validation.
- `FORK_ONLY` → `mlforge resume` exits `3`: `NO VALID CONTINUATION — use mlforge fork or retrain.` Never silently starts from step 0.
- Pre-first-checkpoint failures (validation gate, preflight, artifact resolution) are structurally `FORK_ONLY` — there is nothing to resume.

`STOPPED` and `PAUSED` are both user-initiated but differ in intent: pause preserves resume capability; stop consumes it. `FAILED` and `INTERRUPTED` are both unexpected but differ in protocol: interrupted always recovers *through reconciliation* (resume iff a valid checkpoint exists); failed enters with an explicit disposition recorded at failure time.

#### Checkpoint Manifest Is Recovery Authority; `state.json` Is a Projection

After an unexpected event (crash, power loss, host reboot), the run's recorded state is **suspect**. On next inspection:

```text
AUTHORITY (survives crashes):
    checkpoint manifest + commit markers + content hashes
    → these are written transactionally (§11.1) and are always trusted

PROJECTION (may be stale):
    state.json, events.jsonl tail, logs
    → written asynchronously; may say RUNNING when the process is dead
```

Reconciliation rule: **never trust `state.json` alone.** A run whose `state.json` says `RUNNING` but whose supervisor reports no live process → after heartbeat timeout → `INTERRUPTED`, followed by §11.2/§12.3 recovery. The checkpoint manifest, not the status file, determines *where* training resumes; the status file only determines *what the user is told*.

**Graceful stops use the checkpoint protocol; unexpected deaths do not:**

```text
GRACEFUL (SIGINT / SIGTERM / pause / stop):
    RUNNING → STOPPING → finish safe boundary → checkpoint → commit → STOPPED|PAUSED

UNEXPECTED (crash / kill -9 / power loss / host reboot):
    RUNNING → (no checkpoint written) → heartbeat timeout detected
            → INTERRUPTED → reconciliation scan (§12.3) → resume from last COMMITTED checkpoint
```

Power loss is strictly harder than process crash: in-flight `fsync` may or may not have completed, so the checkpoint transaction protocol (§11.1) is designed to be idempotent — an incomplete write simply lacks a commit marker and is ignored. `checkpoint.tmp` remnants are swept during reconciliation.

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

### 12.3 Reconciliation After Unexpected Events (Deterministic Crash Recovery)

Crash recovery is a **deterministic function** of on-disk state — not a heuristic, not a guess. Any operator run on any machine must derive the same answer from the same run folder:

```text
reconcile(run_folder):
    1. read all checkpoint directories
    2. filter to committed + hash-valid candidates (§11.2 predicate)
    3. select newest valid → resume_point
    4. read state.json + events.jsonl tail  (projection, for reporting only)
    5. compare: is recorded state consistent with authority?
         consistent    → run state = recorded state
         inconsistent  → run state = INTERRUPTED
    6. write RECONCILED event with: prior recorded state, derived state,
       resume_point, skipped checkpoints (with reasons)
    7. emit report to user before any resume
```

Invariants:

- **Reconciliation is idempotent** — running it twice produces identical results and no duplicate events (event has deterministic `reconciliation_id = SHA256(run + authority_state)`).
- **Reconciliation never mutates checkpoints or run_spec** — it only writes the event and projects run state.
- **Reconciliation never starts training** — it only reports. Resume remains an explicit user action (§13.3, `resume = explicit`).
- A run in `INTERRUPTED` with no valid checkpoint → `FAILED` (`recovery: FORK_ONLY`) transition on reconciliation, never silent restart.

### 12.4 Process Ownership (The CLI Is Not the Parent)

Training must survive the shell that launched it:

```text
WRONG (fragile):
    $ mlforge train ...          # CLI process IS the training process
    close terminal / ssh drop → SIGSIGHUP → training dies

RIGHT (supervised):
    $ mlforge train ...          # CLI writes command, supervisord/daemon starts worker
    CLI detaches, returns "started"
    close terminal / ssh drop → worker unaffected
```

Ownership model:

```text
CLI (short-lived)  →  submits command to supervisor daemon  →  spawns training worker
                                                          ↘  heartbeat channel
supervisor daemon (long-lived, per-user, survives shell)
    - owns training worker process (restart policy: none — restarts are user decisions)
    - owns heartbeat emission (worker → status.json, 30s interval)
    - detects worker death → triggers reconciliation (§12.3)
```

Rules:

- **The training worker is never a direct child of an interactive CLI.**
- **The supervisor never auto-restarts a dead worker** — restart = resume = explicit user decision (product spec §1, core decision boundary).
- Supervisor death does not corrupt runs: worker continues; next `mlforge status` rebuilds supervisor state from run folders.
- `mlforge watch` / `status --follow` subscribe to the heartbeat channel; they never assume liveness from `state.json` alone.

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

> **Sequence diagrams for every operation (train, resume, retrain, finetune, pause, stop, prepare, evaluate, infer, export, compare) are in `13_product_specification.md` §6, with the failure behavior matrix in §7.** This section defines the operations' semantics and lineage semantics.

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

#### Inference Contract (The Exported Model Declares How It Must Be Called)

An exported model is not portable without a machine-checkable contract. `model_spec.json` carries:

```yaml
inference_contract:
    input_schema:
        - name: frame
          type: image
          dtype: uint8
          layout: "HWC"
          color_space: RGB
          resize: {width: 640, height: 640, mode: letterbox}
          normalization: {mean: [..], std: [..]}
        - name: prompt
          type: text
          tokenizer: "sha256:..."        # tokenizer artifact, not a name
          max_tokens: 4096
    output_schema:
        - name: detections
          type: structured
          schema: "json_schema://..."
    operator_set: "sha256:..."           # hash of ops the graph requires
    runtime: {framework: "onnx", opset: 17}
    dtype: fp32
    dynamic_axes: {batch: [0]}
    minimal_memory: {vram_mb: 512}
    numerical_tolerance: {atol: 1e-4}
```

Consumers (`mlforge validate`, serving, `infer`) verify against this contract **before** loading:

```text
mlforge infer --model model://abc
  → contract check: preprocessing matches? tokenizer hash matches?
  → operator set supported by runtime? dtype supported?
  → SAFE | BLOCK (with exact mismatch named)
```

A model without an inference contract cannot be packaged for reuse — `mlforge package` BLOCKS.

Inference is independent of training: `mlforge package` → `mlforge validate` → `mlforge infer`.

### 15.4 Evaluation References Immutable Artifacts

```bash
mlforge eval --model model://abc --dataset dataset://xyz
```

Result records `model_hash + dataset_hash + code_hash + environment_hash` → metrics are reproducible.

#### Evaluation Protocol Identity

Same model + same dataset can still yield incomparable numbers if the evaluation *procedure* differs. Therefore:

```text
EVALUATION IDENTITY = model_hash
                    + dataset_hash
                    + code_hash
                    + environment_hash
                    + evaluation_protocol_hash      # ← new
```

```yaml
evaluation_protocol:
    split: "val"                     # never train
    metric_definitions: "sha256:..." # metric code + parameters
    aggregation: "macro|micro|per_class"
    thresholds: {iou: 0.5, conf: 0.05}
    nms: {iou_thresh: 0.5, max_det: 100}
    sample_subset: null              # or hash of index list
    seed: 123456
    harness_code_hash: "sha256:..."
```

An evaluation result is only comparable to another if **all five identity components match**. `mlforge compare` (product spec §6.10) enforces this: mismatched protocol hashes → results marked `NOT_COMPARABLE`, never silently averaged.

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

### 16.1 Five State Concepts (Never Conflated)

| Concept | Medium | Authority | Survives crash? | Role |
|---|---|---|---|---|
| **CHECKPOINT STATE** | `checkpoints/ckpt-*/manifest.json` + commit marker + hashes | checkpoint manifest | **yes** (transactional, §11.1) | recovery point for *where* training continues |
| **RUN STATE** | `state.json` + `events.jsonl` | event journal (state = projection of journal) | partially (projection may be stale) | lifecycle position of the run |
| **LIVE STATE** | supervisor heartbeat channel | supervisor/worker process | no (process death = loss) | "is a process running *right now*" |
| **METRICS** | `metrics.jsonl` | metrics log | yes | observations — never used for control decisions |
| **EVENTS** | `events.jsonl` (append-only) | event journal | yes | history / audit trail |

Rules:

- `RUN STATE` (state.json) is a **projection**, derived from `EVENTS` + `CHECKPOINT STATE`. It can be rebuilt; never authoritative on its own.
- `LIVE STATE` is authoritative only via heartbeat freshness — a `RUNNING` state.json with expired heartbeat ⇒ `INTERRUPTED` (§12.3).
- Recovery reads `CHECKPOINT STATE`; reporting reads `RUN STATE`; liveness reads `LIVE STATE`. Mixing them up is how systems resume from the wrong step.

### 16.2 Single Authority Per Fact

| Information | Authority |
|---|---|
| Experiment semantics | `run_spec` |
| Code | source artifact |
| Dataset | dataset manifest |
| Transform | transform artifact |
| Environment | environment artifact |
| Model | model artifact |
| Training state | checkpoint manifest (+ commit marker) |
| Run lifecycle state | event journal (state.json = projection) |
| Liveness | supervisor heartbeat |
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
| Single-writer run lease held | Block second execution |
| Sampling policy unchanged | Block if changed (semantic) |
| Iteration policy compatible with topology | Record as changed or Block |
| Distributed topology compatible | Block or new segment |
| Host compatibility profile satisfied | Block (exact fix reported) |
| Newest checkpoint valid (predicate, §11.2) | Skip invalid, report recovery path |
| Inference contract present | Block packaging |
| Evaluation protocol matches for compare | Mark NOT_COMPARABLE |

Schema versions everywhere (`schema_version: 1`). Old schemas require explicit migration: `v1 → migration → v2`. Never silently reinterpret.

---

## 18. Resume Flow (19 Steps)

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
        ├── 13. Verify distributed semantics (§8.5 topology)
        ├── 14. Verify checkpoint integrity (newest-valid predicate, §11.2)
        ├── 15. ACQUIRE RUN LEASE (§23.2) — fail → BLOCK, no start
        ├── 16. REVALIDATE volatile subset (§23.3): dataset identity,
        │       disk headroom, lease ownership
        ├── 17. Restore full state (optimizer/scheduler/RNG/sampler)
        ├── 18. Create execution segment
        └── 19. Resume
```

**Any failure in 1–16 → TRAINING DOES NOT START.** Steps 15–16 close the TOCTOU window between validation and execution (§23.3); leases hold single-writer semantics for the rest.

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

> **The complete, normative CLI contract — all commands, arguments, behavior rules, exit codes, and object reference syntax — is defined in `13_product_specification.md` §4.** The surface below is the architecture-level view; the product spec is authoritative for implementation.

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
| Resume = "compatible → continue" | 19-step validation gate; any fail → no start |
| CPU: warn + estimate | CPU: same gate; PORTABLE mode only; explicit user choice |
| No code capture | Source snapshot + git commit + dirty flag in run folder |
| No RNG preservation | Full RNG hierarchy in checkpoint; Exact mode requires it |
| Training state from metrics | Metrics = observations; state = checkpoint manifest only |
| No lineage | Lineage DAG: runs → models → evaluations → exports |
| No schema versioning | `schema_version` on all artifacts; migration required |
| Warning-heavy | Formal invariant table; BLOCK vs WARN distinction |
| Single "resume" command | `train` / `resume` / `fork` — three distinct semantics |
| `N − 1` ordinal checkpoint fallback | Newest-valid identity predicate + reported recovery path (§11.2) |
| "Semantic preservation" informal | Formal semantic-invariant matrix (§8.4); unknown → FORK |
| Image digest = full environment | Environment = image + host compatibility + execution capability (§6.5) |
| world_size = topology | Full distributed topology identity (§8.5) |
| No concurrency control | Run/execution/artifact/command leases (§23) |
| Crash → "state.json says RUNNING" | Manifest authority + reconciliation → INTERRUPTED (§12.3) |
| One "state" concept | Five state concepts with per-concept authority (§16.1) |
| Integrity = security | Four separate boundaries; v1.0 scope recorded (§24) |
| Identity = run id | Five-layer identity stack (§25) |

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

---

## 23. Concurrency Control (Run & Execution Leases)

Nothing prevents two terminals from resuming the same run, or a training worker and a GC from racing on the same store. Single-writer semantics are enforced by **leases**, not conventions.

### 23.1 Lease Types

| Lease | Held by | Prevents | Lifetime | Medium |
|---|---|---|---|---|
| **run lease** | an executing worker | second process resuming/executing the same run | while worker alive; heartbeat renews | `runs/<id>/.lease` (atomic O_EXCL + PID + host + token) |
| **execution lease** | a worker on a *machine* | same machine starting a second training job when resources are reserved | per job | supervisor registry |
| **artifact lease** | GC candidate marking / prepare jobs | GC collecting an artifact mid-use | duration of operation | store registry |
| **command lease** | an idempotent command in flight | duplicate concurrent execution of same `command_id` (§23.4) | command duration | command journal |

### 23.2 Lease Protocol

```text
acquire(run_id):
    create .lease.tmp {run_id, pid, host, session_token, acquired_at, heartbeat_at}
    fsync
    atomic hard-link/rename → .lease          # fails if .lease exists

renew (heartbeat, 30s):
    update heartbeat_at only if session_token matches (owner check)

release (graceful):
    delete .lease only if session_token matches

steal (only via explicit user action):
    mlforge lease break RUN --force           # requires --yes; logs LEASE_BROKEN event
```

Rules:

- **Stale ≠ free.** A lease whose heartbeat exceeds the timeout (default 120s) is **SUSPECT**, not auto-reclaimable — it triggers reconciliation (§12.3) first: is the process actually dead? Only after confirmation may the lease be broken.
- **Lease break is always logged** — `LEASE_BROKEN {previous_owner, reason, operator}` in the event journal.
- **Hard invariant:** at most one live worker may write a run folder's checkpoints at any time. A second `mlforge resume` on a leased run → `BLOCK: run already executing (lease held by pid N on host H)`.
- **Distributed:** exactly one *global* run lease exists; per-rank workers hold **rank leases** subordinate to it, so a partial crash of rank 2/4 does not release the run lease.

### 23.3 Preflight → Acquisition → Revalidation (TOCTOU Window)

Validation at time T does not guarantee validity at time T+Δ (dataset path deleted, disk filled by another job, lease stolen):

```text
1. PREFLIGHT validate (all 17 gates)
2. acquire run lease            ← concurrency control point
3. REVALIDATE the volatile subset:
     dataset identity · disk headroom · lease ownership
     (things that can change between preflight and start)
4. execute
```

The revalidation step closes the time-of-check/time-of-use gap for everything that can change out-of-band. Immutable things (run_spec, code, environment hashes) are not re-checked — they cannot change.

### 23.4 Command Idempotency

State-changing commands carry a client-generated `command_id`:

```bash
mlforge train --config X.yaml --command-id 01JABC...
# duplicate submission:
mlforge train --config X.yaml --command-id 01JABC...
→ returns the ORIGINAL result: "command 01JABC... already completed at T, run_01JDEF"
```

| Command class | Idempotent by `command_id`? | Behavior on duplicate |
|---|---|---|
| `train`, `retrain`, `finetune` | **yes** | return original run id; never create a second run |
| `export`, `evaluate`, `package` | **yes** | return original result id; never recompute |
| `prepare` | **yes** (content-addressed result) | return cached artifact identity |
| `resume`, `stop`, `pause`, `fork` | naturally keyed by run id | second call → conflict/`already in state X` |
| `status`, `inspect`, `list` | read-only | n/a (always safe) |

Rules:
- If no `--command-id` supplied → one is generated and recorded (dedup only within that invocation's retry window).
- A command that **failed** may be retried with the same `command_id` only after recording `command_failed` — retries of failures are allowed; duplicates of *successes* are not.
- The command journal (`commands.jsonl`) is itself append-only, same as events.

---

## 24. Security Boundaries (Four Separate Questions)

Security in MLForge spans four concerns that are frequently conflated. Each has its own boundary and its own answer:

| Concern | Question | MLForge v1.0 answer |
|---|---|---|
| **Integrity** | Is this artifact what it claims to be? | **Cryptographic.** SHA-256 identity on every artifact (§6); transactional writes (§11.1); commit markers; fail-closed validation. This is the primary defense and it is fully implemented. |
| **Authentication** | Who is claiming to act? | **Local, single-user.** Runs are tied to OS user + supervisor session tokens (§23). No multi-user identity in v1.0. |
| **Authorization** | Who may perform this action? | **Local OS permissions.** Filesystem ACLs on project/store; `--force`/`--yes` operations require an interactive operator (no silent escalation). |
| **Confidentiality** | Who may read the data? | **Filesystem-level.** Secrets never enter run_spec/environment/logs/checkpoints (§16) — only references, injected at runtime. Artifacts assumed non-public (lab machine). |

Explicit non-goals for v1.0 (recorded, not implicit):

- No network-facing service; CLI + local supervisor only → no remote attack surface by design.
- No artifact signing / PKI (content hash is a *checksum*, not an authenticity proof against a malicious store) — recorded as a **v1.1+ extension point** (`integrity.json` reserved field: `signature`).
- No encryption at rest beyond OS/disk level.
- Integrity ≠ authenticity: hashes detect corruption and accidental substitution; they do not defend against an adversary who can rewrite both artifact and hash. Documented so nobody over-trusts the guarantee.

---

## 25. Identity Layers (The Conceptual Stack)

Five identity layers exist in MLForge. **They are never conflated** — most reproducibility and recovery bugs trace back to mixing two of them:

| Layer | Answers | Example value | Changes when… |
|---|---|---|---|
| **SEMANTIC IDENTITY** | *What experiment is this?* | `run_spec` hash (model, data, optimizer, losses, seed, global batch) | **only** via `fork` — never by editing or by hardware change |
| **ARTIFACT IDENTITY** | *What bytes is this?* | `sha256:def...` for dataset / checkpoint / model / environment | bytes change → new identity (even if semantics identical) |
| **EXECUTION IDENTITY** | *How was it computed?* | machine, driver, image digest, topology, precision, micro/accum/world | new hardware, new host, new segment |
| **LIFECYCLE IDENTITY** | *Where is it in its life?* | run state (RUNNING / PAUSED / INTERRUPTED…), generation, epoch/step | time passes; crashes; user actions |
| **OBSERVATION** | *What did we see?* | metrics, logs, status projections, heartbeats | continuously; **never authoritative** |

Cross-layer rules:

1. **Same SEMANTIC IDENTITY + different EXECUTION IDENTITY = one run, two segments** (§12.2) — not two runs.
2. **Different SEMANTIC IDENTITY = different run**, regardless of identical artifacts (fork).
3. **Same ARTIFACT IDENTITY does not imply same SEMANTIC IDENTITY** — two forks can start from the same checkpoint and diverge; the checkpoint hash alone never identifies the experiment.
4. **LIFECYCLE changes never alter SEMANTIC or ARTIFACT identity** — resume, pause, migrate: none rewrites the experiment.
5. **OBSERVATION is never an input to identity or control decisions** — metrics do not gate resume; heartbeats only gate *liveness reporting*.
6. **PORTABLE mode changes EXECUTION identity and records it** — semantic identity is preserved by the §8.4 matrix; bitwise identity is honestly disclaimed.

When in doubt: **name the layer first.** Most "reproducibility bugs" are layer confusion, not math.

---

## 26. Related Documents

| Document | Contains | Relationship |
|---|---|---|
| **`13_product_specification.md`** | User mental model, CLI contract, state machines, sequence diagrams, failure matrix, status/telemetry plane | **Product layer for this architecture** — how the human drives the machinery below |
| `11_model_weights_and_disk_space.md` | Model weight sizes, disk budgets, download order | Feeds preflight disk-space validation (§7.3) |
| `02_dataset_preparation.md` | Dataset research, download URLs, quality checks | Feeds dataset registration (§10) |
| `00_master_plan.md` | Training strategy, model inventory, timeline | Governs what gets trained and when |

**Reading order:** `00_master_plan` (what/why) → `02_dataset_preparation` (data) → **`12_training_system`** (machinery) → **`13_product_specification`** (interface + behavior) → implementation.
