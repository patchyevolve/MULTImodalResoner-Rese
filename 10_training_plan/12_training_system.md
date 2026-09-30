# Standalone ML Training System

> A hardware-agnostic, project-agnostic training platform. Install it on any OS, train on any GPU (RTX 3070 today, H100 tomorrow, CPU if desperate), and move trained models between machines via a single self-contained folder. Reusable across every future project.

---

## 1. What This Is

A standalone app (`mlforge`) — installable on any OS, self-configuring on install:

```
┌──────────────────────────────────────────────────────────────┐
│                       mlforge CLI                            │
│   init / configure / prepare / train / open / resume / ...   │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│  Hardware    │  Run Folder  │  Dataset     │  Checkpoint     │
│  Abstraction │  Manager     │  Config      │  Store          │
│  (auto-scale)│  (portable)  │  (user paths)│  (self-contained│
├──────────────┴──────┬───────┴──────────────┴─────────────────┤
│                     │  Ingestion Pipeline                     │
│                     │  (raw → model-ready, cached, validated) │
├─────────────────────┴────────────────────────────────────────┤
│                     Training Engine                           │
│          (Lightning + DDP/FSDP + config freeze)               │
├──────────────────────────────────────────────────────────────┤
│                     Evaluation + Export                        │
└──────────────────────────────────────────────────────────────┘
```

**Design principles:**
1. **Portable run folders** — every training run is one folder. Copy it to any machine, resume there.
2. **Config freeze** — training semantics never change silently. Hardware settings auto-adapt.
3. **User-configured paths** — you point at your data. No autonomous scanning, no "discovery."
4. **Self-configuring install** — one command installs the right PyTorch/CUDA/deps for your OS and GPU.
5. **Dataset-validated resume** — before resuming, system verifies you're pointing at the same data.
6. **Environment-pinned** — run folder records exact library versions; new machine installs those.
7. **Explicit decisions** — training/retraining is always your choice. System reports, you decide.

---

## 2. App Installation (Any OS, Self-Configuring)

```bash
# Linux / macOS
curl -fsSL https://mlforge.dev/install.sh | bash

# Windows (PowerShell)
irm https://mlforge.dev/install.ps1 | iex

# Or via pip (universal)
pip install mlforge
```

What the installer does:

```
[1/6] Detecting OS...           Linux x86_64 (or Windows / macOS)
[2/6] Detecting GPU...          NVIDIA RTX 3070, CUDA 12.4 detected
                                (or: no GPU → CPU-only mode)
[3/6] Installing PyTorch...     torch 2.5.1+cu124 (matches your CUDA)
[4/6] Installing dependencies... lightning, opencv, scikit-learn, ...
[5/6] Creating config dir...    ~/.mlforge/
[6/6] Running smoke test...     GPU matrix multiply OK

mlforge 1.0.0 installed.
Run `mlforge init` to create your first project.
```

No manual CUDA setup. No conda environments. No "which PyTorch version do I need." The installer figures it out from your hardware.

---

## 3. The Portable Run Folder (Core Concept)

**Every training run is ONE folder. It contains everything needed to continue training on any machine.**

```
runs/rf_detr_s_20260930_1430/
│
├── checkpoint.ckpt              # Model weights + optimizer state + scheduler state + epoch
│                                #   (PyTorch Lightning format — hardware-agnostic)
│
├── training_config.yaml         # 🔒 SEMANTIC config — FROZEN after training starts
│                                #   effective_batch, learning_rate, lr_schedule,
│                                #   optimizer, losses, augmentations, seed,
│                                #   target_epochs, early_stopping
│
├── hardware_config.json         # 🔄 DISPOSABLE — last machine's settings, regenerated on open
│                                #   micro_batch, grad_accum, precision, num_workers,
│                                #   gradient_checkpointing, strategy, gpu_name
│
├── state.json                   # Where training is at
│                                #   epoch: 23, max_epochs: 50,
│                                #   metrics: {mAP: 49.1, ...},
│                                #   best_metric: 49.1, best_epoch: 25,
│                                #   status: "interrupted" | "completed" | "early_stopped"
│
├── dataset_manifest.json        # 📋 WHAT data trained this — validated on resume
│                                #   datasets: [{name, version, path_when_trained,
│                                #              sample_count, split, checksum}],
│                                #   transform_hash: "a1b2c3d4"
│
├── environment.json             # 📦 EXACT libraries that trained this (gap #2)
│                                #   python: "3.11.9",
│                                #   packages: {torch: "2.5.1+cu124",
│                                #              lightning: "2.4.0",
│                                #              numpy: "1.26.4", ...},
│                                #   captured_at: "2026-09-30T14:30:00Z"
│
├── metrics.jsonl                # Per-epoch training log (loss, lr, metrics)
│
├── wandb_export/                # Offline experiment log (if W&B used)
│
└── README.txt                   # Human-readable summary
                                   "RF-DETR-S, trained 23/50 epochs on RTX 3070,
                                    best mAP 49.1, needs coco_2017 + custom_clips"
```

**Size:** ~400MB for RF-DETR-S (130MB weights + 260MB optimizer state + configs/logs). Small enough for git-LFS, USB drive, network share, email — any transfer method.

**Portability guarantee:** `checkpoint.ckpt` is FP32 + `map_location="cpu"` on load. Trained on RTX 3070, loads identically on H100, RTX 4090, or CPU.

---

## 4. Config Freeze Model (Semantic vs Hardware)

### 4.1 The Two Types

| Type | What | Changes on new machine? | Why |
|---|---|---|---|
| 🔒 **Semantic** (`training_config.yaml`) | effective_batch, learning_rate, lr_schedule, optimizer, losses, augmentations, seed, target_epochs, early_stopping | **NEVER** silently | Training dynamics depend on these. Changing mid-run destabilizes training. Adam moments were computed under these values. |
| 🔄 **Hardware** (`hardware_config.json`) | micro_batch, grad_accum, precision, num_workers, gradient_checkpointing, strategy, gpu_name | **ALWAYS** regenerated | Just *how* this machine achieves the same training. Disposable. |

### 4.2 The Key Invariant: effective_batch Never Changes

```
RTX 3070 (8GB):                       H100 (80GB):
  micro_batch = 2                       micro_batch = 32
  grad_accum  = 16                      grad_accum  = 1
  ─────────────────                     ─────────────────
  effective   = 2 × 16 = 32             effective   = 32 × 1 = 32
                                       
  ←── SAME gradient statistics per optimizer step ──→
  ←── SAME LR meaning ──→
  ←── Adam moments still valid ──→
```

The training process doesn't know hardware changed. Only the mechanics of *how* each optimizer step is computed changed.

### 4.3 Adaptation Flow (When You Open a Run Folder on a New Machine)

```
$ mlforge open runs/rf_detr_s_20260930_1430/

Reading run folder...
  Model: rf_detr_s | Epoch: 23/50 | Best mAP: 49.1
  Status: interrupted (saved at epoch 23)
  Trained on: RTX 3070 8GB

Detecting current hardware...
  GPU: H100 80GB | CUDA 12.6 | 1 GPU

Config adaptation:
  ┌───────────────────────┬──────────┬──────────┬───────────┐
  │ Setting               │ Stored   │ Current  │ Type      │
  ├───────────────────────┼──────────┼──────────┼───────────┤
  │ micro_batch           │ 2        │ 32       │ 🔄 hw     │
  │ grad_accum            │ 16       │ 1        │ 🔄 hw     │
  │ effective_batch       │ 32       │ 32       │ 🔒 frozen │
  │ learning_rate         │ 1e-4     │ 1e-4     │ 🔒 frozen │
  │ lr_schedule position  │ epoch 23 │ epoch 23 │ 🔒 frozen │
  │ optimizer (Adam) state│ loaded   │ loaded   │ 🔒 frozen │
  │ precision             │ bf16     │ bf16     │ 🔄 hw     │
  │ gradient_checkpointing│ ON       │ OFF      │ 🔄 hw     │
  │ num_workers           │ 8        │ 16       │ 🔄 hw     │
  └───────────────────────┴──────────┴──────────┴───────────┘

  effective_batch unchanged → training dynamics preserved ✅

Validating datasets (gap #1)...
  Manifest says: coco_2017 v1 (118,287 train), custom_clips v1 (200)
  
  Checking paths from your dataset config...
  coco_2017:      /home/user/data/coco/      → 118,287 samples ✅ match
  custom_clips:   /home/user/data/custom/    → 200 samples ✅ match

Environment check (gap #2)...
  Folder expects: python 3.11, torch 2.5.1+cu124, lightning 2.4.0
  Current:        python 3.12, torch 2.5.1+cu126, lightning 2.4.0
  ⚠️ Minor version differences detected.
     torch 2.5.1+cu124 → 2.5.1+cu126: compatible (minor CUDA patch) ✅
     python 3.11 → 3.12: compatible ✅
  (System checks compatibility, not exact match — CUDA patch versions differ by GPU)

Resume from epoch 23? [Y/n]: y

Starting training...
  Epoch 24/50 | mAP 49.3 | lr 7.5e-5 | 8.2 it/s (was 2.1 it/s on 3070)
```

### 4.4 CPU-Only Machine

```
$ mlforge open runs/rf_detr_s_20260930_1430/
  (on a machine with no GPU)

Reading run folder...
  Model: rf_detr_s | Epoch: 23/50 | Best mAP: 49.1

Detecting current hardware...
  GPU: none (CPU only)

Config adaptation:
  micro_batch:       2 → 1     🔄 hw
  grad_accum:        16 → 32   🔄 hw
  effective_batch:   32 → 32   🔒 frozen
  precision:         bf16 → fp32 (CPU)  🔄 hw

  ⚠️ WARNING: CPU-only training speed estimate:
     Current GPU (RTX 3070): ~2.1 it/s
     CPU estimate:           ~0.02 it/s
     Remaining 27 epochs:    estimated 17 DAYS

  Options:
    [a] Continue anyway (very slow — 17 days)
    [b] Stop here — current checkpoint usable as-is (mAP 49.1)
    [c] Cancel

  Choice:
```

### 4.5 Explicit Semantic Change (`--reconfigure`)

Only when YOU decide to change what training means:

```
$ mlforge open runs/rf_detr_s_20260930_1430/ --reconfigure effective_batch=64 lr=2e-4

  ⚠️ You are changing SEMANTIC config mid-training:
     effective_batch: 32 → 64
     learning_rate:   1e-4 → 2e-4 (linearly scaled)

  This may destabilize training:
  - Adam moments were accumulated under effective_batch=32
  - Gradient noise profile will change
  - LR schedule position (epoch 23) was computed for old LR

  Recommendation: finish current run at effective_batch=32,
                  then start a FRESH run at effective_batch=64.

  Force anyway? [y/N]:
```

### 4.6 Rules Summary

| Situation | What happens |
|---|---|
| Resume on bigger GPU | Hardware auto-upgrades, semantics locked, seamless |
| Resume on smaller GPU | Hardware downgrades, semantics locked, seamless |
| Resume on CPU | Hardware downgrades, time estimate warning, user chooses |
| Resume with wrong dataset | ❌ BLOCKED — manifest mismatch, error shown |
| Resume with compatible library versions | ✅ Proceeds (minor CUDA/python patch diffs OK) |
| Resume with incompatible library versions | ⚠️ Warning with fix instructions |
| Change effective_batch/LR | Explicit `--reconfigure`, warning, confirmation |
| Fresh start (no checkpoint) | Everything auto-configured for this machine |

---

## 5. Dataset Configuration (User-Configured Paths)

**No discovery. You tell the system where your data is. Period.**

### 5.1 One-Time Setup Per Machine

```bash
$ mlforge configure datasets

# System shows you what it needs for your project:
This project uses these datasets:

  [1] coco_2017        (detection images + annotations, ~25GB)
  [2] market1501       (person Re-ID crops, ~153MB)
  [3] custom_clips     (your annotated sports videos, ~5GB)
  [4] event_hypothesis_pairs  (labeled CSV for ranker, ~12KB)

Enter path for coco_2017: /home/daksh/data/coco
  → Checking... found 118,287 train images, 5,000 val ✅

Enter path for market1501: /home/daksh/data/market1501
  → Checking... found 15,109 train identities ✅

Enter path for custom_clips: /home/daksh/data/custom
  → Checking... found 200 annotated frames ✅

Enter path for event_hypothesis_pairs: /home/daksh/data/labels/pairs.csv
  → Checking... found 5,000 rows ✅

Saved: ~/.mlforge/datasets_multimodal_reasoner.yaml

All datasets configured. You can change paths anytime with:
  mlforge configure datasets
```

### 5.2 What Gets Saved

```yaml
# ~/.mlforge/datasets_multimodal_reasoner.yaml
# (machine-local — different on each machine)

machine: "lab-pc-3070"
os: "linux"

datasets:
  coco_2017:
    path: "/home/daksh/data/coco"
    train_split: "train2017"
    val_split: "val2017"
    annotations: "annotations/instances_train2017.json"

  market1501:
    path: "/home/daksh/data/market1501"
    train_split: "bounding_box_train"
    query: "query"
    gallery: "gallery"

  custom_clips:
    path: "/home/daksh/data/custom"
    annotations: "annotations/"

  event_hypothesis_pairs:
    path: "/home/daksh/data/labels/pairs.csv"
```

**Different machine → different paths → run `mlforge configure datasets` again.** That's the whole workflow.

### 5.3 If a Path Is Wrong

```
$ mlforge train --config rf_detr_s.yaml

Validating dataset paths...
  coco_2017: /home/daksh/data/coco → ❌ NOT FOUND
  custom_clips: /home/daksh/data/custom → ✅ found

Error: dataset 'coco_2017' not found at /home/daksh/data/coco

Fix options:
  a) Re-run: mlforge configure datasets
  b) Download: mlforge download coco_2017
  c) Edit: ~/.mlforge/datasets_multimodal_reasoner.yaml
```

No guessing. No scanning the whole disk. You get a clear error and clear fix options.

---

## 6. Ingestion: Raw Data → Model-Ready

Every model needs a different data format. This is the transform layer between "datasets you configured" and "DataLoader the model trains on."

### 6.1 Model ↔ Dataset Mapping (`ingestion.yaml`)

```yaml
# projects/multimodal_reasoner/ingestion.yaml

models:
  rf_detr_s:
    task: detection
    transform: coco_detection          # raw → RF-DETR DataLoader
    train_sources:
      - dataset: coco_2017
        split: train
      - dataset: custom_clips
        split: train
    val_sources:
      - dataset: coco_2017
        split: val
    expected: {train: 118487, val: 5000}

  osnet:
    task: reid
    transform: reid_crops
    train_sources:
      - dataset: market1501
        split: train
      - dataset: custom_reid           # extracted from custom_clips
        depends_on: rf_detr_s          # needs trained detector for crops
    val_sources:
      - dataset: market1501
        split: query
      - dataset: market1501
        split: gallery
    expected: {train: 15309}

  gbdt_ranker:
    task: ranking
    transform: tabular
    train_sources:
      - dataset: event_hypothesis_pairs
        split: train
    expected: {train: 5000}

  calibrator:
    task: calibration
    transform: calibration
    train_sources:
      - generated_from: rf_detr_s      # (logits, ground truth) from RF-DETR-S val
    depends_on: [rf_detr_s]
    expected: {train: 5000}
```

### 6.2 Transform Cache

Once transformed, data is cached. Re-running doesn't redo work:

```python
# mlforge/ingestion/cache.py
class TransformCache:
    """
    Keyed by (dataset_path, transform_name, config_hash).
    If the input data and transform config haven't changed, reuse the output.
    """
    def get(self, dataset, transform, config_hash) -> Optional[Path]:
        key = f"{dataset}__{transform}__{config_hash[:8]}"
        manifest = Path(f"data/manifests/{key}.json")
        if manifest.exists():
            out = json.loads(manifest.read_text())["output_path"]
            if Path(out).exists():
                return Path(out)       # ✅ cached, skip transform
        return None                    # ❌ need to transform
```

### 6.3 Prepare Flow

```
$ mlforge prepare --model rf_detr_s

Resolving: rf_detr_s → ingestion.yaml
  Sources: coco_2017:train + custom_clips:train

[1/4] Transform cache...
  coco_2017 → coco_detection:     ✅ cached (118,287)
  custom_clips → coco_detection:  ✅ cached (200)

[2/4] Merge: 118,287 + 200 = 118,487 train samples

[3/4] Validate (gap: data quality)...
  ✅ 500 random images load
  ✅ All bboxes within image bounds
  ✅ No duplicate image IDs

[4/4] Manifest written to data/manifests/rf_detr_s.json

Ready. (0 bytes re-processed)
```

---

## 7. Hardware Abstraction

### 7.1 Detection

```python
# mlforge/hardware/detector.py

@dataclass
class HardwareProfile:
    gpu_name: str
    gpu_vram_gb: float
    gpu_count: int
    compute_capability: str
    cpu_cores: int
    ram_gb: float
    supports_bf16: bool     # Ampere+ (SM 8.0+)
    supports_fp8: bool      # Hopper (SM 9.0+)

def detect() -> HardwareProfile: ...  # Same as before — reads torch.cuda
```

### 7.2 Auto-Config (Fresh Start Only)

Only used when training from scratch. Resuming uses the freeze model (§4).

```python
def auto_configure(hw, model_name) -> dict:
    # Batch table by VRAM tier
    batch_table = {
        80: {"rf_detr_s": 32, "osnet": 64, "rf_detr_seg_s": 16, "detrpose_s": 32},
        40: {"rf_detr_s": 16, "osnet": 32, "rf_detr_seg_s": 8,  "detrpose_s": 16},
        24: {"rf_detr_s": 8,  "osnet": 32, "rf_detr_seg_s": 4,  "detrpose_s": 16},
        16: {"rf_detr_s": 4,  "osnet": 16, "rf_detr_seg_s": 2,  "detrpose_s": 8},
        8:  {"rf_detr_s": 2,  "osnet": 8,  "rf_detr_seg_s": 1,  "detrpose_s": 4},
        0:  {"rf_detr_s": 1,  "osnet": 4,  "rf_detr_seg_s": 1,  "detrpose_s": 1},
    }
    # ... pick by vram, compute grad_accum to reach effective_batch=32,
    #     set precision, workers, strategy (same logic as before)
```

---

## 8. Training Engine

```python
# mlforge/training/engine.py

class TrainingEngine:
    def __init__(self, run_dir: str = None, config_path: str = None):
        """
        Two modes:
          run_dir given    → RESUME (load semantic config from folder, adapt hardware)
          config_path given → FRESH START (auto-configure everything for this machine)
        """
        if run_dir:
            self.mode = "resume"
            self.semantic = load_yaml(f"{run_dir}/training_config.yaml")   # 🔒 frozen
            self.state = load_json(f"{run_dir}/state.json")
            self.manifest = load_json(f"{run_dir}/dataset_manifest.json")  # gap #1
            self.environment = load_json(f"{run_dir}/environment.json")    # gap #2
            self.hw = detect()
            self.hardware = self._adapt_hardware()      # 🔄 regenerate
        else:
            self.mode = "fresh"
            self.semantic = load_yaml(config_path)
            self.hw = detect()
            self.hardware = auto_configure(self.hw, self.semantic["model"]["name"])

    def _adapt_hardware(self) -> dict:
        """Regenerate hardware settings for THIS machine, keeping effective_batch."""
        cfg = auto_configure(self.hw, self.semantic["model"]["name"])
        # Force effective_batch to match frozen semantic value
        target = self.semantic["training"]["target_effective_batch"]
        cfg["micro_batch"] = ...           # max that fits this GPU
        cfg["grad_accum"] = max(1, target // cfg["micro_batch"])
        cfg["effective_batch"] = cfg["micro_batch"] * cfg["grad_accum"]
        assert cfg["effective_batch"] == target, "effective_batch must not change"
        return cfg

    def validate_datasets(self) -> bool:
        """gap #1: verify datasets match what trained this checkpoint."""
        for entry in self.manifest["datasets"]:
            path = get_configured_path(entry["name"])       # from user's machine config
            if not path.exists():
                raise DatasetError(f"{entry['name']} not found at {path}")
            count = count_samples(path, entry["split"])
            if count != entry["sample_count"]:
                raise DatasetError(
                    f"{entry['name']}: expected {entry['sample_count']} samples, "
                    f"found {count}. Wrong dataset version?"
                )
        return True

    def validate_environment(self) -> list:
        """gap #2: check library compatibility against folder's environment.json."""
        warnings = []
        current = snapshot_environment()    # pip freeze equivalent
        for pkg, expected in self.environment["packages"].items():
            got = current.get(pkg)
            if got and not compatible(expected, got):
                warnings.append(f"{pkg}: folder={expected}, current={got}")
        return warnings

    def train(self):
        if self.mode == "resume":
            self.validate_datasets()          # gap #1 — BLOCKS if mismatch
            self.validate_environment()       # gap #2 — WARNS if incompatible
        # ... Lightning Trainer.fit() with semantic + hardware configs ...
```

---

## 9. CLI Execution Flows

### 9.1 First Time on a New Machine

```
$ mlforge configure datasets          # point at your data (§5)
$ mlforge prepare --model rf_detr_s   # transform raw → model-ready (§6)
$ mlforge train --config rf_detr_s.yaml   # fresh start (auto-config for this HW)
```

### 9.2 Training Produces a Run Folder

```
$ mlforge train --config rf_detr_s.yaml

[1/7] Fresh start — auto-configuring for RTX 3070...
      micro_batch=2 | grad_accum=16 | effective=32 | bf16

[2/7] Validating dataset paths...
      coco_2017: ✅ | custom_clips: ✅

[3/7] Preparing data (cached transforms)...
      118,487 train / 5,000 val

[4/7] Training...
      Epoch 1/50 → epoch 23/50... (interrupted by user, Ctrl+C)

[5/7] Saving run folder...
      runs/rf_detr_s_20260930_1430/
        checkpoint.ckpt ✅
        training_config.yaml ✅ (semantic frozen)
        hardware_config.json ✅ (3070 settings)
        state.json ✅ (epoch 23, status: interrupted)
        dataset_manifest.json ✅ (coco_v1: 118287, custom: 200)
        environment.json ✅ (torch 2.5.1+cu124, python 3.11.9, ...)
        metrics.jsonl ✅

[6/7] Size: 412 MB

[7/7] Resume anytime: mlforge open runs/rf_detr_s_20260930_1430/
```

### 9.3 Moving to Another Machine

```
# On old machine (3070):
$ cp -r runs/rf_detr_s_20260930_1430/ /media/usb/
# (or git push, scp, whatever — it's just a folder)

# On new machine (H100):
$ mlforge install                          # self-configuring installer (§2)
$ mlforge configure datasets               # point at data on THIS machine
  (download or copy datasets as needed)

$ mlforge open /path/to/rf_detr_s_20260930_1430/

  Reading run folder...
    Epoch 23/50 | Best mAP 49.1 | RTX 3070 → H100 adaptation (§4.3)
  
  Validating datasets (gap #1)...
    coco_2017: 118,287 ✅ match
    custom_clips: 200 ✅ match
  
  Environment check (gap #2)...
    torch 2.5.1+cu124 → 2.5.1+cu126: compatible ✅
  
  Resume from epoch 23? [Y/n]: y
  
  Training... Epoch 24/50 (8.2 it/s on H100 — was 2.1 on 3070)
```

### 9.4 Status View (Explicit Decisions)

```
$ mlforge status

═══════════════════════════════════════════════════════
 MODELS
═══════════════════════════════════════════════════════
  Model          Metric   Baseline  Status    Run Folder
  ────────────── ──────── ───────── ──────── ──────────────────────────
  rf_detr_s      mAP 49.1  53.0     ⏸ 23/50   rf_detr_s_20260930_1430
  osnet          R@1 96.1  95.0     ✅ done   osnet_20261001_0900
  gbdt_ranker    NDCG 0.87 0.85     ✅ done   gbdt_20261001_1400
  rf_detr_seg_s  —         43.0     ⬜ never  —

═══════════════════════════════════════════════════════
 AVAILABLE ACTIONS (you decide)
═══════════════════════════════════════════════════════
  Resume:      mlforge open runs/rf_detr_s_20260930_1430/
  Fresh:       mlforge train --config rf_detr_s.yaml
  Train new:   mlforge train --config rf_detr_seg_s.yaml
  Evaluate:    mlforge eval --run runs/osnet_20261001_0900
  Export:      mlforge export --run runs/osnet_20261001_0900 --format onnx
```

---

## 10. What Trains Where

### RTX 3070 8GB (Now)

| Model | micro_batch | grad_accum | effective | Est. Time |
|---|---|---|---|---|
| OSNet | 8 | 4 | 32 | ~8h |
| GBDT (CPU) | — | — | — | ~2h |
| Calibrator (CPU) | — | — | — | ~2h |
| RF-DETR-S | 2 | 16 | 32 | ~48h |
| RF-DETR-Seg-S | 1 | 32 | 32 | ~80h (defer) |

### H100 80GB (Later — Same Run Folders Resume Here)

| Model | micro_batch | grad_accum | effective | Est. Time |
|---|---|---|---|---|
| RF-DETR-S | 32 | 1 | 32 | ~3h |
| OSNet | 64 | 1 | 32* | ~1.5h |
| RF-DETR-Seg-S | 16 | 2 | 32 | ~4h |
| VLM LoRA | 4-8 | — | — | ~8h |

*effective_batch stays at frozen value — hardware just computes it faster.

---

## 11. Disk Space

| Item | Size |
|---|---|
| mlforge app + deps | ~5 GB |
| Pre-trained weights | ~5 GB |
| Core datasets | ~50 GB |
| Full datasets (optional) | ~130 GB |
| Run folders (5 models) | ~2 GB |
| Transformed/cached data | ~5 GB |
| **Total** | **~70-195 GB** |

2TB HDD on the 3070 machine is plenty.

---

## 12. Summary

| Concern | Solution |
|---|---|
| Move training between machines | One self-contained run folder — copy it, `mlforge open`, resume |
| Config mismatch (3070 batch=2 vs H100 batch=32) | Semantic config frozen, hardware config regenerated. effective_batch always preserved |
| Weaker machine (CPU) | Time estimate warning, user chooses to continue or stop |
| Want to change batch/LR mid-run | Explicit `--reconfigure`, warning, confirmation |
| Wrong dataset pointed at resume | ❌ Blocked — manifest validation (gap #1) catches sample count mismatch |
| Library version mismatch | environment.json comparison, compatibility check with warnings (gap #2) |
| Dataset paths | User configures per machine via `mlforge configure datasets`. No scanning |
| Fresh install on any OS | Self-configuring installer detects OS/GPU, installs correct PyTorch |
| Reusing already-downloaded data | Point at existing folder. Transforms cached. Nothing re-downloaded unless path is wrong |
| Who decides when to train | You. Always. System reports status, never auto-trains |
