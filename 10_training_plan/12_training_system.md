# Standalone ML Training System

> A hardware-agnostic, project-agnostic training platform. Trains any model on any GPU setup — RTX 3070 8GB today, H100 tomorrow, multi-GPU later. Reusable across every future project. This is the SYSTEM, not just configs for one model.

---

## 1. What This Is

A standalone Python package (`mlforge`) that provides:

```
┌─────────────────────────────────────────────────────────┐
│                    mlforge CLI                          │
│  mlforge train / eval / export / resume / compare       │
├─────────────┬─────────────┬─────────────┬───────────────┤
│  Hardware   │  Experiment │  Dataset    │  Checkpoint   │
│  Abstraction│  Tracker    │  Manager    │  Manager      │
├─────────────┴─────────────┴─────────────┴───────────────┤
│                  Training Engine                         │
│     (PyTorch Lightning + DDP + Auto-scaling)            │
├─────────────────────────────────────────────────────────┤
│              Evaluation Engine                           │
│         (metrics + regression detection)                │
├─────────────────────────────────────────────────────────┤
│              Export Engine                               │
│     (PyTorch → ONNX → TensorRT / CoreML)                │
└─────────────────────────────────────────────────────────┘
```

**Design principles:**
1. **Hardware-agnostic** — one config works on 8GB or 80GB, auto-scales
2. **Project-agnostic** — works for detection, pose, Re-ID, NLP, anything PyTorch
3. **Resumable** — every training run can be interrupted and resumed
4. **Reproducible** — seed everything, log everything, version everything
5. **Offline-first** — works without internet (uni systems often have no GitHub access)

---

## 2. Directory Structure (Standalone Repo)

```
mlforge/
├── mlforge/                      # Core package
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point
│   │
│   ├── hardware/                 # Hardware abstraction
│   │   ├── __init__.py
│   │   ├── detector.py           # Detect GPUs, CPU, RAM, disk
│   │   ├── profiler.py           # Benchmark GPU speed
│   │   ├── auto_config.py        # Auto-set batch_size, precision, workers
│   │   └── profiles/             # Hardware profiles
│   │       ├── rtx3070_8gb.yaml
│   │       ├── rtx4090_24gb.yaml
│   │       ├── h100_80gb.yaml
│   │       └── multi_gpu.yaml
│   │
│   ├── experiment/               # Experiment tracking
│   │   ├── __init__.py
│   │   ├── tracker.py            # W&B / MLflow / local JSON
│   │   ├── config.py             # Config loading + validation
│   │   └── compare.py            # Compare runs
│   │
│   ├── data/                     # Dataset management
│   │   ├── __init__.py
│   │   ├── registry.py           # Dataset registry (download URLs, checksums)
│   │   ├── downloader.py         # Resumable downloads
│   │   ├── preprocess.py         # Preprocessing pipelines
│   │   ├── splits.py             # Train/val/test split management
│   │   └── versioning.py         # DVC integration / manifest hashing
│   │
│   ├── training/                 # Training engine
│   │   ├── __init__.py
│   │   ├── engine.py             # Core training loop (Lightning-based)
│   │   ├── callbacks.py          # Early stopping, checkpointing, LR find
│   │   ├── precision.py          # FP16/BF16/FP8 auto-selection
│   │   ├── distributed.py        # Single-GPU / DDP / FSDP strategies
│   │   └── memory.py             # OOM recovery, gradient checkpointing
│   │
│   ├── evaluation/               # Evaluation engine
│   │   ├── __init__.py
│   │   ├── runner.py             # Run eval after training
│   │   ├── metrics.py            # mAP, Rank-1, HOTA, ECE, etc.
│   │   └── regression.py         # Compare against baseline, detect regressions
│   │
│   ├── export/                   # Export engine
│   │   ├── __init__.py
│   │   ├── onnx.py               # PyTorch → ONNX
│   │   ├── tensorrt.py           # ONNX → TensorRT (FP16/INT8)
│   │   └── verify.py             # Verify exported model matches PyTorch
│   │
│   └── checkpoints/              # Checkpoint management
│       ├── __init__.py
│       ├── store.py              # Save/load/version checkpoints
│       └── registry.py           # Track which checkpoint is "best"
│
├── projects/                     # Project-specific configs
│   ├── multimodal_reasoner/      # Our project
│   │   ├── config/
│   │   │   ├── rf_detr_s.yaml
│   │   │   ├── detrpose_s.yaml
│   │   │   ├── osnet.yaml
│   │   │   ├── rf_detr_seg_s.yaml
│   │   │   ├── gbdt_ranker.yaml
│   │   │   └── calibrator.yaml
│   │   ├── datasets.yaml         # Dataset registry for this project
│   │   └── models/               # Custom model definitions
│   │       ├── rf_detr_wrapper.py
│   │       └── detrpose_wrapper.py
│   │
│   └── future_project/           # Any new project drops in here
│       └── config/
│
├── scripts/                      # One-off scripts
│   ├── setup.sh                  # Environment setup
│   ├── download_all.sh           # Download all datasets
│   └── benchmark_hardware.py     # Run hardware benchmarks
│
├── tests/                        # Tests
│   ├── test_hardware.py
│   ├── test_training.py
│   └── test_export.py
│
├── pyproject.toml                # Package definition
├── Makefile                      # make train, make eval, make export
└── README.md
```

---

## 3. Hardware Abstraction Layer

This is the core innovation. **You write one config. The system auto-adapts to whatever GPU it runs on.**

### 3.1 Hardware Detection

```python
# mlforge/hardware/detector.py

import torch
import psutil
import shutil
from dataclasses import dataclass

@dataclass
class HardwareProfile:
    gpu_name: str
    gpu_vram_gb: float
    gpu_count: int
    compute_capability: str    # e.g. "8.6" for RTX 3070
    cpu_cores: int
    ram_gb: float
    disk_free_gb: float
    has_tensor_cores: bool
    supports_bf16: bool        # Ampere+ (SM 8.0+)
    supports_fp8: bool         # Hopper (SM 9.0+)

def detect() -> HardwareProfile:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_mem / 1e9
        cc = f"{props.major}.{props.minor}"
        return HardwareProfile(
            gpu_name=props.name,
            gpu_vram_gb=round(vram, 1),
            gpu_count=torch.cuda.device_count(),
            compute_capability=cc,
            cpu_cores=psutil.cpu_count(logical=True),
            ram_gb=round(psutil.virtual_memory().total / 1e9, 1),
            disk_free_gb=round(shutil.disk_usage('/').free / 1e9, 1),
            has_tensor_cores=True,
            supports_bf16=props.major >= 8,      # Ampere+
            supports_fp8=props.major >= 9,        # Hopper+
        )
    else:
        return HardwareProfile(
            gpu_name="CPU", gpu_vram_gb=0, gpu_count=0,
            compute_capability="0.0", cpu_cores=psutil.cpu_count(logical=True),
            ram_gb=round(psutil.virtual_memory().total / 1e9, 1),
            disk_free_gb=round(shutil.disk_usage('/').free / 1e9, 1),
            has_tensor_cores=False, supports_bf16=False, supports_fp8=False,
        )
```

### 3.2 Auto-Configuration

```python
# mlforge/hardware/auto_config.py

def auto_configure(hw: HardwareProfile, model_size_mb: float, task: str) -> dict:
    """
    Given hardware + model size, compute optimal training config.
    One config works everywhere — this function adapts it.
    """
    config = {}

    # --- Precision ---
    if hw.supports_fp8:
        config["precision"] = "bf16"       # H100: BF16 preferred
    elif hw.supports_bf16:
        config["precision"] = "bf16"       # 3070/4090: BF16 (safer than FP16)
    else:
        config["precision"] = "16"         # Older GPUs: FP16 with GradScaler

    # --- Batch size (auto-scaled to VRAM) ---
    vram = hw.gpu_vram_gb
    if vram >= 80:          # H100 80GB
        config["batch_size"] = {"rf_detr_s": 32, "osnet": 64, "detrpose_s": 32, "rf_detr_seg_s": 16}
    elif vram >= 40:        # A100 40GB / RTX 6000
        config["batch_size"] = {"rf_detr_s": 16, "osnet": 32, "detrpose_s": 16, "rf_detr_seg_s": 8}
    elif vram >= 24:        # RTX 4090 24GB
        config["batch_size"] = {"rf_detr_s": 8, "osnet": 32, "detrpose_s": 16, "rf_detr_seg_s": 4}
    elif vram >= 16:        # RTX 5070 Ti / 4080 16GB
        config["batch_size"] = {"rf_detr_s": 4, "osnet": 16, "detrpose_s": 8, "rf_detr_seg_s": 2}
    elif vram >= 8:         # RTX 3070 8GB ← YOUR CURRENT SETUP
        config["batch_size"] = {"rf_detr_s": 2, "osnet": 8, "detrpose_s": 4, "rf_detr_seg_s": 1}
    else:                   # CPU only
        config["batch_size"] = {"rf_detr_s": 1, "osnet": 4, "detrpose_s": 1, "rf_detr_seg_s": 1}

    # --- Gradient accumulation (compensates for small batch) ---
    target_effective_batch = 32
    for model, bs in config["batch_size"].items():
        accum = max(1, target_effective_batch // bs)
        config.setdefault("grad_accum", {})[model] = accum

    # --- DataLoader workers ---
    config["num_workers"] = min(hw.cpu_cores // 2, 8)
    config["pin_memory"] = hw.gpu_count > 0

    # --- Distributed strategy ---
    if hw.gpu_count >= 8:
        config["strategy"] = "fsdp"        # Fully Sharded (large models, many GPUs)
    elif hw.gpu_count >= 2:
        config["strategy"] = "ddp"         # Distributed Data Parallel
    elif hw.gpu_count == 1:
        config["strategy"] = "auto"        # Single GPU
    else:
        config["strategy"] = "cpu"         # CPU fallback

    # --- Gradient checkpointing (saves memory, costs 20% speed) ---
    config["gradient_checkpointing"] = vram < 16

    # --- Optimizer ---
    config["optimizer"] = "adamw"
    config["learning_rate_scale"] = 1.0 if vram >= 24 else 0.5  # Scale LR for small batch

    # --- Checkpoint frequency (more frequent on unstable small setups) ---
    config["save_every_n_epochs"] = 1 if vram < 16 else 5
    config["keep_top_k"] = 3

    return config
```

### 3.3 Hardware Profiles (YAML)

```yaml
# mlforge/hardware/profiles/rtx3070_8gb.yaml
name: "RTX 3070 8GB (Temporary)"
vram_gb: 8.0
compute_capability: "8.6"
precision: "bf16"
max_batch_size:
  rf_detr_s: 2
  detrpose_s: 4
  osnet: 8
  rf_detr_seg_s: 1
grad_accum_required: true
gradient_checkpointing: true
strategy: "auto"
estimated_rf_detr_epochs_per_hour: 2      # ~50 epochs = 25 hours
estimated_osnet_epochs_per_hour: 8        # ~60 epochs = 7.5 hours
notes: |
  Sufficient for OSNet, GBDT, calibrator, and RF-DETR-S with batch=2 + grad accum.
  RF-DETR-Seg-S will be very slow — defer to H100 if possible.
```

```yaml
# mlforge/hardware/profiles/h100_80gb.yaml
name: "H100 80GB (Permanent)"
vram_gb: 80.0
compute_capability: "9.0"
precision: "bf16"
max_batch_size:
  rf_detr_s: 32
  detrpose_s: 32
  osnet: 64
  rf_detr_seg_s: 16
grad_accum_required: false
gradient_checkpointing: false
strategy: "auto"
estimated_rf_detr_epochs_per_hour: 15     # ~50 epochs = 3.3 hours
estimated_osnet_epochs_per_hour: 40       # ~60 epochs = 1.5 hours
supports_fp8: true
notes: |
  Everything trains fast. Can run full COCO fine-tune in hours.
  FP8 training available for VLM if needed.
```

---

## 4. Model Training Configs (Hardware-Agnostic)

Each config is written ONCE. The hardware layer auto-adjusts batch size, precision, etc.

```yaml
# projects/multimodal_reasoner/config/rf_detr_s.yaml

model:
  name: "rf_detr_s"
  class: "projects.multimodal_reasoner.models.rf_detr_wrapper:RFDETRSmall"
  pretrained: "auto"          # auto-download COCO weights
  input_size: 512             # Hardware layer may reduce on small GPUs

data:
  train: "coco_train"
  val: "coco_val"
  # Dataset manager resolves these from datasets.yaml

training:
  # These are TARGETS — hardware layer adjusts actual values
  target_epochs: 50
  target_effective_batch: 32  # Hardware layer computes micro_batch × grad_accum
  learning_rate: 1e-4
  optimizer: "adamw"
  scheduler: "cosine_with_restarts"
  early_stopping:
    patience: 10
    metric: "mAP"
    mode: "max"

  # These are FIXED regardless of hardware
  losses:
    - name: "focal"
      weight: 1.0
    - name: "l1"
      weight: 5.0
    - name: "giou"
      weight: 2.0

  augmentations:
    - type: "RandomHorizontalFlip"
      p: 0.5
    - type: "ColorJitter"
      brightness: 0.2
    - type: "RandomScale"
      scale_range: [0.8, 1.2]

  gradient_clip: 0.1
  ema: true
  ema_decay: 0.9999

evaluation:
  metrics: ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
  baseline: 53.0              # COCO AP target — regression if below

export:
  formats: ["onnx", "tensorrt_fp16"]
  input_size: [512, 512]
  batch_size: 1
```

```yaml
# projects/multimodal_reasoner/config/osnet.yaml

model:
  name: "osnet"
  class: "projects.multimodal_reasoner.models.osnet_wrapper:OSNetX10"
  pretrained: "imagenet"

data:
  train: "market1501_train"
  val: "market1501_val"

training:
  target_epochs: 60
  target_effective_batch: 32
  learning_rate: 3e-4
  optimizer: "adam"
  scheduler: "cosine"
  losses:
    - type: "triplet"
      margin: 0.3
      weight: 1.0
    - type: "cross_entropy"
      label_smoothing: 0.1
      weight: 1.0
  sampler: "balanced"
  num_instances: 4

evaluation:
  metrics: ["rank1", "rank5", "mAP"]
  baseline: 95.0              # Rank-1 target

export:
  formats: ["onnx"]
  input_size: [256, 128]
```

---

## 5. Training Engine (Hardware-Agnostic Core)

```python
# mlforge/training/engine.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from mlforge.hardware.detector import detect
from mlforge.hardware.auto_config import auto_configure

class TrainingEngine:
    def __init__(self, config_path: str, project_dir: str):
        self.config = load_yaml(config_path)
        self.hw = detect()
        self.hw_config = auto_configure(self.hw, task=self.config["model"]["name"])

    def train(self, resume_from: str = None):
        # 1. Build model
        model = self._build_model()

        # 2. Build dataloaders (with hardware-appropriate batch size)
        train_loader, val_loader = self._build_data()

        # 3. Build trainer (auto-adapts to hardware)
        trainer = pl.Trainer(
            max_epochs=self.config["training"]["target_epochs"],
            precision=self.hw_config["precision"],
            devices=self.hw.gpu_count or "auto",
            strategy=self.hw_config["strategy"],
            accumulate_grad_batches=self.hw_config["grad_accum"].get(
                self.config["model"]["name"], 1
            ),
            gradient_clip_val=self.config["training"].get("gradient_clip", 0.0),
            gradient_checkpointing=self.hw_config["gradient_checkpointing"],
            callbacks=[
                EarlyStopping(
                    monitor=self.config["evaluation"]["metrics"][0],
                    patience=self.config["training"]["early_stopping"]["patience"],
                    mode=self.config["training"]["early_stopping"]["mode"],
                ),
                ModelCheckpoint(
                    save_top_k=self.hw_config["keep_top_k"],
                    monitor=self.config["evaluation"]["metrics"][0],
                    mode=self.config["training"]["early_stopping"]["mode"],
                    every_n_epochs=self.hw_config["save_every_n_epochs"],
                ),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            logger=self._build_logger(),
            resume_from_checkpoint=resume_from,
        )

        # 4. Train
        trainer.fit(model, train_loader, val_loader)

        # 5. Evaluate
        results = trainer.test(model, val_loader)

        # 6. Check for regression
        self._check_regression(results)

        return results
```

---

## 6. What Trains Where — Migration Plan

### Phase A: RTX 3070 8GB (Days 1-10, Temporary)

**Goal:** Train everything that CAN fit, prepare everything else for H100.

| Model | Batch | Grad Accum | Effective Batch | Est. Time | Feasible? |
|---|---|---|---|---|---|
| **OSNet** | 8 | 4 | 32 | ~8 hours | ✅ Easy |
| **GBDT Ranker** | N/A (CPU) | N/A | N/A | ~2 hours | ✅ CPU |
| **Calibrator** | N/A (CPU) | N/A | N/A | ~2 hours | ✅ CPU |
| **RF-DETR-S** | 2 | 16 | 32 | ~40-50 hours | ✅ Slow but works |
| **DETRPose-S** | 4 | 8 | 32 | ~20 hours | ✅ Works |
| **RF-DETR-Seg-S** | 1 | 32 | 32 | ~80+ hours | ⚠️ Very slow |
| **VLM fine-tune** | N/A | N/A | N/A | N/A | ❌ Needs H100 |

**3070 Strategy:**
```
Day 1-2:  Environment setup + hardware benchmark + dataset download
Day 2-3:  OSNet training (8 hours) → DONE
Day 3:    GBDT ranker + calibrator (CPU, 4 hours) → DONE
Day 3-5:  RF-DETR-S training (~40-50 hours, run overnight)
Day 5-6:  DETRPose-S training (~20 hours)
Day 6-10: RF-DETR-Seg-S IF time permits (or DEFER to H100)
Day 10+:  Migrate to H100 for remaining work
```

### Phase B: H100 (Day 10+, Permanent)

**Goal:** Retrain everything at full quality, plus anything deferred.

| Model | Batch | Time | Notes |
|---|---|---|---|
| **RF-DETR-S** | 32 | ~3 hours | Retrain from scratch at full batch |
| **DETRPose-S** | 32 | ~2 hours | If needed (pretrained may suffice) |
| **OSNet** | 64 | ~1.5 hours | Retrain at full batch |
| **RF-DETR-Seg-S** | 16 | ~4 hours | Now feasible |
| **VLM LoRA** | 4-8 | ~8 hours | Domain adaptation if needed |
| **GBDT + Calibrator** | CPU | ~4 hours | Same as before |
| **LoRA adapters** | 4-8 | ~4 hours | Sports domain detection adaptation |

**H100 Strategy:**
```
Day 10-11: Migrate environment, verify H100 benchmarks
Day 11-12: Retrain RF-DETR-S at full quality (3 hours)
Day 12:    Retrain OSNet + DETRPose-S (3.5 hours)
Day 12-13: Train RF-DETR-Seg-S (4 hours)
Day 13-14: VLM LoRA domain adaptation (if needed, 8 hours)
Day 14:    GBDT + calibrator retrain (4 hours)
Day 14+:   Integration + benchmarking begins
```

### Phase C: Multi-GPU (Future, If Needed)

```
- 2× H100: DDP, batch scales linearly, 2× speed
- 4× H100: FSDP available for larger models
- Config change: strategy: "ddp" or "fsdp" (auto-detected)
- No code changes needed
```

---

## 7. Retraining / Finetuning Workflow

Retraining is not a one-time thing. The system handles ongoing model updates:

```
┌──────────────────────────────────────────────────────────┐
│                  Retraining Trigger                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Trigger Type          Detection Method     Action       │
│  ─────────────────     ────────────────     ──────       │
│  Performance drop      Eval mAP < baseline  Retrain      │
│  New data arrived      Dataset version++    Incremental   │
│  Domain shift          Eval on new domain   LoRA adapt    │
│  Scheduled             Every N weeks        Full retrain  │
│  Manual                mlforge train --new  Full retrain  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                  Retraining Modes                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Mode 1: Full Retrain                                     │
│    - Train from COCO-pretrained weights                   │
│    - All layers trainable                                 │
│    - When: baseline dropped, major data change            │
│    - Time: RF-DETR-S 3h (H100), 48h (3070)              │
│                                                          │
│  Mode 2: Fine-tune (Warm Start)                           │
│    - Train from OUR best checkpoint                       │
│    - Lower LR (1e-5), fewer epochs (10-15)               │
│    - When: small data addition, minor domain shift        │
│    - Time: RF-DETR-S 1h (H100), 12h (3070)              │
│                                                          │
│  Mode 3: LoRA Adaptation                                  │
│    - Freeze base weights, train low-rank adapters         │
│    - Only LoRA params trainable (~1% of model)            │
│    - When: new domain (e.g., basketball → tennis)         │
│    - Time: RF-DETR-S 30min (H100), 4h (3070)            │
│                                                          │
│  Mode 4: Calibrator-only Retrain                          │
│    - No GPU needed, CPU only                              │
│    - Retrain temperature + conformal on new val set       │
│    - When: calibration drift detected (ECE > 0.05)        │
│    - Time: 30 minutes                                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

```bash
# CLI commands for each mode
mlforge train --config rf_detr_s.yaml --mode full        # Full retrain
mlforge train --config rf_detr_s.yaml --mode finetune \
    --checkpoint models/checkpoints/rf_detr_s/best.pth    # Warm start
mlforge train --config rf_detr_s.yaml --mode lora \
    --checkpoint models/checkpoints/rf_detr_s/best.pth    # LoRA adapter
mlforge train --config calibrator.yaml --mode calibrator-only  # CPU only
```

---

## 8. CLI Interface

```bash
# === Setup ===
mlforge setup                                    # Install deps, detect hardware
mlforge hardware                                 # Print hardware profile
mlforge hardware --benchmark                     # Run GPU benchmarks

# === Data ===
mlforge data download --project multimodal_reasoner   # Download all datasets
mlforge data preprocess --dataset coco                # Preprocess one dataset
mlforge data status                                   # Show dataset status

# === Training ===
mlforge train --config projects/multimodal_reasoner/config/rf_detr_s.yaml
mlforge train --config ... --resume experiments/run_42/checkpoints/last.ckpt
mlforge train --config ... --hardware auto         # Explicit auto-config
mlforge train --config ... --hardware rtx3070      # Force specific profile

# === Evaluation ===
mlforge eval --config ... --checkpoint best.pth
mlforge eval --config ... --compare experiments/run_42 experiments/run_43

# === Export ===
mlforge export --checkpoint best.pth --format onnx
mlforge export --checkpoint best.pth --format tensorrt --precision fp16
mlforge export --verify                            # Verify export matches PyTorch

# === Experiment Management ===
mlforge runs list                                  # List all runs
mlforge runs compare run_42 run_43                 # Compare two runs
mlforge runs best --metric mAP                     # Find best run by metric

# === Retraining ===
mlforge retrain --check                            # Check if retraining needed
mlforge retrain --mode finetune --checkpoint ...   # Incremental retrain
```

---

## 9. Checkpoint Manager

```python
# mlforge/checkpoints/store.py

class CheckpointStore:
    """
    Manages model checkpoints across hardware transitions.
    Critical: checkpoints saved on 3070 must load on H100.
    """

    def save(self, model, optimizer, epoch, metrics, name="best"):
        """
        Save checkpoint in hardware-agnostic format:
        - Model weights (FP32, portable across GPUs)
        - Optimizer state (for resume)
        - Training metadata (epoch, metrics, config hash)
        - Hardware info (which GPU trained this)
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),       # FP32 — portable
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config_hash": self.config_hash,
            "hardware": {
                "gpu": torch.cuda.get_device_name(0),
                "vram_gb": torch.cuda.get_device_properties(0).total_mem / 1e9,
                "precision_used": "bf16",
            },
            "mlforge_version": "1.0.0",
        }
        path = f"checkpoints/{name}.ckpt"
        torch.save(checkpoint, path)

        # Also save weights-only (for inference/export)
        torch.save(model.state_dict(), f"checkpoints/{name}_weights.pth")

    def load(self, path, model, optimizer=None):
        """Load checkpoint — works regardless of which GPU saved it."""
        checkpoint = torch.load(path, map_location="cpu")  # CPU first, then move
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def migrate(self, from_hw: str, to_hw: str):
        """
        Verify checkpoints trained on `from_hw` work on `to_hw`.
        Usually works automatically — this is just validation.
        """
        for ckpt in self.list_checkpoints():
            self.load(ckpt, model=model)  # Test load on current hardware
            print(f"✅ {ckpt}: portable from {from_hw} to {to_hw}")
```

**Key insight:** PyTorch checkpoints are hardware-agnostic. A model trained on RTX 3070 loads identically on H100. The only thing that changes is you can now use larger batch sizes and train faster.

---

## 10. Environment Setup (Works on Any Machine)

```bash
# scripts/setup.sh — run once on any machine

#!/bin/bash
set -e

echo "=== MLForge Setup ==="

# 1. Create virtual environment
python -m venv mlforge_env
source mlforge_env/bin/activate

# 2. Install PyTorch (detect CUDA version automatically)
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' | cut -d. -f1,2)
    echo "Detected CUDA $CUDA_VERSION"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | tr -d .)
else
    echo "No GPU detected — installing CPU-only PyTorch"
    pip install torch torchvision
fi

# 3. Install mlforge
pip install -e .

# 4. Install common training deps
pip install pytorch-lightning wandb tensorboard
pip install opencv-python-headless pillow tqdm pyyaml
pip install scikit-learn scipy numpy pandas

# 5. Detect hardware
mlforge hardware

# 6. Run quick benchmark
mlforge hardware --benchmark

echo "=== Setup Complete ==="
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo "VRAM: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB" if torch.cuda.is_available() else "N/A")')"
```

---

## 11. Migration Checklist: 3070 → H100

When the H100 becomes available:

```bash
# On H100 machine:
git clone <mlforge-repo>
cd mlforge
bash scripts/setup.sh                # Same setup script
mlforge hardware                     # Verify H100 detected
mlforge hardware --benchmark         # Verify benchmarks

# Copy checkpoints from 3070 (if any trained):
rsync -avz user@3070-machine:checkpoints/ ./checkpoints/
mlforge train --config rf_detr_s.yaml --resume checkpoints/best.ckpt \
    --hardware h100                  # Resume on H100 with larger batch

# OR retrain from scratch on H100 (recommended — faster than resuming):
mlforge train --config rf_detr_s.yaml --mode full --hardware h100

# Verify checkpoint portability:
mlforge export --checkpoint checkpoints/best.ckpt --format onnx --verify
```

**What transfers:**
- ✅ All checkpoints (PyTorch format, hardware-agnostic)
- ✅ All configs (hardware-agnostic by design)
- ✅ All datasets (just copy or re-download)
- ✅ All experiment logs (W&B is cloud-based)
- ✅ All code (same repo)

**What changes:**
- Hardware profile auto-detected (3070 → H100)
- Batch sizes auto-increase (2 → 32)
- Training speed increases ~15×
- Precision may upgrade (FP16 → BF16 → FP8)

---

## 12. Disk Space Budget (Standalone System)

| Item | 3070 Machine | H100 Machine |
|---|---|---|
| MLForge code | 50 MB | 50 MB |
| Python environment | 5 GB | 5 GB |
| Pre-trained weights | 5 GB | 5 GB |
| Datasets (core) | 50 GB | 50 GB |
| Datasets (full) | 115 GB | 180 GB |
| Checkpoints | 5 GB | 10 GB |
| Experiment logs | 2 GB | 2 GB |
| **Total** | **~130 GB** | **~260 GB** |

The 2TB HDD on the 3070 machine is more than enough.

---

## 13. Summary: One System, Any Hardware

| Capability | RTX 3070 8GB | H100 80GB | Multi-GPU |
|---|---|---|---|
| OSNet training | ✅ 8 hours | ✅ 1.5 hours | ✅ <1 hour |
| RF-DETR-S training | ✅ 48 hours | ✅ 3 hours | ✅ 1.5 hours |
| DETRPose-S training | ✅ 20 hours | ✅ 2 hours | ✅ 1 hour |
| RF-DETR-Seg-S training | ⚠️ 80+ hours | ✅ 4 hours | ✅ 2 hours |
| GBDT + Calibrator | ✅ CPU, 4 hours | ✅ CPU, 4 hours | ✅ CPU, 4 hours |
| VLM LoRA | ❌ | ✅ 8 hours | ✅ 4 hours |
| Resume from 3070 on H100 | — | ✅ instant | ✅ instant |
| Retraining modes | ✅ full/finetune/lora | ✅ all modes | ✅ all modes |
| Config changes needed | ❌ auto | ❌ auto | ❌ auto |

**The system is written once. Hardware is detected automatically. Batch sizes, precision, distributed strategy all adapt. Checkpoints transfer across hardware. The 3070 is a starting point, not a limitation.**
