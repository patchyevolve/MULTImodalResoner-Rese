# Standalone ML Training System

> A hardware-agnostic, project-agnostic training platform. Trains any model on any GPU setup — RTX 3070 8GB today, H100 tomorrow, multi-GPU later. Reusable across every future project. This is the SYSTEM, not just configs for one model.

---

## 1. What This Is

A standalone Python package (`mlforge`) that provides:

```
┌──────────────────────────────────────────────────────────────┐
│                       mlforge CLI                            │
│  discover / prepare / train / eval / export / status / ...   │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│  Hardware    │  Experiment  │  Ingestion   │  Checkpoint     │
│  Abstraction │  Tracker     │  Pipeline    │  Manager        │
│  (auto-scale)│  (W&B/local) │  (raw→train) │  (portable)     │
├──────────────┴──────┬───────┴──────────────┴─────────────────┤
│                     │  Dataset Registry                       │
│                     │  (discovery, checksums, transforms,     │
│                     │   model↔dataset mapping, caching)       │
├─────────────────────┴────────────────────────────────────────┤
│                     Training Engine                           │
│          (Lightning + DDP/FSDP + auto batch/precision)        │
├──────────────────────────────────────────────────────────────┤
│                     Evaluation Engine                         │
│              (metrics + baseline regression check)            │
├──────────────────────────────────────────────────────────────┤
│                     Export Engine                             │
│           (PyTorch → ONNX → TensorRT, verified)               │
└──────────────────────────────────────────────────────────────┘
```

**Design principles:**
1. **Hardware-agnostic** — one config works on 8GB or 80GB, auto-scales
2. **Project-agnostic** — works for detection, pose, Re-ID, NLP, anything PyTorch
3. **Dataset-reuse-first** — discovers already-downloaded data, never re-downloads
4. **Explicit decisions** — you choose when to train/retrain; the system reports, never auto-acts
5. **Resumable** — every training run can be interrupted and resumed
6. **Reproducible** — seed everything, log everything, version everything
7. **Offline-first** — works without internet (uni systems often have no GitHub access)

---

## 2. Directory Structure (Standalone Repo)

```
mlforge/
├── mlforge/                      # Core package
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point + command routing
│   │
│   ├── hardware/                 # Hardware abstraction
│   │   ├── detector.py           # Detect GPUs, CPU, RAM, disk
│   │   ├── profiler.py           # Benchmark GPU speed
│   │   ├── auto_config.py        # Auto-set batch_size, precision, workers
│   │   └── profiles/             # Hardware profiles (3070, 4090, H100, multi)
│   │
│   ├── ingestion/                # Dataset ingestion pipeline  ← NEW
│   │   ├── __init__.py
│   │   ├── registry.py           # Dataset registry (URLs, checksums, paths)
│   │   ├── discovery.py          # Scan disk, detect already-downloaded data
│   │   ├── downloader.py         # Resumable downloads (only if missing)
│   │   ├── transforms/           # Raw format → model-specific format
│   │   │   ├── coco_detection.py # Raw COCO → RF-DETR DataLoader
│   │   │   ├── reid_crops.py     # Market1501 → OSNet DataLoader
│   │   │   ├── coco_keypoints.py # Raw keypoints → DETRPose DataLoader
│   │   │   ├── tabular.py        # Labeled pairs → GBDT feature matrix
│   │   │   └── calibration.py    # Val logits → calibrator dataset
│   │   ├── mapping.py            # Which datasets feed which models
│   │   ├── cache.py              # Transform cache (don't redo work)
│   │   └── validate.py           # Data integrity checks
│   │
│   ├── experiment/               # Experiment tracking
│   │   ├── tracker.py            # W&B / MLflow / local JSON
│   │   ├── config.py             # Config loading + validation
│   │   └── compare.py            # Compare runs
│   │
│   ├── training/                 # Training engine
│   │   ├── engine.py             # Core training loop (Lightning-based)
│   │   ├── callbacks.py          # Early stopping, checkpointing, LR find
│   │   ├── precision.py          # FP16/BF16/FP8 auto-selection
│   │   ├── distributed.py        # Single-GPU / DDP / FSDP strategies
│   │   └── memory.py             # OOM recovery, gradient checkpointing
│   │
│   ├── evaluation/               # Evaluation engine
│   │   ├── runner.py             # Run eval after training
│   │   ├── metrics.py            # mAP, Rank-1, HOTA, ECE, etc.
│   │   └── regression.py         # Compare against baseline, detect regressions
│   │
│   ├── export/                   # Export engine
│   │   ├── onnx.py               # PyTorch → ONNX
│   │   ├── tensorrt.py           # ONNX → TensorRT (FP16/INT8)
│   │   └── verify.py             # Verify exported model matches PyTorch
│   │
│   └── checkpoints/              # Checkpoint management
│       ├── store.py              # Save/load/version checkpoints
│       └── registry.py           # Track which checkpoint is "best"
│
├── projects/                     # Project-specific configs
│   ├── multimodal_reasoner/      # Our project
│   │   ├── config/               # Model training configs
│   │   │   ├── rf_detr_s.yaml
│   │   │   ├── detrpose_s.yaml
│   │   │   ├── osnet.yaml
│   │   │   ├── rf_detr_seg_s.yaml
│   │   │   ├── gbdt_ranker.yaml
│   │   │   └── calibrator.yaml
│   │   ├── datasets.yaml         # Dataset registry (paths, checksums, download URLs)
│   │   ├── ingestion.yaml        # Model ↔ dataset mapping (which data trains what)
│   │   └── models/               # Custom model definitions
│   │
│   └── future_project/           # Any new project drops in here
│       ├── config/
│       ├── datasets.yaml
│       └── ingestion.yaml
│
├── data/                         # Data lives here (or symlinked from ~/data)
│   ├── raw/                      # Downloaded datasets (never modified)
│   │   ├── coco/
│   │   ├── market1501/
│   │   └── custom/
│   ├── processed/                # Transformed, model-specific (cached)
│   │   ├── rf_detr_s/
│   │   ├── osnet/
│   │   └── ...
│   └── manifests/                # What's been prepared, with hashes
│
├── experiments/                  # Training runs
│   ├── run_001/
│   │   ├── config.yaml           # Snapshot of config used
│   │   ├── checkpoints/
│   │   ├── metrics.json
│   │   └── logs/
│   └── ...
│
├── models/                       # Final model artifacts
│   ├── pretrained/               # Downloaded pretrained weights
│   ├── checkpoints/              # Best checkpoints (symlink to experiments)
│   └── exported/                 # ONNX / TensorRT engines
│
├── scripts/
│   ├── setup.sh                  # Environment setup
│   └── benchmark_hardware.py     # Run hardware benchmarks
│
├── tests/
├── pyproject.toml
├── Makefile                      # make discover, make prepare, make train
└── README.md
```

---

## 3. Dataset Ingestion System

**This is how raw datasets become training data for specific models.** Every model has a different data format. The ingestion pipeline bridges the gap.

### 3.1 The Problem

```
Raw datasets on disk:              Models that need training:
┌──────────────────┐               ┌──────────────────┐
│ COCO 2017         │               │ RF-DETR-S         │ ← needs COCO bbox format
│ (images + JSON)   │──┐            │ DETRPose-S        │ ← needs COCO keypoints format
├──────────────────┤  │            │ OSNet             │ ← needs person crops + IDs
│ Market1501        │──┤            │ RF-DETR-Seg-S     │ ← needs COCO polygon masks
│ (crops + IDs)     │  │            │ GBDT ranker       │ ← needs tabular features
├──────────────────┤  ├─INGEST──►  │ Calibrator        │ ← needs (logits, ground truth)
│ Custom clips      │──┤            └──────────────────┘
│ (videos + anno)   │  │
├──────────────────┤  │            Each model needs a DIFFERENT format.
│ SoccerNet         │──┘            Ingestion transforms raw → model-ready.
│ (frames + JSON)   │
└──────────────────┘
```

### 3.2 Dataset Registry (`datasets.yaml`)

Declares what datasets exist, where they are, and how to get them. The system **checks disk first** — if it's already there, it skips download.

```yaml
# projects/multimodal_reasoner/datasets.yaml

datasets:
  coco_2017:
    description: "COCO 2017 train+val (detection + segmentation)"
    path: "~/data/coco/"              # Where it should be on disk
    format: "coco"                    # Native format
    size_gb: 25
    checksum: "sha256:a1b2c3..."     # Verify integrity (optional but recommended)
    download:
      urls:
        - "http://images.cocodataset.org/zips/train2017.zip"
        - "http://images.cocodataset.org/zips/val2017.zip"
        - "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
      extract_to: "~/data/coco/"

  market1501:
    description: "Market-1501 person Re-ID benchmark"
    path: "~/data/market1501/"
    format: "reid"
    size_gb: 0.15
    checksum: "sha256:d4e5f6..."
    download:
      urls:
        - "https://drive.google.com/uc?id=1r0o7gX1Bd..."
      extract_to: "~/data/market1501/"

  custom_clips:
    description: "Our annotated sports video clips"
    path: "~/data/custom/"
    format: "coco_keypoints"          # We annotated in COCO format
    size_gb: 4.8
    checksum: null                    # Our own data — no official checksum
    download: null                    # Already on disk (we made it)

  event_hypothesis_pairs:
    description: "Labeled event-hypothesis pairs for ranker training"
    path: "~/data/labels/hypothesis_pairs.csv"
    format: "tabular"
    size_gb: 0.01
    checksum: null
    download: null
```

### 3.3 Ingestion Map (`ingestion.yaml`)

**This is the model ↔ dataset mapping.** It answers: "Which datasets train which model, in what format?"

```yaml
# projects/multimodal_reasoner/ingestion.yaml

models:
  rf_detr_s:
    task: detection
    input_format: coco_detection       # Transform target
    train_sources:
      - dataset: coco_2017
        split: train
        role: primary                  # Main training data
      - dataset: custom_clips
        split: train
        role: domain                   # Domain-specific addition
        annotations: bbox_only         # Use only bbox annotations
    val_sources:
      - dataset: coco_2017
        split: val
    transform: coco_detection
    expected_train_samples: 118487     # 118287 COCO + 200 custom
    expected_val_samples: 5000

  rf_detr_seg_s:
    task: segmentation
    input_format: coco_segmentation
    train_sources:
      - dataset: coco_2017
        split: train
        role: primary
      - dataset: custom_clips
        split: train
        role: domain
        annotations: polygons_only
    val_sources:
      - dataset: coco_2017
        split: val
    transform: coco_segmentation
    expected_train_samples: 118487

  detrpose_s:
    task: pose
    input_format: coco_keypoints
    train_sources:
      - dataset: coco_2017
        split: train
        annotation_type: keypoints
      - dataset: custom_clips
        split: train
        annotations: keypoints_only
    val_sources:
      - dataset: coco_2017
        split: val
        annotation_type: keypoints
    transform: coco_keypoints
    expected_train_samples: 118487

  osnet:
    task: reid
    input_format: reid_crops
    train_sources:
      - dataset: market1501
        split: train
        role: primary
      - dataset: custom_reid           # Crops extracted from our clips
        split: train
        role: domain
        depends_on: rf_detr_s          # Need detections to extract crops
    val_sources:
      - dataset: market1501
        split: query                   # Query set
      - dataset: market1501
        split: gallery                 # Gallery set
    transform: reid_crops
    expected_train_samples: 15109

  gbdt_ranker:
    task: ranking
    input_format: tabular
    train_sources:
      - dataset: event_hypothesis_pairs
        split: train
    val_sources:
      - dataset: event_hypothesis_pairs
        split: val
    transform: tabular
    expected_train_samples: 5000

  calibrator:
    task: calibration
    input_format: logits_groundtruth
    train_sources:
      - source: generated               # Not a raw dataset — generated
        from_model: rf_detr_s           # Uses RF-DETR-S validation predictions
        from_split: val
        generates: val_logits           # (logits, ground_truth) pairs
    transform: calibration
    depends_on: [rf_detr_s]            # Must train RF-DETR-S first
    expected_train_samples: 5000
```

### 3.4 Discovery: Detecting Already-Downloaded Data

```python
# mlforge/ingestion/discovery.py

from pathlib import Path
import hashlib, json, yaml

class DatasetDiscovery:
    """
    Scan disk for registered datasets. Never re-download what exists.
    """

    def discover(self, project_dir: str) -> dict:
        registry = yaml.safe_load(open(f"{project_dir}/datasets.yaml"))
        results = {}

        for name, spec in registry["datasets"].items():
            path = Path(spec["path"]).expanduser()

            if not path.exists():
                results[name] = {"status": "MISSING", "path": str(path), "size_gb": 0}
                continue

            # Compute actual size
            size_gb = self._dir_size(path) / 1e9

            # Verify checksum if defined
            if spec.get("checksum"):
                if self._verify_checksum(path, spec["checksum"]):
                    checksum_status = "OK"
                else:
                    checksum_status = "MISMATCH"
            else:
                checksum_status = "NO_CHECKSUM"

            results[name] = {
                "status": "FOUND",
                "path": str(path),
                "size_gb": round(size_gb, 2),
                "checksum": checksum_status,
                "needs_download": False,
            }

        return results

    def plan_downloads(self, discovery_results: dict) -> list:
        """Only download what's MISSING. Skip everything already on disk."""
        return [
            name for name, r in discovery_results.items()
            if r["status"] == "MISSING"
        ]
```

### 3.5 Transform Cache: Don't Redo Work

```python
# mlforge/ingestion/cache.py

class TransformCache:
    """
    If we already transformed coco_2017 → coco_detection format,
    don't redo it. Cache keyed by (dataset_version, transform_name, config_hash).
    """

    def get(self, dataset: str, transform: str, config_hash: str) -> Optional[Path]:
        cache_key = f"{dataset}__{transform}__{config_hash[:8]}"
        manifest_path = Path(f"data/manifests/{cache_key}.json")

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            # Verify the cached data still exists
            if Path(manifest["output_path"]).exists():
                return Path(manifest["output_path"])
        return None

    def put(self, dataset: str, transform: str, config_hash: str,
            output_path: str, sample_count: int):
        cache_key = f"{dataset}__{transform}__{config_hash[:8]}"
        manifest = {
            "dataset": dataset,
            "transform": transform,
            "config_hash": config_hash,
            "output_path": output_path,
            "sample_count": sample_count,
            "created_at": datetime.utcnow().isoformat(),
        }
        Path(f"data/manifests/{cache_key}.json").write_text(json.dumps(manifest, indent=2))
```

### 3.6 Ingestion Flow (Step by Step)

```
$ mlforge data discover --project multimodal_reasoner

Scanning disk for registered datasets...

  Dataset               Path                      Status    Size
  ───────────────────── ───────────────────────── ──────── ────────
  coco_2017             ~/data/coco/              ✅ FOUND  25.2 GB
  market1501            ~/data/market1501/        ✅ FOUND  153 MB
  custom_clips          ~/data/custom/            ✅ FOUND  4.8 GB
  event_hypothesis_pairs ~/data/labels/hypothesis_pairs.csv ✅ FOUND 12 KB
  soccernet             ~/data/soccernet/         ❌ MISSING 20.0 GB

  Checksums: coco_2017 OK | market1501 OK | others: no checksum
  Disk free: 1.8 TB (enough for missing: soccernet 20 GB)

  4/5 datasets already on disk. 1 needs download.
  Run: mlforge data download --dataset soccernet  (or skip — it's optional)
```

```
$ mlforge data prepare --model rf_detr_s

Resolving ingestion pipeline for rf_detr_s...
  Declared sources: coco_2017:train (primary) + custom_clips:train (domain)

[1/5] Checking transform cache...
  coco_2017 → coco_detection_v1:      ✅ cached (118287 samples, hash match)
  custom_clips → coco_detection_v1:   ✅ cached (200 samples, hash match)

[2/5] Transform (skipped — all cached)

[3/5] Merging sources...
  coco_2017:train     118,287 samples
  custom_clips:train      200 samples
  ─────────────────────────────────
  Total train:        118,487 samples

[4/5] Validating...
  ✅ All 118,487 images loadable (random sample: 500 checked)
  ✅ All bounding boxes valid (x,y,w,h within image bounds)
  ✅ All category IDs in valid range (1-90)
  ✅ No duplicate image IDs

[5/5] Writing manifest...
  Saved: data/manifests/rf_detr_s__coco_detection__a1b2c3d4.json

Ready. 118,487 train / 5,000 val samples for rf_detr_s.
Cached transforms: 2/2 (0 bytes re-processed)
```

```
$ mlforge data prepare --model osnet

Resolving ingestion pipeline for osnet...
  Declared sources: market1501:train (primary) + custom_reid:train (domain)

[1/5] Checking transform cache...
  market1501 → reid_crops_v1:   ✅ cached (15109 samples)
  custom_reid → reid_crops_v1:  ❌ not cached

[2/5] Transforming custom_reid...
  ⚠️ custom_reid requires detections from rf_detr_s
  Checking rf_detr_s checkpoint... ✅ found (mAP 53.2)
  Extracting person crops from custom clips... 200 crops extracted
  Assigning track IDs via ByteTrack... 47 unique identities found
  Saved: data/processed/osnet/custom_reid/

[3/5] Merging sources...
  market1501:train     15,109 samples
  custom_reid:train        200 samples (47 IDs)
  ─────────────────────────────────
  Total train:         15,309 samples

[4/5] Validating...
  ✅ All crops loadable (256×128 RGB)
  ✅ Minimum 2 images per identity (triplet loss requirement)
  ✅ No identity leakage between train/val

[5/5] Writing manifest...
  Saved: data/manifests/osnet__reid_crops__e5f6a7b8.json

Ready. 15,309 train / 3,368 query / 15,913 gallery for osnet.
```

---

## 4. Hardware Abstraction Layer

**You write one config. The system auto-adapts to whatever GPU it runs on.**

### 4.1 Hardware Detection

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

### 4.2 Auto-Configuration

```python
# mlforge/hardware/auto_config.py

def auto_configure(hw: HardwareProfile, model_name: str) -> dict:
    """
    Given hardware + model name, compute optimal training config.
    One config works everywhere — this function adapts it.
    """
    config = {}

    # --- Precision ---
    if hw.supports_bf16:
        config["precision"] = "bf16"
    else:
        config["precision"] = "16"

    # --- Batch size (auto-scaled to VRAM) ---
    vram = hw.gpu_vram_gb
    batch_table = {
        80: {"rf_detr_s": 32, "osnet": 64, "detrpose_s": 32, "rf_detr_seg_s": 16},
        40: {"rf_detr_s": 16, "osnet": 32, "detrpose_s": 16, "rf_detr_seg_s": 8},
        24: {"rf_detr_s": 8,  "osnet": 32, "detrpose_s": 16, "rf_detr_seg_s": 4},
        16: {"rf_detr_s": 4,  "osnet": 16, "detrpose_s": 8,  "rf_detr_seg_s": 2},
        8:  {"rf_detr_s": 2,  "osnet": 8,  "detrpose_s": 4,  "rf_detr_seg_s": 1},
        0:  {"rf_detr_s": 1,  "osnet": 4,  "detrpose_s": 1,  "rf_detr_seg_s": 1},
    }
    for threshold in sorted(batch_table.keys(), reverse=True):
        if vram >= threshold:
            config["batch_size"] = batch_table[threshold]
            break

    # --- Gradient accumulation (compensates for small batch) ---
    target_effective_batch = 32
    bs = config["batch_size"].get(model_name, 4)
    config["micro_batch"] = bs
    config["grad_accum"] = max(1, target_effective_batch // bs)
    config["effective_batch"] = bs * config["grad_accum"]

    # --- DataLoader workers ---
    config["num_workers"] = min(hw.cpu_cores // 2, 8)
    config["pin_memory"] = hw.gpu_count > 0

    # --- Distributed strategy ---
    if hw.gpu_count >= 8:
        config["strategy"] = "fsdp"
    elif hw.gpu_count >= 2:
        config["strategy"] = "ddp"
    elif hw.gpu_count == 1:
        config["strategy"] = "auto"
    else:
        config["strategy"] = "cpu"

    # --- Memory optimizations ---
    config["gradient_checkpointing"] = vram < 16

    # --- Optimizer ---
    config["optimizer"] = "adamw"
    config["learning_rate_scale"] = 1.0 if vram >= 24 else 0.5

    # --- Checkpoint frequency ---
    config["save_every_n_epochs"] = 1 if vram < 16 else 5
    config["keep_top_k"] = 3

    return config
```

### 4.3 Hardware Profiles (YAML)

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
estimated_rf_detr_epochs_per_hour: 2
estimated_osnet_epochs_per_hour: 8
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
estimated_rf_detr_epochs_per_hour: 15
estimated_osnet_epochs_per_hour: 40
supports_fp8: true
notes: |
  Everything trains fast. Can run full COCO fine-tune in hours.
  FP8 training available for VLM if needed.
```

---

## 5. Model Training Configs (Hardware-Agnostic)

Each config is written ONCE. The hardware layer auto-adjusts batch size, precision, etc. The `data` section references the ingestion map — not raw paths.

```yaml
# projects/multimodal_reasoner/config/rf_detr_s.yaml

model:
  name: "rf_detr_s"
  class: "projects.multimodal_reasoner.models.rf_detr_wrapper:RFDETRSmall"
  pretrained: "auto"          # auto-load COCO checkpoint (cached in models/pretrained/)
  input_size: 512

data:
  # References ingestion.yaml — system resolves which datasets to use
  recipe: "rf_detr_s"         # Looks up rf_detr_s in ingestion.yaml
  # Resolution: coco_2017:train + custom_clips:train → transform → DataLoader

training:
  target_epochs: 50
  target_effective_batch: 32  # Hardware layer computes micro_batch × grad_accum
  learning_rate: 1e-4
  optimizer: "adamw"
  scheduler: "cosine_with_restarts"
  early_stopping:
    patience: 10
    metric: "mAP"
    mode: "max"

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
  baseline: 53.0              # Regression check target

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
  recipe: "osnet"             # Resolves market1501:train + custom_reid:train

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
  baseline: 95.0

export:
  formats: ["onnx"]
  input_size: [256, 128]
```

---

## 6. Training Engine (Hardware-Agnostic Core)

```python
# mlforge/training/engine.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from mlforge.hardware.detector import detect
from mlforge.hardware.auto_config import auto_configure
from mlforge.ingestion.mapping import resolve_data_pipeline

class TrainingEngine:
    def __init__(self, config_path: str, project_dir: str):
        self.config = load_yaml(config_path)
        self.project_dir = project_dir
        self.hw = detect()
        model_name = self.config["model"]["name"]
        self.hw_config = auto_configure(self.hw, model_name)

    def train(self, resume_from: str = None, mode: str = "full"):
        # 1. Resolve datasets via ingestion map
        #    e.g. rf_detr_s → coco_2017:train + custom_clips:train → transform → DataLoader
        train_loader, val_loader = resolve_data_pipeline(
            project_dir=self.project_dir,
            recipe=self.config["data"]["recipe"],
            micro_batch=self.hw_config["micro_batch"],
            num_workers=self.hw_config["num_workers"],
        )

        # 2. Build model (load pretrained or resume from checkpoint)
        model = self._build_model(mode=mode, resume_from=resume_from)

        # 3. Build trainer (auto-adapts to hardware)
        trainer = pl.Trainer(
            max_epochs=self.config["training"]["target_epochs"],
            precision=self.hw_config["precision"],
            devices=self.hw.gpu_count or "auto",
            strategy=self.hw_config["strategy"],
            accumulate_grad_batches=self.hw_config["grad_accum"],
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
        )

        # 4. Train
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)

        # 5. Evaluate
        results = trainer.test(model, val_loader)

        # 6. Check regression against baseline
        self._check_regression(results)

        return results
```

---

## 7. Training Decisions: Explicit, Never Automatic

**Training is always an explicit decision YOU make.** The system never auto-trains, auto-retrains, or auto-overwrites. It reports status, runs what you tell it to run, and warns you about regressions.

### 7.1 Status Command (The Reporting Layer)

```
$ mlforge status --project multimodal_reasoner

═══════════════════════════════════════════════════════════════
 MODEL STATUS
═══════════════════════════════════════════════════════════════

  Model           Metric   Baseline  Status   Dataset     Last Trained
  ─────────────── ──────── ───────── ──────── ─────────── ────────────
  rf_detr_s       mAP 53.2  53.0     ✅ PASS  coco_v1     2 days ago
  osnet           R@1 96.1  95.0     ✅ PASS  market_v1   3 days ago
  gbdt_ranker     NDCG 0.87 0.85     ✅ PASS  pairs_v1    3 days ago
  calibrator      ECE 0.03  <0.05    ✅ PASS  val_v1      3 days ago
  rf_detr_seg_s   —         43.0     ⬜ NOT TRAINED
  detrpose_s      —         —        ⬜ USE PRETRAINED (no fine-tune needed)

═══════════════════════════════════════════════════════════════
 DATASET STATUS
═══════════════════════════════════════════════════════════════

  Dataset                On Disk    Version  Transforms  Used By
  ────────────────────── ────────── ──────── ─────────── ─────────────────
  coco_2017              ✅ 25.2GB  v1       cached      rf_detr_s, seg
  market1501             ✅ 153MB   v1       cached      osnet
  custom_clips           ✅ 4.8GB   v1       cached      rf_detr_s, seg, osnet
  event_hypothesis_pairs ✅ 12KB    v1       cached      gbdt_ranker
  soccernet              ❌ MISSING  —        —           (optional)

═══════════════════════════════════════════════════════════════
 AVAILABLE ACTIONS (you decide which to run)
═══════════════════════════════════════════════════════════════

  Train untrained model:   mlforge train --config rf_detr_seg_s.yaml
  Retrain existing model:  mlforge train --config rf_detr_s.yaml
  Resume interrupted run:  mlforge train --config rf_detr_s.yaml --resume last
  Fine-tune on new data:   mlforge train --config rf_detr_s.yaml --mode finetune
  Check datasets:          mlforge data discover
  Evaluate a checkpoint:   mlforge eval --config rf_detr_s.yaml
```

### 7.2 Training Modes (You Choose One)

| Mode | Command | Starts From | When YOU Choose It |
|---|---|---|---|
| **Full** | `--mode full` | Pretrained weights (COCO/ImageNet) | First training, major retrain |
| **Resume** | `--resume last` | Last checkpoint of interrupted run | Training was interrupted |
| **Fine-tune** | `--mode finetune --checkpoint best` | YOUR best checkpoint | New data added, want to improve |
| **LoRA** | `--mode lora --checkpoint best` | YOUR best checkpoint (frozen base) | New domain, quick adaptation |
| **Calibrator-only** | `--mode calibrator` | CPU only, no GPU | Calibration drift, quick fix |

**No mode runs unless you type the command.** The system does not detect "oh you should retrain" and act on it. `mlforge status` tells you what's true. You decide.

### 7.3 Regression Protection (Warns, Never Overwrites)

```
$ mlforge train --config rf_detr_s.yaml

  ...training...

  Final evaluation:
    mAP: 49.1 | Baseline: 53.0 | Status: ⚠️ BELOW BASELINE (-3.9)

  Regression check:
    ⚠️ mAP 49.1 < baseline 53.0
    ⚠️ Best checkpoint NOT overwritten (existing best: run_040, mAP 53.2)
    ⚠️ This run saved as: experiments/run_043 (for comparison)

  Suggestions:
    - Increase epochs (current: 50, try: 100)
    - Check data quality (mlforge data validate --dataset coco_2017)
    - Try higher LR (current: 1e-4, try: 2e-4)
    - Check if custom_clips domain data is hurting (remove and retest)
```

The system **never silently overwrites a better model with a worse one.** You can force it with `mlforge train ... --force-overwrite-best`.

---

## 8. CLI Execution Flow: Step-by-Step Paths

### 8.1 `mlforge discover` — Find What's on Disk

```
$ mlforge discover --project multimodal_reasoner

[1/3] Loading dataset registry (datasets.yaml)
      5 datasets registered

[2/3] Scanning disk...
      ~/data/coco/              → EXISTS (25.2 GB)
      ~/data/market1501/        → EXISTS (153 MB)
      ~/data/custom/            → EXISTS (4.8 GB)
      ~/data/labels/...csv      → EXISTS (12 KB)
      ~/data/soccernet/         → MISSING

[3/3] Verifying checksums...
      coco_2017: sha256 match ✅
      market1501: sha256 match ✅

Result: 4/5 on disk. 1 missing (soccernet, optional).
Next: mlforge data download --dataset soccernet   (or skip)
```

### 8.2 `mlforge download` — Only Gets What's Missing

```
$ mlforge download --project multimodal_reasoner

Checking what's already on disk...
  coco_2017:      ✅ present, SKIP
  market1501:     ✅ present, SKIP
  custom_clips:   ✅ present, SKIP
  hypotheses:     ✅ present, SKIP
  soccernet:      ❌ missing, DOWNLOAD

Downloading soccernet (20 GB)...
  [████████░░░░░░░░░░░░] 45% — 9.2 GB / 20 GB — ETA 18m (12 MB/s)
  (resumable: if interrupted, run again — continues from where it stopped)

Extracting... done
Verifying checksum... ✅

Done. 1 downloaded, 4 skipped (already on disk).
```

### 8.3 `mlforge prepare` — Transform Raw → Model-Ready

```
$ mlforge prepare --model rf_detr_s

[1/5] Loading ingestion map (ingestion.yaml)
      rf_detr_s sources: coco_2017:train + custom_clips:train

[2/5] Checking transform cache...
      coco_2017 → coco_detection:      ✅ cached
      custom_clips → coco_detection:   ✅ cached

[3/5] Merge + split
      Train: 118,487 samples | Val: 5,000 samples

[4/5] Validate
      ✅ Images load, annotations valid, no duplicates

[5/5] Manifest written

Ready. (0 bytes re-processed — all cached)
```

### 8.4 `mlforge train` — The Full Training Path

```
$ mlforge train --config projects/multimodal_reasoner/config/rf_detr_s.yaml

[1/9] Loading config...
      model: rf_detr_s | mode: full | project: multimodal_reasoner

[2/9] Detecting hardware...
      GPU: RTX 3070 | VRAM: 8.0 GB | CUDA: 12.x
      → micro_batch=2 | grad_accum=16 | effective_batch=32
      → precision=bf16 | grad_ckpt=ON | strategy=auto

[3/9] Resolving data pipeline...
      recipe: rf_detr_s → ingestion.yaml lookup
      coco_2017:train      → ✅ cached transform (118,287)
      custom_clips:train   → ✅ cached transform (200)
      val: coco_2017:val   → ✅ cached transform (5,000)

[4/9] Loading model...
      RF-DETR-S (32.1M params)
      pretrained: models/pretrained/rf_detr_s_coco.pth (cached)
      mode: full → all layers trainable

[5/9] Checkpoint check...
      --resume not given.
      Existing run found: run_042 (epoch 23/50, mAP 49.1)
      Continue from run_042? [y/N]: y

[6/9] Training (resuming from epoch 23)...
      Epoch 24/50 | loss 4.21 | mAP 48.3 | lr 8.2e-5 | 2.1 it/s
      Epoch 25/50 | loss 4.15 | mAP 49.1 | lr 7.8e-5 | 2.1 it/s
      ...
      Epoch 35/50 | loss 3.82 | mAP 52.7 | lr 4.1e-5 | 2.1 it/s
      Early stopping (no improvement for 10 epochs)
      Best epoch: 25 | Best mAP: 49.1

[7/9] Final evaluation on coco_2017:val...
      mAP: 49.1 | AP50: 68.3 | AP75: 52.1

[8/9] Regression check...
      ⚠️ 49.1 < 53.0 baseline → best NOT overwritten
      Saved as run_043 for comparison

[9/9] Logging...
      W&B: https://wandb.ai/.../run_043
      Checkpoint: experiments/run_043/checkpoints/last.ckpt

Done in 47.3h (RTX 3070, batch 2×16).
```

### 8.5 `mlforge eval` — Evaluate Without Training

```
$ mlforge eval --config rf_detr_s.yaml --checkpoint experiments/run_040/best.ckpt

[1/4] Loading model + checkpoint...
      RF-DETR-S loaded from run_040 (mAP 53.2 at save time)

[2/4] Resolving val data...
      coco_2017:val → cached (5,000 samples)

[3/4] Running inference on 5,000 images...
      [████████████████████] 100% — 5000/5000

[4/4] Computing metrics...
      mAP: 53.2 | AP50: 72.1 | AP75: 56.8
      vs baseline 53.0: ✅ PASS (+0.2)
      vs last eval (2 days ago): +0.0 (no change)
```

### 8.6 `mlforge export` — Convert for Deployment

```
$ mlforge export --checkpoint experiments/run_040/best.ckpt --format tensorrt --precision fp16

[1/4] Loading checkpoint...
      RF-DETR-S from run_040 (mAP 53.2)

[2/4] Export to ONNX...
      Saved: models/exported/rf_detr_s.onnx (130 MB)

[3/4] Convert to TensorRT FP16...
      Saved: models/exported/rf_detr_s_fp16.engine (68 MB)

[4/4] Verify parity (ONNX vs PyTorch)...
      100 random images: max diff = 0.003 (within tolerance 0.01)
      ✅ PASS — exported model matches PyTorch

Done. TensorRT engine ready for inference.
```

### 8.7 Complete First-Time Flow (Fresh Machine)

```bash
# Step 1: Setup (one time)
mlforge setup                          # Install deps, detect hardware, benchmark

# Step 2: Discover what we have
mlforge discover --project multimodal_reasoner
# → 4/5 datasets found, 1 missing

# Step 3: Download missing (optional ones can be skipped)
mlforge download --project multimodal_reasoner

# Step 4: Prepare data for first model
mlforge prepare --model rf_detr_s
mlforge prepare --model osnet
mlforge prepare --model gbdt_ranker

# Step 5: Train (in dependency order)
mlforge train --config rf_detr_s.yaml        # First (OSNet custom_reid depends on it)
mlforge train --config osnet.yaml            # Second (needs rf_detr_s for crops)
mlforge train --config gbdt_ranker.yaml      # Third (CPU, independent)
mlforge train --config calibrator.yaml       # Fourth (needs rf_detr_s val logits)
mlforge train --config rf_detr_seg_s.yaml    # Fifth (optional, slow on 3070)

# Step 6: Check everything
mlforge status --project multimodal_reasoner

# Step 7: Export for inference
mlforge export --checkpoint best_rf_detr_s --format tensorrt
mlforge export --checkpoint best_osnet --format onnx

# Step 8: Migrate to H100 (when available)
# On H100 machine: same setup.sh, rsync data/ + experiments/, retrain at full batch
mlforge train --config rf_detr_s.yaml --hardware h100
```

---

## 9. What Trains Where — Migration Plan

### Phase A: RTX 3070 8GB (Days 1-10, Temporary)

| Model | Batch | Grad Accum | Effective | Est. Time | Feasible? |
|---|---|---|---|---|---|
| **OSNet** | 8 | 4 | 32 | ~8 hours | ✅ Easy |
| **GBDT Ranker** | N/A (CPU) | N/A | N/A | ~2 hours | ✅ CPU |
| **Calibrator** | N/A (CPU) | N/A | N/A | ~2 hours | ✅ CPU |
| **RF-DETR-S** | 2 | 16 | 32 | ~40-50 hours | ✅ Slow but works |
| **DETRPose-S** | 4 | 8 | 32 | ~20 hours | ✅ Works |
| **RF-DETR-Seg-S** | 1 | 32 | 32 | ~80+ hours | ⚠️ Defer to H100 |
| **VLM fine-tune** | N/A | N/A | N/A | N/A | ❌ Needs H100 |

**3070 Strategy:**
```
Day 1-2:  Setup + discover + download datasets
Day 2-3:  OSNet training (8 hours) → DONE
Day 3:    GBDT ranker + calibrator (CPU, 4 hours) → DONE
Day 3-5:  RF-DETR-S training (~40-50 hours, run overnight)
Day 5-6:  DETRPose-S training (~20 hours)
Day 6-10: RF-DETR-Seg-S IF time permits (or DEFER)
Day 10+:  Migrate to H100
```

### Phase B: H100 (Day 10+, Permanent)

| Model | Batch | Time | Notes |
|---|---|---|---|
| **RF-DETR-S** | 32 | ~3 hours | Full retrain at full batch |
| **OSNet** | 64 | ~1.5 hours | Full retrain |
| **RF-DETR-Seg-S** | 16 | ~4 hours | Now feasible |
| **VLM LoRA** | 4-8 | ~8 hours | Domain adaptation if needed |
| **GBDT + Calibrator** | CPU | ~4 hours | Same as before |

**H100 Strategy:**
```
Day 10-11: Migrate (setup.sh + rsync data/ + experiments/)
Day 11-12: Retrain RF-DETR-S at full quality (3 hours)
Day 12:    Retrain OSNet + DETRPose-S (3.5 hours)
Day 12-13: Train RF-DETR-Seg-S (4 hours)
Day 13-14: VLM LoRA domain adaptation (if needed)
Day 14:    GBDT + calibrator retrain (4 hours)
Day 14+:   Integration + benchmarking begins
```

### Phase C: Multi-GPU (Future, If Needed)

```
- 2× H100: DDP, batch scales linearly, 2× speed
- 4× H100: FSDP available for larger models
- Config change: strategy auto-detected — no code changes
```

---

## 10. Checkpoint Manager

```python
# mlforge/checkpoints/store.py

class CheckpointStore:
    """
    Manages model checkpoints across hardware transitions.
    Critical: checkpoints saved on 3070 must load on H100.
    """

    def save(self, model, optimizer, epoch, metrics, name="best"):
        checkpoint = {
            "model_state_dict": model.state_dict(),       # FP32 — portable
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config_hash": self.config_hash,
            "hardware": {
                "gpu": torch.cuda.get_device_name(0),
                "vram_gb": torch.cuda.get_device_properties(0).total_mem / 1e9,
            },
            "mlforge_version": "1.0.0",
        }
        torch.save(checkpoint, f"checkpoints/{name}.ckpt")
        torch.save(model.state_dict(), f"checkpoints/{name}_weights.pth")

    def load(self, path, model, optimizer=None):
        """Load checkpoint — works regardless of which GPU saved it."""
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
```

**Key insight:** PyTorch checkpoints are hardware-agnostic. A model trained on RTX 3070 loads identically on H100.

---

## 11. Environment Setup (Works on Any Machine)

```bash
# scripts/setup.sh — run once on any machine

#!/bin/bash
set -e
echo "=== MLForge Setup ==="

# 1. Virtual environment
python -m venv mlforge_env
source mlforge_env/bin/activate

# 2. PyTorch (detect CUDA automatically)
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' | cut -d. -f1,2)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | tr -d .)
else
    pip install torch torchvision
fi

# 3. mlforge + deps
pip install -e .
pip install pytorch-lightning wandb tensorboard
pip install opencv-python-headless pillow tqdm pyyaml
pip install scikit-learn scipy numpy pandas

# 4. Detect + benchmark
mlforge hardware
mlforge hardware --benchmark

echo "=== Setup Complete ==="
```

---

## 12. Migration Checklist: 3070 → H100

```bash
# On H100 machine:
git clone <mlforge-repo>
cd mlforge
bash scripts/setup.sh

# Copy data (don't re-download)
rsync -avz user@3070:~/data/ ~/data/
mlforge discover --project multimodal_reasoner   # Verify all found

# Copy experiments (checkpoints)
rsync -avz user@3070:~/experiments/ ~/experiments/
mlforge status --project multimodal_reasoner      # Verify all checkpoints visible

# Retrain at full quality (recommended — faster than resuming)
mlforge train --config rf_detr_s.yaml --hardware h100
```

**What transfers:**
- ✅ All datasets (rsync — no re-download)
- ✅ All transforms (manifests + processed/ rsync)
- ✅ All checkpoints (PyTorch format, hardware-agnostic)
- ✅ All configs (hardware-agnostic by design)
- ✅ All experiment logs

**What changes:**
- Hardware auto-detected (3070 → H100)
- Batch sizes auto-increase (2 → 32)
- Training speed increases ~15×

---

## 13. Disk Space Budget

| Item | 3070 Machine | H100 Machine |
|---|---|---|
| MLForge code | 50 MB | 50 MB |
| Python environment | 5 GB | 5 GB |
| Pre-trained weights | 5 GB | 5 GB |
| Raw datasets | 50 GB | 50 GB |
| Processed/cached transforms | 5 GB | 5 GB |
| Full datasets (all optional) | 115 GB | 180 GB |
| Checkpoints | 5 GB | 10 GB |
| Experiment logs | 2 GB | 2 GB |
| **Total** | **~130 GB** | **~260 GB** |

The 2TB HDD on the 3070 machine is more than enough.

---

## 14. Summary: One System, Any Hardware

| Capability | RTX 3070 8GB | H100 80GB | Multi-GPU |
|---|---|---|---|
| OSNet training | ✅ 8 hours | ✅ 1.5 hours | ✅ <1 hour |
| RF-DETR-S training | ✅ 48 hours | ✅ 3 hours | ✅ 1.5 hours |
| DETRPose-S training | ✅ 20 hours | ✅ 2 hours | ✅ 1 hour |
| RF-DETR-Seg-S training | ⚠️ 80+ hours | ✅ 4 hours | ✅ 2 hours |
| GBDT + Calibrator | ✅ CPU, 4 hours | ✅ CPU, 4 hours | ✅ CPU, 4 hours |
| VLM LoRA | ❌ | ✅ 8 hours | ✅ 4 hours |
| Dataset reuse across machines | ✅ rsync | ✅ rsync | ✅ rsync |
| Resume from 3070 on H100 | — | ✅ instant | ✅ instant |
| Config changes needed | ❌ auto | ❌ auto | ❌ auto |

**The system is written once. Datasets are discovered, not re-downloaded. Training is explicit, never automatic. One config works on any hardware. Checkpoints and data transfer across machines.**
