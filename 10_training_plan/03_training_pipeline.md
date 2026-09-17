# Phase 2: Training Pipeline

> Exact training configurations, commands, and procedures for every model we fine-tune. One model at a time. Sequential. No shortcuts.

---

## Training Order

```
Day 5-7:  RF-DETR-S (detection)        ← Most critical, everything depends on this
Day 8-9:  OSNet (Re-ID)                 ← Quick, 1-2 days
Day 10-12: RF-DETR-Seg-S (segmentation) ← If time permits, skip if behind
Day 13:   GBDT Hypothesis Ranker        ← CPU only, 2-4 hours
Day 13:   Calibrator training           ← CPU only, 2-4 hours
Day 13+:  Integration + benchmarking
```

---

## Model 1: RF-DETR-S Fine-Tuning (Detection)

### Why This First

Detection is the foundation. Every downstream component (tracking, pose, reasoning) depends on detection quality. If detection fails, everything fails.

### Training Strategy

**Two-stage approach** (consistent with `01_foundations/18_training_strategy.md`):

1. **Baseline (this plan, Days 5-7):** Fine-tune from COCO-pretrained weights on COCO + custom domain. This establishes the baseline accuracy.
2. **Domain Adaptation (Phase 2+, if needed):** If sports domain accuracy is insufficient, apply LoRA adapters and fine-tune on sports-specific data.

### Training Configuration (Baseline Stage)

```yaml
# configs/rf_detr_s_finetune.yaml

model:
  name: "rf-detr-s"
  pretrained: "facebook/rf-detr-small"  # or custom checkpoint
  architecture:
    backbone: "convnext_tiny"
    transformer: "detr_transformer"
    num_queries: 300
    num_classes: 91  # 80 COCO + 11 extra
    d_model: 256
    nhead: 8
    num_encoder_layers: 6
    num_decoder_layers: 6

training:
  # Hardware optimization for RTX 4090
  epochs: 50
  batch_size: 8              # Fits in 24GB with FP16
  gradient_accumulation: 4   # Effective batch = 32
  precision: "fp16"          # Mixed precision training
  compile: true              # torch.compile for 20-30% speedup

  # Learning rate schedule
  learning_rate: 1e-4
  min_lr: 1e-6
  weight_decay: 0.0001
  warmup_epochs: 3
  scheduler: "cosine_with_hard_restarts"
  scheduler_params:
    T_0: 10
    T_mult: 2

  # Optimizer
  optimizer: "adamw"
  adamw_params:
    betas: [0.9, 0.999]
    eps: 1e-8

  # Loss
  losses:
    - "cross_entropy"        # Classification
    - "l1"                   # Bounding box regression
    - "giou"                 # Generalized IoU
  loss_weights:
    classification: 1.0
    l1: 5.0
    giou: 2.0

  # Augmentation
  augmentations:
    train:
      - type: "RandomHorizontalFlip"
        p: 0.5
      - type: "RandomScale"
        scale_range: [0.8, 1.2]
      - type: "RandomCrop"
        crop_size: [640, 640]
      - type: "ColorJitter"
        brightness: 0.2
        contrast: 0.2
        saturation: 0.2
        hue: 0.1
      - type: "MixUp"
        alpha: 0.5
        prob: 0.3
      - type: "Mosaic"
        prob: 0.5
    val:
      - type: "Resize"
        size: [640, 640]

  # Regularization
  dropout: 0.1
  label_smoothing: 0.1
 ema: true                    # Exponential moving average of weights
  ema_decay: 0.9999

  # Training tricks
  early_stopping:
    patience: 10
    metric: "mAP"
    mode: "max"
  gradient_clip: 0.1

checkpointing:
  save_dir: "models/checkpoints/rf_detr_s/"
  save_every_n_epochs: 5
  keep_top_k: 3              # Keep best 3 by mAP
  save_optimizer: true

logging:
  log_every_n_steps: 50
  wandb_project: "multimodal-video-reasoner"
  wandb_run_name: "rf-detr-s-finetune-v1"
  wandb_tags: ["detection", "rf-detr", "finetune"]

evaluation:
  val_every_n_epochs: 1
  metrics: ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
```

### Training Command

```bash
# Single GPU training
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    scripts/train_detection.py \
    --config configs/rf_detr_s_finetune.yaml \
    --gpu 0

# Or simpler (single GPU)
python scripts/train_detection.py \
    --config configs/rf_detr_s_finetune.yaml \
    --gpu 0

# Monitor training
wandb server  # Open browser to see live metrics
```

### Expected Training Metrics

| Epoch | mAP | AP50 | AP75 | Loss | LR | Time |
|---|---|---|---|---|---|---|
| 1 | ~15 | ~30 | ~10 | 4.5 | 1e-4 | ~15 min |
| 5 | ~35 | ~55 | ~30 | 2.1 | 8e-5 | ~15 min |
| 10 | ~42 | ~62 | ~40 | 1.5 | 5e-5 | ~15 min |
| 20 | ~48 | ~68 | ~47 | 1.1 | 2e-5 | ~15 min |
| 30 | ~51 | ~71 | ~50 | 0.9 | 8e-6 | ~15 min |
| 50 | ~53+ | ~73+ | ~52+ | 0.8 | 1e-6 | ~15 min |

**Target: mAP ≥ 53.0** (matching RF-DETR-S on COCO)

### Export to TensorRT

```bash
# After training, export to ONNX
python scripts/export_onnx.py \
    --checkpoint models/checkpoints/rf_detr_s/best.pth \
    --output models/onnx/rf_detr_s.onnx \
    --input-size 640 640 \
    --batch-size 1

# Convert ONNX to TensorRT
trtexec \
    --onnx=models/onnx/rf_detr_s.onnx \
    --saveEngine=models/trt_engines/rf_detr_s_fp16.engine \
    --fp16 \
    --workspace=4096

# Benchmark TensorRT engine
python scripts/benchmark_trt.py \
    --engine models/trt_engines/rf_detr_s_fp16.engine \
    --input-size 640 640 \
    --iterations 1000
```

### Domain Adaptation with LoRA (Phase 2+, If Needed)

If baseline mAP on sports domain is < 50% (below target), apply LoRA adapters:

```yaml
# configs/rf_detr_s_lora_domain.yaml

model:
  name: "rf-detr-s"
  pretrained: "models/checkpoints/rf_detr_s/best.pth"  # Start from baseline
  lora:
    enabled: true
    rank: 16                    # Low-rank adaptation
    alpha: 32                   # Scaling factor
    target_modules:             # Apply LoRA to transformer layers
      - "encoder.layers.*.self_attn"
      - "encoder.layers.*.ffn"
      - "decoder.layers.*.self_attn"
      - "decoder.layers.*.cross_attn"
    dropout: 0.1

training:
  epochs: 20                    # Fewer epochs than baseline
  batch_size: 4                 # Smaller batch for fine-tuning
  learning_rate: 5e-5           # 10× lower than baseline
  freeze_backbone: true         # Freeze ConvNeXt backbone
  # Only LoRA adapters + decoder are trained
```

```bash
# LoRA fine-tune on sports domain data
python scripts/train_detection_lora.py \
    --config configs/rf_detr_s_lora_domain.yaml \
    --data data/processed/sports/ \
    --gpu 0

# Merge LoRA weights into base model
python scripts/merge_lora.py \
    --base models/checkpoints/rf_detr_s/best.pth \
    --lora models/checkpoints/rf_detr_s/lora_best.pth \
    --output models/checkpoints/rf_detr_s/domain_adapted.pth
```

**When to use LoRA:**
- Baseline mAP on sports clips < 50%
- Sports-specific classes (ball, goal, referee) have low recall
- Custom domain video shows detection gaps

**When NOT to use LoRA:**
- Baseline already meets targets (mAP ≥ 53.0)
- Time is limited (LoRA adds 1-2 days)
- Domain is similar to COCO (general objects)

---

## Model 2: OSNet Fine-Tuning (Re-ID)

### Why Second

Re-ID improves tracking quality, especially after occlusion. Quick to train (1-2 days).

### Training Configuration

```yaml
# configs/osnet_reid_finetune.yaml

model:
  name: "osnet_x1_0"
  pretrained: "imagenet"
  architecture:
    channels: [64, 128, 256, 512]
    num_classes: 0  # Set from data (number of identities)

training:
  epochs: 60
  batch_size: 32             # Re-ID uses larger batches
  precision: "fp16"
  compile: true

  # Triplet loss + cross entropy
  losses:
    - type: "triplet"
      margin: 0.3
      weight: 1.0
    - type: "cross_entropy"
      label_smoothing: 0.1
      weight: 1.0

  learning_rate: 3e-4
  scheduler: "cosine"
  optimizer: "adam"

  # Re-ID specific
  sampler: "balanced"        # Ensure equal class sampling
  num_instances: 4           # Per class per batch
  num_crops: 3               # Random crops per image

  augmentations:
    train:
      - type: "RandomHorizontalFlip"
        p: 0.5
      - type: "RandomCrop"
        padding: 10
      - type: "RandomErasing"
        probability: 0.5
        sl: 0.02
        sh: 0.4
      - type: "ColorJitter"
        brightness: 0.2
        contrast: 0.2
    val:
      - type: "Resize"
        size: [256, 128]
```

### Training Command

```bash
python scripts/train_reid.py \
    --config configs/osnet_reid_finetune.yaml \
    --gpu 0
```

### Evaluation

```bash
# Evaluate on Market1501
python scripts/eval_reid.py \
    --checkpoint models/checkpoints/osnet/best.pth \
    --dataset market1501 \
    --metrics rank1 rank5 mAP
```

**Target: Rank-1 > 95%, mAP > 85%** on Market1501

---

## Model 3: RF-DETR-Seg-S Fine-Tuning (Segmentation)

### Why Third

Segmentation is useful but not critical for the demo. Skip if running behind schedule.

### Training Configuration

```yaml
# configs/rf_detr_seg_s_finetune.yaml

model:
  name: "rf-detr-seg-s"
  pretrained: "rf-detr-s-detection"  # Start from detection weights
  architecture:
    backbone: "convnext_tiny"
    mask_head: "point_raster_head"
    num_classes: 91
    num_mask_tokens: 300

training:
  epochs: 40
  batch_size: 6               # Segmentation needs more memory
  precision: "fp16"
  losses:
    - "cross_entropy"
    - "l1"
    - "giou"
    - "dice"                 # Mask loss
  loss_weights:
    classification: 1.0
    l1: 5.0
    giou: 2.0
    dice: 5.0
  learning_rate: 1e-4
```

---

## Model 4: GBDT Hypothesis Ranker (CPU Only)

### Why Fourth

The hypothesis ranker decides which hypotheses to send to the expensive VLM reasoner. A GBDT (Gradient Boosted Decision Tree) model is fast to train, fast to inference, and gives calibrated rankings. This is the "intelligence" that decides what's worth reasoning about.

### Architecture Reference

`02_architecture/05_reasoning/01_hypothesis_engine.md` — the ranker sits inside the hypothesis engine, scoring candidate hypotheses before verification.

### What We're Training

A LightGBM ranking model that takes structured features from the evidence graph and outputs a relevance score for each hypothesis. Features include:
- Prediction error (how far the hypothesis deviates from observed state)
- Evidence count (how many observations support/refute)
- Temporal consistency (does the hypothesis agree with recent history)
- Cross-modal agreement (do vision and audio agree)
- Entity count and occlusion level
- R-score from event detector

### Training Configuration

```python
# scripts/train_hypothesis_ranker.py

"""
Hypothesis Ranker Training:
1. Generate labeled (event, hypothesis, correct_rank) triples
2. Extract structured features from evidence graph
3. Train LightGBM ranker with LambdaMART objective
4. Export to JSON for inference
"""

import lightgbm as lgb
import numpy as np
import json

class HypothesisRanker:
    """GBDT ranker for hypothesis scoring."""

    FEATURE_NAMES = [
        "prediction_error",
        "evidence_count_for",
        "evidence_count_against",
        "temporal_consistency",
        "cross_modal_agreement",
        "entity_count",
        "occlusion_level",
        "r_score",
        "hypothesis_age_ms",
        "similar_past_episodes_count",
    ]

    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        """Train LambdaMART ranker."""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [3, 5, 10],
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.log_evaluation(50)],
        )

        return self.model

    def predict(self, features):
        """Score hypotheses."""
        return self.model.predict(features)

    def export(self, path):
        """Export model to JSON."""
        self.model.dump_model(path)

    def load(self, path):
        """Load model from JSON."""
        self.model = lgb.Booster(model_file=path)
```

### Training Data Preparation

```bash
# Step 1: Generate labeled data from validation set
# Run the pipeline on validation videos, collect (event, hypotheses, ground_truth) triples
python scripts/generate_ranker_data.py \
    --data data/splits/val_videos.json \
    --output experiments/ranker/labeled_data.json \
    --min-hypotheses 3 \
    --max-hypotheses 10

# Step 2: Extract features
python scripts/extract_ranker_features.py \
    --labeled-data experiments/ranker/labeled_data.json \
    --output experiments/ranker/features.npz

# Step 3: Train ranker
python scripts/train_hypothesis_ranker.py \
    --features experiments/ranker/features.npz \
    --output models/hypothesis_ranker/ranker.json \
    --eval-split 0.2
```

### Expected Performance

| Metric | Target | Notes |
|---|---|---|
| NDCG@3 | ≥ 0.85 | Top-3 hypotheses contain correct one |
| NDCG@5 | ≥ 0.80 | Top-5 ranking quality |
| Inference latency | < 1ms | Single prediction on CPU |
| Model size | < 1MB | Lightweight for deployment |

**Target: NDCG@3 ≥ 0.85, inference < 1ms**

---

## Model 5: Calibrator Training (CPU Only)

### Why Last

The calibrator trains on validation set predictions from the other models. No GPU needed.

### What We're Training

1. **Temperature Scaling**: Single parameter T that scales logits
2. **Conformal Prediction**: Nonconformity scores from calibration set
3. **Confidence Decomposition Weights**: Learned weights for 5 components

### Training Configuration

```python
# scripts/train_calibrator.py

"""
Calibration Training:
1. Run trained models on validation set
2. Collect prediction scores and ground truth
3. Learn temperature parameter
4. Compute conformal thresholds
5. Learn confidence decomposition weights
"""

import numpy as np
import json
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve

class TemperatureScaling:
    """Learn optimal temperature for post-hoc calibration."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels):
        """Learn temperature on validation set."""
        def nll_loss(T):
            scaled_logits = logits / T
            probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
            nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-10)
            return nll.mean()

        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x

        # Compute ECE before and after
        ece_before = self._compute_ece(logits, labels, T=1.0)
        ece_after = self._compute_ece(logits, labels, T=self.temperature)

        print(f"Temperature: {self.temperature:.3f}")
        print(f"ECE before: {ece_before:.4f}")
        print(f"ECE after:  {ece_after:.4f}")

        return self.temperature

    def _compute_ece(self, logits, labels, T=1.0, n_bins=15):
        """Compute Expected Calibration Error."""
        scaled = logits / T
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = predictions == labels

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
        return ece

class ConformalCalibrator:
    """Compute conformal prediction thresholds."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.threshold = None

    def fit(self, nonconformity_scores):
        """Compute threshold from calibration set."""
        n = len(nonconformity_scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = np.quantile(nonconformity_scores, min(quantile_level, 1.0))
        print(f"Conformal threshold (alpha={self.alpha}): {self.threshold:.4f}")
        return self.threshold

    def predict_set(self, scores):
        """Return set of hypotheses within threshold."""
        return [i for i, s in enumerate(scores) if s <= self.threshold]
```

### Calibration Training Script

```bash
# Step 1: Generate predictions on validation set
python scripts/eval_detection.py \
    --checkpoint models/checkpoints/rf_detr_s/best.pth \
    --data data/splits/coco_val.json \
    --output experiments/calibration/prediction_logits.json

# Step 2: Train temperature scaler
python scripts/train_calibrator.py \
    --predictions experiments/calibration/prediction_logits.json \
    --output models/calibrator/temperature.json

# Step 3: Compute conformal thresholds
python scripts/train_calibrator.py \
    --mode conformal \
    --predictions experiments/calibration/prediction_logits.json \
    --output models/calibrator/conformal.json \
    --alpha 0.05 0.10

# Step 4: Learn confidence decomposition weights
python scripts/train_calibrator.py \
    --mode decomposition \
    --predictions experiments/calibration/prediction_logits.json \
    --output models/calibrator/decomposition.json
```

**Target: ECE < 0.05, coverage ≥ 95% at α=0.05**

---

## Training Infrastructure

### GPU Memory Management

```python
# Utility: Monitor GPU memory during training
import torch

def log_gpu_memory():
    """Log current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")
```

### Checkpoint Management

```python
# Save checkpoint
def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)

# Load checkpoint
def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']
```

### Reproducibility

```python
# Set all seeds for reproducibility
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## Model Export Summary

After all training, we should have:

```
models/
├── pretrained/
│   ├── rf_detr_s_coco.pth          # Original COCO weights
│   ├── osnet_imagenet.pth           # Original ImageNet weights
│   └── whisper_large_v3.pt          # Original Whisper weights
│
├── checkpoints/
│   ├── rf_detr_s/
│   │   ├── best.pth                 # Best mAP checkpoint
│   │   ├── last.pth                 # Last epoch checkpoint
│   │   └── config.yaml              # Training config used
│   ├── osnet/
│   │   ├── best.pth
│   │   └── config.yaml
│   └── rf_detr_seg_s/
│       ├── best.pth
│       └── config.yaml
│
├── onnx/
│   ├── rf_detr_s.onnx
│   ├── osnet.onnx
│   └── rf_detr_seg_s.onnx
│
├── trt_engines/
│   ├── rf_detr_s_fp16.engine        # TensorRT FP16
│   ├── rf_detr_s_int8.engine        # TensorRT INT8 (if quantized)
│   └── rf_detr_seg_s_fp16.engine
│
├── hypothesis_ranker/
│   └── ranker.json                  # LightGBM ranker (GBDT)
│
└── calibrator/
    ├── temperature.json
    ├── conformal.json
    └── decomposition.json
```
