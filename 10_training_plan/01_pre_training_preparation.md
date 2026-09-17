# Phase 0: Pre-Training Preparation

> Everything we do BEFORE touching a model. Environment, tools, verification. This phase takes 1-3 days and prevents weeks of wasted time from broken setups.

---

## 1. System Requirements Checklist

### Hardware Verification

```bash
# Check GPU is visible and correct
nvidia-smi

# Expected output for RTX 4090:
# - Name: NVIDIA GeForce RTX 4090
# - Memory: 24564 MiB
# - CUDA Version: 12.x
# - Driver Version: 550.x+

# Check GPU can run sustained compute
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv -l 1

# Check available disk space (need ~500GB total)
df -h /home

# Check RAM
free -h
# Need: 32GB+ recommended, 16GB minimum
```

### Disk Space Budget

| Item | Size | Location |
|---|---|---|
| COCO 2017 (train + val) | ~25GB | `~/data/coco/` |
| MOT17 (train + val) | ~10GB | `~/data/mot/` |
| Market1501 | ~1GB | `~/data/market1501/` |
| SoccerNet v2 | ~20GB | `~/data/soccernet/` |
| Custom video clips | ~5GB | `~/data/custom/` |
| Pre-trained weights | ~10GB | `~/models/pretrained/` |
| Fine-tuned checkpoints | ~20GB | `~/models/checkpoints/` |
| TensorRT engines | ~5GB | `~/models/trt_engines/` |
| Experiment logs | ~2GB | `~/experiments/` |
| **Total** | **~100GB** | |

---

## 2. Software Environment Setup

### 2.1 CUDA + cuDNN + PyTorch

```bash
# Install CUDA 12.4 (if not already installed)
# Check version compatibility first

# Create conda environment
conda create -n multimodal python=3.11 -y
conda activate multimodal

# Install PyTorch with CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA is available in PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')"
```

### 2.2 Core Dependencies

```bash
# Detection + Pose + Tracking
pip install ultralytics              # YOLO (baseline comparison)
pip install supervision              # Annotating, processing
pip install opencv-python-headless   # Video I/O
pip install av                       # Video decoding (PyAV)

# Deep learning frameworks
pip install transformers             # HuggingFace (Whisper, OSNet)
pip install timm                     # Pre-trained vision models
pip install einops                   # Tensor operations

# VLM inference
pip install vllm                     # Local VLM serving (if GPU allows)
pip install openai                   # API fallback (GPT-4.1)
pip install google-generativeai      # API fallback (Gemini)

# Calibration
pip install mapie                    # Conformal prediction
pip install scikit-learn             # Calibration utilities

# Monitoring + Logging
pip install wandb                    # Experiment tracking (free for academics)
pip install tensorboard              # Alternative logging
pip install psutil                   # System monitoring
pip install gpustat                  # GPU monitoring

# Deployment
pip install fastapi uvicorn          # REST API
pip install grpcio grpcio-tools      # gRPC (if needed)

# Utilities
pip install numpy scipy pillow tqdm pyyaml rich
```

### 2.3 NVIDIA DeepStream SDK (Optional but Recommended)

```bash
# DeepStream provides optimized video decode + pre-processing
# Download from: https://developer.nvidia.com/deepstream-sdk
# Skip if not available — we have fallback paths

# If available:
# 1. Install DeepStream 7.x
# 2. Verify: deepstream-app --version-all
# 3. Use nvdec for hardware-accelerated decode
```

### 2.4 TensorRT (Required for Production Inference)

```bash
# Install TensorRT (comes with PyTorch CUDA installation usually)
pip install tensorrt

# Verify
python -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"

# For ONNX export + TensorRT conversion:
pip install onnx onnxruntime-gpu
```

---

## 3. Repository Setup

```bash
# Clone our repo
git clone https://github.com/patchyevolve/MULTImodalResoner-Rese.git
cd MULTImodalResoner-Rese

# Create code directories (we've only had docs so far)
mkdir -p src/{perception,fusion,state,memory,reasoning,calibration,scheduler,forensics,domains,infrastructure}
mkdir -p src/utils
mkdir -p configs
mkdir -p scripts
mkdir -p tests
mkdir -p notebooks

# Create data directories
mkdir -p data/{raw,processed,splits,custom}

# Create model directories
mkdir -p models/{pretrained,checkpoints,trt_engines,onnx}

# Create experiment directories
mkdir -p experiments/{runs,logs,configs}

# Create deployment directories
mkdir -p deployment/{docker,k8s,helm}
```

---

## 4. Pre-trained Weight Downloads

### 4.1 Detection Models

```bash
# RF-DETR-S (primary detection model)
# Download from HuggingFace
python -c "
from transformers import AutoModelForObjectDetection
model = AutoModelForObjectDetection.from_pretrained(
    'researcher/rf-detr-s',
    force_download=True
)
print('RF-DETR-S downloaded successfully')
"

# Also download RF-DETR-N (fallback, lighter)
# RF-DETR-L (async high-accuracy path)
```

### 4.2 Pose Models

```bash
# DETRPose-S (primary pose model)
# Check Ultralytics or original paper repo for weights
# If not available as pre-trained, we'll train from scratch on COCO keypoints

# RF-DETR-KP (single-pass detection+pose, fallback)
```

### 4.3 Tracking

```bash
# ByteTrack — no pre-trained weights needed (algorithm-based)
# Just install the library
pip install byte-track
# Or use: https://github.com/ifzhang/ByteTrack
```

### 4.4 Re-ID

```bash
# OSNet (primary Re-ID model)
python -c "
from torchreid.utils import FeatureExtractor
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='',  # Downloads automatically
    device='cuda:0'
)
print('OSNet downloaded successfully')
"
```

### 4.5 Segmentation

```bash
# RF-DETR-Seg-S
# Similar to detection model, with segmentation head
```

### 4.6 OCR

```bash
# PaddleOCR v4
pip install paddlepaddle-gpu paddleocr
python -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
print('PaddleOCR initialized successfully')
"
```

### 4.7 Audio (Whisper)

```bash
# Whisper large-v3
python -c "
import whisper
model = whisper.load_model('large-v3', device='cuda')
print('Whisper large-v3 loaded successfully')
"
```

### 4.8 VLM (Reasoning)

```bash
# Option A: Qwen3-VL-30B-A3B FP8 (fits in 24GB RTX 4090 for inference, 90 tok/s)
# Option B: API access to Qwen3-VL-30B or GPT-4.1

# For local:
pip install qwen-vl-utils
# Download Qwen3-VL-30B-A3B from HuggingFace

# For API:
# Set up API keys as environment variables
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### 4.9 Deepfake Detection

```bash
# CLIP (SigLIP variant)
python -c "
from transformers import AutoProcessor, AutoModel
model = AutoModel.from_pretrained('google/siglip-base-patch16-224')
print('SigLIP downloaded successfully')
"

# EVA-02 (structural analysis)
# SRM + BayarNet (noise analysis)
# These may need custom implementation or HuggingFace search
```

---

## 5. GPU Benchmark Verification

Before any training, verify our GPU matches published benchmarks:

```bash
# Script: scripts/verify_gpu_benchmarks.py

import torch
import time

def benchmark_detection():
    """Verify RF-DETR-S inference latency."""
    # Load model
    # Run 100 warmup iterations
    # Measure p50, p95, p99 latency
    # Compare with published: 3.5ms on T4
    pass

def benchmark_pose():
    """Verify DETRPose-S inference latency."""
    # Expected: 2.39ms on A10
    pass

def benchmark_tracking():
    """Verify ByteTrack throughput."""
    # Expected: 0.2ms per frame
    pass

def benchmark_vlm():
    """Verify VLM inference speed."""
    # Expected: 90 tok/s for Qwen3-VL-30B FP8
    # Or measure API latency
    pass

def benchmark_memory():
    """Verify VRAM allocation for full pipeline."""
    # Perception: ~3GB
    # VLM: depends on model
    # Total must fit in available VRAM
    pass

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    benchmark_detection()
    benchmark_pose()
    benchmark_tracking()
    benchmark_vlm()
    benchmark_memory()

    print("\nAll benchmarks passed!" if all_passed else "\nSome benchmarks failed — investigate before training.")
```

---

## 6. Experiment Tracking Setup

```bash
# Weights & Biases (recommended — free for academics)
wandb login
# Follow browser prompt

# In code:
import wandb
wandb.init(
    project="multimodal-video-reasoner",
    config={
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": torch.cuda.get_device_properties(0).total_mem / 1e9,
    }
)
```

---

## 7. Configuration Templates

### configs/rf_detr_s_finetune.yaml

```yaml
# RF-DETR-S Fine-tuning Configuration
model:
  name: "rf-detr-s"
  pretrained: "researcher/rf-detr-s"
  num_classes: 91  # COCO classes (+ custom if needed)
  input_size: [640, 640]
  precision: "fp16"  # or "bf16" if supported

training:
  epochs: 50
  batch_size: 8  # Adjust based on VRAM
  gradient_accumulation: 4  # Effective batch = 32
  learning_rate: 0.0001
  weight_decay: 0.0001
  warmup_steps: 500
  scheduler: "cosine_with_restarts"
  optimizer: "adamw"

data:
  train: "data/splits/coco_train.json"
  val: "data/splits/coco_val.json"
  test: "data/splits/custom_test.json"
  augment:
    - "random_horizontal_flip"
    - "random_scale jitter 0.8-1.2"
    - "random_crop"
    - "color_jitter"
    - "mixup alpha=0.5"

checkpointing:
  save_dir: "models/checkpoints/rf_detr_s/"
  save_every_n_epochs: 5
  keep_top_k: 3  # by mAP

logging:
  log_every_n_steps: 50
  wandb_project: "multimodal-video-reasoner"
  wandb_run_name: "rf-det-s-finetune-v1"
```

### configs/osnet_reid_finetune.yaml

```bash
# OSNet Fine-tuning Configuration
model:
  name: "osnet_x1_0"
  pretrained: "imagenet"  # or torchreid default
  num_classes: 0  # Will be set from data

training:
  epochs: 60
  batch_size: 32
  learning_rate: 0.0003
  scheduler: "cosine"
  optimizer: "adam"
  label_smooth: true
  label_smooth_eps: 0.1

data:
  train: "data/splits/market1501_train.json"
  val: "data/splits/market1501_val.json"
  test: "data/splits/market1501_test.json"
  height: 256
  width: 128
  augment:
    - "random_horizontal_flip"
    - "random_crop padding=10"
    - "random_erase"
    - "color_jitter"
```

---

## 8. Pre-Training Verification Checklist

Before starting Phase 1 (training), verify ALL of these:

- [ ] GPU visible and correct model detected
- [ ] CUDA version matches PyTorch CUDA build
- [ ] All pip dependencies installed without errors
- [ ] Pre-trained weights downloaded for all models
- [ ] GPU benchmark matches published numbers (±10%)
- [ ] VRAM sufficient for training batch size
- [ ] Disk space > 100GB available
- [ ] Wandb account created and logged in
- [ ] Repository structure created
- [ ] Configuration files written
- [ ] Dataset download scripts ready
- [ ] At least one end-to-end smoke test passing

**Do not proceed to Phase 1 until ALL checkboxes are green.**
