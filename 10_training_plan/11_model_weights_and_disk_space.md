# Model Weights — Download List & Disk Space

> Every model weight file we need to download, exact file sizes, where to get them, how much disk space they take (raw weights + training checkpoints + exported inference engines). Verified September 2026.

---

## 1. Master Model Inventory

### Pre-Trained Weights (Download Only — No Training)

| # | Model | Purpose | Params | File Size | Source | License |
|---|---|---|---|---|---|---|
| M1 | RF-DETR-S | Detection (primary) | 32.1M | ~130 MB | Roboflow / auto-download | Apache 2.0 |
| M2 | RF-DETR-L | Detection (async fallback) | 33.9M | ~136 MB | Roboflow / auto-download | Apache 2.0 |
| M3 | DETRPose-S | Pose estimation | 11.9M | ~48 MB | GitHub releases | Apache 2.0 |
| M4 | ByteTrack | Object tracking | 0 | 0 (algorithm) | pip install | MIT |
| M5 | OSNet x1_0 | Re-ID (pretrained) | 2.2M | ~9 MB | torchreid auto-download | MIT |
| M6 | PaddleOCR v4 | OCR | — | ~200 MB | PaddlePaddle auto-download | Apache 2.0 |
| M7 | Whisper-large-v3 | Audio transcription | 1.55B | 3.09 GB | HuggingFace / whisper auto | MIT |
| M8 | Qwen3-VL-30B-A3B FP8 | VLM reasoning (local) | 31B (3B active) | 22.2 GB | HuggingFace | Apache 2.0 |
| M9 | SigLIP-base | Deepfake detection ensemble | 0.2B | 813 MB | HuggingFace | Apache 2.0 |
| M10 | EVA-02 | Deepfake detection ensemble | 0.3B | ~300 MB | HuggingFace | MIT |

**Pre-trained weights subtotal: ~26.8 GB**

### Fine-Tuned Checkpoints (Created During Training)

| # | Model | Trained From | Output Checkpoint Size |
|---|---|---|---|
| C1 | RF-DETR-S fine-tuned | M1 | ~130 MB (light) / ~500 MB (full w/ optimizer) |
| C2 | OSNet fine-tuned | M5 | ~9 MB (light) / ~30 MB (full) |
| C3 | RF-DETR-Seg-S fine-tuned | M1+seg head | ~140 MB (light) / ~550 MB (full) |
| C4 | GBDT Ranker | CPU training | <1 MB |
| C5 | Calibrator | CPU training | <1 MB |

**Training checkpoints subtotal: ~1.5 GB** (light) / **~6 GB** (full with optimizer states)

### Exported Inference Engines (TensorRT + ONNX)

| # | Model | Format | Precision | Size |
|---|---|---|---|---|
| E1 | RF-DETR-S | TensorRT FP16 | FP16 | ~80 MB |
| E2 | RF-DETR-S | ONNX | FP32 | ~130 MB |
| E3 | DETRPose-S | TensorRT FP16 | FP16 | ~30 MB |
| E4 | DETRPose-S | ONNX | FP32 | ~48 MB |
| E5 | OSNet | ONNX | FP32 | ~9 MB |
| E6 | RF-DETR-Seg-S | TensorRT FP16 | FP16 | ~90 MB |

**Exported engines subtotal: ~390 MB**

---

## 2. Detailed Model Breakdown

### M1: RF-DETR-S (Detection — Primary Model)

```
Parameters:     32.1M
Resolution:     512×512
COCO AP:        53.0 (AP50:95), 72.1 (AP50)
Latency:        3.5ms (T4 TensorRT FP16)
License:        Apache 2.0
```

**Download:**
```bash
# Option A: Auto-download via rfdetr package (RECOMMENDED)
pip install rfdetr
python -c "
from rfdetr import RFDETRSmall
model = RFDETRSmall()  # Auto-downloads COCO pretrained weights
print('RF-DETR-S loaded')
"
# Weights cached at: ~/.cache/rfdetr/rfdetr_small_*.pth (~130 MB)

# Option B: HuggingFace
pip install transformers
python -c "
from transformers import AutoModelForObjectDetection
model = AutoModelForObjectDetection.from_pretrained('Roboflow/rf-detr-small')
"
# Downloads to: ~/.cache/huggingface/hub/models--Roboflow--rf-detr-small/

# Option C: GitHub releases
wget https://github.com/roboflow/rf-detr/releases/download/v1.0.0/rfdetr_small.pth
```

**Disk space:**
| What | Size |
|---|---|
| Pretrained weights (.pth) | ~130 MB |
| Training checkpoint (with optimizer) | ~500 MB |
| Best checkpoint (light, EMA) | ~130 MB |
| ONNX export | ~130 MB |
| TensorRT FP16 engine | ~80 MB |
| **Total for RF-DETR-S** | **~970 MB** |

---

### M2: RF-DETR-L (Detection — Async Fallback)

```
Parameters:     33.9M
Resolution:     704×704
COCO AP:        56.5 (AP50:95), 75.1 (AP50)
Latency:        6.8ms (T4 TensorRT FP16)
License:        Apache 2.0
```

**Download:**
```bash
python -c "
from rfdetr import RFDETRLarge
model = RFDETRLarge()
"
```

**Disk space: ~970 MB** (same structure as RF-DETR-S)

---

### M3: DETRPose-S (Pose Estimation)

```
Parameters:     11.9M
COCO AP:        67.0
Latency:        2.39ms (TensorRT FP16)
License:        Apache 2.0
```

**Download:**
```bash
# GitHub releases (direct link)
wget https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s.pth
# File size: ~48 MB (11.9M params × 4 bytes FP32)

# Or HuggingFace
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='SebasJanampa/DETRPose_S_COCO',
    filename='detrpose_hgnetv2_s.pth'
)
print(path)
"
```

**Disk space:**
| What | Size |
|---|---|
| Pretrained weights (.pth) | ~48 MB |
| ONNX export | ~48 MB |
| TensorRT FP16 engine | ~30 MB |
| **Total for DETRPose-S** | **~126 MB** |

---

### M4: ByteTrack (Tracking — No Weights)

```
Parameters:     0 (algorithm, no learning)
Latency:        0.2-0.5ms per frame
License:        MIT
```

**Install:**
```bash
pip install byte-track
# Or: git clone https://github.com/ifzhang/ByteTrack
```

**Disk space: ~0 MB** (pure Python/C++ code, <1 MB)

---

### M5: OSNet x1_0 (Re-ID)

```
Parameters:     2.2M
Market1501:     Rank-1 94.2%, mAP 87.0%
Latency:        3-10ms
License:        MIT
```

**Download:**
```bash
# Auto-downloads on first use via torchreid
pip install torchreid
python -c "
from torchreid.utils import FeatureExtractor
extractor = FeatureExtractor(model_name='osnet_x1_0', device='cuda:0')
"
# Downloads to: ~/.torch/reid_models/osnet_x1_imagenet.pth (~9 MB)

# Or manually from HuggingFace
wget https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_imagenet.pth
```

**Disk space:**
| What | Size |
|---|---|
| Pretrained weights (ImageNet) | ~9 MB |
| Fine-tuned checkpoint (light) | ~9 MB |
| Fine-tuned checkpoint (full w/ optimizer) | ~30 MB |
| ONNX export | ~9 MB |
| **Total for OSNet** | **~57 MB** |

---

### M6: PaddleOCR v4 (OCR)

```
Detection model:    PP-OCRv4_server_det (110 MB)
Recognition model:  PP-OCRv4_server_rec (88 MB)
Angle classifier:   ppocr_v4 (1-2 MB)
Total:              ~200 MB
License:            Apache 2.0
```

**Download:**
```bash
pip install paddlepaddle-gpu paddleocr
python -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
# Auto-downloads models on first use
"
# Downloads to: ~/.paddleocr/ (~200 MB total)
```

**Disk space: ~200 MB**

---

### M7: Whisper-large-v3 (Audio)

```
Parameters:     1.55B
WER:            ~3% (English)
License:        MIT
```

**Download:**
```bash
# Option A: OpenAI whisper package
pip install openai-whisper
python -c "
import whisper
model = whisper.load_model('large-v3', device='cuda')
"
# Downloads to: ~/.cache/whisper/large-v3.pt (3.09 GB)

# Option B: HuggingFace (safetensors format)
python -c "
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3')
"
# Downloads to: ~/.cache/huggingface/hub/models--openai--whisper-large-v3/
# Files: model.safetensors (3.09 GB) + config + tokenizer (~50 MB)

# Option C: Only need transcription? Use faster-whisper (CTranslate2)
pip install faster-whisper
# Downloads ~1.5 GB (quantized version, half the size)
```

**Disk space:**
| Format | Size |
|---|---|
| PyTorch (.pt) | 3.09 GB |
| Safetensors | 3.09 GB |
| FP32 safetensors (HF) | 6.17 GB |
| faster-whisper (CTranslate2) | ~1.5 GB |
| **Recommended: PyTorch .pt** | **3.09 GB** |

---

### M8: Qwen3-VL-30B-A3B (VLM — Local)

```
Parameters:     31B total, 3B active per token (MoE)
Precision:      FP8 (fine-grained, block size 128)
License:        Apache 2.0
Inference:      ~90 tok/s on RTX 4090 24GB
```

**Download:**
```bash
# Option A: FP8 (RECOMMENDED — fits in 24GB)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-VL-30B-A3B-Instruct-FP8',
    local_dir='models/pretrained/qwen3-vl-30b-fp8'
)
"
# Files:
#   model-00001-of-00004.safetensors    10.6 GB
#   model-00002-of-00004.safetensors    10.6 GB
#   model-00003-of-00004.safetensors    10.7 GB
#   model-00004-of-00004.safetensors    325 MB
#   model.safetensors.index.json        112 KB
#   config.json, tokenizer, etc.        ~50 MB
#   TOTAL:                              ~22.2 GB

# Option B: GGUF Q4_K_M (for llama.cpp / Ollama — smaller)
# Q4_K_M: 18.29 GB (run with ~20.4 GB VRAM)
# Q8_0:   31.26 GB (run with ~33.4 GB VRAM)

# Option C: Full BF16 (NOT recommended — 62 GB, won't fit in 24GB)
```

**Disk space:**
| Quantization | Disk Size | VRAM Required | Fits 24GB? |
|---|---|---|---|
| FP8 (recommended) | 22.2 GB | ~23 GB | ✅ Tight |
| Q4_K_M (GGUF) | 18.29 GB | ~20.4 GB | ✅ |
| Q8_0 (GGUF) | 31.26 GB | ~33.4 GB | ❌ |
| BF16 (full) | 62.2 GB | ~65 GB | ❌ |

**Note:** For RTX 4090 24GB, use FP8 (22.2 GB) + token compression (StreamingTOM/HybridKV) to fit. For API fallback, no local download needed.

---

### M9: SigLIP-base (Deepfake Detection Ensemble)

```
Parameters:     0.2B (200M)
Resolution:     224×224
License:        Apache 2.0
```

**Download:**
```bash
python -c "
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('google/siglip-base-patch16-224')
processor = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
"
# Downloads to: ~/.cache/huggingface/hub/models--google--siglip-base-patch16-224/
# Files: model.safetensors (813 MB) + preprocessor + config (~1 MB)
```

**Disk space: ~815 MB**

---

### M10: EVA-02 (Deepfake Detection Ensemble)

```
Parameters:     ~300M (EVA02-B or EVA02-L)
License:        MIT
```

**Download:**
```bash
# From HuggingFace
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('facebook/eva02_large_patch14_448')
"
# ~1.2 GB (Large) or ~300 MB (Base)

# Or use timm
import timm
model = timm.create_model('eva_large_patch14_448.clip_ft_in22k_in1k')
# ~1.2 GB
```

**Disk space: ~300 MB (Base) to ~1.2 GB (Large)**

---

## 3. Complete Disk Space Budget

### Scenario A: API Fallback (No Local VLM)

| Category | Items | Size |
|---|---|---|
| **Pre-trained weights** | RF-DETR-S + RF-DETR-L + DETRPose-S + OSNet + PaddleOCR + SigLIP + EVA-02 | ~1.6 GB |
| **Whisper** | large-v3 (.pt) | 3.09 GB |
| **Training checkpoints** | 5 models (light) | ~1.5 GB |
| **Exported engines** | TensorRT + ONNX | ~390 MB |
| **VLM** | API (no download) | 0 |
| **Models subtotal** | | **~6.6 GB** |
| **Datasets** | COCO + MOT17 + Market1501 + SportsMOT + SoccerNet + custom | ~115 GB |
| **Experiment logs** | W&B local cache + TensorBoard | ~2 GB |
| **Grand total** | | **~124 GB** |

### Scenario B: Local VLM (Qwen3-VL FP8 on RTX 4090)

| Category | Items | Size |
|---|---|---|
| **Pre-trained weights** | Same as Scenario A | ~1.6 GB |
| **Whisper** | large-v3 (.pt) | 3.09 GB |
| **VLM** | Qwen3-VL-30B-A3B FP8 | 22.2 GB |
| **Training checkpoints** | 5 models (light) | ~1.5 GB |
| **Exported engines** | TensorRT + ONNX | ~390 MB |
| **Models subtotal** | | **~28.8 GB** |
| **Datasets** | Same as Scenario A | ~115 GB |
| **Experiment logs** | | ~2 GB |
| **Grand total** | | **~146 GB** |

### Scenario C: Full Local (Everything + Full Training Checkpoints)

| Category | Items | Size |
|---|---|---|
| **All models** | Scenario B | ~28.8 GB |
| **Full training checkpoints** | With optimizer states (resumable) | ~6 GB |
| **Multiple model variants** | RF-DETR-N + S + M + L, DETRPose-S + N | ~1.5 GB |
| **Models subtotal** | | **~36.3 GB** |
| **Datasets** | All 11 datasets incl. Celeb-DF++, FaceForensics++, MOT20, AudioSet | ~180 GB |
| **Experiment logs + TensorBoard** | | ~5 GB |
| **Grand total** | | **~221 GB** |

---

## 4. Download Priority Order

### Day 1-2 (Critical — Needed for Training)

```bash
# 1. RF-DETR-S (~130 MB) — auto-downloads with rfdetr package
pip install rfdetr
python -c "from rfdetr import RFDETRSmall; RFDETRSmall()"

# 2. DETRPose-S (~48 MB)
wget https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s.pth

# 3. OSNet (~9 MB) — auto-downloads with torchreid
pip install torchreid
python -c "from torchreid.utils import FeatureExtractor; FeatureExtractor(model_name='osnet_x1_0')"

# 4. Whisper large-v3 (3.09 GB)
pip install openai-whisper
python -c "import whisper; whisper.load_model('large-v3')"

# 5. PaddleOCR v4 (~200 MB) — auto-downloads
pip install paddlepaddle-gpu paddleocr
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"

# Subtotal Day 1-2: ~3.5 GB
```

### Day 3 (High Priority — For Reasoning + Forensics)

```bash
# 6. SigLIP (813 MB)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('google/siglip-base-patch16-224')"

# 7. EVA-02 (~300 MB)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/eva02_large_patch14_448')"

# 8. RF-DETR-L for async path (~136 MB)
python -c "from rfdetr import RFDETRLarge; RFDETRLarge()"

# Subtotal Day 3: ~1.2 GB
```

### Day 4+ (VLM — Only If Running Local)

```bash
# 9. Qwen3-VL-30B-A3B FP8 (22.2 GB) — LARGE DOWNLOAD
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-VL-30B-A3B-Instruct-FP8',
    local_dir='models/pretrained/qwen3-vl-30b-fp8'
)
"
# Takes 30-60 minutes depending on connection
# SKIP if using API fallback (GPT-4.1 / Gemini)

# Subtotal Day 4: ~22.2 GB (or 0 GB with API)
```

---

## 5. GPU VRAM Budget (Inference)

```
┌─────────────────────────────────────────────────┐
│ RTX 4090 24GB — Inference Memory Layout         │
├─────────────────────────────────────────────────┤
│ CUDA Context + cuDNN:          1.0 GB           │
│ RF-DETR-S (TensorRT FP16):     1.5 GB           │
│ DETRPose-S (TensorRT FP16):    0.5 GB           │
│ ByteTrack:                     0.05 GB          │
│ OSNet:                         0.05 GB          │
│ Whisper large-v3:              3.1 GB           │
│ PaddleOCR:                     0.5 GB           │
│ Working buffers:               2.0 GB           │
│ ─────────────────────────────────────────        │
│ Perception Subtotal:           ~8.7 GB          │
│                                          │
│ Qwen3-VL-30B-A3B FP8:         22.2 GB  ← WON'T FIT│
│ (with token compression):     ~15 GB    ← STILL TIGHT│
│                                          │
│ OPTION 1: VLM via API          → 8.7 GB used ✓  │
│ OPTION 2: Load VLM, skip Whisper → 23.2 GB ✓    │
│ OPTION 3: Stream VLM weights   → varies          │
└─────────────────────────────────────────────────┘

Recommended: Perception on GPU + VLM via API
  GPU used: ~8.7 GB (leaves 15 GB headroom)
  VLM latency: 400ms-1s (API) vs 800-3500ms (local)
```

---

## 6. Quick Reference Card

| Model | Download Size | VRAM | Source | Auto? |
|---|---|---|---|---|
| RF-DETR-S | 130 MB | 1.5 GB | rfdetr package | ✅ |
| RF-DETR-L | 136 MB | 1.5 GB | rfdetr package | ✅ |
| DETRPose-S | 48 MB | 0.5 GB | GitHub releases | ❌ wget |
| ByteTrack | 0 MB | 0.05 GB | pip install | ✅ |
| OSNet | 9 MB | 0.05 GB | torchreid | ✅ |
| PaddleOCR v4 | 200 MB | 0.5 GB | paddleocr package | ✅ |
| Whisper-large-v3 | 3.09 GB | 3.1 GB | whisper package | ✅ |
| Qwen3-VL FP8 | 22.2 GB | 22.2 GB | HuggingFace | ❌ explicit |
| SigLIP | 813 MB | 0.5 GB | transformers | ✅ |
| EVA-02 | 300 MB | 0.3 GB | transformers | ✅ |
| **TOTAL (no VLM)** | **~4.7 GB** | **~8.7 GB** | | |
| **TOTAL (with VLM)** | **~26.9 GB** | **~24 GB** | | |
