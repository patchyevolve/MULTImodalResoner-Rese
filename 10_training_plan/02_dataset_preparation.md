# Dataset Research & Preparation — Complete Guide

> Every dataset, model weight, and video source we use. Where to get them, exact sizes, formats, license terms, download commands, preprocessing steps, and quality checks. Verified September 2026.

---

## 1. Master Inventory

### Training Datasets

| # | Dataset | Purpose | Size | Source | License | Priority |
|---|---|---|---|---|---|---|
| 1 | COCO 2017 | Detection + Segmentation + Keypoints training | 25 GB | cocodataset.org | CC BY 4.0 | Critical |
| 2 | MOT17 | Tracking evaluation + Re-ID crops | 5.5 GB | motchallenge.net | Research use | Critical |
| 3 | Market1501 | Re-ID training | 153 MB | liangzheng.com.cn | Research use | Critical |
| 4 | SportsMOT | Sports tracking training + eval | 36.5 GB | HuggingFace / OneDrive | Research use | High |
| 5 | SoccerNet v2 | Sports domain action spotting + tracking | 20 GB | soccer-net.org (NDA) | Research use | High |

### Evaluation Datasets

| # | Dataset | Purpose | Size | Source | License | Priority |
|---|---|---|---|---|---|---|
| 6 | MOT20 | Crowded scene tracking evaluation | 6.8 GB | motchallenge.net | Research use | Medium |
| 7 | Celeb-DF++ | Deepfake detection evaluation | ~50 GB | github.com/OUC-VAS | Research use | Medium |
| 8 | FaceForensics++ | Deepfake detection training + eval | ~15 GB | github.com/ondyari | Research use | Medium |
| 9 | DFDC (Preview) | Deepfake detection evaluation | 5K videos | ai.meta.com/datasets/dfdc | Research use | Low |

### Audio

| # | Dataset | Purpose | Size | Source | License | Priority |
|---|---|---|---|---|---|---|
| 10 | AudioSet (subset) | Audio event detection for Whisper evaluation | 2.4 GB features | research.google.com/audioset | Research use | Low |

### Custom / Self-Collected

| # | Dataset | Purpose | Size | Source | License | Priority |
|---|---|---|---|---|---|---|
| 11 | Custom video clips | Domain demo + evaluation | 5 GB | YouTube / TV recording | Fair use | High |

### Model Pre-Trained Weights (NOT datasets, but needed for training)

| # | Model | Purpose | Source | License |
|---|---|---|---|---|
| W1 | RF-DETR-S | Detection backbone | Roboflow GitHub / HuggingFace | Apache 2.0 |
| W2 | RF-DETR-L | Async high-accuracy detection | Roboflow GitHub | Apache 2.0 |
| W3 | DETRPose-S | Pose estimation | GitHub SebastianJanampa | Apache 2.0 |
| W4 | OSNet | Re-ID backbone | torchreid / HuggingFace | MIT |
| W5 | ByteTrack | Object tracking | GitHub ifzhang | MIT |
| W6 | PaddleOCR v4 | OCR | PaddlePaddle | Apache 2.0 |
| W7 | Whisper-large-v3 | Audio transcription | OpenAI | MIT |
| W8 | Qwen3-VL-30B-A3B | VLM reasoning | Alibaba Qwen / HuggingFace | Apache 2.0 |
| W9 | SigLIP | Deepfake detection ensemble | Google / HuggingFace | Apache 2.0 |

**Total disk budget: ~120 GB** (datasets + weights + checkpoints)

---

## 2. COCO 2017 — Detection + Segmentation + Keypoints

### What It Is
The canonical object detection benchmark. 330K images, 80 object categories, bounding boxes + segmentation masks + 17-keypoint human pose annotations.

### Exact Numbers

| Split | Images | Annotations |
|---|---|---|
| train2017 | 118,287 | instances + keypoints + captions |
| val2017 | 5,000 | instances + keypoints + captions |
| test-dev2017 | 20,288 | no public annotations |
| **Total** | **143,597** | |

- 80 object categories (COCO defines 91, but only 80 used in detection)
- Person category: ~260K instances with 17-keypoint pose annotations
- Keypoint schema: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

### Download

```bash
mkdir -p data/raw/coco && cd data/raw/coco

# Images (~19 GB total)
wget http://images.cocodataset.org/zips/train2017.zip    # 19 GB
wget http://images.cocodataset.org/zips/val2017.zip      # 1 GB

# Annotations (~250 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Keypoint annotations (separate file)
wget http://images.cocodataset.org/annotations/person_keypoints_trainval2017.zip

# Test-dev (for final eval, no public GT)
wget http://images.cocodataset.org/zips/test2017.zip     # 7 GB

# Unzip all
unzip "*.zip" -d .
```

### Annotation Format (COCO JSON)

```json
{
  "images": [{"id": 1, "file_name": "000000000009.jpg", "width": 640, "height": 480}],
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 1,
    "bbox": [x, y, width, height],  // [float] pixels, top-left corner
    "segmentation": [[x1,y1,x2,y2,...]],  // polygon OR RLE
    "area": 1234.5,
    "iscrowd": 0,
    "keypoints": [x1,y1,v1, x2,y2,v2, ...],  // 17 * 3 = 51 values
    "num_keypoints": 10
  }],
  "categories": [{"id": 1, "name": "person", "keypoints": [...], "skeleton": [...]}]
}
```

### How We Use It

| Task | COCO Split | What We Extract |
|---|---|---|
| Detection fine-tuning | train2017 + val2017 | All 80 classes, bounding boxes |
| Segmentation fine-tuning | train2017 + val2017 | Instance masks |
| Pose estimation evaluation | val2017 | Keypoint annotations (person only) |
| Re-ID crop extraction | train2017 | Person bounding box crops |
| Hypothesis ranker features | val2017 | Scene context for feature engineering |

### Preprocessing Script

```python
# scripts/preprocess_coco.py
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

def verify_images(image_dir):
    """Check all images load without corruption."""
    corrupted = []
    for img_file in tqdm(list(image_dir.glob("*.jpg")), desc="Verifying"):
        try:
            img = Image.open(img_file)
            img.verify()
        except Exception as e:
            corrupted.append((img_file.name, str(e)))
    return corrupted

def extract_person_crops(ann_file, image_dir, output_dir, min_size=50):
    """Extract person crops for Re-ID training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_file) as f:
        coco = json.load(f)

    person_id = next(c["id"] for c in coco["categories"] if c["name"] == "person")
    img_dict = {img["id"]: img for img in coco["images"]}
    count = 0

    for ann in tqdm(coco["annotations"], desc="Extracting crops"):
        if ann["category_id"] != person_id or ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w < min_size or h < min_size:
            continue

        img = Image.open(image_dir / img_dict[ann["image_id"]]["file_name"])
        crop = img.crop((x, y, x + w, y + h))
        crop.save(output_dir / f"{ann['image_id']:06d}_{count:04d}.jpg")
        count += 1

    print(f"Extracted {count} person crops from {ann_file.name}")

if __name__ == "__main__":
    coco_dir = Path("data/raw/coco")

    # Verify
    for split in ["train2017", "val2017"]:
        bad = verify_images(coco_dir / split)
        if bad:
            print(f"WARNING: {len(bad)} corrupted in {split}")

    # Extract person crops
    extract_person_crops(
        coco_dir / "annotations" / "instances_train2017.json",
        coco_dir / "train2017",
        Path("data/processed/coco/person_crops_train")
    )
```

---

## 3. MOT17 — Multi-Object Tracking

### What It Is
The standard pedestrian tracking benchmark. 7 sequences (train + test), each with 3 detector variants (DPM, FRCNN, SDP). Provides bounding boxes with unique track IDs.

### Exact Numbers

| Split | Sequences | Total Frames | Total Boxes | Avg Density |
|---|---|---|---|---|
| Train | 7 (×3 detectors = 21) | 11,518 | 549,740 | ~48/seq |
| Test | 7 (×3 detectors = 21) | 16,638 | 917,092 | ~55/seq |
| **Total** | **14 unique** | **28,156** | **1,466,832** | |

- 2D MOT16-format annotations: frame_id, track_id, bbox [x,y,w,h], confidence, class, visibility
- Resolution: mixed (540p to 1080p)
- FPS: 25-30

### Download

```bash
mkdir -p data/raw/mot && cd data/raw/mot

# Full dataset (includes images, GT, detections) — 5.5 GB
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip

# Or from HuggingFace (smaller, individual sequences)
# pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Lekim89/MOT17', repo_type='dataset', local_dir='data/raw/mot')
"
```

### Annotation Format (MOTChallenge CSV)

```
frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence, class, visibility
1, 1, 1348, 562, 106, 256, 1, 1, 0.85
1, 2, 752, 432, 98, 224, 1, 1, 0.72
```

- class=1 is pedestrian (only class we care about)
- visibility: 0-1 ratio of visible area

### How We Use It

| Task | What We Extract |
|---|---|
| ByteTrack evaluation | HOTA, MOTA, IDF1 metrics on test set |
| Re-ID training crops | Person crops from tracking sequences |
| World state validation | Entity lifecycle tracking quality |

---

## 4. MOT20 — Crowded Scene Tracking

### What It Is
Extension of MOT17 with extremely crowded scenes (up to 246 pedestrians per frame). 8 sequences from 3 scenes.

### Exact Numbers

| Split | Sequences | Frames | Boxes | Avg Density |
|---|---|---|---|---|
| Train | 4 | 4,479 | 517,426 | 115/seq |
| Test | 4 | 8,931 | 1,134,614 | 127/seq |

### Download

```bash
wget https://motchallenge.net/data/MOT20.zip    # 6.8 GB
unzip MOT20.zip
```

### When We Use It
Only for evaluating tracker robustness under extreme crowding. Not needed for training.

---

## 5. Market1501 — Person Re-Identification

### What It Is
The standard person Re-ID benchmark. 1,501 identities captured by 6 cameras in front of a supermarket at Tsinghua University.

### Exact Numbers

| Split | Identities | Images |
|---|---|---|
| bounding_box_train | 751 | 12,936 |
| bounding_box_test | 750 | 15,913 |
| query | 750 | 3,368 |
| **Total** | **1,501** | **32,217** |

- Resolution: 128×256 pixels (standard Re-ID crop size)
- 6 cameras, field-of-view overlap
- DPM-detected bounding boxes (not hand-drawn)
- 2,793 distractor identities in test gallery

### Download

```bash
mkdir -p data/raw/market1501 && cd data/raw/market1501

# Primary link
wget http://161.117.101.58:8080/data/market1501/Market-1501-v15.09.15.zip
unzip Market-1501-v15.09.15.zip

# Alternative: Google Drive or Baidu Disk (see github.com/mikewilliamson/market1501)
```

### Annotation Format

```
# File naming: {person_id}_{camera_id}.jpg
# Example: 0002_c1s1_000151_01.jpg
#   0002 = person ID 2
#   c1 = camera 1
#   s1 = sequence 1
#   000151 = frame 151
#   01 = snapshot 1
```

### How We Use It

| Task | What We Extract |
|---|---|
| OSNet fine-tuning | Train split → train/val (80/20) |
| Re-ID evaluation | query + gallery → Rank-1, Rank-5, mAP |

### Preprocessing

```python
# Already in standard format. Just need to:
# 1. Split train into train/val (80/20 by identity)
# 2. Create person_id → contiguous_id mapping
# 3. Create query/gallery evaluation protocol

# torchreid handles this automatically:
from torchreid.data.datasets.image import Market1501
dataset = Market1501(root='data/raw')
# Returns train, query, gallery splits
```

---

## 6. SportsMOT — Sports Multi-Object Tracking

### What It Is
Large-scale sports tracking dataset. 240 video clips from basketball, football, and volleyball. Only tracks players on the court (not spectators/referees).

### Exact Numbers

| Category | Clips | Avg Frames | Total Boxes |
|---|---|---|---|
| Basketball | ~80 | 422 | ~500K |
| Football | ~80 | 673 | ~600K |
| Volleyball | ~80 | 360 | ~500K |
| **Total** | **240** | **485 avg** | **1.6M+** |

- Resolution: 720p, 25 FPS
- MOTChallenge format (same as MOT17)
- ~150K frames total (15× MOT17)
- ~3,401 unique tracks

### Download

```bash
# OneDrive (primary)
# https://1drv.ms/u/s!AtjeLq7YnYGRgQRrmqGr4B-k-xsC

# HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='MCG-NJU/SportsMOT', repo_type='dataset', local_dir='data/raw/sportsmot')
"

# Baidu Netdisk (password: 4dnw)
# Also on Kaggle: kaggle.com/datasets/ayushspai/sportsmot (36 GB)
```

### Annotation Format
Same as MOTChallenge CSV:
```
frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence, class, visibility
```

### Splits

```
splits_txt/
├── train.txt      # ~140 videos
├── val.txt        # ~50 videos
├── test.txt       # ~50 videos
├── basketball.txt
├── football.txt
└── volleyball.txt
```

### How We Use It

| Task | What We Extract |
|---|---|
| ByteTrack sports eval | HOTA on val/test splits |
| Custom tracker training | Train split for sports domain |
| Detection validation | RF-DETR-S on sports frames |

---

## 7. SoccerNet v2 — Soccer Video Understanding

### What It Is
500 complete broadcast soccer games (764 hours), with ~300K temporal annotations. 17 action classes, camera shot segmentation, replay grounding.

### Exact Numbers

| Annotation Type | Count |
|---|---|
| Action spots (17 classes) | 110,458 |
| Camera shot transitions (13 types) | 158,493 |
| Replay shots | 32,932 |
| Games | 500 (+ 50 challenge) |
| Resolution | 720p / 224p |
| FPS | 25 |

### 17 Action Classes
```
Penalty, Kick-off, Goal, Substitution, Offside, Shots on target,
Shots off target, Clearance, Ball out of play, Throw-in, Foul,
Indirect free-kick, Direct free-kick, Corner, Yellow card,
Red card, Yellow→red card
```

### Download

```bash
# Install SoccerNet pip package
pip install SoccerNet

# Download labels (free, no NDA)
python -c "
from SoccerNet.Downloader import SoccerNetDownloader
dl = SoccerNetDownloader(LocalDirectory='data/raw/soccernet')
dl.downloadGames(files=['Labels-v2.json'], split=['train', 'valid', 'test'])
dl.downloadGames(files=['Labels-cameras.json'], split=['train', 'valid', 'test'])
dl.downloadGames(files=['video.ini'], split=['train', 'valid', 'test'])
"

# Download pre-computed features (free)
python -c "
from SoccerNet.Downloader import SoccerNetDownloader
dl = SoccerNetDownloader(LocalDirectory='data/raw/soccernet')
dl.downloadGames(files=['1_ResNET_TF2_PCA512.npy', '2_ResNET_TF2_PCA512.npy'],
                 split=['train', 'valid', 'test'])
"

# Download videos (REQUIRES NDA from soccer-net.org)
dl.password = input('Enter NDA password: ')
dl.downloadGames(files=['1_720p.mkv', '2_720p.mkv'], split=['train', 'valid', 'test'])
```

### Data Format

```
SoccerNet/
├── england_epl/
│   └── 2016-2017/
│       └── 2017-02-04 - 12-30 Chelsea 1 - 1 Liverpool/
│           ├── video.ini                    # start/duration per half
│           ├── Labels-v2.json               # action spotting annotations
│           ├── Labels-cameras.json          # camera shot annotations
│           ├── 1_720p.mkv                   # 1st half video
│           ├── 2_720p.mkv                   # 2nd half video
│           ├── 1_ResNET_TF2_PCA512.npy     # pre-extracted features
│           └── 2_ResNET_TF2_PCA512.npy
```

### How We Use It

| Task | What We Extract |
|---|---|
| Action spotting eval | Labels-v2.json event timestamps |
| Sports reasoning | Event sequences + player tracking |
| Custom sports clips | Extract frames from broadcast videos |

---

## 8. Celeb-DF++ — Deepfake Detection (Evaluation)

### What It Is
State-of-the-art deepfake benchmark. 22 deepfake methods across 3 scenarios: Face-swap, Face-reenactment, Talking-face. 53,196 forged videos + 590 real.

### Exact Numbers

| Scenario | Methods | Videos |
|---|---|---|
| Face-swap | 8 (SimSwap, InSwapper, HifiFace, GHOST, etc.) | 15,200+ |
| Face-reenactment | 7 (DaGAN, FSRT, LivePortrait, etc.) | 13,300+ |
| Talking-face | 7 (SadTalker, AniTalker, EchoMimic, etc.) | 21,000+ |
| Real (Celeb-real) | — | 590 |
| Real (YouTube-real) | — | 590 |
| **Total** | **22 methods** | **54,376+** |

- ~15 million total frames
- Each video ~10 seconds average
- 59 celebrities of varying demographics

### Download

```bash
# GitHub: github.com/OUC-VAS/Celeb-DF-PP
git clone https://github.com/OUC-VAS/Celeb-DF-PP.git

# Follow instructions in repo for video downloads
# Requires agreement to dataset terms
```

### How We Use It
Evaluation only. Test our deepfake detection ensemble (CLIP + EVA-02 + SRM) on held-out videos.

---

## 9. FaceForensics++ — Deepfake Detection (Training)

### What It Is
1,000 real video sequences + 4 manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures). 1.8M+ manipulated images.

### Exact Numbers

| Method | Videos | Images |
|---|---|---|
| Real | 1,000 | 509,914 |
| Deepfakes | 1,000 | ~450K |
| Face2Face | 1,000 | ~450K |
| FaceSwap | 1,000 | ~450K |
| NeuralTextures | 1,000 | ~450K |
| **Total** | **5,000** | **~1.8M** |

- Compression levels: raw, c23 (light), c40 (heavy)
- Ground-truth manipulation masks included

### Download

```bash
# GitHub: github.com/ondyari/FaceForensics
git clone https://github.com/ondyari/FaceForensics.git
cd FaceForensics

# Download (requires agreeing to terms)
python download.py --dataset faceforensics --compression c40 --type real --download-type videos
python download.py --dataset faceforensics --compression c40 --type manipulation --methods deepfakes face2face faceswap neuraltextures
```

### How We Use It

| Task | What We Extract |
|---|---|
| Deepfake detector training | c40 compressed videos (most realistic) |
| Ensemble calibration | Validation split for threshold tuning |

---

## 10. AudioSet — Audio Events (Optional)

### What It Is
2M+ 10-second audio clips from YouTube, 527 event classes. Used to evaluate Whisper audio features on sports commentary.

### Exact Numbers

| Split | Segments |
|---|---|
| balanced_train | 22,160 |
| unbalanced_train | 2,041,789 |
| eval | 20,371 |
| **Total** | **2,084,320** |

- 128-dim audio features (VGG-based), extracted at 1Hz
- Total features: 2.4 GB
- Total audio: ~1.9 TB (if downloading raw)

### Download

```bash
# Features only (2.4 GB) — sufficient for our use
gsutil rsync -d -r features gs://us_audioset/youtube_corpus/v1/features

# Or download pre-extracted features as tar.gz
wget https://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz
```

### How We Use It
Only for evaluating Whisper audio features on sports commentary. Not for training.

---

## 11. Custom Domain Video — Self-Collected

### What We Need

| Content Type | Count | Duration | Source | Purpose |
|---|---|---|---|---|
| Soccer match | 3-5 clips | 2-5 min | YouTube (fair use) | Primary demo |
| Basketball game | 2-3 clips | 2-5 min | YouTube | Secondary demo |
| News broadcast | 2-3 clips | 1-2 min | TV recording | OCR + general eval |
| Surveillance | 2-3 clips | 3-5 min | Public datasets | General eval |
| Deepfake samples | 5-10 clips | 5-10s | Celeb-DF / FF++ | Forensics demo |
| **Total** | **15-25 clips** | **~30 min** | | |

### Collection Strategy

```bash
# YouTube clips (fair use for research)
# Use yt-dlp to download short clips
yt-dlp --download-sections "*0:00-2:00" -o "data/raw/custom/soccer/%(title)s.%(ext)s" "URL"

# For broadcast soccer: search for "full match highlights" or "extended highlights"
# These are typically 3-10 minutes and contain the key events we need
```

### Annotation Requirements

```python
# For each custom clip, annotate:
# 1. Bounding boxes for first 100 frames (CVAT / Label Studio)
# 2. Key events with timestamps (manual)
# 3. Text regions (scoreboards, jersey numbers)
# 4. Ground truth claims for VLM evaluation

# Tools:
# - CVAT (cvat.ai) — free, open-source, COCO format export
# - Label Studio (labelstud.io) — free, multi-modal
# - FiftyOne (voxel51.com) — dataset visualization + analysis
```

---

## 12. Pre-Trained Weight Downloads

### RF-DETR-S (Detection — Primary Model)

```bash
# Option 1: Roboflow package (recommended)
pip install rfdetr

# Option 2: HuggingFace
pip install transformers
python -c "
from transformers import AutoModelForObjectDetection, AutoImageProcessor
processor = AutoImageProcessor.from_pretrained('Roboflow/rf-detr-base')
model = AutoModelForObjectDetection.from_pretrained('Roboflow/rf-detr-base')
print('RF-DETR-S loaded')
"

# Option 3: From source
pip install https://github.com/roboflow/rf-detr/archive/refs/heads/develop.zip

# Model sizes:
# RF-DETR-N: 30.5M params, 48.4 AP, 2.3ms T4
# RF-DETR-S: 32.1M params, 53.0 AP, 3.5ms T4  ← WE USE THIS
# RF-DETR-M: 33.7M params, 54.7 AP, 4.4ms T4
# RF-DETR-L: 33.9M params, 56.5 AP, 6.8ms T4  ← ASYNC FALLBACK
```

### DETRPose-S (Pose Estimation)

```bash
# Download weights from GitHub releases
wget https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s.pth

# Or from HuggingFace
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='SebasJanampa/DETRPose_S_COCO', filename='detrpose_hgnetv2_s.pth')
print(f'Weights at: {path}')
"

# Model performance (COCO val2017):
# DETRPose-S: 67.0 AP, 11.5M params, 2.39ms (TensorRT FP16)
# DETRPose-N: 57.2 AP, 4.1M params, 1.55ms (lighter fallback)
```

### OSNet (Re-ID)

```bash
# Auto-downloads on first use via torchreid
pip install torchreid

python -c "
from torchreid.utils import FeatureExtractor
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='',  # Downloads automatically
    device='cuda:0'
)
print('OSNet loaded')
"

# Performance: Rank-1 > 95%, mAP > 85% on Market1501
```

### ByteTrack (Tracking — No Weights Needed)

```bash
pip install byte-track
# Or: https://github.com/ifzhang/ByteTrack
# Algorithm-based, no training required
```

### PaddleOCR v4 (OCR)

```bash
pip install paddlepaddle-gpu paddleocr

python -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
print('PaddleOCR initialized')
"
# 3% WER, no fine-tuning needed
```

### Whisper-large-v3 (Audio)

```bash
pip install openai-whisper

python -c "
import whisper
model = whisper.load_model('large-v3', device='cuda')
print('Whisper loaded')
"
# 3% WER, no fine-tuning needed
```

### Qwen3-VL-30B-A3B (VLM — Local or API)

```bash
# Option A: API (recommended for initial development)
pip install openai
export OPENAI_API_KEY="your-key"

# Option B: Local inference (requires ~20GB VRAM)
pip install qwen-vl-utils vllm
# Download from HuggingFace: Qwen/Qwen3-VL-30B-A3B
# FP8 fits in 24GB RTX 4090 with token compression
```

---

## 13. Disk Space Budget

| Item | Size | Location |
|---|---|---|
| COCO 2017 (images + annotations) | 25 GB | `data/raw/coco/` |
| MOT17 | 5.5 GB | `data/raw/mot/` |
| MOT20 (optional) | 6.8 GB | `data/raw/mot20/` |
| Market1501 | 153 MB | `data/raw/market1501/` |
| SportsMOT | 36.5 GB | `data/raw/sportsmot/` |
| SoccerNet v2 (features + labels) | 20 GB | `data/raw/soccernet/` |
| Celeb-DF++ (eval subset) | 10 GB | `data/raw/celebdf/` |
| FaceForensics++ (c40) | 15 GB | `data/raw/ffpp/` |
| Custom video clips | 5 GB | `data/raw/custom/` |
| Pre-trained weights | 10 GB | `models/pretrained/` |
| Fine-tuned checkpoints | 20 GB | `models/checkpoints/` |
| TensorRT engines | 5 GB | `models/trt_engines/` |
| Experiment logs | 2 GB | `experiments/` |
| **Total** | **~160 GB** | |

---

## 14. Download Priority Order

Execute in this exact order (most critical first):

```bash
# DAY 2 (Critical — needed for training)
# 1. COCO 2017 (detection + pose training)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/person_keypoints_trainval2017.zip

# 2. MOT17 (tracking evaluation)
wget https://motchallenge.net/data/MOT17.zip

# DAY 3 (High priority)
# 3. Market1501 (Re-ID training)
wget http://161.117.101.58:8080/data/market1501/Market-1501-v15.09.15.zip

# 4. SportsMOT (sports tracking)
# Download from OneDrive or HuggingFace

# 5. SoccerNet v2 labels + features (sports domain)
pip install SoccerNet
python -c "from SoccerNet.Downloader import SoccerNetDownloader; ..."

# DAY 4 (Medium priority — evaluation only)
# 6. Celeb-DF++ (deepfake eval)
# 7. FaceForensics++ (deepfake training)
# 8. Custom video clips

# DAY 5 (Low priority)
# 9. AudioSet features (audio eval)
# 10. MOT20 (crowded eval, optional)
```

---

## 15. Data Quality Checklist

After downloading and preprocessing, verify ALL of these:

```bash
# scripts/verify_datasets.py

# COCO
- [ ] All 118,287 train images load without corruption
- [ ] All 5,000 val images load without corruption
- [ ] Annotation JSON is valid and parseable
- [ ] Bounding boxes within image bounds (x>=0, y>=0, x+w<=W, y+h<=H)
- [ ] Keypoint annotations have correct format (51 values per person)
- [ ] No duplicate image IDs
- [ ] Person crops extracted: ~260K from train

# MOT17
- [ ] All 21 sequences (7×3) have complete frame sequences
- [ ] Ground truth CSV files parse correctly
- [ ] Track IDs are unique within each sequence
- [ ] No gaps in frame numbering

# Market1501
- [ ] 12,936 train images present
- [ ] 3,368 query images present
- [ ] 15,913 gallery images present
- [ ] Person IDs match across splits
- [ ] Camera IDs in range [1,6]

# SportsMOT
- [ ] 240 sequences present
- [ ] Split files match actual sequence directories
- [ ] Annotations in MOTChallenge format

# SoccerNet
- [ ] Labels-v2.json parseable for all 500 games
- [ ] Feature .npy files load correctly
- [ ] Action class distribution matches published stats

# Custom
- [ ] All clips play without errors
- [ ] Annotations in COCO format
- [ ] Event labels have correct timestamps
```

---

## 16. Dataset Versioning

```bash
# Use DVC for large file tracking
pip install dvc

dvc init
dvc remote add -d storage /path/to/storage

# Track each dataset
dvc add data/raw/coco
dvc add data/raw/mot
dvc add data/raw/market1501
dvc add data/raw/sportsmot
dvc add data/raw/soccernet
dvc add data/raw/custom

# Version commits
git add data/*.dvc .gitignore
git commit -m "Dataset v1: COCO + MOT17 + Market1501 + SportsMOT + SoccerNet"
```
