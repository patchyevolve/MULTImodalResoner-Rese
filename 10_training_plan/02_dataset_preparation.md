# Phase 1: Dataset Preparation

> Every dataset we use, how to get it, how to preprocess it, and what splits we create. This phase runs in parallel with Phase 0.

---

## 1. Dataset Inventory

| Dataset | Purpose | Size | Source | License |
|---|---|---|---|---|
| COCO 2017 | Detection + Segmentation training | 25GB | cocodataset.org | CC BY 4.0 |
| MOT17 | Tracking evaluation | 10GB | MOTChallenge | Research use |
| Market1501 | Re-ID training | 1GB | ics.uci.edu | Research use |
| SoccerNet v2 | Sports domain reasoning | 20GB | soccernet.com | Research use |
| Custom clips | Domain-specific evaluation | 5GB | Self-collected | Own data |
| Deepfake test set | Forensics evaluation | 2GB | Various | Research use |

---

## 2. COCO 2017 (Detection + Segmentation)

### Download

```bash
# Create directory
mkdir -p data/raw/coco
cd data/raw/coco

# Download train images (~19GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Download val images (~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Download annotations (~250MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# Download test-dev images (for final evaluation)
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
```

### Preprocessing

```python
# scripts/preprocess_coco.py

"""
COCO Preprocessing Pipeline:
1. Verify all images load without errors
2. Filter annotations (remove crowd annotations, keep only person for Re-ID)
3. Create detection splits (train/val/test)
4. Create person crops for Re-ID training
5. Create keypoint annotations for pose
6. Verify data integrity
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

def verify_images(image_dir):
    """Verify all images load without corruption."""
    corrupted = []
    for img_file in tqdm(list(image_dir.glob("*.jpg")), desc="Verifying images"):
        try:
            img = Image.open(img_file)
            img.verify()
        except Exception as e:
            corrupted.append((img_file.name, str(e)))
    return corrupted

def create_detection_split(annotation_file, output_dir):
    """Create train/val detection splits with our format."""
    with open(annotation_file) as f:
        coco = json.load(f)

    # Our detection format
    split = {
        "images": [],
        "annotations": [],
        "categories": coco["categories"]
    }

    img_id_map = {}
    for i, img in enumerate(coco["images"]):
        split["images"].append({
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        })
        img_id_map[img["id"]] = i

    for ann in tqdm(coco["annotations"], desc="Processing annotations"):
        if ann.get("iscrowd", 0):
            continue
        split["annotations"].append({
            "id": ann["id"],
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],  # [x, y, w, h]
            "area": ann["area"],
            "iscrowd": 0
        })

    output_path = output_dir / "detection_split.json"
    with open(output_path, "w") as f:
        json.dump(split, f)
    print(f"Created {output_path} with {len(split['images'])} images, {len(split['annotations'])} annotations")

def extract_person_crops(annotation_file, image_dir, output_dir):
    """Extract person crops for Re-ID training."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_file) as f:
        coco = json.load(f)

    person_category_id = None
    for cat in coco["categories"]:
        if cat["name"] == "person":
            person_category_id = cat["id"]
            break

    img_dict = {img["id"]: img for img in coco["images"]}

    person_id_counter = defaultdict(int)
    for ann in tqdm(coco["annotations"], desc="Extracting person crops"):
        if ann["category_id"] != person_category_id:
            continue
        if ann.get("iscrowd", 0):
            continue

        img_info = img_dict[ann["image_id"]]
        img_path = image_dir / img_info["file_name"]

        x, y, w, h = ann["bbox"]
        if w < 50 or h < 50:  # Skip tiny detections
            continue

        # Load and crop
        img = Image.open(img_path)
        crop = img.crop((x, y, x + w, y + h))

        # Save with person-level labeling
        person_id_counter[ann["image_id"]] += 1
        crop_name = f"{ann['image_id']:06d}_{person_id_counter[ann['image_id']]:03d}.jpg"
        crop.save(output_dir / crop_name)

    print(f"Extracted {sum(person_id_counter.values())} person crops")

if __name__ == "__main__":
    coco_dir = Path("data/raw/coco")
    output_dir = Path("data/processed/coco")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Verify images
    corrupted = verify_images(coco_dir / "train2017")
    corrupted += verify_images(coco_dir / "val2017")
    if corrupted:
        print(f"WARNING: {len(corrupted)} corrupted images found")
        for name, err in corrupted[:10]:
            print(f"  {name}: {err}")

    # Step 2: Create detection splits
    create_detection_split(
        coco_dir / "annotations" / "instances_train2017.json",
        output_dir
    )
    create_detection_split(
        coco_dir / "annotations" / "instances_val2017.json",
        output_dir
    )

    # Step 3: Extract person crops for Re-ID
    extract_person_crops(
        coco_dir / "annotations" / "instances_train2017.json",
        coco_dir / "train2017",
        output_dir / "person_crops_train"
    )
```

---

## 3. MOT17 (Tracking)

### Download

```bash
mkdir -p data/raw/mot
cd data/raw/mot

# MOT17 train + test
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip

# MOT20 (more crowded scenes, optional)
# wget https://motchallenge.net/data/MOT20.zip
# unzip MOT20.zip
```

### Preprocessing

```python
# scripts/preprocess_mot.py

"""
MOT Preprocessing:
1. Parse MOT annotation format (gt.txt)
2. Convert to our track format
3. Create evaluation splits
4. Verify frame sequences are complete
"""

def parse_mot_gt(gt_file):
    """Parse MOT ground truth file."""
    tracks = {}
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            confidence = float(parts[6]) if len(parts) > 6 else 1.0
            class_id = int(parts[7]) if len(parts) > 7 else 1

            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append({
                "frame": frame_id,
                "bbox": [x, y, w, h],
                "confidence": confidence,
                "class": class_id
            })
    return tracks
```

---

## 4. Market1501 (Re-ID)

### Download

```bash
mkdir -p data/raw/market1501
cd data/raw/market1501

wget http://161.117.101.58:8080/data/market1501/Market-1501-v15.09.15.zip
unzip Market-1501-v15.09.15.zip
```

### Preprocessing

```python
# Market1501 already has train/test splits in bounding_box_train/ and bounding_box_test/
# Just need to:
# 1. Create our Re-ID format (image_path, person_id, camera_id)
# 2. Split train into train/val (80/20)
# 3. Create gallery and query sets for evaluation
```

---

## 5. SoccerNet v2 (Sports Domain)

### Download

```bash
mkdir -p data/raw/soccernet
cd data/raw/soccernet

# SoccerNet v2 requires registration at soccernet.com
# Download: actions, replays, tracking
# ~20GB total

# If not available, use alternative:
# - Sports dataset from Roboflow (smaller but sufficient)
# - Self-collected soccer clips from YouTube (fair use for research)
```

### Preprocessing

```python
"""
SoccerNet Preprocessing:
1. Extract frames at 30 FPS from match videos
2. Create action labels (goal, foul, corner, throw-in, etc.)
3. Create player tracking annotations
4. Create event timestamps
5. Split by match (no overlap between train/val/test matches)
"""
```

---

## 6. Custom Domain Video

### Collection Strategy

We need **10-20 video clips** of real-world content for our demonstration:

| Content Type | Count | Duration | Source |
|---|---|---|---|
| Soccer match | 3-5 clips | 2-5 min each | YouTube (fair use) |
| Basketball game | 2-3 clips | 2-5 min each | YouTube |
| News broadcast | 2-3 clips | 1-2 min each | TV recording |
| Surveillance (outdoor) | 2-3 clips | 3-5 min each | Public dataset |
| Social media | 3-5 clips | 30s-2 min each | TikTok/Instagram |

### Annotation

```python
"""
Custom Video Annotation:
1. Annotate bounding boxes for first 100 frames of each clip
2. Annotate key events (goals, fouls, scene changes)
3. Annotate text regions (scoreboards, jerseys)
4. Create ground truth claims for evaluation

Tools:
- CVAT (free, open-source) for bounding boxes
- Label Studio for multi-modal annotation
- Manual annotation for events/claims
"""
```

---

## 7. Data Format Standards

### Detection Format

```json
{
  "image_id": "000001",
  "file_name": "frame_000001.jpg",
  "width": 1920,
  "height": 1080,
  "detections": [
    {
      "bbox": [100.0, 200.0, 50.0, 120.0],
      "category": "person",
      "score": 0.95,
      "id": "det_001"
    }
  ]
}
```

### Track Format

```json
{
  "track_id": "person_001",
  "frames": [
    {
      "frame_id": 1,
      "timestamp_ms": 33,
      "bbox": [100.0, 200.0, 50.0, 120.0],
      "keypoints_2d": [[120, 210], [125, 215], ...],
      "confidence": 0.95,
      "occluded": false
    }
  ],
  "attributes": {
    "class": "person",
    "first_frame": 1,
    "last_frame": 300
  }
}
```

### Event Format

```json
{
  "event_id": "evt_001",
  "event_type": "goal",
  "timestamp_ms": 45000,
  "duration_ms": 3000,
  "entity_ids": ["person_003", "ball_001"],
  "description": "Player 3 kicks ball into goal",
  "confidence": 1.0,
  "ground_truth": true
}
```

---

## 8. Data Quality Checks

Run these checks after preprocessing:

```bash
# Script: scripts/verify_datasets.py

# Checks to run:
# 1. All images load without errors
# 2. All bounding boxes are within image bounds
# 3. No duplicate image IDs
# 4. All annotation files are valid JSON
# 5. Train/val/test splits have no overlap
# 6. Class distribution is reasonable
# 7. Image dimensions are consistent
# 8. No empty annotations (every image has at least one object)
```

---

## 9. Dataset Versioning

```bash
# Use DVC (Data Version Control) for dataset versioning
pip install dvc

# Initialize
dvc init
dvc remote add -d storage /path/to/storage

# Track datasets
dvc add data/processed/coco
dvc add data/processed/mot
dvc add data/processed/market1501
dvc add data/processed/soccernet
dvc add data/processed/custom

# Commit dataset versions with git
git add data/*.dvc .gitignore
git commit -m "Dataset v1: all preprocessed datasets"
```

---

## 10. Preprocessing Timeline

| Day | Task | Duration |
|---|---|---|
| Day 2 | Download COCO 2017 | 2-4 hours |
| Day 2 | Download MOT17 | 1-2 hours |
| Day 3 | Download Market1501 | 10 minutes |
| Day 3 | Download SoccerNet v2 | 2-4 hours |
| Day 3 | Collect custom video clips | 2-3 hours |
| Day 4 | Preprocess COCO (verify, split, extract crops) | 2-3 hours |
| Day 4 | Preprocess MOT17 | 1-2 hours |
| Day 5 | Preprocess Market1501 | 30 minutes |
| Day 5 | Preprocess SoccerNet v2 | 2-3 hours |
| Day 5 | Annotate custom video clips | 4-8 hours |
| Day 5 | Run data quality checks | 1 hour |
| Day 5 | Set up DVC versioning | 30 minutes |

**Total: ~20-30 hours across 4 days**
