# Pose Estimation Architecture

## Purpose

Estimate 2D and 3D body pose from detected person regions. Produces joint positions, visibility scores, and confidence per joint. Critical for action recognition, sports analysis, and occlusion reasoning.

---

## Interfaces

### Input

```
PoseInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  image:           Tensor[H, W, 3] uint8
  person_detections: Detection[]            # from detection pipeline, filtered to person class
  crop_padding:    float                    # padding around bbox for crop, default 0.1
}
```

### Output

```
PoseBatch {
  frame_id:        uint64
  timestamp_ns:    uint64
  poses:           PoseResult[]
  model_id:        string                   # e.g. "DETRPose-S/1.0"
  inference_ms:    float
}

PoseResult {
  entity_id:       string                   # linked to detection
  keypoints_2d:    [K][2]float              # K joints, (x, y) normalized to crop
  keypoint_scores: [K]float                 # per-joint confidence
  pose_score:      float                    # aggregate OKS-based score
  body_parts:      BodyPart[]               # occluded/visible per part
  occluded_joints: uint64                   # bitmask of occluded joints
}

BodyPart {
  joint_indices:   int[]
  visibility:      float                    # 0=occluded, 1=visible
  estimation_method: Enum                   # DIRECT_OBSERVATION | TEMPORAL_EXTRAPOLATION | PRIOR
}
```

### API

```
estimate_pose(input: PoseInput, config: PoseConfig) -> PoseBatch
```

---

## Data Contracts

### Model Selection

| Constraint | Model | Latency | AP50:95 | Params |
|---|---|---|---|---|
| Ultra-fast (<2.5ms) | DETRPose-S (A10) | 2.39ms | 67.0 | 11.9M |
| Fast (2.5-5ms) | DETRPose-M (A10) | 3.67ms | 69.4 | 23.5M |
| Balanced (5-10ms) | DETRPose-L (A10) | 5.08ms | 72.5 | 36.8M |
| Peak (8-10ms) | DETRPose-X (A10) | 8.59ms | 73.3 | 82.3M |
| Single-pass (T4) | RF-DETR-KP | 9.7ms | 71.8 | 40.7M |

### Runtime Configuration

```
PoseConfig {
  model:           string                   # "DETRPose-S" | "DETRPose-L" | ...
  input_size:      [2]int                   # e.g. [640, 640]
  num_keypoints:   int                      # 17 (COCO) or 23 (body)
  confidence_threshold: float               # default 0.3
  use_3d:          bool                     # run IK for 3D lift
  device:          string
  precision:       string                   # "fp16"
}
```

### Keypoint Schema (COCO 17)

```
COCO_KEYPOINTS = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
  [15,13],[13,11],[16,14],[14,12],[11,12],[11,13],[12,14],
  [1,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8,10],
  [1,0],[2,0],[0,11],[0,12]
]
```

---

## Latency Budget

### 30 FPS Critical Path

| Stage | Budget | Notes |
|---|---|---|
| Crop from detections | 0.2–0.5 ms | GPU crop, no CPU copy |
| Preprocess (resize, normalize) | 0.3–0.8 ms | |
| Model inference | 2.39–5.08 ms | DETRPose-S or M |
| Keypoint decode | 0.1–0.3 ms | Argmax + softnms |
| Output publish | 0.2–0.5 ms | |
| **Total pose** | **3–7 ms** | |

### 3D Pose Lift (async, 3–10 Hz)

- HybrIK-lite or SMPL-lite for 2D→3D lifting.
- Adds 5–15ms per entity.
- Runs on subset of entities (high-confidence, action-relevant).

---

## Dependencies

### Upstream
- `01_perception/01_detection.md` — Provides person crops
- `01_perception/03_tracking.md` — Entity IDs for cross-frame linking

### Downstream
- `03_state/01_world_state.md` — Pose in world state
- `03_state/03_trajectory_model.md` — Joint trajectories
- `03_state/04_event_detection.md` — Pose-based event triggers
- `05_reasoning/03_deep_vlm_reasoner.md` — Pose context for VLM

---

## Failure Modes

| Failure | Detection | Response |
|---|---|---|
| No person detections | Empty input | Skip pose, return empty |
| Low-confidence keypoints | score < threshold | Mark as occluded, use temporal extrapolation |
| Occluded body parts | visibility < 0.3 | Use prior/kinematic model |
| GPU OOM (large batch) | CUDA OOM | Reduce batch size, process sequentially |
| Model mismatch | Different joint count | Reproject to COCO format |

---

## Reality Check 2026

### DETRPose vs YOLO-pose (from benchmark_report_2026.md):
- DETRPose-S matches YOLOv8-X and YOLO11-X accuracy (67.0 vs 67.3/67.2 AP) with **81% fewer params** and **52% faster** inference.
- DETRPose-L at 72.5 AP, 5.08ms — best accuracy/latency tradeoff for real-time.
- On CrowdPose (occluded): DETRPose-X reaches 75.1 AP, 81.3 APE (easy), 68.1 APH (hard).

### Resolution Scaling (RF-DETR-KP):
- 312×312: 61.1 AP, 4.5ms
- 576×576: 71.8 AP, 9.8ms
- 888×888: 74.2 AP, 25.9ms
- Same checkpoint — only input resolution changes, no retraining.

### 3D Pose (research, not production-verified):
- HybrIK-lite: ~5ms additional, reasonable for async path.
- ViDiHand (hands): 13.5mm PA-MPJPE but 140ms latency — not real-time.
- MoRo (multi-person occluded): +3.5mm MPJPE improvement, 63ms — async only.
