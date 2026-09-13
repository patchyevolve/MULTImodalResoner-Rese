# Camera Motion Estimation Architecture

## Purpose

Estimate camera ego-motion (pan, tilt, zoom, rotation) from consecutive frames. Critical for: distinguishing camera motion from object motion, video stabilization, and camera-based hallucination detection (OmniVChall 2026: ~38.57% accuracy drop when camera motion is involved).

---

## Interfaces

### Input

```
CameraMotionInput {
  frame_id:        uint64
  prev_frame_id:   uint64
  timestamp_ns:    uint64
  image:           Tensor[H, W, 3] uint8
  prev_image:      Tensor[H, W, 3] uint8
  detections:      DetectionBatch            # optional, for object-relative motion
}
```

### Output

```
CameraMotionOutput {
  frame_id:        uint64
  timestamp_ns:    uint64
  motion_type:     Enum                     # STATIC | PAN_LEFT | PAN_RIGHT | TILT_UP | TILT_DOWN | ZOOM_IN | ZOOM_OUT | ROTATION | HANDHELD
  homography:      [3][3]float              # 2D homography between frames
  optical_flow:    Tensor[H, W, 2] float    # dense optical flow (optional)
  ego_motion:      [6]float                 # 3D rotation + translation (if available)
  motion_magnitude: float                   # scalar motion intensity
  confidence:      float
  inference_ms:    float
}
```

### API

```
estimate_motion(input: CameraMotionInput, config: CameraConfig) -> CameraMotionOutput
compensate_motion(image: Tensor, motion: CameraMotionOutput) -> Tensor
```

---

## Data Contracts

### Method Selection

| Method | Latency | Accuracy | Notes |
|---|---|---|---|
| Lucas-Kanade (sparse) | 0.5–1ms | Low | Fast, feature-based |
| Farneback (dense) | 2–5ms | Medium | Good balance |
| RAFT (learning-based) | 5–15ms | High | State-of-the-art |
| ORB + RANSAC | 1–3ms | Medium | Robust to outliers |

### Runtime Configuration

```
CameraConfig {
  method:          Enum                     # SPARSE | DENSE | LEARNING
  use_gpu:         bool                     # default true
  grid_size:       int                      # for sparse features, default 20
  min_features:    int                      # default 100
  ransac_threshold: float                   # default 3.0
}
```

---

## Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Feature detection | 0.5–1 ms | Sparse: ORB/SIFT |
| Feature matching | 0.5–1 ms | BF matcher or FLANN |
| Motion estimation | 0.5–3 ms | RANSAC + homography |
| **Total (sparse)** | **1–3 ms** | |
| **Total (dense)** | **2–5 ms** | |
| **Total (learning)** | **5–15 ms** | Async path |

---

## Dependencies

### Upstream
- `10_infrastructure/01_data_schemas.md` — Frame format

### Downstream
- `01_perception/03_tracking.md` — Motion-compensated detection
- `01_perception/02_pose_estimation.md` — Ego-motion compensation
- `03_state/01_world_state.md` — Camera state in world model
- `09_domains/02_general_multimedia.md` — Camera motion as context

---

## Reality Check 2026

### OmniVChall Finding (ICML 2026):
- Camera-based reasoning is the **worst-performing hallucination type** (~38.57% accuracy drop).
- Models confuse lens motion with physical object motion.
- Camera type (lens/viewpoint dynamics) is a novel hallucination category.

### Practical Implications:
- Camera motion estimation is **not optional** — it's critical for correct object motion interpretation.
- Sports: camera follows ball/player — ego-motion must be subtracted to get true object velocity.
- News: frequent cuts — camera motion type helps segment different shots.
- Handheld: high-frequency jitter — must distinguish from object vibration.
