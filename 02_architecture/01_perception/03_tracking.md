# Object Tracking Architecture

## Purpose

Maintain consistent identity of detected objects across frames. Links detections into tracks, handles occlusion, and provides motion predictions for the next frame's detector. This is the glue between per-frame detection and temporal reasoning.

---

## Interfaces

### Input

```
TrackingInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  detections:      DetectionBatch            # from detection pipeline
  frame_image:     Tensor[H, W, 3] uint8    # for appearance features (optional)
}
```

### Output

```
TrackingOutput {
  frame_id:        uint64
  timestamp_ns:    uint64
  tracks:          Track[]
  statistics:      TrackingStats
}

Track {
  entity_id:       string                   # persistent ID across frames
  bbox_norm:       [4]float                 # current bounding box
  class_id:        uint16
  age:             uint32                   # frames since first detection
  hits:            uint32                   # consecutive frames matched
  time_since_update: uint32                 # frames since last match
  velocity:        [2]float                 # predicted velocity (px/frame)
  score:           float                    # track confidence
  is_occluded:     bool
  features:        string[]                 # recent appearance feature refs
}

TrackingStats {
  active_tracks:   int
  new_tracks:      int
  lost_tracks:     int
  removed_tracks:  int
  inference_ms:    float
}
```

### API

```
update_tracks(input: TrackingInput, config: TrackConfig) -> TrackingOutput
get_track_history(entity_id: string, max_frames: int) -> Track[]
```

---

## Data Contracts

### Tracker Selection

| Tracker | HOTA | IDF1 | Latency | Notes |
|---|---|---|---|---|
| ByteTrack | 67.7 | 79.5 | 0.2ms | Default. Simple, fast, robust. |
| NvSORT (DeepStream) | ~65 | ~75 | 0.1ms | NVIDIA optimized, minimal |
| NvDeepSORT | ~70 | ~82 | 0.5ms | Appearance features, slower |
| Bot-SORT | 68.5 | 80.1 | 0.3ms | Camera motion compensation |

### Runtime Configuration

```
TrackConfig {
  tracker:         string                   # "ByteTrack" | "Bot-SORT" | ...
  track_buffer:    int                      # frames to keep lost tracks, default 30
  match_threshold: float                    # IoU threshold for matching, default 0.8
  new_track_threshold: float                # min score for new track, default 0.6
  max_age:         int                      # frames before track removal, default 60
  min_hits:        int                      # frames before track confirmed, default 3
  use_reid:        bool                     # appearance features for re-matching
  camera_motion_compensation: bool          # global motion compensation
}
```

### State Machine

```
Track States:
  TENTATIVE  →  CONFIRMED  →  LOST  →  REMOVED
     ↑              ↑           ↑
     └── new det ───┘── matched ─┘── lost > max_age ──→ REMOVED
```

---

## Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Feature extraction (appearance) | 0–0.3ms | Optional, only if use_reid=true |
| IoU/feature matching | 0.1–0.3ms | Hungarian algorithm on small matrix |
| State update (Kalman filter) | 0.1–0.2ms | Per-track predict + update |
| Track management | 0.05–0.1ms | Create/confirm/remove |
| Output publish | 0.1–0.2ms | |
| **Total tracking** | **0.2–0.5ms** | Very lightweight |

---

## Dependencies

### Upstream
- `01_perception/01_detection.md` — DetectionBatch input

### Downstream
- `01_perception/02_pose_estimation.md` — Entity-linked crops
- `01_perception/06_reid.md` — Feature matching for re-identification
- `03_state/01_world_state.md` — Track → Entity mapping
- `03_state/03_trajectory_model.md` — Track velocity as trajectory input
- `07_scheduler/01_multi_rate_scheduler.md` — Detector skip decisions

---

## Failure Modes

| Failure | Detection | Response |
|---|---|---|
| ID switch | Entity ID changes mid-track | Log, report, downstream must handle |
| Track fragmentation | One object → multiple tracks | Re-ID merge on recovery |
| Ghost tracks | Track without detections | Remove after max_age |
| Occlusion → identity swap | Two objects swap IDs | Use appearance features, physics constraints |
| Tracker overflow | >1000 active tracks | Force-remove oldest low-score tracks |

---

## Reality Check 2026

### ByteTrack (default choice):
- Simplest tracker with competitive HOTA (67.7) and IDF1 (79.5).
- No appearance features — relies on IoU + motion prediction.
- 0.2ms per frame — negligible overhead.
- Handles up to ~500 objects per frame comfortably.

### Bot-SORT (if camera motion compensation needed):
- Global motion compensation (GMC) module handles camera shake/pan.
- Slightly higher latency (0.3ms) but better HOTA (68.5).
- Recommended for handheld/UAV footage.

### SAM 3 Memory Tracker (for segmentation tracks):
- SAM 3 includes built-in memory tracker for mask propagation.
- 30ms per image with 100+ objects (H200).
- Only use if pixel-level masks are required — much heavier than ByteTrack.
