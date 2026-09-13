# Multimodal Video Reasoner — Complete Architecture

> **One file. Everything.** Every component, every interface, every data contract, every latency budget, every failure mode, every dependency, every reality check. No references to other files needed.
>
> **Approach:** Bottom-up composition. Each component is a self-contained architecture. The full system is composed from these pieces.
>
> **Consistency rule:** Every component follows: Purpose → Interfaces → Data Contracts → Latency Budget → Dependencies → Failure Modes → Reality Check 2026.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Composition Rules](#2-composition-rules)
3. [Layer 01 — Perception](#3-layer-01--perception)
4. [Layer 02 — Fusion](#4-layer-02--fusion)
5. [Layer 03 — State](#5-layer-03--state)
6. [Layer 04 — Memory](#6-layer-04--memory)
7. [Layer 05 — Reasoning](#7-layer-05--reasoning)
8. [Layer 06 — Calibration](#8-layer-06--calibration)
9. [Layer 07 — Scheduler](#9-layer-07--scheduler)
10. [Layer 08 — Forensics](#10-layer-08--forensics)
11. [Layer 09 — Domains](#11-layer-09--domains)
12. [Layer 10 — Infrastructure](#12-layer-10--infrastructure)
13. [Core Data Contracts](#13-core-data-contracts)
14. [Complete Data Flow](#14-complete-data-flow)
15. [Hardware Reference](#15-hardware-reference)
16. [Deployment Architecture](#16-deployment-architecture)
17. [Reality Check 2026](#17-reality-check-2026)

---

# 1. System Overview

## 1.1 Purpose

A real-time multimodal video reasoning system that ingests live video streams, detects and tracks objects, reasons about events, and outputs structured claims with calibrated confidence scores, prediction sets, and provenance metadata.

## 1.2 Pipeline Summary

```
MULTIMODAL INPUT
  +-- video (RTSP/HTTP/File, 30 FPS, H.264/H.265)
  +-- audio (PCM 16kHz mono)
  +-- metadata/provenance (C2PA manifests, SynthID watermarks)
          |
          v
FAST PERCEPTION BUS  ---------------------------- 30 FPS state path
  +-- detector (RF-DETR-S, 2.3-6.8ms)
  +-- tracker (ByteTrack, 0.2-0.5ms)
  +-- pose (DETRPose-S, 2.39-9.7ms)
  +-- segmentation (RF-DETR-Seg, async 5-10Hz)
  +-- OCR (PaddleOCR, event-triggered)
  +-- embeddings/re-ID (OSNet, event-triggered)
  +-- audio features (Whisper, stream latency)
  +-- camera motion (ORB+RANSAC, 1-3ms)
          |
          v
WORLD STATE (Ring Buffer, <0.1ms)
  +-- entities + attributes
  +-- trajectories + relations
  +-- scene context
  +-- events + uncertainty
          |
   +------+-----------------------+
   |      |          |            |
   v      v          v            v
memory  motion     evidence     prediction
   |      model      graph        model
   +------+----------+------------+
                       |
                       v
               HYPOTHESIS ENGINE (5-15ms)
                       |
               +-------+-------+
               |               |
               v               v
         fast verifier     deep reasoner
         (1-5ms)          (800-3500ms async)
               |               |
               +-------+-------+
                       |
                       v
               CALIBRATION LAYER (3-8ms)
                       |
                       v
          CLAIM + EVIDENCE + CONFIDENCE
          (structured JSON/Protobuf)
```

## 1.3 Critical Path

The real-time loop should never wait on deep reasoning. Deep reasoning consumes state snapshots and writes back higher-level beliefs asynchronously.

## 1.4 Hardware Reference (2026)

**Tier 1 (Minimum viable):** 1x RTX 4090 24GB.
- Hosts: RF-DETR-S/pose (~3 GB) + ByteTrack state (~0.1 GB) + Qwen 2.5-VL 32B AWQ (~20.4 GB peak).
- 30 FPS perception + 0.3-0.5 Hz deep reasoning. Tight but fits.

**Tier 2 (Recommended):** 1x A100 80GB OR 2x RTX 4090 24GB.
- 1 GPU: all of the above + forensic ensemble + 0.5-1 Hz 32B VLM.
- 2 GPU: Perception GPU + Reasoning GPU. 1 Hz VLM reasoning comfortable.

**Tier 3 (Server/Production):** 2+ A100/H100.
- Perception on one; reasoning + forensic on others. 1-2 Hz deep reasoning + redundancy.

---

# 2. Composition Rules

1. **Dependencies flow downward only.** Higher layers depend on lower layers, never the reverse.
2. **Each component publishes typed outputs.** No shared mutable state between components.
3. **Latency budgets are hard constraints.** If a component exceeds its budget, it must degrade gracefully.
4. **Every component has a failure mode.** No silent failures. All errors propagate as structured error objects.
5. **Async components must be idempotent.** VLM reasoning, forensics, and summarization can be retried safely.

---

# 3. Layer 01 — Perception

## 3.1 Detection Pipeline

### Purpose

Detect and classify objects in each video frame. Produces bounding boxes, class labels, and confidence scores for every detected entity. This is the foundational component — all downstream tracking, pose, and reasoning depend on detection quality.

### Interfaces

**Input:**
```
FrameInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  image:           Tensor[H, W, 3] uint8   # BGR or RGB
  stream_id:       string
  decode_latency_ms: float
}
```

**Output:**
```
DetectionBatch {
  frame_id:        uint64
  timestamp_ns:    uint64
  detections:      Detection[]
  model_id:        string                   # e.g. "RF-DETR-S/1.8.2"
  inference_ms:    float
  total_ms:        float
}

Detection {
  bbox_norm:       [4]float                 # [x_center, y_center, w, h] normalized 0-1
  class_id:        uint16
  class_name:      string
  score:           float
  feature_ref:     string                   # embedding pointer for Re-ID
}
```

**API:**
```
detect_batch(frames: FrameInput[], config: DetectConfig) -> DetectionBatch
detect_single(frame: FrameInput, config: DetectConfig) -> DetectionBatch
```

### Data Contracts

**Model Selection Decision Matrix:**

| Constraint | Model | Latency T4 | AP50:95 | Params |
|---|---|---|---|---|
| Ultra-fast (<2.5ms) | RF-DETR-N | 2.3ms | 48.4 | 30.5M |
| Fast (2.5-4ms) | RF-DETR-S | 3.5ms | 53.0 | 32.1M |
| Balanced (4-7ms) | RF-DETR-M | 4.4ms | 54.7 | 33.7M |
| High-accuracy (7-12ms) | RF-DETR-L | 6.8ms | 56.5 | 33.9M |
| Peak accuracy (12-18ms) | RF-DETR-XL | 11.5ms | 58.6 | 126.4M |

**Runtime Configuration:**
```
DetectConfig {
  model:           string
  input_size:      [2]int                   # e.g. [512, 512]
  confidence_threshold: float               # default 0.5
  nms_threshold:   float                    # default 0.7
  max_detections:  int                      # default 100
  device:          string
  precision:       string                   # "fp16" | "fp8" | "int8"
}
```

**Quality Flags:**
```
DetectionQuality {
  blur_score:               float           # 0=sharp, 1=blurry
  detector_in_distribution: float           # model confidence on OOD-ness
  input_occluded_frac:      float
  compression_artifacts:    bool
}
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Preprocess (resize, normalize) | 1-2 ms | Pinned memory, zero-copy |
| Model inference | 2.3-6.8 ms | RF-DETR-S or M |
| NMS / postprocess | 0.5-1 ms | CPU or GPU NMS |
| Output publish | 0.2-0.5 ms | Ring buffer write |
| **Total detection** | **4-10 ms** | Leaves 23-29ms for rest of pipeline |

### Dependencies

**Upstream:** Hardware topology (GPU assignment), Camera motion (motion-compensated input, optional)
**Downstream:** Tracking, Pose estimation, Re-ID, World state

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Model OOM | CUDA OOM error | Downgrade to RF-DETR-N, reduce resolution |
| Latency spike >15ms | Timing monitor | Skip frame, propagate tracker prediction |
| All scores < threshold | Empty detection set | Report "no detections", don't trigger events |
| GPU thermal throttle | Utilization >95% | Reduce to N model, increase detector skip |
| Input corruption | Decoder error | Drop frame, log, continue |

### Reality Check 2026

- RF-DETR-S beats YOLO26-S at every matched latency tier (53.0 vs 47.7 AP50:95).
- YOLO26-N is faster (1.7ms vs 2.3ms) but gives up 8.1 AP.
- RF-DETR-2XL reaches 60.1 AP50:95 — no YOLO variant comes close.
- On CPU: YOLO26 wins decisively. RF-DETR has 30M+ params, no meaningful CPU path.
- RTX 5090: RF-DETR-L TRT FP16: 1.33ms, 753 FPS.
- RF-DETR-L RTX 4090 JIT FP16: 6.1ms, 163 FPS. TRT FP16: 1.33ms, 753 FPS.

---

## 3.2 Pose Estimation

### Purpose

Estimate 2D and 3D body pose from detected person regions. Produces joint positions, visibility scores, and confidence per joint. Critical for action recognition, sports analysis, and occlusion reasoning.

### Interfaces

**Input:**
```
PoseInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  image:           Tensor[H, W, 3] uint8
  person_detections: Detection[]
  crop_padding:    float                    # default 0.1
}
```

**Output:**
```
PoseBatch {
  frame_id:        uint64
  timestamp_ns:    uint64
  poses:           PoseResult[]
  model_id:        string
  inference_ms:    float
}

PoseResult {
  entity_id:       string
  keypoints_2d:    [K][2]float
  keypoint_scores: [K]float
  pose_score:      float
  body_parts:      BodyPart[]
  occluded_joints: uint64                   # bitmask
}

BodyPart {
  joint_indices:   int[]
  visibility:      float
  estimation_method: Enum                   # DIRECT_OBSERVATION | TEMPORAL_EXTRAPOLATION | PRIOR
}
```

**API:**
```
estimate_pose(input: PoseInput, config: PoseConfig) -> PoseBatch
```

### Data Contracts

**Model Selection:**

| Constraint | Model | Latency | AP50:95 | Params |
|---|---|---|---|---|
| Ultra-fast (<2.5ms) | DETRPose-S (A10) | 2.39ms | 67.0 | 11.9M |
| Fast (2.5-5ms) | DETRPose-M (A10) | 3.67ms | 69.4 | 23.5M |
| Balanced (5-10ms) | DETRPose-L (A10) | 5.08ms | 72.5 | 36.8M |
| Peak (8-10ms) | DETRPose-X (A10) | 8.59ms | 73.3 | 82.3M |
| Single-pass (T4) | RF-DETR-KP | 9.7ms | 71.8 | 40.7M |

**COCO 17 Keypoint Schema:**
```
COCO_KEYPOINTS = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle"
]
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Crop from detections | 0.2-0.5 ms | GPU crop, no CPU copy |
| Preprocess (resize, normalize) | 0.3-0.8 ms | |
| Model inference | 2.39-5.08 ms | DETRPose-S or M |
| Keypoint decode | 0.1-0.3 ms | Argmax + softnms |
| Output publish | 0.2-0.5 ms | |
| **Total pose** | **3-7 ms** | |

3D Pose Lift (async, 3-10 Hz): HybrIK-lite or SMPL-lite, adds 5-15ms per entity.

### Dependencies

**Upstream:** Detection (person crops), Tracking (entity IDs)
**Downstream:** World state (pose), Trajectory model (joint trajectories), Event detection (pose-based triggers), Deep VLM (pose context)

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| No person detections | Empty input | Skip pose, return empty |
| Low-confidence keypoints | score < threshold | Mark as occluded, use temporal extrapolation |
| Occluded body parts | visibility < 0.3 | Use prior/kinematic model |
| GPU OOM | CUDA OOM | Reduce batch size |

### Reality Check 2026

- DETRPose-S matches YOLOv8-X and YOLO11-X accuracy (67.0 vs 67.3/67.2 AP) with 81% fewer params and 52% faster.
- DETRPose-L: 72.5 AP, 5.08ms — best accuracy/latency tradeoff for real-time.
- CrowdPose (occluded): DETRPose-X 75.1 AP, 68.1 APH (hard).
- Resolution scaling (RF-DETR-KP): 312x312→61.1 AP 4.5ms, 576x576→71.8 AP 9.8ms, 888x888→74.2 AP 25.9ms.

---

## 3.3 Object Tracking

### Purpose

Maintain consistent identity of detected objects across frames. Links detections into tracks, handles occlusion, and provides motion predictions for the next frame's detector.

### Interfaces

**Input:**
```
TrackingInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  detections:      DetectionBatch
  frame_image:     Tensor[H, W, 3] uint8   # optional appearance features
}
```

**Output:**
```
TrackingOutput {
  frame_id:        uint64
  timestamp_ns:    uint64
  tracks:          Track[]
  statistics:      TrackingStats
}

Track {
  entity_id:       string                   # persistent ID across frames
  bbox_norm:       [4]float
  class_id:        uint16
  age:             uint32
  hits:            uint32
  time_since_update: uint32
  velocity:        [2]float
  score:           float
  is_occluded:     bool
  features:        string[]
}

TrackingStats {
  active_tracks:   int
  new_tracks:      int
  lost_tracks:     int
  removed_tracks:  int
  inference_ms:    float
}
```

### Data Contracts

**Tracker Selection:**

| Tracker | HOTA | IDF1 | Latency | Notes |
|---|---|---|---|---|
| ByteTrack | 67.7 | 79.5 | 0.2ms | Default. Simple, fast, robust. |
| NvSORT (DeepStream) | ~65 | ~75 | 0.1ms | NVIDIA optimized |
| NvDeepSORT | ~70 | ~82 | 0.5ms | Appearance features |
| Bot-SORT | 68.5 | 80.1 | 0.3ms | Camera motion compensation |

**State Machine:**
```
Track States:
  TENTATIVE  ->  CONFIRMED  ->  LOST  ->  REMOVED
     ^              ^           ^
     +-- new det ----+-- matched -+-- lost > max_age --> REMOVED
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Feature extraction (appearance) | 0-0.3ms | Optional |
| IoU/feature matching | 0.1-0.3ms | Hungarian algorithm |
| State update (Kalman filter) | 0.1-0.2ms | Per-track predict + update |
| Track management | 0.05-0.1ms | Create/confirm/remove |
| **Total tracking** | **0.2-0.5ms** | Very lightweight |

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| ID switch | Entity ID changes mid-track | Log, report |
| Track fragmentation | One object → multiple tracks | Re-ID merge on recovery |
| Ghost tracks | Track without detections | Remove after max_age |
| Occlusion → identity swap | Two objects swap IDs | Appearance features + physics |
| Tracker overflow | >1000 active tracks | Force-remove oldest low-score |

### Reality Check 2026

- ByteTrack: simplest tracker with competitive HOTA (67.7) and IDF1 (79.5), 0.2ms per frame.
- Bot-SORT: global motion compensation for handheld/UAV footage, 0.3ms, HOTA 68.5.
- SAM 3 Memory Tracker: 30ms per image with 100+ objects (H200) — only for segmentation tracks.

---

## 3.4 Instance Segmentation

### Purpose

Produce pixel-level masks for detected objects. Used when precise object boundaries matter — occlusion reasoning, spatial relationship analysis, fine-grained visual understanding.

### Interfaces

**Input:**
```
SegmentationInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  image:           Tensor[H, W, 3] uint8
  detections:      DetectionBatch            # optional: prompt-based
  prompt_type:     Enum                     # BOX | POINT | TEXT | EVERYTHING
}
```

**Output:**
```
SegmentationBatch {
  frame_id:        uint64
  masks:           MaskResult[]
  model_id:        string
  inference_ms:    float
}

MaskResult {
  entity_id:       string
  mask_rle:        string                   # run-length encoded
  mask_area:       int
  mask_iou_score:  float
  bbox_overlap:    float
}
```

### Data Contracts

**Model Selection:**

| Model | AP50:95 (Mask) | Latency T4 | Params | Use Case |
|---|---|---|---|---|
| RF-DETR-Seg-S | 43.1 | 4.4ms | 33.7M | Fast, closed-set |
| RF-DETR-Seg-M | 45.3 | 5.9ms | 35.7M | Balanced |
| RF-DETR-Seg-L | 47.1 | 8.8ms | 36.2M | High accuracy |
| YOLO26-M-Seg | 44.0 | 6.32ms | 23.6M | CPU-friendly |
| SAM 3 | 47.0 (LVIS) | 30ms (H200) | 848M | Open-vocabulary |
| SAM 3.1 | 47.0+ | ~16fps (H100) | 848M | Multi-object (16/pass) |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Crop + preprocess | 0.3-0.5 ms | |
| RF-DETR-Seg inference | 4.4-8.8 ms | |
| Mask decode | 0.5-1 ms | RLE encoding |
| **Total (RF-DETR-Seg)** | **5-10 ms** | Async path, 5-10 Hz |
| SAM 3 prompt-based | 30ms (H200) | Async, event-triggered |
| SAM 3 everything | 2921ms (3090) | NOT real-time, batch only |

### Dependencies

**Upstream:** Detection (bounding boxes), World state (entity regions)
**Downstream:** World state (pixel masks), Deep VLM (visual context), Deepfake detection (spatial artifacts)

### Reality Check 2026

- RF-DETR-Seg leads at every matched latency tier vs YOLO26-Seg.
- SAM 3: 30ms per image with 100+ objects on H200 (prompt-based). Everything mode is 100x slower.
- SAM 3.1 adds Object Multiplex: 16 objects per pass, 7x faster throughput.
- Real-time (30 FPS): RF-DETR-Seg-S or skip entirely. Async (5-10 Hz): RF-DETR-Seg-M or L.

---

## 3.5 OCR Pipeline

### Purpose

Extract text from video frames — jerseys, scoreboards, signs, captions, overlays. Event-triggered, not per-frame.

### Interfaces

**Input:**
```
OCRInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  image:           Tensor[H, W, 3] uint8
  roi:             [4]float                 # optional region of interest
  trigger_reason:  string                   # "scene_cut" | "new_text_detected" | "periodic"
}
```

**Output:**
```
OCRBatch {
  frame_id:        uint64
  results:         OCRResult[]
  inference_ms:    float
}

OCRResult {
  text:            string
  bbox_norm:       [4]float
  confidence:      float
  language:        string
  text_type:       Enum                     # JERSEY | SCOREBOARD | SIGN | CAPTION | OVERLAY
}
```

### Data Contracts

| Model | WER | Latency | Notes |
|---|---|---|---|
| PaddleOCR v4 | ~3% | 15-30ms | Best general-purpose |
| EasyOCR | ~5% | 20-40ms | Good multilingual |
| Tesseract 5 | ~8% | 10-20ms | Fast, less accurate |
| TrOCR (HuggingFace) | ~2% | 50-100ms | Transformer-based, slow |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Text region detection | 3-5 ms | EAST/DB detector |
| Text recognition | 10-20 ms | Per region |
| Postprocess (NMS, decode) | 1-2 ms | |
| **Total (2-5 regions)** | **15-40 ms** | Event-triggered only |

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| No text detected | Empty results | Log, continue |
| Low confidence | score < threshold | Report as uncertain |
| OCR on blurry frame | blur_score > 0.7 | Skip, wait for keyframe |
| Wrong language | Low confidence | Try next language model |

### Reality Check 2026

- PaddleOCR v4: ~3% WER, 15-30ms per frame on GPU.
- Sports jersey numbers: accuracy >95% (high contrast, large).
- VLM as OCR alternative: Qwen3-VL can read text directly, higher latency (100ms+) but handles complex layouts.

---

## 3.6 Re-Identification

### Purpose

Match detected persons/objects across non-overlapping camera views or after long occlusion. Event-triggered, not per-frame.

### Interfaces

**Input:**
```
ReIDInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  crop:            Tensor[H, W, 3] uint8
  entity_id:       string
  query_type:      Enum                     # GALLERY_MATCH | ENROLL | VERIFY
}
```

**Output:**
```
ReIDOutput {
  entity_id:       string
  embedding:       [512]float
  matched_id:      string
  match_score:     float
  rank:            int
  candidates:      ReIDCandidate[]
}
```

### Data Contracts

| Model | Rank-1 (Market1501) | Latency | Notes |
|---|---|---|---|
| OSNet | 95.6% | 2-5ms | Lightweight, default |
| TransReID | 97.0% | 20-50ms | Transformer, slow |
| CLIP-based | 94.0% | 10-20ms | Cross-modal capable |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Crop + preprocess | 0.3-0.5 ms | |
| Embedding extraction | 2-5 ms | OSNet |
| Gallery search (FAISS) | 0.5-2 ms | 10K gallery |
| **Total** | **3-8 ms** | Event-triggered |

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| No gallery match | score < threshold | Create new entity |
| Duplicate enrollments | Multiple embeddings for same entity | Consolidation job |
| Gallery overflow | >10K entries | LRU eviction + re-enroll |

### Reality Check 2026

- OSNet: 95.6% Rank-1, 2-5ms — sufficient for most scenarios.
- TransReID: 97.0% but 20-50ms — only for high-value re-identification.
- Cross-camera matching requires appearance + spatial reasoning.

---

## 3.7 Audio Feature Extraction

### Purpose

Extract speech, speaker diarization, and audio events from the audio stream. Provides text transcripts, speaker IDs, and audio event tags to the fusion layer.

### Interfaces

**Input:**
```
AudioInput {
  chunk_id:        uint64
  timestamp_ns:    uint64
  audio:           Tensor[T] float32        # T samples at 16kHz
  sample_rate:     int                      # 16000
}
```

**Output:**
```
AudioOutput {
  chunk_id:        uint64
  transcript:      string
  speaker_id:      string
  audio_events:    AudioEvent[]
  confidence:      float
  inference_ms:    float
}

AudioEvent {
  event_class:     string                   # "whistle" | "cheer" | "siren" | "silence"
  start_ms:        float
  end_ms:          float
  score:           float
}
```

### Data Contracts

| Model | WER | Latency | Notes |
|---|---|---|---|
| Whisper-large-v3 | ~3% | 50-100ms (GPU) | Best accuracy |
| Whisper-base | ~8% | 20-40ms | Faster, less accurate |
| NVIDIA NeMo | ~4% | 30-60ms | Streaming capable |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Audio chunk receive | 0 ms | Already in memory |
| Feature extraction | 5-10 ms | Mel spectrogram |
| Model inference | 50-100 ms | Whisper-large-v3 |
| **Total** | **55-110 ms** | Stream latency (acceptable) |

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| No audio stream | Missing input | Skip audio features |
| Low SNR | Confidence < 0.3 | Mark transcript uncertain |
| Speaker overlap | Multiple active speakers | Diarization uncertainty flag |

---

## 3.8 Camera Motion Estimation

### Purpose

Estimate ego-motion from consecutive frames. Provides camera shake, pan, tilt, and zoom information for motion-compensated detection and spatial reasoning.

### Interfaces

**Input:**
```
CameraMotionInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  image_curr:      Tensor[H, W] float32    # grayscale
  image_prev:      Tensor[H, W] float32
}
```

**Output:**
```
CameraMotionOutput {
  frame_id:        uint64
  motion_type:     Enum                     # STATIC | PAN | TILT | ZOOM | SHAKE | UNKNOWN
  homography:      [3][3]float              # 3x3 transform matrix
  translation:     [3]float                 # estimated translation
  rotation:        [3]float                 # estimated rotation
  confidence:      float
  inference_ms:    float
}
```

### Data Contracts

| Method | Latency | Accuracy | Notes |
|---|---|---|---|
| ORB + RANSAC | 1-3ms | Good | Default, fast |
| Farneback optical flow | 3-8ms | Better | Denser motion field |
| RAFT optical flow | 10-30ms | Best | Deep learning, async |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Feature extraction (ORB) | 0.5-1 ms | |
| Feature matching | 0.3-0.5 ms | |
| RANSAC homography | 0.5-1.5 ms | |
| **Total** | **1-3 ms** | |

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Few features (blank scene) | <10 matches | Mark as UNKNOWN |
| Sudden cut (no correspondence) | RANSAC inlier ratio < 10% | Mark as scene cut |
| Rolling shutter artifacts | Homography residual high | Downgrade to translation-only |

---

# 4. Layer 02 — Fusion

## 4.1 Multi-Modal Fusion

### Purpose

Combine vision, audio, and text features into a unified representation. Handles missing modalities gracefully and weights each modality by its reliability.

### Interfaces

**Input:**
```
FusionInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  vision_features: Tensor                    # from perception pipeline
  audio_features:  Tensor                    # from audio pipeline
  text_features:   Tensor                    # from OCR pipeline
  modality_mask:   [3]bool                   # which modalities are present
  confidence:      [3]float                  # per-modality confidence
}
```

**Output:**
```
FusionOutput {
  frame_id:        uint64
  fused_features:  Tensor                    # unified representation
  modality_weights: [3]float                 # learned/reported weights
  fusion_confidence: float
  inference_ms:    float
}
```

### Data Contracts

**Fusion Strategies:**

| Strategy | Latency | Notes |
|---|---|---|
| Late fusion (concatenate) | 0.5-1ms | Simplest, default |
| Attention-based fusion | 2-5ms | Cross-modal attention |
| Gated fusion | 1-3ms | Learned gates per modality |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Feature alignment | 0.5-1 ms | Temporal sync |
| Fusion computation | 1-5 ms | Depends on strategy |
| Output normalization | 0.1-0.3 ms | |
| **Total** | **2-6 ms** | Fast path |

### Dependencies

**Upstream:** Detection, Pose, Audio features, OCR, Camera motion
**Downstream:** Temporal fusion → Cross-modal alignment → World state

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing modality | modality_mask false | Use available modalities only |
| Conflicting modalities | Cross-modal disagreement > 0.5 | Flag uncertainty, weight by confidence |
| Feature dimension mismatch | Shape error | Reproject to common dimension |

---

## 4.2 Temporal Fusion

### Purpose

Aggregate fused features across time windows. Provides smoothed, temporally consistent features that reduce noise and capture temporal patterns.

### Interfaces

**Input:**
```
TemporalFusionInput {
  current_features: FusionOutput
  history_window:   FusionOutput[]           # last N frames
  window_size:      int                      # default 5
}
```

**Output:**
```
TemporalFusionOutput {
  frame_id:        uint64
  smoothed_features: Tensor
  temporal_consistency: float                # 0=highly variable, 1=consistent
  motion_energy:   float                     # aggregate motion in window
  inference_ms:    float
}
```

### Data Contracts

**Temporal Aggregation Methods:**

| Method | Latency | Notes |
|---|---|---|
| Exponential Moving Average | 0.5-1ms | Default, simple |
| Sliding window average | 0.3-0.5ms | Uniform weights |
| Transformer self-attention | 2-5ms | Async, high-accuracy |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Window assembly | 0.1-0.2 ms | Ring buffer read |
| Aggregation | 0.3-2 ms | Depends on method |
| Consistency scoring | 0.1-0.3 ms | |
| **Total** | **0.5-3 ms** | |

---

## 4.3 Cross-Modal Alignment

### Purpose

Synchronize features across modalities that may have different timestamps, resolutions, or frame rates. Ensures temporal alignment before state update.

### Interfaces

**Input:**
```
AlignmentInput {
  vision_timestamp: uint64
  audio_timestamp:  uint64
  text_timestamp:   uint64
  vision_features:  Tensor
  audio_features:   Tensor
  text_features:    Tensor
  max_offset_ms:    float                    # default 100ms
}
```

**Output:**
```
AlignmentOutput {
  aligned_timestamp: uint64
  aligned_features:  [3]Tensor               # temporally aligned
  alignment_scores:  [3]float                # 0=poor, 1=exact
  max_offset_ms:     float
}
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Timestamp comparison | 0.1 ms | |
| Feature interpolation | 0.5-1 ms | If offset > threshold |
| Correlation scoring | 0.3-0.5 ms | |
| **Total** | **1-2 ms** | |

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Offset > max_offset_ms | Alignment score < 0.3 | Use closest available, flag misalignment |
| Modality missing | Timestamp = 0 | Skip alignment for that modality |

---

# 5. Layer 03 — State

## 5.1 World State

### Purpose

Typed entity/attribute store. The single source of truth for the current state of the world. Updated every frame from perception, read by reasoning and memory.

### Interfaces

**Input:**
```
WorldStateUpdate {
  frame_id:        uint64
  timestamp_ns:    uint64
  entities:        EntityUpdate[]
  scene:           SceneUpdate
  events:          EventTrigger[]
  camera_motion:   CameraMotionOutput
}
```

**Output:**
```
WorldStateSnapshot {
  frame_id:        uint64
  timestamp_ns:    uint64
  entities:        Entity[]
  trajectories:    Trajectory[]
  relations:       Relation[]
  scene:           Scene
  events:          Event[]
  uncertainty:     UncertaintySummary
  staleness_ms:    float
}

Entity {
  id:              string
  class:           string
  bbox_norm:       [4]float
  pose:            PoseResult
  velocity:        [3]float
  acceleration:    [3]float
  attributes:      map<string, Any>
  confidence:      float
  occlusion_level: float
  tracking_state:  string                   # TENTATIVE | CONFIRMED | LOST
}

Relation {
  subject:         string                   # entity_id
  predicate:       string                   # "kicking" | "near" | "facing"
  object:          string                   # entity_id
  confidence:      float
  spatial:         SpatialRelation
}

Scene {
  location:        string
  activity:        string
  num_entities:    int
  lighting:        string
  weather:         string
}
```

### Data Contracts

**Storage: Lock-free Ring Buffer**
- 30 slots (one per frame at 30 FPS = 1 second of history)
- Pre-allocated at startup, no runtime allocation
- SPSC or MPMC depending on reader count
- Write: perception pipeline (single writer)
- Read: reasoning, memory, API (multiple readers)

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Entity merge (new + existing) | 0.1-0.3 ms | Per entity |
| Relation computation | 0.05-0.1 ms | Spatial + semantic |
| Scene update | 0.05-0.1 ms | |
| Ring buffer write | 0.02-0.05 ms | Lock-free |
| **Total** | **0.3-1 ms** | Per frame |

### Dependencies

**Upstream:** Detection, Pose, Tracking, Fusion (all perception), Camera motion
**Downstream:** Memory (short-term), Reasoning (hypothesis engine), Event detection

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Entity count exceeds capacity | >100 entities | Evict lowest-confidence |
| State staleness > threshold | timestamp difference | Flag stale reads |
| Concurrent write conflict | Atomic failure | Retry with backoff |

---

## 5.2 Entity Tracker

### Purpose

Manage entity lifecycle — creation, confirmation, occlusion handling, re-identification, and removal. Owns the persistent identity of each entity.

### Interfaces

**Input:**
```
EntityTrackerInput {
  frame_id:        uint64
  detections:      DetectionBatch
  tracks:          TrackingOutput
  pose:            PoseBatch
}
```

**Output:**
```
EntityTrackerOutput {
  entities:        Entity[]
  lifecycle_events: EntityLifecycleEvent[]  # CREATED | CONFIRMED | OCCLUDED | RECOVERED | REMOVED
  statistics:      EntityStats
}
```

### Data Contracts

**Entity Lifecycle States:**

| State | Duration | Action |
|---|---|---|
| NEW | <3 frames | Track not confirmed yet |
| CONFIRMED | Active | Track established, updates every frame |
| OCCLUDED | ≥5 frames | Object hidden, extrapolate from trajectory |
| RECOVERED | After occlusion | Re-identified via appearance + spatial |
| LOST | ≥15 frames | No detections, search window active |
| REMOVED | After timeout | Profile archived to long-term memory |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Lifecycle evaluation | 0.05-0.1 ms | Per entity |
| ID assignment | 0.02-0.05 ms | Hash map lookup |
| State transition | 0.02-0.05 ms | |
| **Total per entity** | **0.1-0.2 ms** | |

---

## 5.3 Trajectory Model

### Purpose

Predict future positions of entities based on velocity, acceleration, and physics constraints. Provides extrapolated trajectories for occlusion handling and event prediction.

### Interfaces

**Input:**
```
TrajectoryInput {
  entity_id:       string
  position_history: [N][3]float              # last N positions
  velocity:        [3]float
  acceleration:    [3]float
  prediction_horizon_ms: float               # default 500ms
}
```

**Output:**
```
TrajectoryOutput {
  entity_id:       string
  predicted_positions: [M][3]float           # future positions
  prediction_variance: [M][3]float           # uncertainty per step
  trajectory_type: Enum                      # LINEAR | CURVED | STATIONARY | ERRATIC
  prediction_confidence: float
}
```

### Data Contracts

**Kalman Filter Configuration:**
```
KalmanConfig {
  process_noise:   [3]float                 # per-axis, default [0.1, 0.1, 0.1]
  measurement_noise: [3]float               # per-axis
  dt:              float                    # time step (1/30s for 30 FPS)
  max_prediction_steps: int                 # default 15 (0.5s at 30 FPS)
}
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Kalman predict | 0.05-0.1 ms | Per entity |
| Physics constraint projection | 0.03-0.05 ms | Joint limits, velocity bounds |
| Trajectory classification | 0.02-0.05 ms | |
| **Total** | **0.1-0.3 ms** | Per entity |

---

## 5.4 Event Detection

### Purpose

Detect meaningful events from entity states, trajectories, and relations. Computes an R (relevance) score that drives the scheduler's priority allocation.

### Interfaces

**Input:**
```
EventDetectionInput {
  world_state:     WorldStateSnapshot
  rules:           EventRule[]
  prediction_error: map<string, float>      # per-entity
  uncertainty:     map<string, float>       # per-entity
}
```

**Output:**
```
EventTrigger {
  event_id:        string
  event_type:      string                   # "goal" | "foul" | "scene_cut" | "anomaly"
  timestamp_ns:    uint64
  entity_ids:      string[]                 # involved entities
  r_score:         float                    # 0-1, drives scheduling priority
  evidence:        Evidence[]
  confidence:      float
}
```

### Data Contracts

**R Score Formulation:**
```
R = f(prediction_error, uncertainty, model_disagreement, event_importance, user_priority)

Where:
  prediction_error:  How much the current frame deviates from prediction
  uncertainty:       Current state uncertainty (from calibration)
  model_disagreement: Fast verifier vs deep reasoner disagreement
  event_importance:  Domain-specific importance (e.g., goal > pass)
  user_priority:     User-specified priority override
```

### Event Rule Schema:
```
EventRule {
  name:            string
  conditions:      Condition[]              # all must be true
  priority:        int                      # 1=highest
  compute_path:    Enum                     # FAST_ONLY | FAST_THEN_DEEP | DEEP_ONLY
  cooldown_ms:     int                      # minimum time between triggers
}

Condition {
  entity_class:    string                   # or "*" for any
  attribute:       string                   # "velocity" | "pose.left_knee" | etc.
  operator:        Enum                     # GT | LT | EQ | IN_RANGE | CHANGED
  value:           float
  window_ms:       int                      # time window for condition
}
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Rule evaluation | 0.05-0.1 ms | Per rule per entity |
| R score computation | 0.02-0.05 ms | |
| Cooldown check | 0.01-0.02 ms | Hash map lookup |
| **Total** | **0.1-0.3 ms** | Per frame |

### Dependencies

**Upstream:** World state, Entity tracker, Trajectory model
**Downstream:** Scheduler (priority), Hypothesis engine (triggers), Memory (event log)

---

# 6. Layer 04 — Memory

## 6.1 Short-Term Memory

### Purpose

Ring buffer of recent world state snapshots. Provides immediate access to the last N frames of state for temporal reasoning and feature extraction.

### Interfaces

**Input:**
```
ShortTermWrite {
  snapshot:        WorldStateSnapshot
  ttl_ms:          int                      # default 5000 (5 seconds)
}
```

**Output:**
```
ShortTermRead {
  query_window:    [2]uint64                # [start_ns, end_ns]
  snapshots:       WorldStateSnapshot[]
  count:           int
}
```

### Data Contracts

**Storage: Pre-allocated Ring Buffer**
- 30 slots (1 second at 30 FPS)
- Each slot: ~1-5 KB depending on entity count
- Flush to long-term memory every 5 seconds
- Promote to working memory on event trigger

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Write (append to ring) | <0.01 ms | Lock-free SPSC |
| Read (window query) | 0.05-0.1 ms | Linear scan, small N |
| Flush to long-term | 0.5-2 ms | Async, every 5s |
| **Total write** | **<0.1 ms** | |

---

## 6.2 Working Memory

### Purpose

Active hypothesis context. Holds the current top-K hypotheses being investigated, along with their evidence chains and pending verifications.

### Interfaces

**Input:**
```
WorkingMemoryWrite {
  hypothesis:      Hypothesis
  evidence:        Evidence[]
  priority:        float                    # R score
}
```

**Output:**
```
WorkingMemoryRead {
  active_hypotheses: Hypothesis[]
  context_summary: string                   # text summary for VLM prompt
  total_evidence:  int
}
```

### Data Contracts

**Working Memory Structure:**
- Capacity: Top 10 hypotheses by R score
- Per hypothesis: ~5-10 KB (hypothesis + evidence chain)
- Eviction: Lowest priority hypothesis when full
- Access: Reasoning engine reads, event detection writes

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Insert/update hypothesis | 0.1-0.5 ms | Sorted list maintenance |
| Context assembly for VLM | 1-3 ms | Text formatting |
| **Total** | **1-5 ms** | |

---

## 6.3 Long-Term Memory

### Purpose

Persistent entity profiles, historical trajectories, and learned patterns. SQLite for structured data, FAISS for vector similarity search.

### Interfaces

**Input:**
```
LongTermWrite {
  entity_id:       string
  profile:         EntityProfile
  trajectory_summary: TrajectorySummary
  event_summaries: EventSummary[]
}
```

**Output:**
```
LongTermRead {
  entity_id:       string
  profile:         EntityProfile
  similar_entities: EntityProfile[]          # FAISS similarity search
  historical_events: EventSummary[]
  avg_confidence:  float                    # historical average
}
```

### Data Contracts

**Entity Profile Schema:**
```
EntityProfile {
  entity_id:       string
  first_seen_ns:   uint64
  last_seen_ns:    uint64
  total_frames:    int
  avg_confidence:  float
  avg_velocity:    [3]float
  dominant_pose:   PoseResult               # most common pose
  appearance_embedding: [512]float           # average Re-ID embedding
  known_attributes: map<string, Any>
  event_history:   string[]                 # event_id references
}
```

**FAISS Index Configuration:**
```
FAISSConfig {
  dimension:       512                      # Re-ID embedding size
  index_type:      "IVF100,Flat"            # or "HNSW64" for speed
  nprobe:          10                       # search candidates
  max_gallery:     100000                   # entities
}
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| SQLite read/write | 5-20 ms | Structured data |
| FAISS similarity search | 5-15 ms | 10K gallery |
| Profile assembly | 2-5 ms | |
| **Total (read)** | **10-50 ms** | Async, non-blocking |

### Dependencies

**Upstream:** Short-term memory (periodic flush), Entity tracker (profiles)
**Downstream:** Deep VLM (historical context), Episodic memory (event summaries)

---

## 6.4 Episodic Memory

### Purpose

Event summaries, scene descriptions, and narrative context. Used for VLM prompt assembly and long-term reasoning about sequences of events.

### Interfaces

**Input:**
```
EpisodeWrite {
  episode_id:      string
  start_frame:     uint64
  end_frame:       uint64
  summary:         string                   # text description
  key_entities:    string[]
  key_events:      Event[]
  embedding:       [512]float               # episode embedding for search
}
```

**Output:**
```
EpisodeRead {
  episode_id:      string
  summary:         string
  key_events:      Event[]
  relevance_score: float                    # for query matching
}
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Episode creation | 50-200 ms | Async summarization |
| FAISS search | 5-30 ms | Similarity retrieval |
| **Total (read)** | **5-30 ms** | |

---

# 7. Layer 05 — Reasoning

## 7.1 Hypothesis Engine

### Purpose

Generate and rank hypotheses from world state, events, and memory context. Applies rule-based pattern matching, domain knowledge, and prediction comparison.

### Interfaces

**Input:**
```
HypothesisInput {
  world_state:     WorldStateSnapshot
  event:           EventTrigger
  working_memory:  WorkingMemoryRead
  domain_rules:    DomainRule[]
}
```

**Output:**
```
HypothesisSetUpdate {
  hypotheses:      Hypothesis[]
  top_hypothesis:  Hypothesis
  ranking:         HypothesisRanking[]
  computation_ms:  float
}

Hypothesis {
  id:              string
  claim:           string                   # natural language
  type:            Enum                     # OBSERVED | INFERRED | PREDICTED
  status:          Enum                     # CANDIDATE | SUPPORTED | REFUTED | INCONCLUSIVE
  evidence:        Evidence[]
  contradictions:  Evidence[]
  confidence:      ConfidenceDecomposition
  priority:        float                    # derived from R score
}
```

### Data Contracts

**Hypothesis Generation Pipeline:**
1. Pattern matching against domain rules
2. Entity relation analysis
3. Trajectory prediction comparison
4. Cross-modal evidence gathering
5. Ranking by combined confidence

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Rule evaluation | 2-5 ms | Per active rule |
| Pattern matching | 1-3 ms | |
| Evidence gathering | 2-5 ms | |
| Ranking | 0.5-1 ms | Sort by confidence |
| **Total** | **5-15 ms** | CPU-bound |

### Dependencies

**Upstream:** World state, Event detection, Working memory, Prediction model, Domain rules
**Downstream:** Fast verifier (candidate hypotheses), Evidence graph (links), Deep VLM (if inconclusive)

---

## 7.2 Fast Verifier

### Purpose

Quick rule-based verification of hypotheses. Checks trajectory consistency, pose validity, spatial constraints, and temporal coherence. Resolves most hypotheses without VLM.

### Interfaces

**Input:**
```
VerificationInput {
  hypothesis:      Hypothesis
  world_state:     WorldStateSnapshot
  trajectory:      TrajectoryOutput
  evidence_graph:  EvidenceGraph
}
```

**Output:**
```
VerificationResult {
  hypothesis_id:   string
  verdict:         Enum                     # SUPPORTED | REFUTED | INCONCLUSIVE
  confidence:      float
  evidence_used:   Evidence[]
  verification_ms: float
  needs_deep:      bool                     # true if inconclusive → send to VLM
}
```

### Data Contracts

**Verification Checks:**

| Check | Method | Latency |
|---|---|---|
| Trajectory consistency | Position predicted vs observed within σ | 0.1-0.3ms |
| Pose validity | Joint angles within biomechanical limits | 0.1-0.2ms |
| Spatial constraint | Entity proximity within expected range | 0.05-0.1ms |
| Temporal coherence | Action duration within normal range | 0.05-0.1ms |
| Physics consistency | Velocity/acceleration within bounds | 0.05-0.1ms |
| Cross-entity consistency | Interaction plausibility | 0.1-0.3ms |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| All verification checks | 0.5-2 ms | Parallel execution |
| Verdict aggregation | 0.1-0.3 ms | |
| **Total** | **1-5 ms** | |

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| All checks pass | Verdict = SUPPORTED | Publish claim |
| Any check fails | Verdict = REFUTED | Publish refutation |
| Partial evidence | Verdict = INCONCLUSIVE | Route to deep VLM |
| Verification timeout | >5ms | Default to INCONCLUSIVE |

---

## 7.3 Deep VLM Reasoner

### Purpose

Multimodal deep reasoning using Vision-Language Models. Consumes state snapshots, assembles prompts with context, and produces natural language reasoning with structured output. Only invoked when fast verification is inconclusive.

### Interfaces

**Input:**
```
VLMReasoningInput {
  hypothesis:      Hypothesis
  state_snapshots: WorldStateSnapshot[]     # key frames
  working_memory:  WorkingMemoryRead
  episode_context: EpisodeRead[]            # similar past episodes
  prompt_template: string
}
```

**Output:**
```
VLMReasoningOutput {
  hypothesis_id:   string
  verdict:         Enum                     # SUPPORTED | REFUTED | INCONCLUSIVE
  reasoning_text:  string                   # natural language explanation
  confidence:      ConfidenceDecomposition
  evidence_used:   Evidence[]
  inference_ms:    float
  model_id:        string                   # "Qwen3-VL-30B-A3B"
  tokens_generated: int
}
```

### Data Contracts

**Model Selection:**

| Model | Latency | Quality | VRAM | Notes |
|---|---|---|---|---|
| Qwen3-VL-30B-A3B FP8 | 800-1500ms | High | 20GB | Default on Tier 1 |
| Qwen2.5-VL-32B AWQ | 1000-2000ms | High | 20GB | Alternative |
| GPT-4.1 (API) | 880ms TTFT | Very High | N/A | API fallback |
| Gemini 2.5 Flash (API) | 420ms TTFT | High | N/A | Fastest API |
| Qwen 3.5-397B (Tier 3) | 2000-3500ms | Very High | 4×A100 | Production |

**VLM Output Schema (structured JSON):**
```json
{
  "verdict": "supported|refuted|inconclusive",
  "confidence": {
    "perception": 0.0-1.0,
    "temporal": 0.0-1.0,
    "reasoning": 0.0-1.0
  },
  "evidence_refs": ["ev_104", "ev_110"],
  "reasoning": "natural language explanation",
  "alternative_hypotheses": [
    {"claim": "...", "probability": 0.2}
  ]
}
```

### Prompt Assembly Strategy:
1. System prompt: role definition, output format constraints
2. State summary: current world state (compressed)
3. Hypothesis: claim to evaluate
4. Evidence: relevant observations and trajectories
5. Context: similar past episodes from episodic memory
6. Instructions: structured JSON output format

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Frame selection | 5-10 ms | Key frame extraction |
| Prompt assembly | 1-2 ms | Template filling |
| VLM inference | 800-3500 ms | Model dependent |
| Response parsing | 5-10 ms | JSON extraction |
| **Total** | **811-3532 ms** | Async, non-blocking |

### Dependencies

**Upstream:** Fast verifier (inconclusive hypotheses), Working memory, Long-term memory, Episodic memory
**Downstream:** Confidence decomposition, Evidence graph, Claim output

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| VLM OOM | CUDA OOM | Reduce context, retry |
| Malformed output | JSON parse error | Retry with stricter prompt |
| Timeout >5s | Timer | Return INCONCLUSIVE with partial reasoning |
| Queue depth = 4 (max) | Queue monitor | Drop oldest, submit newest (highest R) |
| Staleness >5s | Timestamp check | Tag as stale, discount confidence |

### Queue & Backpressure Rules (MANDATORY):
1. **VLM queue depth = 4 max.** If depth = 4 and new high-R event arrives, drop oldest pending snapshot.
2. **Coalescing rule:** Same entity/event window has ≥2 snapshots pending → keep newest with highest R score, drop others.
3. **Staleness tag:** Every VLM output arrives with `claim_staleness_ms = current_time - snapshot_time`. If >5000ms → auto-tagged `stale = true`, confidence discounted.
4. **Backpressure propagate:** VLM queue depth >3 for >30s → disable low-priority triggers.

---

## 7.4 Evidence Graph

### Purpose

Typed directed graph connecting observations, states, hypotheses, and evidence. Supports uncertainty propagation, causal chain discovery, and evidence strength computation.

### Interfaces

**Input:**
```
EvidenceGraphUpdate {
  new_nodes:       EvidenceNode[]
  new_edges:       EvidenceEdge[]
  claim_id:        string
}
```

**Output:**
```
EvidenceGraph {
  nodes:           EvidenceNode[]
  edges:           EvidenceEdge[]
  support_paths:   EvidencePath[]
  weakest_link:    EvidenceEdge
  total_llr:       float                    # total log-likelihood ratio
}

EvidenceNode {
  id:              string
  type:            Enum                     # PERCEPTION | TRACKING | ACTION | SPATIAL | TEMPORAL | CAUSAL | CLAIM_SUPPORT
  data:            map<string, Any>
  timestamp_ns:    uint64
  confidence:      float
}

EvidenceEdge {
  from:            string                   # node_id
  to:              string                   # node_id
  type:            Enum                     # TEMPORAL | SPATIAL | CAUSAL | SEMANTIC | TRACKING | EVIDENCE | NEGATION
  weight_llr:      float                    # log-likelihood ratio
  confidence:      float
  metadata:        map<string, Any>
}
```

### Node Types:

| Type | Data Fields | Source |
|---|---|---|
| PERCEPTION | frame_id, entity_id, bbox, pose, confidence | Detection/Pose |
| TRACKING | track_id, avg_velocity, direction | Tracking |
| ACTION | action_class, start/end frame, tempo | Pose + Rules |
| SPATIAL | region, zone, density | Spatial analysis |
| TEMPORAL | window_start/end, frequency, anomaly | Temporal analysis |
| CAUSAL | cause_id, effect_id, delay, probability | Causal inference |
| CLAIM_SUPPORT | claim_id, support_score, num_paths | Aggregation |

### Edge Types:

| Type | Fields | Semantics |
|---|---|---|
| TEMPORAL | lag_ms, confidence | A happened before B |
| SPATIAL | distance, overlap, direction | A is near/above/left of B |
| CAUSAL | probability, delay, evidence_count | A caused B |
| SEMANTIC | relation, weight | A is related to B |
| TRACKING | continuity, appearance_sim | A is same entity as B |
| EVIDENCE | weight_llr, supportive/refutive | A supports/refutes B |
| NEGATION | negation_type, confidence | A negates B |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Node insertion | 0.1-0.3 ms | Per node |
| Edge insertion | 0.1-0.2 ms | Per edge |
| Support path query | 1-5 ms | BFS/DFS traversal |
| Uncertainty propagation | 1-3 ms | Message passing |
| **Total (per update)** | **1-5 ms** | |

### Storage:
- SQLite: nodes, edges, metadata
- FAISS: node embeddings for similarity
- NetworkX: in-memory graph for queries
- Periodic snapshots every 100 frames

---

## 7.5 Prediction Model

### Purpose

Extrapolate entity trajectories and predict future states. Provides predictions that the hypothesis engine compares against observations.

### Interfaces

**Input:**
```
PredictionInput {
  entity_id:       string
  trajectory:      TrajectoryOutput
  context:         WorldStateSnapshot
  horizon_ms:      float                    # prediction horizon
}
```

**Output:**
```
PredictionOutput {
  entity_id:       string
  predicted_state: EntityState
  prediction_variance: float
  prediction_confidence: float
  contributing_factors: string[]
}
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Trajectory extrapolation | 0.05-0.1 ms | Kalman prediction |
| Context adjustment | 0.03-0.05 ms | Scene-aware correction |
| **Total** | **0.1-0.2 ms** | Per entity |

---

# 8. Layer 06 — Calibration

## 8.1 Confidence Decomposition

### Purpose

Decompose overall confidence into interpretable components: perception quality, temporal consistency, motion evidence, cross-modal agreement, and reasoning strength.

### Interfaces

**Input:**
```
ConfidenceInput {
  perception_confidence: float
  temporal_consistency: float
  motion_evidence: float
  cross_modal_agreement: float
  reasoning_confidence: float
  calibration_method: string
}
```

**Output:**
```
ConfidenceDecomposition {
  perception:      float                    # 0-1
  temporal:        float                    # 0-1
  motion:          float                    # 0-1
  cross_modal_agreement: float              # 0-1
  reasoning:       float                    # 0-1
  calibrated:      float                    # 0-1, final calibrated score
  calibration_method: string
  uncertainty_sources: string[]             # list of factors reducing confidence
  overall:         float                    # weighted combination
}
```

### Data Contracts

**Decomposition Formula:**
```
overall = w_p * perception + w_t * temporal + w_m * motion + w_cm * cross_modal + w_r * reasoning

Where weights are learned from calibration set:
  w_p = 0.25, w_t = 0.20, w_m = 0.15, w_cm = 0.15, w_r = 0.25 (defaults)
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Component aggregation | 0.1-0.3 ms | Weighted sum |
| Uncertainty source detection | 0.2-0.5 ms | Rule-based |
| **Total** | **<1 ms** | |

---

## 8.2 Conformal Prediction

### Purpose

Provide prediction sets with guaranteed coverage. Instead of a single point estimate, outputs a set of plausible hypotheses with formal coverage guarantees.

### Interfaces

**Input:**
```
ConformalInput {
  hypothesis_set:  Hypothesis[]
  calibration_data: CalibrationDataset      # held-out set
  alpha:           float                    # significance level (0.05 or 0.10)
}
```

**Output:**
```
ConformalOutput {
  prediction_set:  Hypothesis[]             # set with guaranteed coverage
  coverage_guarantee: float                 # 1 - alpha
  set_size:        int
  calibration_error: float                  # observed vs expected coverage
  alpha:           float
}
```

### Data Contracts

**Split Conformal Prediction:**
1. Split calibration data into proper train and calibration sets
2. Compute nonconformity scores on calibration set
3. Determine threshold for desired coverage level
4. At inference: include all hypotheses with score above threshold

**Coverage guarantees:**
- α = 0.05 → 95% coverage guaranteed
- α = 0.10 → 90% coverage guaranteed
- Set size varies: typically 1-3 hypotheses

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Score computation | 1-3 ms | Per hypothesis |
| Threshold application | 0.1-0.2 ms | |
| Set assembly | 0.5-1 ms | |
| **Total** | **2-5 ms** | |

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Set too large (>5) | set_size > 5 | Increase alpha, report uncertainty |
| Coverage below guarantee | calibration_error > 0.05 | Retrain calibration |
| Calibration data stale | timestamp > 24h | Trigger recalibration |

---

## 8.3 Temperature Scaling

### Purpose

Post-hoc calibration of neural network confidence scores. Learns a single temperature parameter on a validation set to align predicted probabilities with observed frequencies.

### Interfaces

**Input:**
```
TemperatureInput {
  logits:          float[]                  # raw model outputs
  temperature:     float                    # learned parameter
}
```

**Output:**
```
TemperatureOutput {
  calibrated_prob: float                    # temperature-scaled probability
  temperature:     float
  ece_before:      float                    # expected calibration error before
  ece_after:       float                    # expected calibration error after
}
```

### Data Contracts

**Temperature Scaling Formula:**
```
p_calibrated = softmax(logits / T)

Where T is learned by minimizing NLL on validation set:
  T* = argmin_T E[-sum y_i * log(softmax(z_i / T))]
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Temperature application | <0.1 ms | Single division + softmax |
| **Total** | **<0.1 ms** | |

---

## 8.4 Claim Output

### Purpose

Assemble final structured claim with all metadata: confidence decomposition, prediction set, evidence references, staleness tags, and epistemic status.

### Interfaces

**Input:**
```
ClaimAssemblyInput {
  hypothesis:      Hypothesis
  confidence:      ConfidenceDecomposition
  conformal:       ConformalOutput
  evidence_graph:  EvidenceGraph
  temperature:     TemperatureOutput
  staleness_ms:    float
}
```

**Output:**
```
Claim {
  id:              string
  schema_version:  string                   # "1.0.0"
  timestamp_ns:    uint64
  published_ts_ns: uint64
  claim_text:      string
  claim_type:      Enum                     # OBSERVED | INFERRED | PREDICTED
  epistemic_status: Enum                    # PHYSICAL | RULE_BASED | UNCERTAIN

  confidence:      ConfidenceDecomposition

  conformal_prediction_set_alpha_05: string[]
  conformal_prediction_set_alpha_10: string[]
  prediction_set_size_at_alpha_05: int

  evidence:        Evidence[]
  evidence_graph_snapshot: EvidenceGraph

  provenance:      Provenance
  output_quality:  OutputQuality
}

Provenance {
  producing_model: string
  producing_model_version: string
  checkpoint_sha256: string
  claim_staleness_ms: float
  stale:           bool
  source_stream:   string
  source_frames:   uint64[]
  event_id:        string
  hypothesis_id:   string
  domain:          string
  domain_metadata: map<string, string>
}

OutputQuality {
  schema_version:  string
  confidence_method: string
  calibration_dataset: string
  coverage_verified: bool
  p95_latency_ms:  float
  p99_latency_ms:  float
  gpu_utilization: float
}
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Claim assembly | 0.5-1 ms | Field mapping |
| Validation | 0.1-0.2 ms | Schema check |
| Serialization | 0.3-0.5 ms | JSON or Protobuf |
| **Total** | **1-2 ms** | |

---

# 9. Layer 07 — Scheduler

## 9.1 Multi-Rate Scheduler

### Purpose

Orchestrate processing rates for all components based on R scores. Ensures the critical path never blocks on lower-priority work.

### Interfaces

**Input:**
```
SchedulerInput {
  event:           EventTrigger
  current_queue_depths: map<string, int>
  gpu_utilization: float
  system_load:     float
}
```

**Output:**
```
SchedulerOutput {
  priority_stream: Enum                     # P0_CRITICAL | P1_HIGH | P2_NORMAL | P3_LOW
  target_rate_hz:  float
  should_skip:     bool
  should_coalesce: bool
  should_drop:     bool
}
```

### Data Contracts

**Priority Mapping:**

| R Score | Priority Stream | Rate | Behavior |
|---|---|---|---|
| ≥ 0.85 | P0_CRITICAL | 30 FPS | Never skip, never preempt |
| 0.5 - 0.85 | P1_HIGH | 10-15 Hz | May skip frames if overloaded |
| 0.2 - 0.5 | P2_NORMAL | 2-5 Hz | Periodic, coalesceable |
| < 0.2 | P3_LOW | 0.1-1 Hz | Background, droppable |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Priority computation | 0.1-0.2 ms | |
| Rate adjustment | 0.05-0.1 ms | |
| **Total** | **<0.5 ms** | Overhead per decision |

---

## 9.2 Queue Management

### Purpose

Bounded queues for all async workers. Enforces depth limits, coalescing policies, and FIFO ordering within priority levels.

### Data Contracts

**Queue Configuration:**

| Queue | Max Depth | Coalescing | Drop Policy |
|---|---|---|---|
| Frame input | 8 | No | Drop oldest |
| Fast verify | 32 | No | Reject new |
| Deep VLM | 4 | Yes (newest wins) | Drop oldest |
| Forensics | 8 | No | Drop oldest |
| Long-term write | 4 | No | Drop oldest |
| Hypothesis gen | 8 | Yes (same entity) | Drop oldest |

### Coalescing Rules:
1. Same entity + same event window → keep newest with highest R score
2. Same hypothesis + new evidence → merge evidence, keep hypothesis
3. VLM queue: if 2+ pending for same entity → drop older snapshots

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Enqueue | <0.01 ms | Lock-free SPSC |
| Dequeue | <0.01 ms | Lock-free SPSC |
| Coalesce check | 0.05-0.1 ms | Hash map lookup |
| **Total per operation** | **<0.1 ms** | |

---

## 9.3 Backpressure Controller

### Purpose

Detect system overload and trigger degradation. Prevents cascade failures by reducing compute load before resources are exhausted.

### Data Contracts

**Backpressure Triggers:**

| Condition | Threshold | Action |
|---|---|---|
| VLM queue depth > 3 | >30 seconds | Disable low-priority triggers |
| GPU utilization > 95% | >60 seconds | Downgrade detection to RF-DETR-N |
| Memory usage > 80% | >30 seconds | Flush long-term memory, reduce ring buffer |
| Frame drop rate > 5% | >10 seconds | Skip all async work |
| Latency p99 > 30ms | >5 seconds | Reduce detection resolution |

### Degradation Cascade:
1. Disable low-priority triggers (P3)
2. Disable normal triggers (P2) — user query + scene cut still force through
3. Downgrade detection model (RF-DETR-S → RF-DETR-N)
4. Reduce VLM context length
5. Emergency: flush all async queues, perception-only mode

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Threshold evaluation | 0.1-0.2 ms | Per metric |
| Action dispatch | 0.05-0.1 ms | |
| **Total** | **<0.2 ms** | |

---

## 9.4 GPU Work Distribution

### Purpose

Allocate GPU resources across concurrent workloads using CUDA streams, MPS, or MIG. Ensures critical path gets priority access to GPU.

### Data Contracts

**GPU Allocation Strategies:**

| Strategy | Isolation | Utilization | Use Case |
|---|---|---|---|
| CUDA Streams | None | High | Tier 1 (single GPU) |
| CUDA MPS | Soft (SM sharing) | High | Tier 1-2 (complementary workloads) |
| NVIDIA MIG | Hard (hardware) | Medium | Tier 3 (strict QoS) |

**CUDA Stream Priorities:**
| Stream | Priority | Workload |
|---|---|---|
| Stream 0 (highest) | 0 | Detection + Pose (critical path) |
| Stream 1 | 1 | Tracking + State update |
| Stream 2 | 2 | Fast verifier + Hypothesis |
| Stream 3 | 3 | VLM inference |
| Stream 4 (lowest) | 4 | Forensics + Memory |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Stream dispatch | <0.05 ms | CUDA API call |
| MPS context switch | <0.1 ms | If using MPS |
| **Total** | **<0.2 ms** | |

### Reality Check 2026

- UnifiedServe (MPS-based): 3.0x more requests or 1.5x tighter SLOs, 4.4x higher throughput.
- HeteroServe: two GPU pools (consumer for vision, producer for LLM), cross-type work stealing.
- FlashCodec: GOP-level parallel decoding, 2.8-9.1x speedup over CPU on 4 A100.

---

# 10. Layer 08 — Forensics

## 10.1 Deepfake Detection

### Purpose

Detect AI-generated or manipulated video content using an ensemble of specialized detectors. Event-triggered on scene cuts, new entities, or explicit requests.

### Interfaces

**Input:**
```
ForensicInput {
  media_id:        string
  frames:          Tensor[]                 # key frames to analyze
  trigger_reason:  string                   # "scene_cut" | "new_entity" | "user_request"
  c2pa_result:     ProvenanceOutput         # from C2PA check
}
```

**Output:**
```
ForensicResult {
  media_id:        string
  verdict:         Enum                     # AUTHENTIC | SYNTHETIC | INCONCLUSIVE
  confidence:      float
  detector_results: DetectorResult[]
  ensemble_agreement: float                 # 0-1, how much detectors agree
  analysis_ms:     float
}

DetectorResult {
  detector_name:   string
  verdict:         string
  confidence:      float
  latency_ms:      float
}
```

### Data Contracts

**Ensemble Models:**

| Model | Type | Latency | Accuracy | Notes |
|---|---|---|---|---|
| CLIP-based (SigLIP) | Semantic | 50-100ms | 85-90% | Semantic consistency |
| EVA-02 | Structural | 30-80ms | 80-85% | Frequency domain |
| SRM + BayarNet | Noise | 20-50ms | 75-80% | Statistical traces |
| EffNet-B4 | Texture | 40-80ms | 82-88% | Local texture |
| capsule network | Spatial | 50-100ms | 78-83% | Spatial artifacts |

**Fusion Strategy:**
- Logit averaging with learned weights
- Confidence threshold: 0.7 for consensus
- Disagreement → INCONCLUSIVE

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Frame selection | 5-10 ms | Key frame extraction |
| CLIP inference | 50-100 ms | |
| EVA-02 inference | 30-80 ms | |
| SRM+BayarNet | 20-50 ms | |
| EffNet-B4 | 40-80 ms | |
| Capsule network | 50-100 ms | |
| Ensemble fusion | 1-2 ms | Logit averaging |
| **Total** | **200-500 ms** | Parallel execution |

### Dependencies

**Upstream:** C2PA/SynthID (provenance check), Detection (key frames)
**Downstream:** Synthetic media domain, Claim output

### Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Detector OOM | CUDA error | Run detectors sequentially |
| All detectors disagree | consensus < 0.5 | INCONCLUSIVE verdict |
| High false positive rate | confidence < 0.6 | Flag as uncertain |

### Reality Check 2026

- AI-generated content doubled 2024→2025, 96% of deepfakes are AI-generated.
- Vision models 66-73% accuracy on deepfakes (limited training data).
- Audio deepfakes: 15-20% of fraud attempts use voice synthesis.
- Best ensemble: >95% accuracy on controlled benchmarks, 85-90% in-the-wild.

---

## 10.2 Provenance / C2PA

### Purpose

Verify content provenance using C2PA manifests, SynthID watermarks, and content credentials. Determines if content is from a trusted source.

### Interfaces

**Input:**
```
ProvenanceInput {
  media_id:        string
  media_bytes:     bytes                    # or reference to storage
  media_type:      string                   # "image" | "video" | "audio"
}
```

**Output:**
```
ProvenanceOutput {
  media_id:        string
  has_manifest:    bool
  manifest_valid:  bool
  signer_trusted:  bool
  signer_name:     string
  signature_valid: bool
  creation_date:   string
  synthid_detected: bool
  synthid_confidence: float
  c2pa_version:    string
  validation_ms:   float
}
```

### Data Contracts

**C2PA v2.4 Structure:**
```
C2PAManifest {
  version:         "2.4"
  signature:       COSE_Sign1
  certificate:     X.509
  assertions:      Assertion[]
}

Assertion Types:
  creation         - creator, when, where
  c2pa.hash        - Merkle tree root
  stds.iptc.location - GPS coordinates
  stds.exif        - camera settings
  c2pa.cloud-data  - SynthID watermark
  ai-inference     - model used, settings
```

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Manifest extraction | 5-20 ms | Parse container |
| Signature verification | 10-30 ms | COSE validation |
| Certificate chain check | 5-15 ms | X.509 validation |
| SynthID detection | 20-50 ms | Watermark extraction |
| Merkle hash verification | 10-30 ms | Video chunks |
| **Total** | **50-200 ms** | |

### Reality Check 2026

- C2PA v2.4 (April 2026): new BMFF hash boxes for live video streaming.
- Security: IACR analysis (ePrint 2026/804) found 7 serious problems in C2PA — validators accept known compromised certificates.
- SynthID (Jul 2026): survives H.264 re-encoding, screenshots, up to 20% crop. 50% crop breaks at ~250 iterations.
- No cross-provider watermark detection (Google SynthID ≠ OpenAI).
- C2PA adoption growing but no requirement for AI-generated content labeling.

---

## 10.3 Audio Forensics

### Purpose

Detect AI-generated speech, voice cloning, and audio manipulation. Complements visual deepfake detection for audio-heavy content.

### Interfaces

**Input:**
```
AudioForensicInput {
  media_id:        string
  audio:           Tensor[T] float32
  sample_rate:     int
}
```

**Output:**
```
AudioForensicResult {
  media_id:        string
  verdict:         Enum                     # AUTHENTIC | SYNTHETIC | MANIPULATED | INCONCLUSIVE
  confidence:      float
  detector_results: AudioDetectorResult[]
  analysis_ms:     float
}
```

### Data Contracts

| Model | Type | Latency | Accuracy |
|---|---|---|---|
| Teffic-Audio | Classification | 50-100ms | 85-90% |
| FlowFake | Flow-based | 100-200ms | 80-85% |
| Wav2Vec2 fine-tuned | Embedding | 30-60ms | 78-82% |
| MesoNet-Audio | Frequency | 20-40ms | 75-80% |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Audio preprocessing | 5-10 ms | Feature extraction |
| Detector inference | 30-200 ms | Model dependent |
| Ensemble fusion | 1-2 ms | |
| **Total** | **100-500 ms** | |

---

# 11. Layer 09 — Domains

## 11.1 Sports Reasoning

### Purpose

Domain-specific reasoning for sports analysis. Applies sport rules (soccer, basketball, etc.), player roles, field geometry, and temporal patterns.

### Interfaces

**Input:**
```
SportsReasoningInput {
  world_state:     WorldStateSnapshot
  sport:           string                   # "soccer" | "basketball" | "football"
  event:           EventTrigger
  rulebook:        SportRules
}
```

**Output:**
```
SportsReasoningOutput {
  event_classification: string              # "goal" | "foul" | "offside" | "pass"
  rule_reference: string                    # "FIFA Law 12"
  confidence:      float
  involved_players: string[]
  temporal_context: string                  # "buildup" | "climax" | "aftermath"
  replay_value:    float                    # 0-1, how interesting for replay
}
```

### Data Contracts

**Sport-Specific Rules:**

| Sport | Key Events | Data Source |
|---|---|---|
| Soccer | Goal, foul, offside, corner, throw-in | SoccerNet + custom rules |
| Basketball | Shot, foul, travel, 3-point, block | Play-by-play data |
| Football | Touchdown, interception, penalty, sack | Tracking + play structure |

### Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Rule evaluation | 2-5 ms | Per active rule |
| Player role assignment | 1-2 ms | Position-based |
| Temporal context | 1-2 ms | Sequence analysis |
| **Total (fast path)** | **10-30 ms** | |

### Dependencies

**Upstream:** World state, Event detection, Entity tracker
**Downstream:** Claim output (domain-specific claims)

### Reality Check 2026

- SoccerNet V2: action recognition benchmarks, ~85% accuracy on temporal localization.
- Sports rule reasoning requires both perception (did contact occur?) and judgment (was it a foul?).
- VLM can reason about sports but needs structured context (player positions, ball trajectory).

---

## 11.2 General Multimedia

### Purpose

Domain-agnostic fallback pipeline. Uses the base perception→reasoning pipeline without sport-specific rules. Handles news, surveillance, social media, and general video content.

### Interfaces

**Input/Output:** Same as base pipeline (World State → Hypothesis → Claim)

### Data Contracts

**Default Domain Settings:**
```
GeneralDomainConfig {
  use_sports_rules: false
  use_news_context: false
  min_entities_for_event: 2
  default_confidence_discount: 0.1          # general content is less certain
}
```

### Latency Budget

Same as base pipeline — no additional overhead.

---

## 11.3 Synthetic Media

### Purpose

Detection + provenance verification for AI-generated content. Combines deepfake detection, C2PA verification, and SynthID watermark detection.

### Interfaces

**Input/Output:** Forensic results feed into Claim output with synthetic media metadata.

### Data Contracts

**Synthetic Media Verdict Flow:**
1. C2PA check → trusted source? → AUTHENTIC
2. No C2PA → run deepfake ensemble → SYNTHETIC / INCONCLUSIVE
3. SynthID detected → cross-reference with provider database
4. Combine all evidence → final verdict

---

# 12. Layer 10 — Infrastructure

## 12.1 Data Schemas

### Purpose

Versioned message schemas for all inter-component communication. Protobuf for performance-critical paths, JSON for API/debugging.

### Schema Standards:

**Non-negotiable additions for every message:**
1. Monotonic nanosecond timestamp (`ts_ns`)
2. Producing model identifier (model_name + version + checkpoint SHA256)
3. Hardware context tag (`context_id`)
4. JSON schema version — semver

**Serialization Selection:**

| Format | Wire Size | Decode Speed | Use Case |
|---|---|---|---|
| FlatBuffers | 344 bytes | 0.08s/1M ops | Hot path (perception → state) |
| Protobuf | 228 bytes | 302s/1M ops | Kafka, storage, API |
| JSON | 1475 bytes | 583s/1M ops | Debugging, logs, config |

**NvSchema Message Types (NVIDIA compatible):**

| Message | Key Fields | Topic |
|---|---|---|
| FrameObservationBatch | version, id, timestamp, sensorId, objects | perception.objects |
| WorldStateSnapshot | entities, trajectories, relations, scene | state.world |
| EventTrigger | type, timestamp, objectIds, category | events.triggered |
| Claim | claim_text, confidence, evidence, provenance | claims.published |
| VisionLLM | queries, responses, embeddings | vlm.captions |

**C2PA Manifest (CBOR/JSON-LD):**
- Formats: CBOR, JSON, JSON-LD
- Containers: JUMBF (JPEG), BMFF (MP4)
- Signatures: COSE (CBOR), X.509 certificates
- Hashing: Merkle trees for video chunks

---

## 12.2 Hardware Topology

### Purpose

GPU/CPU/edge layout for different deployment tiers. Defines which components run where.

### Tier 1: Single GPU (1× RTX 4090, 24GB)

```
GPU 0:
  Detection (RF-DETR-S): 1.5GB
  Pose (DETRPose-S): 0.5GB
  Tracking (ByteTrack): 0.05GB
  VLM (Qwen3-VL-30B FP8): 20GB
  CUDA Context: 1GB
  Total: 23.05GB / 24GB

CPU:
  Ring Buffers: Pre-allocated
  Job Queues: Bounded
  REST API: FastAPI
  SQLite: Entity/Event/Claim DB
  FAISS: Vector Index
```

### Tier 2: Dual GPU (2× RTX 4090 or 1× A100)

```
GPU 0 (Perception):
  Detection (RF-DETR-M): 1.5GB
  Pose (DETRPose-M): 0.5GB
  Tracking: 0.05GB
  Segmentation: 1GB
  Total: ~3GB / 24GB

GPU 1 (Reasoning):
  VLM (Qwen3-VL-30B): 20GB
  Fast Verifier: 0.1GB
  Evidence Graph: 0.1GB
  Total: ~20GB / 24GB
```

### Tier 3: Server (4× A100/H100)

```
GPU 0: Perception (RF-DETR-L, DETRPose-L, ByteTrack, Segmentation)
GPU 1: Fast Reasoning (Hypothesis, Fast Verifier, Evidence Graph)
GPU 2: Deep VLM (Qwen 3.5-397B, TP=4 across GPUs 2-3)
GPU 3: Deep VLM (continued) + Forensics (Deepfake ensemble)
CPU Node: State management, Memory, Scheduler, API
Storage Node: Kafka, SQLite, FAISS
```

### Network:
- 10GbE: API/Streaming
- NVLink: GPU↔GPU 600+GB/s (Tier 3)
- PCIe: GPU↔CPU 64GB/s

---

## 12.3 Deployment

### Purpose

Docker/Kubernetes/Helm deployment architecture. Defines containers, services, scaling, and monitoring.

### Container Architecture:

```
multimodal-reasoner/
├── perception/          # Detection + Pose + Tracking
│   ├── Dockerfile
│   ├── requirements.txt
│   └── config.yaml
├── state/              # World state + Entity tracker + Events
├── reasoning/          # Hypothesis + Fast verify + Evidence graph
├── vlm/               # Deep VLM reasoning
├── calibration/        # Confidence + Conformal + Temperature
├── forensics/          # Deepfake + C2PA + Audio
├── memory/             # Short/Working/Long/Episodic
├── scheduler/          # Multi-rate + Queue + Backpressure + GPU
├── api/               # FastAPI gateway
└── helm/              # Kubernetes deployment charts
```

### Kubernetes Deployment:

**Tier 1 (single node):**
```
namespace: multimodal-reasoner
Pods:
  - perception (GPU: 1× RTX 4090, CPU: 8 cores, Memory: 16GB)
  - reasoning (GPU: shared, CPU: 8 cores, Memory: 32GB)
  - state-management (CPU: 4 cores, Memory: 8GB)
  - api-gateway (CPU: 2 cores, Memory: 4GB)
  - scheduler (CPU: 2 cores, Memory: 4GB)
```

**Tier 3 (multi-node):**
```
Node 1 (GPU): perception + fast-reasoning
Node 2 (GPU): deep-vlm + forensics
Node 3 (CPU): state + memory + scheduler
Node 4 (CPU): api + kafka + prometheus + grafana
```

### Monitoring:
- Prometheus: metrics collection (latency, throughput, queue depths, GPU utilization)
- Grafana: dashboards (per-component latency, error rates, resource usage)
- Jaeger: distributed tracing (request flow through all components)
- NVIDIA DCGM: GPU-specific metrics (SM utilization, memory, thermals)

---

# 13. Core Data Contracts

## 13.1 Observation (Real 2026)

```json
{
  "schema_version": "1.0.0",
  "id": "obs_001",
  "ts_ns": 12530000000,
  "context_id": "stream_17",
  "time_ms_offset": 12530,
  "modality": "vision",
  "source": {
    "stream_id": "cam_main",
    "frame_num": 376,
    "frame_ts_ns": 12530000000
  },
  "producing_model": {
    "name": "RF-DETR-S",
    "version": "1.8.2",
    "checkpoint_sha256": "a1b2c3..."
  },
  "producing_latency_ms": 3.4,
  "entity": "person_7",
  "entity_alternate_ids_topk": [{"id": "person_7b", "score": 0.23}],
  "region_norm": [0.31, 0.22, 0.18, 0.52],
  "keypoints_2d_norm": { "joints": [], "visibility_score": [] },
  "feature_ref": "sha256:abc123...",
  "detector_score_raw": 0.96,
  "input_quality_flags": {
    "blur_score": 0.08,
    "occluded_frac": 0.0,
    "compressed": false,
    "detector_in_distribution": 0.91
  }
}
```

## 13.2 State Estimate (Distributions, Not Points)

```json
{
  "schema_version": "1.0.0",
  "entity": "person_7",
  "ts_ns": 12530000000,
  "state_window_ms": [12520, 12530],
  "producing_model": {"name": "Kalman+SMPL_lite", "version": "0.3"},
  "pose_3d": {
    "joints_mean_meters": [],
    "per_joint_covariance": [[],[]],
    "per_joint_visibility_score": [0.97, 0.94, 0.31],
    "per_joint_estimation_method": ["direct_observation", "direct_observation", "temporal_extrapolation"]
  },
  "velocity_mps": { "mean": [], "covariance": [] },
  "acceleration_mps2": { "mean": [], "covariance": [] },
  "occluded_parts": ["left_hand"],
  "state_distribution_type": "gaussian_mixture_3",
  "state_distribution": [
    {"weight": 0.72, "joints_mean": [], "joints_cov": []},
    {"weight": 0.21, "joints_mean": [], "joints_cov": []},
    {"weight": 0.07, "joints_mean": [], "joints_cov": []}
  ],
  "uncertainty_summary": {
    "perception": 0.96,
    "temporal_consistency": 0.91,
    "occlusion_penalty_applied": true,
    "kalman_innovation_mahalanobis": 1.2
  },
  "physics_constraint_violated": false,
  "projected_to_feasible": true
}
```

## 13.3 Evidence (Log-Likelihood, Not Arbitrary 0-1)

```json
{
  "schema_version": "1.0.0",
  "id": "ev_104",
  "ts_ns": 12532000000,
  "source": ["obs_001", "state_81"],
  "source_modality": ["vision", "state_estimate"],
  "producing_model": [{"name":"RF-DETR-S"}, {"name":"HybrIK-lite"}],
  "relation": "supports",
  "target": "hyp_21",
  "weight_llr": 1.47,
  "weight_estimator": "calibration_set_logistic_estimate_v2",
  "evidence_spatial_region": {"entity": "ball_2", "bbox_norm":[]},
  "evidence_temporal_range_ms": [12400, 12540],
  "input_quality_summary": {
    "min_visibility": 0.4,
    "compressed": true
  }
}
```

## 13.4 Hypothesis (Prediction Sets, Conformal, Stale Aware)

```json
{
  "schema_version": "1.0.0",
  "id": "hyp_21",
  "claim": "player_A_kicked_ball",
  "type": "inferred",
  "epistemic_status": "physical_discontinuity_plus_rule",
  "status": "supported",
  "support": ["ev_104", "ev_110", "rule_soccer_14_1"],
  "contradictions": [],
  "hypothesis_set_ranking": [
    {"hyp_id": "hyp_21", "posterior": 0.72},
    {"hyp_id": "hyp_22", "posterior": 0.21},
    {"hyp_id": "hyp_23", "posterior": 0.07}
  ],
  "conformal_prediction_set_alpha_05": ["hyp_21", "hyp_22"],
  "conformal_prediction_set_alpha_10": ["hyp_21"],
  "confidence": {
    "perception": 0.96,
    "temporal": 0.91,
    "motion": 0.94,
    "cross_modal_agreement": 0.88,
    "reasoning": 0.84,
    "calibrated": 0.89,
    "calibration_method": "split_conformal_alpha_01_v3",
    "prediction_set_size_at_alpha_05": 2
  },
  "snapshot_ts_ns": 12500000000,
  "published_ts_ns": 14220000000,
  "claim_staleness_ms": 1720,
  "inconclusive_justification": null
}
```

---

# 14. Complete Data Flow

## 14.1 Critical Path (30 FPS)

```
Frame arrives
  → Decode (1-3ms)
  → Preprocess (1-2ms)
  → Track (0.2ms)
  → Detect (3.5ms)
  → Pose (2.4ms)
  → State Update (0.5ms)
  → Event Score (0.2ms)
  → Publish (0.3ms)
  → Snapshot Ready (total: 8-12ms)
```

## 14.2 Fast Reasoning Path (5-10 Hz)

```
Event R≥0.5
  → Hypothesis Generation (5-15ms)
  → Fast Verify (1-5ms)
  → SUPPORTED/REFUTED → Confidence (0.8-0.95) → Claim
  → INCONCLUSIVE → Deep VLM path
```

## 14.3 Deep Reasoning Path (0.3-0.5 Hz)

```
Inconclusive from fast verify
  → Select Frames (5-10ms)
  → Build Prompt (1-2ms)
  → VLM Inference (800-3500ms)
  → Parse Response (5-10ms)
  → Confidence (0.6-0.9) → Claim
```

## 14.4 Calibration Output (always)

```
Confidence Input (from fast or deep path)
  → Decompose (<1ms)
  → Conformal Prediction (2-5ms)
  → Temperature Scaling (<0.1ms)
  → Assemble Claim (1ms)
  → Validate (0.1ms)
  → Serialize (0.5ms)
  → Claim Published
```

## 14.5 Forensics Path (async)

```
Scene Cut / New Entity
  → C2PA Valid?
    → Valid + Trusted → AUTHENTIC (0.95)
    → No Manifest → Run Ensemble (200-2000ms)
      → Semantic (CLIP) + Structural (EVA-02) + Spectral (SRM)
      → Logit Fusion → Consensus?
        → Yes → SYNTHETIC (0.8-0.95)
        → No → INCONCLUSIVE (0.5)
  → → Claim Output
```

## 14.6 Memory Path (async)

```
State Snapshot
  → Short-Term Ring Buffer
  → flush (5s) → Long-Term SQLite
  → promote → Working Memory
  → episode → Episodic Store
  → FAISS Index (embeddings)
  → historical context → VLM
```

---

# 15. Hardware Reference

## 15.1 Latency Budget Summary

| Path | Latency | Frequency |
|---|---|---|
| Critical Path (perception) | 8-12ms | 30 FPS |
| Fast Reasoning | 6-20ms | 5-10 Hz |
| Deep Reasoning | 811-3532ms | 0.3-0.5 Hz |
| Calibration | 3-8ms | Always |
| **Total Fast Path** | **17-40ms** | 30 FPS |
| **Total Deep Path** | **814-3540ms** | Async |

## 15.2 Storage Estimates (1 hour of 30 FPS video)

| Store | Size |
|---|---|
| State Ring: 30 frames × 26 entities × 1KB | ~780 KB |
| Entity History: 26 × 30 × 0.5KB | ~390 KB |
| Event Log: 100 × 2KB | ~200 KB |
| Working Memory: 10 × 5KB | ~50 KB |
| Entity Profiles: 26 × 5KB | ~130 KB |
| Event Summaries: 100 × 3KB | ~300 KB |
| Claims: 50 × 2KB | ~100 KB |
| FAISS Index: 26 × 512 × 4 bytes | ~53 KB |
| **Total per hour** | **~2 MB (memory) + ~5 MB (persistent)** |

---

# 16. Deployment Architecture

## 16.1 Docker Compose (Development)

```yaml
version: '3.8'
services:
  perception:
    image: reasoning/perception:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./config:/app/config
    ports:
      - "8081:8080"

  state:
    image: reasoning/state:latest
    volumes:
      - ./data:/app/data
    ports:
      - "8082:8080"

  reasoning:
    image: reasoning/vlm:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8083:8080"

  api:
    image: reasoning/api:latest
    ports:
      - "8000:8000"
    depends_on:
      - perception
      - state
      - reasoning

  scheduler:
    image: reasoning/scheduler:latest
    ports:
      - "8084:8080"
```

## 16.2 Kubernetes (Production)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: perception
  namespace: multimodal-reasoner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: perception
  template:
    spec:
      containers:
      - name: perception
        image: reasoning/perception:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
          requests:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
      nodeSelector:
        accelerator: nvidia-rtx-4090
```

---

# 17. Reality Check 2026

## 17.1 What's Real

| Component | Status | Evidence |
|---|---|---|
| RF-DETR detection | Production-ready | NVIDIA benchmarks, third-party verification |
| DETRPose pose | Production-ready | Paper + benchmarks, beats YOLO-pose |
| ByteTrack tracking | Production-proven | Widely deployed in DeepStream |
| Qwen3-VL reasoning | Available (API + local) | vLLM support, FP8 quantization |
| Conformal prediction | Established theory | Split conformal, guaranteed coverage |
| C2PA provenance | Shipping (v2.4) | Adobe + partners, but security concerns found |
| SynthID watermarks | Shipping (Google) | Survives re-encoding, but no cross-provider |

## 17.2 What's Uncertain

| Component | Uncertainty | Risk |
|---|---|---|
| VLM reasoning quality | Varies by domain | May need domain-specific fine-tuning |
| Ensemble deepfake detection | 85-90% in-the-wild | High false positive rate possible |
| Cross-modal fusion | Limited benchmarks | May not improve over vision-only |
| Audio forensics | Early stage | Voice cloning improving rapidly |

## 17.3 What's Banned

1. **Batched VLM on every N frames** — defeats event-triggered compute, makes latency unpredictable.
2. **Single-process Python GIL-threaded everything** — use separate processes or CUDA streams.
3. **Unbounded queues** — memory explodes, 30 FPS path blocks on memory alloc.
4. **CPU↔GPU copy inside per-frame critical path** — use pinned memory + persistent GPU tensors.
5. **"Shared state" dict with locks** — use lock-free ring buffers (SPSC/MPMC).

## 17.4 Banned Architecture Patterns

1. **Batched VLM on every N frames** — defeats event-triggered compute, makes latency unpredictable.
2. **Single-process Python GIL-threaded everything** — use separate processes or CUDA streams. Critical path 30 FPS code should be C++/CUDA or compiled (torch.compile / TensorRT graph with fixed inputs).
3. **Unbounded queues** — they grow, memory explodes, and eventually the 30 FPS path blocks on memory alloc. Every queue has a depth limit + drop policy.
4. **CPU↔GPU copy inside the per-frame critical path** — use pinned memory + persistent GPU tensors allocated once at init.
5. **"Shared state" dict with locks** — use lock-free ring buffers (SPSC/MPMC) for cross-worker state snapshot publication. Locks = jitter.

---

## Appendix A: Production Reference Architectures (2026 Verified)

### NVIDIA VSS/RT-VLM (Shipping Product)
- FastAPI-based REST API on DeepStream-accelerated pipeline.
- Default chunking: 10-second segments, 80 frames per chunk at 448x448 resolution.
- Inference engine: vLLM backend. Output: Server-Sent Events (SSE) or Kafka.
- Per-stream latency (H100, FP8): 0.58s alerting (OSL=1), 1.23s captioning (OSL=100) at 1 stream.
- At 51 concurrent alerting streams: 3.6s avg, 4.95s p95.
- GPU utilization: 94.8% (alerting), 83.8% (captioning).

### NVIDIA Patent US20250292557A1 (VLM Scheduler for Vehicles)
- Prompt scheduler queues detection requests at 30 or 60 FPS from multiple pipelines.
- Safety-prioritized: ADAS (pedestrian, trajectory) > driver monitoring (drowsiness every 2s).
- Single shared VLM serves multiple concurrent detection pipelines.

### UnifiedServe (MPS-based, Dec 2025)
- Uses CUDA MPS to share GPU between vision encoder and LLM decode.
- Three async workers: vision_process, encode-prefill, decode.
- Decode stream has priority over encode; encoder uses leftover SM cycles.
- Result: 3.0x more requests or 1.5x tighter SLOs, 4.4x higher throughput.

### HeteroServe (Edge-Cloud, Mar 2026)
- Two GPU pools: consumer GPUs (RTX 4090) for vision, producer GPUs for LLM.
- Phase-aware Multimodal Scheduler coordinates pools.
- Cross-type work stealing: consumer GPUs steal LLM decode work when idle.

### Co-VStream (Edge-Cloud, Jun 2026)
- Edge: Dual-Condensed Perception Pipeline. Cloud: Multi-process async.
- End-to-end latency: 2.99s (vs 4.08s cloud-only). Subgraph retrieval: 0.16s.

### MOSS-Video-Preview (Two-Channel, Jun 2026)
- Perception and generation on separate non-blocking pathways.
- Cross-attention backbone with visual features through side channel.
- 5x faster TTFT, 2.7x higher decoding throughput.

### GOP-Level Parallel Decoding (FlashCodec, Dec 2025)
- Video partitioned into GOPs for parallel decoding across GPUs.
- Stall-free scheduling: dispatches next GOP immediately when NVDEC idle.
- 2.8-9.1x speedup over CPU decoding on 4 A100 GPUs.

---

## Appendix B: Cross-References

- `02_architecture/01_perception/` through `10_infrastructure/` — 45 individual component files with identical structure
- `02_architecture/diagrams/` — 60 Mermaid diagrams (system, DB, API, protobuf, data flow, state machines, deployment)
- `benchmark_report_2026.md` — Per-component benchmark data
- `09_sources/README.md` — All 139 sources across 18 categories
- `01_foundations/` — 18 research tracks for each component
- `08_implementation/` — Build plan, ablation plan, risk register
- `VLM_Inference_Research_2025-2026.md` — VLM inference metrics with Sep 2026 updates
