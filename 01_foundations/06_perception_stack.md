# 6. Perception stack

## Objective
Build the fast evidence acquisition layer.

## Vision
- Detection
- Tracking
- Segmentation
- Pose
- Depth
- Optical flow
- Camera motion
- Face recognition/re-identification
- OCR
- Scene understanding

## Audio
- ASR
- Speaker identification/diarization
- Sound event detection
- Audio localization
- Audio-video synchronization

## Metadata
- Container/codec information
- Frame rate
- Editing history where available
- Provenance
- Cryptographic/signature information

## Research question
Which perception outputs should be generated continuously, periodically, or only on demand?

---

## REALITY CHECK 2026 (concrete cadence + real latencies)

### ✅ CONTINUOUS (every frame, 30 FPS — proven, fits in budget):
| Task | Typical Model | T4/4090 batch-1 latency | Notes |
|---|---|---|---|
| Tracking (propagation only) | ByteTrack, NvTrack | 0.1–0.5 ms | Propagates tracks; no detector needed |
| Object detection (lightweight) | RF-DETR-N/S / YOLO26-N/S | 1.7–3.5 ms | Cadence: 10–30 Hz; for 30Hz use smallest variants |
| Pose estimation (2D) | RF-DETR Keypoint / YOLO26-Pose | 2.7–9.8 ms | DETRPose-S: 2.39 ms A10 (67.0 AP); DETRPose-B: 3.55 ms (70.1 AP); DETRPose-X: 12.0 ms (73.3 AP); RF-DETR-KP-S: 3.4 ms T4 (67.9 AP); RF-DETR-KP-B: 9.8 ms (71.8 AP); RF-DETR-KP-X: 17.2 ms (74.3 AP) |
| Optical flow (lightweight) | RAFT-tiny, FlowFormer-small | 3–8 ms | Optional; can run at 15 Hz |
| Camera motion (rotation estimate) | Essential matrix + RANSAC | 0.2 ms | OpenCV-based, no GPU needed |

### ⚠️ PERIODIC (5–15 Hz, or re-detect every N frames — fits):
| Task | Typical Model | Latency | Cadence |
|---|---|---|---|
| Object detection (high-accuracy variant) | RF-DETR-L/XL / YOLO26-X | 6.8–17.2 ms | 5–10 Hz; re-detect tracker every 3–6 frames |
| Instance segmentation | RF-DETR-Seg / SAM 2.1 | RF-DETR-Seg-S: 3.4 ms; RF-DETR-Seg-L: 8.8 ms T4; SAM 2.1 Hiera-L: 25.3 ms A100 | RF-DETR-Seg at 5–10 Hz; SAM 2.1 on-demand |
| 3D pose (SMPL/GHUM estimation) | HiPART / HybrIK / RAM / 4DHumans | HiPART: 2.53 ms single frame (396 FPS); HiPART seq-243: 1.73 ms (577 FPS); HybrIK: ~10–20 ms; RAM: 96.9 ms (10.32 FPS) | HiPART at 30 Hz; RAM at 10 Hz; fuse with 2D tracking between calls |
| Re-identification embedding | OSNet / ViT backbone | 2–5 ms | On new track + every 2 s |
| Scene classification | CLIP base | 2–4 ms | 1–2 Hz on scene-change |
| Depth estimation | DPT-Lite / Depth Anything small | 5–10 ms | 3–5 Hz |

### 🎯 ON DEMAND (event-triggered, NOT continuous):
| Task | Trigger | Typical Latency |
|---|---|---|
| OCR | Text region detected, or new document object | 5–30 ms (PaddleOCR / Qwen-VL OCR) |
| Face recognition/re-ID | New face track appears, or explicit query | 10–50 ms |
| Sound event detection (non-continuous) | Audio anomaly trigger, or every 1 s | 10–100 ms |
| Speaker diarization | On speech segment, or batch every 3–5 s | 100–500 ms per clip |
| Metadata / provenance / C2PA inspection | New file loaded, or scene cut | 1–5 ms (parse only; cryptographic verify 5–50 ms) |
| High-accuracy segmentation (SAM 2.1/3) | User query + new object of interest | SAM 2.1 Hiera-L: 25.3 ms A100; SAM 3: 30 ms H200 |
| Full ASR | Every 500 ms–2 s clip buffered | 50–300 ms (Whisper small-medium) |

### ❌ DO NOT RUN AT 30 FPS EVER:
- SAM 2 segmentation on all entities.
- Whisper-large ASR per frame (or even per 100 ms).
- Any 32B+ VLM.
- Face recognition on every single frame on 100+ entities.
- Audio-video sync verification per-frame (run every 0.5–2 s).

### TOTAL BUDGET CHECK:
Continuous critical path = tracking (~0.5 ms) + lightweight detect (~2.5 ms) + 2D pose (~7 ms) + pre/post-processing (~3–5 ms) = **~13–15 ms per frame at 30 Hz**.
Fits easily in 33.3 ms budget. Leaves 18–20 ms headroom for periodic tasks and async dispatch. This is achievable on a single T4-class GPU.
