# 15. Real-time inference

## Objective
Make the system operational at a 30 FPS media input rate.

## Constraint
33.3 ms per input frame is the nominal budget at 30 FPS.

## Research questions
- What must be on the critical path?
- Which models can be skipped or cascaded?
- How should CPU/GPU/NPU workloads be partitioned?
- What is batch-1 latency for each component?
- How much memory bandwidth is consumed?

## Optimize
- FP16/BF16/INT8/INT4
- TensorRT/CUDA or equivalent accelerators
- Distillation
- Kernel fusion
- Token pruning/compression
- Frame skipping with tracking

## Deliverable
Latency budget and runtime architecture.

---

## REALITY CHECK 2026 (CONCRETE LATENCY BUDGET FOR 30 FPS)

30 FPS = **33.33 ms** per frame nominal budget. DO NOT spend the entire budget in one model. Reserve 20–30% headroom for decode, preprocessing, synchronization, memory movement, scheduling, and OS jitter.

### ✅ PROVEN BUDGET BREAKDOWN, single T4 / RTX 4090 GPU:
| Stage | Component | Allocated Budget | Measurable Reality 2026 |
|---|---|---|---|
| 0 | Video decode (H.264/H.265) | 1–3 ms | Hardware decoder; 1080p30 is ~1 ms. |
| 1 | Pre-processing (resize, normalize, tensor transfer) | 1–3 ms | Use pinned memory; avoid CPU↔GPU copies mid-pipeline. |
| 2 | Tracking propagation (ByteTrack/NvTrack) | 0.2–0.5 ms | No detector; just Kalman + matching. |
| 3 | Lightweight detection (RF-DETR-S or YOLO26-S) | 2–4 ms | RF-DETR-S: 3.5 ms T4; YOLO26-S: 2.5 ms T4. Run at 30 Hz. |
| 4 | 2D pose (lightweight) | 2.5–9.8 ms | DETRPose-S: 2.39 ms A10; RF-DETR-KP: 9.8 ms T4 (71.8 AP) |
| 5 | State update + motion model + physics constraints | 0.3–1 ms | Pure math; negligible. |
| 6 | Lightweight event/anomaly trigger (prediction error) | 0.1–0.3 ms | Threshold checks. |
| 7 | Publish state snapshot + queue dispatch | 0.2–0.5 ms | IPC/ring buffer. |
| **Headroom reserved** | For OS jitter, periodic tasks, p99 latency | **8–15 ms** | Non-negotiable; if headroom < 5ms you'll miss deadlines under load. |
| **TOTAL CRITICAL PATH** | | **~8–20 ms p50** | Fits easily in 33.3 ms. |

### ⚠️ PERIODIC / ASYNC TASKS (NOT ON CRITICAL PATH):
Scheduled to run in the headroom, on alternate frames, or on separate workers:
| Task | Latency per call | Recommended Cadence |
|---|---|---|
| High-accuracy detection (RF-DETR-L/XL) | 7–17 ms | 5–10 Hz; re-detect tracker every 3–6 frames |
| 3D pose (SMPL via HybrIK) | 10–25 ms | 3–10 Hz |
| Optical flow | 5–10 ms | 5–15 Hz |
| Instance segmentation | 5–9 ms | 3–10 Hz if needed |
| Depth estimate | 5–10 ms | 3–5 Hz |
| Re-ID embedding | 2–5 ms | On new track + every 2 s |
| Event classifier (action) | 10–30 ms | 3–10 Hz |
| **VLM deep reasoning** | **800–3500 ms** | **0.3–0.5 Hz (every 2–3 s) or event-triggered ONLY** |
| Forensic / synthetic media ensemble | 200–2000 ms | Event-driven or 0.1–0.5 Hz |

### ✅ PROVEN OPTIMIZATION TECHNIQUES (production-grade):
- FP16 / BF16: Default. Accuracy impact negligible.
- INT8 / INT4 quantization: Via TensorRT / AWQ / GPTQ. Perception models <1% mAP loss; VLMs ~1–3% quality loss but 2–3× speedup. Use.
- TensorRT / CUDA graphs: Mandatory for batch-1 latency. Cuts kernel launch overhead by 30–70%.
- Model cascades (cheap → expensive, early exit): Detect at 10 Hz; only run high-accuracy if tracker confidence drops.
- **Detector skipping + tracker propagation:** Single biggest optimization. Run detector every 3rd frame; tracker runs 30 FPS. 3× less detector compute for ~1–2% mAP loss.
- Feature + token caching: Re-use ViT features across frames for 10–30 Hz perception; temporal token compression (StreamingTOM/STC) gives 2–4× VLM context reduction.
- Persistent GPU memory / pinned host memory: Avoid allocation overhead per frame.

### ❌ FORGET ABOUT:
- Running any VLM (even 7B) on the 30 FPS critical path. Minimum 7B VLM image-encode + 100 tok output = ~800 ms optimistic. That's ~24 frames of 30 FPS.
- Running high-accuracy segmentation on every entity on every frame.
- Running Whisper large every 100 ms.
- CPU-only full perception stack. Raspberry Pi 5 CPU-only = multi-second per frame latency for large models.

### GPU WORKLOAD PARTITIONING (2026 realistic):
- **30 FPS fast path:** Dedicated GPU stream (highest priority). No preemption allowed.
- **Periodic perception:** Same GPU, lower priority stream; TensorRT-optimized. Time-sliced in gaps between 30 FPS frames.
- **VLM deep reasoning:** Can be SAME GPU if VRAM budget allows (RTX 4090 = 24 GB fits Qwen 2.5-VL 32B AWQ + perception models). Or separate second GPU for >32B VLMs.
- **CPU:** Decode, pre/post-processing not on critical path, async scheduler, vector DB, forensic CPU-only models.
- **MEASURE p50 / p95 / p99.** p99 latency is what kills 30 FPS continuity; p50 is easy.
