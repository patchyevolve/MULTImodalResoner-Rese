# Model selection matrix

Do not select models solely by benchmark score.

| Component | Primary metric | Secondary metric | Runtime constraint |
|---|---|---|---|
| Detector | mAP/recall | small-object robustness | batch-1 latency |
| Tracker | HOTA/IDF1 | occlusion recovery | per-frame latency |
| Pose | PCK/MPJPE | occlusion robustness | per-frame latency |
| Segmentation | IoU/J&F | temporal stability | memory |
| OCR | CER/WER | blur robustness | event-triggered |
| Audio | F1/WER | sync accuracy | stream latency |
| Video VLM | temporal QA | grounding | asynchronous |
| World model | state error | long-horizon stability | inference cost |
| Hypothesis model | calibrated ranking | contradiction detection | event-triggered |
| Calibration | ECE/Brier | risk-coverage | low overhead |

## Selection rule
A model is acceptable only if its contribution improves system-level accuracy or uncertainty enough to justify its compute and memory cost.

---

## REALITY CHECK 2026: VERIFIED MODEL SELECTION DATA

### DETECTOR SELECTION (from benchmark_report_2026.md):

| Latency Tier | Best mAP50:95 | Model | Latency T4 |
|---|---|---|---|
| Ultra-fast (<2ms) | 48.4 | RF-DETR-N | 2.3ms |
| Fast (2-4ms) | 54.7 | RF-DETR-M | 4.4ms |
| Balanced (4-7ms) | 56.5 | RF-DETR-L | 6.8ms |
| Accuracy (7-12ms) | 58.6 | RF-DETR-XL | 11.5ms |
| Peak (17ms) | 60.1 | RF-DETR-2XL | 17.2ms |

**Decision:** RF-DETR-S or RF-DETR-M for 30 FPS critical path (3.5-4.4ms). RF-DETR-L for async high-accuracy (5-10 Hz).

### POSE SELECTION (from benchmark_report_2026.md):

| Model | AP50:95 | Latency | Params | Notes |
|---|---|---|---|---|
| DETRPose-S | 67.0 | 2.39ms (A10) | 11.9M | Matches YOLOv8-X accuracy at 81% fewer params |
| DETRPose-L | 72.5 | 5.08ms (A10) | 36.8M | Best accuracy/latency tradeoff |
| RF-DETR-KP | 71.8 | 9.7ms (T4) | 40.7M | Preview, single resolution |
| YOLO26-pose X | 71.0 | 9.8ms (T4) | 57.6M | Largest, comparable to RF-DETR-KP |

**Decision:** DETRPose-S for 30 FPS critical path (2.39ms). DETRPose-L for async (5 Hz).

### VLM SELECTION (from VLM_Inference_Research and agent findings):

| Model | Size | Latency | Throughput | VRAM | Use Case |
|---|---|---|---|---|---|
| Qwen3-VL-30B-A3B | 30B (3B active) | — | 90 tok/s | ~20GB FP8 | Local reasoning (RTX 4090) |
| Qwen3.5-397B-A17B | 397B (17B active) | — | — | Multi-GPU | Deep reasoning (2+ A100) |
| Gemini 3.5 Flash | API | ~0.4s TTFT | ~213 tok/s | Cloud | Fast API reasoning |
| GPT-5.4 | API | — | — | Cloud | Complex reasoning |
| StreamingVLM | 7B | <100ms/tok | 8 FPS | H100 | Real-time video |

**Decision:** Qwen3-VL-30B-A3B for local 0.3-0.5 Hz reasoning. StreamingVLM for real-time video. API models for deep investigation.

### TOKEN COMPRESSION (for VLM inference):

| Method | Compression | Speedup | Accuracy | Training |
|---|---|---|---|---|
| StreamingTOM | 15.7× KV cache | 2× TTFT | 63.8% VideoMME | None |
| HybridKV | 7.9× memory | 1.52× decode | ~100% (7B) | None |
| KVCapsule | 2.4× memory | 2× TPS | Negligible loss | Light |
| PruneSID | 88.9% tokens | 6× FLOPs | 96.3% (1.5-7B) | None |

**Decision:** StreamingTOM for KV cache compression (training-free). HybridKV for decode speedup.

### HARDWARE SELECTION:

| Hardware | VLM Support | Perception | Cost |
|---|---|---|---|
| RTX 4090 (24GB) | Qwen3-VL-30B-A3B FP8 (tight) | RF-DETR + DETRPose | ~$1,600 |
| RTX 5090 (32GB) | Qwen3-VL-30B-A3B FP8 (comfortable) | RF-DETR + DETRPose | ~$2,000 |
| A100 80GB | Qwen3.5-122B or 32B full | Full pipeline | ~$10K |
| H100 80GB | Qwen3.5-397B (TP=4) | Full pipeline + streaming | ~$25K |
| Jetson Thor | 20B at 52 tok/s, 35B at 35 tok/s | Lightweight only | ~$2K |
