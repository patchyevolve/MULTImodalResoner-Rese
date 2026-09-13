# 30 FPS budget

30 FPS means one input frame arrives every:

33.33 ms

Do not spend the entire budget in one model. Reserve headroom for decoding, preprocessing, synchronization, memory movement and scheduling.

## Target architecture
- 30 FPS: tracker/state update
- 10–30 FPS: selected lightweight perception
- 10–15 Hz: event detection
- 5–10 Hz: fast reasoning
- 1–5 Hz: deep multimodal reasoning
- event-driven: expensive forensic/causal investigations

## Measure
- end-to-end frame latency
- per-stage latency
- p50/p95/p99 latency
- jitter
- GPU utilization
- VRAM
- CPU utilization
- memory bandwidth
- queue depth
- dropped/coalesced jobs
- answer staleness

## Important distinction
30 FPS *input/state continuity* is realistic as a systems target; 30 FPS *large-VLM full reasoning* is a materially different and much harder target.

---

## REALITY CHECK 2026: VERIFIED LATENCY BUDGETS

### Critical Path Budget (RTX 4090, from 01_reference_architecture.md):
1. Decode: 1–3 ms
2. Fast preprocess: 1–2 ms
3. Tracker propagate: 0.2–0.5 ms
4. Lightweight detect + pose: 5–14 ms total (RF-DETR-S 3.5ms + DETRPose-S 2.39ms A10)
5. State update: 0.3–1 ms
6. Event trigger scoring: 0.1–0.3 ms
7. Publish state snapshot: 0.2–0.5 ms

**Total: ~8–20 ms p50, ~15–28 ms p99.** Leaves 5–20 ms headroom per frame.

### NVIDIA RT-VLM Real-World Latencies (H100, FP8):
| Concurrency | Alerting (OSL=1) | Captioning (OSL=100) |
|---|---|---|
| 1 stream | 0.58s | 1.23s |
| 10 streams | 1.10s | 2.21s |
| 20 streams | 1.33s | 3.72s |
| 51 streams | 3.6s avg | — |

**Note:** These are per-chunk latencies (10s chunks), not per-frame. The 30 FPS path runs independently of VLM inference.

### FlashCodec Decoding Speedup (Dec 2025):
- GOP-level parallel decoding across GPUs: 2.8-9.1× speedup over CPU.
- Stall-free scheduling: next GOP dispatched immediately when NVDEC idle.
- Video decoding is no longer the bottleneck for multi-stream systems.

### Hardware Decoding Capability:
- RTX 4090: hardware decode H.264/H.265/AV1 at >100 streams.
- Jetson Thor: hardware decode + 2,070 FP4 TFLOPS for inference.
- Bottleneck is inference, not decoding.
