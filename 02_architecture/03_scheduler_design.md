# Scheduler design

## Principle
Use a multi-rate asynchronous system with bounded queues.

## Critical path
1. Decode
2. Tracking/state update
3. Lightweight event detection
4. Publish current state

## Asynchronous paths
- Specialist perception
- Hypothesis generation
- Deep VLM reasoning
- Synthetic-media forensics
- Long-term summarization

## Trigger score
Potential research formulation:

R = f(prediction_error, uncertainty, model_disagreement, event_importance, user_priority)

Higher R causes more compute to be allocated to the corresponding event/window.

## Failure behavior
If deep reasoning lags:
- never block the 30 FPS path
- coalesce stale reasoning jobs
- prioritize newest high-value state
- preserve previous hypothesis with explicit age/staleness

---

## REALITY CHECK 2026: PRODUCTION SCHEDULER DESIGNS

### NVIDIA RT-VLM Scheduler (verified production system):
- Chunk-based scheduling: 10-second video segments (80 frames at 448×448).
- vLLM backend with configurable output: OSL=1 (alerting single token) or OSL=100 (captioning).
- Bounded queue with backpressure: GPU utilization monitored at 94.8% (alerting) / 83.8% (captioning).
- Per-stream latency scaling: 0.58s (1 stream) → 3.6s (51 streams) for alerting.

### TCM-Serve Modality-Aware Scheduling (May 2026):
- Classifies requests: video = "trucks" (dominate GPU), image = "cars", text = "motorcycles".
- Reduces TTFT by 54% overall, 78.5% for latency-critical requests.
- Addresses head-of-line blocking from large video requests.

### ReaLB Real-Time Load Balancing (May 2026):
- Modality-aware scheduling with zero overhead.
- Up to 1.32× throughput improvement for multimodal MoE.
- Evaluated on 8× NVIDIA RTX 5090 (32GB each).

### Phase-Aware GPU Scheduling (HeteroServe, Mar 2026):
- Two GPU pools: consumer (vision) + producer (LLM).
- Cross-type work stealing: idle consumer GPUs steal LLM decode work.
- Embedding transfer: ~4.5 MB per image over PCIe.

### GPU Partitioning Options:
- **MPS (software sharing):** Better utilization when workloads complementary. No memory isolation.
- **MIG (hardware partitioning):** Up to 7 isolated instances on A100/H100. Deterministic QoS. Each instance has dedicated SMs, L2 cache, memory controllers.
- **CUDA Stream Priorities:** Higher priority streams scheduled first. Used in UnifiedServe: decode stream priority over encode.

### GOP-Level Parallel Decoding (FlashCodec, Dec 2025):
- Video partitioned into GOPs for parallel decoding across GPUs.
- Stall-free scheduling: dispatches next GOP immediately when NVDEC idle.
- 2.8-9.1× speedup over CPU decoding on 4 A100 GPUs.
