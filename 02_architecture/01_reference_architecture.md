# Reference architecture

## Data flow

```text
MULTIMODAL INPUT
  ├── video
  ├── image
  ├── audio
  └── metadata/provenance
          |
          v
FAST PERCEPTION BUS  ---------------------------- 30 FPS state path
  ├── detector
  ├── tracker
  ├── pose
  ├── segmentation
  ├── OCR
  ├── embeddings/re-ID
  ├── audio features
  └── camera motion
          |
          v
WORLD STATE
  ├── entities
  ├── attributes
  ├── trajectories
  ├── relations
  ├── scene
  ├── events
  └── uncertainty
          |
   +------+-----------------------+
   |      |          |            |
   v      v          v            v
memory  motion     evidence     prediction
   |      model      graph        model
   +------+----------+------------+
                       |
                       v
               HYPOTHESIS ENGINE
                       |
               +-------+-------+
               |               |
               v               v
         fast verifier     deep reasoner
               |               |
               +-------+-------+
                       v
               CALIBRATION LAYER
                       |
                       v
          CLAIM + EVIDENCE + CONFIDENCE
```

## Principle
The real-time loop should never wait on deep reasoning. Deep reasoning consumes state snapshots and writes back higher-level beliefs asynchronously.

---

## REALITY CHECK 2026: CONCRETE ARCHITECTURE

### HARDWARE REFERENCE TARGET (2026 realistic):
**Tier 1 (Minimum viable):** 1× RTX 4090 24GB.
- Hosts: RF-DETR-S/pose (~3 GB) + ByteTrack state (~0.1 GB) + Qwen 2.5-VL 32B AWQ (~20.4 GB peak).
- 30 FPS perception **+** 0.3–0.5 Hz deep reasoning. Tight but fits. Queue depth must be monitored.

**Tier 2 (Recommended):** 1× A100 80GB OR 2× RTX 4090 24GB.
- 1 GPU: all of the above **+** forensic ensemble + 0.5–1 Hz 32B VLM + headroom for p99.
- 2 GPU: Perception GPU + Reasoning GPU. 1 Hz VLM reasoning comfortable.

**Tier 3 (Server/Production):** 2+ A100/H100.
- Perception on one; reasoning + forensic on others. 1–2 Hz deep reasoning + redundancy.

### CRITICAL PATH EXECUTION (30 FPS, 33.3 ms frame, PRIORITY STREAM 0 — NO PREEMPTION):
1. **Decode** (1–3 ms): H.264/H.265 hardware decoder.
2. **Fast preprocess** (1–2 ms): Resize to detector input 512/640 px; pinned memory / zero copy.
3. **Tracker propagate** (0.2–0.5 ms): ByteTrack/NvTrack.
4. **Lightweight detect + pose** (5–14 ms total): RF-DETR-S (3.5 ms) or YOLO26-S (2.5 ms) + lightweight pose (DETRPose-S: 2.39 ms A10, or RF-DETR-KP: 9.8 ms T4).
5. **State update** (0.3–1 ms): Kalman filter per entity per joint; physics constraints projection (joint limits, velocity bounds).
6. **Event trigger scoring** (0.1–0.3 ms): Prediction error + uncertainty thresholds computed.
7. **Publish state snapshot** (0.2–0.5 ms): Ring buffer write + notify async workers.

Total critical path: **~8–20 ms p50, ~15–28 ms p99**.

Leaves ~5–20 ms headroom on every single frame for the following periodic tasks (time-sliced into the GPU's idle windows between critical path calls):
- High-accuracy detection (5–10 Hz)
- 3D pose IK (3–10 Hz)
- Optical flow / depth (3–5 Hz)
- Instance segmentation (optional, 3–10 Hz)

### ASYNCHRONOUS PATHS (LOWER PRIORITY GPU STREAMS + CPU THREADS):
All run on bounded queues with explicit max depth. Coalescing is MANDATORY.

| Worker | Queue Depth Max | Rate (typical) | Latency per Job | Cadence |
|---|---|---|---|---|
| Specialist perception (Re-ID, OCR, Face) | 16 | 2–10 Hz | 5–50 ms | Event-triggered + periodic |
| Hypothesis generation & ranking | 8 | 2–5 Hz | 10–100 ms (CPU) | On event trigger or periodic |
| **Deep VLM reasoning** | **4** | **0.3–0.5 Hz** | **800–3500 ms** | **Event-triggered (high R score) or oldest state snapshot if queue empty** |
| Synthetic-media forensics ensemble | 8 | 0.1–1 Hz | 200–2000 ms | Event-triggered or first-seen per video / scene cut |
| Long-term summarization / episodic memory write | 4 | 0.05–0.2 Hz | 500–5000 ms | Buffer-driven; write to vector DB |

**VLM LATENCY REFERENCE (2026 verified):**
- API: GPT-4.1 = 0.88s TTFT, 171 tok/s. Gemini 2.5 Flash = 0.42s TTFT, 213 tok/s. Gemini 2.5 Flash-Lite = 0.29s TTFT, 323 tok/s.
- Local: Qwen3-VL-30B-A3B FP8 = 90 tok/s (RTX 4090D). Qwen2.5-VL-32B AWQ = 312 tok/s (2×A100, vLLM 0.7.3).
- StreamingVLM: 8 FPS, <100ms/token (ICLR 2026).

### QUEUE & BACKPRESSURE RULES (MANDATORY):
1. **VLM queue depth = 4 max.** If depth = 4 and a new high-R event arrives, drop the OLDEST pending snapshot in queue, submit new. Never exceed 4: VLM latency 2–3 s × 4 = 8–12 s of pending work is already too stale.
2. **Coalescing rule:** Same entity/event window has ≥ 2 snapshots pending → keep newest with highest R score, drop others.
3. **Staleness tag:** Every VLM output arrives with `claim_staleness_ms = current_time - snapshot_time`. If >5000 ms it gets auto-tagged `stale = true` and confidence is discounted heuristically.
4. **Backpressure propagate:** VLM queue depth >3 for >30 seconds → disable low-priority triggers (user query + scene cut still force through; rule violation downgrades to heuristic-only). Eventually trigger perception model downgrade to YOLO26-N from RF-DETR-S if system-wide utilization >95% for >60 s.

### CORE DATA FLOW — TYPED OBJECTS (not blobs):
All messages use protobuf / strict JSON schemas (see 02_core_data_contracts.md), with versioning. No undocumented fields.

```
FrameObservationBatch → EntityTracker → WorldStateSnapshot [every 33 ms]
                         → EventTriggerScorer → PriorityScore R
                         → WorkerDispatch
WorkerDispatch (R ≥ 0.85 → VLM max priority; R≥0.5 → normal; else fast-rule-only)
  ├─> PerceptionAsyncWorker → PerceptionResult
  ├─> HypothesisEngine (rule + ranking) → HypothesisSetUpdate
  ├─> DeepVLMWorker → VLMReasoningResult
  ├─> ForensicWorker → AuthenticityAssessment
  └─> (all) → EvidenceGraphWriter → CalibrationLayer
CalibrationLayer → Claim (with decomposed conf + conformal set + staleness)
```

### ❌ ARCHITECTURAL CHOICES THAT ARE BANNED (2026 reality):
1. **Batched VLM on every N frames.** It defeats the purpose of event-triggered compute and makes latency unpredictable.
2. **Single-process Python GIL-threaded everything.** Use separate processes or CUDA streams. Critical path 30 FPS code should be C++/CUDA or compiled (torch.compile / TensorRT graph with fixed inputs).
3. **Unbounded queues.** They grow, memory explodes, and eventually the 30 FPS path blocks on memory alloc. Every queue has a depth limit + drop policy.
4. **CPU↔GPU copy inside the per-frame critical path.** Use pinned memory + persistent GPU tensors allocated once at init.
5. **"Shared state" dict with locks.** Use lock-free ring buffers (SPSC/MPMC) for cross-worker state snapshot publication. Locks = jitter.

### PRODUCTION REFERENCE ARCHITECTURES (2026 verified):

**NVIDIA VSS/RT-VLM (shipping product, not research):**
- FastAPI-based REST API on DeepStream-accelerated pipeline.
- Default chunking: 10-second segments, 80 frames per chunk at 448×448 resolution.
- Inference engine: vLLM backend. Output: Server-Sent Events (SSE) or Kafka.
- Per-stream latency (H100, FP8): 0.58s alerting (OSL=1), 1.23s captioning (OSL=100) at 1 stream.
- At 51 concurrent alerting streams: 3.6s avg, 4.95s p95. At 33 captioning streams: 4.78s avg, 6.44s p95.
- GPU utilization: 94.8% (alerting), 83.8% (captioning).
- Supported models: Cosmos Reason2 8B, Qwen3-VL-30B-A3B, Nemotron-3-Nano-Omni-30B.

**NVIDIA Patent US20250292557A1 (VLM scheduler for vehicles):**
- Prompt scheduler queues detection requests at 30 or 60 FPS from multiple pipelines.
- Safety-prioritized: ADAS (pedestrian, trajectory) > driver monitoring (drowsiness every 2s, sign evaluation every 5-10s).
- Single shared VLM serves multiple concurrent detection pipelines.

**UnifiedServe (MPS-based multi-worker, Dec 2025):**
- Uses CUDA MPS to share GPU between vision encoder and LLM decode.
- Three async workers: vision_process, encode-prefill, decode.
- Decode stream has priority over encode; encoder uses leftover SM cycles.
- Result: 3.0× more requests or 1.5× tighter SLOs, 4.4× higher throughput.

**HeteroServe (edge-cloud, Mar 2026):**
- Two GPU pools: consumer GPUs (RTX 4090) for vision, producer GPUs for LLM.
- Phase-aware Multimodal Scheduler coordinates pools.
- Cross-type work stealing: consumer GPUs steal LLM decode work when idle.

**Co-VStream (edge-cloud, Jun 2026):**
- Edge: Dual-Condensed Perception Pipeline. Cloud: Multi-process async.
- End-to-end latency: 2.99s (vs 4.08s cloud-only). Subgraph retrieval: 0.16s.

**MOSS-Video-Preview (two-channel, Jun 2026):**
- Perception and generation on separate non-blocking pathways.
- Cross-attention backbone with visual features through side channel.
- 5× faster TTFT, 2.7× higher decoding throughput.

### DELIVERABLE (2026 realistic):
Reference architecture with:
- Concrete per-stage latency budgets and measured p50/p95 profiles on RTX 4090.
- Explicit per-worker queue depths, coalescing policies, backpressure triggers.
- Typed message schemas end-to-end (NvSchema Protobuf as reference).
- Demo implementation on 1 GPU with sports video achieving sustained 30 FPS p99.
