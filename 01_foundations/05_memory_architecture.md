# 5. Memory architecture

## Objective
Design bounded memory that preserves useful evidence without feeding an ever-growing video history into a VLM.

## Memory tiers
- Frame/feature buffer: milliseconds to seconds
- Motion/event memory: seconds to tens of seconds
- Episodic memory: current event/play
- Semantic memory: minutes to hours

## Research questions
- Which information should be retained verbatim?
- Which can be compressed into tokens, embeddings, graphs or summaries?
- How should memory uncertainty decay?
- How should relevant historical evidence be retrieved?
- Can temporal token compression preserve reasoning accuracy?

## Deliverable
A hierarchical memory design with latency and retention policies.

---

## REALITY CHECK 2026

### ✅ PROVEN memory tiers (implement today):

| Tier | Timescale | Content | Retention Policy | Reality Check |
|---|---|---|---|---|
| **Frame/feature buffer** | ms → 2 s | Raw frames, precomputed detector features, optical flow | FIFO ring buffer, ~1–4 GB VRAM typical | This is standard in DeepStream/nvtracker; cost is known. |
| **Motion/event memory** | 2 s → 30 s | Entity trajectories, per-joint pose history, velocity/acceleration buffers, event log (last N events) | LRU bounded by entity count; top-20–100 entities retained | Compute ~0; storage tiny. |
| **Episodic memory** | 30 s → 30 min | Structured event graph, entity relation snapshots, key-frame thumbnails with embeddings | RAG-style vector DB; top-K retrieved by embedding similarity per query | FAISS/ScaNN/Qdrant are production-grade; no research issue. |
| **Semantic memory** | 30 min → hours | Abstracted fact table, per-entity persisted attributes (e.g., "player_7 = jersey #10, team red"), claim history with confidence | Write-on-change only; persist to DB | Plain old SQL or document DB. No innovation needed. |

### ⚠️ PLAUSIBLE (engineering, not science):
- **Temporal token compression for VLM context:** StreamingTOM (CVPR 2026): **15.7× KV-cache compression**, 2× TTFT speedup. STC (CVPR 2026): **24.5% ViT + 45.3% LLM latency reduction**, 99% accuracy. FastVLM (CVPR 2025): **75% visual tokens compressed** (576→144). KVCapsule (May 2026): **2× TPS**, 2.4× KV reduction. HYBRIDKV (ACL 2026): **7.9× KV reduction**, 1.52× speedup. These work; integrate as preprocessing BEFORE the VLM.
- **Video memory architectures:** FlexMem (CVPR 2026): training-free, processes **>1,000 frames** on single 3090 via dual-pathway compression. VideoMem (2026): adaptive memory buffer with GRPO training. ReWind: long-term memory bank with 32 read queries for 10+ min videos.
- **Memory uncertainty decay:** Heuristic; uncertainty grows linearly or sqrt(t) with time since last observation. Tune slope on calibration set.

### ❌ SET ASIDE (no value for this system):
- End-to-end-learned "neural memory" layers outside the VLM itself (e.g., memory-augmented transformers): They don't outperform simple RAG + typed state for this kind of structured perception, and they make calibration nearly impossible.
- Retaining frames verbatim beyond ~2–5 seconds: Wasteful. Store features/embeddings/thumbnails; retrieve source from disk if needed later via timestamp.

### HARD NUMBERS for budget planning on a single-GPU system (RTX 4090 or A100):
- Feature buffer: 60 frames × 256 features × entity count → ~1 GB VRAM acceptable.
- Trajectory buffers: 100 entities × 300 frames × 20 joints × 6 floats = ~3 MB. Negligible.
- Episodic semantic DB: Can live on CPU RAM/disk; vector DB query adds ~10–50 ms. Acceptable for async path.
- DO NOT attempt to feed "all of history" into the VLM. Token cost is prohibitive. Always compress/summarize/retrieve top-K.
