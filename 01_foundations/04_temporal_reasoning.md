# 4. Temporal reasoning

## Objective
Reason over ordered sequences rather than isolated frames.

## Research questions
- What temporal context is actually necessary for different event types?
- How should short-term, medium-term and long-term memory interact?
- How can the model identify event boundaries online?
- How should before/after evidence constrain hidden events?
- How should temporal hallucinations be detected?

## Focus
- Online action detection
- Temporal localization
- Action anticipation
- Sequence consistency
- Event segmentation
- Long-video reasoning
- Temporal causal chains

## Deliverable
A streaming temporal state machine and benchmark suite.

---

## REALITY CHECK 2026

### ✅ PROVEN (production-grade):
- **Event segmentation / boundary detection online:** Action detection Transformers (e.g., TimeSformer derivatives, VideoMAE) at 5–10 Hz are doable; per-frame temporal scoring works.
- **Long-video reasoning via summaries + retrieval:** Standard RAG architecture; embeddings per clip + vector DB. Works for hours-long video at low cost.
- **Sequence consistency via temporal smoothing:** Simple to implement; fixes 80% of per-frame flicker artifacts.

### ⚠️ PLAUSIBLE, design carefully:
- **Event boundaries online:** Works for coarse events (~1 s granularity). Fine-grained (<100 ms) event timing from single viewpoint video remains noisy; expect ±50–200 ms jitter on event boundaries and design the state machine accordingly.
- **Short/medium/long-term memory interaction:** This is systems engineering, not algorithmic research. See track 5 (memory architecture) for realistic tier design.
- **Before/after evidence constraining hidden events:** This works IF you have a state model and can afford the lookahead (i.e., do it in the async path, not the 30 FPS path). The 30 FPS path has ~0 ms lookahead (streaming); before/after constraints by definition require a window.
- **Temporal hallucination detection:** Check temporal consistency against tracking state. VLMs hallucinate temporal order about 15–30% of the time on complex multi-event clips; use tracker state as ground-of-truth to catch ordering errors.

### ❌ SET ASIDE:
- **Fine-grained (<50 ms) temporal event boundary detection from single video, zero-shot:** In practice, frame rate + motion blur + viewpoint limit temporal precision.

### RECOMMENDED design for this system:
- **30 FPS path (no lookahead):** Track state; fire coarse event triggers (scene cut, new object, 3σ velocity change).
- **1 Hz+1 s window async path:** Run event boundary refinement + temporal consistency checks with lookahead.
- **Benchmark suite MUST specify evaluation tolerance on event timing (±100 ms, ±200 ms, etc.).** Evaluating "did it detect the event?" without timing tolerance is uninformative for streaming systems.
