# 16. Multi-rate scheduling

## Objective
Allocate computation according to information value rather than processing every operation at every frame.

## Candidate rates
- 30 Hz: tracking/state update
- 10–15 Hz: event/anomaly detection
- 5–10 Hz: fast reasoning
- 1–5 Hz: deep VLM reasoning
- Event-triggered: expensive forensic/causal analysis

## Triggers
- prediction error
- model disagreement
- high uncertainty
- new object/entity
- scene cut
- rapid acceleration
- unusual pose
- possible rule violation
- user query

## Deliverable
Scheduler, queues, priorities, backpressure policy and compute budget controller.

---

## REALITY CHECK 2026

### ✅ PROVEN / STANDARD SYSTEMS ENGINEERING (not research):
- Multi-rate async scheduling with bounded queues: nvtracker, DeepStream, GStreamer all do this in production. No novelty here; it's engineering.
- Bounded queues + coalescing + drop policy: Standard.
- Priority tiers: Standard thread-pool pattern.

### ⚠️ RESEARCH POTENTIAL (what's new for this system):
- **Scheduler trigger score driven by quantified uncertainty + prediction error:** Most schedulers use fixed rates; this one is adaptive. The trigger formula R = f(prediction_error, uncertainty, model_disagreement, event_importance, user_priority) is where the design work is. Tune weights on a calibration set for Pareto optimal accuracy/compute.
- **Event-based triggers:** The following triggers are MEASURABLE in 30 FPS path; implement them:
  - Prediction error (state residual vs motion model): ≥ 2σ → trigger.
  - High state uncertainty (covariance large): threshold trigger.
  - Model disagreement (RF-DETR vs YOLO differ by class): trigger.
  - New entity appearance: trigger immediately (always run deep reasoning on first appearance).
  - Scene cut / rapid camera motion: trigger.
  - Rapid acceleration (≥5g human, or ball 20g): trigger.
  - Physics constraint violation (impossible pose detected): trigger.
  - Possible rule violation (domain-specific): trigger.
  - Explicit user query: always max priority.

### REALISTIC RATES (based on 2026 hardware, see track 15 for latencies):
| Component | Cadence | Notes (why this rate) |
|---|---|---|
| Tracking + state update | 30 Hz (every frame) | Mandatory; <1 ms. |
| Lightweight event trigger checks | 30 Hz | Threshold math; negligible. |
| Lightweight detection (RF-DETR-N/S) | **10–30 Hz** | ~2–4 ms; can run every frame, but typically run every 2–3 frames to save + tracker propagates. |
| High-accuracy redetect + re-ID | 5–10 Hz | 7–17 ms; uses up periodic headroom. |
| Event/anomaly classification (action) | 5–10 Hz | 10–30 ms. |
| Fast reasoning (rule engine + evidence graph heuristics) | 2–5 Hz | CPU; ~50 ms per call. |
| **Deep VLM reasoning** | **0.3–0.5 Hz (every 2–3 s)** + event-triggered bursts | 800–3500 ms per call. Do NOT run at fixed 1 Hz unless hardware budget is multi-GPU. |
| Expensive forensic/causal analysis | Event-triggered only (expect 0.1 Hz avg) | 200–2000 ms ensemble. |

### TRIGGER SCORE CALIBRATION:
R ∈ [0, 1] threshold into tiers:
| R ≥ threshold | Action |
|---|---|
| 0.85 | Immediate VLM + forensic queue; bump to max priority; drop stale lower-R jobs if needed. |
| 0.5 | Add to VLM queue at normal priority; coalesce with nearby events. |
| 0.2 | Fast rule-engine pass only; no VLM. |
| <0.2 | Do nothing beyond 30 FPS path. |

### FAILURE BEHAVIOR (MANDATORY, not optional):
If deep reasoning lags → queue depth > threshold:
1. **NEVER block the 30 FPS path.** This is a hard design constraint; if you violate this you don't have a 30 FPS system.
2. Coalesce stale reasoning jobs: If 3 state snapshots from the same event window are pending, drop oldest 2, keep newest.
3. Prioritize newest high-R state: Fresh state at 0.9 R is better than stale state at 1.0 R.
4. Preserve previous hypothesis, explicitly tag with `staleness_ms`: Don't drop outputs; age them and show staleness in UI/API.
5. Backpressure to producers: If all queues deep, switch perception to even-lighter variant for 1–2 seconds (RF-DETR-N → YOLO26-N) until backlog clears.

### DELIVERABLE (practical 2026):
- Scheduler implementation with 4 priority tiers + bounded queues (depth 16-64 typical per tier).
- Coalescing policy by event window + entity overlap.
- Measurable Pareto curve: X-axis compute budget (GPU util %, dollar cost), Y-axis accuracy/calibration. Show that event-triggered adaptive scheduling beats fixed-rate scheduling at every compute budget. This is the research deliverable for the scheduler.
