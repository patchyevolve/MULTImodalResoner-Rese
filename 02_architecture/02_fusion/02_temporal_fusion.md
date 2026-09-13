# Temporal Fusion Architecture

## Purpose

Aggregate information across time windows to capture temporal patterns — motion, events, trends, and context that single frames cannot represent. Provides the bridge between per-frame perception and temporal reasoning.

---

## Interfaces

### Input

```
TemporalInput {
  current_state:   WorldStateSnapshot        # current frame state
  state_history:   WorldStateSnapshot[]      # last N frames (ring buffer)
  window_size_ms:  int                      # aggregation window, default 1000
  stride_ms:       int                      # window stride, default 100
}
```

### Output

```
TemporalOutput {
  timestamp_ns:    uint64
  aggregated:      TemporalAggregate
  features:        TemporalFeatures
  inference_ms:    float
}

TemporalAggregate {
  entity_trajectories: Map[string, Trajectory]
  motion_patterns:  MotionPattern[]
  temporal_events:  TemporalEvent[]
  scene_changes:    SceneChange[]
}

TemporalFeatures {
  optical_flow:    Tensor[H, W, 2]          # if computed
  motion_embeddings: [N][128]float          # per-entity motion features
  scene_embedding: [256]float               # scene-level temporal feature
}
```

### API

```
aggregate_temporal(input: TemporalInput, config: TemporalConfig) -> TemporalOutput
```

---

## Data Contracts

### Aggregation Methods

| Method | Latency | Memory | Notes |
|---|---|---|---|
| Sliding window (last N) | 0.5–1ms | O(N×entities) | Simple, fixed window |
| Exponential moving average | 0.2–0.5ms | O(entities) | Lightweight, recent bias |
| Keyframe interpolation | 1–3ms | O(keyframes) | Variable rate |
| RNN/GRU state | 2–5ms | O(state_dim) | Learned temporal model |

### Configuration

```
TemporalConfig {
  method:          Enum                     # SLIDING_WINDOW | EMA | KEYFRAME | GRU
  window_size_ms:  int                      # default 1000 (1 second)
  max_history_frames: int                   # default 30 (1 second at 30 FPS)
  keyframe_interval_ms: int                 # default 500
  compute_optical_flow: bool                # default false (expensive)
}
```

---

## Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| History retrieval | 0.1–0.3ms | Ring buffer access |
| Aggregation | 0.3–1ms | Depending on method |
| Motion feature extraction | 0.5–2ms | If computed |
| **Total** | **0.5–3ms** | |

---

## Dependencies

### Upstream
- `04_memory/01_short_term.md` — State history ring buffer
- `03_state/01_world_state.md` — Current state

### Downstream
- `03_state/03_trajectory_model.md` — Trajectory input
- `03_state/04_event_detection.md` — Temporal event triggers
- `05_reasoning/05_prediction_model.md` — Temporal context

---

## Reality Check 2026

### Key Insight:
- 30 FPS provides 33ms between frames — temporal fusion must be lightweight.
- Most temporal information comes from trajectory history, not complex temporal models.
- EMA is sufficient for 90% of use cases; GRU only if learned temporal patterns are needed.
- Optical flow is expensive (5–15ms) — compute only when explicitly needed (e.g., action recognition).
