# Prediction Model Architecture

## Purpose

Predict future states — where entities will be, what events will happen next, what the scene will look like in N seconds. Provides anticipatory capabilities for proactive reasoning and early event detection.

---

## Interfaces

### Input

```
PredictionInput {
  current_state:   WorldStateSnapshot
  trajectory:      Trajectory
  context:         TemporalFeatures
  horizon_ms:      int                      # prediction horizon, default 2000 (2s)
  domain_rules:    DomainRule[]
}
```

### Output

```
PredictionOutput {
  entity_id:       string
  predicted_state: EntityState
  predicted_events: EventTrigger[]
  confidence:      float
  horizon_ms:      int
  prediction_method: string
  inference_ms:    float
}
```

### API

```
predict_next_event(state: WorldStateSnapshot, horizon_ms: int) -> PredictionOutput
predict_entity_state(entity_id: string, horizon_ms: int) -> PredictionOutput
assess_prediction_quality(prediction: PredictionOutput, actual: WorldStateSnapshot) -> float
```

---

## Data Contracts

### Prediction Methods

| Method | Latency | Accuracy | Notes |
|---|---|---|---|
| Trajectory extrapolation | 0.1ms | Good (short) | Constant velocity/acceleration |
| Rule-based prediction | 1–5ms | Moderate | Domain rules + current state |
| Statistical model | 5–10ms | Good | Learned from data |
| VLM-based prediction | 800–3500ms | Best | Expensive, async |

### SoccerNet 2026 Baseline:
- 5s horizon: best achieves 24.08% avg mAP (vs 16.76% baseline).
- 2s horizon: 18.18% (vs 13.00% baseline).
- **Key insight:** Prediction is hard. Even SOTA is far from reliable at 5s.

---

## Latency Budget

| Method | Budget | Notes |
|---|---|---|
| Trajectory extrapolation | 0.1ms | |
| Rule-based | 1–5ms | |
| Statistical | 5–10ms | |
| VLM-based | 800–3500ms | Async |

---

## Dependencies

### Upstream
- `03_state/03_trajectory_model.md` — Trajectory data
- `03_state/01_world_state.md` — Current state

### Downstream
- `03_state/04_event_detection.md` — Predicted events
- `05_reasoning/01_hypothesis_engine.md` — Predictions as hypotheses
- `05_reasoning/03_deep_vlm_reasoner.md` — VLM prediction

---

## Reality Check 2026

### Prediction Quality:
- At 0.5–2s: trajectory extrapolation works well for linear motion.
- At 2–5s: accuracy drops significantly. Report uncertainty explicitly.
- At 5s+: prediction is unreliable for most scenarios. Use for hypothesis generation only.
- **Never present predictions as facts.** Always tag as `epistemic_status: PREDICTED`.
