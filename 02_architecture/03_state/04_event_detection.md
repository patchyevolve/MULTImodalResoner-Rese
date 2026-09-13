# Event Detection Architecture

## Purpose

Detect events from world state changes — rule-based triggers that fire when specific conditions are met. Events are the bridge between perception and reasoning: they convert continuous state into discrete claims that the hypothesis engine can evaluate.

---

## Interfaces

### Input

```
EventDetectionInput {
  current_state:   WorldStateSnapshot
  prev_state:      WorldStateSnapshot
  trajectories:    Map[string, Trajectory]
  domain_rules:    DomainRule[]             # sport-specific or general rules
}
```

### Output

```
EventTrigger {
  event_id:        string
  event_type:      string                   # "goal" | "foul" | "offside" | "collision" | "anomaly"
  timestamp_ns:    uint64
  confidence:      float
  evidence:        Evidence[]               # what triggered this event
  priority_score:  float                    # R score for compute allocation
  requires_vlm:    bool                     # does this need deep reasoning?
  entities:        string[]                 # involved entity IDs
}

Evidence {
  type:            Enum                     # TRAJECTORY | POSE | RELATION | RULE | AUDIO
  description:     string
  weight:          float
  data:            Map[string, Any]         # supporting data
}
```

### API

```
detect_events(input: EventDetectionInput) -> EventTrigger[]
score_event_priority(event: EventTrigger) -> float
```

---

## Data Contracts

### Event Types

| Event | Trigger | Priority | VLM Required |
|---|---|---|---|
| Goal/Score | Ball crosses line + trajectory change | High | Yes |
| Foul | Pose anomaly + proximity + rule | High | Yes |
| Offside | Trajectory + line + timing | High | Yes |
| Collision | Proximity + sudden deceleration | Medium | No |
| Formation change | Entity positions over time | Low | No |
| Camera cut | Scene change detection | Medium | No |
| Anomaly | Prediction error > threshold | High | Yes |

### Priority Scoring (R Score)

```
R = α × prediction_error
  + β × uncertainty
  + γ × model_disagreement
  + δ × event_importance
  + ε × user_priority

Where:
  prediction_error: how far state deviates from expectation (0-1)
  uncertainty: system uncertainty at this moment (0-1)
  model_disagreement: how much perception models disagree (0-1)
  event_importance: domain-specific importance (0-1)
  user_priority: user request priority (0-1)
```

### Configuration

```
EventConfig {
  rules:           DomainRule[]             # loaded from domain config
  min_confidence:  float                    # default 0.3
  priority_thresholds: Map[string, float]   # per event type
  cooldown_ms:     int                      # min time between same events, default 1000
}
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Rule evaluation (per rule) | 0.01–0.05ms | Simple comparisons |
| Trajectory-based triggers | 0.05–0.1ms | Threshold checks |
| Priority scoring | 0.01ms | Linear combination |
| **Total** | **0.1–0.3ms** | |

---

## Dependencies

### Upstream
- `03_state/01_world_state.md` — Current and previous state
- `03_state/03_trajectory_model.md` — Trajectory predictions
- `09_domains/01_sports_reasoning.md` — Domain rules

### Downstream
- `05_reasoning/01_hypothesis_engine.md` — Events as evidence
- `07_scheduler/01_multi_rate_scheduler.md` — Priority for compute allocation
- `04_memory/04_episodic_memory.md` — Event summaries

---

## Reality Check 2026

### Priority Scoring:
- R ≥ 0.85 → VLM max priority
- R ≥ 0.5 → normal VLM processing
- R < 0.5 → fast-rule-only, no VLM

### Sports Events:
- Goals, fouls, offsides: high priority, always VLM-verified.
- Passes, dribbles: low priority, rule-based only.
- Anomalies: prediction error > 2σ triggers investigation.

### General Events:
- Camera cuts: scene change triggers re-initialization.
- New entity: first detection triggers identity assignment.
- Anomaly: any state that deviates from prediction triggers investigation.
