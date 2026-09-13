# Entity Tracker Architecture (State-Level)

> **Note:** This is the state-level entity lifecycle manager, not the low-level ByteTrack (see `01_perception/03_tracking.md`). This component manages entity creation, attribute updates, occlusion handling, and identity merging across the system.

## Purpose

Manage the lifecycle of each tracked entity — creation, attribute accumulation, occlusion recovery, identity merging, and removal. Provides the entity abstraction that reasoning, memory, and domain logic consume.

---

## Interfaces

### Input

```
EntityUpdate {
  entity_id:       string
  frame_id:        uint64
  timestamp_ns:    uint64
  detection:       Detection                # new detection (if any)
  pose:            PoseResult               # new pose (if any)
  reid_match:      ReIDOutput               # re-identification result (if any)
  track_update:    Track                    # tracker state update
}
```

### Output

```
Entity {
  id:              string
  class:           string
  first_seen_ns:   uint64
  last_seen_ns:    uint64
  lifetime_frames: int
  current_state:   EntityState
  history:         EntityState[]            # last N states (ring buffer)
  identity:        EntityIdentity           # cross-view identity
  attributes:      Map[string, Any]
}

EntityState {
  frame_id:        uint64
  timestamp_ns:    uint64
  bbox_norm:       [4]float
  pose:            PoseResult
  velocity:        [2]float
  confidence:      float
  occlusion_level: float
  is_keyframe:     bool
}

EntityIdentity {
  canonical_id:    string                   # persistent ID across views/sessions
  aliases:         string[]                 # track IDs from different cameras
  reid_features:   [][512]float             # recent appearance features
  last_reid_match: uint64                   # timestamp of last re-ID check
}
```

### API

```
create_entity(update: EntityUpdate) -> Entity
update_entity(entity_id: string, update: EntityUpdate) -> Entity
merge_entities(primary_id: string, secondary_id: string) -> Entity
remove_entity(entity_id: string) -> bool
get_entity(entity_id: string) -> Entity
get_all_active_entities() -> Entity[]
```

---

## Data Contracts

### Entity Lifecycle

```
DETECTED (first detection, score > threshold)
  → CONFIRMED (3+ consecutive frames matched)
    → ACTIVE (regular updates)
      → OCCLUDED (no detection but tracker interpolating)
        → RECOVERED (detection reappears, identity verified)
        → LOST (no detection for max_age frames)
          → REMOVED
```

### Attribute Accumulation

| Attribute | Source | Update Rule |
|---|---|---|
| team/color | Detection classifier | Majority vote over lifetime |
| jersey_number | OCR | Highest-confidence match |
| position/role | Reasoning/domain | Most recent inference |
| speed | Trajectory model | Exponential moving average |
| pose | Pose pipeline | Most recent frame |

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Entity update | 0.05–0.1ms | Per entity |
| Entity creation | 0.1ms | |
| Identity merge | 0.5–1ms | Re-ID comparison |
| Entity removal | 0.02ms | Hash map delete |
| Active entity query | <0.1ms | Lock-free read |

---

## Dependencies

### Upstream
- `01_perception/03_tracking.md` — Track updates
- `01_perception/06_reid.md` — Identity matching
- `01_perception/02_pose_estimation.md` — Pose data

### Downstream
- `03_state/03_trajectory_model.md` — Entity histories
- `05_reasoning/01_hypothesis_engine.md` — Entity context
- `04_memory/03_long_term.md` — Entity persistence

---

## Reality Check 2026

### Multi-Person Sports:
- Up to 22 players + 3 officials + ball = 26 entities simultaneously.
- Re-ID across camera cuts: OSNet at 2–5ms per check, gallery of ~100.
- Occlusion recovery: kinematic model predicts position during occlusion (28% gap reduction with MoRo).

### Identity Merging:
- When two tracks are merged (same person, different cameras), canonical ID persists.
- History from both tracks is combined, timestamps maintained.
- Re-ID features from both tracks are stored for future matching.
