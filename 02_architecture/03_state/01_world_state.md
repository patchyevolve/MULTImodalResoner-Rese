# World State Architecture

## Purpose

Maintain a typed, per-frame snapshot of all entities, attributes, relationships, and scene context. This is the single source of truth for what is happening right now. Every downstream component reads from or writes to world state.

---

## Interfaces

### Input

```
WorldStateUpdate {
  frame_id:        uint64
  timestamp_ns:    uint64
  entities:        EntityUpdate[]            # from perception
  scene:           SceneUpdate               # scene-level attributes
  events:          Event[]                   # triggered events
  uncertainty:     UncertaintySummary         # system-wide uncertainty
}
```

### Output

```
WorldStateSnapshot {
  frame_id:        uint64
  timestamp_ns:    uint64
  entities:        Entity[]
  trajectories:    Trajectory[]
  relations:       Relation[]
  scene:           Scene
  events:          Event[]
  uncertainty:     UncertaintySummary
  staleness_ms:    float                    # time since this snapshot was current
}

Entity {
  id:              string
  class:           string                   # "person" | "ball" | "object"
  bbox_norm:       [4]float
  pose:            PoseResult               # from pose pipeline
  velocity:        [2]float
  acceleration:    [2]float
  attributes:      Map[string, Any]         # team, jersey_number, role, etc.
  confidence:      float
  occlusion_level: float                    # 0=visible, 1=fully occluded
  tracking_state:  Enum                     # TENTATIVE | CONFIRMED | LOST
}

Relation {
  subject:         string                   # entity_id
  predicate:       string                   # "near" | "faster_than" | "behind" | "interacting_with"
  object:          string                   # entity_id
  confidence:      float
  spatial:         SpatialRelation          # distance, angle, relative_position
}
```

### API

```
get_current_state() -> WorldStateSnapshot
get_state_at(timestamp_ns: uint64) -> WorldStateSnapshot
get_entity(entity_id: string) -> Entity
update_state(update: WorldStateUpdate) -> WorldStateSnapshot
```

---

## Data Contracts

### Storage

- **Current state:** Lock-free ring buffer (SPSC/MPMC), one slot per frame.
- **State history:** Ring buffer of last 30 frames (1 second at 30 FPS).
- **Entity map:** Concurrent hash map (entity_id → Entity).
- **Relations:** Computed on-demand from entity positions, not stored.

### Update Rules

1. **Create entity:** First detection with score > threshold.
2. **Update entity:** Merge detection + tracking + pose + state estimate.
3. **Mark occluded:** If detector confidence drops or tracking interpolates.
4. **Remove entity:** After max_age frames without detection.

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Entity update (per entity) | 0.05–0.1ms | Merge new data |
| State snapshot publish | 0.2–0.5ms | Ring buffer write |
| Snapshot retrieval | <0.1ms | Lock-free read |
| **Total update** | **0.3–1ms** | |

---

## Dependencies

### Upstream
- `01_perception/` — All perception outputs
- `02_fusion/` — Fused features

### Downstream
- Everything downstream reads world state.

---

## Reality Check 2026

### Design Principles:
- World state is **append-only per frame** — never mutate in-place.
- Every snapshot is immutable once published — safe for concurrent reads.
- Entity IDs are strings, not integers — allows cross-session persistence.
- Relations are computed, not stored — avoids stale relation bugs.
- Staleness is explicit — every consumer knows how old the data is.
