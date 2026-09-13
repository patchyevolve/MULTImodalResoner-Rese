# Short-Term Memory Architecture

## Purpose

Provide lock-free, sub-millisecond access to recent state snapshots and entity histories. This is the fast cache that the real-time path reads from — no locks, no allocations, no surprises.

---

## Interfaces

### Output

```
ShortTermMemory {
  state_ring:      RingBuffer[WorldStateSnapshot]    # last 30 frames
  entity_history:  Map[string, RingBuffer[EntityState]]  # per-entity history
  event_log:       RingBuffer[EventTrigger]           # last 100 events
  current_index:   int                                # write pointer
}
```

### API

```
write_state(snapshot: WorldStateSnapshot) -> void
read_current() -> WorldStateSnapshot
read_at(frame_id: uint64) -> WorldStateSnapshot
read_history(entity_id: string, frames: int) -> EntityState[]
write_event(event: EventTrigger) -> void
recent_events(count: int) -> EventTrigger[]
```

---

## Data Contracts

### Ring Buffer Configuration

```
ShortTermConfig {
  state_buffer_size: int                    # default 30 (1 second at 30 FPS)
  entity_history_size: int                  # default 30 per entity
  event_buffer_size: int                    # default 100
  max_entities:    int                      # default 500
  allocation:      Enum                     # PREALLOCATED | ON_DEMAND
}
```

### Storage

- **State ring:** Pre-allocated array of 30 WorldStateSnapshot structs. Lock-free SPSC write, lock-free read.
- **Entity history:** Pre-allocated per-entity ring buffers. Created on entity creation, destroyed on removal.
- **Event log:** Pre-allocated ring of 100 EventTrigger structs.
- **No dynamic allocation** in the hot path. All memory allocated at init.

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Write state | <0.05ms | Ring buffer push |
| Read current | <0.01ms | Atomic read of index |
| Read at frame_id | <0.01ms | Array lookup |
| Entity history | <0.1ms | Ring buffer slice |
| **Total** | **<0.1ms** | |

---

## Dependencies

### Upstream
- `03_state/01_world_state.md` — State snapshots to store

### Downstream
- Everything reads from short-term memory.
- `04_memory/02_working_memory.md` — Promotes relevant data
- `04_memory/03_long_term.md` — Periodic flush to persistent store

---

## Reality Check 2026

### Design Rules:
1. **No malloc in hot path.** All buffers pre-allocated at system start.
2. **No locks.** SPSC/MPMC ring buffers only.
3. **No copying.** Read directly from ring buffer slots.
4. **Staleness is explicit.** Every reader checks `timestamp_ns` of what it reads.
5. **Entity histories are bounded.** Max 30 frames per entity. Older data is in long-term memory.
