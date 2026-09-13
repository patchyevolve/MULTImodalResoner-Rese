# Queue Management Architecture

## Purpose

Manage bounded queues for each processing rate. Enforces depth limits, coalescing rules, staleness detection, and drop policies. Prevents unbounded memory growth and ensures the system degrades gracefully under load.

---

## Interfaces

### Input/Output

```
Queue[T] {
  name:            string
  max_depth:       int
  current_depth:   int
  items:           T[]                      # bounded array
  head:            int                      # read pointer
  tail:            int                      # write pointer
  strategy:        Enum                     # DROP_OLDEST | DROP_LOWEST_PRIORITY | COALESCE
}

QueueOperation[T] {
  enqueue:         (item: T) -> bool         # false if dropped
  dequeue:         () -> T | null
  peek:            () -> T | null
  coalesce:        (key_fn: (T) -> string) -> void
  drop_stale:      (max_age_ms: int) -> T[]
}
```

### API

```
create_queue(name: string, max_depth: int, strategy: string) -> Queue
enqueue(queue_name: string, item: Any) -> bool
dequeue(queue_name: string) -> Any
get_depth(queue_name: string) -> int
coalesce(queue_name: string, key_fn: string) -> void
drop_stale(queue_name: string, max_age_ms: int) -> Any[]
```

---

## Data Contracts

### Queue Configurations

| Queue | Max Depth | Strategy | Staleness |
|---|---|---|---|
| perception | 1 | DROP_LOWEST_PRIORITY | N/A |
| fast_reason | 4 | DROP_OLDEST | 1000ms |
| deep_vlm | 4 | COALESCE + DROP_OLDEST | 5000ms |
| forensics | 8 | DROP_OLDEST | 10000ms |
| summary | 4 | DROP_OLDEST | 30000ms |

### Coalescing Rules

```
1. Same entity + same event window + ≥2 pending → keep newest with highest R score.
2. Drop lower-priority items for same entity.
3. Never exceed max_depth. If enqueue would exceed → drop lowest priority.
```

### Staleness Detection

```
For each item in queue:
  staleness_ms = current_time - item.created_at
  if staleness_ms > max_staleness_ms:
    item.stale = true
    item.confidence *= 0.5  # discount confidence for stale items
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Enqueue | <0.05ms | Ring buffer write |
| Dequeue | <0.05ms | Ring buffer read |
| Coalesce | 0.1–0.5ms | Scan + remove |
| Drop stale | 0.1–0.3ms | Scan + remove |
| **Total** | **<0.5ms** | |

---

## Reality Check 2026

### Backpressure Propagation (from reference architecture):
- VLM queue depth >3 for >30s → disable low-priority triggers.
- User query + scene cut still force through.
- Eventually trigger perception model downgrade if system utilization >95% for >60s.
- **Every queue has a depth limit + drop policy.** No exceptions.
