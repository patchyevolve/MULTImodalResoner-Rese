# Backpressure Architecture

## Purpose

Detect and respond to system overload — when queues fill up, GPU utilization spikes, or latency degrades. Triggers protective actions to prevent cascading failures and maintain the 30 FPS critical path.

---

## Interfaces

### Input

```
SystemMetrics {
  queue_depths:    Map[string, int]
  gpu_utilization: float
  gpu_memory_used: float
  frame_latency_p50: float
  frame_latency_p99: float
  vlm_queue_depth: int
  dropped_frames:  int
  dropped_jobs:    int
}
```

### Output

```
BackpressureAction {
  action:          Enum                     # DISABLE_TRIGGER | DOWNGRADE_MODEL | DROP_LOW_PRIORITY | THROTTLE_VLM
  target:          string                   # which component to adjust
  reason:          string
  severity:        Enum                     # WARNING | CRITICAL | EMERGENCY
  duration_ms:     int                      # how long to apply
}
```

### API

```
evaluate(metrics: SystemMetrics) -> BackpressureAction[]
apply_action(action: BackpressureAction) -> void
```

---

## Data Contracts

### Trigger Thresholds

| Metric | WARNING | CRITICAL | EMERGENCY |
|---|---|---|---|
| VLM queue depth | >3 | >4 | Full |
| GPU utilization | >85% | >95% | >99% |
| Frame latency p99 | >28ms | >30ms | >33ms |
| Dropped frames | >0 | >3 in 10s | >5 in 10s |
| System memory | >80% | >90% | >95% |

### Response Escalation

```
WARNING → Disable low-priority triggers
CRITICAL → Disable low-priority + downgrade perception model
EMERGENCY → Disable all non-critical + throttle VLM to 0.1 Hz
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Metrics evaluation | 0.1ms | Threshold comparison |
| Action dispatch | 0.05ms | |
| **Total** | **<0.2ms** | |

---

## Reality Check 2026

### Backpressure Rules (from reference architecture):
1. **VLM queue depth = 4 max.** If full and new high-R event arrives, drop OLDEST pending.
2. **Coalescing mandatory.** Same entity/window ≥2 pending → keep newest with highest R.
3. **Staleness tag.** Every output carries `claim_staleness_ms`. If >5000ms → `stale = true`.
4. **Backpressure propagate.** VLM queue >3 for >30s → disable low-priority triggers.
5. **Perception downgrade.** System utilization >95% for >60s → YOLO26-N from RF-DETR-S.
