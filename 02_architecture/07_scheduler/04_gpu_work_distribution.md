# GPU Work Distribution Architecture

## Purpose

Efficiently distribute work across GPU resources — CUDA streams, MPS/MIG partitions, and multi-GPU topologies. Ensures perception and reasoning share GPU without starving either.

---

## Interfaces

### Input/Output

```
GPUAllocation {
  gpu_id:          int
  streams:         CUDAStream[]
  workloads:       Workload[]
  utilization:     float
  memory_used:     float
}

CUDAStream {
  stream_id:       int
  priority:        int                      # higher = scheduled first
  workload_type:   string                   # "perception" | "reasoning" | "both"
}

Workload {
  id:              string
  type:            Enum                     # PERCEPTION | VLM_ENCODE | VLM_DECODE | FORENSICS
  gpu_stream:      int
  priority:        int
  estimated_ms:    float
}
```

### API

```
allocate_workload(workload: Workload) -> GPUAllocation
get_stream_priority(workload_type: string) -> int
get_gpu_utilization(gpu_id: int) -> float
```

---

## Data Contracts

### CUDA Stream Priorities

| Stream | Priority | Workload |
|---|---|---|
| Perception | Highest | Detection, tracking, pose |
| Fast reasoning | High | Rule checks, trajectory |
| VLM prefill | Medium | Vision encoding + prompt |
| VLM decode | Medium-Low | Token generation |
| Forensics | Low | Deepfake detection |
| Summary | Lowest | Episodic memory |

### MPS Configuration (for single-GPU sharing)

```
MPSConfig {
  enable_mps:      bool                     # default true for multi-workload
  perception_share: float                   # SM percentage, default 0.4
  reasoning_share: float                    # SM percentage, default 0.4
  reserve:         float                    # headroom, default 0.2
}
```

### MIG Configuration (for strict isolation)

```
MIGConfig {
  enable_mig:      bool                     # false by default (less flexible)
  instance_count:  int                      # default 3
  per_instance_memory: int                  # GB, default 10
}
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Stream dispatch | 0.05–0.1ms | CUDA stream launch |
| MPS context switch | 0.1–0.2ms | Software partitioning |
| MIG context switch | <0.01ms | Hardware partitioning |
| **Total** | **<0.2ms** | |

---

## Dependencies

### Upstream
- `10_infrastructure/02_hardware_topology.md` — GPU configuration
- `07_scheduler/01_multi_rate_scheduler.md` — Job scheduling

### Downstream
- All GPU-accelerated components

---

## Reality Check 2026

### UnifiedServe (MPS, Dec 2025):
- Shares GPU between vision encoder and LLM decode.
- 3.0× more requests or 1.5× tighter SLOs, 4.4× throughput.
- Decode stream has priority; encoder uses leftover SM cycles.

### MIG (A100/H100):
- Up to 7 isolated instances, each with dedicated SMs, L2, memory.
- Better for strict isolation but less flexible than MPS.
- GB200: 2×93GB, 4×46GB, or 7×23GB instances.

### Practical Decision:
- **Single GPU, complementary workloads:** MPS (vision + LLM share).
- **Single GPU, strict isolation needed:** MIG.
- **Multi-GPU:** Separate perception GPU + reasoning GPU.
- **Streaming priorities:** Perception stream always highest priority.
