# Hardware Topology Architecture

## Purpose

Define the physical hardware layout and GPU allocation for the system. Maps components to specific GPUs, specifies interconnects, and defines deployment tiers from minimum viable to production.

---

## Hardware Tiers

### Tier 1: Minimum Viable (1× RTX 4090 24GB)

```
GPU 0: RTX 4090 24GB
  ├── RF-DETR-S/pose (~3 GB)
  ├── ByteTrack state (~0.1 GB)
  └── Qwen 3-VL-30B-A3B FP8 (~20 GB peak)
  Total: ~23.1 GB (tight fit)
  Perception: 30 FPS ✓
  Deep reasoning: 0.3-0.5 Hz ✓ (queue depth monitoring required)
```

### Tier 2: Recommended (1× A100 80GB or 2× RTX 4090)

```
Option A: 1× A100 80GB
  ├── All Tier 1 components
  ├── Forensic ensemble (~5 GB)
  ├── Higher-quality VLM (Qwen 32B AWQ ~16 GB)
  └── Headroom for p99
  Perception: 30 FPS ✓
  Deep reasoning: 0.5-1 Hz ✓

Option B: 2× RTX 4090
  GPU 0: Perception
    ├── RF-DETR-S/pose
    ├── ByteTrack
    └── Fast reasoning
  GPU 1: Reasoning
    ├── Qwen 3-VL-30B-A3B
    ├── Forensic ensemble
    └── VLM reasoning
  Perception: 30 FPS ✓
  Deep reasoning: 1 Hz ✓
```

### Tier 3: Server/Production (2+ A100/H100)

```
GPU 0: Perception
  ├── RF-DETR-L (async high-accuracy)
  ├── DETRPose-L
  ├── ByteTrack
  └── Camera motion
GPU 1: Fast Reasoning
  ├── Hypothesis engine
  ├── Fast verifier
  └── Evidence graph
GPU 2: Deep Reasoning
  ├── Qwen 3.5-397B (TP=4)
  └── VLM reasoning
GPU 3: Forensics + Memory
  ├── Deepfake ensemble
  ├── C2PA/SynthID
  └── Long-term memory
  Perception: 30 FPS ✓
  Deep reasoning: 1-2 Hz ✓
  Redundancy: Yes
```

### Edge Tier: Jetson Thor

```
Jetson AGX Thor (128GB, 2070 FP4 TFLOPS)
  ├── Lightweight perception (RF-DETR-N)
  ├── Lightweight pose (DETRPose-S)
  ├── ByteTrack
  └── Small VLM (20B at 52 tok/s)
  Perception: 30 FPS ✓
  Deep reasoning: 0.1-0.3 Hz ✓
  Power: 40-130W
```

---

## Interconnect Requirements

| Connection | Bandwidth | Latency | Notes |
|---|---|---|---|
| GPU ↔ GPU (NVLink) | 600+ GB/s | <1μs | H100/H200 |
| GPU ↔ GPU (PCIe) | 64 GB/s | ~1μs | RTX 4090 |
| GPU ↔ CPU (PCIe) | 32 GB/s | ~2μs | Use pinned memory |
| Network (10GbE) | 1.25 GB/s | ~100μs | API/cluster |
| Network (100GbE) | 12.5 GB/s | ~10μs | Multi-node |

---

## GPU Memory Budget

### RTX 4090 (24GB) Allocation:

| Component | Memory | Notes |
|---|---|---|
| RF-DETR-S | 1.5 GB | Detection |
| DETRPose-S | 0.5 GB | Pose |
| ByteTrack | 0.05 GB | Tracking |
| Qwen 3-VL-30B-A3B FP8 | 20 GB | VLM |
| CUDA context + buffers | 1 GB | Runtime |
| **Total** | **23.05 GB** | 96% utilization — tight |

### A100 (80GB) Allocation:

| Component | Memory | Notes |
|---|---|---|
| RF-DETR-L | 2 GB | Detection |
| DETRPose-L | 1 GB | Pose |
| ByteTrack | 0.05 GB | Tracking |
| Qwen 32B AWQ | 16 GB | VLM |
| Forensic ensemble | 5 GB | Detection |
| Long-term memory | 10 GB | FAISS index |
| CUDA context + buffers | 2 GB | Runtime |
| **Total** | **36.05 GB** | 45% utilization — comfortable |

---

## Reality Check 2026

### RTX 5090 (32GB, $2K):
- 1.27× FP16 of RTX 4090, 1.78× bandwidth.
- Can fit 70B INT4 with long context.
- Better value than 2× RTX 4090 for single-GPU builds.

### Jetson Thor (128GB, ~$2K):
- 2,070 FP4 TFLOPS, 7.5× performance vs AGX Orin.
- 20B at 52 tok/s, 35B at 35 tok/s.
- Best edge option for portable deployments.

### NVIDIA DGX Spark:
- 2 alerting streams, 1 captioning stream (RT-VLM).
- Good for development/testing, not production multi-stream.
