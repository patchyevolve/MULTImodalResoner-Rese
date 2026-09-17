# Optimization Strategies

> Every optimization technique we apply, why we apply it, and how to verify it works. This is what separates "it runs" from "it runs fast."

---

## 1. Optimization Stack (Priority Order)

| Priority | Optimization | Speedup | Complexity | Apply To |
|---|---|---|---|---|
| 1 | Mixed Precision (FP16/BF16) | 1.5-2x | Low | All GPU training + inference |
| 2 | TensorRT export | 2-5x | Medium | All deployed models |
| 3 | torch.compile | 1.2-1.3x | Low | Training loop |
| 4 | CUDA Graphs | 1.1-1.3x | Medium | Inference pipeline |
| 5 | Batch coalescing | 1.5-2x | Low | Multi-stream inference |
| 6 | Pinned memory + zero-copy | 1.2-1.5x | Low | CPU↔GPU transfers |
| 7 | INT8 quantization | 2-3x | High | Deployment if needed |
| 8 | Model pruning | 1.3-2x | High | Only if VRAM constrained |

---

## 2. Mixed Precision Training (FP16/BF16)

### What It Does

Uses FP16 (16-bit floating point) for forward/backward passes, FP32 for master weights. Reduces memory usage by ~40% and speeds up computation on Tensor Cores.

### Implementation

```python
# Automatic Mixed Precision (AMP)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in FP16
    with autocast(dtype=torch.float16):
        outputs = model(batch)
        loss = criterion(outputs, targets)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    scaler.step(optimizer)
    scaler.update()
```

### When to Use

| Component | FP16 | BF16 | FP32 | Why |
|---|---|---|---|---|
| RF-DETR training | ✅ | ✅ | | Proven stable with DETR models |
| DETRPose training | ✅ | ✅ | | Same |
| OSNet training | ✅ | ✅ | | Re-ID models are small, FP16 sufficient |
| Whisper inference | ✅ | ✅ | | Encoder/decoder both support FP16 |
| VLM inference | ✅ | ✅ | | vLLM handles this automatically |
| Calibration training | | | ✅ | CPU-based, no GPU needed |

### Verification

```bash
# Compare FP32 vs FP16 accuracy
python scripts/compare_precision.py \
    --model rf_detr_s \
    --data val_split.json \
    --precisions fp32 fp16 bf16

# Expected: <0.1% mAP difference between FP32 and FP16
```

---

## 3. TensorRT Export (Critical for Inference)

### What It Does

Compiles PyTorch models into optimized CUDA kernels. Fuses operations, optimizes memory layout, and enables FP16/INT8 inference.

### Export Pipeline

```bash
# Step 1: Export to ONNX
python scripts/export_onnx.py \
    --model rf_detr_s \
    --checkpoint models/checkpoints/rf_detr_s/best.pth \
    --output models/onnx/rf_detr_s.onnx \
    --input-shape 1 3 640 640 \
    --opset 17

# Step 2: Convert to TensorRT FP16
trtexec \
    --onnx=models/onnx/rf_detr_s.onnx \
    --saveEngine=models/trt_engines/rf_detr_s_fp16.engine \
    --fp16 \
    --workspace=4096 \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:1x3x640x640 \
    --maxShapes=input:4x3x640x640

# Step 3: Validate accuracy
python scripts/validate_trt.py \
    --engine models/trt_engines/rf_detr_s_fp16.engine \
    --onnx-model models/onnx/rf_detr_s.onnx \
    --data val_split.json \
    --tolerance 0.01  # Allow 1% accuracy difference
```

### Expected Speedups

| Model | PyTorch FP32 | PyTorch FP16 | TensorRT FP16 | Speedup |
|---|---|---|---|---|
| RF-DETR-S | 6.8ms | 4.5ms | 2.3ms | 3.0x |
| DETRPose-S | 5.0ms | 3.2ms | 2.0ms | 2.5x |
| OSNet | 5.0ms | 3.0ms | 1.5ms | 3.3x |
| Whisper encoder | 50ms | 30ms | 15ms | 3.3x |

---

## 4. torch.compile (Training Speedup)

### What It Does

PyTorch 2.0+ compilation. Fuses operations, generates optimized kernels, and reduces Python overhead. 20-30% training speedup with minimal code changes.

### Implementation

```python
# Simple one-liner
model = torch.compile(model, mode="max-autotune")

# Or with specific options
model = torch.compile(
    model,
    mode="reduce-overhead",  # Optimize for inference
    fullgraph=True,          # Capture full computation graph
    dynamic=False,           # Static shapes for best performance
)
```

### When to Use

| Phase | Use torch.compile? | Why |
|---|---|---|
| Training | ✅ `mode="default"` | 20-30% speedup |
| Inference | ✅ `mode="max-autotune"` | 30-40% speedup |
| Export to ONNX | ⚠️ May cause issues | Disable before export |
| Debugging | ❌ | Harder to debug compiled models |

---

## 5. CUDA Graphs (Inference Latency)

### What It Does

Captures a sequence of CUDA operations and replays them without CPU overhead. Eliminates kernel launch latency.

### Implementation

```python
# For inference pipeline
g = torch.cuda.CUDAGraph()

# Warmup: run once to capture
with torch.cuda.graph(g):
    static_input = input_tensor.clone()
    static_output = model(static_input)

# Subsequent runs: replay graph (no CPU overhead)
static_input.copy_(new_input)
g.replay()
output = static_output.clone()
```

### Expected Impact

| Metric | Without CUDA Graphs | With CUDA Graphs | Improvement |
|---|---|---|---|
| Kernel launch latency | 5-10μs per op | 0 | Eliminated |
| End-to-end latency | 5ms | 4.2ms | 16% |
| CPU utilization | 30% | 10% | 3x reduction |

---

## 6. Pinned Memory + Zero-Copy Transfers

### What It Does

Allocates CPU memory as "pinned" (page-locked), enabling async DMA transfers to GPU. Overlaps CPU preparation with GPU computation.

### Implementation

```python
# Wrong way (slow)
tensor = torch.tensor(data)                    # pageable memory
tensor = tensor.to('cuda')                     # synchronous copy

# Right way (fast)
tensor = torch.tensor(data).pin_memory()       # pinned memory
tensor = tensor.to('cuda', non_blocking=True)  # async copy

# Even better: pre-allocate pinned buffers
class PinnedBuffer:
    def __init__(self, shape, dtype=torch.float32):
        self.cpu = torch.empty(shape, dtype=dtype).pin_memory()
        self.gpu = torch.empty(shape, dtype=dtype, device='cuda')

    def copy_async(self, data):
        self.cpu.copy_(data, non_blocking=True)
        self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu
```

---

## 7. Batch Coalescing (Multi-Stream Inference)

### What It Does

Groups small inference requests into batches for better GPU utilization. Instead of running 1 detection per frame, batch 2-4 frames together.

### Implementation

```python
class BatchCoalescer:
    def __init__(self, max_batch_size=4, max_wait_ms=10):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending = []

    def add(self, tensor, callback):
        self.pending.append((tensor, callback))
        if len(self.pending) >= self.max_batch_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        batch = torch.stack([t for t, _ in self.pending])
        outputs = model(batch)
        for i, (_, callback) in enumerate(self.pending):
            callback(outputs[i])
        self.pending = []
```

---

## 8. INT8 Quantization (If VRAM Constrained)

### What It Does

Reduces model weights from FP32 (4 bytes) to INT8 (1 byte). 4x memory reduction, 2-3x inference speedup. Small accuracy loss.

### When to Use

| Scenario | Use INT8? |
|---|---|
| RTX 4090 (24GB) | No — FP16 sufficient |
| RTX 5070 Ti (16GB) | Maybe — if VRAM tight |
| Edge deployment | Yes — memory critical |
| Accuracy-critical | No — FP16 better |

### Implementation

```python
# PTQ (Post-Training Quantization) — no retraining needed
from torch.quantization import quantize_dynamic

model_int8 = quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize linear layers
    dtype=torch.qint8
)

# QAT (Quantization-Aware Training) — better accuracy, needs retraining
# Only if INT8 accuracy is insufficient
```

---

## 9. Multi-GPU Inference (If Available)

### What It Does

Splits model across multiple GPUs for larger models or higher throughput.

### Implementation

```python
# Tensor Parallelism for VLM
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    tensor_parallel_size=2,  # Split across 2 GPUs
    dtype="float16",
    gpu_memory_utilization=0.9,
)
```

---

## 10. Performance Profiling

### How to Measure

```bash
# PyTorch Profiler
python -c "
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(6):
        output = model(input_tensor)
        prof.step()
"

# Nsight Systems (NVIDIA profiling)
nsys profile -o profile_report python scripts/inference_benchmark.py

# ncu (CUDA kernel profiler)
ncu --set full python scripts/inference_benchmark.py
```

### Optimization Checklist

After profiling, check:

- [ ] GPU utilization > 80% during inference
- [ ] No CPU bottlenecks (data loading, preprocessing)
- [ ] Memory transfers are async (pinned memory)
- [ ] No unnecessary .cpu() calls in hot path
- [ ] CUDA graphs capturing repeatable patterns
- [ ] No Python overhead in critical path (use torch.compile)
- [ ] Batch sizes are power-of-2 where possible
- [ ] No memory leaks (memory usage is stable over time)

---

## 11. Optimization Results Target

| Component | Baseline | After Optimization | Target |
|---|---|---|---|
| Detection (RF-DETR-S) | 6.8ms | 2.3ms | <3ms |
| Pose (DETRPose-S) | 5.0ms | 2.0ms | <3ms |
| Tracking (ByteTrack) | 0.2ms | 0.2ms | <0.5ms |
| VLM (Qwen3-VL-30B) | 3500ms | 800ms | <1500ms |
| Calibration | 8ms | 3ms | <5ms |
| **End-to-end fast path** | 40ms | 15ms | <20ms |
