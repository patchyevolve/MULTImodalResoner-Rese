# Runtime optimization research

Investigate:

- model quantization
- mixed precision
- TensorRT/CUDA or equivalent acceleration
- kernel fusion
- batching vs batch-1 behavior
- persistent GPU memory
- zero-copy or pinned memory strategies
- token pruning
- temporal token caching
- feature reuse across frames
- detector skipping with tracker propagation
- early exit
- model cascades
- speculative reasoning
- workload partitioning across CPU/GPU/NPU

## Rule
Optimize the complete pipeline, not isolated inference benchmarks. A faster model can still reduce throughput if it increases transfers, synchronization or memory pressure.

---

## REALITY CHECK 2026: VERIFIED OPTIMIZATION TECHNIQUES

### QUANTIZATION:

| Method | Hardware | Speedup | Accuracy Loss | Status |
|---|---|---|---|---|
| FP8 | H100/H200 | 1.3–2.1× | <1.5% MMLU | Production-ready |
| NVFP4 | Blackwell (B200) | Higher | TBD | Production-ready on Blackwell |
| INT4 AWQ | Any | 0.83× (worse) | 5.66% MMLU | Avoid for inference |
| W4A8 AWQ | Any | 1.15× | Moderate | Better than INT4 |
| 2:4 structured sparsity | H100 | 1.3–1.62× | Negligible | Production-ready |

**Recommendation:** FP8 is the sweet spot for H100/H200. FP4 requires Blackwell and is still maturing.

### KERNEL FUSION:

**FlashAttention-4 (Mar 2026):**
- Written in CuTe-DSL (Python), JIT compiled to PTX → SASS.
- Compile time: 2.5s (vs 55s for FA-3) = 22× faster compilation.
- Performance: 1,613 TFLOPs/s on B200 (~71% of theoretical max).
- 1.3× faster than cuDNN 9.13, 2.7× faster than Triton.

**FlashFormer:**
- Fuses entire transformer forward pass into single kernel.
- Targets low-batch inference (memory-bandwidth bound).

### TOKEN COMPRESSION (2026):

| Method | Compression | Speedup | Accuracy | Training | Venue |
|---|---|---|---|---|---|
| StreamingTOM | 15.7× KV cache | 2× TTFT | 63.8% VideoMME | None | CVPR 2026 |
| HybridKV | 7.9× memory | 1.52× decode | ~100% (7B) | None | ACL 2026 |
| KVCapsule | 2.4× memory | 2× TPS | Negligible loss | Light | May 2026 |
| SelKV | 3.3× decode | At 100k tokens | 25% retention | None | Jul 2026 |
| EarlyTom | 6.8× FLOPs | Vision encoder | 40.7% FLOPs | None | CVPR 2026 |
| PruneSID | 88.9% tokens | 6× FLOPs | 96.3% (1.5-7B) | None | ICLR 2026 |
| DynaKV | 6% retained | — | 94% LongBench | None | Mar 2026 |

### SPECULATIVE DECODING FOR VLMS:

| Method | Speedup | Training Required | Venue |
|---|---|---|---|
| SpecVLM (EagleVLM) | 2.5–2.9× | Yes (5 epochs) | arXiv 2509.11815 |
| SpecVLM (Video) | 90% token pruning | None | EMNLP 2025 |
| ViSpec | First meaningful VLM speedup | Yes | NeurIPS 2025 |
| HCSpec | High-efficiency cascade | — | ACL 2026 |

### EDGE DEPLOYMENT:

| Hardware | AI Perf | VLM Support | Power | Price |
|---|---|---|---|---|
| Jetson Thor | 2,070 FP4 TFLOPS | 20B at 52 tok/s, 35B at 35 tok/s | 40-130W | ~$2K |
| Jetson Orin Nano Super | — | 7B at 21.75 tok/s, 8B at 19.14 tok/s | 15W | ~$250 |
| RTX 5090 | 209.5 FP16 TFLOPS | 8B at 140 tok/s, 70B INT4 at 38 tok/s | 575W | $2K |

### MULTI-GPU SERVING:

**M* (ICLR 2026):**
- Walk Graph abstraction for composite model execution.
- Qwen3-Omni: 2.9× lower RTF, 2.7× higher throughput vs vLLM-Omni.
- V-JEPA 2 rollouts: 12.5× faster.

**SGLang (2026):**
- 25× inference performance on GB300 NVL72.
- DFlash speculative decoding, Spec V2.
- Hardware: NVIDIA, AMD, TPU, Ascend NPU.

**vLLM (v26.08):**
- NIM integration, EAGLE/MTP speculative decoding.
- Tensor/pipeline/expert/data parallelism.
