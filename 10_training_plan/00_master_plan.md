# Training Master Plan

> This is the complete execution plan for building, training, and deploying the Multimodal Video Reasoner. Every step is detailed. Nothing is assumed. This document is the single source of truth for what we do, how we do it, and why.

---

## The Goal

We are building a **real-time multimodal video reasoning system** that:
1. Processes live video at 30 FPS
2. Detects and tracks objects with production accuracy
3. Reasons about events using vision-language models
4. Outputs structured claims with calibrated confidence
5. Runs on a single GPU (RTX 4090 or equivalent provided by professor)

This is NOT a research prototype. This is a **production-quality demonstration** that showcases every component working together at real-time speed with proper uncertainty quantification.

---

## What We're Training (Model Inventory)

We are NOT training everything from scratch. We are:

| Component | Strategy | Why |
|---|---|---|
| **Object Detection** | Fine-tune RF-DETR-S on custom data | Pre-trained on COCO, fine-tune for our domain |
| **Pose Estimation** | Use pre-trained DETRPose-S, no fine-tuning | Already 67.0 AP on COCO, sufficient for our needs |
| **Object Tracking** | Use ByteTrack as-is, tune parameters | Algorithm-based, no learning needed |
| **Segmentation** | Use RF-DETR-Seg-S pre-trained | May fine-tune if time permits |
| **OCR** | Use PaddleOCR v4 as-is | 3% WER, no fine-tuning needed |
| **Re-ID** | Fine-tune OSNet on custom person data | Domain-specific appearance features |
| **Audio (Whisper)** | Use Whisper-large-v3 as-is | 3% WER, no fine-tuning needed |
| **Camera Motion** | ORB+RANSAC, no model | Classical CV, no learning |
| **VLM Reasoning** | Prompt engineering + few-shot | Use Qwen3-VL or API, no weight updates |
| **Confidence Calibration** | Train calibrator on validation set | Temperature scaling + conformal prediction |
| **Deepfake Detection** | Use pre-trained ensemble | CLIP + EVA-02 + SRM, no fine-tuning |
| **C2PA/SynthID** | Use existing tools | No training, just integration |

### Models We Actually Fine-Tune (Sequential, One at a Time)

| Priority | Model | Dataset | Duration | GPU |
|---|---|---|---|---|
| 1st | RF-DETR-S (detection) | COCO + custom domain | 2-3 days | Full GPU |
| 2nd | OSNet (Re-ID) | Market1501 + custom | 1-2 days | Full GPU |
| 3rd | RF-DETR-Seg-S (segmentation) | COCO + custom | 2-3 days | Full GPU |
| 4th | Calibrator (temperature + conformal) | Held-out validation | 2-4 hours | CPU |

**Total training time: 7-9 days** (sequential on one GPU)

Everything else uses pre-trained weights or classical algorithms.

---

## Execution Phases

### Phase 0: Pre-Training Preparation (Days 1-3)
**Goal:** Environment ready, data collected, baseline established

```
01_pre_training_preparation.md →
  - Install CUDA, PyTorch, dependencies
  - Download pre-trained model weights
  - Set up experiment tracking (W&B or MLflow)
  - Verify GPU benchmarks match published numbers
  - Set up Docker environment
  - Clone and test NVIDIA DeepStream (if available)
```

### Phase 1: Dataset Preparation (Days 2-5)
**Goal:** All datasets downloaded, preprocessed, split, and versioned

```
02_dataset_preparation.md →
  - Download COCO 2017 (detection + segmentation)
  - Download MOT17/MOT20 (tracking)
  - Download Market1501 (Re-ID)
  - Download SoccerNet v2 (sports domain)
  - Collect custom domain video (10-20 sports clips)
  - Preprocess all datasets to standard format
  - Create train/val/test splits
  - Verify data quality (check for corruption, label errors)
```

### Phase 2: Model Fine-Tuning (Days 5-14)
**Goal:** All models fine-tuned, benchmarked, and exported

```
03_training_pipeline.md →
  - Fine-tune RF-DETR-S on COCO + custom domain
  - Fine-tune OSNet on Market1501 + custom
  - Fine-tune RF-DETR-Seg-S on COCO + custom
  - Export all models to TensorRT/ONNX for inference
  - Benchmark each model individually
```

### Phase 3: Integration (Days 14-21)
**Goal:** All components connected, pipeline runs end-to-end

```
05_inference_integration.md →
  - Connect perception pipeline (detection → pose → tracking)
  - Connect state management (ring buffer, entity tracker)
  - Connect reasoning pipeline (hypothesis → fast verify → VLM)
  - Connect calibration (confidence → conformal → claim output)
  - Connect scheduler (multi-rate, backpressure)
  - End-to-end pipeline test on sample video
```

### Phase 4: Benchmarking & Demo (Days 21-28)
**Goal:** Full evaluation, polished demo, final paper

```
06_benchmarking_plan.md →
  - Per-component latency benchmarks
  - End-to-end latency measurement
  - Accuracy evaluation on test set
  - Calibration quality (ECE, coverage)
  - Stress test (multiple streams, high entity count)
  - Demo video recording
  - Final paper with results
```

---

## Hardware Utilization Strategy

### RTX 4090 (24GB) — Primary Training GPU

```
Training Phase:
  GPU Memory: 23GB allocated
  - Model weights: varies
  - Optimizer states: 2-4x model size
  - Activations: depends on batch size
  - Gradient buffers: 1x model size
  Training runs: ONE MODEL AT A TIME
  Background: Nothing else on GPU during training

Inference Phase:
  GPU Memory: ~8GB allocated
  - RF-DETR-S: 1.5GB
  - DETRPose-S: 0.5GB
  - ByteTrack: 0.05GB
  - CUDA context: 1GB
  - Working space: 2GB
  Total: ~5GB, leaves headroom for VLM inference
```

### If Only RTX 5070 Ti (16GB) Available

```
Training Phase:
  - Reduce batch size
  - Use gradient accumulation
  - Use mixed precision (FP16/BF16)
  - One model at a time, no exceptions
  - May need to reduce RF-DETR input resolution during training

Inference Phase:
  - Perception pipeline: ~3GB
  - VLM: API fallback (GPT-4.1 / Gemini Flash)
  - OR local Qwen2.5-VL-7B (fits in remaining 13GB)
```

---

## Quality Bar

This is not "make it work." This is "make it impressive."

### Minimum Viable Demo
- [ ] Detection running at 30 FPS on live video
- [ ] Tracking maintaining consistent IDs
- [ ] At least 3 event types detected (goal, foul, scene cut)
- [ ] VLM producing structured reasoning for each event
- [ ] Confidence scores reported with decomposition
- [ ] Prediction sets with coverage guarantee
- [ ] Claims output as structured JSON

### Impressive Demo (Target)
- [ ] All of the above PLUS:
- [ ] Real-time sports analysis (soccer match)
- [ ] Deepfake detection on sample synthetic videos
- [ ] C2PA provenance verification
- [ ] Live dashboard showing entity tracking + claims
- [ ] Latency overlay showing real-time FPS
- [ ] Comparison: our system vs naive VLM-every-frame approach
- [ ] Ablation: with/without calibration, with/without fast verifier

### Benchmark Report
- [ ] Per-component latency (p50, p95, p99)
- [ ] End-to-end pipeline latency
- [ ] Detection accuracy vs baseline
- [ ] Calibration ECE < 0.05
- [ ] Conformal coverage ≥ 95% at α=0.05
- [ ] GPU utilization during operation
- [ ] Memory usage over time (no leaks)
- [ ] Throughput (frames per second sustained)

---

## File Map

| File | Content |
|---|---|
| `00_master_plan.md` | This file — overview and strategy |
| `01_pre_training_preparation.md` | Environment setup, dependencies, verification |
| `02_dataset_preparation.md` | Every dataset, every preprocessing step |
| `03_training_pipeline.md` | Exact training configs for each model |
| `04_optimization_strategies.md` | Mixed precision, TensorRT, quantization |
| `05_inference_integration.md` | How models connect into the pipeline |
| `06_benchmarking_plan.md` | How we measure everything |
| `07_timeline_milestones.md` | Day-by-day execution plan |

---

## Non-Negotiable Rules

1. **Benchmark before optimizing.** Never guess at performance. Measure first.
2. **One model at a time on GPU.** No concurrent training runs.
3. **Version everything.** Every model checkpoint, every dataset version, every config.
4. **Reproducible results.** Seed everything. Log everything.
5. **TensorRT for inference.** No raw PyTorch in the critical path.
6. **No silent failures.** Every component must have error handling.
7. **Latency budgets are law.** If a component exceeds its budget, degrade or skip.
