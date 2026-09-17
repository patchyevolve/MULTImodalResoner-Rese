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
| **Object Detection** | Fine-tune RF-DETR-S on COCO | Baseline training on COCO. If domain gap is large (sports), LoRA fine-tune on domain data in Phase 2+ (`01_foundations/18_training_strategy.md`) |
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

| Priority | Model | Dataset | Duration | GPU | Architecture Ref |
|---|---|---|---|---|---|
| 1st | RF-DETR-S (detection) | COCO + custom domain | 2-3 days | Full GPU | `02_architecture/01_perception/01_detection.md` |
| 2nd | OSNet (Re-ID) | Market1501 + custom | 1-2 days | Full GPU | `02_architecture/01_perception/06_reid.md` |
| 3rd | RF-DETR-Seg-S (segmentation) | COCO + custom | 2-3 days | Full GPU | `02_architecture/01_perception/04_segmentation.md` |
| 4th | GBDT Hypothesis Ranker | Labeled event-hypothesis pairs | 2-4 hours | CPU | `02_architecture/05_reasoning/01_hypothesis_engine.md` |
| 5th | Calibrator (temperature + conformal) | Held-out validation | 2-4 hours | CPU | `02_architecture/06_calibration/03_temperature_scaling.md` |

**Total training time: 7-9 days** (sequential on one GPU for GPU models, CPU models same day)

Everything else uses pre-trained weights or classical algorithms.

### Components That Use Pre-Trained Weights (No Fine-Tuning)

| Component | Model | Source | Architecture Ref |
|---|---|---|---|
| Pose Estimation | DETRPose-S | Pre-trained, 67.0 AP COCO | `02_architecture/01_perception/02_pose_estimation.md` |
| Object Tracking | ByteTrack | Algorithm, no weights | `02_architecture/01_perception/03_tracking.md` |
| OCR | PaddleOCR v4 | Pre-trained, 3% WER | `02_architecture/01_perception/05_ocr.md` |
| Audio | Whisper-large-v3 | Pre-trained, 3% WER | `02_architecture/01_perception/07_audio_features.md` |
| Camera Motion | ORB+RANSAC | Classical CV | `02_architecture/01_perception/08_camera_motion.md` |
| Deepfake Detection | CLIP+EVA-02+SRM ensemble | Pre-trained | `02_architecture/08_forensics/01_deepfake_detection.md` |
| C2PA/SynthID | Existing tools | Integration only | `02_architecture/08_forensics/02_provenance_c2pa.md` |

### Components That Are Code-Only (No Training)

| Component | Strategy | Architecture Ref |
|---|---|---|
| Multi-Modal Fusion | Weighted combination, rule-based | `02_architecture/02_fusion/01_multimodal_fusion.md` |
| Temporal Fusion | Exponential moving average | `02_architecture/02_fusion/02_temporal_fusion.md` |
| Cross-Modal Alignment | Timestamp matching | `02_architecture/02_fusion/03_cross_modal_alignment.md` |
| World State | Ring buffer, typed store | `02_architecture/03_state/01_world_state.md` |
| Entity Tracker | Lifecycle state machine | `02_architecture/03_state/02_entity_tracker.md` |
| Trajectory Model | Kalman filter | `02_architecture/03_state/03_trajectory_model.md` |
| Event Detection | Rule-based triggers | `02_architecture/03_state/04_event_detection.md` |
| Short-Term Memory | Ring buffer | `02_architecture/04_memory/01_short_term.md` |
| Working Memory | Top-K active hypotheses | `02_architecture/04_memory/02_working_memory.md` |
| Long-Term Memory | SQLite + vector search | `02_architecture/04_memory/03_long_term.md` |
| Episodic Memory | FAISS event store | `02_architecture/04_memory/04_episodic_memory.md` |
| Fast Verifier | Rule-based checks | `02_architecture/05_reasoning/02_fast_verifier.md` |
| VLM Reasoner | Prompt engineering + few-shot | `02_architecture/05_reasoning/03_deep_vlm_reasoner.md` |
| Evidence Graph | Graph construction | `02_architecture/05_reasoning/04_evidence_graph.md` |
| Prediction Model | Kalman + learned residual | `02_architecture/05_reasoning/05_prediction_model.md` |
| Confidence Decomposition | Weighted combination | `02_architecture/06_calibration/01_confidence_decomposition.md` |
| Conformal Prediction | Nonconformity scores | `02_architecture/06_calibration/02_conformal_prediction.md` |
| Claim Output | JSON assembly | `02_architecture/06_calibration/04_claim_output.md` |
| Scheduler | Priority queue | `02_architecture/07_scheduler/01_multi_rate_scheduler.md` |
| Queue Manager | Bounded deque | `02_architecture/07_scheduler/02_queue_management.md` |
| Backpressure | Threshold triggers | `02_architecture/07_scheduler/03_backpressure.md` |
| GPU Distributor | CUDA stream assignment | `02_architecture/07_scheduler/04_gpu_work_distribution.md` |
| Sports Reasoning | Domain rules + trajectories | `02_architecture/09_domains/01_sports_reasoning.md` |
| General Multimedia | Domain-agnostic fallback | `02_architecture/09_domains/02_general_multimedia.md` |
| Synthetic Media | Detection + provenance | `02_architecture/09_domains/03_synthetic_media.md` |

---

## Execution Phases

> **Important scope clarification:** This training plan covers model training + integration + demo (28 days). The full system development (including all 41 architecture components, domains, forensics, production deployment) follows the staged build plan in `08_implementation/01_staged_build_plan.md` (22-30 weeks). This plan is the FIRST 4 weeks of that larger effort.

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
  - OR local Qwen3-VL-30B-A3B FP8 (fits in remaining 13GB if 4090D 48GB, else API fallback)
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

## Evaluation Experiments Mapping

> How this training plan maps to the formal experiment matrix in `07_evaluation/01_experiment_matrix.md`.

| Experiment | Description | Training Plan Coverage | Target |
|---|---|---|---|
| **E1** | Direct observation (perception accuracy) | RF-DETR-S training + benchmarking | Within 2% of COCO baselines |
| **E2** | Occluded-state reconstruction | Not in this plan (Phase 2+ from `08_implementation/`) | <40% degradation at 50% occlusion |
| **E3** | Temporal event inference | Event detection + fast verifier | Top-3 recall ≥85% at 2s horizon |
| **E4** | Hypothesis competition | Hypothesis engine + GBDT ranker | recall@3 ≥85% for correct hypothesis |
| **E5** | Contradictory modalities | Fusion layer + confidence decomposition | Audio-contradiction claims downweighted |
| **E6** | Synthetic media robustness | Not in this plan (Phase 4 from `08_implementation/`) | Inconclusive on >80% attacked samples |
| **E7** | Prediction-driven scheduling | Scheduler integration | Event-triggered ≥15% better than fixed-rate |
| **E8** | 30 FPS stress test | Pipeline integration + benchmarking | 30 FPS sustained, p99 <33ms |
| **E9** | Long-term memory | Not in this plan (Phase 2+ from `08_implementation/`) | State continuity across 30+ min sessions |
| **E10** | Distribution shift | Not in this plan (Phase 5 from `08_implementation/`) | <10pp degradation across 3+ domain shifts |

**Experiments covered in this plan:** E1, E3, E4, E7, E8 (5 of 10)
**Experiments deferred to later phases:** E2, E5, E6, E9, E10 (5 of 10)

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
