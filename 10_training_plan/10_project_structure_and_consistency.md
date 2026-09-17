# Project Structure & Consistency Audit

> Honest assessment of what exists, what needs to be built, and what's inconsistent.

---

## 1. What Actually Exists Right Now

The repo is **100% markdown documentation**. Zero code. 97 `.md` files across 12 folders.

```
multimodal_reasoner_research/
│
├── README.md                           ← repo navigation
├── MASTER_RESEARCH_PLAN.md             ← master plan
├── CAPSTONE_PAPER.md                   ← professor pitch paper
├── benchmark_report_2026.md            ← per-component benchmark data (414 lines)
├── VLM_Inference_Research_2025-2026.md ← VLM inference metrics
│
├── 00_core/                            ← 3 files: thesis, terminology, principles
├── 01_foundations/                      ← 18 files: one per research track
├── 02_architecture/                    ← 42 files: 10 layers × 3-8 components + diagrams
│   ├── 01_perception/                  ← 8 component specs
│   ├── 02_fusion/                      ← 3 component specs
│   ├── 03_state/                       ← 4 component specs
│   ├── 04_memory/                      ← 4 component specs
│   ├── 05_reasoning/                   ← 5 component specs
│   ├── 06_calibration/                 ← 4 component specs
│   ├── 07_scheduler/                   ← 4 component specs
│   ├── 08_forensics/                   ← 3 component specs
│   ├── 09_domains/                     ← 3 component specs
│   ├── 10_infrastructure/              ← 3 component specs
│   └── diagrams/                       ← 6 diagram files (60 Mermaid diagrams)
├── 03_models/                          ← 1 file: model selection matrix
├── 04_uncertainty/                     ← 2 files: confidence model, claim taxonomy
├── 05_realtime/                        ← 2 files: 30 FPS budget, optimization research
├── 06_domains/                         ← 3 files: sports, general, synthetic
├── 07_evaluation/                      ← 2 files: experiment matrix, success criteria
├── 08_implementation/                  ← 3 files: staged build, ablation, risk register
├── 09_sources/                         ← 2 files: source index, search targets
└── 10_training_plan/                   ← 10 files: training plan + infrastructure + system flow
```

**There is no `src/` directory. There is no Python code. There is no `models/` directory with weights. There is no `data/` directory. There is no `config/` directory.**

---

## 2. Consistency Audit: Training Plan vs Research

### CHECK 1: Model Selection Matrix vs Training Plan

| What Research Says (`03_models/`) | What Training Plan Says (`10_training_plan/`) | Consistent? |
|---|---|---|
| Detector: RF-DETR-S or M (3.5-4.4ms T4) | Training pipeline uses RF-DETR-S | ✅ Yes |
| Pose: DETRPose-S (2.39ms A10) | Training pipeline doesn't mention pose training | ✅ OK — pose is pretrained, no training needed |
| Tracker: ByteTrack (pretrained, no training) | Training plan doesn't train tracker | ✅ Yes |
| VLM: Qwen3-VL-30B-A3B for local reasoning | Training plan references VLM fine-tuning with QLoRA | ⚠️ Partially — training plan says "Qwen2.5-VL" in some places, should be "Qwen3-VL-30B-A3B" |
| Hypothesis ranker: GBDT (XGBoost/LightGBM) | Training plan doesn't cover this | ⚠️ Missing — hypothesis ranker training not in plan |
| Calibration: post-hoc temperature scaling | Training plan includes temperature scaling | ✅ Yes |
| Token compression: StreamingTOM + HybridKV | Training plan doesn't mention these | ⚠️ Missing — VLM optimization not in training plan |

### CHECK 2: Staged Build Plan vs Training Plan

| What Research Says (`08_implementation/`) | What Training Plan Says | Consistent? |
|---|---|---|
| 10 stages, 22-30 weeks total | 28-day execution timeline | ❌ **INCONSISTENT** — staged plan says 5-7 months, training plan says 28 days |
| Stage 1: perception backbone (2-3 weeks) | Phase 1: train detector (days 1-7) | ⚠️ Different scope — staged plan includes integration, training plan is just model training |
| Stage 6: deep reasoning (3-4 weeks) | Phase 4: VLM fine-tuning | ⚠️ Different scope |
| Total compute: 500-1,000 RTX 4090 hours | Training plan: ~$5-15 API costs | ⚠️ Different scope — staged plan includes ALL development, training plan is just model training |

### CHECK 3: 30 FPS Budget vs Training Plan

| What Research Says (`05_realtime/`) | What Training Plan Says | Consistent? |
|---|---|---|
| Critical path: 8-20ms p50 | Critical path: 8-12ms | ✅ Consistent (training plan is more specific) |
| Multi-rate: 30 FPS tracking, 5-10 Hz fast reasoning, 1-5 Hz deep | Same multi-rate design | ✅ Yes |
| RTX 4090 hardware decode >100 streams | Training plan assumes RTX 4090/5070 Ti | ✅ Yes |

### CHECK 4: Training Strategy vs Training Plan

| What Research Says (`01_foundations/18_training_strategy.md`) | What Training Plan Says | Consistent? |
|---|---|---|
| Compose pretrained modules, don't train end-to-end | Training plan trains 4 models independently | ✅ Yes — matches modular strategy |
| Detector: pretrained RF-DETR, LoRA if needed | Trains RF-DETR-S from scratch on COCO | ⚠️ Slight mismatch — research says "pretrained + LoRA", training plan says "train from scratch" |
| Hypothesis ranker: GBDT, <1 GPU day | Not in training plan | ❌ Missing |
| Confidence calibration: post-hoc, 0 training | Training plan includes calibration training | ⚠️ Calibration is post-hoc (no backbone retraining), but training plan describes training a calibration model |
| VLM: prompt engineering first, QLoRA if needed | Training plan does QLoRA fine-tuning | ✅ Consistent — training plan is the "if needed" path |
| Data: 50-200K labeled samples | Training plan uses COCO (118K), MOT17, Market1501, SoccerNet | ✅ Consistent |
| Compute: 500-1,000 RTX 4090 hours | Training plan: ~$5-15 API costs | ⚠️ Different scope — research includes ALL development |

### CHECK 5: Risk Register vs Training Plan

| Risk | Training Plan Coverage | Consistent? |
|---|---|---|
| R1: False confidence | Calibration pipeline covers this | ✅ Yes |
| R2: Temporal hallucination | Not addressed in training | ⚠️ Not covered |
| R3: Compounding state error | Not addressed in training | ⚠️ Not covered |
| R4: Scheduler starvation | Not addressed in training | ⚠️ Not covered |
| R6: Synthetic media brittleness | Not addressed in training | ⚠️ Not covered |
| R13: EU AI Act compliance | Not addressed in training | ⚠️ Not covered |

### CHECK 6: Evaluation Experiments vs Training Plan

| Experiment | Training Plan Coverage | Consistent? |
|---|---|---|
| E1: Direct observation | Benchmarking plan includes detection mAP | ✅ Yes |
| E2: Occluded-state reconstruction | Not in training plan | ❌ Missing |
| E3: Temporal event inference | Not in training plan | ❌ Missing |
| E4: Hypothesis competition | Not in training plan | ❌ Missing |
| E5: Contradictory modalities | Not in training plan | ❌ Missing |
| E6: Synthetic media robustness | Not in training plan | ❌ Missing |
| E7: Prediction-driven scheduling | Not in training plan | ❌ Missing |
| E8: 30 FPS stress test | Benchmarking plan includes latency | ✅ Yes |
| E9: Long-form memory | Not in training plan | ❌ Missing |
| E10: Distribution shift | Not in training plan | ❌ Missing |

---

## 3. Key Inconsistencies Found

### INCONSISTENCY 1: Timeline Mismatch
- **Staged build plan** (`08_implementation/`): 22-30 weeks (5-7 months) for full system
- **Training plan** (`10_training_plan/`): 28 days for training + infrastructure
- **Reality**: The training plan covers ONLY model training. The staged build plan covers the ENTIRE system development. These are different scopes. The training plan timeline is a SUBSET of the staged build plan.

### INCONSISTENCY 2: VLM Name Mismatch
- **Model selection matrix**: "Qwen3-VL-30B-A3B" (latest, MoE architecture)
- **Training plan**: References "Qwen2.5-VL" in some places
- **Fix**: Training plan should consistently reference "Qwen3-VL-30B-A3B"

### INCONSISTENCY 3: Missing Components
- **Hypothesis ranker training** (GBDT) — not in training plan
- **Fusion layer** — not in training plan
- **Memory layer** — not in training plan
- **Forensics** — not in training plan
- **Evaluation experiments** (E1-E10) — not in training plan

### INCONSISTENCY 4: Training Approach
- **Research**: "Use pretrained RF-DETR + LoRA if domain gap is large"
- **Training plan**: "Train RF-DETR-S from scratch on COCO"
- **Reality**: Training from scratch on COCO is valid for getting a baseline, but the research suggests pretrained + LoRA for domain adaptation. Both are valid, but the training plan should clarify this is the BASELINE training, not the domain adaptation.

---

## 4. Proper Project Structure

This is what the project SHOULD look like when code is written. Everything below the `src/` line is what needs to be CREATED. Everything above it already EXISTS.

```
multimodal_reasoner_research/
│
│ ═══════════════════════════════════════════════════════
│  EXISTING: Research & Documentation (97 .md files)
│ ═══════════════════════════════════════════════════════
│
├── README.md
├── MASTER_RESEARCH_PLAN.md
├── CAPSTONE_PAPER.md
├── benchmark_report_2026.md
├── VLM_Inference_Research_2025-2026.md
│
├── 00_core/                           ← thesis, terminology, principles
├── 01_foundations/                     ← 18 research tracks
├── 02_architecture/                   ← 41 component specs + diagrams
├── 03_models/                         ← model selection matrix
├── 04_uncertainty/                    ← confidence model, claim taxonomy
├── 05_realtime/                       ← 30 FPS budget, optimization
├── 06_domains/                        ← sports, general, synthetic
├── 07_evaluation/                     ← experiment matrix, success criteria
├── 08_implementation/                 ← staged build, ablation, risk register
├── 09_sources/                        ← source index
├── 10_training_plan/                  ← training plan, infrastructure, system flow
│
│ ═══════════════════════════════════════════════════════
│  TO BE CREATED: Source Code
│ ═══════════════════════════════════════════════════════
│
├── src/                               ← MAIN SOURCE CODE
│   ├── __init__.py
│   │
│   ├── ingestor/                      ← Pre-layer: video/audio input
│   │   ├── __init__.py
│   │   ├── file_ingestor.py           ← Read video files (MP4, AVI, MOV)
│   │   ├── rtsp_ingestor.py           ← Read RTSP streams
│   │   ├── audio_ingestor.py          ← Extract audio tracks
│   │   └── frame_buffer.py            ← Pre-allocated GPU frame pool
│   │
│   ├── perception/                    ← LAYER 01: Detection, tracking, pose
│   │   ├── __init__.py
│   │   ├── detector.py                ← RF-DETR-S TensorRT inference
│   │   ├── pose_estimator.py          ← DETRPose-S inference
│   │   ├── tracker.py                 ← ByteTrack wrapper
│   │   ├── segmenter.py               ← RF-DETR-Seg-S (Phase 2)
│   │   ├── ocr.py                     ← PaddleOCR (Phase 2)
│   │   ├── reid.py                    ← OSNet (Phase 2)
│   │   ├── audio_features.py          ← Whisper (Phase 2)
│   │   ├── camera_motion.py           ← optical flow (Phase 2)
│   │   └── pipeline.py                ← Perception orchestrator
│   │
│   ├── fusion/                        ← LAYER 02: Multimodal fusion
│   │   ├── __init__.py
│   │   ├── multimodal_fusion.py       ← Combine vision+audio+text
│   │   ├── temporal_fusion.py         ← Smooth across time windows
│   │   └── cross_modal_alignment.py   ← Synchronize modalities
│   │
│   ├── state/                         ← LAYER 03: World state
│   │   ├── __init__.py
│   │   ├── world_state.py             ← Ring buffer entity store
│   │   ├── entity_tracker.py          ← Multi-entity lifecycle
│   │   ├── trajectory_model.py        ← Position/velocity prediction
│   │   └── event_detector.py          ← Rule-based event triggers
│   │
│   ├── memory/                        ← LAYER 04: Memory systems
│   │   ├── __init__.py
│   │   ├── short_term.py              ← Ring buffer (current state)
│   │   ├── working_memory.py          ← Active hypothesis context
│   │   ├── long_term.py               ← SQLite + vector search (Phase 2)
│   │   └── episodic.py                ← FAISS event store (Phase 2)
│   │
│   ├── reasoning/                     ← LAYER 05: Reasoning engine
│   │   ├── __init__.py
│   │   ├── hypothesis_engine.py       ← Generate & rank hypotheses
│   │   ├── fast_verifier.py           ← Rule-based quick checks
│   │   ├── vlm_reasoner.py            ← VLM deep reasoning
│   │   ├── evidence_graph.py          ← Causal/evidential links
│   │   └── prediction_model.py        ← Future state prediction
│   │
│   ├── calibration/                   ← LAYER 06: Confidence calibration
│   │   ├── __init__.py
│   │   ├── confidence_decomposition.py ← 5-component decomposition
│   │   ├── conformal.py               ← Prediction sets
│   │   ├── temperature_scaling.py      ← Post-hoc calibration
│   │   └── claim_output.py            ← Structured claim JSON
│   │
│   ├── scheduler/                     ← LAYER 07: Job scheduling
│   │   ├── __init__.py
│   │   ├── priority_scheduler.py      ← Multi-rate priority
│   │   ├── queue_manager.py           ← Bounded queues
│   │   ├── backpressure.py            ← Overload protection
│   │   └── gpu_distributor.py         ← CUDA stream assignment
│   │
│   ├── forensics/                     ← LAYER 08: Synthetic media
│   │   ├── __init__.py
│   │   ├── deepfake_detection.py      ← Ensemble detectors (Phase 3)
│   │   ├── provenance_c2pa.py         ← C2PA verification (Phase 3)
│   │   └── audio_forensics.py         ← Audio deepfake (Phase 3)
│   │
│   ├── domains/                       ← LAYER 09: Domain specialization
│   │   ├── __init__.py
│   │   ├── sports_reasoning.py        ← Sports rules + trajectories (Phase 3)
│   │   ├── general_multimedia.py      ← Domain-agnostic fallback
│   │   └── synthetic_media.py         ← Detection + provenance (Phase 3)
│   │
│   ├── schemas/                       ← LAYER 10: Data contracts
│   │   ├── perception.proto           ← Detection, pose, tracking schemas
│   │   ├── state.proto                ← World state, entity schemas
│   │   ├── reasoning.proto            ← Hypothesis, evidence schemas
│   │   ├── calibration.proto          ← Confidence, claim schemas
│   │   └── claim.proto                ← Final claim output schema
│   │
│   ├── output/                        ← Output delivery
│   │   ├── __init__.py
│   │   ├── rest_api.py                ← FastAPI endpoints
│   │   ├── websocket_server.py        ← Live streaming
│   │   └── kafka_producer.py          ← Async message publishing
│   │
│   ├── monitoring/                    ← Observability
│   │   ├── __init__.py
│   │   ├── metrics.py                 ← Prometheus metrics
│   │   └── health.py                  ← Health checks
│   │
│   ├── pipeline.py                    ← MAIN ORCHESTRATOR
│   └── server.py                      ← ENTRY POINT
│
├── models/                            ← DOWNLOADED/FINE-TUNED MODEL WEIGHTS
│   ├── detection/
│   │   ├── rf_detr_s/                 ← RF-DETR-S TensorRT engine
│   │   │   ├── model.plan             ← TensorRT engine file
│   │   │   ├── config.yaml            ← Model config
│   │   │   └── labels.txt             ← Class labels
│   │   └── rf_detr_l/                 ← RF-DETR-L (async high-accuracy)
│   ├── pose/
│   │   ├── detrpose_s/                ← DETRPose-S TensorRT engine
│   │   │   ├── model.plan
│   │   │   └── config.yaml
│   │   └── detrpose_l/                ← DETRPose-L (async)
│   ├── tracking/
│   │   └── bytetrack/                 ← ByteTrack config (no weights needed)
│   │       └── bytetrack_s.yaml
│   ├── reid/
│   │   └── osnet/                     ← OSNet weights (Phase 2)
│   │       ├── osnet_ain_x1_0.pth
│   │       └── config.yaml
│   ├── segmentation/
│   │   └── rf_detr_seg_s/             ← RF-DETR-Seg-S (Phase 2)
│   ├── vlm/
│   │   ├── qwen3_vl_30b/             ← Qwen3-VL-30B-A3B (Phase 2)
│   │   │   ├── config.json
│   │   │   ├── model-*.safetensors
│   │   │   └── tokenizer/
│   │   └── qwen3_vl_30b_awq/         ← AWQ quantized version
│   ├── calibration/
│   │   ├── temperature_scaler.pkl     ← Learned temperature
│   │   ├── conformal_quantiles.pkl    ← Conformal prediction sets
│   │   └── calibration_set/           ← Held-out calibration data
│   └── hypothesis/
│       └── ranker/                    ← GBDT hypothesis ranker (Phase 2)
│           └── ranker.json
│
├── data/                              ← DATASETS (downloaded/preprocessed)
│   ├── coco/
│   │   ├── train2017/                 ← 118K training images
│   │   ├── val2017/                   ← 5K validation images
│   │   └── annotations/
│   ├── mot17/
│   │   ├── train/                     ← MOT17 tracking sequences
│   │   └── val/
│   ├── market1501/                    ← ReID dataset (Phase 2)
│   ├── soccernet/
│   │   ├── tracking/                  ← SoccerNet tracking (Phase 3)
│   │   ├── action_spotting/           ← SoccerNet action spotting (Phase 3)
│   │   └── foul_recognition/          ← SoccerNet foul (Phase 3)
│   └── custom/
│       ├── sports/                    ← Custom sports footage
│       └── synthetic/                 ← Deepfake test samples (Phase 3)
│
├── config/                            ← CONFIGURATION FILES
│   ├── pipeline.yaml                  ← Full pipeline config
│   ├── hardware.yaml                  ← GPU/CPU topology
│   ├── models/
│   │   ├── detector.yaml              ← Detector config
│   │   ├── pose.yaml                  ← Pose config
│   │   ├── tracker.yaml               ← Tracker config
│   │   ├── vlm.yaml                   ← VLM config
│   │   └── calibration.yaml           ← Calibration config
│   ├── domains/
│   │   ├── sports.yaml                ← Sports rules
│   │   └── general.yaml               ← General config
│   └── docker/
│       ├── Dockerfile                 ← Container build
│       └── docker-compose.yml         ← Multi-container setup
│
├── tests/                             ← TEST SUITE
│   ├── unit/                          ← Per-component tests
│   │   ├── test_detector.py
│   │   ├── test_tracker.py
│   │   ├── test_world_state.py
│   │   ├── test_event_detector.py
│   │   ├── test_hypothesis_engine.py
│   │   ├── test_fast_verifier.py
│   │   ├── test_calibration.py
│   │   └── test_claim_output.py
│   ├── integration/                   ← Pipeline integration tests
│   │   ├── test_perception_pipeline.py
│   │   ├── test_reasoning_pipeline.py
│   │   └── test_full_pipeline.py
│   └── stress/                        ← Performance tests
│       ├── test_30fps_sustained.py
│       └── test_memory_leak.py
│
├── scripts/                           ← UTILITY SCRIPTS
│   ├── download_models.py             ← Download pre-trained weights
│   ├── export_tensorrt.py             ← PyTorch → TensorRT export
│   ├── prepare_calibration_set.py     ← Create calibration dataset
│   ├── run_benchmark.py               ← Performance benchmarking
│   └── generate_report.py             ← Generate evaluation report
│
├── deployment/                        ← DEPLOYMENT
│   ├── docker/
│   │   ├── Dockerfile                 ← Production container
│   │   └── docker-compose.yml         ← Full stack
│   ├── k8s/                           ← Kubernetes manifests (Phase 3)
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── hpa.yaml
│   └── scripts/
│       ├── setup_gpu.sh               ← GPU driver setup
│       └── deploy.sh                  ← Deployment script
│
├── notebooks/                         ← JUPYTER NOTEBOOKS (exploration)
│   ├── 01_exploration.ipynb           ← Data exploration
│   ├── 02_training.ipynb              ← Model training
│   ├── 03_evaluation.ipynb            ← Results analysis
│   └── 04_visualization.ipynb         ← Pipeline visualization
│
└── docs/                              ← DOCUMENTATION
    ├── api.md                         ← API reference
    ├── architecture.md                ← Architecture overview
    ├── training.md                    ← Training guide
    ├── deployment.md                  ← Deployment guide
    └── troubleshooting.md             ← Common issues
```

---

## 5. File Count Summary

| Category | Existing | To Create | Total |
|---|---|---|---|
| Research docs (`.md`) | 97 | 5 | 102 |
| Source code (`.py`, `.proto`) | 0 | ~65 | ~65 |
| Config files (`.yaml`, `.json`) | 0 | ~15 | ~15 |
| Test files (`.py`) | 0 | ~15 | ~15 |
| Scripts (`.py`, `.sh`) | 0 | ~7 | ~7 |
| Docker/Deploy | 0 | ~5 | ~5 |
| Notebooks (`.ipynb`) | 0 | ~4 | ~4 |
| **Total files** | **97** | **~111** | **~208** |

---

## 6. Build Order (Consistent with All Research)

Phase 1 (Weeks 1-4): Core pipeline
```
Step 1:  src/ingestor/file_ingestor.py         ← Read video
Step 2:  src/perception/detector.py             ← Detect objects
Step 3:  src/perception/tracker.py              ← Track objects
Step 4:  src/perception/pose_estimator.py       ← Estimate pose
Step 5:  src/perception/pipeline.py             ← Full perception
Step 6:  src/state/world_state.py               ← Ring buffer
Step 7:  src/state/event_detector.py            ← Detect events
Step 8:  src/scheduler/priority_scheduler.py    ← Schedule jobs
Step 9:  src/reasoning/hypothesis_engine.py     ← Generate hypotheses
Step 10: src/reasoning/fast_verifier.py         ← Quick verification
Step 11: src/memory/short_term.py               ← Store snapshots
Step 12: src/pipeline.py                        ← Connect everything
Step 13: tests/unit/test_core_pipeline.py       ← Verify it works
```

Phase 2 (Weeks 5-8): Reasoning + Calibration
```
Step 14: src/reasoning/vlm_reasoner.py          ← VLM reasoning
Step 15: src/reasoning/evidence_graph.py        ← Evidence tracking
Step 16: src/calibration/confidence_decomposition.py ← Confidence
Step 17: src/calibration/conformal.py           ← Prediction sets
Step 18: src/calibration/temperature_scaling.py ← Calibration
Step 19: src/calibration/claim_output.py        ← Structured output
Step 20: src/memory/working_memory.py           ← Active context
Step 21: src/output/rest_api.py                 ← REST endpoints
Step 22: tests/integration/test_reasoning.py    ← Verify reasoning
```

Phase 3 (Weeks 9-12): Async + Optimization
```
Step 23: src/perception/segmenter.py            ← Segmentation
Step 24: src/perception/ocr.py                  ← OCR
Step 25: src/perception/reid.py                 ← Re-identification
Step 26: src/perception/audio_features.py       ← Audio
Step 27: src/fusion/multimodal_fusion.py        ← Fusion
Step 28: src/fusion/temporal_fusion.py          ← Temporal smoothing
Step 29: src/scheduler/queue_manager.py         ← Queue management
Step 30: src/scheduler/backpressure.py          ← Overload protection
Step 31: src/scheduler/gpu_distributor.py       ← GPU distribution
Step 32: src/monitoring/metrics.py              ← Prometheus metrics
Step 33: src/monitoring/health.py               ← Health checks
Step 34: tests/stress/test_30fps.py             ← Performance test
```

Phase 4 (Weeks 13-16): Domains + Forensics
```
Step 35: src/domains/sports_reasoning.py        ← Sports rules
Step 36: src/domains/general_multimedia.py      ← Generic domain
Step 37: src/domains/synthetic_media.py         ← Synthetic detection
Step 38: src/forensics/deepfake_detection.py    ← Deepfake ensemble
Step 39: src/forensics/provenance_c2pa.py       ← C2PA verification
Step 40: src/forensics/audio_forensics.py       ← Audio forensics
Step 41: src/memory/long_term.py                ← Long-term memory
Step 42: src/memory/episodic.py                 ← Episodic memory
Step 43: src/reasoning/prediction_model.py      ← Future prediction
Step 44: src/output/websocket_server.py         ← Live streaming
Step 45: src/output/kafka_producer.py           ← Async messaging
Step 46: tests/integration/test_full_pipeline.py ← Full system test
```

Phase 5 (Weeks 17-20): Production
```
Step 47:  config/ (all YAML files)
Step 48:  src/schemas/ (all .proto files)
Step 49:  deployment/ (Docker, K8s)
Step 50:  scripts/ (utility scripts)
Step 51:  notebooks/ (exploration)
Step 52:  docs/ (documentation)
Step 53:  End-to-end testing on real video
Step 54:  Performance optimization
Step 55:  Security audit
```

---

## 7. What to Do Next

1. **Fix inconsistencies in training plan** — update VLM name, add missing components
2. **Start coding Phase 1** — begin with `src/ingestor/file_ingestor.py`
3. **Set up project structure** — create all directories and `__init__.py` files
4. **Download models** — get RF-DETR-S, DETRPose-S, ByteTrack configs
5. **Write first test** — verify file_ingestor reads video correctly
