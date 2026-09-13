# Multimodal Video Reasoner

> A real-time multimodal video reasoning system with calibrated confidence, prediction sets, and production-grade architecture. Capstone project.

## What This Is

A system that watches live video, detects and tracks objects, reasons about events using vision-language models, and outputs **structured claims** with calibrated confidence scores, prediction sets, and provenance metadata — all at 30 FPS on a single consumer GPU.

## Quick Start for Teammates

**If you have 5 minutes**, read these files in order:

1. `CAPSTONE_PAPER.md` — What we're building, why, and how (the pitch paper for our professor)
2. `02_architecture/FULL_ARCHITECTURE.md` — The complete system architecture in one file
3. `README.md` (this file) — Navigate the rest of the repo

**If you have 30 minutes**, also read:

4. `02_architecture/00_architecture_index.md` — Index of all 41 components
5. `02_architecture/diagrams/04_master_architecture.md` — Full system diagram (all layers, all connections)
6. `benchmark_report_2026.md` — Per-component benchmark data

**If you have 2 hours**, read:

7. `01_foundations/` — Pick the track relevant to your work (18 tracks)
8. `02_architecture/` — Read the component files for your layer

---

## Repository Structure

```
multimodal_reasoner_research/
│
├── CAPSTONE_PAPER.md                 ← START HERE: Professor pitch paper
├── MASTER_RESEARCH_PLAN.md           ← Research coordination plan
├── README.md                         ← This file
├── benchmark_report_2026.md          ← Per-component benchmark data
├── VLM_Inference_Research_2025-2026.md ← VLM inference metrics
│
├── 00_core/                          ← Project foundation
│   ├── 01_research_thesis.md         ← Central thesis and research question
│   ├── 02_terminology.md             ← Key terms and definitions
│   └── 03_system_principles.md       ← Design principles
│
├── 01_foundations/                    ← 18 research tracks (read these to understand the "why")
│   ├── 01_fundamental_reasoning.md
│   ├── 02_latent_world_models.md
│   ├── 03_hidden_state_reconstruction.md
│   ├── 04_temporal_reasoning.md
│   ├── 05_memory_architecture.md
│   ├── 06_perception_stack.md
│   ├── 07_multimodal_fusion.md
│   ├── 08_evidence_representation.md
│   ├── 09_hypothesis_engine.md
│   ├── 10_uncertainty_calibration.md
│   ├── 11_prediction_anticipation.md
│   ├── 12_physics_biomechanics.md
│   ├── 13_causal_counterfactual_reasoning.md
│   ├── 14_synthetic_media_forensics.md
│   ├── 15_realtime_inference.md
│   ├── 16_multirate_scheduling.md
│   ├── 17_evaluation_benchmarks.md
│   └── 18_training_strategy.md
│
├── 02_architecture/                  ← THE CORE: System architecture (read this to understand the "how")
│   ├── FULL_ARCHITECTURE.md          ← ★ SINGLE FILE: Complete architecture (3,096 lines)
│   ├── 00_architecture_index.md      ← Index of all 41 components
│   ├── 01_reference_architecture.md  ← Reference architecture overview
│   ├── 02_core_data_contracts.md     ← JSON/Protobuf message schemas
│   ├── 03_scheduler_design.md        ← Multi-rate scheduler design
│   │
│   ├── 01_perception/                ← Layer 01: Perception (8 components)
│   │   ├── 01_detection.md           ← RF-DETR detection pipeline
│   │   ├── 02_pose_estimation.md     ← DETRPose body pose
│   │   ├── 03_tracking.md            ← ByteTrack object tracking
│   │   ├── 04_segmentation.md        ← RF-DETR-Seg / SAM 3 masks
│   │   ├── 05_ocr.md                 ← PaddleOCR text extraction
│   │   ├── 06_reid.md                ← OSNet re-identification
│   │   ├── 07_audio_features.md      ← Whisper audio processing
│   │   └── 08_camera_motion.md       ← ORB+RANSAC ego-motion
│   │
│   ├── 02_fusion/                    ← Layer 02: Fusion (3 components)
│   │   ├── 01_multimodal_fusion.md
│   │   ├── 02_temporal_fusion.md
│   │   └── 03_cross_modal_alignment.md
│   │
│   ├── 03_state/                     ← Layer 03: State (4 components)
│   │   ├── 01_world_state.md         ← Ring buffer world state
│   │   ├── 02_entity_tracker.md      ← Entity lifecycle
│   │   ├── 03_trajectory_model.md    ← Kalman filter prediction
│   │   └── 04_event_detection.md     ← R score event triggers
│   │
│   ├── 04_memory/                    ← Layer 04: Memory (4 components)
│   │   ├── 01_short_term.md          ← Ring buffer (30 frames)
│   │   ├── 02_working_memory.md      ← Active hypotheses
│   │   ├── 03_long_term.md           ← SQLite + FAISS
│   │   └── 04_episodic_memory.md     ← Episode store
│   │
│   ├── 05_reasoning/                 ← Layer 05: Reasoning (5 components)
│   │   ├── 01_hypothesis_engine.md   ← Rule-based hypothesis gen
│   │   ├── 02_fast_verifier.md       ← 1-5ms rule verification
│   │   ├── 03_deep_vlm_reasoner.md   ← Qwen3-VL deep reasoning
│   │   ├── 04_evidence_graph.md      ← Typed evidence graph
│   │   └── 05_prediction_model.md    ← Trajectory extrapolation
│   │
│   ├── 06_calibration/               ← Layer 06: Calibration (4 components)
│   │   ├── 01_confidence_decomposition.md ← 5-component confidence
│   │   ├── 02_conformal_prediction.md    ← Prediction sets
│   │   ├── 03_temperature_scaling.md     ← Post-hoc calibration
│   │   └── 04_claim_output.md            ← Structured claim assembly
│   │
│   ├── 07_scheduler/                 ← Layer 07: Scheduler (4 components)
│   │   ├── 01_multi_rate_scheduler.md
│   │   ├── 02_queue_management.md
│   │   ├── 03_backpressure.md
│   │   └── 04_gpu_work_distribution.md
│   │
│   ├── 08_forensics/                 ← Layer 08: Forensics (3 components)
│   │   ├── 01_deepfake_detection.md
│   │   ├── 02_provenance_c2pa.md
│   │   └── 03_audio_forensics.md
│   │
│   ├── 09_domains/                   ← Layer 09: Domains (3 components)
│   │   ├── 01_sports_reasoning.md
│   │   ├── 02_general_multimedia.md
│   │   └── 03_synthetic_media.md
│   │
│   ├── 10_infrastructure/            ← Layer 10: Infrastructure (3 components)
│   │   ├── 01_data_schemas.md
│   │   ├── 02_hardware_topology.md
│   │   └── 03_deployment.md
│   │
│   └── diagrams/                     ← 60 Mermaid diagrams
│       ├── 00_system_diagrams.md     ← 15 system-level diagrams
│       ├── 01_db_schema_diagrams.md  ← 15 database schema diagrams
│       ├── 02_api_protobuf_diagrams.md ← 10 API/protobuf diagrams
│       ├── 03_complete_reference.md  ← Master reference + storage estimates
│       ├── 04_master_architecture.md ← Full system (all layers, all connections)
│       └── 05_complete_system.md     ← Pipeline + state machines + deployment
│
├── 03_models/                        ← Model selection research
│   └── 01_model_selection_matrix.md
│
├── 04_uncertainty/                   ← Confidence and calibration
│   ├── 01_confidence_model.md
│   └── 02_claim_taxonomy.md
│
├── 05_realtime/                      ← Real-time execution
│   ├── 01_30fps_budget.md
│   └── 02_optimization_research.md
│
├── 06_domains/                       ← Domain-specific reasoning
│   ├── 01_sports_reasoning.md
│   ├── 02_general_multimedia.md
│   └── 03_synthetic_media.md
│
├── 07_evaluation/                    ← Evaluation plan
│   ├── 01_experiment_matrix.md
│   └── 02_success_criteria.md
│
├── 08_implementation/                ← Implementation plan
│   ├── 01_staged_build_plan.md
│   ├── 02_ablation_plan.md
│   └── 03_risk_register.md
│
└── 09_sources/                       ← 139 research sources
    ├── README.md                     ← Source index (18 categories)
    └── 02_search_targets.md
```

---

## How to Navigate by Role

### If you're working on **Perception** (Detection, Pose, Tracking)
```
Read: 02_architecture/01_perception/01_detection.md
      02_architecture/01_perception/02_pose_estimation.md
      02_architecture/01_perception/03_tracking.md
Then: 01_foundations/06_perception_stack.md
      benchmark_report_2026.md (perception section)
```

### If you're working on **Reasoning** (Hypothesis, VLM, Evidence)
```
Read: 02_architecture/05_reasoning/01_hypothesis_engine.md
      02_architecture/05_reasoning/02_fast_verifier.md
      02_architecture/05_reasoning/03_deep_vlm_reasoner.md
Then: 01_foundations/09_hypothesis_engine.md
      01_foundations/08_evidence_representation.md
```

### If you're working on **Calibration** (Confidence, Conformal Prediction)
```
Read: 02_architecture/06_calibration/01_confidence_decomposition.md
      02_architecture/06_calibration/02_conformal_prediction.md
      02_architecture/06_calibration/03_temperature_scaling.md
Then: 01_foundations/10_uncertainty_calibration.md
      04_uncertainty/01_confidence_model.md
```

### If you're working on **Scheduler / Infrastructure**
```
Read: 02_architecture/07_scheduler/01_multi_rate_scheduler.md
      02_architecture/07_scheduler/02_queue_management.md
      02_architecture/10_infrastructure/02_hardware_topology.md
      02_architecture/10_infrastructure/03_deployment.md
Then: 01_foundations/16_multirate_scheduling.md
```

### If you're working on **Forensics** (Deepfake, C2PA, Audio)
```
Read: 02_architecture/08_forensics/01_deepfake_detection.md
      02_architecture/08_forensics/02_provenance_c2pa.md
      02_architecture/08_forensics/03_audio_forensics.md
Then: 01_foundations/14_synthetic_media_forensics.md
```

### If you're writing the **paper / presentation**
```
Read: CAPSTONE_PAPER.md                     ← The pitch paper
      02_architecture/FULL_ARCHITECTURE.md  ← Complete architecture
      02_architecture/diagrams/04_master_architecture.md ← System diagrams
      benchmark_report_2026.md              ← Benchmark data for tables
```

---

## System Overview (TL;DR)

```
Video (30 FPS) + Audio + Metadata
         │
         ▼
┌─────────────────────────┐
│  PERCEPTION (<20ms)     │  Detection, Pose, Tracking, Segmentation
│  RF-DETR-S + DETRPose   │  ByteTrack, PaddleOCR, OSNet, Whisper
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  STATE (<1ms)           │  World State Ring Buffer (30 frames)
│  Entity Tracker         │  Trajectory Model (Kalman)
│  Event Detection        │  R Score → Scheduler Priority
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  REASONING              │  Fast Path: Rules (1-5ms) → 80% resolved
│  Hypothesis Engine      │  Deep Path: VLM (800-3500ms) → 20% inconclusive
│  Evidence Graph         │  Typed evidence with log-likelihood ratios
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  CALIBRATION (<8ms)     │  5-component confidence decomposition
│  Conformal Prediction   │  Prediction sets (95% coverage guarantee)
│  Temperature Scaling    │  Staleness tags, provenance metadata
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  OUTPUT                 │  Structured Claim (JSON/Protobuf)
│  + Confidence           │  + Evidence References
│  + Prediction Sets      │  + Staleness + Epistemic Status
└─────────────────────────┘
```

**Key design principle:** The 30 FPS critical path (8-12ms) never waits on deep reasoning. VLM inference runs asynchronously on event-triggered snapshots.

---

## Hardware Requirements

| Tier | GPU | VRAM | What Runs |
|---|---|---|---|
| **Minimum** | RTX 5070 Ti | 16GB | Perception (3GB) + API-based VLM reasoning |
| **Recommended** | RTX 4090 | 24GB | Perception + local VLM (Qwen3-VL-30B FP8) |
| **Production** | 2× A100 | 160GB | Full system with redundancy |

**For this capstone:** Partner's friend's RTX 5070 Ti. Sequential model training (one at a time, even if days per model).

---

## Key Numbers

| Metric | Value | Source |
|---|---|---|
| Critical path latency | 8-12ms | RF-DETR-S + DETRPose-S benchmarks |
| Fast reasoning latency | 6-20ms | Rule-based verification |
| Deep reasoning latency | 800-3500ms | VLM inference (async) |
| End-to-end fast path | 17-40ms | Perception + fast reasoning + calibration |
| Detection AP50:95 | 53.0 | RF-DETR-S on COCO val2017 |
| Pose AP50:95 | 67.0 | DETRPose-S on COCO val2017 |
| Tracking HOTA | 67.7 | ByteTrack on MOT17 |
| VLM throughput | 90 tok/s | Qwen3-VL-30B FP8 on RTX 4090D |
| Total components | 41 | Across 10 layers |
| Research sources | 139 | Verified, 2024-2026 |

---

## Team Coordination

### File Ownership

| Area | Owner | Key Files |
|---|---|---|
| Perception | _assign_ | `02_architecture/01_perception/` |
| Reasoning | _assign_ | `02_architecture/05_reasoning/` |
| Calibration | _assign_ | `02_architecture/06_calibration/` |
| Scheduler | _assign_ | `02_architecture/07_scheduler/` |
| Forensics | _assign_ | `02_architecture/08_forensics/` |
| Paper | _assign_ | `CAPSTONE_PAPER.md` |

### Workflow

1. **Architecture is decided** — read `FULL_ARCHITECTURE.md` before changing anything
2. **Interfaces are typed** — every component has defined input/output schemas
3. **Latency budgets are hard** — if your component exceeds budget, degrade gracefully
4. **No shared mutable state** — use lock-free ring buffers, typed messages
5. **Test against benchmarks** — `benchmark_report_2026.md` has the numbers

---

## Common Tasks

### "I need to understand the full system"
→ Read `02_architecture/FULL_ARCHITECTURE.md` (single file, 3,096 lines)

### "I need to see the diagrams"
→ Read `02_architecture/diagrams/` (6 files, 60 Mermaid diagrams)

### "I need to check if a model fits on our GPU"
→ Check `02_architecture/01_perception/01_detection.md` (memory table)
→ Check `benchmark_report_2026.md` (VRAM requirements)

### "I need to understand the data flow"
→ Read `02_architecture/FULL_ARCHITECTURE.md` Section 14 (Complete Data Flow)
→ Or `02_architecture/diagrams/04_master_architecture.md`

### "I need to understand the calibration framework"
→ Read `02_architecture/06_calibration/01_confidence_decomposition.md`
→ Read `02_architecture/06_calibration/02_conformal_prediction.md`

### "I need the benchmark numbers for the paper"
→ Read `benchmark_report_2026.md`
→ Read `VLM_Inference_Research_2025-2026.md`

### "I need to understand the scheduler"
→ Read `02_architecture/07_scheduler/01_multi_rate_scheduler.md`
→ Read `02_architecture/03_scheduler_design.md`

---

## File Statistics

| Category | Files | Lines |
|---|---|---|
| Architecture (components) | 45 | ~8,000 |
| Architecture (diagrams) | 6 | ~2,500 |
| Architecture (reference) | 4 | ~1,500 |
| Research foundations | 18 | ~3,000 |
| Capstone paper | 1 | 464 |
| Benchmarks & research | 3 | ~1,500 |
| Implementation plan | 3 | ~500 |
| Evaluation plan | 2 | ~300 |
| **Total** | **93** | **~16,484** |

---

## License

Internal capstone project. Do not distribute without team approval.
