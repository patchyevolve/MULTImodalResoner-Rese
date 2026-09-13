# Architecture Index

> **Approach:** Bottom-up composition. Each small architecture defines a self-contained component with interfaces, contracts, and latency targets. The full system is composed from these pieces.
>
> **Consistency rule:** Every architecture file follows the same template: Purpose → Interfaces → Data Contracts → Latency Targets → Dependencies → Reality Check → Failure Modes.

---

## Layer 01 — Perception

| File | Component | Purpose | Latency Target |
|------|-----------|---------|----------------|
| `01_perception/01_detection.md` | Detection Pipeline | Detect objects in frames | 2.3–6.8ms per frame |
| `01_perception/02_pose_estimation.md` | Pose Estimation | 2D/3D body pose | 2.39–9.7ms per frame |
| `01_perception/03_tracking.md` | Object Tracking | Identity maintenance across frames | 0.2–0.5ms per frame |
| `01_perception/04_segmentation.md` | Instance Segmentation | Pixel-level masks | 3.4–21.8ms per frame |
| `01_perception/05_ocr.md` | OCR Pipeline | Text extraction from video | Event-triggered |
| `01_perception/06_reid.md` | Re-Identification | Person/instance matching | Event-triggered |
| `01_perception/07_audio_features.md` | Audio Feature Extraction | Audio signal processing | Stream latency |
| `01_perception/08_camera_motion.md` | Camera Motion Estimation | Ego-motion and stabilization | 1–3ms per frame |

## Layer 02 — Fusion

| File | Component | Purpose | Latency Target |
|------|-----------|---------|----------------|
| `02_fusion/01_multimodal_fusion.md` | Multi-Modal Fusion | Combine vision, audio, text | <5ms (fast path) |
| `02_fusion/02_temporal_fusion.md` | Temporal Fusion | Aggregate across time windows | 1–10ms |
| `02_fusion/03_cross_modal_alignment.md` | Cross-Modal Alignment | Synchronize modalities | <2ms |

## Layer 03 — State

| File | Component | Purpose | Latency Target |
|------|-----------|---------|----------------|
| `03_state/01_world_state.md` | World State | Typed entity/attribute store | 0.3–1ms per update |
| `03_state/02_entity_tracker.md` | Entity Tracker | Multi-entity lifecycle | 0.2–0.5ms per entity |
| `03_state/03_trajectory_model.md` | Trajectory Model | Position/velocity prediction | 0.1–0.3ms |
| `03_state/04_event_detection.md` | Event Detection | Rule-based event triggers | 0.1–0.3ms |

## Layer 04 — Memory

| File | Component | Purpose | Latency Target |
|------|-----------|---------|----------------|
| `04_memory/01_short_term.md` | Short-Term Memory | Ring buffer, current state | <0.1ms |
| `04_memory/02_working_memory.md` | Working Memory | Active hypothesis context | 1–5ms |
| `04_memory/03_long_term.md` | Long-Term Memory | Vector DB, episodic store | 10–100ms (async) |
| `04_memory/04_episodic_memory.md` | Episodic Memory | Event summaries, summaries | 50–500ms (async) |

## Layer 05 — Reasoning

| File | Component | Purpose | Latency Target |
|------|-----------|---------|----------------|
| `05_reasoning/01_hypothesis_engine.md` | Hypothesis Engine | Generate & rank hypotheses | 10–100ms (CPU) |
| `05_reasoning/02_fast_verifier.md` | Fast Verifier | Quick rule-based checks | 1–5ms |
| `05_reasoning/03_deep_vlm_reasoner.md` | Deep VLM Reasoner | Multimodal deep reasoning | 800–3500ms (async) |
| `05_reasoning/04_evidence_graph.md` | Evidence Graph | Causal/evidential links | 5–20ms per update |
| `05_reasoning/05_prediction_model.md` | Prediction Model | Future state prediction | 10–50ms |

## Layer 06 — Calibration

| File | Component | Purpose | Latency Target |
|------|-----------|---------|----------------|
| `06_calibration/01_confidence_decomposition.md` | Confidence Decomposition | Multi-component confidence | <1ms |
| `06_calibration/02_conformal_prediction.md` | Conformal Prediction | Prediction sets with coverage | 5–20ms |
| `06_calibration/03_temperature_scaling.md` | Temperature Scaling | Post-hoc calibration | <1ms |
| `06_calibration/04_claim_output.md` | Claim Output | Structured claim with metadata | <1ms |

## Layer 07 — Scheduler

| File | Component | Purpose | Latency Target |
|------|-----------|---------|----------------|
| `07_scheduler/01_multi_rate_scheduler.md` | Multi-Rate Scheduler | Orchestrate processing rates | <0.5ms overhead |
| `07_scheduler/02_queue_management.md` | Queue Management | Bounded queues, coalescing | <0.1ms per op |
| `07_scheduler/03_backpressure.md` | Backpressure | System overload protection | <1ms trigger |
| `07_scheduler/04_gpu_work_distribution.md` | GPU Work Distribution | CUDA streams, MPS/MIG | <0.2ms dispatch |

## Layer 08 — Forensics

| File | Component | Purpose | Latency Target |
|------|-----------|---------|----------------|
| `08_forensics/01_deepfake_detection.md` | Deepfake Detection | Synthetic media detection | 200–2000ms (async) |
| `08_forensics/02_provenance_c2pa.md` | Provenance / C2PA | Content credentials | 50–200ms per check |
| `08_forensics/03_audio_forensics.md` | Audio Forensics | Audio deepfake detection | 100–500ms |

## Layer 09 — Domains

| File | Component | Purpose | Latency Target |
|------|-----------|---------|----------------|
| `09_domains/01_sports_reasoning.md` | Sports Reasoning | Domain-specific rules | 10–50ms (fast path) |
| `09_domains/02_general_multimedia.md` | General Multimedia | Domain-agnostic fallback | Same as base pipeline |
| `09_domains/03_synthetic_media.md` | Synthetic Media | Detection + provenance | Same as forensics |

## Layer 10 — Infrastructure

| File | Component | Purpose | Latency Target |
|------|-----------|---------|----------------|
| `10_infrastructure/01_data_schemas.md` | Data Schemas | Protobuf/JSON contracts | N/A |
| `10_infrastructure/02_hardware_topology.md` | Hardware Topology | GPU/CPU/edge layout | N/A |
| `10_infrastructure/03_deployment.md` | Deployment | Docker/K8s/helm charts | N/A |

---

## Composition Rules

1. **Dependencies flow downward only.** Higher layers depend on lower layers, never the reverse.
2. **Each component publishes typed outputs.** No shared mutable state between components.
3. **Latency budgets are hard constraints.** If a component exceeds its budget, it must degrade gracefully.
4. **Every component has a failure mode.** No silent failures. All errors propagate as structured error objects.
5. **Async components must be idempotent.** VLM reasoning, forensics, and summarization can be retried safely.

## Cross-References

- `diagrams/00_system_diagrams.md` — All Mermaid diagrams (15 diagrams)
- `benchmark_report_2026.md` — Per-component benchmark data
- `09_sources/README.md` — All 139 sources
- `01_foundations/` — Research tracks for each component
- `08_implementation/` — Build plan, ablation plan, risk register
