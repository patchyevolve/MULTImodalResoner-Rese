# Agent Context Prompt — Multimodal Video Reasoner

> **Purpose:** This document gives you complete understanding of the project so you can build the code implementation consistent with all architecture, research, and training plan documents. Read this ENTIRE document before writing any code.

---

## 1. Project Overview

**What we are building:** A real-time multimodal video reasoning system that processes live video at 30 FPS, detects and tracks objects, reasons about events using vision-language models, and outputs structured claims with calibrated confidence.

**Where you are working:** The code lives in `src/` at the project root. Everything else (`00_core/` through `10_training_plan/`) is research, architecture, and planning documents that define WHAT to build. You build HOW.

**Repository:** `https://github.com/patchyevolve/MULTImodalResoner-Rese`

---

## 2. What Exists (Research Repo)

The repo contains 97 markdown files across 12 directories. These are the authoritative specifications. Your code must be consistent with them.

```
00_core/              ← Thesis, terminology, 10 system principles
01_foundations/        ← 18 research tracks (perception, fusion, reasoning, etc.)
02_architecture/       ← 41 component specs + 60 Mermaid diagrams
  01_perception/      ← Detection, pose, tracking, segmentation, OCR, ReID, audio, camera
  02_fusion/          ← Multimodal, temporal, cross-modal alignment
  03_state/           ← World state, entity tracker, trajectory, event detection
  04_memory/          ← Short-term, working, long-term, episodic
  05_reasoning/       ← Hypothesis engine, fast verifier, VLM reasoner, evidence graph, prediction
  06_calibration/     ← Confidence decomposition, conformal, temperature scaling, claim output
  07_scheduler/       ← Multi-rate, queue management, backpressure, GPU distribution
  08_forensics/       ← Deepfake detection, C2PA, audio forensics
  09_domains/         ← Sports, general multimedia, synthetic media
  10_infrastructure/  ← Schemas, hardware topology, deployment
  diagrams/           ← 60 Mermaid diagrams (system, DB, API, complete reference)
03_models/           ← Model selection matrix (RF-DETR-S, DETRPose-S, Qwen3-VL-30B, etc.)
04_uncertainty/      ← Confidence model, claim taxonomy
05_realtime/         ← 30 FPS budget, optimization research
06_domains/          ← Sports reasoning, general multimedia, synthetic media
07_evaluation/       ← 10 experiments (E1-E10), success criteria
08_implementation/   ← Staged build plan (10 stages), ablation plan, risk register
09_sources/          ← 139 sources in 18 categories
10_training_plan/    ← 13 files: master plan, preparation, datasets, training, optimization,
                       inference integration, benchmarking, timeline, infrastructure,
                       system flow, project structure audit, model weights/disk space,
                       MLForge v1.0 training system (immutable runs, fail-closed resume validation)
```

---

## 3. Architecture Summary — 10 Layers, 41 Components

Every component follows this template: Purpose → Interfaces → Data Contracts → Latency Targets → Dependencies → Failure Modes → Reality Check 2026.

### Layer 01 — Perception (8 components, 30 FPS, <20ms total)

| # | Component | Model | Latency | Output Type |
|---|---|---|---|---|
| 01 | Detection | RF-DETR-S TensorRT | 2.3-6.8ms | DetectionBatch |
| 02 | Pose Estimation | DETRPose-S TensorRT | 2.39-9.7ms | PoseBatch |
| 03 | Object Tracking | ByteTrack | 0.2-0.5ms | TrackingOutput |
| 04 | Segmentation | RF-DETR-Seg-S | 5-10ms | SegmentationBatch |
| 05 | OCR | PaddleOCR v4 | 15-40ms | OCRBatch |
| 06 | Re-Identification | OSNet | 3-10ms | ReIDOutput |
| 07 | Audio Features | Whisper-large-v3 | 50-100ms | AudioOutput |
| 08 | Camera Motion | ORB+RANSAC | 1-3ms | CameraMotionOutput |

### Layer 02 — Fusion (3 components, 2-10ms)

| # | Component | Strategy | Output Type |
|---|---|---|---|
| 01 | Multi-Modal Fusion | Late/Attention fusion | FusionOutput |
| 02 | Temporal Fusion | EMA sliding window | TemporalFusionOutput |
| 03 | Cross-Modal Alignment | Correlation sync | AlignmentOutput |

### Layer 03 — State (4 components, 0.3-1ms)

| # | Component | Strategy | Output Type |
|---|---|---|---|
| 01 | World State | Lock-free ring buffer | WorldStateSnapshot |
| 02 | Entity Tracker | Lifecycle state machine | Entity[] |
| 03 | Trajectory Model | Kalman filter | Trajectory |
| 04 | Event Detection | Rule engine + R score | EventTrigger[] |

### Layer 04 — Memory (4 components)

| # | Component | Strategy | Latency |
|---|---|---|---|
| 01 | Short-Term | Ring buffer, 30 frames | <0.1ms |
| 02 | Working Memory | Top-10 active hypotheses | 1ms |
| 03 | Long-Term | SQLite + FAISS | 10-50ms |
| 04 | Episodic | Episode store + search | 5-30ms |

### Layer 05 — Reasoning (5 components)

| # | Component | Strategy | Latency |
|---|---|---|---|
| 01 | Hypothesis Engine | Rules + GBDT ranker | 5-15ms |
| 02 | Fast Verifier | Trajectory/pose/rule checks | 1-5ms |
| 03 | Deep VLM Reasoner | Qwen3-VL-30B-A3B | 800-3500ms |
| 04 | Evidence Graph | Typed nodes + edges | 1-5ms |
| 05 | Prediction Model | Trajectory extrapolation | 0.1ms |

### Layer 06 — Calibration (4 components, 3-8ms)

| # | Component | Strategy | Output Type |
|---|---|---|---|
| 01 | Confidence Decomposition | 5-component weighted sum | ConfidenceDecomposition |
| 02 | Conformal Prediction | Nonconformity scores | PredictionSet |
| 03 | Temperature Scaling | Post-hoc NLL minimization | CalibratedProb |
| 04 | Claim Output | JSON/Protobuf assembly | Claim |

### Layer 07 — Scheduler (4 components)

| # | Component | Strategy |
|---|---|---|
| 01 | Multi-Rate Scheduler | R-score priority assignment |
| 02 | Queue Manager | Bounded deques with coalescing |
| 03 | Backpressure | Threshold triggers, graceful degradation |
| 04 | GPU Distributor | CUDA stream assignment, MPS/MIG |

### Layer 08 — Forensics (3 components, async)

| # | Component | Strategy |
|---|---|---|
| 01 | Deepfake Detection | CLIP+EVA-02+SRM ensemble |
| 02 | C2PA/SynthID | Cryptographic provenance |
| 03 | Audio Forensics | Teffic-Audio + FlowFake |

### Layer 09 — Domains (3 components)

| # | Component | Strategy |
|---|---|---|
| 01 | Sports Reasoning | SoccerNet rules + trajectories |
| 02 | General Multimedia | Domain-agnostic fallback |
| 03 | Synthetic Media | Detection + provenance |

### Layer 10 — Infrastructure (3 components)

| # | Component | Strategy |
|---|---|---|
| 01 | Data Schemas | Protobuf + FlatBuffers |
| 02 | Hardware Topology | GPU/CPU/Edge layout |
| 03 | Deployment | Docker/K8s/Helm |

---

## 4. Data Contracts — Every Typed Interface

These are the EXACT types that pass between components. Your code MUST use these field names and types.

### 4.1 Ingestor Output → Perception Input

```python
@dataclass
class RawFrame:
    frame_id: int          # uint64
    timestamp_ns: int      # uint64, nanosecond timestamp
    image: np.ndarray      # Tensor[H, W, 3] uint8, on GPU
    stream_id: str

@dataclass
class AudioChunk:
    chunk_id: int          # uint64
    audio: np.ndarray      # PCM float32
    sample_rate: int       # 16000
```

### 4.2 Detection Output → Pose/Tracking/Fusion

```python
@dataclass
class Detection:
    bbox_norm: list[float]    # [4] — x_center, y_center, w, h (normalized 0-1)
    class_id: int             # uint16
    class_name: str           # "person", "ball", etc.
    score: float              # detection confidence 0-1
    feature_ref: str          # pointer to appearance embedding

@dataclass
class DetectionBatch:
    frame_id: int
    timestamp_ns: int
    detections: list[Detection]
    model_id: str             # "RF-DETR-S"
    inference_ms: float
```

### 4.3 Pose Output → World State

```python
@dataclass
class PoseResult:
    entity_id: str
    keypoints_2d: list[list[float]]    # [17][2] — COCO 17 keypoints
    keypoint_scores: list[float]       # [17]
    pose_score: float
    occluded_joints: int               # bitmask

@dataclass
class PoseBatch:
    frame_id: int
    timestamp_ns: int
    poses: list[PoseResult]
    model_id: str                      # "DETRPose-S"
    inference_ms: float
```

**COCO 17 Keypoint Order:** nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

### 4.4 Tracking Output → World State/Entity Tracker

```python
@dataclass
class Track:
    entity_id: str
    bbox_norm: list[float]     # [4]
    class_id: int
    age: int                   # frames since first detection
    hits: int                  # total confirmations
    time_since_update: int     # frames since last update
    velocity: list[float]      # [2] — vx, vy normalized
    score: float
    is_occluded: bool

@dataclass
class TrackingOutput:
    frame_id: int
    timestamp_ns: int
    tracks: list[Track]
    statistics: dict           # active_tracks, new_tracks, lost_tracks, removed_tracks
```

### 4.5 World State → Everything Downstream

```python
@dataclass
class Entity:
    id: str
    class_name: str            # "person", "ball", "object"
    bbox_norm: list[float]     # [4]
    pose: PoseResult | None
    velocity: list[float]      # [3]
    acceleration: list[float]  # [3]
    attributes: dict           # team, jersey_number, role, etc.
    confidence: float
    occlusion_level: float     # 0=visible, 1=fully occluded
    tracking_state: str        # "TENTATIVE" | "CONFIRMED" | "LOST"

@dataclass
class Relation:
    subject: str               # entity_id
    predicate: str             # "near", "has_possession", "blocking"
    object: str                # entity_id
    confidence: float

@dataclass
class Scene:
    location: str
    activity: str
    num_entities: int
    lighting: str
    weather: str

@dataclass
class WorldStateSnapshot:
    frame_id: int
    timestamp_ns: int
    entities: list[Entity]
    trajectories: list          # Trajectory objects
    relations: list[Relation]
    scene: Scene
    events: list                # Event objects
    uncertainty: dict           # per-component uncertainty scores
    staleness_ms: float
```

### 4.6 Event Detection → Scheduler/Reasoning

```python
@dataclass
class EventTrigger:
    event_id: str
    event_type: str             # "goal", "foul", "scene_cut", "anomaly", "new_entity"
    timestamp_ns: int
    entity_ids: list[str]
    r_score: float              # 0-1, priority/importance score
    evidence: list              # Evidence objects
    confidence: float
```

**R-Score Formula:**
```
R = f(prediction_error, uncertainty, model_disagreement, event_importance, user_priority)
```
Higher R → more compute allocated. R ≥ 0.5 triggers fast reasoning. R ≥ 0.7 gets priority scheduling.

### 4.7 Hypothesis Engine → Fast Verifier → VLM

```python
@dataclass
class Hypothesis:
    id: str
    claim: str                  # natural language hypothesis
    type: str                   # "OBSERVED", "INFERRED", "PREDICTED"
    status: str                 # "CANDIDATE", "SUPPORTED", "REFUTED", "INCONCLUSIVE"
    evidence: list              # Evidence objects
    contradictions: list
    confidence: dict            # 5-component decomposition
    priority: float             # R score

@dataclass
class VerificationResult:
    hypothesis_id: str
    verdict: str                # "SUPPORTED", "REFUTED", "INCONCLUSIVE"
    confidence: float
    evidence_used: list
    verification_ms: float
    needs_deep: bool            # True → send to VLM
```

### 4.8 VLM Reasoning

```python
@dataclass
class VLMReasoningInput:
    hypothesis: Hypothesis
    state_snapshots: list[WorldStateSnapshot]   # recent frames
    working_memory: dict                         # active context
    episode_context: list                        # similar past episodes
    prompt_template: str

@dataclass
class VLMReasoningOutput:
    hypothesis_id: str
    verdict: str               # "SUPPORTED", "REFUTED", "INCONCLUSIVE"
    reasoning_text: str        # natural language explanation
    confidence: dict           # 5-component decomposition
    evidence_used: list
    inference_ms: float
    model_id: str              # "Qwen3-VL-30B-A3B"
    tokens_generated: int
```

**VLM Output JSON Schema:**
```json
{
  "verdict": "supported|refuted|inconclusive",
  "confidence": {
    "perception": 0.0-1.0,
    "temporal": 0.0-1.0,
    "reasoning": 0.0-1.0
  },
  "evidence_refs": ["ev_104", "ev_110"],
  "reasoning": "natural language explanation",
  "alternative_hypotheses": [
    {"claim": "...", "probability": 0.2}
  ]
}
```

### 4.9 Evidence Graph

```python
@dataclass
class EvidenceNode:
    id: str
    node_type: str    # "PERCEPTION", "TRACKING", "ACTION", "SPATIAL", "TEMPORAL", "CAUSAL", "CLAIM_SUPPORT"
    data: dict        # type-specific fields
    timestamp_ns: int
    confidence: float

@dataclass
class EvidenceEdge:
    source: str       # node_id
    target: str       # node_id
    edge_type: str    # "TEMPORAL", "SPATIAL", "CAUSAL", "SEMANTIC", "TRACKING", "EVIDENCE", "NEGATION"
    weight_llr: float # log-likelihood ratio
    confidence: float
    metadata: dict
```

### 4.10 Calibration → Claim Output

```python
@dataclass
class ConfidenceDecomposition:
    perception: float              # 0-1
    temporal: float                # 0-1
    motion: float                  # 0-1
    cross_modal_agreement: float   # 0-1
    reasoning: float               # 0-1
    calibrated: float              # 0-1, after temperature scaling
    calibration_method: str
    uncertainty_sources: list[str]
    overall: float                 # weighted sum

@dataclass
class Claim:
    id: str
    schema_version: str            # "1.0.0"
    timestamp_ns: int
    published_ts_ns: int
    claim_text: str                # natural language claim
    claim_type: str                # "OBSERVED", "INFERRED", "PREDICTED"
    epistemic_status: str          # "CONFIRMED", "PROBABLE", "POSSIBLE", "SPECULATIVE"
    confidence: ConfidenceDecomposition
    conformal_prediction_set_alpha_05: list[str]   # hypothesis IDs
    conformal_prediction_set_alpha_10: list[str]
    prediction_set_size_at_alpha_05: int
    evidence: list                 # Evidence objects
    provenance: dict               # model info, staleness, source
    output_quality: dict           # latency, coverage, gpu_util
```

**Confidence Decomposition Formula:**
```
overall = 0.25 * perception + 0.20 * temporal + 0.15 * motion + 0.15 * cross_modal + 0.25 * reasoning
```

---

## 5. Data Flow — How Components Connect

```
FRAME ARRIVES
  → FileIngestor.decode()                    [1-3ms]   → RawFrame
  → Preprocessor.preprocess()                [1-2ms]   → GPU tensor [1,3,640,640]
  → TensorRTDetector.detect()                [2.3-6.8ms] → DetectionBatch
  → ByteTrack.update()                       [0.2-0.5ms] → TrackingOutput
  → PoseEstimator.estimate()                 [2.39-5ms]  → PoseBatch
  → WorldState.update()                      [0.3-1ms]   → WorldStateSnapshot
  → EventDetector.detect()                   [0.1-0.3ms] → EventTrigger[]
  → RingBuffer.write()                       [0.2-0.5ms] → Published

TOTAL CRITICAL PATH: 8-12ms (leaves 21-25ms headroom for 30 FPS)

EVENT TRIGGERED (R ≥ 0.5)
  → HypothesisEngine.generate()              [5-15ms]  → Hypothesis[]
  → FastVerifier.verify()                    [1-5ms]   → VerificationResult
  → IF SUPPORTED/REFUTED:
      → ConfidenceDecomposition.decompose()  [<1ms]
      → ConformalPredict.predict()           [2-5ms]
      → TemperatureScaling.scale()           [<0.1ms]
      → ClaimOutput.assemble()               [1-2ms]   → Claim
  → IF INCONCLUSIVE:
      → VLMReasoner.inference()              [800-3500ms] → Claim (async)
```

---

## 6. Latency Budgets — Hard Constraints

| Component | Target (p50) | Target (p99) | Max Allowed |
|---|---|---|---|
| Decode | 2ms | 3ms | 5ms |
| Preprocess | 1.5ms | 2ms | 3ms |
| Detection (RF-DETR-S) | 3.5ms | 6.8ms | 10ms |
| Pose (DETRPose-S) | 2.4ms | 5ms | 8ms |
| Tracking (ByteTrack) | 0.3ms | 0.5ms | 1ms |
| State Update | 0.5ms | 1ms | 2ms |
| Event Detection | 0.2ms | 0.3ms | 1ms |
| **Critical Path Total** | **~10ms** | **~18ms** | **<33ms** |
| Hypothesis Generation | 10ms | 15ms | 20ms |
| Fast Verify | 3ms | 5ms | 8ms |
| VLM Inference | 1500ms | 3500ms | 5000ms |
| Calibration | 4ms | 8ms | 15ms |
| **Fast Reasoning Total** | **~20ms** | **~35ms** | **<50ms** |

---

## 7. Hardware Constraints

### Training Hardware (Two-Phase)

| Phase | Hardware | Duration | Role |
|---|---|---|---|
| **Temporary** | RTX 3070 8GB, 32GB RAM, 2TB HDD | Days 1-10 | Small model training (OSNet, GBDT, calibrator), RF-DETR-S with batch=2 + grad accum |
| **Permanent** | H100 ~80GB | Day 10+ | Full-quality retraining, VLM LoRA, everything fast |
| **Future** | Multi-GPU (2-4× H100) | If needed | DDP/FSDP — auto-detected, no code changes |

Training system design: `10_training_plan/12_training_system.md` (MLForge v1.0 — immutable runs, cryptographic artifact identities, fail-closed validation, exact vs portable resume modes).

### Inference Hardware (Tier 1: Single RTX 4090 24GB — Production Target)

```
GPU Memory Budget:
  RF-DETR-S:          1.5 GB
  DETRPose-S:         0.5 GB
  ByteTrack:          0.05 GB
  CUDA Context:       1 GB
  Working Space:      2 GB
  ─────────────────────────
  Perception Total:   ~5 GB
  Remaining for VLM:  ~19 GB
  
  Qwen3-VL-30B-A3B FP8: ~20 GB (TIGHT — needs token compression)
  OR: API fallback (GPT-4.1 / Gemini Flash)
```

### Tier 2: RTX 5070 Ti (16GB) — Fallback

```
  Perception: ~5 GB
  VLM: API only (no local VLM fits)
```

### Token Compression (Required for Tier 1 Local VLM)

- StreamingTOM: 15.7× KV cache reduction, 2× TTFT speedup
- HybridKV: 7.9× memory reduction, 1.52× decode speedup
- Applied at inference time, not training

---

## 8. Model Selection (Verified 2026)

| Component | Model | Latency | Accuracy | Source |
|---|---|---|---|---|
| Detection | RF-DETR-S | 3.5ms T4 | 53.0 mAP COCO | Roboflow SAB |
| Detection (async) | RF-DETR-L | 6.8ms T4 | 56.5 mAP COCO | Roboflow SAB |
| Pose | DETRPose-S | 2.39ms A10 | 67.0 AP COCO | Official |
| Pose (async) | DETRPose-L | 5.08ms A10 | 72.5 AP COCO | Official |
| Tracking | ByteTrack | 0.2-0.5ms | HOTA 87.2 SportsMOT | SAM 3 |
| Segmentation | RF-DETR-Seg-S | 5-10ms | — | Roboflow |
| OCR | PaddleOCR v4 | 15-40ms | 3% WER | Official |
| Re-ID | OSNet | 3-10ms | Rank-1 95%+ Market1501 | Official |
| Audio | Whisper-large-v3 | 50-100ms | 3% WER | OpenAI |
| VLM (local) | Qwen3-VL-30B-A3B FP8 | 800-1500ms | 90 tok/s RTX 4090D | Qwen |
| VLM (API) | GPT-4.1 / Gemini Flash | 400ms TTFT | — | API |
| Hypothesis Ranker | LightGBM (GBDT) | <1ms | NDCG@3 ≥ 0.85 | Train |

---

## 9. Training Strategy

**Do NOT train end-to-end.** Compose pretrained modules. This is the verified 2026 strategy.

### Models We Fine-Tune (Sequential, One GPU at a Time)

| Priority | Model | Dataset | Duration | GPU |
|---|---|---|---|---|
| 1st | RF-DETR-S | COCO + custom domain | 2-3 days | Full GPU |
| 2nd | OSNet | Market1501 + custom | 1-2 days | Full GPU |
| 3rd | RF-DETR-Seg-S | COCO + custom | 2-3 days | Full GPU |
| 4th | GBDT Ranker | Labeled event-hypothesis pairs | 2-4 hours | CPU |
| 5th | Calibrator | Held-out validation | 2-4 hours | CPU |

### Two-Stage Detection Training

1. **Baseline (Days 5-7):** Fine-tune from COCO-pretrained weights on COCO + custom domain
2. **Domain Adaptation (Phase 2+, if needed):** LoRA adapters on sports-specific data

### What Uses Pre-Trained Weights (No Training)

- DETRPose-S, ByteTrack, PaddleOCR, Whisper, ORB+RANSAC
- Deepfake ensemble (CLIP+EVA-02+SRM)
- C2PA/SynthID tools

### What Is Code-Only (No Training)

- All fusion, state, memory, scheduler, reasoning (except VLM prompt engineering), calibration (post-hoc), forensics integration, domains, infrastructure

---

## 10. Evaluation Targets (E1-E10)

| Experiment | Target | Covered In |
|---|---|---|
| E1: Direct observation | Within 2% of COCO baselines | Training plan |
| E2: Occluded-state | <40% degradation at 50% occlusion | Architecture |
| E3: Temporal event | Top-3 recall ≥85% at 2s horizon | Training plan |
| E4: Hypothesis competition | recall@3 ≥85% correct in top-3 | Training plan |
| E5: Contradictory modalities | Audio-contradiction downweighted | Architecture |
| E6: Synthetic media | Inconclusive on >80% attacked samples | Architecture |
| E7: Prediction scheduling | Event-triggered ≥15% better | Training plan |
| E8: 30 FPS stress | 30 FPS sustained, p99 <33ms | Training plan |
| E9: Long-term memory | State continuity 30+ min sessions | Architecture |
| E10: Distribution shift | <10pp degradation 3+ domain shifts | Architecture |

---

## 11. Code Structure — What to Build

```
src/
├── ingestor/                      ← Pre-layer
│   ├── file_ingestor.py           ← Read video files
│   ├── rtsp_ingestor.py           ← Read RTSP streams
│   ├── audio_ingestor.py          ← Extract audio
│   └── frame_buffer.py            ← Pre-allocated GPU buffer pool
│
├── perception/                    ← LAYER 01
│   ├── detector.py                ← RF-DETR-S TensorRT
│   ├── pose_estimator.py          ← DETRPose-S TensorRT
│   ├── tracker.py                 ← ByteTrack wrapper
│   ├── segmenter.py               ← RF-DETR-Seg-S (Phase 2)
│   ├── ocr.py                     ← PaddleOCR (Phase 2)
│   ├── reid.py                    ← OSNet (Phase 2)
│   ├── audio_features.py          ← Whisper (Phase 2)
│   ├── camera_motion.py           ← ORB+RANSAC (Phase 2)
│   └── pipeline.py                ← Perception orchestrator
│
├── fusion/                        ← LAYER 02
│   ├── multimodal_fusion.py
│   ├── temporal_fusion.py
│   └── cross_modal_alignment.py
│
├── state/                         ← LAYER 03
│   ├── world_state.py
│   ├── entity_tracker.py
│   ├── trajectory_model.py
│   └── event_detector.py
│
├── memory/                        ← LAYER 04
│   ├── short_term.py
│   ├── working_memory.py
│   ├── long_term.py
│   └── episodic.py
│
├── reasoning/                     ← LAYER 05
│   ├── hypothesis_engine.py
│   ├── fast_verifier.py
│   ├── vlm_reasoner.py
│   ├── evidence_graph.py
│   └── prediction_model.py
│
├── calibration/                   ← LAYER 06
│   ├── confidence_decomposition.py
│   ├── conformal.py
│   ├── temperature_scaling.py
│   └── claim_output.py
│
├── scheduler/                     ← LAYER 07
│   ├── priority_scheduler.py
│   ├── queue_manager.py
│   ├── backpressure.py
│   └── gpu_distributor.py
│
├── forensics/                     ← LAYER 08
│   ├── deepfake_detection.py
│   ├── provenance_c2pa.py
│   └── audio_forensics.py
│
├── domains/                       ← LAYER 09
│   ├── sports_reasoning.py
│   ├── general_multimedia.py
│   └── synthetic_media.py
│
├── schemas/                       ← LAYER 10
│   ├── perception.proto
│   ├── state.proto
│   ├── reasoning.proto
│   ├── calibration.proto
│   └── claim.proto
│
├── output/
│   ├── rest_api.py
│   ├── websocket_server.py
│   └── kafka_producer.py
│
├── monitoring/
│   ├── metrics.py
│   └── health.py
│
├── pipeline.py                    ← MAIN ORCHESTRATOR
└── server.py                      ← ENTRY POINT
```

---

## 12. Build Order — Exactly What to Build, In What Order

### Phase 1: Core Pipeline (Weeks 1-4) — BUILD THIS FIRST

```
Step 1:  src/ingestor/file_ingestor.py         ← Read video, output RawFrame
Step 2:  src/perception/detector.py             ← TensorRT RF-DETR-S, output DetectionBatch
Step 3:  src/perception/tracker.py              ← ByteTrack, output TrackingOutput
Step 4:  src/perception/pose_estimator.py       ← TensorRT DETRPose-S, output PoseBatch
Step 5:  src/perception/pipeline.py             ← Connect detect+pose+track
Step 6:  src/state/world_state.py               ← Ring buffer, output WorldStateSnapshot
Step 7:  src/state/event_detector.py            ← Rule engine, output EventTrigger[]
Step 8:  src/scheduler/priority_scheduler.py    ← R-score priority
Step 9:  src/reasoning/hypothesis_engine.py     ← Generate hypotheses
Step 10: src/reasoning/fast_verifier.py         ← Rule-based verify
Step 11: src/memory/short_term.py               ← Ring buffer snapshots
Step 12: src/pipeline.py                        ← Connect everything
Step 13: tests/unit/test_core_pipeline.py       ← Verify it works
```

### Phase 2: Reasoning + Calibration (Weeks 5-8)

```
Step 14: src/reasoning/vlm_reasoner.py
Step 15: src/reasoning/evidence_graph.py
Step 16: src/calibration/confidence_decomposition.py
Step 17: src/calibration/conformal.py
Step 18: src/calibration/temperature_scaling.py
Step 19: src/calibration/claim_output.py
Step 20: src/memory/working_memory.py
Step 21: src/output/rest_api.py
Step 22: tests/integration/test_reasoning.py
```

### Phase 3: Async + Optimization (Weeks 9-12)

```
Step 23-34: segmentation, OCR, ReID, audio, camera, fusion, queue, backpressure, GPU, monitoring
```

### Phase 4: Domains + Forensics (Weeks 13-16)

```
Step 35-46: sports, general, synthetic, deepfake, C2PA, audio forensics, long-term memory, episodic, prediction, websocket, kafka
```

### Phase 5: Production (Weeks 17-20)

```
Step 47-55: config files, protobuf schemas, Docker, K8s, scripts, notebooks, docs, testing, optimization, security
```

---

## 13. System Principles (Non-Negotiable)

From `00_core/03_system_principles.md`:

1. **Do not conflate visibility with truth.** Hidden facts can be inferred; inferred facts are not direct observation.
2. **Do not collapse hypotheses prematurely.** Maintain competing explanations until evidence separates them.
3. **Use time as evidence.** Before/after observations constrain hidden events.
4. **Use structural priors.** Human biomechanics, object dynamics, scene topology constrain hidden states.
5. **Make uncertainty compositional.** A claim can have high detection confidence but low intent confidence.
6. **Exploit redundancy.** Trackers, motion models, temporal memory carry information between expensive inference calls.
7. **Use disagreement as signal.** Cross-model or cross-modal conflict increases uncertainty or triggers deeper reasoning.
8. **Optimize for information, not uniform computation.** Spend compute when uncertainty or prediction error is high.
9. **Keep an evidence trail.** Every claim traceable to source media and intermediate reasoning artifacts.
10. **Evaluate under degradation.** Occlusion, blur, compression, missing audio, camera cuts are part of the target environment.

### Reality-Based Principles (2026)

- **R1:** VLM cadence ≠ perception cadence. VLM outputs are ALWAYS stale by 500-3000ms. Every VLM claim MUST carry `claim_staleness_ms`.
- **R2:** Detection/tracking are imperfect. Expect 5-15% ID switches. Track 2-5 ID hypotheses per entity in crowded scenes.
- **R3:** For occluded joints: output distributions, NEVER a single "best guess". Required: `visibility_score`, `posterior_covariance`, `estimation_method`.
- **R4:** Synthetic media detection is multi-evidence, uncertain judgment — never a binary flag. Minimum 3 independent evidence channels before anything other than "inconclusive".
- **R5:** Calibration is post-hoc. Temperature scaling, isotonic regression, conformal prediction — NO retraining.
- **R6:** Open-vocabulary recall has hard ceilings (~60-65% on LVIS). Prefer closed-vocabulary for known domain ontology.
- **R7:** Causal claims require explicit grounding. "A then B" = temporal_correlation, NOT causal_claim.

---

## 14. Risk Mitigation — How Code Addresses Each Risk

| Risk | Code Mitigation |
|---|---|
| R1: False confidence | `calibration/temperature_scaling.py` + `calibration/conformal.py` — ECE < 0.05, coverage ≥ 95% |
| R2: Temporal hallucination | `reasoning/fast_verifier.py` — temporal consistency check before VLM |
| R3: Compounding state error | `state/world_state.py` — staleness tracking, confidence decay on unseen entities |
| R4: Scheduler starvation | `scheduler/backpressure.py` — queue depth > 4 → coalesce stale jobs |
| R5: Domain overfitting | Two-stage: baseline on COCO, optional LoRA adaptation |
| R6: Synthetic media brittleness | `forensics/` — ensemble + C2PA + "inconclusive" output |
| R7: Overinterpretation | `calibration/confidence_decomposition.py` — occlusion penalty |
| R8: Latency collapse | `monitoring/metrics.py` — stress test, memory leak detection |
| R9: Adversarial attacks | Qwen3-VL has 6.5% ASR (vs LLaVA 52.6-66.9%) |
| R10: Distribution shift | VLM-RobustBench testing, async scheduling |
| R11: Production failure | Every component has try/except, graceful degradation |
| R12: Bias | Cross-modal agreement check, provenance, human review |
| R13: EU AI Act | C2PA credentials in claim provenance |

---

## 15. Configuration System

All config lives in `config/` as YAML. The orchestrator loads `config/pipeline.yaml` which references component configs.

```yaml
# config/pipeline.yaml
perception:
  detection:
    model: "rf-detr-s"
    input_size: [640, 640]
    confidence_threshold: 0.5
    nms_threshold: 0.7
    max_detections: 100
    precision: "fp16"
  pose:
    model: "detrpose-s"
    confidence_threshold: 0.3
  tracker:
    tracker_type: "bytetrack"
    track_thresh: 0.5
    match_thresh: 0.8
    max_age: 30
    min_hits: 3

state:
  ring_buffer_size: 30
  stale_threshold_frames: 10
  entity_confidence_decay: 0.9

events:
  rules:
    - name: "goal"
      r_score: 0.8
    - name: "foul"
      r_score: 0.7
    - name: "scene_cut"
      r_score: 0.9
    - name: "new_entity"
      r_score: 0.5

reasoning:
  hypothesis_max: 10
  fast_verify_timeout_ms: 5
  vlm_mode: "api"                    # api | local
  vlm_model: "gpt-4o"                # or qwen3-vl-30b-a3b
  vlm_api_key_env: "OPENAI_API_KEY"
  vlm_queue_depth: 4
  vlm_timeout_ms: 5000

calibration:
  temperature_model: "models/calibrator/temperature.json"
  conformal_model: "models/calibrator/conformal.json"
  decomposition_model: "models/calibrator/decomposition.json"

output:
  rest_api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
  websocket:
    enabled: true
    port: 8001
```

---

## 16. What NOT to Build (By Design)

These are explicitly excluded from the training plan and initial implementation:

- **End-to-end training** — research says don't do it (destroys calibration, 100-1000× more compute)
- **VLM fine-tuning with QLoRA** — prompt engineering first, QLoRA only if quality insufficient
- **Continual learning** — use LoRA swaps instead
- **RL/preference training** — overkill until heuristic + LTR baseline exhausted
- **Self-supervised learning** — all perception components already pretrained on massive data

---

## 17. How to Use This Document

1. **Read the architecture file** for the component you're building (`02_architecture/XX_component/`)
2. **Check the data contracts** in Section 4 above for exact types
3. **Check the latency budget** in Section 6 for your component's target
4. **Follow the build order** in Section 12 — don't skip ahead
5. **Run tests** after each step
6. **Be consistent** with the field names and types in this document — other components depend on them
