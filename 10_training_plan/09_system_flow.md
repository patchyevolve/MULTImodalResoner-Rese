# System Flow Document

> This document maps the architecture to actual code. Every component from the 10-layer architecture, how data flows between them, what each component produces, what each component consumes, and where the code lives.

---

## 1. Architecture → Code Mapping

Every architecture component maps to a code module. No exceptions.

| Layer | Component (Architecture) | Code Module | Input Type | Output Type |
|---|---|---|---|---|
| **INGEST** | Video Source | `src/ingestor/file_ingestor.py` | RTSP/File/URL | Raw frame (numpy) |
| **INGEST** | Audio Source | `src/ingestor/audio_ingestor.py` | Audio stream | PCM chunks |
| **INGEST** | Frame Buffer | `src/ingestor/frame_buffer.py` | Raw frames | GPU tensors (pinned) |
| **01** | Detection Pipeline | `src/perception/detector.py` | GPU tensor [1,3,640,640] | DetectionBatch |
| **01** | Pose Estimation | `src/perception/pose_estimator.py` | GPU tensor + crops | PoseBatch |
| **01** | Object Tracking | `src/perception/tracker.py` | DetectionBatch | TrackingOutput |
| **01** | Instance Segmentation | `src/perception/segmenter.py` | GPU tensor + boxes | SegmentationBatch |
| **01** | OCR Pipeline | `src/perception/ocr.py` | GPU tensor + ROI | OCRBatch |
| **01** | Re-Identification | `src/perception/reid.py` | Person crops | Embeddings |
| **01** | Audio Features | `src/perception/audio_features.py` | PCM chunks | Transcript + events |
| **01** | Camera Motion | `src/perception/camera_motion.py` | Frame pair | Homography |
| **02** | Multi-Modal Fusion | `src/fusion/multimodal_fusion.py` | Vision+Audio+Text features | FusedFeatures |
| **02** | Temporal Fusion | `src/fusion/temporal_fusion.py` | FusedFeatures + history | SmoothedFeatures |
| **02** | Cross-Modal Alignment | `src/fusion/cross_modal_alignment.py` | Misaligned features | AlignedFeatures |
| **03** | World State | `src/state/world_state.py` | EntityUpdate[] | WorldStateSnapshot |
| **03** | Entity Tracker | `src/state/entity_tracker.py` | Tracks + Poses | Entity[] |
| **03** | Trajectory Model | `src/state/trajectory_model.py` | Entity history | TrajectoryOutput |
| **03** | Event Detection | `src/state/event_detector.py` | WorldStateSnapshot | EventTrigger[] |
| **04** | Short-Term Memory | `src/memory/short_term.py` | WorldStateSnapshot | Recent snapshots |
| **04** | Working Memory | `src/memory/working_memory.py` | Hypothesis | Active hypotheses |
| **04** | Long-Term Memory | `src/memory/long_term.py` | Entity profiles | Historical context |
| **04** | Episodic Memory | `src/memory/episodic.py` | Event summaries | Similar episodes |
| **05** | Hypothesis Engine | `src/reasoning/hypothesis_engine.py` | Event + State + Memory | Hypothesis[] |
| **05** | Fast Verifier | `src/reasoning/fast_verifier.py` | Hypothesis + State | VerificationResult |
| **05** | Deep VLM Reasoner | `src/reasoning/vlm_reasoner.py` | Hypothesis + State + Memory | VLMReasoningOutput |
| **05** | Evidence Graph | `src/reasoning/evidence_graph.py` | Nodes + Edges | EvidenceGraph |
| **05** | Prediction Model | `src/reasoning/prediction_model.py` | Trajectory + Context | PredictedState |
| **06** | Confidence Decomposition | `src/calibration/confidence_decomposition.py` | Raw confidences | DecomposedConfidence |
| **06** | Conformal Prediction | `src/calibration/conformal.py` | Hypothesis set | PredictionSet |
| **06** | Temperature Scaling | `src/calibration/temperature_scaling.py` | Logits | CalibratedProb |
| **06** | Claim Output | `src/calibration/claim_output.py` | All calibration data | Claim (JSON) |
| **07** | Multi-Rate Scheduler | `src/scheduler/priority_scheduler.py` | Event R-score | Priority assignment |
| **07** | Queue Manager | `src/scheduler/queue_manager.py` | Jobs | Bounded queues |
| **07** | Backpressure | `src/scheduler/backpressure.py` | Queue depths + GPU util | Degradation triggers |
| **07** | GPU Distributor | `src/scheduler/gpu_distributor.py` | Work items | CUDA stream assignment |
| **08** | Deepfake Detection | `src/forensics/deepfake_detection.py` | Media frames | ForensicResult |
| **08** | C2PA/SynthID | `src/forensics/provenance_c2pa.py` | Media bytes | ProvenanceOutput |
| **08** | Audio Forensics | `src/forensics/audio_forensics.py` | Audio chunks | AudioForensicResult |
| **09** | Sports Reasoning | `src/domains/sports_reasoning.py` | Event + Rules | SportsEvent |
| **09** | General Multimedia | `src/domains/general_multimedia.py` | Event | GenericEvent |
| **09** | Synthetic Media | `src/domains/synthetic_media.py` | Forensic results | SyntheticVerdict |
| **10** | Data Schemas | `src/schemas/` | N/A (definitions) | Protobuf/JSON schemas |
| **10** | Hardware Topology | `src/config/hardware.yaml` | N/A (config) | GPU allocation |
| **10** | Deployment | `deployment/docker/` | N/A (infra) | Containers |

---

## 2. Complete Data Flow (Input → Output)

Every arrow below is a typed data flow. No untyped "stuff" passes between components.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           VIDEO / AUDIO INPUT                                       │
│  RTSP stream, file upload, HTTP stream, audio track                                 │
└───────────────────────────────┬─────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  INGEST LAYER                                                                        │
│                                                                                      │
│  FileIngestor / RTSPIngestor / AudioIngestor                                         │
│                                                                                      │
│  Output: RawFrame { frame_id: uint64, timestamp_ns: uint64, image: np.ndarray,      │
│                      stream_id: string }                                             │
│  Output: AudioChunk { chunk_id: uint64, audio: np.ndarray, sample_rate: int }        │
└───────────────────────────────┬─────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 01 — PERCEPTION (Critical Path: 8-12ms)                                      │
│                                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                           │
│  │ 01_detection │───▶│ 02_pose      │───▶│ 03_tracking  │                           │
│  │              │    │              │    │              │                            │
│  │ Input:       │    │ Input:       │    │ Input:       │                            │
│  │  GPU tensor  │    │  GPU tensor  │    │  Detection-  │                            │
│  │  [1,3,640,  │    │  + person    │    │  Batch       │                            │
│  │   640] FP16  │    │  crops       │    │              │                            │
│  │              │    │              │    │              │                            │
│  │ Output:      │    │ Output:      │    │ Output:      │                            │
│  │  Detection-  │    │  PoseBatch   │    │  Tracking-   │                           │
│  │  Batch       │    │  {poses[],   │    │  Output      │                            │
│  │  {det[],     │    │   scores[],  │    │  {tracks[],  │                            │
│  │   scores[],  │    │   bodyparts} │    │   stats}     │                            │
│  │   labels[]}  │    │              │    │              │                            │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                           │
│         │                   │                   │                                    │
│  ┌──────┴───────┐    ┌──────┴───────┐    ┌──────┴───────┐                           │
│  │ 04_segmen-   │    │ 05_ocr       │    │ 06_reid      │  (async, on-demand)      │
│  │ tation       │    │              │    │              │                            │
│  │ Input: tensor│    │ Input: tensor│    │ Input: crop  │                           │
│  │ + boxes      │    │ + ROI        │    │              │                            │
│  │ Output: masks│    │ Output: text │    │ Output:      │                           │
│  └──────────────┘    └──────────────┘    │  embed[512]  │                           │
│                                          └──────────────┘                           │
│  ┌──────────────┐    ┌──────────────┐                                               │
│  │ 07_audio     │    │ 08_camera    │  (async)                                      │
│  │ features     │    │ motion       │                                               │
│  │ Input: audio │    │ Input: frame │                                               │
│  │ chunk        │    │ pair         │                                               │
│  │ Output:      │    │ Output:      │                                               │
│  │  transcript  │    │  homography  │                                               │
│  └──────────────┘    └──────────────┘                                               │
│                                                                                      │
│  CRITICAL PATH OUTPUT: TrackingOutput (frame_id, tracks[], stats)                   │
│  ALL PERCEPTION OUTPUTS: DetectionBatch, PoseBatch, TrackingOutput,                 │
│                          SegmentationBatch, OCRBatch, Embeddings,                   │
│                          AudioOutput, CameraMotionOutput                             │
└───────────────────────────────┬─────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 02 — FUSION (2-6ms)                                                          │
│                                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐                │
│  │ 01_multimodal    │─▶│ 02_temporal      │─▶│ 03_cross_modal     │                │
│  │ fusion           │  │ fusion           │  │ alignment          │                 │
│  │                  │  │                  │  │                    │                  │
│  │ Input:           │  │ Input:           │  │ Input:             │                  │
│  │  vision features │  │  FusedFeatures   │  │  misaligned        │                  │
│  │  audio features  │  │  + history[5]    │  │  features from     │                  │
│  │  text features   │  │                  │  │  different         │                  │
│  │                  │  │ Output:          │  │  timestamps        │                  │
│  │ Output:          │  │  SmoothedFeatures│  │                    │                  │
│  │  FusedFeatures   │  │  + consistency   │  │ Output:            │                  │
│  │  + weights[3]    │  │  + motion_energy │  │  AlignedFeatures   │                  │
│  └──────────────────┘  └──────────────────┘  └────────────────────┘                │
│                                                                                      │
│  FUSION OUTPUT: AlignedFeatures { frame_id, features, modality_weights,             │
│                                    fusion_confidence }                               │
└───────────────────────────────┬─────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 03 — STATE (0.3-1ms)                                                         │
│                                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                  │
│  │ 01_world_state   │─▶│ 02_entity        │─▶│ 03_trajectory    │                  │
│  │                  │  │ tracker          │  │ model            │                   │
│  │ Input:           │  │                  │  │                  │                   │
│  │  TrackingOutput  │  │ Input:           │  │ Input:           │                   │
│  │  + PoseBatch     │  │  tracks + poses  │  │  entity history  │                   │
│  │  + FusedFeatures │  │                  │  │                  │                   │
│  │                  │  │ Output:          │  │ Output:          │                   │
│  │ Output:          │  │  Entity[] with   │  │  Trajectory      │                   │
│  │  WorldState-     │  │  lifecycle state │  │  {predicted_pos, │                   │
│  │  Snapshot        │  │                  │  │   variance,      │                   │
│  │  {entities,      │  └──────────────────┘  │   confidence}    │                   │
│  │   trajectories,  │                        └──────────────────┘                   │
│  │   relations,     │  ┌──────────────────┐                                         │
│  │   scene, events} │  │ 04_event         │                                         │
│  │                  │─▶│ detection        │                                         │
│  └──────────────────┘  │                  │                                          │
│                        │ Input:           │                                          │
│                        │  WorldState      │                                          │
│                        │  + rules         │                                          │
│                        │                  │                                          │
│                        │ Output:          │                                          │
│                        │  EventTrigger[]  │                                          │
│                        │  {event_type,    │                                          │
│                        │   r_score,       │                                          │
│                        │   entity_ids,    │                                          │
│                        │   confidence}    │                                          │
│                        └──────────────────┘                                         │
│                                                                                      │
│  STATE OUTPUT: WorldStateSnapshot + EventTrigger[]                                  │
└───────────────────────────────┬─────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
┌─────────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  LAYER 04 — MEMORY  │ │  LAYER 07 —      │ │  LAYER 09 —      │
│                     │ │  SCHEDULER       │ │  DOMAINS         │
│ 01_short_term       │ │                  │ │                  │
│  (<0.1ms, ring buf) │ │ 01_multi_rate    │ │ 01_sports        │
│                     │ │  scheduler       │ │  reasoning       │
│ 02_working_memory   │ │  (<0.5ms)        │ │  (10-50ms)       │
│  (1-5ms, top-10)    │ │                  │ │                  │
│                     │ │ 02_queue_mgr     │ │ 02_general       │
│ 03_long_term        │ │  (<0.1ms)        │ │  multimedia      │
│  (10-50ms, SQLite)  │ │                  │ │                  │
│                     │ │ 03_backpressure  │ │ 03_synthetic     │
│ 04_episodic         │ │  (<0.2ms)        │ │  media           │
│  (5-30ms, FAISS)    │ │                  │ │                  │
│                     │ │ 04_gpu_distrib   │ │                  │
│ Output feeds into:  │ │  (<0.2ms)        │ │                  │
│  Reasoning layer    │ │                  │ │                  │
│  (context for VLM)  │ │ Controls:       │ │ Output:          │
│                     │ │  priority of     │ │  DomainEvent     │
└─────────────────────┘ │  reasoning jobs  │ │  (with rules)    │
                        │                  │ └──────────────────┘
                        └──────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 05 — REASONING                                                               │
│                                                                                      │
│  ┌──────────────────┐                                                               │
│  │ 05_prediction    │──┐                                                            │
│  │ model            │  │                                                            │
│  │ (0.1ms)          │  │                                                            │
│  └──────────────────┘  │                                                            │
│                        │                                                            │
│  ┌──────────────────┐  │    ┌──────────────────┐                                   │
│  │ 01_hypothesis    │◀─┘───▶│ 04_evidence      │◀──┐                               │
│  │ engine           │       │ graph            │   │                               │
│  │ (5-15ms)         │──────▶│ (1-5ms)          │   │                               │
│  └────────┬─────────┘       └──────────────────┘   │                               │
│           │                                         │                               │
│           ▼                                         │                               │
│  ┌──────────────────┐       ┌──────────────────┐   │                               │
│  │ 02_fast_verifier │──────▶│ 03_deep_vlm      │───┘                               │
│  │ (1-5ms)          │       │ reasoner          │                                   │
│  │                  │       │ (800-3500ms)      │                                   │
│  │ SUPPORTED/       │       │ (async)           │                                   │
│  │ REFUTED ─────────│──────▶│                   │                                   │
│  │ INCONCLUSIVE ────│──────▶│ Produces:         │                                   │
│  │                  │       │ VLMReasoningOutput│                                   │
│  └────────┬─────────┘       └────────┬──────────┘                                  │
│           │                           │                                              │
│           └───────────┬───────────────┘                                              │
│                       │                                                              │
│  REASONING OUTPUT: VerificationResult {                                             │
│    hypothesis_id, verdict (SUPPORTED|REFUTED|INCONCLUSIVE),                         │
│    confidence (raw), evidence_used[], verification_ms                                │
│  }                                                                                   │
└───────────────────────┬─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 06 — CALIBRATION (3-8ms)                                                     │
│                                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                  │
│  │ 01_confidence    │─▶│ 02_conformal     │─▶│ 03_temperature   │                  │
│  │ decomposition    │  │ prediction       │  │ scaling          │                   │
│  │ (<1ms)           │  │ (2-5ms)          │  │ (<0.1ms)         │                   │
│  │                  │  │                  │  │                  │                   │
│  │ Input:           │  │ Input:           │  │ Input:           │                   │
│  │  5 raw confid-   │  │  Hypothesis[]    │  │  Logits          │                   │
│  │  ence components │  │  + calibration   │  │  + temperature T │                   │
│  │  (perception,    │  │  data            │  │                  │                   │
│  │   temporal,      │  │                  │  │ Output:          │                   │
│  │   motion,        │  │ Output:          │  │  Calibrated      │                   │
│  │   cross_modal,   │  │  PredictionSet   │  │  Probability     │                   │
│  │   reasoning)     │  │  {set[],         │  │                  │                   │
│  │                  │  │   coverage,      │  └──────────────────┘                   │
│  │ Output:          │  │   set_size}      │                                         │
│  │  DecomposedConf  │  └──────────────────┘                                         │
│  │  {perception,    │                                                               │
│  │   temporal,      │  ┌──────────────────┐                                         │
│  │   motion,        │  │ 04_claim_output  │                                         │
│  │   cross_modal,   │◀─│                  │                                         │
│  │   reasoning,     │  │ Input: ALL above │                                         │
│  │   overall}       │  │                  │                                         │
│  └──────────────────┘  │ Output:          │                                         │
│                        │  Claim (JSON)    │                                         │
│                        │  {id, claim_text,│                                         │
│                        │   confidence,    │                                         │
│                        │   prediction_set,│                                         │
│                        │   evidence[],    │                                         │
│                        │   staleness,     │                                         │
│                        │   provenance}    │                                         │
│                        └────────┬─────────┘                                        │
│                                 │                                                    │
│  CALIBRATION OUTPUT: Claim (structured JSON)                                        │
└─────────────────────────────────┬───────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼──────────────────────┐
              │                   │                       │
              ▼                   ▼                       ▼
┌─────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│  REST API           │ │  WebSocket       │ │  Kafka Producer      │
│  (FastAPI)          │ │  (Live stream)   │ │  (Async messages)    │
│                     │ │                  │ │                      │
│  GET /api/v1/claims │ │  WS /api/v1/     │ │  Topic:              │
│  GET /api/v1/state  │ │  stream/live     │ │  claims.published    │
│  GET /api/v1/health │ │                  │ │                      │
│                     │ │  Pushes Claim    │ │  Publishes Claim     │
│  Returns: Claim[]   │ │  as SSE event    │ │  as Protobuf binary  │
└─────────────────────┘ └──────────────────┘ └──────────────────────┘
```

---

## 3. Data Flow by Priority Path

### Path A: Critical Path (30 FPS, must complete in <33ms)

```
Frame arrives
  → FileIngestor.decode()                    [1-3ms]   → RawFrame
  → Preprocessor.preprocess()                [1-2ms]   → GPU tensor [1,3,640,640]
  → TensorRTDetector.detect()                [2.3-6.8ms] → DetectionBatch
  → ByteTrack.update()                       [0.2-0.5ms] → TrackingOutput
  → PoseEstimator.estimate()                 [2.39-5ms]  → PoseBatch
  → WorldState.update()                      [0.3-1ms]   → WorldStateSnapshot
  → EventDetector.detect()                   [0.1-0.3ms] → EventTrigger[]
  → RingBuffer.write()                       [0.2-0.5ms] → Published

TOTAL: 8-12ms (leaves 21-25ms headroom)
```

### Path B: Fast Reasoning (5-10 Hz, triggered by R score ≥ 0.5)

```
EventTrigger (R ≥ 0.5)
  → HypothesisEngine.generate()              [5-15ms]  → Hypothesis[]
  → FastVerifier.verify()                    [1-5ms]   → VerificationResult
  → IF SUPPORTED/REFUTED:
      → ConfidenceDecomposition.decompose()  [<1ms]    → DecomposedConfidence
      → ConformalPredict.predict()           [2-5ms]   → PredictionSet
      → TemperatureScaling.scale()           [<0.1ms]  → CalibratedProb
      → ClaimOutput.assemble()               [1-2ms]   → Claim
  → IF INCONCLUSIVE: → Path C

TOTAL FAST PATH: 17-40ms
```

### Path C: Deep Reasoning (0.3-0.5 Hz, async)

```
INCONCLUSIVE from fast verify
  → VLMReasoner.select_frames()              [5-10ms]  → key_frames
  → VLMReasoner.build_prompt()               [1-2ms]   → prompt_string
  → VLMReasoner.inference()                  [800-3500ms] → raw_response
  → VLMReasoner.parse_response()             [5-10ms]  → VLMReasoningOutput
  → EvidenceGraph.add()                      [1-5ms]   → updated graph
  → ConfidenceDecomposition.decompose()      [<1ms]    → DecomposedConfidence
  → ConformalPredict.predict()               [2-5ms]   → PredictionSet
  → TemperatureScaling.scale()               [<0.1ms]  → CalibratedProb
  → ClaimOutput.assemble()                   [1-2ms]   → Claim

TOTAL DEEP PATH: 814-3540ms (async, non-blocking)
```

### Path D: Async Perception (event-triggered, not per-frame)

```
Event: scene_cut / new_entity / user_request
  → Segmenter.segment()                      [5-10ms]  → MaskResult[]
  → OCR.extract()                            [15-40ms] → OCRResult[]
  → ReID.match()                             [3-8ms]   → ReIDOutput
  → AudioFeatures.extract()                  [55-110ms] → AudioOutput
  → CameraMotion.estimate()                  [1-3ms]   → CameraMotionOutput

ALL feed into Fusion layer → State layer
```

### Path E: Forensics (async, event-triggered)

```
Event: scene_cut / new_entity / user_request
  → C2PA.provenance_check()                  [50-200ms] → ProvenanceOutput
  → IF no C2PA manifest:
      → DeepfakeDetection.ensemble()         [200-2000ms] → ForensicResult
  → AudioForensics.analyze()                 [100-500ms]  → AudioForensicResult
  → SyntheticMediaDomain.classify()          [10-30ms]    → SyntheticVerdict
  → Feed into Claim output (domain metadata)

ALL async, never block critical path
```

---

## 4. Component Interface Contracts

### What passes between layers (typed, no dictionaries)

```
INGESTOR OUTPUT → PERCEPTION INPUT:
  RawFrame {
    frame_id:      uint64
    timestamp_ns:  uint64
    image:         Tensor[H, W, 3] uint8  (on GPU)
    stream_id:     string
  }

DETECTION OUTPUT → POSE/TRACKING/FUSION INPUT:
  DetectionBatch {
    frame_id:      uint64
    timestamp_ns:  uint64
    detections:    Detection[]
    model_id:      string
    inference_ms:  float
  }

  Detection {
    bbox_norm:     [4]float               (x_center, y_center, w, h)
    class_id:      uint16
    class_name:    string
    score:         float
    feature_ref:   string
  }

POSE OUTPUT → WORLD STATE INPUT:
  PoseBatch {
    frame_id:      uint64
    poses:         PoseResult[]
    model_id:      string
    inference_ms:  float
  }

  PoseResult {
    entity_id:     string
    keypoints_2d:  [17][2]float
    keypoint_scores: [17]float
    pose_score:    float
    occluded_joints: uint64
  }

TRACKING OUTPUT → WORLD STATE/ENTITY TRACKER INPUT:
  TrackingOutput {
    frame_id:      uint64
    tracks:        Track[]
    statistics:    TrackingStats
  }

  Track {
    entity_id:     string
    bbox_norm:     [4]float
    class_id:      uint16
    age:           uint32
    hits:          uint32
    time_since_update: uint32
    velocity:      [2]float
    score:         float
    is_occluded:   bool
  }

WORLD STATE OUTPUT → REASONING/MEMORY/SCHEDULER INPUT:
  WorldStateSnapshot {
    frame_id:      uint64
    timestamp_ns:  uint64
    entities:      Entity[]
    trajectories:  Trajectory[]
    relations:     Relation[]
    scene:         Scene
    events:        Event[]
    uncertainty:   UncertaintySummary
    staleness_ms:  float
  }

EVENT TRIGGER → SCHEDULER/REASONING INPUT:
  EventTrigger {
    event_id:      string
    event_type:    string
    timestamp_ns:  uint64
    entity_ids:    string[]
    r_score:       float
    evidence:      Evidence[]
    confidence:    float
  }

REASONING OUTPUT → CALIBRATION INPUT:
  VerificationResult {
    hypothesis_id: string
    verdict:       enum (SUPPORTED|REFUTED|INCONCLUSIVE)
    confidence:    float
    evidence_used: Evidence[]
    verification_ms: float
    needs_deep:    bool
  }

CALIBRATION OUTPUT → FINAL OUTPUT:
  Claim {
    id:            string
    schema_version: string
    timestamp_ns:  uint64
    published_ts_ns: uint64
    claim_text:    string
    claim_type:    enum
    epistemic_status: enum
    confidence:    ConfidenceDecomposition
    conformal_prediction_set_alpha_05: string[]
    conformal_prediction_set_alpha_10: string[]
    prediction_set_size_at_alpha_05: int
    evidence:      Evidence[]
    provenance:    Provenance
    output_quality: OutputQuality
  }
```

---

## 5. Code File → Architecture Layer Map

### Directory Structure Matches Architecture Exactly

```
src/
├── ingestor/                    ← (not in architecture layers — pre-layer)
│   ├── file_ingestor.py         ← Video file reader
│   ├── rtsp_ingestor.py         ← RTSP stream reader
│   ├── audio_ingestor.py        ← Audio stream splitter
│   └── frame_buffer.py          ← Pre-allocated GPU buffer pool
│
├── perception/                  ← LAYER 01 — Perception
│   ├── detector.py              ← 01_detection
│   ├── pose_estimator.py        ← 02_pose_estimation
│   ├── tracker.py               ← 03_tracking
│   ├── segmenter.py             ← 04_segmentation
│   ├── ocr.py                   ← 05_ocr
│   ├── reid.py                  ← 06_reid
│   ├── audio_features.py        ← 07_audio_features
│   ├── camera_motion.py         ← 08_camera_motion
│   └── pipeline.py              ← Perception orchestrator
│
├── fusion/                      ← LAYER 02 — Fusion
│   ├── multimodal_fusion.py     ← 01_multimodal_fusion
│   ├── temporal_fusion.py       ← 02_temporal_fusion
│   └── cross_modal_alignment.py ← 03_cross_modal_alignment
│
├── state/                       ← LAYER 03 — State
│   ├── world_state.py           ← 01_world_state
│   ├── entity_tracker.py        ← 02_entity_tracker
│   ├── trajectory_model.py      ← 03_trajectory_model
│   └── event_detector.py        ← 04_event_detection
│
├── memory/                      ← LAYER 04 — Memory
│   ├── short_term.py            ← 01_short_term
│   ├── working_memory.py        ← 02_working_memory
│   ├── long_term.py             ← 03_long_term
│   └── episodic.py              ← 04_episodic_memory
│
├── reasoning/                   ← LAYER 05 — Reasoning
│   ├── hypothesis_engine.py     ← 01_hypothesis_engine
│   ├── fast_verifier.py         ← 02_fast_verifier
│   ├── vlm_reasoner.py          ← 03_deep_vlm_reasoner
│   ├── evidence_graph.py        ← 04_evidence_graph
│   └── prediction_model.py      ← 05_prediction_model
│
├── calibration/                 ← LAYER 06 — Calibration
│   ├── confidence_decomposition.py ← 01_confidence_decomposition
│   ├── conformal.py             ← 02_conformal_prediction
│   ├── temperature_scaling.py   ← 03_temperature_scaling
│   └── claim_output.py          ← 04_claim_output
│
├── scheduler/                   ← LAYER 07 — Scheduler
│   ├── priority_scheduler.py    ← 01_multi_rate_scheduler
│   ├── queue_manager.py         ← 02_queue_management
│   ├── backpressure.py          ← 03_backpressure
│   └── gpu_distributor.py       ← 04_gpu_work_distribution
│
├── forensics/                   ← LAYER 08 — Forensics
│   ├── deepfake_detection.py    ← 01_deepfake_detection
│   ├── provenance_c2pa.py       ← 02_provenance_c2pa
│   └── audio_forensics.py       ← 03_audio_forensics
│
├── domains/                     ← LAYER 09 — Domains
│   ├── sports_reasoning.py      ← 01_sports_reasoning
│   ├── general_multimedia.py    ← 02_general_multimedia
│   └── synthetic_media.py       ← 03_synthetic_media
│
├── schemas/                     ← LAYER 10 — Data Schemas
│   ├── perception.proto
│   ├── state.proto
│   ├── reasoning.proto
│   ├── calibration.proto
│   └── claim.proto
│
├── output/                      ← Output delivery (not architecture layer)
│   ├── rest_api.py
│   ├── websocket_server.py
│   └── kafka_producer.py
│
├── monitoring/                  ← Observability (not architecture layer)
│   ├── metrics.py
│   └── health.py
│
├── config/
│   ├── hardware.yaml            ← LAYER 10 — Hardware Topology
│   └── pipeline.yaml            ← Full pipeline configuration
│
├── pipeline.py                  ← MAIN ORCHESTRATOR
└── server.py                    ← ENTRY POINT
```

---

## 6. The Orchestrator (How Everything Connects)

```python
# src/pipeline.py

class MultimodalReasoner:
    """Main orchestrator. Connects all 10 architecture layers."""

    def __init__(self, config_path):
        cfg = load_config(config_path)

        # Layer 01: Perception
        self.detector = TensorRTDetector(cfg.detection)
        self.pose_estimator = PoseEstimator(cfg.pose)
        self.tracker = ByteTrackWrapper(cfg.tracker)

        # Layer 02: Fusion
        self.fusion = MultiModalFusion(cfg.fusion)
        self.temporal = TemporalFusion(cfg.temporal)
        self.alignment = CrossModalAlignment(cfg.alignment)

        # Layer 03: State
        self.world_state = WorldState(cfg.state)
        self.entity_tracker = EntityTracker(cfg.entity)
        self.trajectory = TrajectoryModel(cfg.trajectory)
        self.event_detector = EventDetector(cfg.events)

        # Layer 04: Memory
        self.short_term = ShortTermMemory(cfg.memory.short_term)
        self.working_memory = WorkingMemory(cfg.memory.working)
        self.long_term = LongTermMemory(cfg.memory.long_term)
        self.episodic = EpisodicMemory(cfg.memory.episodic)

        # Layer 05: Reasoning
        self.hypothesis_engine = HypothesisEngine(cfg.reasoning)
        self.fast_verifier = FastVerifier(cfg.reasoning)
        self.vlm_reasoner = VLMReasoner(cfg.reasoning.vlm)
        self.evidence_graph = EvidenceGraph(cfg.reasoning.evidence)
        self.prediction_model = PredictionModel(cfg.reasoning.prediction)

        # Layer 06: Calibration
        self.confidence_decomp = ConfidenceDecomposition(cfg.calibration)
        self.conformal = ConformalPrediction(cfg.calibration.conformal)
        self.temperature = TemperatureScaling(cfg.calibration.temperature)
        self.claim_output = ClaimOutput(cfg.calibration.claim)

        # Layer 07: Scheduler
        self.scheduler = MultiRateScheduler(cfg.scheduler)
        self.queue_manager = QueueManager(cfg.scheduler.queue)
        self.backpressure = Backpressure(cfg.scheduler.backpressure)
        self.gpu_distributor = GPUDistributor(cfg.scheduler.gpu)

        # Layer 08: Forensics
        self.deepfake = DeepfakeDetection(cfg.forensics.deepfake)
        self.c2pa = C2PAVerifier(cfg.forensics.c2pa)
        self.audio_forensics = AudioForensics(cfg.forensics.audio)

        # Layer 09: Domains
        self.sports = SportsReasoning(cfg.domains.sports)
        self.general = GeneralMultimedia(cfg.domains.general)
        self.synthetic = SyntheticMediaDomain(cfg.domains.synthetic)

        # Output
        self.output = ClaimPublisher(cfg.output)

    def process_frame(self, raw_frame):
        """CRITICAL PATH: frame → perception → state → event (<12ms)"""

        # Layer 01
        detections = self.detector.detect(raw_frame)
        poses = self.pose_estimator.estimate(raw_frame, detections)
        tracks = self.tracker.update(detections)

        # Layer 03
        snapshot = self.world_state.update(tracks, poses)
        events = self.event_detector.detect(snapshot)

        # Store in memory
        self.short_term.write(snapshot)

        # Layer 07: Schedule reasoning
        for event in events:
            priority = self.scheduler.compute_priority(event)
            if priority >= 0.5:
                self._reason(event, snapshot)

        return snapshot, events

    def _reason(self, event, snapshot):
        """REASONING PATH: event → hypothesis → verify → claim"""

        # Layer 04: Get context
        context = self.working_memory.get_context()

        # Layer 05: Hypothesis + Fast Verify
        hypotheses = self.hypothesis_engine.generate(event, snapshot, context)
        for hyp in hypotheses:
            verdict = self.fast_verifier.verify(hyp, snapshot)

            if verdict.needs_deep:
                verdict = self.vlm_reasoner.reason(hyp, snapshot, context)
                self.evidence_graph.add(hyp, verdict)

            # Layer 06: Calibration
            decomposed = self.confidence_decomp.decompose(verdict)
            pred_set = self.conformal.predict(hypotheses)
            calibrated = self.temperature.scale(decomposed)
            claim = self.claim_output.assemble(
                verdict, decomposed, pred_set, calibrated
            )

            # Output
            self.output.publish(claim)
```

---

## 7. Build Order (Architecture-Aligned)

Build in this order. Each step produces a testable artifact.

| Step | What to Build | Architecture Layer | Test |
|---|---|---|---|
| 1 | `ingestor/file_ingestor.py` | Pre-layer | Can read video, output frames |
| 2 | `perception/detector.py` | L01-01 | Detects objects in single frame |
| 3 | `perception/tracker.py` | L01-03 | Tracks objects across frames |
| 4 | `perception/pose_estimator.py` | L01-02 | Estimates pose on person crops |
| 5 | `perception/pipeline.py` | L01 orchestrator | Full perception at 30 FPS |
| 6 | `state/world_state.py` | L03-01 | Ring buffer updates correctly |
| 7 | `state/event_detector.py` | L03-04 | Detects events, computes R score |
| 8 | `scheduler/priority_scheduler.py` | L07-01 | Assigns priority to events |
| 9 | `reasoning/hypothesis_engine.py` | L05-01 | Generates hypotheses from events |
| 10 | `reasoning/fast_verifier.py` | L05-02 | Verifies hypotheses in 1-5ms |
| 11 | `reasoning/vlm_reasoner.py` | L05-03 | VLM produces reasoning output |
| 12 | `reasoning/evidence_graph.py` | L05-04 | Graph accumulates evidence |
| 13 | `memory/short_term.py` | L04-01 | Ring buffer stores snapshots |
| 14 | `memory/working_memory.py` | L04-02 | Active hypotheses managed |
| 15 | `calibration/confidence_decomposition.py` | L06-01 | 5-component decomposition |
| 16 | `calibration/conformal.py` | L06-02 | Prediction sets with coverage |
| 17 | `calibration/temperature_scaling.py` | L06-03 | Temperature calibration |
| 18 | `calibration/claim_output.py` | L06-04 | Structured claim JSON |
| 19 | `pipeline.py` | Full system | End-to-end on video file |
| 20 | `output/rest_api.py` | Output | REST endpoints serving claims |
| 21 | `monitoring/metrics.py` | Observability | Prometheus metrics live |
| 22 | Docker + deployment | L10 | Container runs pipeline |

**Skip for now (Phase 2+):**
- Fusion layer (L02) — works without it initially
- Segmentation, OCR, ReID, Audio, Camera Motion (L01 async) — add after core works
- Forensics (L08) — add after reasoning works
- Domains (L09) — add after reasoning works
- Long-term/Episodic memory (L04) — add after short-term works
- Kafka output — add after REST API works

---

## 8. Summary: What Goes Where

```
Architecture Layer          Code Module                    Build Step
─────────────────────────────────────────────────────────────────────
Pre-layer (Ingest)          src/ingestor/                  Step 1
L01 Perception (critical)   src/perception/detector.py     Step 2
                            src/perception/tracker.py      Step 3
                            src/perception/pose_estimator.py Step 4
                            src/perception/pipeline.py     Step 5
L01 Perception (async)      src/perception/segmenter.py    Phase 2
                            src/perception/ocr.py          Phase 2
                            src/perception/reid.py         Phase 2
                            src/perception/audio_features.py Phase 2
                            src/perception/camera_motion.py Phase 2
L02 Fusion                  src/fusion/                    Phase 2
L03 State                   src/state/world_state.py       Step 6
                            src/state/event_detector.py    Step 7
                            src/state/entity_tracker.py    Phase 2
                            src/state/trajectory_model.py  Phase 2
L04 Memory (short)          src/memory/short_term.py       Step 13
L04 Memory (working)        src/memory/working_memory.py   Step 14
L04 Memory (long+episodic)  src/memory/long_term.py        Phase 2
                            src/memory/episodic.py         Phase 2
L05 Reasoning               src/reasoning/hypothesis_engine.py Step 9
                            src/reasoning/fast_verifier.py     Step 10
                            src/reasoning/vlm_reasoner.py     Step 11
                            src/reasoning/evidence_graph.py   Step 12
                            src/reasoning/prediction_model.py Phase 2
L06 Calibration             src/calibration/confidence_decomposition.py Step 15
                            src/calibration/conformal.py           Step 16
                            src/calibration/temperature_scaling.py Step 17
                            src/calibration/claim_output.py        Step 18
L07 Scheduler               src/scheduler/priority_scheduler.py    Step 8
                            src/scheduler/queue_manager.py         Phase 2
                            src/scheduler/backpressure.py          Phase 2
                            src/scheduler/gpu_distributor.py       Phase 2
L08 Forensics               src/forensics/                  Phase 3
L09 Domains                 src/domains/                    Phase 3
L10 Infrastructure          src/schemas/ + deployment/      Step 22
Output                      src/output/                     Step 20
Monitoring                  src/monitoring/                 Step 21
```
