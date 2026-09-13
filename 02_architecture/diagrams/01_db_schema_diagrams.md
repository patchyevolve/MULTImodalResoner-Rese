# Database Schema Diagrams

> All schemas use Mermaid ER diagram syntax. Covers every data store in the system.

---

## 1. Entity Store (SQLite — Primary Entity Database)

```mermaid
erDiagram
    ENTITY {
        string id PK "UUID, persistent across sessions"
        string canonical_id UK "cross-view identity"
        string entity_class "person | ball | object"
        string name "if known"
        float first_seen_ns "creation timestamp"
        float last_seen_ns "last update"
        int lifetime_frames "total frames observed"
        string tracking_state "TENTATIVE | CONFIRMED | ACTIVE | LOST"
        float occlusion_level "0=visible, 1=fully occluded"
        float confidence "aggregate confidence"
        json attributes "team, jersey_number, role, etc"
        float created_at "row creation time"
        float updated_at "last row update"
    }

    ENTITY_STATE {
        string id PK "auto-increment"
        string entity_id FK "→ ENTITY.id"
        uint64 frame_id "frame reference"
        float timestamp_ns "nanosecond timestamp"
        float bbox_x "normalized center x"
        float bbox_y "normalized center y"
        float bbox_w "normalized width"
        float bbox_h "normalized height"
        json pose_2d "keypoints array"
        json pose_3d "3D joints if available"
        float velocity_x "normalized units/frame"
        float velocity_y "normalized units/frame"
        float acceleration_x "normalized units/frame²"
        float acceleration_y "normalized units/frame²"
        float confidence "detection confidence"
        float occlusion_level "per-state occlusion"
        bool is_keyframe "keyframe flag"
    }

    ENTITY_IDENTITY {
        string id PK "auto-increment"
        string entity_id FK "→ ENTITY.id"
        string canonical_id "persistent cross-view ID"
        json aliases "track IDs from different cameras"
        blob reid_features "appearance embedding array"
        float last_reid_match "timestamp of last re-ID"
        int match_count "total re-ID matches"
    }

    ENTITY_EMBEDDING {
        string id PK "auto-increment"
        string entity_id FK "→ ENTITY.id"
        blob embedding "512-dim float array"
        string source_model "OSNet | TransReID"
        float timestamp_ns "when extracted"
        float confidence "embedding quality"
    }

    ENTITY ||--o{ ENTITY_STATE : "has states"
    ENTITY ||--o{ ENTITY_IDENTITY : "has identity"
    ENTITY ||--o{ ENTITY_EMBEDDING : "has embeddings"
```

---

## 2. Event Store (SQLite — Event Database)

```mermaid
erDiagram
    EVENT {
        string id PK "UUID"
        string event_type "goal | foul | offside | anomaly | scene_cut"
        float timestamp_ns "when event occurred"
        float confidence "event confidence"
        float priority_score "R score 0-1"
        string status "TRIGGERED | VERIFIED | REFUTED | INCONCLUSIVE"
        json entities "involved entity IDs"
        string domain "sports | news | general"
        json metadata "event-specific data"
        float created_at "row creation time"
    }

    EVENT_EVIDENCE {
        string id PK "auto-increment"
        string event_id FK "→ EVENT.id"
        string evidence_type "TRAJECTORY | POSE | RULE | VLM | AUDIO"
        string description "human-readable"
        float weight "log-likelihood ratio"
        json data "supporting data"
        string source_model "model that produced this"
        float timestamp_ns "evidence timestamp"
    }

    EVENT_FRAMES {
        string id PK "auto-increment"
        string event_id FK "→ EVENT.id"
        uint64 frame_id "frame reference"
        float timestamp_ns "frame timestamp"
        string role "key_frame | context | before | after"
        int sequence_order "temporal order"
    }

    EVENT ||--o{ EVENT_EVIDENCE : "has evidence"
    EVENT ||--o{ EVENT_FRAMES : "has frames"
```

---

## 3. Claim Store (SQLite — Final Claims)

```mermaid
erDiagram
    CLAIM {
        string id PK "UUID"
        string schema_version "1.0.0"
        float timestamp_ns "observation time"
        float published_ts_ns "publication time"
        string claim_text "natural language claim"
        string claim_type "OBSERVED | INFERRED | PREDICTED | HYPOTHESIS | CAUSAL | COUNTERFACTUAL"
        string epistemic_status "CONFIRMED | PROBABLE | POSSIBLE | SPECULATIVE | UNKNOWN"
        float confidence_perception "0-1"
        float confidence_temporal "0-1"
        float confidence_motion "0-1"
        float confidence_cross_modal "0-1"
        float confidence_reasoning "0-1"
        float confidence_calibrated "0-1"
        float confidence_overall "0-1"
        string calibration_method "temperature | conformal | isotonic"
        int prediction_set_size_alpha05 "size at α=0.05"
        json conformal_set_alpha05 "hypothesis IDs"
        json conformal_set_alpha10 "hypothesis IDs"
        string producing_model "model name"
        string producing_model_version "model version"
        string checkpoint_sha256 "weight hash"
        float claim_staleness_ms "time from snapshot"
        bool stale "true if >5s"
        string source_stream "camera/stream ID"
        json source_frames "frame IDs used"
        string event_id "triggering event"
        string hypothesis_id "source hypothesis"
        string domain "sports | news | general"
        json domain_metadata "domain-specific fields"
        float created_at "row creation time"
    }

    CLAIM_EVIDENCE {
        string id PK "auto-increment"
        string claim_id FK "→ CLAIM.id"
        string role "support | contradiction"
        string evidence_id FK "→ EVENT_EVIDENCE.id (optional)"
        string description "evidence description"
        float weight "log-likelihood ratio"
        string source_model "model that produced this"
    }

    CLAIM_uncertainty_sources {
        string id PK "auto-increment"
        string claim_id FK "→ CLAIM.id"
        string source "uncertainty source description"
    }

    CLAIM ||--o{ CLAIM_EVIDENCE : "has evidence"
    CLAIM ||--o{ CLAIM_uncertainty_sources : "has uncertainty sources"
```

---

## 4. Hypothesis Store (In-Memory — Working Memory)

```mermaid
erDiagram
    HYPOTHESIS {
        string id PK "UUID"
        string claim "natural language hypothesis"
        string type "OBSERVED | INFERRED | PREDICTED | CAUSAL | COUNTERFACTUAL"
        string epistemic_status "CONFIRMED | PROBABLE | POSSIBLE | SPECULATIVE"
        string status "CANDIDATE | SUPPORTED | REFUTED | INCONCLUSIVE"
        float posterior "Bayesian posterior 0-1"
        int rank "current ranking"
        float created_at "creation timestamp"
        float age_ms "time since creation"
        float last_updated "last posterior update"
        string generation_method "rule | pattern | vlm"
        string event_id "triggering event"
    }

    HYPOTHESIS_EVIDENCE {
        string id PK "auto-increment"
        string hypothesis_id FK "→ HYPOTHESIS.id"
        string evidence_id FK "→ EVENT_EVIDENCE.id"
        string role "support | contradiction"
        float weight "log-likelihood ratio"
        float added_at "when evidence was linked"
    }

    HYPOTHESIS ||--o{ HYPOTHESIS_EVIDENCE : "has evidence"
```

---

## 5. Evidence Graph (In-Memory — Graph Structure)

```mermaid
erDiagram
    EVIDENCE_NODE {
        string id PK "UUID"
        string node_type "OBSERVATION | INFERENCE | RULE | VLM_OUTPUT"
        string claim "evidence claim text"
        float confidence "0-1"
        json source "model IDs that produced this"
        float timestamp_ns "when evidence was created"
        json metadata "additional data"
        float created_at "row creation time"
    }

    EVIDENCE_EDGE {
        string id PK "auto-increment"
        string source_node_id FK "→ EVIDENCE_NODE.id"
        string target_node_id FK "→ EVIDENCE_NODE.id"
        string relation "SUPPORTS | CONTRADICTS | ENABLES | WEAKENS"
        float weight "log-likelihood ratio"
        string weight_estimator "how weight was computed"
        float created_at "row creation time"
    }

    EVIDENCE_NODE ||--o{ EVIDENCE_EDGE : "as source"
    EVIDENCE_NODE ||--o{ EVIDENCE_EDGE : "as target"
```

---

## 6. Calibration Store (SQLite — Calibration Data)

```mermaid
erDiagram
    CALIBRATION_SET {
        string id PK "UUID"
        string model_name "model being calibrated"
        string model_version "model version"
        string calibration_method "temperature | conformal | isotonic"
        float alpha "significance level"
        int sample_count "calibration set size"
        float created_at "calibration timestamp"
        bool active "currently in use"
    }

    CALIBRATION_POINT {
        string id PK "auto-increment"
        string calibration_set_id FK "→ CALIBRATION_SET.id"
        string hypothesis_id "reference hypothesis"
        float nonconformity_score "how surprising"
        bool is_correct "ground truth"
        float timestamp_ns "when scored"
    }

    TEMPERATURE_PARAMS {
        string id PK "auto-increment"
        string model_name "model being calibrated"
        string model_version "model version"
        float temperature "learned T parameter"
        float validation_nll "negative log-likelihood"
        float validation_ece "expected calibration error"
        int sample_count "validation set size"
        float trained_at "training timestamp"
        bool active "currently in use"
    }

    CALIBRATION_SET ||--o{ CALIBRATION_POINT : "has points"
```

---

## 7. Long-Term Entity Profiles (SQLite + FAISS)

```mermaid
erDiagram
    ENTITY_PROFILE {
        string id PK "canonical_id (UUID)"
        string name "if known"
        string entity_class "person | ball | object"
        json attributes "persistent attributes"
        blob average_embedding "512-dim average"
        int event_count "total events involving this entity"
        float first_seen_ns "first observation"
        float last_seen_ns "last observation"
        json sessions "session summaries"
        float created_at "row creation time"
        float updated_at "last update"
    }

    ENTITY_SESSION {
        string id PK "auto-increment"
        string entity_id FK "→ ENTITY_PROFILE.id"
        string session_id "session identifier"
        float start_ns "session start"
        float end_ns "session end"
        int frame_count "frames in session"
        json highlights "key moments"
        float summary_embedding "256-dim session summary"
    }

    ENTITY_TRAIT {
        string id PK "auto-increment"
        string entity_id FK "→ ENTITY_PROFILE.id"
        string trait_name "team | role | jersey_number | skill"
        string trait_value "trait value"
        float confidence "trait confidence"
        int observation_count "times observed"
        float last_observed "timestamp"
    }

    ENTITY_PROFILE ||--o{ ENTITY_SESSION : "has sessions"
    ENTITY_PROFILE ||--o{ ENTITY_TRAIT : "has traits"
```

---

## 8. Event Summary Store (SQLite — Event Summaries)

```mermaid
erDiagram
    EVENT_SUMMARY {
        string id PK "UUID"
        string event_id FK "→ EVENT.id (optional)"
        float timestamp_ns "event timestamp"
        string event_type "goal | foul | anomaly | etc"
        string description "natural language summary"
        json entities "involved entity IDs"
        float confidence "summary confidence"
        string video_ref "pointer to video segment"
        blob embedding "256-dim for retrieval"
        string domain "sports | news | general"
        float created_at "row creation time"
    }

    EVENT_SUMMARY_TAG {
        string id PK "auto-increment"
        string summary_id FK "→ EVENT_SUMMARY.id"
        string tag_name "tag category"
        string tag_value "tag value"
    }

    EVENT_SUMMARY ||--o{ EVENT_SUMMARY_TAG : "has tags"
```

---

## 9. Episode Store (SQLite — Episodic Memory)

```mermaid
erDiagram
    EPISODE {
        string id PK "UUID"
        float start_ns "episode start time"
        float end_ns "episode end time"
        json event_ids "ordered event IDs"
        string summary "natural language summary"
        json entities "involved entity IDs"
        string domain "sports | news | general"
        json key_frames "frame IDs for visual summary"
        blob embedding "256-dim for retrieval"
        float duration_ms "episode duration"
        int event_count "number of events"
        float created_at "row creation time"
    }

    EPISODE_EVENT {
        string id PK "auto-increment"
        string episode_id FK "→ EPISODE.id"
        string event_id FK "→ EVENT.id"
        int sequence_order "temporal order in episode"
        float relative_time_ms "time from episode start"
    }

    EPISODE ||--o{ EPISODE_EVENT : "contains events"
```

---

## 10. Forensic Results Store (SQLite)

```mermaid
erDiagram
    FORENSIC_RESULT {
        string id PK "UUID"
        string media_id "reference to media content"
        string media_type "IMAGE | VIDEO | AUDIO"
        float timestamp_ns "analysis time"
        bool is_synthetic "final verdict"
        float confidence "ensemble confidence"
        string manipulation_type "face_swap | voice_clone | ai_generated"
        string detection_method "ensemble | provenance | combined"
        json detector_scores "per-detector scores"
        json evidence "detected artifacts"
        string recommendation "trust | flag_for_review | reject"
        float latency_ms "analysis latency"
        float created_at "row creation time"
    }

    FORENSIC_PROVENANCE {
        string id PK "auto-increment"
        string result_id FK "→ FORENSIC_RESULT.id"
        bool has_c2pa "C2PA manifest present"
        bool c2pa_valid "signature valid"
        bool synthid_detected "SynthID watermark found"
        string issuer "signing organization"
        float issued_at "certificate issue time"
        json claims "provenance claims"
        float confidence "provenance confidence"
    }

    FORENSIC_DETECTOR {
        string id PK "auto-increment"
        string result_id FK "→ FORENSIC_RESULT.id"
        string detector_name "CNNSpot | ForgeLens | etc"
        string detector_type "spatial | temporal | spectral | semantic"
        float score "detector output score"
        float auc "detector AUC on this sample"
        float latency_ms "detector inference time"
    }

    FORENSIC_RESULT ||--o{ FORENSIC_PROVENANCE : "has provenance"
    FORENSIC_RESULT ||--o{ FORENSIC_DETECTOR : "has detector scores"
```

---

## 11. Configuration Store (JSON/SQLite)

```mermaid
erDiagram
    SYSTEM_CONFIG {
        string id PK "config identifier"
        string config_version "semver"
        string environment "dev | staging | production"
        json gpu_config "GPU allocation settings"
        json model_config "model selections"
        json scheduler_config "scheduler parameters"
        json queue_config "queue depths and policies"
        json calibration_config "calibration parameters"
        json domain_config "domain-specific rules"
        bool active "currently loaded"
        float created_at "creation time"
        float updated_at "last update"
    }

    MODEL_VERSION {
        string id PK "auto-increment"
        string model_name "RF-DETR-S | DETRPose-S | Qwen3-VL"
        string model_version "version string"
        string checkpoint_path "file path"
        string checkpoint_sha256 "hash"
        float loaded_at "when loaded into GPU"
        float gpu_memory_mb "memory usage"
        bool active "currently in use"
    }

    AUDIT_LOG {
        string id PK "auto-increment"
        string event_type "config_change | model_load | error | backpressure"
        string component "which component"
        json details "event details"
        float timestamp_ns "event time"
        string severity "INFO | WARNING | ERROR | CRITICAL"
    }
```

---

## 12. Ring Buffer Schema (In-Memory — Not SQLite)

```mermaid
graph TB
    subgraph STATE_RING["State Ring Buffer (30 slots)"]
        SR0["[0] frame_id=1000<br/>timestamp=12530000000<br/>entities=26<br/>events=2"]
        SR1["[1] frame_id=1001<br/>timestamp=12530333333<br/>entities=26<br/>events=1"]
        SR2["[2] frame_id=1002<br/>timestamp=12530666666<br/>entities=25<br/>events=0"]
        SR3["..."]
        SR29["[29] frame_id=1029<br/>timestamp=12539666666<br/>entities=27<br/>events=3"]
    end

    subgraph ENTITY_RING["Entity History Ring (per entity, 30 slots)"]
        ER0["[0] state at t=12530"]
        ER1["[1] state at t=12533"]
        ER2["[2] state at t=12536"]
        ER3["..."]
        ER29["[29] state at t=12620"]
    end

    subgraph EVENT_RING["Event Ring Buffer (100 slots)"]
        EVT0["[0] event: goal, t=12531"]
        EVT1["[1] event: foul, t=12545"]
        EVT2["[2] event: camera_cut, t=12560"]
        EVT3["..."]
        EVT99["[99] event: anomaly, t=12800"]
    end

    subgraph VLM_QUEUE["VLM Queue (4 slots, COALESCE)"]
        VQ0["slot 0: hyp_21, R=0.92"]
        VQ1["slot 1: hyp_25, R=0.87"]
        VQ2["slot 2: — empty —"]
        VQ3["slot 3: — empty —"]
    end

    WRITE[Write Pointer] --> SR0
    READ[Read Pointer] --> SR29

    style STATE_RING fill:#c8e6c9,stroke:#388e3c
    style ENTITY_RING fill:#bbdefb,stroke:#1976d2
    style EVENT_RING fill:#fff9c4,stroke:#f9a825
    style VLM_QUEUE fill:#ffcdd2,stroke:#d32f2f
```

---

## 13. Complete Entity Relationship Diagram

```mermaid
erDiagram
    ENTITY ||--o{ ENTITY_STATE : "has states"
    ENTITY ||--o{ ENTITY_IDENTITY : "has identity"
    ENTITY ||--o{ ENTITY_EMBEDDING : "has embeddings"
    ENTITY ||--o{ ENTITY_PROFILE : "is profiled as"

    ENTITY_PROFILE ||--o{ ENTITY_SESSION : "has sessions"
    ENTITY_PROFILE ||--o{ ENTITY_TRAIT : "has traits"

    EVENT ||--o{ EVENT_EVIDENCE : "has evidence"
    EVENT ||--o{ EVENT_FRAMES : "has frames"
    EVENT ||--o{ EVENT_SUMMARY : "summarized as"

    EVENT_SUMMARY ||--o{ EVENT_SUMMARY_TAG : "has tags"

    EPISODE ||--o{ EPISODE_EVENT : "contains events"
    EPISODE_EVENT }o--|| EVENT : "references event"

    CLAIM ||--o{ CLAIM_EVIDENCE : "has evidence"
    CLAIM ||--o{ CLAIM_uncertainty_sources : "has uncertainty"

    HYPOTHESIS ||--o{ HYPOTHESIS_EVIDENCE : "has evidence"

    EVIDENCE_NODE ||--o{ EVIDENCE_EDGE : "as source"
    EVIDENCE_NODE ||--o{ EVIDENCE_EDGE : "as target"

    CALIBRATION_SET ||--o{ CALIBRATION_POINT : "has points"

    FORENSIC_RESULT ||--o{ FORENSIC_PROVENANCE : "has provenance"
    FORENSIC_RESULT ||--o{ FORENSIC_DETECTOR : "has detector scores"

    HYPOTHESIS_EVIDENCE }o--o| EVENT_EVIDENCE : "references evidence"
    CLAIM_EVIDENCE }o--o| EVENT_EVIDENCE : "references evidence"
```

---

## 14. Index Strategy

```mermaid
graph TB
    subgraph INDEXES["Critical Indexes"]
        direction TB
        I1["ENTITY.id<br/>PRIMARY KEY, B-tree"]
        I2["ENTITY.canonical_id<br/>UNIQUE INDEX"]
        I3["ENTITY_STATE.entity_id + timestamp_ns<br/>COMPOSITE INDEX"]
        I4["EVENT.timestamp_ns<br/>B-tree INDEX"]
        I5["EVENT.event_type<br/>INDEX for filtering"]
        I6["CLAIM.timestamp_ns<br/>B-tree INDEX"]
        I7["CLAIM.claim_type + domain<br/>COMPOSITE INDEX"]
        I8["CLAIM_EVIDENCE.claim_id<br/>FK INDEX"]
        I9["FORENSIC_RESULT.media_id<br/>INDEX"]
        I10["EPISODE.start_ns<br/>B-tree INDEX"]
    end

    subgraph PERFORMANCE["Performance Rules"]
        P1["All queries use indexed columns"]
        P2["No full table scans on hot path"]
        P3["Entity history: composite index on entity_id + timestamp"]
        P4["Event queries: index on timestamp + event_type"]
        P5["Claim queries: index on timestamp + claim_type + domain"]
        P6["WAL mode for concurrent reads"]
        P7["Connection pooling for async workers"]
    end
```

---

## 15. Data Lifecycle Diagram

```mermaid
graph TB
    subgraph HOT["Hot Path (< 1ms)"]
        H1[Frame Input] --> H2[Ring Buffer Write]
        H2 --> H3[Entity Map Update]
        H3 --> H4[Event Detection]
    end

    subgraph WARM["Warm Path (1-100ms)"]
        W1[Hypothesis Generation] --> W2[Working Memory]
        W2 --> W3[Fast Verification]
        W3 --> W4[Confidence Decomposition]
    end

    subgraph COLD["Cold Path (100ms-5s)"]
        C1[VLM Reasoning] --> C2[Conformal Prediction]
        C2 --> C3[Claim Generation]
        C3 --> C4[Claim Store]
    end

    subgraph ARCHIVE["Archive Path (async)"]
        A1[State Flush] --> A2[Long-Term Entity Store]
        A1 --> A3[Event Summary Store]
        A3 --> A4[Episode Store]
        A2 --> A5[FAISS Index]
    end

    subgraph LIFECYCLE["Retention Policy"]
        L1["Ring Buffers: 30 frames (1s)"]
        L2["Working Memory: 30s TTL"]
        L3["Claims: Permanent"]
        L4["Entity Profiles: Permanent"]
        L5["Event Summaries: 30 days"]
        L6["Episodes: 30 days"]
        L7["Forensic Results: 90 days"]
        L8["Calibration Sets: Until recalibrated"]
    end

    HOT --> WARM
    WARM --> COLD
    COLD --> ARCHIVE
```
