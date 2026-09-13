# System Architecture Diagrams

> All diagrams use Mermaid syntax. Render in GitHub, VS Code (with Mermaid extension), or any Mermaid-compatible viewer.

## Diagram Files

| File | Diagrams | What's Inside |
|------|----------|---------------|
| **`00_system_diagrams.md`** | 15 | System overview, critical path, perception, fusion, state, memory, reasoning, calibration, scheduler, forensics, data flow, hardware tiers, entity state machine, hypothesis decision tree |
| **`01_db_schema_diagrams.md`** | 15 | Entity store, event store, claim store, hypothesis store, evidence graph, calibration store, entity profiles, event summaries, episodes, forensic results, config store, ring buffers, complete ER diagram, index strategy, data lifecycle |
| **`02_api_protobuf_diagrams.md`** | 10 | Protobuf hierarchy, FrameObservationBatch, WorldStateSnapshot, Claim schemas, API endpoints, NvSchema flow, C2PA manifest, SynthID flow, FlatBuffers vs Protobuf, schema versioning |
| **`03_complete_reference.md`** | 3 | Master reference (all layers + all stores), latency budget summary, storage estimates |
| **`04_master_architecture.md`** | 7 | Complete system architecture (all layers all connections), end-to-end data flow, complete I/O flow, all data paths (6 paths A-F), component interaction map, full system latency budgets, system topology (tier 1-3) |
| **`05_complete_system.md`** | 10 | Complete pipeline (every data path), entity lifecycle state machine, hypothesis decision tree, evidence graph structure, data buffers & flows, API endpoints with contracts, deployment topology (K8s), latency budget frame-to-claim, component integration matrix, system summary (all components, stores, schemas, latency, hardware, deployment) |

---

## 1. System Overview — All Layers

```mermaid
graph TB
    subgraph INPUT["MULTIMODAL INPUT"]
        V[Video]
        A[Audio]
        T[Text/Metadata]
        P[Provenance/C2PA]
    end

    subgraph L1["LAYER 01 — PERCEPTION (30 FPS)"]
        DET[Detection<br/>RF-DETR-S 3.5ms]
        POSE[Pose Estimation<br/>DETRPose-S 2.4ms]
        TRK[Object Tracking<br/>ByteTrack 0.2ms]
        SEG[Segmentation<br/>RF-DETR-Seg 5ms]
        OCR[OCR<br/>PaddleOCR 20ms]
        REID[Re-Identification<br/>OSNet 3ms]
        AUD[Audio Features<br/>Whisper 50ms]
        CAM[Camera Motion<br/>ORB+RANSAC 2ms]
    end

    subgraph L2["LAYER 02 — FUSION"]
        MF[Multi-Modal Fusion<br/>Late/Attention 2-10ms]
        TF[Temporal Fusion<br/>EMA 0.5ms]
        XMA[Cross-Modal Alignment<br/>Correlation 2ms]
    end

    subgraph L3["LAYER 03 — STATE"]
        WS[World State<br/>Ring Buffer <0.1ms]
        ET[Entity Tracker<br/>Lifecycle 0.1ms]
        TM[Trajectory Model<br/>Kalman 0.1ms]
        ED[Event Detection<br/>Rules 0.2ms]
    end

    subgraph L4["LAYER 04 — MEMORY"]
        STM[Short-Term<br/>Ring Buffer <0.1ms]
        WM[Working Memory<br/>Active Hypotheses 1ms]
        LTM[Long-Term<br/>Vector DB 10-50ms]
        EM[Episodic<br/>Episode Store 5-30ms]
    end

    subgraph L5["LAYER 05 — REASONING"]
        HE[Hypothesis Engine<br/>Rules 5-15ms]
        FV[Fast Verifier<br/>Rules 1-5ms]
        VLM[Deep VLM Reasoner<br/>Qwen3-VL 800-3500ms]
        EG[Evidence Graph<br/>Typed Graph 1-5ms]
        PM[Prediction Model<br/>Trajectory 0.1ms]
    end

    subgraph L6["LAYER 06 — CALIBRATION"]
        CD[Confidence Decomposition<br/><1ms]
        CP[Conformal Prediction<br/>2-5ms]
        TS[Temperature Scaling<br/><0.1ms]
        CO[Claim Output<br/>Structured 1-2ms]
    end

    subgraph L7["LAYER 07 — SCHEDULER"]
        MRS[Multi-Rate Scheduler<br/>Priority <0.5ms]
        QM[Queue Management<br/>Bounded <0.5ms]
        BP[Backpressure<br/>Threshold <0.2ms]
        GPU[GPU Distribution<br/>MPS/MIG <0.2ms]
    end

    subgraph L8["LAYER 08 — FORENSICS"]
        DD[Deepfake Detection<br/>Ensemble 200-2000ms]
        C2PA[Provenance/C2PA<br/>50-200ms]
        AF[Audio Forensics<br/>100-500ms]
    end

    subgraph L9["LAYER 09 — DOMAINS"]
        SR[Sports Reasoning<br/>Rules 10-30ms]
        GM[General Multimedia<br/>Fallback]
        SM[Synthetic Media<br/>Detection + Provenance]
    end

    subgraph L10["LAYER 10 — INFRASTRUCTURE"]
        DS[Data Schemas<br/>Protobuf/JSON]
        HW[Hardware Topology<br/>GPU Layout]
        DEP[Deployment<br/>Docker/K8s]
    end

    subgraph OUTPUT["OUTPUT"]
        CLAIM[Structured Claims<br/>+ Confidence + Evidence]
    end

    %% Input to Perception
    V --> DET
    V --> CAM
    A --> AUD
    T --> OCR
    P --> C2PA

    %% Perception internal
    DET --> POSE
    DET --> TRK
    DET --> SEG
    DET --> REID
    TRK --> ET

    %% Perception to Fusion
    DET --> MF
    AUD --> MF
    OCR --> MF
    MF --> TF
    TF --> XMA

    %% Fusion to State
    XMA --> WS
    ET --> WS
    TM --> WS

    %% State to Memory
    WS --> STM
    WS --> WM

    %% State to Reasoning
    WS --> HE
    ED --> HE
    HE --> FV
    FV -->|inconclusive| VLM
    HE --> EG
    PM --> HE

    %% Reasoning to Calibration
    VLM --> CD
    EG --> CD
    CD --> CP
    CP --> TS
    TS --> CO

    %% Scheduler controls everything
    MRS --> QM
    QM --> BP
    BP --> GPU
    GPU -.->|controls| DET
    GPU -.->|controls| VLM
    GPU -.->|controls| DD

    %% Forensics
    C2PA --> SM
    DD --> SM
    AF --> SM

    %% Domains
    SR --> CO
    GM --> CO
    SM --> CO

    %% Memory
    STM --> LTM
    WM --> EM

    %% Output
    CO --> CLAIM

    %% Infrastructure
    DS -.->|schemas| WS
    HW -.->|GPU alloc| GPU
    DEP -.->|deployment| DEP
```

---

## 2. Critical Path — 30 FPS Data Flow

```mermaid
graph LR
    subgraph CRITICAL["CRITICAL PATH (< 20ms)"]
        direction LR
        D1["Decode<br/>1-3ms"] --> D2["Preprocess<br/>1-2ms"]
        D2 --> D3["Track<br/>0.2-0.5ms"]
        D3 --> D4["Detect+Pose<br/>5-8ms"]
        D4 --> D5["State Update<br/>0.3-1ms"]
        D5 --> D6["Event Score<br/>0.1-0.3ms"]
        D6 --> D7["Publish<br/>0.2-0.5ms"]
    end

    subgraph ASYNC["ASYNCHRONOUS PATHS"]
        direction TB
        A1["Hypothesis Engine<br/>5-15ms"] --> A2{"Fast Verifier<br/>1-5ms"}
        A2 -->|supported/refuted| A3["Claim Output"]
        A2 -->|inconclusive| A4["Deep VLM<br/>800-3500ms"]
        A4 --> A3
        A5["Forensics<br/>200-2000ms"] --> A3
        A6["Summarization<br/>500-5000ms"] --> A7["Long-Term Memory"]
    end

    D7 -->|R score ≥ 0.85| A1
    D7 -->|R score ≥ 0.5| A1
    D7 -->|R score < 0.5| A3

    style CRITICAL fill:#e1f5fe,stroke:#0288d1
    style ASYNC fill:#fff3e0,stroke:#f57c00
```

---

## 3. Layer 01 — Perception Detail

```mermaid
graph TB
    subgraph INPUT["Input"]
        F[Frame 33ms]
    end

    subgraph DET["Detection (2.3-6.8ms)"]
        D1[RF-DETR-S/M]
        D2[NMS/Postprocess]
    end

    subgraph POSE["Pose (2.39-9.7ms)"]
        P1[Crop from Detections]
        P2[DETRPose-S/L]
        P3[Keypoint Decode]
    end

    subgraph TRK["Tracking (0.2-0.5ms)"]
        T1[ByteTrack/Bot-SORT]
        T2[Kalman Predict+Update]
        T3[Track Management]
    end

    subgraph OPTIONAL["Optional (Async 5-10 Hz)"]
        O1[Segmentation<br/>RF-DETR-Seg]
        O2[OCR<br/>PaddleOCR]
        O3[Re-ID<br/>OSNet]
        O4[Camera Motion<br/>ORB+RANSAC]
    end

    subgraph OUTPUT["Output"]
        DB[DetectionBatch]
        PB[PoseBatch]
        TO[TrackOutput]
    end

    F --> D1
    D1 --> D2
    D2 --> DB
    DB --> P1
    P1 --> P2
    P2 --> P3
    P3 --> PB
    DB --> T1
    T1 --> T2
    T2 --> T3
    T3 --> TO

    F -.-> O1
    F -.-> O2
    F -.-> O4
    DB -.-> O3
    O1 -.-> OPTIONAL
    O2 -.-> OPTIONAL
    O3 -.-> OPTIONAL
    O4 -.-> OPTIONAL
```

---

## 4. Layer 02 — Fusion Detail

```mermaid
graph TB
    subgraph INPUT["Inputs from Perception"]
        VS[World State Snapshot]
        AF[Audio Features]
        TF[Text/OCR Features]
    end

    subgraph MF["Multi-Modal Fusion"]
        direction TB
        M1{"Fusion Strategy"}
        M1 -->|Early| M2[Concatenate + Linear<br/><1ms]
        M1 -->|Late| M3[Weighted Scores<br/>1-2ms]
        M1 -->|Attention| M4[Cross-Attention<br/>5-10ms]
        M1 -->|Gated| M5[Learned Gate<br/>2-5ms]
    end

    subgraph TF2["Temporal Fusion"]
        T1[Sliding Window / EMA]
        T2[Motion Features]
    end

    subgraph XMA["Cross-Modal Alignment"]
        X1[Timestamp Sync]
        X2[Feature Correlation]
        X3[Sync Score]
    end

    subgraph OUTPUT["Fused Output"]
        FR[FusedRepresentation<br/>embedding + agreement + conflicts]
    end

    VS --> MF
    AF --> MF
    TF --> MF
    MF --> TF2
    TF2 --> XMA
    XMA --> FR

    X3 -->|low sync| CONF[Confidence Penalty]
    CONF --> FR
```

---

## 5. Layer 03 — State Management

```mermaid
graph TB
    subgraph INPUT["Inputs"]
        PB[PoseBatch]
        TO[TrackOutput]
        DB[DetectionBatch]
        ED[EventTrigger]
    end

    subgraph WS["World State (Ring Buffer)"]
        W1[Entity Map<br/>ConcurrentHashMap]
        W2[State Ring<br/>Last 30 frames]
        W3[Relation Compute<br/>On-demand]
    end

    subgraph ET["Entity Lifecycle"]
        E1["DETECTED<br/>(first detection)"]
        E2["CONFIRMED<br/>(3+ frames)"]
        E3["ACTIVE<br/>(regular updates)"]
        E4["OCCLUDED<br/>(no detection)"]
        E5["LOST<br/>(max_age exceeded)"]
        E6["REMOVED"]

        E1 -->|3+ frames| E2
        E2 --> E3
        E3 -->|no detection| E4
        E4 -->|reappear| E3
        E4 -->|max_age| E5
        E5 --> E6
    end

    subgraph TM["Trajectory"]
        TR1[Constant Velocity]
        TR2[Kalman Filter]
        TR3[Polynomial Fit]
    end

    subgraph ED2["Event Detection"]
        R1[Rule Evaluation]
        R2[Priority Scoring R]
        R3[Event Trigger]
    end

    subgraph OUTPUT["Output"]
        WS_OUT[WorldStateSnapshot]
    end

    PB --> WS
    TO --> WS
    DB --> WS
    WS --> ET
    ET --> TM
    TM --> ED2
    ED2 --> WS_OUT
```

---

## 6. Layer 04 — Memory Hierarchy

```mermaid
graph TB
    subgraph STM["Short-Term Memory (Ring Buffer)"]
        S1[State Ring<br/>30 frames]
        S2[Entity History<br/>30/entity]
        S3[Event Log<br/>100 events]
    end

    subgraph WM["Working Memory"]
        W1[Active Hypotheses<br/>Top 10]
        W2[Evidence Buffer<br/>100 items]
        W3[Reasoning Trace<br/>50 steps]
    end

    subgraph LTM["Long-Term Memory (Async)"]
        L1[Entity Profiles<br/>SQLite + FAISS]
        L2[Event Summaries<br/>SQLite]
        L3[Video Segments<br/>S3/minio]
    end

    subgraph EM["Episodic Memory"]
        E1[Episode Store]
        E2[Episode Search<br/>FAISS]
        E3[Episode Summaries<br/>VLM-generated]
    end

    STM -->|"flush every 5s"| LTM
    STM -->|"promote active"| WM
    WM -->|"periodic"| EM
    WM -->|"context"| LTM

    WS[WorldState] --> STM
    HE[Hypotheses] --> WM
    EV[Events] --> EM

    style STM fill:#c8e6c9,stroke:#388e3c
    style WM fill:#fff9c4,stroke:#f9a825
    style LTM fill:#bbdefb,stroke:#1976d2
    style EM fill:#e1bee7,stroke:#7b1fa2
```

---

## 7. Layer 05 — Reasoning Pipeline

```mermaid
graph TB
    subgraph INPUT["Input"]
        EV[Event Trigger]
        WS[World State]
        WM[Working Memory]
    end

    subgraph HE["Hypothesis Engine (5-15ms)"]
        H1[Rule-Based Generation]
        H2[Pattern Matching]
        H3[Posterior Update]
        H4[Hypothesis Ranking]
    end

    subgraph FV["Fast Verifier (1-5ms)"]
        F1[Trajectory Check]
        F2[Pose Check]
        F3[Rule Check]
    end

    subgraph VLM["Deep VLM Reasoner (800-3500ms)"]
        V1[Frame Selection]
        V2[Prompt Construction]
        V3[VLM Inference]
        V4[Response Parse]
    end

    subgraph EG["Evidence Graph"]
        G1[Evidence Nodes]
        G2[Typed Edges]
        G3[Posterior Computation]
    end

    subgraph OUTPUT["Output"]
        CO[Claim Output]
    end

    EV --> HE
    WS --> HE
    WM --> HE
    HE --> FV

    FV -->|SUPPORTED| CO
    FV -->|REFUTED| CO
    FV -->|INCONCLUSIVE| VLM
    VLM --> CO

    HE --> EG
    FV --> EG
    VLM --> EG
    EG --> G3
    G3 --> CO

    style FV fill:#c8e6c9,stroke:#388e3c
    style VLM fill:#ffcdd2,stroke:#d32f2f
```

---

## 8. Layer 06 — Calibration Flow

```mermaid
graph TB
    subgraph INPUT["Inputs"]
        RC[Reasoning Components<br/>perception, temporal, motion,<br/>cross_modal, reasoning]
        HYPS[Hypotheses + Scores]
    end

    subgraph CD["Confidence Decomposition"]
        CD1[Weighted Sum]
        CD2[Quality Flags]
        CD3[Uncertainty Sources]
    end

    subgraph CP["Conformal Prediction"]
        CP1[Calibration Set]
        CP2[Nonconformity Scores]
        CP3[Threshold τ]
        CP4[Prediction Set]
    end

    subgraph TS["Temperature Scaling"]
        TS1[Learned T]
        TS2[Softmax Adjustment]
    end

    subgraph OUTPUT["Claim Output"]
        CO1[Structured Claim<br/>+ confidence components<br/>+ prediction sets<br/>+ staleness<br/>+ epistemic status]
    end

    RC --> CD
    CD --> CP
    CD --> TS
    CP --> CO1
    TS --> CO1
    HYPS --> CP
```

---

## 9. Layer 07 — Scheduler & Backpressure

```mermaid
graph TB
    subgraph INPUT["System Metrics"]
        M1[Queue Depths]
        M2[GPU Utilization]
        M3[Frame Latency]
        M4[Dropped Jobs]
    end

    subgraph MRS["Multi-Rate Scheduler"]
        R1[R Score Computation]
        R2[Priority Queue]
        R3[Job Dispatch]
    end

    subgraph QUEUES["Bounded Queues"]
        Q1["Perception<br/>depth=1"]
        Q2["Fast Reason<br/>depth=4"]
        Q3["Deep VLM<br/>depth=4"]
        Q4["Forensics<br/>depth=8"]
        Q5["Summary<br/>depth=4"]
    end

    subgraph BP["Backpressure"]
        B1{"VLM queue > 3<br/>for >30s?"}
        B2{"GPU > 95%<br/>for >60s?"}
        B3[Disable Low-Priority]
        B4[Downgrade Model<br/>RF-DETR-S → N]
    end

    subgraph GPU["GPU Distribution"]
        G1[CUDA Streams<br/>Priority-based]
        G2[MPS<br/>SM Sharing]
        G3[MIG<br/>HW Isolation]
    end

    M1 --> MRS
    M2 --> MRS
    M3 --> MRS
    M4 --> MRS

    MRS --> R1
    R1 --> R2
    R2 --> R3

    R3 --> Q1
    R3 --> Q2
    R3 --> Q3
    R3 --> Q4
    R3 --> Q5

    M1 --> BP
    M2 --> BP
    BP --> B1
    B1 -->|yes| B3
    B1 -->|no| B2
    B2 -->|yes| B4
    B2 -->|no| OK[Normal Operation]

    GPU --> Q1
    GPU --> Q3
```

---

## 10. Layer 08 — Forensics Pipeline

```mermaid
graph TB
    subgraph INPUT["Input Media"]
        IMG[Image]
        VID[Video]
        AUD[Audio]
    end

    subgraph C2PA["Provenance Check (50-200ms)"]
        C1[C2PA Manifest]
        C2[SynthID Watermark]
        C3[Content Credentials]
    end

    subgraph DD["Deepfake Detection (200-2000ms)"]
        D1[Semantic Branch<br/>CLIP/SigLIP 62.4%]
        D2[Structural Branch<br/>EVA-02 28.5%]
        D3[Spectral Branch<br/>SRM/Bayar 9.1%]
        D4[Logit Fusion<br/>+ Rank Normalization]
    end

    subgraph AF["Audio Forensics (100-500ms)"]
        A1[Teffic-Audio<br/>EER 1.45%]
        A2[FlowFake<br/>34K params]
        A3[XLSR+AASIST<br/>Open-source]
    end

    subgraph DECISION["Decision Fusion"]
        DF1{Provenance Valid?}
        DF2{Ensemble Consensus?}
        DF3[AUTHENTIC]
        DF4[SYNTHETIC]
        DF5[INCONCLUSIVE]
    end

    subgraph OUTPUT["Output"]
        OUT[Authenticity Assessment<br/>+ confidence<br/>+ evidence<br/>+ recommendation]
    end

    IMG --> C2PA
    VID --> C2PA
    AUD --> AF
    IMG --> DD
    VID --> DD

    C2PA --> DF1
    DF1 -->|valid + trusted| DF3
    DF1 -->|no manifest| DD
    DD --> DF2
    DF2 -->|consensus| DF4
    DF2 -->|no consensus| DF5
    AF --> DF2

    DF3 --> OUT
    DF4 --> OUT
    DF5 --> OUT
```

---

## 11. Full Data Flow — End to End

```mermaid
graph TB
    subgraph TIME["Time Budget: 33.3ms per frame"]
        direction LR
        T1["0-3ms<br/>Decode"] --> T2["3-5ms<br/>Preprocess"]
        T2 --> T3["5-6ms<br/>Track"]
        T3 --> T4["6-14ms<br/>Detect+Pose"]
        T4 --> T5["14-15ms<br/>State Update"]
        T5 --> T6["15-16ms<br/>Event Score"]
        T6 --> T7["16-17ms<br/>Publish"]
    end

    subgraph ASYNC_PATH["Async Processing"]
        direction TB
        E1[Event R≥0.85] --> H1[Hypothesis 5-15ms]
        H1 --> V1{Fast Verify 1-5ms}
        V1 -->|Yes| C1[Claim 1-2ms]
        V1 -->|No| V2[VLM 800-3500ms]
        V2 --> C1
        E2[Forensics Trigger] --> F1[Ensemble 200-2000ms]
        F1 --> C1
        C1 --> OUT1[Structured Claim]
    end

    subgraph MEMORY_PATH["Memory Path"]
        direction TB
        M1[State Snapshot] --> M2[Short-Term Ring]
        M2 --> M3[Long-Term DB]
        M3 --> M4[Episodic Store]
    end

    T7 -->|R score| E1
    T7 -->|scene cut| E2
    T7 --> M1
```

---

## 12. Hardware Topology — Tier 1 (1× RTX 4090)

```mermaid
graph TB
    subgraph GPU0["GPU 0: RTX 4090 24GB"]
        direction TB
        subgraph CRITICAL["Critical Path (30 FPS)"]
            DET[RF-DETR-S<br/>1.5 GB]
            POSE[DETRPose-S<br/>0.5 GB]
            TRK[ByteTrack<br/>0.05 GB]
        end
        subgraph ASYNC["Async Path (0.3-0.5 Hz)"]
            VLM[Qwen3-VL-30B-A3B FP8<br/>20 GB]
        end
        CTX[CUDA Context<br/>1 GB]
    end

    subgraph HOST["Host CPU"]
        RING[Ring Buffers<br/>Pre-allocated]
        QUEUE[Job Queues<br/>Bounded]
        API[REST API<br/>FastAPI]
    end

    DET --> POSE
    POSE --> TRK
    TRK --> RING
    RING --> VLM
    VLM --> API

    style GPU0 fill:#e3f2fd,stroke:#1976d2
    style HOST fill:#f3e5f5,stroke:#7b1fa2
```

---

## 13. Hardware Topology — Tier 3 (Multi-GPU)

```mermaid
graph TB
    subgraph G0["GPU 0: Perception"]
        DET[RF-DETR-L]
        POSE[DETRPose-L]
        TRK[ByteTrack]
        CAM[Camera Motion]
    end

    subgraph G1["GPU 1: Fast Reasoning"]
        HE[Hypothesis Engine]
        FV[Fast Verifier]
        EG[Evidence Graph]
    end

    subgraph G2["GPU 2: Deep Reasoning"]
        VLM[Qwen 3.5-397B TP=4]
    end

    subgraph G3["GPU 3: Forensics + Memory"]
        DD[Deepfake Ensemble]
        C2PA[C2PA/SynthID]
        LTM[Long-Term Memory]
        FAISS[FAISS Index]
    end

    subgraph CPU["Host CPU"]
        SCHED[Scheduler]
        RING[Ring Buffers]
        API[API Server]
    end

    G0 -->|state snapshots| RING
    RING --> G1
    G1 -->|hypotheses| G2
    G2 -->|VLM output| G1
    G1 -->|claims| G3
    G0 -->|events| SCHED
    SCHED -->|jobs| G1
    SCHED -->|jobs| G2
    SCHED -->|jobs| G3
```

---

## 14. State Machine — Entity Lifecycle

```mermaid
stateDiagram-v2
    [*] --> DETECTED: First detection\nscore > threshold
    DETECTED --> CONFIRMED: 3+ consecutive\nframes matched
    CONFIRMED --> ACTIVE: Regular updates\nreceived
    ACTIVE --> OCCLUDED: No detection\nbut tracker interpolating
    OCCLUDED --> ACTIVE: Detection reappears\nidentity verified
    OCCLUDED --> LOST: No detection\nfor max_age frames
    LOST --> REMOVED: Cleanup
    ACTIVE --> LOST: No detection\nfor max_age frames
    REMOVED --> [*]

    note right of DETECTED: TENTATIVE state
    note right of CONFIRMED: Track is stable
    note right of OCCLUDED: Kinematic model\npredicts position
    note right of LOST: Identity preserved\nfor re-identification
```

---

## 15. Hypothesis Decision Tree

```mermaid
graph TB
    START[Event Triggered] --> GEN[Generate Hypotheses<br/>Top 3-5]
    GEN --> FAST{Fast Verify<br/>1-5ms}
    FAST -->|Trajectory match| SUP[SUPPORTED<br/>confidence = 0.8-0.95]
    FAST -->|Rule violation| REF[REFUTED<br/>confidence = 0.9-1.0]
    FAST -->|Inconclusive| VLM{Deep VLM<br/>800-3500ms}
    VLM -->|Evidence found| VLM_SUP[SUPPORTED<br/>confidence = 0.6-0.9]
    VLM -->|Contradiction found| VLM_REF[REFUTED<br/>confidence = 0.7-0.95]
    VLM -->|Still unclear| INC[INCONCLUSIVE<br/>confidence = 0.3-0.5]

    SUP --> CAL[Calibration]
    REF --> CAL
    VLM_SUP --> CAL
    VLM_REF --> CAL
    INC --> CAL

    CAL --> CONFORMAL{Conformal<br/>Prediction Set}
    CONFORMAL -->|Set size = 1| CLAIM[Claim Published]
    CONFORMAL -->|Set size = 2-3| CLAIM2[Claim + Alternatives]
    CONFORMAL -->|Set size > 3| UNCERTAIN[High Uncertainty Flag]
```
