# Complete System Reference Diagram

> Master diagram showing every component, every data store, every connection, and every latency budget.

---

## Full System — All Layers + All Data Stores

```mermaid
graph TB
    subgraph INPUT["INPUT"]
        V[Video Stream<br/>RTSP/HTTP]
        A[Audio Stream]
        M[Metadata/Provenance]
    end

    subgraph GPU_PERCEPTION["GPU 0: PERCEPTION (30 FPS, <20ms)"]
        direction TB
        DEC[Decode<br/>1-3ms]
        PRE[Preprocess<br/>1-2ms]
        DET["Detection<br/>RF-DETR-S 3.5ms<br/>DB: DetectionBatch"]
        POSE["Pose<br/>DETRPose-S 2.4ms<br/>DB: PoseBatch"]
        TRK["Tracking<br/>ByteTrack 0.2ms<br/>DB: TrackOutput"]
        CAM["Camera Motion<br/>ORB+RANSAC 2ms<br/>DB: CameraMotionOutput"]
    end

    subgraph GPU_REASONING["GPU 1: REASONING (0.3-0.5 Hz)"]
        direction TB
        HE["Hypothesis Engine<br/>5-15ms<br/>DB: Hypothesis"]
        FV["Fast Verifier<br/>1-5ms<br/>DB: VerificationResult"]
        EG["Evidence Graph<br/>1-5ms<br/>DB: EvidenceGraph"]
        VLM["Deep VLM<br/>Qwen3-VL-30B<br/>800-3500ms<br/>DB: VLMReasoningOutput"]
    end

    subgraph GPU_CALIBRATION["GPU/CPU: CALIBRATION"]
        direction TB
        CD["Confidence Decomposition<br/><1ms<br/>DB: ConfidenceDecomposition"]
        CP["Conformal Prediction<br/>2-5ms<br/>DB: ConformalOutput"]
        TS["Temperature Scaling<br/><0.1ms<br/>DB: TemperatureOutput"]
        CLM["Claim Output<br/>1-2ms<br/>DB: Claim"]
    end

    subgraph GPU_FORENSICS["GPU 2: FORENSICS (async)"]
        direction TB
        C2PA["C2PA/SynthID<br/>50-200ms<br/>DB: ProvenanceOutput"]
        DDF["Deepfake Ensemble<br/>200-2000ms<br/>DB: ForensicResult"]
        AFF["Audio Forensics<br/>100-500ms<br/>DB: AudioForensicsResult"]
    end

    subgraph STATE["STATE MANAGEMENT"]
        WS["World State<br/>Ring Buffer 30 slots<br/>Latency: <0.1ms"]
        ET["Entity Tracker<br/>Lifecycle Manager<br/>Latency: 0.1ms"]
        TM["Trajectory Model<br/>Kalman Filter<br/>Latency: 0.1ms"]
        ED["Event Detection<br/>Rule Engine<br/>Latency: 0.2ms"]
    end

    subgraph MEMORY["MEMORY HIERARCHY"]
        STM["Short-Term<br/>Ring Buffer<br/>30 frames<br/>Latency: <0.1ms"]
        WM["Working Memory<br/>Active Hypotheses<br/>Top 10<br/>Latency: 1ms"]
        LTM["Long-Term<br/>SQLite + FAISS<br/>Entity Profiles<br/>Latency: 10-50ms"]
        EMP["Episodic<br/>Episode Store<br/>FAISS Retrieval<br/>Latency: 5-30ms"]
    end

    subgraph SCHEDULER["SCHEDULER"]
        MRS["Multi-Rate Scheduler<br/>R Score: 0-1<br/>Latency: <0.5ms"]
        QM["Queue Manager<br/>Bounded Queues<br/>Depth: 1-8"]
        BP["Backpressure<br/>Threshold Triggers<br/>Latency: <0.2ms"]
        GPUD["GPU Distribution<br/>MPS/MIG<br/>Latency: <0.2ms"]
    end

    subgraph DOMAINS["DOMAINS"]
        SR["Sports Reasoning<br/>Rules + VLM<br/>10-30ms"]
        GMM["General Multimedia<br/>Fallback Pipeline"]
        SM["Synthetic Media<br/>Detection + Provenance"]
    end

    subgraph OUTPUT["OUTPUT"]
        CLAIM["Structured Claim<br/>+ Confidence Components<br/>+ Prediction Sets<br/>+ Evidence References<br/>+ Staleness + Epistemic Status"]
    end

    subgraph STORES["DATA STORES"]
        direction TB
        DS1[("Entity DB<br/>SQLite<br/>Entities + States")]
        DS2[("Event DB<br/>SQLite<br/>Events + Evidence")]
        DS3[("Claim DB<br/>SQLite<br/>Claims")]
        DS4[("Calibration DB<br/>SQLite<br/>Sets + Params")]
        DS5[("Forensic DB<br/>SQLite<br/>Results")]
        DS6[("Vector DB<br/>FAISS<br/>Entity Embeddings")]
        DS7[("Episode DB<br/>SQLite<br/>Episodes")]
        DS8[("Config DB<br/>JSON/SQLite<br/>System Config")]
    end

    subgraph INFRA["INFRASTRUCTURE"]
        SCHEMA["Data Schemas<br/>Protobuf + FlatBuffers"]
        HW["Hardware Topology<br/>GPU Allocation"]
        DEP["Deployment<br/>Docker/K8s/Helm"]
    end

    %% Input flow
    V --> DEC
    A --> AUD_IN[Audio Pipeline]
    M --> C2PA

    %% Perception flow
    DEC --> PRE --> DET --> POSE --> TRK
    DEC --> CAM

    %% State management
    DET --> WS
    POSE --> WS
    TRK --> WS
    CAM --> WS
    WS --> ET
    ET --> TM
    TM --> ED

    %% Memory
    WS --> STM
    STM --> LTM
    STM --> WM
    WM --> EMP

    %% Scheduler
    ED --> MRS
    MRS --> QM
    QM --> BP
    BP --> GPUD
    GPUD -.->|controls| DET
    GPUD -.->|controls| VLM
    GPUD -.->|controls| DDF

    %% Reasoning
    ED -->|R ≥ 0.85| HE
    HE --> FV
    FV -->|inconclusive| VLM
    HE --> EG
    FV --> EG
    VLM --> EG

    %% Calibration
    FV -->|verified| CD
    VLM -->|reasoned| CD
    EG --> CD
    CD --> CP
    CP --> TS
    TS --> CLM

    %% Forensics
    C2PA --> SM
    DDF --> SM
    AFF --> SM

    %% Domains
    SR --> CLM
    GMM --> CLM
    SM --> CLM

    %% Output
    CLM --> CLAIM

    %% Data stores
    WS -.-> DS1
    ED -.-> DS2
    CLM -.-> DS3
    CP -.-> DS4
    DDF -.-> DS5
    LTM -.-> DS6
    EMP -.-> DS7
    MRS -.-> DS8

    %% Infrastructure
    SCHEMA -.->|schemas| WS
    HW -.->|GPU alloc| GPUD
    DEP -.->|deployment| DEP
```

---

## Latency Budget Summary

```mermaid
graph LR
    subgraph BUDGET["33.3ms Frame Budget"]
        direction LR
        B1["Decode<br/>1-3ms"]
        B2["Preprocess<br/>1-2ms"]
        B3["Track<br/>0.2-0.5ms"]
        B4["Detect+Pose<br/>5-8ms"]
        B5["State Update<br/>0.3-1ms"]
        B6["Event Score<br/>0.1-0.3ms"]
        B7["Publish<br/>0.2-0.5ms"]
        B8["HEADROOM<br/>5-20ms"]
    end

    B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8

    style B8 fill:#c8e6c9,stroke:#388e3c
```

---

## Data Store Size Estimates

```mermaid
graph TB
    subgraph SIZES["Storage Estimates (1 hour of 30 FPS video)"]
        direction TB
        S1["State Ring: 30 frames × 26 entities × 1KB = ~780 KB"]
        S2["Entity History: 26 entities × 30 frames × 0.5KB = ~390 KB"]
        S3["Event Log: 100 events × 2KB = ~200 KB"]
        S4["Working Memory: 10 hypotheses × 5KB = ~50 KB"]
        S5["Entity Profiles (long-term): 26 × 5KB = ~130 KB"]
        S6["Event Summaries: ~100 × 3KB = ~300 KB"]
        S7["Claims: ~50 × 2KB = ~100 KB"]
        S8["FAISS Index: 26 × 512 × 4 bytes = ~53 KB"]
        S9["Total per hour: ~2 MB (in-memory) + ~5 MB (persistent)"]
    end
```
