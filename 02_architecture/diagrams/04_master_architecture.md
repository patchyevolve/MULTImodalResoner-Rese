# Master Architecture — Complete System View

> Every component, every connection, every data path, every latency budget. One diagram to see the whole system.

---

## 1. Complete System Architecture — All Layers, All Connections

```mermaid
graph TB
    subgraph INPUT["MULTIMODAL INPUT"]
        V["Video Stream<br/>RTSP / HTTP / File<br/>30 FPS, H.264/H.265"]
        A["Audio Stream<br/>PCM 16kHz mono<br/>1s chunks"]
        T["Text/Metadata<br/>OCR triggers, camera info<br/>provenance data"]
        P["Provenance<br/>C2PA manifests<br/>SynthID watermarks"]
    end

    subgraph PERCEPTION["LAYER 01 — PERCEPTION (30 FPS, <20ms)"]
        direction LR
        subgraph CRITICAL["Critical Path"]
            D1["Decode<br/>1-3ms<br/>H/W decoder"] --> D2["Preprocess<br/>1-2ms<br/>Resize, normalize"]
            D2 --> D3["Tracking<br/>0.2-0.5ms<br/>ByteTrack"]
            D3 --> D4["Detection<br/>2.3-6.8ms<br/>RF-DETR-S/M"]
            D4 --> D5["Pose<br/>2.39-9.7ms<br/>DETRPose-S"]
            D5 --> D6["State Update<br/>0.3-1ms<br/>Kalman filter"]
            D6 --> D7["Event Score<br/>0.1-0.3ms<br/>Priority R"]
            D7 --> D8["Publish<br/>0.2-0.5ms<br/>Ring buffer"]
        end
        subgraph ASYNC_PERC["Async Perception (5-10 Hz)"]
            O1["Segmentation<br/>RF-DETR-Seg 5-10ms"]
            O2["OCR<br/>PaddleOCR 15-40ms"]
            O3["Re-ID<br/>OSNet 3-10ms"]
            O4["Camera Motion<br/>ORB+RANSAC 1-3ms"]
            O5["Audio Features<br/>Whisper 50-100ms"]
        end
    end

    subgraph FUSION["LAYER 02 — FUSION"]
        direction LR
        F1["Multi-Modal<br/>Late/Attention<br/>2-10ms"]
        F2["Temporal<br/>EMA Sliding Window<br/>0.5-3ms"]
        F3["Cross-Modal<br/>Correlation Sync<br/>1-3ms"]
    end

    subgraph STATE["LAYER 03 — STATE"]
        direction LR
        S1["World State<br/>Lock-free Ring Buffer<br/>30 frames, <0.1ms"]
        S2["Entity Tracker<br/>Lifecycle Manager<br/>0.1ms per entity"]
        S3["Trajectory<br/>Kalman Filter<br/>0.1ms"]
        S4["Event Detection<br/>Rule Engine + R Score<br/>0.2ms"]
    end

    subgraph MEMORY["LAYER 04 — MEMORY"]
        direction LR
        M1["Short-Term<br/>Ring Buffer 30 frames<br/><0.1ms"]
        M2["Working<br/>Active Hypotheses Top-10<br/>1ms"]
        M3["Long-Term<br/>SQLite + FAISS<br/>10-50ms"]
        M4["Episodic<br/>Episode Store + Search<br/>5-30ms"]
    end

    subgraph REASONING["LAYER 05 — REASONING"]
        direction LR
        R1["Hypothesis Engine<br/>Rules + Pattern<br/>5-15ms"]
        R2["Fast Verify<br/>Trajectory/Pose/Rule<br/>1-5ms"]
        R3["Deep VLM<br/>Qwen3-VL-30B<br/>800-3500ms"]
        R4["Evidence Graph<br/>Typed Nodes+Edges<br/>1-5ms"]
        R5["Prediction Model<br/>Trajectory Extrapolation<br/>0.1ms"]
    end

    subgraph CALIBRATION["LAYER 06 — CALIBRATION"]
        direction LR
        C1["Confidence Decomposition<br/>5 components<br/><1ms"]
        C2["Conformal Prediction<br/>Prediction Sets alpha=0.05<br/>2-5ms"]
        C3["Temperature Scaling<br/>Post-hoc scaling<br/><0.1ms"]
        C4["Claim Output<br/>Structured JSON/Proto<br/>1-2ms"]
    end

    subgraph SCHEDULER["LAYER 07 — SCHEDULER"]
        direction LR
        Q1["Multi-Rate<br/>R Score Priority<br/><0.5ms"]
        Q2["Queue Mgmt<br/>Bounded + Coalesce<br/><0.5ms"]
        Q3["Backpressure<br/>Threshold Triggers<br/><0.2ms"]
        Q4["GPU Distrib<br/>MPS/MIG/Streams<br/><0.2ms"]
    end

    subgraph FORENSICS["LAYER 08 — FORENSICS (async)"]
        direction LR
        K1["Deepfake Detection<br/>Ensemble 4-6 models<br/>200-2000ms"]
        K2["C2PA/SynthID<br/>Provenance Check<br/>50-200ms"]
        K3["Audio Forensics<br/>Teffic-Audio + FlowFake<br/>100-500ms"]
    end

    subgraph DOMAINS["LAYER 09 — DOMAINS"]
        direction LR
        N1["Sports Reasoning<br/>Rules + SoccerNet<br/>10-30ms"]
        N2["General Multimedia<br/>Domain-Agnostic Fallback"]
        N3["Synthetic Media<br/>Detection + Provenance"]
    end

    subgraph INFRA["LAYER 10 — INFRASTRUCTURE"]
        direction LR
        I1["Schemas<br/>Protobuf + FlatBuffers<br/>Versioned contracts"]
        I2["Hardware Topology<br/>GPU/CPU/Edge layout<br/>Tier 1-3"]
        I3["Deployment<br/>Docker/K8s/Helm<br/>Production-ready"]
    end

    subgraph STORES["DATA STORES"]
        direction LR
        DB1[("Entity DB<br/>SQLite")]
        DB2[("Event DB<br/>SQLite")]
        DB3[("Claim DB<br/>SQLite")]
        DB4[("Calibration DB<br/>SQLite")]
        DB5[("Forensic DB<br/>SQLite")]
        DB6[("Vector DB<br/>FAISS")]
        DB7[("Episode DB<br/>SQLite")]
        DB8[("Config DB<br/>JSON")]
    end

    subgraph OUTPUT["OUTPUT"]
        O_CLAIM["Structured Claim<br/>Confidence Components<br/>Prediction Sets<br/>Evidence References<br/>Staleness Tags<br/>Epistemic Status"]
        O_API["REST API<br/>FastAPI<br/>JSON/Protobuf"]
        O_WS["WebSocket<br/>Live Stream<br/>SSE Events"]
        O_KFK["Kafka<br/>NvSchema<br/>Async Messages"]
    end

    %% INPUT CONNECTIONS
    V --> D1
    A --> O5
    T --> O2
    P --> K2

    %% PERCEPTION INTERNAL
    D4 -->|crops| D5
    D4 -->|detections| O1
    D4 -->|detections| O2
    D4 -->|embeddings| O3
    D4 -->|features| O5

    %% PERCEPTION to FUSION
    D8 -->|state snapshot| F1
    O5 -->|audio features| F1
    O2 -->|text features| F1

    %% FUSION to STATE
    F1 --> F2
    F2 --> F3
    F3 -->|fused features| S1

    %% STATE INTERNAL
    D6 -->|entity updates| S1
    D5 -->|pose data| S1
    D4 -->|detections| S2
    S1 --> S2
    S2 --> S3
    S3 --> S4

    %% STATE to MEMORY
    S1 -->|snapshot| M1
    M1 -->|flush 5s| M3
    M1 -->|promote| M2

    %% STATE to REASONING
    S4 -->|event trigger| Q1
    S1 -->|world state| R1
    M2 -->|hypothesis context| R1

    %% REASONING INTERNAL
    R1 --> R2
    R2 -->|supported/refuted| C4
    R2 -->|inconclusive| R3
    R1 --> R4
    R2 --> R4
    R3 --> R4
    S3 --> R5
    R5 -->|predictions| R1

    %% REASONING to CALIBRATION
    R2 -->|verified result| C1
    R3 -->|VLM result| C1
    R4 -->|evidence| C1
    C1 --> C2
    C2 --> C3
    C3 --> C4

    %% SCHEDULER CONTROLS
    Q1 --> Q2
    Q2 --> Q3
    Q3 --> Q4
    Q4 -.->|"control perceive"| D4
    Q4 -.->|"control reason"| R3
    Q4 -.->|"control forensics"| K1
    Q4 -.->|"control schedule"| Q1

    %% FORENSICS
    K2 -->|authenticity| N3
    K1 -->|detection| N3
    K3 -->|audio| N3
    N3 --> C4

    %% DOMAINS
    S4 -->|event| N1
    N1 --> C4
    N2 --> C4

    %% CALIBRATION to OUTPUT
    C4 --> O_CLAIM
    O_CLAIM --> O_API
    O_CLAIM --> O_WS
    O_CLAIM --> O_KFK

    %% DATA STORES
    S1 -.->|write| DB1
    S4 -.->|write| DB2
    C4 -.->|write| DB3
    C2 -.->|write| DB4
    K1 -.->|write| DB5
    M3 -.->|write| DB6
    M4 -.->|write| DB7
    Q1 -.->|read/write| DB8

    %% MEMORY to REASONING
    M3 -.->|historical context| R3
    M4 -.->|episode context| R3
    M2 -.->|active context| R1

    %% EVIDENCE GRAPH CONNECTIONS
    R4 -.->|evidence for| C2
    R4 -.->|graph snapshot| DB3
```

---

## 2. End-to-End Data Flow — Input to Output

```mermaid
graph LR
    subgraph TIMELINE["TIMELINE"]
        T0["T=0ms<br/>Frame arrives"]
        T3["T=3ms<br/>Decoded"]
        T5["T=5ms<br/>Preprocessed"]
        T6["T=6ms<br/>Tracked"]
        T14["T=14ms<br/>Detected + Posed"]
        T15["T=15ms<br/>State Updated"]
        T16["T=16ms<br/>Event Scored"]
        T17["T=17ms<br/>Published"]
        T20["T=20ms<br/>Fused"]
        T21["T=21ms<br/>State Written"]
        T22["T=22ms<br/>Event Triggered"]
        T37["T=37ms<br/>Hypothesis Generated"]
        T42["T=42ms<br/>Fast Verified"]
        T44["T=44ms<br/>Confidence Decomposed"]
        T46["T=46ms<br/>Claim Published"]
        T3500["T=800-3500ms<br/>VLM Reasoned (async)"]
    end

    T0 --> T3 --> T5 --> T6 --> T14 --> T15 --> T16 --> T17
    T17 --> T20 --> T21 --> T22
    T22 --> T37 --> T42 --> T44 --> T46
    T42 -->|inconclusive| T3500 --> T44

    style T0 fill:#e3f2fd
    style T17 fill:#c8e6c9
    style T46 fill:#fff9c4
    style T3500 fill:#ffcdd2
```

---

## 3. Complete Input/Output Flow

```mermaid
graph TB
    subgraph IN["INPUT LAYER"]
        IN1["Video Input<br/>RTSP stream: cam_001<br/>Resolution: 1920x1080<br/>FPS: 30<br/>Codec: H.264"]
        IN2["Audio Input<br/>PCM 16kHz mono<br/>Chunk: 1 second<br/>Source: same stream"]
        IN3["Metadata<br/>Camera ID, timestamp<br/>GPS (if available)<br/>Stream quality flags"]
        IN4["Provenance<br/>C2PA manifest bytes<br/>SynthID watermark<br/>Content Credentials"]
    end

    subgraph PROC["PROCESSING LAYER"]
        direction TB
        subgraph P1["Phase 1: Perception (<20ms)"]
            P1A["Decode > Preprocess > Detection > Pose > Tracking > State Update"]
            P1A2["Output: Entity[], Trajectory[], EventTrigger[]"]
        end
        subgraph P2["Phase 2: Reasoning (5-3500ms)"]
            P2A["Hypothesis > Fast Verify > (VLM Deep Reasoning) > Evidence Graph"]
            P2A2["Output: Hypothesis (SUPPORTED/REFUTED/INCONCLUSIVE)"]
        end
        subgraph P3["Phase 3: Calibration (<10ms)"]
            P3A["Confidence Decomposition > Conformal > Temperature Scaling"]
            P3A2["Output: ConfidenceDecomposition + PredictionSet"]
        end
        subgraph P4["Phase 4: Output (1-2ms)"]
            P4A["Claim Assembly > Format Serialization > Quality Check"]
            P4A2["Output: Structured Claim (JSON/Protobuf)"]
        end
    end

    subgraph OUT["OUTPUT LAYER"]
        OUT1["REST API Response<br/>GET /api/v1/claims<br/>JSON format<br/>HTTP 200"]
        OUT2["WebSocket Stream<br/>WS /api/v1/stream/live<br/>SSE events<br/>Real-time"]
        OUT3["Kafka Message<br/>Topic: claims.published<br/>Protobuf binary<br/>Async"]
        OUT4["Dashboard UI<br/>Claims list<br/>Confidence charts<br/>Evidence graph"]
    end

    IN1 --> P1
    IN2 --> P1
    IN3 --> P1
    IN4 --> P4
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> OUT1
    P4 --> OUT2
    P4 --> OUT3
    P4 --> OUT4
```

---

## 4. Complete Data Flow — All Paths

```mermaid
graph TB
    subgraph PATH_A["PATH A: CRITICAL PATH (30 FPS)"]
        direction LR
        A1["Frame In"] --> A2["Decode 1-3ms"]
        A2 --> A3["Preprocess 1-2ms"]
        A3 --> A4["Track 0.2ms"]
        A4 --> A5["Detect 3.5ms"]
        A5 --> A6["Pose 2.4ms"]
        A6 --> A7["State Update 0.5ms"]
        A7 --> A8["Event Score 0.2ms"]
        A8 --> A9["Publish 0.3ms"]
        A9 --> A10["Snapshot Ready"]
    end

    subgraph PATH_B["PATH B: FAST REASONING (5-10 Hz)"]
        direction LR
        B1["Event R>=0.5"] --> B2["Hypothesis 5-15ms"]
        B2 --> B3{"Fast Verify 1-5ms"}
        B3 -->|SUPPORTED| B4["Confidence 0.8-0.95"]
        B3 -->|REFUTED| B5["Confidence 0.9-1.0"]
        B3 -->|INCONCLUSIVE| B6["to Path C"]
        B4 --> B7["Claim Output 1-2ms"]
        B5 --> B7
    end

    subgraph PATH_C["PATH C: DEEP REASONING (0.3-0.5 Hz)"]
        direction LR
        C1["Inconclusive"] --> C2["Select Frames 5-10ms"]
        C2 --> C3["Build Prompt 1-2ms"]
        C3 --> C4["VLM Inference 800-3500ms"]
        C4 --> C5["Parse Response 5-10ms"]
        C5 --> C6{"VLM Verdict"}
        C6 -->|SUPPORTED| C7["Confidence 0.6-0.9"]
        C6 -->|REFUTED| C8["Confidence 0.7-0.95"]
        C6 -->|INCONCLUSIVE| C9["Confidence 0.3-0.5"]
        C7 --> C10["to Path D"]
        C8 --> C10
        C9 --> C10
    end

    subgraph PATH_D["PATH D: CALIBRATION OUTPUT"]
        direction LR
        D1["Confidence Input"] --> D2["Decompose <1ms"]
        D2 --> D3["Conformal 2-5ms"]
        D3 --> D4["Temperature <0.1ms"]
        D4 --> D5["Assemble Claim 1ms"]
        D5 --> D6["Validate 0.1ms"]
        D6 --> D7["Serialize 0.5ms"]
        D7 --> D8["Claim Published"]
    end

    subgraph PATH_E["PATH E: FORENSICS PATH (async)"]
        direction LR
        E1["Scene Cut / New Entity"] --> E2{"C2PA Valid?"}
        E2 -->|Valid + Trusted| E3["AUTHENTIC 0.95"]
        E2 -->|No Manifest| E4["Run Ensemble 200-2000ms"]
        E4 --> E5["Semantic CLIP"]
        E4 --> E6["Structural EVA-02"]
        E4 --> E7["Spectral SRM"]
        E5 --> E8["Logit Fusion"]
        E6 --> E8
        E7 --> E8
        E8 --> E9{"Consensus?"}
        E9 -->|Yes| E10["SYNTHETIC 0.8-0.95"]
        E9 -->|No| E11["INCONCLUSIVE 0.5"]
        E3 --> E12["to Claim Output"]
        E10 --> E12
        E11 --> E12
    end

    subgraph PATH_F["PATH F: MEMORY PATH (async)"]
        direction LR
        F1["State Snapshot"] --> F2["Short-Term Ring"]
        F2 -->|flush 5s| F3["Long-Term SQLite"]
        F2 -->|promote| F4["Working Memory"]
        F3 --> F5["FAISS Index"]
        F4 -->|episode| F6["Episodic Store"]
        F3 -.->|historical| F7["to VLM Context"]
        F6 -.->|episode| F7
    end

    %% Cross-path connections
    A10 -->|R score| B1
    A10 -->|scene cut| E1
    A10 -->|state| F1
    B6 --> C1
    B7 --> D1
    C10 --> D1
    E12 --> D5
    F7 --> C4
```

---

## 5. Complete Component Interaction Map

```mermaid
graph TB
    subgraph PERC["Perception Components"]
        DET["Detection<br/>RF-DETR-S/M"]
        POSE["Pose<br/>DETRPose-S"]
        TRK["Tracking<br/>ByteTrack"]
        SEG["Segmentation<br/>RF-DETR-Seg"]
        OCR["OCR<br/>PaddleOCR"]
        REID["Re-ID<br/>OSNet"]
        AUD["Audio<br/>Whisper"]
        CAM["Camera Motion<br/>ORB+RANSAC"]
    end

    subgraph ST["State Components"]
        WS["World State<br/>Ring Buffer"]
        ET["Entity Tracker<br/>Lifecycle"]
        TM["Trajectory<br/>Kalman"]
        ED["Event Detection<br/>Rules"]
    end

    subgraph REAS["Reasoning Components"]
        HE["Hypothesis<br/>Engine"]
        FV["Fast Verifier<br/>Rules"]
        VLM["Deep VLM<br/>Qwen3-VL"]
        EG["Evidence<br/>Graph"]
        PM["Prediction<br/>Model"]
    end

    subgraph CAL["Calibration Components"]
        CD["Confidence<br/>Decomposition"]
        CP["Conformal<br/>Prediction"]
        TS["Temperature<br/>Scaling"]
        CO["Claim<br/>Output"]
    end

    subgraph SCHED["Scheduler Components"]
        MRS["Multi-Rate<br/>Scheduler"]
        QM["Queue<br/>Manager"]
        BP["Backpressure<br/>Controller"]
        GPU["GPU<br/>Distributor"]
    end

    %% PERCEPTION to STATE
    DET -->|"detections"| WS
    POSE -->|"pose data"| WS
    TRK -->|"tracks"| WS
    CAM -->|"motion"| WS
    DET -->|"crops"| POSE
    DET -->|"embeddings"| REID
    DET -->|"regions"| SEG
    DET -->|"regions"| OCR
    AUD -->|"speech"| WS

    %% STATE INTERNAL
    WS --> ET
    ET --> TM
    TM --> ED

    %% STATE to REASONING
    WS -->|"state"| HE
    ED -->|"events"| HE
    TM -->|"trajectories"| PM
    PM -->|"predictions"| HE
    HE --> FV
    FV -->|"inconclusive"| VLM
    HE --> EG
    FV --> EG
    VLM --> EG

    %% REASONING to CALIBRATION
    FV -->|"verified"| CD
    VLM -->|"reasoned"| CD
    EG -->|"evidence"| CD
    CD --> CP
    CP --> TS
    TS --> CO

    %% SCHEDULER CONTROLS
    MRS --> QM
    QM --> BP
    BP --> GPU
    GPU -.->|"priority stream"| DET
    GPU -.->|"async stream"| VLM
    GPU -.->|"background"| SEG
    ED -->|"R score"| MRS

    %% OUTPUT
    CO -->|"claim"| OUT["Output"]
```

---

## 6. Full System with Latency Budgets

```mermaid
graph TB
    subgraph L1["LAYER 01 PERCEPTION"]
        L1A["Decode: 1-3ms"]
        L1B["Preprocess: 1-2ms"]
        L1C["Track: 0.2ms"]
        L1D["Detect: 3.5ms"]
        L1E["Pose: 2.4ms"]
        L1F["State: 0.5ms"]
        L1G["Event: 0.2ms"]
        L1H["Publish: 0.3ms"]
        L1T["TOTAL: 8-12ms"]
    end

    subgraph L2["LAYER 02 FUSION"]
        L2A["Multi-Modal: 2-10ms"]
        L2B["Temporal: 0.5-3ms"]
        L2C["Alignment: 1-3ms"]
        L2T["TOTAL: 3-16ms"]
    end

    subgraph L3["LAYER 03 STATE"]
        L3A["World State: <0.1ms"]
        L3B["Entity: 0.1ms"]
        L3C["Trajectory: 0.1ms"]
        L3D["Event: 0.2ms"]
        L3T["TOTAL: 0.3-0.5ms"]
    end

    subgraph L5["LAYER 05 REASONING"]
        L5A["Hypothesis: 5-15ms"]
        L5B["Fast Verify: 1-5ms"]
        L5C["Deep VLM: 800-3500ms"]
        L5D["Evidence Graph: 1-5ms"]
        L5T["FAST: 6-20ms"]
        L5T2["DEEP: 800-3500ms"]
    end

    subgraph L6["LAYER 06 CALIBRATION"]
        L6A["Confidence: <1ms"]
        L6B["Conformal: 2-5ms"]
        L6C["Temperature: <0.1ms"]
        L6D["Claim: 1-2ms"]
        L6T["TOTAL: 3-8ms"]
    end

    L1T --> L2T
    L2T --> L3T
    L3T --> L5T
    L5T --> L6T
    L5T2 -.->|"if needed"| L6T
```

---

## 7. System Topology — Physical Layout

```mermaid
graph TB
    subgraph TIER1["TIER 1: 1x RTX 4090 (24GB)"]
        direction TB
        subgraph G0["GPU 0"]
            direction TB
            G0A["Detection: RF-DETR-S 1.5GB"]
            G0B["Pose: DETRPose-S 0.5GB"]
            G0C["Tracking: ByteTrack 0.05GB"]
            G0D["VLM: Qwen3-VL-30B FP8 20GB"]
            G0E["CUDA Context: 1GB"]
            G0T["Total: 23.05GB / 24GB"]
        end
        subgraph CPU0["Host CPU"]
            CPU0A["Ring Buffers: Pre-allocated"]
            CPU0B["Job Queues: Bounded"]
            CPU0C["REST API: FastAPI"]
            CPU0D["SQLite: Entity/Event/Claim DB"]
            CPU0E["FAISS: Vector Index"]
        end
    end

    subgraph TIER3["TIER 3: 4x A100/H100"]
        direction LR
        subgraph G0T3["GPU 0: Perception"]
            G0T3A["RF-DETR-L"]
            G0T3B["DETRPose-L"]
            G0T3C["ByteTrack"]
        end
        subgraph G1T3["GPU 1: Fast Reasoning"]
            G1T3A["Hypothesis Engine"]
            G1T3B["Fast Verifier"]
            G1T3C["Evidence Graph"]
        end
        subgraph G2T3["GPU 2: Deep Reasoning"]
            G2T3A["Qwen 3.5-397B"]
            G2T3B["TP=4 across 4 GPUs"]
        end
        subgraph G3T3["GPU 3: Forensics + Memory"]
            G3T3A["Deepfake Ensemble"]
            G3T3B["C2PA/SynthID"]
            G3T3C["Long-Term Memory"]
        end
    end

    subgraph NETWORK["Network Topology"]
        direction LR
        N1["10GbE: API/Streaming"]
        N2["NVLink: GPU to GPU 600+GB/s"]
        N3["PCIe: GPU to CPU 64GB/s"]
    end
```
