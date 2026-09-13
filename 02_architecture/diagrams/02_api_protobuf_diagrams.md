# API & Protobuf Schema Diagrams

> Visual representations of all message formats, API endpoints, and serialization schemas.

---

## 1. Protobuf Message Hierarchy

```mermaid
graph TB
    subgraph CORE["Core Messages"]
        FOB[FrameObservationBatch]
        WSS[WorldStateSnapshot]
        CLM[Claim]
        EVT[Event]
    end

    subgraph PERCEPTION["Perception Messages"]
        DET[Detection]
        POSE[PoseResult]
        TRK[Track]
        MASK[MaskResult]
    end

    subgraph STATE["State Messages"]
        ENT[Entity]
        REL[Relation]
        SCENE[Scene]
        TRAJ[Trajectory]
    end

    subgraph REASONING["Reasoning Messages"]
        HYP[Hypothesis]
        EVD[Evidence]
        EGN[EvidenceGraph]
        CONF[ConfidenceDecomposition]
    end

    subgraph CALIBRATION["Calibration Messages"]
        CPS[ConformalPredictionSet]
        TS[TemperatureScaling]
        IQF[InputQualityFlags]
    end

    FOB --> DET
    FOB --> POSE
    FOB --> TRK
    WSS --> ENT
    WSS --> REL
    WSS --> SCENE
    WSS --> TRAJ
    CLM --> CONF
    CLM --> EVD
    HYP --> EVD
    EGN --> EVD
    CONF --> CPS
    CONF --> TS
```

---

## 2. FrameObservationBatch — Detailed Schema

```mermaid
graph TB
    subgraph FOB["FrameObservationBatch"]
        V["version: string"]
        ID["id: string"]
        TS["timestamp: Timestamp"]
        SID["sensor_id: string"]
        OBJ["objects: repeated Object"]
        FD["frame_data: bytes"]
        CODEC["codec: string"]
        META["metadata: map<string,string>"]
    end

    subgraph OBJ_DEF["Object"]
        OID["id: string"]
        BBOX["bbox: Bbox"]
        SCORE["confidence: float"]
        CID["class_id: int32"]
        CN["class_name: string"]
        EMB["embedding: repeated float"]
        ATTR["attributes: map<string,string>"]
    end

    subgraph BBOX_DEF["Bbox"]
        XC["x_center: float"]
        YC["y_center: float"]
        W["width: float"]
        H["height: float"]
    end

    OBJ --> BBOX_DEF
    FOB --> OBJ_DEF
```

---

## 3. WorldStateSnapshot — Detailed Schema

```mermaid
graph TB
    subgraph WSS["WorldStateSnapshot"]
        FID["frame_id: uint64"]
        TS["timestamp_ns: uint64"]
        ENTS["entities: repeated Entity"]
        TRAJ["trajectories: repeated Trajectory"]
        RELS["relations: repeated Relation"]
        SCN["scene: Scene"]
        EVTS["events: repeated Event"]
        UNC["uncertainty: UncertaintySummary"]
        STALE["staleness_ms: float"]
    end

    subgraph ENT_DEF["Entity"]
        EID["id: string"]
        ECLS["class: string"]
        EBBOX["bbox_norm: Bbox"]
        EPOSE["pose: PoseResult"]
        EVEL["velocity: repeated float"]
        EACC["acceleration: repeated float"]
        EATTR["attributes: map<string,Any>"]
        ECONF["confidence: float"]
        EOCCL["occlusion_level: float"]
        ETSTATE["tracking_state: string"]
    end

    subgraph REL_DEF["Relation"]
        RSUBJ["subject: string (entity_id)"]
        RPRED["predicate: string"]
        ROBJ["object: string (entity_id)"]
        RCONF["confidence: float"]
        RSPAT["spatial: SpatialRelation"]
    end

    subgraph TRAJ_DEF["Trajectory"]
        TID["entity_id: string"]
        TPOS["predicted_positions: repeated"]
        TVEL["velocity: repeated float"]
        TACC["acceleration: repeated float"]
        TTYPE["trajectory_type: string"]
        TCONF["prediction_confidence: float"]
    end

    WSS --> ENT_DEF
    WSS --> REL_DEF
    WSS --> TRAJ_DEF
```

---

## 4. Claim — Detailed Schema

```mermaid
graph TB
    subgraph CLM["Claim"]
        CID["id: string"]
        SVER["schema_version: string"]
        TS["timestamp_ns: uint64"]
        PTS["published_ts_ns: uint64"]
        CTXT["claim_text: string"]
        CTYPE["claim_type: enum"]
        ESTAT["epistemic_status: enum"]
    end

    subgraph CONF_DEF["ConfidenceDecomposition"]
        CP["perception: float"]
        CT["temporal: float"]
        CM["motion: float"]
        CCM["cross_modal_agreement: float"]
        CR["reasoning: float"]
        CCAL["calibrated: float"]
        CMETH["calibration_method: string"]
        COVR["overall: float"]
        CUS["uncertainty_sources: repeated string"]
    end

    subgraph EVD_DEF["Evidence"]
        EID["id: string"]
        ESRC["source: repeated string"]
        EREL["relation: string"]
        ETGT["target: string"]
        EW["weight_llr: float"]
        EWEST["weight_estimator: string"]
    end

    subgraph PROVENANCE["Provenance"]
        PMOD["producing_model: string"]
        PVER["producing_model_version: string"]
        PSHA["checkpoint_sha256: string"]
        PSTALE["claim_staleness_ms: float"]
        PSTALEN["stale: bool"]
        PSTRM["source_stream: string"]
        PFRM["source_frames: repeated uint64"]
        PEID["event_id: string"]
        PHID["hypothesis_id: string"]
        PDOM["domain: string"]
        PDM["domain_metadata: map<string,string>"]
    end

    subgraph CONFORMAL["Conformal Sets"]
        CP05["conformal_prediction_set_alpha_05: repeated string"]
        CP10["conformal_prediction_set_alpha_10: repeated string"]
        PSS["prediction_set_size_at_alpha_05: int"]
    end

    CLM --> CONF_DEF
    CLM --> EVD_DEF
    CLM --> PROVENANCE
    CLM --> CONFORMAL
```

---

## 5. API Endpoint Map

```mermaid
graph TB
    subgraph REST["REST API (FastAPI)"]
        direction TB
        GET1["GET /api/v1/state/current<br/>→ WorldStateSnapshot"]
        GET2["GET /api/v1/state/{timestamp}<br/>→ WorldStateSnapshot"]
        GET3["GET /api/v1/entity/{id}<br/>→ Entity"]
        GET4["GET /api/v1/entity/{id}/history<br/>→ EntityState[]"]
        GET5["GET /api/v1/events?from=&to=&type=<br/>→ Event[]"]
        GET6["GET /api/v1/claims?from=&to=&domain=<br/>→ Claim[]"]
        GET7["GET /api/v1/claims/{id}<br/>→ Claim with evidence"]
        GET8["GET /api/v1/hypotheses/active<br/>→ HypothesisSet"]
        GET9["GET /api/v1/forensics/{media_id}<br/>→ ForensicResult"]
        GET10["GET /api/v1/health<br/>→ SystemHealth"]
        GET11["GET /api/v1/metrics<br/>→ SystemMetrics"]
    end

    subgraph POST["POST Endpoints"]
        POST1["POST /api/v1/stream/register<br/>→ Register video stream"]
        POST2["POST /api/v1/stream/{id}/query<br/>→ Query about stream"]
        POST3["POST /api/v1/forensics/analyze<br/>→ Submit media for analysis"]
        POST4["POST /api/v1/reasoning/investigate<br/>→ Trigger deep investigation"]
    end

    subgraph WEBSOCKET["WebSocket"]
        WS1["WS /api/v1/stream/{id}/live<br/>→ Live claim stream"]
        WS2["WS /api/v1/events/live<br/>→ Live event feed"]
    end

    subgraph KAFKA["Kafka Topics"]
        K1["topic: perception.detections<br/>→ DetectionBatch"]
        K2["topic: state.snapshots<br/>→ WorldStateSnapshot"]
        K3["topic: events.triggered<br/>→ EventTrigger"]
        K4["topic: claims.published<br/>→ Claim"]
        K5["topic: forensics.results<br/>→ ForensicResult"]
        K6["topic: vlm.requests<br/>→ VLMReasoningInput"]
        K7["topic: vlm.responses<br/>→ VLMReasoningOutput"]
    end
```

---

## 6. NvSchema Message Flow (NVIDIA Compatible)

```mermaid
graph LR
    subgraph INPUT["Input"]
        RTSP[RTSP Stream]
    end

    subgraph DECODE["DeepStream Decode"]
        DEC[H.264/H.265/AV1]
    end

    subgraph NVM["NvSchema Messages"]
        NVM1["Frame Message<br/>version, id, timestamp,<br/>sensorId, objects"]
        NVM2["Object Message<br/>id, bbox, confidence,<br/>class, embedding"]
        NVM3["Event Message<br/>type, timestamp,<br/>objectIds, category"]
        NVM4["VisionLLM Message<br/>queries, responses,<br/>embeddings, info"]
    end

    subgraph BROKER["Kafka Broker"]
        K1["perception.objects"]
        K2["state.world"]
        K3["events.triggered"]
        K4["vlm.captions"]
    end

    subgraph OUTPUT["Consumers"]
        C1[Dashboard]
        C2[Alert System]
        C3[Storage]
        C4[Analytics]
    end

    RTSP --> DEC
    DEC --> NVM1
    NVM1 --> NVM2
    NVM2 --> NVM3
    NVM3 --> NVM4
    NVM1 --> K1
    NVM2 --> K2
    NVM3 --> K3
    NVM4 --> K4
    K1 --> C1
    K2 --> C2
    K3 --> C3
    K4 --> C4
```

---

## 7. C2PA Manifest Structure

```mermaid
graph TB
    subgraph MANIFEST["C2PA Manifest (CBOR/JSON-LD)"]
        VER["version: 2.4"]
        SIG["signature: COSE_Sign1"]
        CERT["certificate: X.509"]
        CLAIMS["claim: Assertion[]"]
    end

    subgraph ASSERTIONS["Assertions"]
        A1["creation<br/>creator, when, where"]
        A2["c2pa.hash<br/>Merkle tree root"]
        A3["stds.iptc.location<br/>GPS coordinates"]
        A4["stds.exif<br/>camera settings"]
        A5["c2pa.cloud-data<br/>SynthID watermark"]
        A6["ai-infference<br/>model used, settings"]
    end

    subgraph BMFF["BMFF (MP4) Structure"]
        B1["Init Segment<br/>initHash"]
        B2["Media Segment 1<br/>merkle.hashes[0]"]
        B3["Media Segment 2<br/>merkle.hashes[1]"]
        B4["Media Segment N<br/>merkle.hashes[N]"]
    end

    subgraph VALIDATION["Validation Chain"]
        V1[Extract Manifest]
        V2[Verify Signature]
        V3[Check Certificate Chain]
        V4[Verify Merkle Hashes]
        V5[Check Revocation]
        V6[Result: VALID/INVALID]
    end

    MANIFEST --> ASSERTIONS
    MANIFEST --> BMFF
    MANIFEST --> VALIDATION
```

---

## 8. SynthID Watermark Flow

```mermaid
graph TB
    subgraph EMBED["Embedding (at generation)"]
        E1[AI Model Output]
        E2[SynthID Encoder]
        E3[Watermark Embedded in Pixels/Audio]
    end

    subgraph DETECT["Detection (at verification)"]
        D1[Input Media]
        D2[SynthID Decoder]
        D3{Watermark Found?}
        D4[Confidence Score]
        D5[Provider: Google/OpenAI/etc]
    end

    subgraph ROBUSTNESS["Robustness (verified Jul 2026)"]
        R1["✓ Survives H.264 re-encoding"]
        R2["✓ Survives screenshots"]
        R3["✓ Survives cropping up to ~20%"]
        R4["✗ 50% crop breaks at ~250 iterations"]
        R5["✗ Fragmentation: no cross-provider detection"]
    end

    E1 --> E2
    E2 --> E3
    D1 --> D2
    D2 --> D3
    D3 -->|yes| D4
    D3 -->|no| D5
```

---

## 9. FlatBuffers vs Protobuf Comparison

```mermaid
graph TB
    subgraph FLATBUFFERS["FlatBuffers (Zero-Copy)"]
        FB1["Wire Format: 344 bytes"]
        FB2["Decode+Traverse: 0.08s / 1M ops"]
        FB3["Memory to Decode: 0 bytes"]
        FB4["Transient Memory: 0 KB"]
        FB5["Use: Perception → State (hot path)"]
    end

    subgraph PROTOBUF["Protobuf (Standard)"]
        PB1["Wire Format: 228 bytes"]
        PB2["Decode+Traverse: 302s / 1M ops"]
        PB3["Memory to Decode: 760 bytes"]
        PB4["Transient Memory: 1 KB"]
        PB5["Use: Kafka, Storage, API"]
    end

    subgraph JSON["JSON (Debugging)"]
        J1["Wire Format: 1,475 bytes"]
        J2["Decode+Traverse: 583s / 1M ops"]
        J3["Memory to Decode: 65,689 bytes"]
        J4["Transient Memory: 131 KB"]
        J5["Use: Debugging, logs, config"]
    end

    style FLATBUFFERS fill:#c8e6c9,stroke:#388e3c
    style PROTOBUF fill:#bbdefb,stroke:#1976d2
    style JSON fill:#ffcdd2,stroke:#d32f2f
```

---

## 10. Schema Versioning Timeline

```mermaid
graph LR
    V10["v1.0.0<br/>Initial schemas<br/>Core messages"]
    V11["v1.1.0<br/>+ Conformal sets<br/>+ Staleness tags"]
    V12["v1.2.0<br/>+ Domain metadata<br/>+ Forensic results"]
    V13["v1.3.0<br/>+ Evidence graph<br/>+ Episode store"]
    V20["v2.0.0<br/>Breaking: field renames<br/>+ C2PA v2.4 support"]

    V10 --> V11
    V11 --> V12
    V12 --> V13
    V13 --> V20

    style V10 fill:#c8e6c9,stroke:#388e3c
    style V20 fill:#ffcdd2,stroke:#d32f2f
```
