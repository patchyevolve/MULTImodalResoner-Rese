# Data Schemas Architecture

## Purpose

Define the typed message contracts used across the system. Every component communicates through these schemas — no undocumented fields, no unversioned blobs. Production systems use Protobuf (default) or JSON Schema.

---

## Core Schemas

### FrameObservationBatch (Perception Input)

```protobuf
syntax = "proto3";
package vision;

message FrameObservationBatch {
  string version = 1;
  string id = 2;
  google.protobuf.Timestamp timestamp = 3;
  string sensor_id = 4;
  repeated Object objects = 5;
  bytes frame_data = 6;
  string codec = 7;
  map<string, string> metadata = 8;
}

message Object {
  string id = 1;
  Bbox bbox = 2;
  float confidence = 3;
  int32 class_id = 4;
  string class_name = 5;
  repeated float embedding = 6;
  map<string, string> attributes = 7;
}

message Bbox {
  float x_center = 1;
  float y_center = 2;
  float width = 3;
  float height = 4;
}
```

### WorldStateSnapshot (State Output)

```protobuf
message WorldStateSnapshot {
  uint64 frame_id = 1;
  uint64 timestamp_ns = 2;
  repeated Entity entities = 3;
  repeated Trajectory trajectories = 4;
  repeated Relation relations = 5;
  Scene scene = 6;
  repeated Event events = 7;
  UncertaintySummary uncertainty = 8;
  float staleness_ms = 9;
}

message Entity {
  string id = 1;
  string class = 2;
  Bbox bbox_norm = 3;
  PoseResult pose = 4;
  repeated float velocity = 5;
  repeated float acceleration = 6;
  map<string, string> attributes = 7;
  float confidence = 8;
  float occlusion_level = 9;
  string tracking_state = 10;
}
```

### Claim (Final Output)

```protobuf
message Claim {
  string id = 1;
  string schema_version = 2;
  google.protobuf.Timestamp timestamp = 3;
  google.protobuf.Timestamp published_ts = 4;
  string claim_text = 5;
  string claim_type = 6;
  string epistemic_status = 7;
  ConfidenceDecomposition confidence = 8;
  repeated string conformal_set_05 = 9;
  repeated string conformal_set_10 = 10;
  repeated Evidence support = 11;
  repeated Evidence contradictions = 12;
  string producing_model = 13;
  string producing_model_version = 14;
  float claim_staleness_ms = 15;
  bool stale = 16;
  string source_stream = 17;
  repeated uint64 source_frames = 18;
  string domain = 19;
  map<string, string> domain_metadata = 20;
}

message ConfidenceDecomposition {
  float perception = 1;
  float temporal = 2;
  float motion = 3;
  float cross_modal_agreement = 4;
  float reasoning = 5;
  float calibrated = 6;
  string calibration_method = 7;
  float overall = 8;
  repeated string uncertainty_sources = 9;
}
```

---

## Versioning

### Schema Version Rules

```
Schema versioning: semver (MAJOR.MINOR.PATCH)
  MAJOR: breaking changes (field removal, type change)
  MINOR: new optional fields, new message types
  PATCH: documentation, comments

Every message includes: string version = MAX_FIELD_NUM + 1
```

### Compatibility Rules

1. **Never remove fields.** Deprecate with `deprecated = true`.
2. **Never change field types.** Add new field instead.
3. **Never reuse field numbers.** If field is removed, mark as reserved.
4. **New optional fields are always safe.** Old consumers ignore them.
5. **New required fields are breaking.** Use optional + default value instead.

---

## FlatBuffers (Zero-Copy Hot Path)

For latency-critical paths (perception → state update):

| Property | FlatBuffers | Protobuf | JSON |
|---|---|---|---|
| Decode+Traverse (1M ops) | 0.08s | 302s | 583s |
| Wire format size | 344B | 228B | 1,475B |
| Memory to decode | 0B | 760B | 65,689B |
| Transient memory | 0KB | 1KB | 131KB |

**Use FlatBuffers** for: perception → state update (30 FPS path).
**Use Protobuf** for: Kafka messaging, async workers, long-term storage.

---

## Reality Check 2026

### NVIDIA NvSchema (production):
- Default format: Protobuf for Kafka deployments.
- Timestamp: RFC3339 UTC (`2006-01-02T15:04:05.999Z`).
- VisionLLM Message: queries, responses, embeddings, info map.
- Incident Message: sensor, analytics, anomaly flag, LLM info.

### Design Rules:
1. **Schemas are contracts.** Change only through versioned releases.
2. **All fields documented.** No undocumented fields allowed.
3. **Validation at boundaries.** Validate on ingress, trust internally.
4. **Schema evolution is mandatory.** Plan for v1.1, v1.2, v2.0 from day one.
