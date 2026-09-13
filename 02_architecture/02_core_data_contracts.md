# Core data contracts

## Observation

```json
{
  "id": "obs_001",
  "time": 12.53,
  "modality": "vision",
  "entity": "person_7",
  "region": [0.31,0.22,0.18,0.52],
  "feature_ref": "...",
  "detector_confidence": 0.96
}
```

## State estimate

```json
{
  "entity": "person_7",
  "time": 12.53,
  "pose": "...",
  "velocity": "...",
  "occluded_parts": ["left_hand"],
  "state_distribution": "...",
  "uncertainty": "..."
}
```

## Evidence

```json
{
  "id": "ev_104",
  "source": ["obs_001","state_81"],
  "relation": "supports",
  "target": "hyp_21",
  "weight": 0.82
}
```

## Hypothesis

```json
{
  "id": "hyp_21",
  "claim": "player_A_kicked_ball",
  "status": "candidate",
  "support": ["ev_104","ev_110"],
  "contradictions": [],
  "confidence": {
    "perception": 0.96,
    "temporal": 0.91,
    "motion": 0.94,
    "reasoning": 0.84,
    "calibrated": 0.89
  }
}
```

Illustrative values are schema examples only.

---

## PRODUCTION DATA CONTRACT STANDARDS (2026 verified):

**NVIDIA NvSchema (production default):**
- Format: Protobuf (default for Kafka deployments) or JSON Schema.
- Timestamp: RFC3339 UTC (`2006-01-02T15:04:05.999Z`).
- Core types: Boolean, Integer, Float, Long, String, lvalue (enum), Timestamp, Array.
- VisionLLM Message: version, timestamp, startFrameId, endFrameId, sensor info, llm (queries, responses, embeddings), info (map<string,string>).
- Incident Message: sensorId, timestamp, objectIds, frameIds, category, isAnomaly, LLM info.

**FlatBuffers (latency-critical zero-copy paths):**
- Decode: 0.08s/1M ops vs Protobuf 302s vs JSON 583s.
- Zero memory for decoded objects. Wire format 344 bytes vs Protobuf 228 vs JSON 1,475.
- Use when decode latency matters more than wire size.

**C2PA v2.4 (April 2026):**
- Formats: CBOR, JSON, JSON-LD. Containers: JUMBF (JPEG), BMFF (MP4).
- Signatures: COSE (CBOR), X.509 certificates. Hashing: Merkle trees for video chunks.
- Live video: Per-segment manifest boxes + Verifiable Segment Info method.
- Security: IACR analysis (ePrint 2026/804) found 7 serious problems — validators accept known compromised certificates, exclusion range allows undetectable alterations.

---

## REALITY CHECK 2026: CONTRACT UPDATES

### NON-NEGOTIABLE ADDITIONS FOR REALISM:
Every contract below has:
1. **Monotonic nanosecond timestamp (`ts_ns`). Wall clock for ordering; not user-supplied.
2. **Producing model identifier** (model_name + version + hash of weights or checkpoint**). Audit trail mandatory.
3. **Hardware context tag** (`context_id` from context window reference) links all objects.
4. **JSON schema version** — semver. No unversioned blobs.

### Observation (REAL 2026 contract):
```json
{
  "schema_version": "1.0.0",
  "id": "obs_001",
  "ts_ns": 12530000000,
  "context_id": "stream_17",
  "time_ms_offset": 12530,
  "modality": "vision",
  "source": {
    "stream_id": "cam_main",
    "frame_num": 376,
    "frame_ts_ns": 12530000000
  },
  "producing_model": {
    "name": "RF-DETR-S",
    "version": "1.8.2",
    "checkpoint_sha256": "a1b2c3..."
  },
  "producing_latency_ms": 3.4,
  "entity": "person_7",
  "entity_alternate_ids_topk": [{"id": "person_7b", "score": 0.23}],
  "region_norm": [0.31, 0.22, 0.18, 0.52],
  "keypoints_2d_norm": { "joints": [...], "visibility_score": [...] },
  "feature_ref": "sha256:abc123...",
  "feature_embedding_ref": "...",
  "detector_score_raw": 0.96,
  "input_quality_flags": {
    "blur_score": 0.08,
    "occluded_frac": 0.0,
    "compressed": false,
    "detector_in_distribution": 0.91
  }
}
```

### State estimate (REAL 2026 contract — DISTRIBUTIONS, NOT POINTS):
```json
{
  "schema_version": "1.0.0",
  "entity": "person_7",
  "ts_ns": 12530000000,
  "state_window_ms": [12520, 12530],
  "producing_model": {"name": "Kalman+SMPL_lite", "version":"0.3"},
  "pose_3d": {
    "joints_mean_meters": [...],
    "per_joint_covariance": [[...],...],
    "per_joint_visibility_score": [0.97, 0.94, 0.31, ...],
    "per_joint_estimation_method": ["direct_observation", "direct_observation", "temporal_extrapolation", ...]
  },
  "velocity_mps": { "mean": [...], "covariance": [...] },
  "acceleration_mps2": { "mean": [...], "covariance": [...] },
  "occluded_parts": ["left_hand"],
  "state_distribution_type": "gaussian_mixture_3",
  "state_distribution": [
    {"weight": 0.72, "joints_mean": [...], "joints_cov": [...]},
    {"weight": 0.21, "joints_mean": [...], "joints_cov": [...]},
    {"weight": 0.07, "joints_mean": [...], "joints_cov": [...]}
  ],
  "uncertainty_summary": {
    "perception": 0.96,
    "temporal_consistency": 0.91,
    "occlusion_penalty_applied": true,
    "kalman_innovation_mahalanobis": 1.2
  },
  "physics_constraint_violated": false,
  "projected_to_feasible": true
}
```

### Evidence (REAL 2026 contract — log-likelihood, not arbitrary 0-1):
```json
{
  "schema_version": "1.0.0",
  "id": "ev_104",
  "ts_ns": 12532000000,
  "source": ["obs_001", "state_81"],
  "source_modality": ["vision", "state_estimate"],
  "producing_model": [{"name":"RF-DETR-S",...}, {"name":"HybrIK-lite"}],
  "relation": "supports",
  "target": "hyp_21",
  "weight_llr": 1.47,
  "weight_estimator": "calibration_set_logistic_estimate_v2",
  "evidence_spatial_region": {"entity": "ball_2", "bbox_norm":[...]},
  "evidence_temporal_range_ms": [12400, 12540],
  "input_quality_summary": {
    "min_visibility": 0.4,
    "compressed": true
  }
}
```

### Hypothesis (REAL 2026 contract — prediction sets, conformal, STALE AWARE):
```json
{
  "schema_version": "1.0.0",
  "id": "hyp_21",
  "claim": "player_A_kicked_ball",
  "type": "inferred",
  "epistemic_status": "physical_discontinuity_plus_rule",
  "status": "supported",
  "support": ["ev_104", "ev_110", "rule_soccer_14_1"],
  "contradictions": [],
  "hypothesis_set_ranking": [
    {"hyp_id": "hyp_21", "posterior": 0.72},
    {"hyp_id": "hyp_22", "posterior": 0.21},
    {"hyp_id": "hyp_23", "posterior": 0.07}
  ],
  "conformal_prediction_set_alpha_05": ["hyp_21", "hyp_22"],
  "conformal_prediction_set_alpha_10": ["hyp_21"],
  "confidence": {
    "perception": 0.96,
    "temporal": 0.91,
    "motion": 0.94,
    "cross_modal_agreement": 0.88,
    "reasoning": 0.84,
    "calibrated": 0.89,
    "calibration_method": "split_conformal_alpha_01_v3",
    "calibration_coverage_verified_on": "heldout_soccer_v1_split2",
    "prediction_set_size_at_alpha_05": 2
  },
  "snapshot_ts_ns": 12500000000,
  "published_ts_ns": 14220000000,
  "claim_staleness_ms": 1720,
  "inconclusive_justification": null
}
```
