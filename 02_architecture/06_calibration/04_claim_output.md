# Claim Output Architecture

## Purpose

Produce the final structured claim with all metadata — confidence decomposition, prediction sets, evidence references, staleness, and epistemic status. This is the output contract that consumers (UI, API, downstream systems) receive.

---

## Interfaces

### Output

```
Claim {
  id:              string
  schema_version:  string                   # "1.0.0"
  timestamp_ns:    uint64
  published_ts_ns: uint64

  # Core claim
  claim:           string                   # natural language claim
  type:            Enum                     # OBSERVED | INFERRED | PREDICTED | HYPOTHESIS | CAUSAL | COUNTERFACTUAL | UNKNOWN | INCONCLUSIVE
  epistemic_status: Enum                    # CONFIRMED | PROBABLE | POSSIBLE | SPECULATIVE | UNKNOWN

  # Confidence
  confidence:      ConfidenceDecomposition
  conformal_prediction_set_alpha_05: string[]  # hypothesis IDs at α=0.05
  conformal_prediction_set_alpha_10: string[]  # hypothesis IDs at α=0.10
  prediction_set_size_at_alpha_05: int

  # Evidence
  support:         Evidence[]               # supporting evidence
  contradictions:  Evidence[]               # contradicting evidence
  evidence_graph_snapshot: string           # pointer to graph snapshot

  # Source tracking
  producing_model: string                   # model that generated this claim
  producing_model_version: string
  checkpoint_sha256: string

  # Quality
  claim_staleness_ms: float                 # time from snapshot to publication
  stale:            bool                    # true if staleness > threshold
  input_quality:   InputQualityFlags

  # Provenance
  source_stream:   string
  source_frames:   uint64[]
  event_id:        string                   # triggering event (if any)
  hypothesis_id:   string                   # source hypothesis (if any)

  # Domain
  domain:          string                   # "sports" | "news" | "general"
  domain_metadata: Map[string, Any]         # domain-specific fields
}
```

### API

```
create_claim(hypothesis: Hypothesis, evidence: Evidence[], confidence: ConfidenceDecomposition) -> Claim
validate_claim(claim: Claim) -> bool
serialize_claim(claim: Claim, format: string) -> bytes  # "json" | "protobuf"
```

---

## Data Contracts

### Claim Validation Rules

```
1. claim is non-empty string
2. type is valid enum value
3. confidence.overall is in [0, 1]
4. confidence components are in [0, 1]
5. prediction_set_size_at_alpha_05 >= 1
6. support is non-empty for CONFIRMED/PROBABLE status
7. stale flag is set if claim_staleness_ms > 5000
8. producing_model is non-empty
```

### Serialization Formats

| Format | Use Case | Size | Speed |
|---|---|---|---|
| JSON | API response, debugging | Larger | Slower |
| Protobuf | Kafka, internal messaging | Smaller | Faster |
| FlatBuffer | Zero-copy, hot path | Smallest | Fastest |

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Claim creation | 0.5–1ms | Assembly |
| Validation | 0.1ms | Rule checks |
| Serialization (JSON) | 0.5–1ms | |
| Serialization (Protobuf) | 0.1–0.3ms | |
| **Total** | **1–2ms** | |

---

## Dependencies

### Upstream
- `05_reasoning/01_hypothesis_engine.md` — Hypothesis data
- `05_reasoning/04_evidence_graph.md` — Evidence
- `06_calibration/01_confidence_decomposition.md` — Confidence
- `06_calibration/02_conformal_prediction.md` — Prediction sets

### Downstream
- User-facing API/UI
- `04_memory/03_long_term.md` — Persistent storage
- `04_memory/04_episodic_memory.md` — Episode creation

---

## Reality Check 2026

### Claim Taxonomy (from research):

| Type | When to Use | Confidence Range |
|---|---|---|
| OBSERVED | Directly visible in frames | 0.8–1.0 |
| INFERRED | Derived from multiple observations | 0.5–0.9 |
| PREDICTED | Future state | 0.2–0.7 |
| HYPOTHESIS | Candidate explanation | 0.1–0.5 |
| CAUSAL | Cause-effect relationship | 0.3–0.8 |
| COUNTERFACTUAL | "What if" scenario | 0.05–0.3 |
| INCONCLUSIVE | Insufficient evidence | N/A |

### ClaimFlow (2026):
- 63.5% of claims never reused; only 11.1% ever challenged.
- Widely propagated claims are more often reshaped through qualification than refuted.
- Report `epistemic_status` as part of every claim — it's not decoration.
