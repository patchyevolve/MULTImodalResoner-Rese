# Multi-Modal Fusion Architecture

## Purpose

Combine evidence from vision, audio, text, and metadata into a unified representation. Ensures modalities reinforce rather than contradict each other. Critical for claims that span multiple modalities (e.g., "the referee blew the whistle while raising the red card").

---

## Interfaces

### Input

```
FusionInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  vision:          WorldStateSnapshot        # entity states from perception
  audio:           AudioFeatures             # from audio pipeline
  text:            TextFeatures              # from OCR pipeline
  metadata:        Map[string]string         # source, camera, provenance
  modality_mask:   uint8                    # bitmask: 0b0001=vision, 0b0010=audio, 0b0100=text, 0b1000=meta
}
```

### Output

```
FusedRepresentation {
  frame_id:        uint64
  timestamp_ns:    uint64
  fused_embedding: [768]float               # multimodal embedding
  modality_contributions: Map[string, float] # weight per modality
  cross_modal_agreement: float              # 0-1, inter-modal consistency
  confidence:      float
  conflicts:       ModalityConflict[]        # contradictions between modalities
}

ModalityConflict {
  modality_a:      string
  modality_b:      string
  conflict_type:   Enum                     # TEMPORAL_MISMATCH | SEMANTIC_MISMATCH | SPATIAL_MISMATCH
  severity:        float                    # 0-1
  resolution:      string                   # which modality to trust
}
```

### API

```
fuse(input: FusionInput, config: FusionConfig) -> FusedRepresentation
```

---

## Data Contracts

### Fusion Strategies

| Strategy | Latency | When to Use |
|---|---|---|
| Early fusion (concatenation) | <1ms | Simple, fast, same-space embeddings |
| Late fusion (score voting) | 1–2ms | Different modality outputs |
| Attention fusion (cross-attention) | 5–10ms | Complex relationships |
| Gated fusion (learned gate) | 2–5ms | Adaptive modality weighting |

### Configuration

```
FusionConfig {
  strategy:        Enum                     # EARLY | LATE | ATTENTION | GATED
  modality_weights: Map[string, float]      # default weights per modality
  conflict_resolution: Enum                 # TRUST_HIGHER_CONFIDENCE | TRUST_VISION | LEARNED
  min_modalities:  int                      # minimum modalities for reliable fusion, default 2
  embedding_dim:   int                      # output dimension, default 768
}
```

---

## Latency Budget

| Strategy | Budget | Notes |
|---|---|---|
| Early fusion | <1ms | Concatenate + linear |
| Late fusion | 1–2ms | Weighted score combination |
| Attention fusion | 5–10ms | Cross-attention layers |
| Gated fusion | 2–5ms | Gate network + weighted sum |

---

## Dependencies

### Upstream
- `03_state/01_world_state.md` — Vision state
- `01_perception/07_audio_features.md` — Audio features
- `01_perception/05_ocr.md` — Text features

### Downstream
- `05_reasoning/01_hypothesis_engine.md` — Fused input for hypothesis generation
- `06_calibration/01_confidence_decomposition.md` — Cross-modal agreement score
- `03_state/04_event_detection.md` — Multi-modal event triggers

---

## Reality Check 2026

### CLASH (contrastive audio-language, AAAI 2026):
- 69–75% accuracy on AV-contradiction detection.
- Detects when audio contradicts visual (e.g., crowd cheers vs player falls).
- Useful for cross-modal conflict detection.

### RAPID (ICLR 2026):
- Perception-reasoning disentanglement via dual encoders.
- 20+ point impact on temporal reasoning tasks.
- Suggests: keep perception features and reasoning features separate until deep reasoning stage.

### Practical Decision:
- **Fast path (<5ms):** Late fusion — weighted scores from each modality.
- **Deep path (async):** Attention fusion via VLM — cross-attention over all modalities.
- **Always report:** Cross-modal agreement score and modality conflicts.
