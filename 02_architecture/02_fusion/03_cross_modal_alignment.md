# Cross-Modal Alignment Architecture

## Purpose

Synchronize and align features across different modalities and time bases. Ensures that when audio says "goal" at T=5.0s, it aligns with the visual event at T=5.0s, not T=4.8s or T=5.3s. Critical for accurate multi-modal claims.

---

## Interfaces

### Input

```
AlignmentInput {
  vision_features: Tensor[T_v][D_v]         # vision features over time
  audio_features:  Tensor[T_a][D_a]         # audio features over time
  text_timestamps: Timestamp[]              # OCR/event timestamps
  modality_offsets: Map[string, float]      # known clock offsets
}
```

### Output

```
AlignmentOutput {
  aligned_features: Tensor[T][D]            # time-aligned multimodal features
  sync_score:      float                    # 0-1, overall sync quality
  alignment_map:   Map[string, int[]]       # which frames map to which audio chunks
  drift_estimate:  float                    # estimated clock drift (ms)
  confidence:      float
}
```

### API

```
align(input: AlignmentInput, config: AlignmentConfig) -> AlignmentOutput
estimate_sync(audio: AudioFeatures, vision: VisionFeatures) -> float
```

---

## Data Contracts

### Alignment Methods

| Method | Latency | Accuracy | Notes |
|---|---|---|---|
| Timestamp-based (NTP) | <0.1ms | Depends on clock | Fastest, if clocks synced |
| Feature correlation | 1–3ms | Good | Cross-modal cosine similarity |
| Attention-based | 5–10ms | Best | Learned alignment |

### Configuration

```
AlignmentConfig {
  method:          Enum                     # TIMESTAMP | CORRELATION | ATTENTION
  max_drift_ms:    float                    # max allowed drift, default 100
  sync_window_ms:  int                      # window for correlation, default 500
  min_sync_score:  float                    # below this, flag as out-of-sync
}
```

---

## Latency Budget

| Method | Budget | Notes |
|---|---|---|
| Timestamp | <0.1ms | Simple subtraction |
| Correlation | 1–3ms | Sliding window correlation |
| Attention | 5–10ms | Cross-attention |

---

## Dependencies

### Upstream
- `01_perception/07_audio_features.md` — Audio features
- `02_fusion/01_multimodal_fusion.md` — Pre-fusion features

### Downstream
- `02_fusion/01_multimodal_fusion.md` — Aligned input for fusion
- `06_calibration/01_confidence_decomposition.md` — Sync quality → confidence

---

## Reality Check 2026

### Video-MME-v2 Finding:
- Text modality is critical: Thinking mode +3.8 to +5.8 with subtitles.
- Without text cues, Thinking mode can cause regression.
- Cross-modal alignment score should be reported as a confidence factor.

### Practical Notes:
- Most production video has A/V synced at encoding (latency <33ms).
- Desync becomes significant in: live streams (buffering), multi-camera setups, edited content.
- Report `sync_score` in every fused output — downstream reasoning needs it.
