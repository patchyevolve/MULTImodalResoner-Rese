# Confidence Decomposition Architecture

## Purpose

Decompose claim confidence into component factors — perception quality, temporal consistency, cross-modal agreement, reasoning quality, and calibration. Never expose a single confidence number; always store components so downstream systems can determine WHY confidence is low.

---

## Interfaces

### Input

```
ConfidenceInput {
  perception_score: float                    # detection/pose confidence
  temporal_consistency: float                # cross-frame consistency
  cross_modal_agreement: float              # vision-audio-text alignment
  reasoning_confidence: float               # VLM/rule reasoning quality
  calibration_method: string                # "temperature" | "conformal" | "isotonic"
}
```

### Output

```
ConfidenceDecomposition {
  perception:      float                    # 0-1
  temporal:        float                    # 0-1
  motion:          float                    # 0-1
  cross_modal_agreement: float              # 0-1
  reasoning:       float                    # 0-1
  calibrated:      float                    # 0-1, after calibration
  calibration_method: string
  calibration_verified_on: string           # dataset used for calibration
  overall:         float                    # weighted combination
  uncertainty_sources: string[]             # human-readable list of what's uncertain
}
```

### API

```
decompose(input: ConfidenceInput) -> ConfidenceDecomposition
recalibrate(decomp: ConfidenceDecomposition, method: string) -> ConfidenceDecomposition
```

---

## Data Contracts

### Decomposition Formula

```
overall = w_p × perception
        + w_t × temporal
        + w_m × motion
        + w_c × cross_modal
        + w_r × reasoning

Where:
  w_p = 0.30, w_t = 0.20, w_m = 0.15, w_c = 0.15, w_r = 0.20
  (weights tunable per domain)
```

### Quality Flags

```
ConfidenceQualityFlags {
  low_perception:     bool    # perception < 0.5
  low_temporal:       bool    # temporal < 0.5
  modal_conflict:     bool    # cross_modal_agreement < 0.3
  stale:              bool    # claim_staleness_ms > 5000
  occluded:           bool    # key entity occluded
  low_vlm_confidence: bool    # VLM output confidence < 0.5
}
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Decomposition | <0.1ms | Weighted sum |
| Recalibration | <0.5ms | Temperature scaling |
| **Total** | **<1ms** | |

---

## Dependencies

### Upstream
- `01_perception/` — Perception confidence scores
- `02_fusion/01_multimodal_fusion.md` — Cross-modal agreement
- `05_reasoning/` — Reasoning confidence

### Downstream
- `06_calibration/02_conformal_prediction.md` — Calibration input
- `06_calibration/04_claim_output.md` — Confidence in claims

---

## Reality Check 2026

### VL-Calibration (ACL 2026):
- Reduces ECE from 0.421 → 0.098 on Qwen3-VL-4B (76.7% reduction).
- RL-based framework that decouples visual and reasoning confidence.
- Training overhead: 11%, converges in <100 steps.

### Key Insight:
- VLM-RobustBench: geometric distortions cause up to 34pp accuracy drops.
- Low-severity perturbations often degrade more than visually severe ones.
- Confidence decomposition must include input quality assessment.
