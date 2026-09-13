# Temperature Scaling Architecture

## Purpose

Post-hoc calibration of model confidence scores. Adjusts the temperature of softmax outputs to match observed accuracy. Simple, fast, and effective — the first calibration method to apply before more complex approaches.

---

## Interfaces

### Input

```
TemperatureInput {
  logits:          float[]                  # raw model logits
  temperature:     float                    # learned temperature parameter
}
```

### Output

```
TemperatureOutput {
  calibrated_probs: float[]                 # calibrated probability distribution
  confidence:      float                    # max calibrated probability
  entropy:         float                    # distribution entropy
}
```

### API

```
calibrate(logits: float[], temperature: float) -> TemperatureOutput
learn_temperature(val_logits: float[][], val_labels: int[], lr: float) -> float
```

---

## Data Contracts

### Temperature Scaling Formula

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)

Where:
  z_i: raw logit for class i
  T: temperature parameter (T > 1 = softer, T < 1 = sharper)
  Learned on validation set by minimizing NLL
```

### Configuration

```
TemperatureConfig {
  temperature:     float                    # learned, typically 1.0-2.0
  learning_rate:   float                    # default 0.01
  max_iterations:  int                      # default 1000
  retrain_interval: int                     # default 1000 predictions
}
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Apply scaling | <0.1ms | Single division + softmax |
| Learn temperature | 5–20ms | On validation set, async |
| **Total (apply)** | **<0.1ms** | |

---

## Dependencies

### Upstream
- Model outputs (logits)

### Downstream
- `06_calibration/01_confidence_decomposition.md` — Calibrated confidence
- `06_calibration/04_claim_output.md` — Calibrated claims

---

## Reality Check 2026

### ICML 2024 (Temperature Scaling for VLMs):
- "Significantly and consistently improves calibration, even across distribution shifts."
- VLMs can be calibrated with a very small set of examples (data-efficient).
- Single-prompt calibration transfers effectively to other prompts.
- After temperature scaling, VLMs surpass other model classes in uncertainty estimation.

### Process Supervision (COLM 2026):
- Process-level margin supervision outperforms final-step-only supervision.
- Margin-based supervision produces more reliable calibration than Brier-style score matching.
- Baseline ECE on AIME25: ~0.33-0.37 range for GRPO variants.

### Practical Notes:
- Temperature scaling is **always worth applying** — near-zero cost, meaningful improvement.
- Learn temperature on a held-out calibration set (500+ samples).
- Retrain periodically as model behavior changes.
