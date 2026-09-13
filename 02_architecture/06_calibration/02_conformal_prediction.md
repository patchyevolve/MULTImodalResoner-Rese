# Conformal Prediction Architecture

## Purpose

Provide prediction sets with formal coverage guarantees — instead of a single point prediction, output a set of hypotheses that contains the true answer with probability ≥ 1-α. This is the mathematically principled way to quantify uncertainty.

---

## Interfaces

### Input

```
ConformalInput {
  hypotheses:      Hypothesis[]             # ranked hypotheses
  scores:          float[]                  # nonconformity scores
  alpha:           float                    # significance level, default 0.05
  calibration_set: CalibrationPoint[]       # held-out calibration data
}
```

### Output

```
ConformalOutput {
  prediction_set:  string[]                 # hypothesis IDs in set
  alpha:           float
  coverage_guarantee: float                 # 1 - alpha
  set_size:        int
  calibrated:      bool
  calibration_dataset: string
  confidence:      float
}

CalibrationPoint {
  hypothesis_id:   string
  score:           float                    # nonconformity score
  is_correct:      bool                     # ground truth
}
```

### API

```
calibrate(calibration_set: CalibrationPoint[], alpha: float) -> ConformalCalibrator
predict(hypotheses: Hypothesis[], scores: float[], calibrator: ConformalCalibrator) -> ConformalOutput
adaptive_predict(hypotheses: Hypothesis[], scores: float[], calibrator: ConformalCalibrator, budget: int) -> ConformalOutput
```

---

## Data Flow

```mermaid
graph TB
    A[Calibration Set<br/>500 samples] --> B[Compute Scores<br/>1-2ms]
    B --> C[Find Threshold τ<br/>0.1ms]
    C --> D[Prediction Set<br/>{h: score ≤ τ}]
    D --> E{Set Size}
    E -->|= 1| F[High Confidence<br/>Publish Claim]
    E -->|= 2-3| G[Report Alternatives]
    E -->|> 3| H[High Uncertainty<br/>Trigger Investigation]
```

---

## Data Contracts

### Split Conformal Method

```
1. Split data into calibration set (held-out) and test set.
2. Compute nonconformity scores on calibration set:
   score_i = 1 - P(y_true | hypothesis_i)   # how "surprising" the true label is
3. Find threshold τ = quantile(1-α) of calibration scores.
4. Prediction set = {hypotheses where score ≤ τ}
5. Coverage guarantee: P(y_true ∈ prediction set) ≥ 1 - α
```

### Adaptive Conformal (from EACL 2026):

```
CP_r-value method:
- Reduces prediction set sizes by 7-44% vs standard CP.
- At α=0.05 on ImageNet: set size 5.7 vs 10.1 (43.6% reduction).
- Gains shrink as model accuracy increases.
```

### Configuration

```
ConformalConfig {
  alpha:           float                    # default 0.05 (95% coverage)
  calibration_size: int                     # default 500
  adaptive:        bool                     # default true (CP_r-value)
  retrain_interval: int                     # recalibrate every N predictions, default 100
  min_set_size:    int                      # default 1
  max_set_size:    int                      # default 5
}
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Score computation | 1–2ms | Per hypothesis |
| Threshold lookup | 0.1ms | Quantile on sorted array |
| Set construction | 0.5–1ms | Filter by threshold |
| Recalibration | 5–20ms | Periodic, async |
| **Total (prediction)** | **2–5ms** | |
| **Total (recalibration)** | **5–20ms** | Async |

---

## Dependencies

### Upstream
- `05_reasoning/01_hypothesis_engine.md` — Hypotheses and scores
- `04_memory/02_working_memory.md` — Active hypothesis set

### Downstream
- `06_calibration/04_claim_output.md` — Prediction sets in claims
- `05_reasoning/03_deep_vlm_reasoner.md` — Adaptive verification budget

---

## Reality Check 2026

### EACL 2026 ("The Art of Saying 'Maybe'"):
- 18 VLMs, 6 datasets, 21,000 questions.
- Larger models = better uncertainty quantification.
- Mathematical/reasoning tasks elicit poorest uncertainty across ALL models.

### Proof-of-Perception (CVPR 2026):
- Conformal sets at each node of reasoning DAG.
- Adaptive controller uses set size to decide: accept, retry, or expand.
- +4.2% DocVQA, +3.6% ChartQA over baselines.

### Critical Finding (arXiv:2608.19376, Aug 2026):
- On ImageNet-Sketch, worst-class coverage falls to ≈0 despite marginal coverage of ~0.86.
- Marginal coverage guarantees do NOT ensure class-conditional safety under distribution shift.
- **Use conformal sets as a tool, not a guarantee.** Report coverage alongside confidence.

### Practical Implementation:
- Recalibrate every 100 predictions on held-out data.
- Report `conformal_prediction_set_alpha_05` and `conformal_prediction_set_alpha_10` in every claim.
- Set size > 3 → system is uncertain, trigger investigation.
