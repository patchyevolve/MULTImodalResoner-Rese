# Fast Verifier Architecture

## Purpose

Quickly verify or reject hypotheses using rule-based checks and lightweight models. Runs in the critical path (<5ms). If a hypothesis can be verified or rejected quickly, skip the expensive VLM reasoning.

---

## Interfaces

### Input

```
VerificationInput {
  hypothesis:      Hypothesis
  world_state:     WorldStateSnapshot
  evidence:        Evidence[]
  domain_rules:    DomainRule[]
}
```

### Output

```
VerificationResult {
  hypothesis_id:   string
  verdict:         Enum                     # SUPPORTED | REFUTED | INCONCLUSIVE
  confidence:      float
  method:          string                   # "trajectory_check" | "pose_check" | "rule_check"
  evidence_used:   Evidence[]
  latency_ms:      float
  needs_vlm:       bool                     # true if inconclusive, needs deep reasoning
}
```

### API

```
verify(hypothesis: Hypothesis, state: WorldStateSnapshot) -> VerificationResult
batch_verify(hypotheses: Hypothesis[], state: WorldStateSnapshot) -> VerificationResult[]
```

---

## Data Contracts

### Verification Methods

| Method | Latency | Precision | Notes |
|---|---|---|---|
| Trajectory check | 0.1–0.5ms | High for motion claims | Does trajectory match claim? |
| Pose check | 0.5–1ms | High for action claims | Does pose match claimed action? |
| Rule check | 0.01–0.1ms | High for domain rules | Does state violate domain rules? |
| Proximity check | 0.01ms | Moderate | Are entities close enough? |
| Temporal check | 0.01ms | High | Is timing consistent? |

### Decision Tree

```
1. Run rule_check → REFUTED? → Done (refuted)
2. Run trajectory_check → SUPPORTED? → Done (supported)
3. Run pose_check → SUPPORTED? → Done (supported)
4. All inconclusive → needs_vlm = true → delegate to deep VLM
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Rule check | 0.01–0.1ms | Simple comparisons |
| Trajectory check | 0.1–0.5ms | |
| Pose check | 0.5–1ms | |
| Proximity check | 0.01ms | |
| **Total** | **1–5ms** | |

---

## Dependencies

### Upstream
- `05_reasoning/01_hypothesis_engine.md` — Hypotheses to verify
- `03_state/01_world_state.md` — Current state

### Downstream
- `05_reasoning/03_deep_vlm_reasoner.md` — Inconclusive hypotheses
- `06_calibration/04_claim_output.md` — Verified/refuted claims

---

## Reality Check 2026

### Why Fast Verification Matters:
- VLM inference: 800–3500ms. Fast verification: 1–5ms.
- If 80% of hypotheses can be verified/rejected fast, only 20% need VLM.
- This reduces VLM load by 5×, keeping the system responsive.

### Verification Accuracy:
- Trajectory checks: >95% accuracy for motion claims.
- Pose checks: >90% accuracy for action claims (depends on pose quality).
- Rule checks: 100% accuracy (deterministic).
- Overall fast verification precision: ~85-90%.
