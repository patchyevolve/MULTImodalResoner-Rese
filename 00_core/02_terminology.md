# Terminology and claim semantics

## Observation
Directly supported by the current media sample.

Example: "A person is visible at image coordinates ..."

**Reality note:** Detection confidence is measurable (detector AP). Observations carry provenance: which model, which frame, which spatial region. Never treat an observation as "ground truth" — it is an output of a fallible detector with known error rates. RF-DETR-2XL achieves **60.1 mAP** on COCO (ICLR 2026); even the best detector misses ~40% of objects at the 0.5 IoU threshold.

## Inferred state
Not directly visible but supported by temporal or structural evidence.

Example: "The person's wrist is probably behind the occluder."

**Reality note:** Inference is error-prone. State estimates MUST include a distribution or covariance, not a single point. MoRo (3DV 2026) achieves **37.83mm MPJPE** on visible joints and **48.53mm** on occluded joints (28% gap) — but this is SOTA; typical methods see 20–40% degradation. The system should internally track "visibility fraction" or "occlusion score" for every entity part.

## Hypothesis
One of multiple possible explanations for observations.

Example: "The person probably kicked the ball."

**Reality note:** Maintain a BOUNDED hypothesis set, not an unbounded generator. In practice top-3 to top-5 is the computational sweet spot. Hypotheses below a minimum posterior threshold are dropped immediately. SoccerNet 2026 Challenge best action anticipation achieves only **24.08% avg mAP** at 5s horizon — even SOTA models are uncertain about future events. "Active falsification" in 2026 terms = check whether ANY competing hypothesis explains the data equally well before promoting the leading hypothesis to "supported."

## Prediction
A forecast of a future state or event.

Example: "The ball will probably move toward the left side."

**Reality note:** Short-horizon predictions (≤0.5 s, ≤15 frames) for tracked entities are reliable via constant-velocity / Kalman. Long-horizon (>2 s) for human behavior is unreliable across all current model families; forecast distribution width grows quickly. Always report prediction horizon.

## Causal claim
An assertion that an event produced another event.

Example: "The collision caused the change in ball direction."

**Reality note:** DO NOT PERMIT OPEN-WORLD CAUSAL CLAIMS. They are not reliably supportable from raw video alone in 2026. Causal claims may only be emitted when:
1. Domain-specific rules explicitly encode the causal relation (sports rulebook), OR
2. A physical discontinuity in tracked state coincides in time AND the physics model rules out alternative causes, OR
3. An intervention (A/B test) was performed (rare outside synthetic data).
Correlation in time ≠ causation. The system must tag "temporal_correlation" separately from "causal_claim."

## Unknown / inconclusive
Evidence is insufficient for a reliable claim.

**Reality note:** This is NOT a failure mode — it is the expected default for many realistic inputs. In synthetic media detection, RA-Bench 2026 shows 60–70% of real-world degraded clips should correctly land in "inconclusive" for at least one detector family. SOTA open-source detectors lose **45–50% AUC** on real-world content. Design metrics that reward CORRECT USE of inconclusive, not just answer accuracy.

---

## Recommended claim object

```json
{
  "claim": "player_A_kicked_ball",
  "type": "inferred",
  "epistemic_status": "temporal_correlation",
  "time_range": [12.43, 12.71],
  "horizon_seconds": null,
  "direct_evidence": ["player_A", "ball", "leg_motion"],
  "indirect_evidence": ["ball_velocity_change", "pose_transition"],
  "contradictions": [],
  "confidence": {
    "perception": 0.96,
    "temporal": 0.91,
    "motion": 0.94,
    "reasoning": 0.84,
    "calibrated": 0.89,
    "calibration_method": "split_conformal_alpha_01",
    "prediction_set_size": 2
  },
  "inconclusive_justification": null,
  "claim_staleness_ms": 1420
}
```

**Required additions for 2026 realism:**
- `epistemic_status`: one of [observed_only, temporal_correlation, rule_supported, physical_discontinuity, causal_intervention] — distinguishes correlation from actually-supported causation.
- `horizon_seconds`: null for observations/inferences; how far into the future for predictions.
- `calibration_method`: which post-hoc calibration was applied (temperature, isotonic, conformal, ensemble).
- `prediction_set_size`: conformal prediction returns a SET, not a point. 1 = fully confident, ≥2 = alternatives remain.
- `inconclusive_justification`: if type=inconclusive, a machine-readable enum of WHY (insufficient_data, modality_conflict, occlusion, detector_disagreement, outside_distribution).
- `claim_staleness_ms`: milliseconds since the deep-reasoning snapshot that generated this claim was taken. State updates every 33 ms, but deep reasoning may be 1000–3000 ms stale.

These numbers are illustrative schema examples, not measured results.
