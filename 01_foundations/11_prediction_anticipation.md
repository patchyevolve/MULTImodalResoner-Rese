# 11. Prediction and anticipation

## Objective
Predict likely future states and events from the current belief state.

## Research questions
- Next-state vs next-event prediction?
- How far ahead can the system predict usefully?
- How should multiple plausible futures be represented?
- Can prediction be used to detect anomalies before events occur?
- How should forecast confidence be calibrated?

## Formal target
P(S_t+k, E_t+k | S_1:t, E_1:t)

## Deliverable
Multi-hypothesis future predictor with calibration and anticipation metrics.

---

## REALITY CHECK 2026

### ✅ PROVEN, deploy today:
- **Next-state (1-step) prediction for tracked objects:** Kalman filter / constant-acceleration / IMM. Ballistics (balls) ~95% accurate to 0.5 s. Human pose linear extrapolation ~85% joint consistency to 0.2 s.
- **Multi-hypothesis representation:** Gaussian mixture (2–5 modes) for trajectories. Standard in self-driving stacks for years.
- **Anomaly detection via prediction error:** State residual vs motion model threshold → trigger expensive reasoning. This works; it's exactly the scheduler trigger signal we need.

### ⚠️ PLAUSIBLE, lower your expectations:
- **Next-event prediction (not next-state):** Top-3 accuracy ~50–70% on closed sports datasets (SoccerNet) at 0.5–2 s horizons. Treat as hypothesis, not fact. Return prediction SET (top-K) conformal. SoccerNet 2026 Challenge: best action anticipation achieves **24.08% avg mAP** at 5s horizon (vs 16.76% baseline). At 2s: **18.18%** (vs 13.00% baseline).
- **How far ahead can the system predict usefully?**
  - **≤0.2 s (≤6 frames):** Very reliable. Kalman/linear. Use for scheduler trigger.
  - **0.2–1 s:** Moderate reliability. Multi-modal distributions (≥2 hypotheses).
  - **1–3 s:** Only for rigid ballistics + very coarse human motion. Distribution is WIDE (3–5 hypotheses), confidence ≤50% for most events.
  - **>3 s for human actions:** No current model is reliable. Strategic predictions are hypotheses with huge uncertainty.
- **Anomaly detection BEFORE event occurs:** Works only for very simple anomalies (ball suddenly changing direction with no contact, player accelerating 3σ beyond human max). Detection rate ~30–60% depending on anomaly type; lots of false positives.

### ❌ SPECULATIVE / DO NOT PROMISE:
- **Single-point "what will happen" prediction at >1 s for humans.** Even SOTA models are <40% top-1 accuracy for next soccer pass recipient at 2 s horizon.
- Prediction used as direct evidence for hypothesis. Prediction error is a scheduler TRIGGER, not a claim support.
- "Useful" prediction at >5 s for open-world scenes.

### FORMAL TARGET (realistic 2026 version):
P(S_t+k, E_t+k | S_1:t, E_1:t) represented as:
- For each entity: Gaussian mixture (2–5 components) OR conformal prediction region (90% coverage).
- For each event: Conformal prediction set (top-K hypotheses with coverage guarantee).
- Confidence decays monotonically with k (time horizon).

### PREDICTION CALIBRATION:
Each forecast horizon k needs SEPARATE calibration transform.
- ECE / coverage guaranteed PER-HORIZON, not a global number.
- A prediction at k=30 frames cannot use the same calibration as k=3 frames.

### DELIVERABLE (practical 2026):
- Multi-hypothesis (GMM + conformal sets) future predictor.
- Per-horizon calibrated uncertainty.
- Scheduler trigger based on prediction error with tunable thresholds.
- Do NOT attempt single-point forecasts for the UI. Always present the prediction as range/region with confidence.
