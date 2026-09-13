# System principles

## Core engineering principles (10)

1. **Do not conflate visibility with truth.** A hidden fact can be inferred; an inferred fact is not equivalent to direct observation.
2. **Do not collapse hypotheses prematurely.** Maintain competing explanations until evidence separates them.
3. **Use time as evidence.** Before/after observations can constrain hidden events.
4. **Use structural priors.** Human biomechanics, object dynamics, scene topology and domain rules can constrain hidden states.
5. **Make uncertainty compositional.** A claim can have high object-detection confidence but low intent confidence.
6. **Exploit redundancy.** Trackers, motion models and temporal memory can carry information between expensive inference calls.
7. **Use disagreement as a signal.** Cross-model or cross-modal conflict should increase uncertainty or trigger deeper reasoning.
8. **Optimize for information, not uniform computation.** Spend compute when uncertainty or prediction error is high.
9. **Keep an evidence trail.** Every important claim should be traceable to source media and intermediate reasoning artifacts.
10. **Evaluate under degradation.** Occlusion, blur, compression, missing audio, camera cuts and synthetic transformations are part of the target environment.

---

## REALITY-BASED PRINCIPLES ADDED (2026, derived from benchmark data)

These are NOT negotiable design choices — they are constraints imposed by demonstrated limitations of current technology.

### R1. VLM cadence ≠ perception cadence; never force them to be the same.
- VLM inference takes 500 ms (best API case, p50) to 3500 ms (32B local model). Perception takes 2–20 ms.
- Engineering implication: Deep reasoning outputs are ALWAYS stale by 500–3000 ms relative to current state.
- Every VLM-derived claim MUST carry `claim_staleness_ms`. UI and API MUST surface staleness.
- Do NOT block the 30 FPS state update on VLM completion. If queues back up, coalesce and drop.

### R2. Detection and tracking are imperfect by default; assume 5–15% ID switches per scene for difficult footage.
- HOTA scores on real-world datasets are typically 45–75%, not 100%. Sports tracking SOTA: SAM 3-Deep-EIoU achieves **87.2 HOTA** on SportsMOT, McByte++ achieves **85.0 HOTA** on SoccerNet (training-free), SAMIDARE achieves **85.6 HOTA** on SportsMOT (CVPR 2026W).
- Track 2–5 ID hypotheses per entity in crowded scenes. Use appearance embeddings + trajectory continuity to disambiguate.
- Every entity ID should have an `identity_confidence` and `track_age_ms` exposed.

### R3. For occluded joints: output distributions, NEVER a single "best guess" point.
- MPJPE rises 20–40% on occluded vs. visible joints even with SOTA temporal models.
- Hide-and-disappear occlusions (hand going into pocket, behind back) are the hardest; expect MPJPE 2–3× visible.
- Required fields on every joint estimate: `visibility_score` [0,1], `posterior_covariance` (2×2 or 3×3), `estimation_method` (direct_observation / temporal_extrapolation / kinematic_propagation / diffusion_offline).

### R4. Synthetic media detection is a multi-evidence, uncertain judgment — never a binary flag.
- RA-Bench 2026 demonstrates that detectors collapse to 51–61% on real-world degraded crisis videos (17,886 videos, 19 detectors).
- Lab accuracy (92% on clean benchmarks) has ZERO predictive value for real-world accuracy. SOTA open-source detectors lose **45–50% AUC** on real-world content.
- NTIRE 2026 (CVPR 2026): Best detector DINO-MAC achieves 91.68 AUC under degraded conditions; MICV achieves 97.23% robust ROC-AUC on 42 generators.
- Required: minimum 3 independent evidence channels (e.g., pixel artifacts + temporal consistency + C2PA provenance) before emitting anything other than "inconclusive."
- C2PA / cryptographic provenance ALWAYS outweighs pixel-level detectors if the signature chain is valid. **152 C2PA-conformant products** (132 generators + 20 validators) as of Aug 2026, specification v2.4.
- Accept and design for: 60–70% of real-world social-media-compressed clips will correctly be tagged `inconclusive`.

### R5. Calibration is post-hoc in production; no retraining-based method survives distribution shift.
- Temperature scaling, isotonic regression, and conformal prediction are POST-HOC, model-agnostic, and require NO retraining. These are the deployment methods.
- **NEW (2026):** QaTS (2026) adapts temperature by confidence quantile with only 2 parameters, outperforming vanilla TS, IR, AdaTS. SMART (2025) uses logit gap with only 4 parameters, SOTA across 6 datasets.
- Any training-loss "calibration" (e.g., focal loss weighting, ECE in training loss) helps but is NOT SUFFICIENT — always apply post-hoc on a held-out calibration set from the target domain.
- Conformal prediction sets (not single classes) are the only way to get FINITE-SAMPLE coverage guarantees. Validated across **18 VLMs** on 6 multimodal datasets (EACL 2026). For high-stakes claims, return the prediction set with guaranteed α-level coverage, not a single label.
- **VL-Calibration (ACL 2026):** First to decouple visual + reasoning confidence in VLMs. ECE reduced from 0.421 → 0.098 on Qwen3-VL (76.7% reduction) while improving accuracy +2.3–3.0%.

### R6. Open-vocabulary recall has hard ceilings (~60–65% on LVIS for SOTA); design around recall gaps.
- No open-vocabulary detector in 2026 exceeds ~65 mAP on LVIS (which contains long-tail categories).
- Closed vocab (sports entities: person, ball, goal, referee): ~85–95% recall achievable.
- Strategy: Prefer closed-vocabulary detection for the known domain ontology. Use open-vocabulary VLMs ONLY as an async fallback for OOD entities, with high uncertainty tags.

### R7. Causal claims require explicit grounding; never infer causality from temporal order alone.
- "A then B happened" → `temporal_correlation`, NOT `causal_claim`.
- To upgrade to `causal_claim`, at least one must hold:
  (a) Domain rule explicitly encodes it (e.g., soccer Law 14: penalty kick → ball moves forward when kicked correctly),
  (b) Physical state discontinuity in both entities at the same timestamp, with physics model rejecting alternative explanations (e.g., ball velocity changes 2σ at t=12.57s AND player foot is within 5cm of ball at same timestamp),
  (c) Explicit intervention / A-B test data exists.
- Open-vocabulary "intent" or "reason" attribution (e.g., "the player kicked the ball to score") is by default `hypothesis`, never a supported claim.

### R8. Long-horizon predictions (>2 s for human motion, >5 s for ballistics) are NOT reliable.
- Kalman-filter ballistic prediction for balls: reliable ~200–500 ms into the future.
- Human pose extrapolation: reliable ~100–300 ms (walking/running); <100 ms for rapid direction changes.
- Strategic/tactical predictions (next player to pass): treat as low-confidence hypotheses ONLY, with huge prediction sets.
- Engineering implication: Prediction error as scheduler trigger is meaningful only for ≤500 ms horizons for entities.

### R9. Design for graceful degradation; every module failure mode has a cheaper fallback.
- Detector overloaded → use only tracker propagation for up to N frames, then trigger redetect when `track_confidence < threshold`.
- VLM queue depth > 20 → route high-priority events (scene cut, high-uncertainty) only; delay low-priority.
- All modalities missing except video → continue; tag all cross-modal terms as `missing` and inflate uncertainty appropriately.
- GPU OOM → fall back to smaller model variant (Nano vs Large) and tag `model_degraded: true`.

### R10. System metrics (FPS, latency, staleness, queue depth) are FIRST-CLASS evaluation metrics, not afterthoughts.
- A system that answers 1% more accurately but drops to 8 FPS has FAILED the 30 FPS target.
- Report: p50, p95, p99 for end-to-end latency; per-component latency breakdown; queue depth over time; % of frames where deep reasoning is >3 s stale; detection recall vs. processing load (under backoff).
- Every ablation in the ablation plan records BOTH quality metrics AND compute/latency/staleness metrics.
