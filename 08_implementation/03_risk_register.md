# Risk register

## R1 — False confidence
Mitigation: calibration (VL-Calibration ACL 2026: ECE 0.421→0.098 on Qwen3-VL-4B, 0.401→0.071 on 8B), decomposed confidence, abstention (CAP ACML 2025: 82.9% ECE reduction vs APS, +22.2% hallucination detection AUROC).

## R2 — Temporal hallucination
Mitigation: evidence-grounded temporal memory, state verification. VidHalluc (CVPR 2025) shows VLMs hallucinate temporal order 15–30% of the time. Use tracker state as ground-of-truth. CLASH (CVPR 2026 Findings): best models achieve only 69–75% contradiction detection.

## R3 — Compounding state error
Mitigation: periodic re-detection, uncertainty growth, correction events. RAM (CVPR 2026) reduces ID switches from 349 to 15 via adaptive Kalman filtering. MoRo (3DV 2026) achieves only 28% MPJPE gap between visible (37.83mm) and occluded (48.53mm) joints.

## R4 — Scheduler starvation
Mitigation: bounded queues, priorities, coalescing, deadlines. Neuron Systems (FIFA WC 2026): 42ms glass-to-glass latency across 104 matches, 5.14M events on Final day, zero swaps.

## R5 — Domain overfitting
Mitigation: separate domain priors from generic state machinery. DeepSport (Nov 2025) generalizes across 12 sports with 7B params, beating GPT-5 (35.70) and Qwen3-VL-235B (35.36) with 37.67 overall. SportR (2026): image→video transfer achieves 59.52% on Infraction (from 25.49%).

## R6 — Synthetic-media detector brittleness
Mitigation: ensembles (TRIDENT CVPR 2026: 0.860 AUC), provenance (152 C2PA-conformant products, spec v2.4), robustness tests, inconclusive output. RA-Bench: 26 of 63 detector–source pairs below 50% AUC (worse than random). Fine-tuned MLLMs collapse from 46.0% to 1.4% under social dissemination. Adversarial attacks achieve 83–97% misclassification on deepfake detectors.

## R7 — Hidden-state overinterpretation
Mitigation: probability distributions and explicit epistemic status. MoRo (3DV 2026): 28% MPJPE gap visible vs occluded. HiPART (CVPR 2025): 42.0mm MPJPE on H36M at 396 FPS.

## R8 — Latency collapse at high load
Mitigation: profile queue depth and p99 latency; degrade gracefully rather than blocking 30 FPS. Sports-LiteDet (2026): 68 FPS on Jetson Orin Nano with 5.8 MB model. SAM 3.1: 32 FPS at medium object counts on H100.

## R9 — Adversarial attacks on VLMs
Mitigation: Qwen2.5-VL shows significantly better resilience (6.5–15.5% ASR) vs LLaVA (52.6–66.9% ASR) under white-box attacks. Use drift-gated test-time defense (Beyond False Stability, Jun 2026). Certified YOLO robustness >98% under specific threat models.

## R10 — VLM robustness under distribution shift
Mitigation: VLM-RobustBench (Mar 2026): geometric distortions cause up to 34pp accuracy drops. Low-severity perturbations often degrade more than visually severe ones. Use VLM-RobustBench's 49 augmentation types for stress testing. STREAM-OOD (CVPR 2026W): reduces OOD false alarms from 3.4→1.6/hr.

## R11 — Production failure rate
Mitigation: 73% multimodal RAG production failure rate documented (2026). 362 AI incidents in 2025 (55% YoY increase, Stanford HAI). 36+ novel VLM failure modes identified. Use SABRE (Aug 2026) for reusable stress testing, MUSE (Mar 2026) for multi-turn safety evaluation.

## R12 — Bias and fairness
Mitigation: VIGNETTE (ACL 2026): VLMs reinforce complex contradictory biases. FairLens (Sep 2026): unwarranted inference on 99% of unanswerable questions. VLM Reality Check (CVPR 2026W): counterfactual accuracy drops 12–38%. Use FOCUS (ACL 2026) face-only counterfactuals for bias auditing.

## R13 — EU AI Act compliance
Mitigation: Article 50 transparency obligations in force 2 Aug 2026. Deepfake disclosure required at beginning, after interruptions, at regular intervals. Penalties up to €15M or 3% turnover. C2PA Content Credentials as compliance mechanism. High-risk biometric obligations deferred to Dec 2027.
