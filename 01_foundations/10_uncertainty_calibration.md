# 10. Uncertainty and calibration

## Objective
Make confidence meaningful and decomposable.

## Distinguish
- Observation confidence
- Grounding confidence
- State-estimation uncertainty
- Temporal consistency
- Hypothesis uncertainty
- Causal uncertainty
- Model uncertainty
- Input-quality uncertainty

## Research methods
- Temperature scaling
- Isotonic regression
- Platt scaling
- Deep ensembles
- Evidential approaches
- Conformal prediction
- Selective prediction
- Risk-coverage analysis

## Key requirement
Do not report one scalar where several materially different uncertainties exist.

## Deliverable
A calibrated confidence model and confidence schema.

---

## REALITY CHECK 2026

### ✅ PROVEN methods, deploy today:
- **Split conformal prediction (SCP):** Model-agnostic, post-hoc, FINITE-SAMPLE coverage guarantees. Validated across **18 VLMs** on 6 multimodal datasets (EACL 2026): strictly controls error rate at user-specified α. Requires NO retraining. All empirical error rates remain strictly below α across all tested α values (0.1–0.9). Average empirical error at α=0.2: **0.1934** (ScienceQA), **0.1763** (MMMU) — all below guarantee.
- **Isotonic regression:** Stronger calibration than temperature scaling when enough calibration data; monotonic, well-understood. NA-FIR and SCIR (ICML 2025) improve multi-class isotonic for better ECE.
- **Temperature scaling:** Simplest, works for classification-style outputs, requires <100 calibration samples. VLM average ECE drops to **≈0.05** after temperature scaling across 35 VLMs. ImageNet models rise to **≈0.15** (worse than VLMs after TS).
- **Ensemble disagreement:** Cheap if you already run 2+ variants (e.g., RF-DETR-S vs YOLO26-S); disagreement = proxy for epistemic uncertainty.
- **VL-Calibration (ACL 2026):** First to decouple visual + reasoning confidence in VLMs. Qwen3-VL-4B: ECE **0.421→0.098** (76.7% reduction), accuracy **0.704→0.727** (+2.3%). Qwen3-VL-8B: ECE **0.401→0.071**, accuracy **0.731→0.761** (+3.0%). Qwen3-VL-30B: ECE **0.388→0.082**. A-OKVQA: ECE **0.112→0.017**.

### ⚠️ PLAUSIBLE, implement carefully:
- **Decomposed confidence (8 components stored):** Design is valid, but each component must have its OWN calibration transform, and then the composition must be re-calibrated. This requires more calibration data but yields MUCH more useful info than a single scalar.
- **Conformal prediction sets for claims (not just classification):** Return a SET of claims (or hypotheses) with guaranteed coverage α, not just single best claim. This is the state of the art for high-stakes in 2026.
- **Evidential approaches (Dirichlet networks):** Work well in research but often underperform simple post-hoc conformal in real deployment; keep as research track, not production default.

### ❌ SPECULATIVE / UNRELIABLE in 2026:
- **Single ECE / Brier score as "good enough":** Mathematical reasoning tasks have 2–3× worse calibration than recognition tasks across all VLMs. Per-task calibration transforms are required.
- **Trusting a VLM's own verbalized confidence ("I am 90% sure"):** Research 2025 EMNLP shows verbalized confidence is poorly calibrated; ECE 15–25% on average, biased toward overconfidence on hard examples. Always use independent calibration transforms, never LM self-reported confidence alone.
- **Deep ensembles of 72B VLMs for uncertainty:** Financially impossible ($$$). Use cheaper ensembles of small models for disagreement proxy + post-hoc conformal on the final output.

### 🔬 GAPS IDENTIFIED (2026, needs research):
1. **Conformal prediction for video temporal reasoning chains:** No 2026 work extends conformal coverage guarantees to video VLM reasoning. ConfLVLM (2025) and CAP (ACML 2025) are image-only. Confirmed gap.
2. **Temporal uncertainty propagation:** No 2026 work propagates uncertainty through event detection → temporal ordering → causal inference chains in video VLMs. Confirmed gap.
3. **Multi-model ensemble calibration (detector + VLM + pose):** No 2026 paper calibrates the combined output of a perception pipeline. Cocoon (ICLR 2025) covers camera+LiDAR but not detector+VLM+pose.
4. **Class-conditional conformal coverage:** Sharma & Dutta (Aug 2026) show split-conformal on zero-shot VLMs provides marginal coverage (~0.86) but worst-class coverage collapses to ≈0 under distribution shift.
5. **Abstention at 90% coverage:** Current SOTA VLMs abstain on 60–96% of queries at 90% coverage (vanilla). InstructBLIP with ReCoVERR reaches 65% at 20% risk tolerance. CALM-VLM (2026) integrates temperature scaling + selective prediction but lacks published numbers.
6. **CDF calibration for regression outputs:** CCNet (ICML 2024) for pose calibration, isotonic regression on residual CDFs for bounding boxes — mature for individual modules but no end-to-end pipeline calibration.
7. **Open-set uncertainty for VLMs:** UNI-OOD (CVPR 2026) achieves SOTA on object/image-level OOD, but VLMs are NOT inherently open-set — finite query set creates closed-set assumptions (ECCV 2024).

### DECOMPOSED CONFIDENCE: PER-COMPONENT CALIBRATION 2026:
Store 8 uncertainties. Each has raw → calibrated via independent transform:
| Component | Raw source | Calibration method | 2026 typical residual ECE after calibration |
|---|---|---|---|
| Observation quality | Blur detector, compressed ratio estimate | Isotonic | 1–3% |
| Perception confidence | Detector score, track smoothness | Temperature scaling | 2–5% |
| State-estimation uncertainty | Kalman covariance, occlusion frac | Linear calibrator | 3–7% |
| Temporal consistency | State residual vs motion model | Histogram binning | 3–6% |
| Cross-modal agreement | Modality KL divergence | Isotonic | 4–8% |
| Hypothesis support/contradiction | LLR weights (from ev. graph) | Beta calibrator | 4–9% |
| Reasoning confidence | VLM logit entropy or verifier agree | Conformal prediction sets | 5–10% |
| Final calibrated claim | Composition of above | Split conformal with held-out | 2–5% (goal; VL-Calibration achieves 0.098 ECE) |

**NEW CALIBRATION METHODS (2025-2026):**
- QaTS (2026): 2-parameter temperature scaling adapting by confidence quantile. CIFAR-10 ResNet-50: ECE **1.38→0.85** (38% reduction). CIFAR-100 ResNet-50: ECE **5.61→1.37** (76% reduction). ImageNet-1K ResNet-50: ECE **2.17→0.61** (72% reduction). Outperforms vanilla TS, IR, AdaTS across all settings.
- SMART (2025): 4-parameter method using logit gap. Same benchmarks as QaTS with matching performance. SOTA across 6 datasets.
- CAP (ACML 2025): Conformalized abstention policies. 82.9% ECE reduction vs APS, 74.8% vs LAC. +22.2% hallucination detection AUROC, +21.2% selective generation AUARC.
- BCEA (2026): Budgeted conformal evidence acquisition. Restores finite-sample guarantee while improving coverage. At α=0.05: achieves cov=0.22/risk=0.06 (maintains guarantee).
- CDRL (CVPR 2026): Confidence-driven RL for test-time scaling.

### REQUIRED TESTS, 2026 CONCRETE:
1. Reliability diagrams per claim type (observed, inferred, hypothesis, causal).
2. ECE and Brier score — per-domain AND per-task.
3. Risk-coverage curves at α ∈ [0.01, 0.05, 0.10].
4. Selective accuracy: % of claims correctly "unknown" at fixed coverage.
5. Confidence slope under occlusion: occluded_frac from 0 → 100% → expect confidence monotonically decreasing. If not, calibration is wrong.
6. Confidence under modality failure: remove audio → expect audio-dependent claims' confidence to drop proportionally.
7. Confidence under synthetic manipulation: synthetic confidence for authentic sample should RISE only if provenance + multiple detectors agree; otherwise inconclusive.
8. **Conformal coverage validation:** On a fresh test set, empirical error rate ≤ α + 1/sqrt(n). This is a mathematical requirement, not a suggestion.
