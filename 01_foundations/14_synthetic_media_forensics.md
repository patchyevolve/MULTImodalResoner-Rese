# 14. Synthetic/artificial media reasoning

## Objective
Determine whether media is authentic, synthetic, manipulated, or inconclusive using multiple independent evidence channels.

## Evidence families
- Pixel artifacts
- Frequency-domain artifacts
- Facial/biometric inconsistencies
- Geometry
- Lighting/shadow/reflection consistency
- Temporal consistency
- Audio-video synchronization
- Compression/editing traces
- Metadata/provenance
- Watermarks/signatures
- Generator-specific forensic models

## Critical distinction
Not detecting synthesis is not proof of authenticity.

## Deliverable
Evidence-based authenticity reasoner with an inconclusive state.

---

## REALITY CHECK 2026 (THE MOST IMPORTANT REALITY IN THIS ENTIRE RESEARCH)

### ⚠️ RA-Bench 2026 HARD TRUTHS (17,886 real-world crisis videos, 19 detectors tested):
| Detector Family | Public-Reference AUC | RA-Bench Mean AUC (Open) | RA-Bench Mean AUC (Closed) |
|---|---|---|---|
| Traditional (7-detector mean) | **84.2%** | **57.3%** | **43.9%** |
| CNNSpot | 81.4% | 45.8% | 39.2% |
| UnivFD | 89.9% | 39.9% | 38.4% |
| ForgeLens | 92.9% | 63.0% | 59.5% |
| ReStraV | 98.6% | 68.3% | 45.8% |
| Zero-shot VLMs | ~78% | **~51%** | — |
| Fine-tuned MLLMs | ~88% | **~61%** (clean) → **1.4%** (social dissemination) | — |

- 26 of 63 detector–source pairs scored **below 50% AUC** (worse than random).
- Spearman correlation between public rankings and RA-Bench rankings: only **0.26** (rankings completely invert).
- Fine-tuned MLLMs under full social dissemination (LastMile): FakeR collapses from 46.0% to **1.4%** (97% relative drop).

### ✅ PROVEN / MUST-DO:
- **Three-state output, never binary:** `authentic_supported`, `synthetic_supported`, `inconclusive`. The third state is NOT failure — it's the correct output for ~60–70% of real-world compressed social media videos.
- **Multi-evidence corroboration, NEVER single detector:** TRIDENT (CVPR 2026): tri-modal ensemble (semantic + structural + spectral) achieves **0.860 AUC** vs 0.838 single best (+0.022 gain). Performance plateaus beyond diversity-governed threshold.
- **C2PA provenance is KING:** **152 conformant products** (132 generators + 20 validators) as of Aug 2026. Specification v2.4. EU AI Act Article 50 marking obligations applied from Aug 2, 2026.
- **Watermark detection (SynthID, etc.):** Over **10 billion images and video frames** watermarked as of May 2026. SynthID-O: **100% TPR** at 0.1% FPR on identity, **99.72%** worst-case aggregated. Survived **300 JPEG compression cycles**. Now integrated with OpenAI, Nvidia, ElevenLabs. Google is ~3× more robust than OpenAI for removal resistance.

### ⚠️ PLAUSIBLE, treat as additional weak signals:
- **Pixel/frequency artifacts** — Weak once compressed.
- **Temporal consistency (flicker, warping, jitter)** — Better for Sora/Runway-style video; weak for low-quality real footage.
- **Facial/biometric inconsistencies** — Good if face is high-res; fails on small/blurry faces.
- **Geometry/perspective anomalies** — Strong when present; rare as sole signal.
- **Lighting/shadow/reflection consistency** — Moderate; needs accurate scene geometry.
- **Audio-video sync (lip sync, event timing)** — Good for talking-head deepfakes; weak for general action.
- **Compression/editing traces (GOP anomaly, frame timestamp)** — Good if metadata preserved; destroyed by re-upload.
- **Ensemble agreement across ≥3 detectors** — Abductive corroboration 2026: multi-detector agree positive with provenance = low false positive.

### ❌ ABSOLUTELY SET ASIDE (they are dangerous claims):
- **"99% accurate deepfake detector" on real-world video:** Vendor marketing numbers from internal clean benchmarks have ZERO predictive value outside their benchmark. RA-Bench 2026 proves every headline number collapses under real compression/distribution shift.
- **"Absence of synthetic evidence = proof of authenticity":** EXPLICITLY FORBIDDEN. You cannot prove authenticity from pixel absence; you can only support it via provenance + multi-signal corroboration. If no evidence either way → `inconclusive`.
- **Single-pipeline detector as ground truth for any high-stakes decision.** Any forensic system that matters uses ≥3 independent channels plus provenance.
- **Generalization to next-generation generators unseen during training.** Detectors lag new models by 2–6 months. Always allow `inconclusive` for new-model uncertainty.

### OPERATIONAL REQUIREMENTS (non-negotiable 2026):
1. Minimum 3 independent evidence channels agreeing before moving from `inconclusive` to either supported state.
2. Channel weight hierarchy:
   - `cryptographic_provenance_valid` (C2PA/Truepic) >> everything.
   - `generator_specific_watermark` (SynthID, etc.) > general forensics.
   - `ensemble_of_3_detectors_agree` > single detector.
   - `pixel/freq artifacts` = weakest.
3. Newly released generator (e.g., "new Midjourney v8 dropped yesterday") → default all detections to `inconclusive` for that generator class until calibration set is collected and detector ensemble updated.
4. Design evaluation metrics that REWARD correct use of `inconclusive`. Penalize incorrect claims MORE heavily than correctly saying "I don't know."
5. For every detector, track per-generator-family performance on an ongoing basis. Detectors degrade over time as generators improve; continuous monitoring is mandatory.

### DELIVERABLE (2026 REALITY-BASED):
Evidence-based authenticity reasoner with:
- 3 explicit states (authentic_supported / synthetic_supported / inconclusive)
- 12 evidence channels integrated with calibrated weight hierarchy
- Multi-detector abductive corroboration logic
- C2PA provenance as highest-weight evidence
- Automatic inconclusive fallback for out-of-distribution detectors / new generators
- Benchmarked under social media compression + crop + resize + re-encoding, not lab-clean data.
