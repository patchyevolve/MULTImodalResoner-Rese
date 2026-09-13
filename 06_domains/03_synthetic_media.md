# Artificial/synthetic media

## States
- authentic supported
- synthetic/manipulated supported
- inconclusive

## Evidence
- spatial artifacts
- temporal consistency
- facial/pose consistency
- physical plausibility
- audio-video synchronization
- metadata/provenance
- watermark/signature evidence
- compression history
- detector ensembles

## Critical requirement
Use multiple independent forensic signals and explicitly model detector blind spots and distribution shift.

---

## REALITY CHECK 2026: SYNTHETIC MEDIA DETECTION STATE

### RA-Bench (Aug 2026, arXiv:2608.14391):
- 17,886 videos, 9 generators (4 open, 5 closed). **No detector generalizes across all generators.**

| Detector | Public Ref AUC | RA-Bench Mean (Open) | RA-Bench Mean (Closed) |
|---|---|---|---|
| CNNSpot | 81.4 | 39.2 | 31.9 |
| ForgeLens | 92.9 | 64.9 | 59.5 |
| DeCoF | 81.5 | 63.4 | 55.2 |
| **7-detector mean** | **84.2** | **54.6** | **49.1** |

- 26 of 63 detector-source pairs have AUC below 50% (worse than random).
- At 5% FPR, 7-detector mean identifies only 6.0-7.5% of generated videos.
- HumanProof subset (fooling all 5 reviewers): detectors average 47.5% AUC.
- Social dissemination: fine-tuned MLLMs drop from 46.0% → 1.4% FakeR after re-encoding chain.

### XceptionNet Cross-Dataset Collapse:
- Within-distribution: 99.26% accuracy (FF++).
- Cross-dataset: 51.31% on DFDC (fake recall: 13.16%).
- Transformer architectures show 11.33% less decline than CNN's >15% in cross-dataset.

### Audio Deepfake Detection (2026):
- Resemble AI: 98.1% accuracy, 0.33 RTF (Podonos benchmark).
- Teffic-Audio: 1.454% pooled EER across 14 test sets (#1 Speech-DF-Arena).
- FlowFake (34K params): 75.29% cross-domain accuracy, 2s per 512 clips.
- **Cascaded distortion effect:** AASIST clean EER 0.83% → after full AFE pipeline: 47.11%.

### Provenance & Watermarking:
- **C2PA 2.4 (Apr 2026):** Live video support, HTML embedding, Merkle trees for chunks.
- **C2PA security:** IACR analysis found 7 serious problems — validators accept compromised certs.
- **C2PA adoption:** 4,000+ CAI members, 500+ C2PA members. But only 22% natively recognized on platforms.
- **Real-world test (520 assets, Jun 2026):** 60% preserved C2PA on upload, only 3 of 9 platforms consistent.
- **Google SynthID:** 100B+ images watermarked, 60,000 years of audio, 50M verification checks.
  - Survives typical re-encoding, screenshots, cropping up to ~20%.
  - Fragmentation: Google's detector doesn't recognize OpenAI's watermark and vice versa.

### Ensemble Design Principles (2026 evidence):
1. **Diversity > count:** Peak at 6 models; 7th decreases AUC by -0.0088. Correlation ρ ≈ 0.82 threshold.
2. **Logit-space fusion > probability averaging:** Consistently +0.001-0.010 AUC improvement.
3. **Degradation curriculum dominates:** +0.062 AUC from augmentation alone (more than all architectural choices).
4. **Multi-modal decomposition:** Semantic (62.4%) + Structural (28.5%) + Spectral (9.1%) weights.
5. **Adaptive weighting:** CLIP-conditioned dynamic weights outperform fixed weights under degradation.

### TRIDENT (CVPR 2026 NTIRE, 6th place):
- Tri-modal: CLIP/SigLIP (semantic) + EVA-02 (structural) + SRM/Bayar (spectral).
- Logit-space fusion with rank-based normalization. Total gain: +0.158 AUC.

### LOGER (CVPR 2026 NTIRE, 2nd place):
- Local-Global: heterogeneous VFMs (SigLIP, EVA-02, MetaCLIP) + DINOv3-Large with MIL top-10%.
- Logit averaging: 0.8901 vs probability averaging 0.8887 vs voting 0.8812.

### Critical Requirements:
1. **Do not rely on single detectors** — ensemble 4-6 diverse models.
2. **Plan for social dissemination** — test under compression, resizing, frame rate reduction.
3. **Treat automated detection as triage, not adjudication** (RA-Bench's central finding).
4. **Implement both SynthID (detection) and C2PA (provenance)** — complementary systems.
5. **Expect C2PA credential stripping** on most social platforms — design for graceful degradation.
