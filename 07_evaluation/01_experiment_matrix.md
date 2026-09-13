# Experiment matrix

## E1 — Direct observation
Measure basic perception and grounding. RF-DETR-2XL: **60.1 mAP** COCO. DETRPose-X: **73.3 AP**. SAM 3-Deep-EIoU: **87.2 HOTA** SportsMOT. Target: within 2% of these baselines after integration.

## E2 — Occluded-state reconstruction
Hide hands/feet/objects and evaluate predicted state distributions. MoRo: **37.83mm** visible / **48.53mm** occluded (28% gap). Target: maintain <40% degradation at 50% occlusion fraction.

## E3 — Temporal event inference
Remove key frames and test reconstruction of event identity and timing. SoccerNet 2026 anticipation: best **24.08% avg mAP** at 5s. Target: top-3 recall ≥85% at 2s horizon on closed domain.

## E4 — Hypothesis competition
Provide ambiguous clips with multiple plausible explanations. SPIKE-RL (ICLR 2026): **68.2%** surprise localization, **40.3%** hypothesis diversity. Target: recall@3 ≥85% for correct hypothesis in top-3.

## E5 — Contradictory modalities
Inject audio or OCR errors and test whether confidence falls appropriately. CLASH (CVPR 2026 Findings): best models achieve only **69–75%** contradiction detection. Target: audio-contradiction claims correctly downweighted to inconclusive.

## E6 — Synthetic-media robustness
Apply generation/manipulation plus recompress, crop and resize. RA-Bench 2026: 26 of 63 detector–source pairs below **50% AUC** (worse than random). Target: forensic ensemble outputs inconclusive (not confident-wrong) on >80% of attacked samples.

## E7 — Prediction-driven scheduling
Compare always-on deep reasoning against uncertainty/prediction-error-triggered reasoning. NVIDIA RT-VLM: **51 concurrent streams** at **94.8% GPU util** with adaptive scheduling. Target: event-triggered beats fixed-rate by ≥15% accuracy at same compute budget.

## E8 — 30 FPS stress test
Measure sustained frame rate, jitter, dropped frames and answer freshness. Critical path: tracking (0.5ms) + detect (2.5ms) + pose (7ms) + pre/post (3–5ms) = **~13–15ms/frame**. Target: 30 FPS sustained, p99 <33ms.

## E9 — Long-form memory
Test minute-scale to hour-scale retrieval and state continuity. FlexMem (CVPR 2026): **>1,000 frames** on single 3090. MemWeaver (ACL 2026): reduces input context by **95%** while improving multi-hop reasoning. Target: state continuity maintained across 30+ minute sessions.

## E10 — Distribution shift
Train/test across sports, cameras, lighting, compression and media genres. VLM-RobustBench (Mar 2026): geometric distortions cause up to **34pp accuracy drops**. Target: <10pp degradation across 3+ domain shifts.
