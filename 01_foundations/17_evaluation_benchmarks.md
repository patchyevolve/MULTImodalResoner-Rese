# 17. Evaluation and benchmarks

## Objective
Build evaluation that measures reasoning rather than only caption/VQA accuracy.

## Metric families
### Perception
Detection, tracking, pose, segmentation.

### Hidden-state inference
Hidden-pose error, trajectory error, uncertainty coverage.

### Temporal reasoning
Event localization, ordering, anticipation, temporal consistency.

### Hypothesis reasoning
Precision, recall, alternative coverage, contradiction detection.

### Calibration
ECE, Brier score, risk-coverage, selective accuracy, conformal coverage.

### Streaming
FPS, latency, jitter, queue depth, freshness, online accuracy.

## Required stress tests
- occlusion
- motion blur
- camera cuts
- compression
- missing audio
- conflicting modalities
- adversarial synthetic media
- distribution shift

## Deliverable
A benchmark suite combining natural videos, sports, synthetic media and controlled occlusion experiments.

---

## REALITY CHECK 2026 (concrete metrics)

### ✅ STANDARD, PROVEN METRICS (use off-the-shelf implementations):
| Category | Metrics | Reality 2026 Baseline Numbers (for calibration) |
|---|---|---|
| **Detection** | mAP50, mAP50:95, per-class AP | RF-DETR-2XL = **60.1 mAP50:95** (COCO, ICLR 2026). DETRPose-X = **73.3 AP** (pose). Aim to match within 2% after integration overhead. |
| **Tracking** | HOTA, IDF1, MOTA, track ID switch count | SAM 3-Deep-EIoU: **87.2 HOTA** on SportsMOT (new SOTA). McByte++: **85.0 HOTA** on SoccerNet. HOTA < 50 = integration failed. |
| **Pose** | PCK@0.1, MPJPE, MPJPE_occluded | HiPART: **42.0mm MPJPE** at **396 FPS** (H36M). MoRo: **37.83mm** visible / **48.53mm** occluded. RAM: **53.0mm** on 3DPW. |
| **Segmentation** | IoU, J&F (DAVIS) | RF-DETR-Seg-2XL = **49.9 AP**. SAM 2.1 Hiera-L = **79.5 J&F** (SA-V test). SAM 3: **92.0 J&F** (DAVIS 2017). |
| **Temporal reasoning** | Event localization mAP, ordering accuracy, anticipation recall@k | SoccerNet 2026 anticipation: best avg mAP = **24.08** (FAANTRA-WS). Video-MME-v2: Gemini-3-Pro **49.4** non-linear score vs human **90.7**. |
| **Streaming** | FPS (sustained, not peak), E2E latency p50/p95/p99, jitter, queue depth over time, staleness | NVIDIA RT-VLM: 51 concurrent streams at **94.8% GPU utilization**, E2E avg **3.6s** alerting. 30 FPS sustained p99 is SUCCESS. |

### ⚠️ METRICS THAT NEED CUSTOM BENCHMARK DESIGN (the actual research):
These are unique to this system and don't have off-the-shelf benchmarks:
| Category | Metrics | How to Measure | 2026 Realistic Success Bar |
|---|---|---|---|
| **Hidden-state inference** | Hidden-pose MPJPE, trajectory error (ATE), **uncertainty coverage probability** (does the 95% predicted ellipse actually contain GT 95% of the time?), NLL | Controlled occlusion dataset: hide hands/feet/ball at known timestamps, measure distribution coverage. | Coverage ≥ 90% of stated coverage level (e.g., 95% nominal ellipse covers ≥ 90% GT). |
| **Hypothesis reasoning** | Precision@1, recall@3, alternative coverage (% of time true hypothesis is in top-3 set), contradiction detection precision/recall | Ambiguous clips with 2–3 annotated plausible explanations. | Recall@3 ≥ 85% i.e., correct hypothesis survives in top-3 on ambiguous clips. |
| **Calibration** | ECE (per claim type!), Brier score, risk-coverage AUC, selective accuracy (abstain correct %), conformal coverage validation | Reliability diagrams per each claim type: observed, inferred, hypothesis, causal. | Final claim ECE ≤ 5%. Conformal α=0.05 → empirical error ≤ 6% on fresh test. |
| **Staleness** | Answer freshness = time(state_snapshot) - time(claim_published). Median, p95. | Tag every claim. | Median staleness ≤ 1500 ms for VLM-derived claims; ≤ 50 ms for perception-derived. |

### REQUIRED STRESS TESTS (every benchmark suite must include):
1. **Occlusion:** Vary occlusion duration (0.2 s → 2 s) and occluded fraction (0% → 100% of entity area).
2. **Motion blur:** Vary blur severity, expect detection mAP to drop gracefully.
3. **Camera cuts:** Hard cuts every 1 s → 5 s. Measure ID switch rate, event continuity.
4. **Compression:** H.264 CRF 23 → 40 + social media re-encoding (X/Telegram/WhatsApp style). Synthetic detectors should gracefully go inconclusive.
5. **Missing audio / missing modalities:** Drop audio channel entirely. Cross-modal claims should correctly go to inconclusive or audio_weight=0.
6. **Conflicting modalities:** Inject known OCR errors or audio misclassifications. Check fusion downgrades confidence appropriately.
7. **Adversarial synthetic media:** Sora + Runway + Flux outputs → recompress → crop → resize. Detector ensemble should output INCONCLUSIVE most of the time, not confident-wrong.
8. **Distribution shift:** Train on soccer daytime → test on basketball nighttime. Test generalization of perception + calibration (not just accuracy).

### EXISTING DATASETS TO BUILD ON TOP OF:
- **General:** MS COCO, LVIS, Kinetics, Something-Something v2.
- **Sports:** SoccerNet (v2/v3), FineSports, SportsMOT, Towards Universal Soccer Video Understanding (CVPR 2025), SoccerReplay-1988, SoccerNet 2026 Challenge (5 tasks, 427 teams).
- **3D Pose:** 3DPW, Human3.6M, Fit3D (extreme fitness motions with occlusions), PoseTrack18/21.
- **Occluded hands:** ARCTIC, HOT3D, EPIC-Contact, HOI4D.
- **Synthetic:** RA-Bench (17,886 videos), FVBench (120K+ videos, 42 generators), GenVidBench (6.78M videos), NTIRE 2026 (42 generators), FaceForensics++, C2PA test corpus, DeepfakeImpact (33 methods, 12 datasets).
- **Temporal hallucination:** VidHalluc benchmark (CVPR 2025).
- **Multi-sport VQA:** DeepSport (12 sports), SportR (6,841 CoT annotations).
- **NEW 2026 benchmarks:** Video-MME-v2 (non-linear scoring, 49.4 vs 90.7 human), OmniVCHall (823 videos, 9,027 QA, 8 hallucination types), SEED-Bench-R1 (perception+reasoning generalization), LongVideo-Reason (52K long video QA), CausalVQA (1,786 items, best model 61.66% vs human 84.78%).
- **Video QA SOTA:** Video-MME: Qwen3.6-27B **87.7%**. MVBench: Qwen3.5-Omni-Plus **79.0%**. EgoSchema: Qwen2-VL-72B **77.9%**.

### ❌ BAD / UNINFORMATIVE EVALUATION PRACTICES (SET ASIDE):
- Single number "VQA accuracy" without breakdown by claim type.
- No p99 latency numbers (only p50).
- Evaluating on clean benchmarks only, no compression/occlusion.
- Evaluating synthetic detection on in-domain (same generator) test only.
- Reporting only headline numbers without per-category breakdowns.

### DELIVERABLE (2026 practical):
- Benchmark suite with 10 experiments (E1-E10 per experiment matrix file) + 8 stress tests implemented.
- Full metric per experiment with per-category breakdowns.
- Every result reports BOTH quality metrics AND compute/latency/staleness metrics for the ablation baseline configs.
- Public reproducible eval scripts + fixed hardware target reference (e.g., RTX 4090, CUDA 12.x, TensorRT 10.x).
