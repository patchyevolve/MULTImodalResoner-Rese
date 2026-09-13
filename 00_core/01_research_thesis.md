# Research thesis

## 1. Problem statement

Most current multimodal systems are optimized around a pattern similar to:

```text
media -> encoder -> VLM/LLM -> answer
```

This is insufficient for the intended system because many useful conclusions are not directly visible in one frame. The system must reason about hidden states, temporal continuity, physical constraints, causality, alternative explanations, and uncertainty.

## 2. Proposed abstraction

Model the world as a latent state S_t and the media as observations O_t:

P(S_t | O_1:t)

The system continuously updates a belief state rather than simply classifying each frame.

## 3. Example: hidden hand/foot state

A hand may be occluded, yet the system can estimate its probable state using:

- prior pose
- current pose
- velocity
- acceleration
- skeletal constraints
- object interactions
- scene geometry
- future visible state

The result should be a probability distribution or uncertainty region, not an invented exact pose.

## 4. Example: occluded sports event

Observed:

- foot approaching ball
- later ball velocity changes
- player pose changes

Possible hypotheses:

- kick/contact
- missed contact
- deflection by another object/person
- viewpoint artifact

The reasoner should rank these hypotheses and retain evidence supporting or contradicting them.

## 5. Core loop

```text
observe -> estimate state -> predict -> compare observation -> update belief -> reason when needed
```

This is closer to a streaming world model than a conventional video QA system.

---

## 6. REALITY CHECK (2026 Benchmark Grounding)

### ✅ PROVEN / ACHIEVABLE TODAY (no new research required):

**Perception at 30 FPS on single GPU:**
- RF-DETR-N: 48.4 mAP50:95 @ 2.3 ms / batch-1 (NVIDIA T4, TensorRT FP16)
- RF-DETR-S: 53.0 mAP @ 3.5 ms (T4); 53.3 mAP @ 4.0 ms (RTX 4090)
- RF-DETR-M: 54.7 mAP @ 4.4 ms
- RF-DETR-L: 56.5 mAP @ 6.8 ms
- RF-DETR-2XL: **60.1 mAP** @ 17.2 ms — first real-time model to exceed 60 mAP on COCO (ICLR 2026)
- RF-DETR Keypoint: **71.8 AP** @ 9.8 ms (576×576) — beats YOLO26x-pose (71.6 AP, 12.2 ms)
- YOLO26-N: 40.9 mAP @ 1.7 ms (2.4M params, NMS-free end-to-end)
- YOLO26-X: 57.5 mAP @ 11.8 ms
- DETRPose-S: **67.0 AP** @ 2.39 ms A10 — matches YOLO11-Pose-X (67.2 AP) with 81% fewer params
- DETRPose-X: **73.3 AP** @ 7.36 ms V100 — new SOTA for real-time pose
- RTMO-l: **74.8 AP** @ 19.1 ms V100 — one-stage whole-body SOTA
- RF-DETR-Seg-N: **40.3 AP** @ 3.4 ms — beats YOLOv11-Seg-X (40.1 AP) at 4× faster
- SAM 3: **30 ms** per image on H200 (100+ objects), 47.0 LVIS zero-shot AP
- SAM 3.1: **32 FPS** at medium object counts on H100 (2× throughput improvement)
- **Bottom line:** Detect + Track + Pose in ~10–15 ms TOTAL per frame is production-feasible on a T4/4090-class GPU. 30 FPS perception is NOT speculative.

**Tracking:**
- McByte++ (training-free): HOTA = **85.0** on SoccerNet-Tracking, MOTA = 96.8
- ByteTrack: HOTA = 63.1 (MOT17), ~60 HOTA on SportsMOT baseline
- Sports tracking SOTA: SAM 3-Deep-EIoU **87.2 HOTA** on SportsMOT (new SOTA), SAMIDARE **85.6 HOTA** on SportsMOT (CVPR 2026W)

**Conformal prediction calibration:**
- Split conformal prediction provides FINITE-SAMPLE coverage guarantees with NO RETRAINING required on any pretrained model.
- Validated across **18 VLMs** on 6 multimodal datasets (EACL 2026): strictly controls error rate at user-specified α.
- ECE on uncalibrated VLMs is **10–42%** (VL-Calibration paper: 0.421 ECE on Qwen3-VL); post-hoc calibration reduces to **2–5%** (temperature scaling) or **0.098** (VL-Calibration).
- VLM average ECE drops to **0.05** after temperature scaling across 35 VLMs.

**Multi-rate asynchronous scheduling:**
- NVIDIA DeepStream nvtracker and similar SDKs have separated tracking cadence from inference cadence in production for years.
- Bounded queues, coalescing, priorities, backpressure are all standard production systems engineering — no research novelty here.

**VLM async execution at 0.3–1 Hz (not 30):**
- **API MODELS (2026):** GPT-4.1: 161–165 tok/s, TTFT 0.91–0.96s ($2/$8 per M tokens). GPT-4.1 nano: 134–142 tok/s, TTFT 0.68s ($0.10/$0.40). Gemini 2.5 Flash: 130–182 tok/s ($0.30/$2.50). Gemini 2.5 Flash-Lite: **~305 tok/s** ($0.10/$0.40). Claude 4.5 Haiku: 86–118 tok/s ($1/$5). Claude Sonnet 5: 61 tok/s ($2/$10).
- **LOCAL RTX 4090:** Qwen3-VL-30B-A3B FP8: **90 tok/s** (4090D 48GB). Qwen2.5-VL-32B AWQ: **62 tok/s**, 240ms image encode. Qwen2.5-VL-7B FP8: **150 tok/s**, 180ms image encode. Phi-4-multimodal FP8: **320 tok/s**, 140ms image encode. Llama 3.2 Vision 11B FP8: **115 tok/s**, 220ms image encode. InternVL2 8B FP8: **140 tok/s**, 200ms image encode.
- Video: 8 frames @720p = ~1.4 s encode; 64 frames @540p = ~8.8 s encode.
- **HARD FACT:** 30 FPS VLM is impossible. Even 1 Hz full VLM reasoning is a stretch for 32B+ models on a single GPU. 0.3–0.5 Hz (every 2–3 seconds) is the realistic ceiling for local deep reasoning.
- **Adversarial vulnerability:** LLaVA white-box ASR: 52.6–66.9%. Qwen2.5-VL: 6.5–15.5%. Deepfake detectors: 83–97% misclassification under adversarial attack. Multi-turn attacks bypass safety with 90–100% ASR (MUSE, Mar 2026).
- **Distribution shift:** VLM-RobustBench (Mar 2026): geometric distortions cause up to **34pp accuracy drops**. Low-severity perturbations often degrade more than visually severe ones.
- **Token compression:** StreamingTOM (CVPR 2026): **15.7× KV-cache compression**, memory reduced to 6.4% of baseline, 2× TTFT speedup. STC (CVPR 2026): **24.5% ViT + 45.3% LLM latency reduction**, 99% accuracy retained. FastVLM (CVPR 2025): **75% visual tokens compressed** (576→144), 31.2% faster generation. EarlyTom (CVPR 2026): **2.65× TTFT reduction**, 61% FLOPs reduction. ALVTS (2026): **89% tokens compressed**, 96.7% accuracy retained, 1.6× speedup. KVCapsule (May 2026): **2× TPS improvement**, 2.4× KV cache reduction. HYBRIDKV (ACL 2026): **7.9× KV cache reduction**, 1.52× decoding speedup on Qwen2.5-VL-7B. MixKV (ICLR 2026): +5.1% avg over baselines at budget=64.
- **StreamingVLM:** 8 FPS on H100, 66.18% win vs GPT-4o-mini (ICLR 2026).
- **Inference engines:** vLLM (easy, highest throughput at 5,333 tok/s on H100), TensorRT-LLM (lowest latency: 235ms TTFT at concurrency 32, 10–30% higher throughput ceiling), SGLang (29% advantage on prefix-heavy/RAG workloads via RadixAttention: 16,200 tok/s). Cost: vLLM $0.158/M tokens (FP8), SGLang $0.170.
- **Prefix caching:** UniCache (SIGMetrics 2026): 3.86–17.32% higher hit ratio, 1.10–3.63× lower TTFT vs LRU. GraniKV (Aug 2026): 2.16× throughput at 16K shared prefix. LMCache: 3.0× lower TTFT avg, 2.1× lower p95 under stress.

### ⚠️ PLAUSIBLE / REQUIRES CAREFUL ENGINEERING (not magic, but needs work):

**Online 30 FPS hidden-state inference for occluded pose:**
- OFFLINE: ViDiHand (Jun 2026): **21.67mm MPJPE** on ARCTIC egocentric hands, **0.997 frame accuracy**, 5.5 FPS on 4× A100 (offline clip-level). 30–61% MPJPE reduction vs baselines.
- **ONLINE SOTA (2026):** MoRo (3DV 2026): **70 FPS on H200**, MPJPE visible = 37.83mm, occluded = 48.53mm (only 28% gap). RAM (CVPR 2026): **10.32 FPS**, MPJPE = 53.0mm on 3DPW, reduces ID switches from 349 to 15. HiPART (CVPR 2025): **396 FPS** (single frame), **577 FPS** (seq=243), MPJPE = 42.0mm on H36M. EvaPose (CVPR 2026): **48 FPS** on V100.
- 30 FPS online path requires: lightweight 2D/3D pose model (HybrIK-style hybrid analytical-neural) + Kalman/extended Kalman smoothing + joint-limit clamping + linear extrapolation.
- Expected error: MPJPE will increase ~20–40% during occlusion compared to visible. Distribution output (not point estimate) is mandatory.

**Evidence graph + hypothesis ranking:**
- Schema design is straightforward engineering. What needs work: (a) bounded hypothesis set size (top-K, K=3–5 typically), (b) efficient contradiction detection via cross-modal agreement checks, (c) hypothesis coalescing/splitting heuristics.
- "Falsification" in practice is: maintain top-3 hypotheses, run a lightweight verifier pass that looks FOR evidence against the leading hypothesis, not just for it.

**Decomposed confidence with semantic meaning:**
- The 8-way decomposition (observation → perception → state-estimation → temporal → cross-modal → hypothesis → reasoning → calibrated) is architecturally sound.
- VL-Calibration (ACL 2026): Qwen3-VL-4B: ECE **0.421→0.098** (76.7% reduction), accuracy +2.3% (0.704→0.727). Qwen3-VL-8B: ECE **0.401→0.071**, accuracy +3.0% (0.731→0.761). Qwen3-VL-30B: ECE **0.388→0.082**, accuracy +0.151. A-OKVQA: ECE **0.112→0.017**.
- Challenge: calibrating the COMPOSITION, not just the final number. Conformal prediction on the final scalar is easy; propagating coverage guarantees through the composition chain needs a formal treatment.
- CAP (ACML 2025): >70% calibration error reduction (82.9% vs APS, 74.8% vs LAC), +22.2% hallucination detection AUROC, +21.2% selective generation AUARC. Coverage consistently meets 90% target.

### ❌ CURRENTLY IMPOSSIBLE / SPECULATIVE WITH CURRENT TECHNOLOGY (set these aside):

**"General" counterfactual simulation from arbitrary video:**
- Causal graph induction from raw video without domain-specific supervision is an UNSOLVED research problem at scale. Video contains correlations; distinguishing causation requires interventions or domain priors (e.g., sports rules).
- Do NOT promise general counterfactual simulation. Limit causal claims to: (a) explicitly rule-governed domains (sports fouls), (b) A/B-tested interventions in synthetic data, (c) before/after physical discontinuities with clear physics priors.

**30 FPS end-to-end "deep reasoner" on every frame:**
- Mathematically impossible with 2026 hardware. A single 32B VLM forward pass for one 1024×1024 image takes ~1.7 s minimum on a 4090. 30 FPS would require ~18× 4090s in parallel for the VLM alone, ignoring the perception stack.
- The multi-rate design IS the only viable architecture — this is not a choice, it is a physical constraint.

**"Perfect" or 95%+ accurate synthetic media detection across all models:**
- RA-Bench 2026 (17,886 videos, 19 detectors): Traditional detectors collapse from 84.2% public AUC → **57.3% mean RA-Bench AUC** (open generators), **43.9%** (closed generators). 26 of 63 detector–source pairs scored **below 50% AUC** (worse than random). Spearman correlation between public and RA-Bench rankings: only **0.26**.
- FVBench (CVPR 2026): 120K+ videos, 42 generators. Best zero-shot: InternLM-XComposer2.5 **92.98%**. GPT-4o: **49.86%** (near random). After fine-tuning: 3 models achieve **100%** on their training generators.
- NTIRE 2026 (CVPR 2026): DINO-MAC **0.9168 AUC** (1st), INTSIG **0.8824** (2nd), AntInternational **0.8691** (3rd). TRIDENT ensemble: **0.860 AUC**.
- SOTA open-source detectors lose **45–50% AUC** on real-world content (Miyachi & Uys 2026).
- C2PA: **152 conformant products** (132 generators + 20 validators) as of Aug 2026, specification v2.4.
- SynthID: **10B+ images** watermarked. Survives **300 JPEG compression cycles**. Now integrated with OpenAI, Nvidia, ElevenLabs.
- **Set aside binary/authentic classification entirely.** The three-state (authentic_supported / synthetic_supported / inconclusive) + multi-detector corroboration + C2PA provenance architecture is NOT an option — it is the ONLY defensible design.

**Open-world single-detector "general" entity/relation recognition from video with 99% recall:**
- No model family achieves this. SAM 3 (Nov 2025) achieves **47.0 LVIS zero-shot AP** (2× prior SOTA of 38.5), but that's detection not open-world relation recognition. Open-vocabulary detectors top out at ~60–65 mAP on LVIS, not 95%. Expect recall gaps on rare entities; design around them with evidence-level "don't know" rather than hallucinating.

---

## 7. Research novelty candidates (REALISTIC VERSION)

Potentially novel system-level contributions that are BUILDABLE in 2026:

1. An evidence-centric world-state representation shared across perception and reasoning — **plausible system integration novelty, not algorithmic novelty.**
2. A multi-rate scheduler driven by quantified uncertainty + prediction error, with measurable compute vs. accuracy Pareto curves — **doable; prior work on scheduling exists, but not with this exact trigger formulation.**
3. Explicit separation of observed/inferred/hypothesis/prediction/causal claim types in a streaming claim protocol with UI/API parity — **protocol/systems novelty.**
4. Calibrated confidence decomposition across perception/state/temporal/reasoning with validated per-component coverage (not just final scalar) — **research potential; conformal composition is non-trivial.**
5. Hypothesis falsification as first-class: mandatory "evidence against leading hypothesis" check before claim elevation from candidate to supported — **implementation novelty, straightforward heuristic but rarely built in.**
6. Unified reasoning over natural/fast/synthetic media using the SAME evidence-graph substrate (different detector plugins, same claim infrastructure) — **architectural novelty if executed well.**
7. 30 FPS state continuity + 0.3–1 Hz deep VLM reasoning with explicit staleness/age tags on async beliefs — **systems engineering achievement, very buildable with current parts.**
