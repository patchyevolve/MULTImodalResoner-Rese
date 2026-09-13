# Source index

The research direction is grounded in the following representative literature. The list is a starting point, not an exhaustive bibliography.

## Perception & Detection
1. RF-DETR — Roboflow, ICLR 2026. 60.1 mAP (2XL), first real-time >60.
   https://arxiv.org/abs/2511.09554
2. YOLO26 — Ultralytics, Jan 2026. NMS-free, 57.5 mAP (X), 1.7ms (N).
   https://arxiv.org/abs/2606.03748
3. DETRPose — Jul 2025. 73.3 AP (X), 2.39ms A10 (S).
   https://arxiv.org/abs/2506.13027
4. RF-DETR Keypoint — Jun 2026. 71.8 AP at 9.8ms (576px).
   https://blog.roboflow.com/rf-detr-keypoint/
5. RF-DETR-Seg — Jan 2026. 40.3 AP (N) at 3.4ms.
6. SAM 3 — Meta, Nov 2025. 47.0 LVIS zero-shot AP, 30ms H200.
   https://github.com/facebookresearch/sam2
7. SAM 3.1 — Meta, Mar 2026. 32 FPS medium objects H100.

## Tracking
8. ByteTrack — ECCV 2022. Baseline multi-object tracker.
   https://arxiv.org/abs/2110.06864
9. McByte++ — Training-free MOT, HOTA 85.0 SoccerNet, Aug 2026.
   https://arxiv.org/abs/2608.15688
10. SAMIDARE — CVPR 2026W. HOTA 85.6 on SportsMOT.
11. SAM 3-Deep-EIoU — HOTA 86.8 on SportsMOT (new SOTA).

## VLMs & Inference
43. StreamingVLM — Streaming video language modeling, ICLR 2026. 8 FPS H100.
    https://research.nvidia.com/labs/eai/publication/streamingvlm/
44. StreamingTOM — CVPR 2026. 15.7× KV-cache compression, 2× TTFT speedup.
    https://openaccess.thecvf.com/content/CVPR2026/html/Chen_StreamingTOM_Streaming_Token_Compression_for_Efficient_Video_Understanding_CVPR_2026_paper.html
45. STC — CVPR 2026. 24.5% ViT + 45.3% LLM latency reduction.
    https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Accelerating_Streaming_Video_Large_Language_Models_via_Hierarchical_Token_Compression_CVPR_2026_paper.html
46. FastVLM — CVPR 2025. 75% visual tokens compressed, 31.2% faster generation.
47. EarlyTom — CVPR 2026. 2.65× TTFT reduction, 61% FLOPs reduction.
48. ALVTS — 2026. 89% tokens compressed, 96.7% accuracy retained, 1.6× speedup.
    https://arxiv.org/abs/2606.14277
49. KVCapsule — May 2026. 2× TPS improvement, 2.4× KV cache reduction.
    https://arxiv.org/abs/2605.16439
50. HYBRIDKV — ACL 2026. 7.9× KV cache reduction, 1.52× decoding speedup on Qwen2.5-VL-7B.
51. MixKV — ICLR 2026. +5.1% avg over baselines at budget=64.
    https://arxiv.org/abs/2510.20707
52. UniCache — SIGMetrics 2026. 3.86–17.32% higher hit ratio, 1.10–3.63× lower TTFT vs LRU.
53. GraniKV — Aug 2026. 2.16× throughput at 16K shared prefix.
    https://arxiv.org/abs/2608.15584

## Calibration & Uncertainty
54. VL-Calibration — ACL 2026. ECE 0.421→0.098 (4B), 0.401→0.071 (8B), 0.388→0.082 (30B).
    https://arxiv.org/abs/2604.09529
55. Conformal prediction for 18 VLMs — EACL 2026. Empirical error < α across all α values.
    https://aclanthology.org/2026.findings-eacl.274/
56. SCP for LVLMs — arXiv 2025. α=0.2 avg error 0.1934 (ScienceQA), 0.1763 (MMMU).
    https://arxiv.org/abs/2504.17671
57. QaTS — Quantile-Adaptive Temperature Scaling, 2026. 40–75% ECE reduction vs TS.
    https://arxiv.org/abs/2606.21749
58. SMART — Sample Margin-Aware Recalibration, 2025. 7 parameters, SOTA across 6 datasets.
    https://arxiv.org/abs/2506.23492
59. CAP — Conformalized Abstention Policies, ACML 2025. 82.9% ECE reduction vs APS.
    https://proceedings.mlr.press/v304/tayebati26a.html
60. BCEA — Budgeted Conformal Evidence Acquisition, 2026.
    https://arxiv.org/abs/2606.16667
61. CCNet — Calibrated ConfidenceNet for pose, ICML 2024.
62. UNI-OOD — CVPR 2026. Unified object/image-level OOD detection in VLMs.
63. VLM-UQBench — Feb 2026. 9 UQ methods on 4 VLMs across 3 datasets.
64. FUSE — Jun 2026. Bayesian fusion of aleatoric + epistemic uncertainty in VLMs.

## Occluded Pose & Hidden State
19. ViDiHand — 21.67mm MPJPE, 0.997 frame accuracy, 5.5 FPS 4×A100, Jun 2026.
    https://arxiv.org/abs/2606.30308
20. MoRo — 70 FPS H200, 37.83mm visible / 48.53mm occluded MPJPE, 3DV 2026.
    https://arxiv.org/abs/2601.16079
21. RAM — CVPR 2026. 10.32 FPS, 53.0mm MPJPE, 15 ID switches, MOTA 74.4 PoseTrack21.
    https://arxiv.org/abs/2603.19929
22. HiPART — CVPR 2025. 396 FPS single, 577 FPS seq, 42.0mm MPJPE on H36M.
23. EvaPose — CVPR 2026. 48 FPS V100, 35.6mm MPJPE.
24. HybrIK — CVPR 2021. 45.0mm 3DPW PA-MPJPE, 34.5mm H36M PA-MPJPE.
25. MoPO — 2026. Motion de-occlusion + completion, 64.9mm 3DPW-OC.
    https://arxiv.org/abs/2605.09856
26. PHAC — CVPR 2026. Promptable Human Amodal Completion.
27. BMP — ICCV 2025. +39% detection in large instance overlap, SOTA on OCHuman.
28. VisOR — WACV 2026. +7% on occluded benchmarks, BOW benchmark.

## Synthetic Media & Forensics
25. RA-Bench — 17,886 videos, 19 detectors, 26 pairs below 50% AUC, Aug 2026.
    https://arxiv.org/abs/2608.14391
26. FVBench — CVPR 2026. 120K+ videos, 42 generators. Best zero-shot: 92.98%.
27. GenVidBench — AAAI 2026. 6.78M videos, 11 generators.
28. NTIRE 2026 — CVPR 2026 Workshop. DINO-MAC 0.9168 AUC (1st), INTSIG 0.8824 (2nd).
29. C2PA Specification v2.4 — 152 conformant products (132 generators + 20 validators).
    https://c2pa.org/
30. SynthID — 10B+ images watermarked. 99.72% worst-case TPR. Survives 300 JPEG cycles.
    https://arxiv.org/abs/2510.09263
31. BitMind Forensics — 0.936 AUC Sumsub, 0.915 Deepfake-Eval-2024, 0.918 GenVidBench.
    https://arxiv.org/abs/2607.13234
32. Resemble DETECT-World — 99.5% audio, 98% video, 96% image accuracy.

## Sports AI
33. SoccerMaster — CVPR 2026 Oral. 82.0 AP@50 detection, 86.0% event classification, 59.1 HOTA MOT.
34. DeepSport — Nov 2025. First end-to-end multi-sport MLLM. 37.67 overall (beats GPT-5 35.70).
    https://arxiv.org/abs/2511.12908
35. SoccerNet 2025/2026 Challenge — 5 tasks, 427 teams, 1,129 submissions.
    https://arxiv.org/abs/2607.07320
36. SportsGrounder — ACM MM 2026. 51.8% SoccerNet VQA, 53.6% FineSports VQA (2B params).
    https://arxiv.org/abs/2608.07932
37. McByte++ — Training-free MOT, HOTA 85.0 SoccerNet, Aug 2026.
    https://arxiv.org/abs/2608.15688
38. SAM 3-Deep-EIoU — HOTA 87.2 on SportsMOT (new SOTA, Jun 2026).
    https://github.com/holma91/selective-mask-propagation
39. Sports-LiteDet — 68 FPS Jetson Orin Nano, 5.8 MB, Nature 2026.
40. MatchVision — CVPR 2025. SoccerReplay-1988 dataset.
41. SoccerDETR — 78 FPS RTX 4090, 94.2% mAP@50 Soccana, Feb 2026.
    https://www.mdpi.com/2227-7080/14/3/142
42. TOTNet — Occluded ball tracking, RMSE 7.19 (from 37.30), Aug 2025.
    https://arxiv.org/abs/2508.09650

## Video Understanding
55. SAM 2 — Segment anything model for video with streaming memory.
    https://arxiv.org/abs/2408.00714
56. VidHalluc — Temporal hallucinations in VLMs, CVPR 2025.
    https://openaccess.thecvf.com/content/CVPR2025/html/Li_VidHalluc_Evaluating_Temporal_Hallucinations_in_Multimodal_Large_Language_Models_for_CVPR_2025_paper.html
57. VideoWorld 2 — Transferable knowledge from real-world videos, CVPR 2026.
    https://openaccess.thecvf.com/content/CVPR2026/html/Ren_VideoWorld_2_Learning_Transferable_Knowledge_from_Real-world_Videos_CVPR_2026_paper.html
58. CaST-Bench — Causal-chain spatiotemporal reasoning, CVPR 2026.
    https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_CaST-Bench_Benchmarking_Causal_Chain-Grounded_Spatio-Temporal_Reasoning_for_Video_Question_Answering_CVPR_2026_paper.html
59. Cover — Conformal coverage for video temporal grounding, Aug 2026.
    https://arxiv.org/abs/2608.07434
60. Video-MME-v2 — Non-linear scoring benchmark, Apr 2026.
    https://arxiv.org/abs/2604.05015
61. OmniVCHall — 823 videos, 9,027 QA, 8 hallucination types, ICML 2026.
    https://github.com/BMRETURN/OmniVCHall
62. EgoMemReason — Week-long egocentric video benchmark, May 2026.
    https://arxiv.org/abs/2605.09874
63. FlexMem — Training-free visual memory, >1,000 frames on 3090, CVPR 2026.
    https://arxiv.org/abs/2603.29252
64. CLASH — Cross-modal contradiction detection, CVPR 2026 Findings.
    https://arxiv.org/abs/2511.19199

## Systems
65. DeepStream nvtracker — Separating inference cadence and tracking.
    https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html
66. VAGEN — World model reasoning for multi-turn VLM agents.
    https://www.microsoft.com/en-us/research/publication/vagen-reinforcing-world-model-reasoning-for-multi-turn-vlm-agents/
67. VirtueBench — CVPR 2026. 25 VLMs trustworthiness evaluation.
68. NVIDIA RT-VLM — Cosmos Reason 2, 51 concurrent streams, 94.8% GPU util.
    https://docs.nvidia.com/vss/3.2.0/performance-rt-vlm.html
69. SpecVLM — Speculative decoding for VLMs, 2.5–2.9× speedup.
    https://arxiv.org/abs/2509.11815
70. ViSpec — Vision-aware speculative decoding, up to 3.22× speedup.
    https://arxiv.org/abs/2509.15235
71. Jetson Thor — 128GB edge VLM, Qwen3-VL-30B 218 tok/s, 47.75ms TTFT p99.
    https://developer.nvidia.com/embedded/jetson-benchmarks
72. GRPO for video VLMs — Temporal-R1, VIPO-R1, DeepVideo-R1, GRPO-CARE.
    https://arxiv.org/abs/2506.01908

## Adversarial Robustness
73. VLM-RobustBench — 49 augmentations, 133 settings, 15 VLMs. Up to 34pp drops.
    https://arxiv.org/abs/2603.06148
74. Adversarial attacks on VLMs — LLaVA 52.6–66.9% ASR, Qwen2.5-VL 6.5–15.5%.
    https://arxiv.org/abs/2603.16960
75. PHANTOM — 47,524 adversarial samples, 10 categories, 55 subcategories.
    https://arxiv.org/abs/2606.24388
76. Certified YOLO — CRA >98% under 10⁶ perturbations, ICLR 2026.
77. STREAM-OOD — Streaming OOD for video, reduces false alarms 3.4→1.6/hr, CVPR 2026W.
78. SABRE — Reusable VLM stress testing framework, Aug 2026.
    https://arxiv.org/abs/2608.07435
79. MUSE — Multi-turn safety evaluation, 90–100% ASR, Mar 2026.
    https://arxiv.org/abs/2603.02482

## Evidence Graphs & Reasoning
80. DSFlash — 56 FPS scene graph generation, CVPR 2026.
    https://arxiv.org/abs/2603.10538
81. MECD+ — Event-level causal graphs from video, TPAMI 2026.
    https://arxiv.org/abs/2501.07227
82. GraphThinker — Event-based Video Scene Graph, +4% IoU@0.3, Feb 2026.
    https://arxiv.org/abs/2602.17555
83. SPIKE-RL — Bayesian surprise for video hypothesis generation, ICLR 2026.
    https://arxiv.org/abs/2509.23433
84. MM-GoT — Multimodal Graph-of-Thoughts, +3.1pp, CVPR 2026W.
85. ENTER — Event graphs for VideoQA, 96.5% node precision, Apr 2026.
    https://arxiv.org/abs/2501.14194

## Ethical & Legal
86. EU AI Act Article 50 — Transparency obligations, effective 2 Aug 2026, penalties up to €15M/3%.
    https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act
87. C2PA Conformance — 54 products, 31 companies, Assurance Level 2 (Pixel 10).
    https://contentauthenticity.org/blog/raising-the-bar-for-trust-introducing-the-c2pa-conformance-program
88. VIGNETTE — 30M+ images, 150+ identities, 8 bias dimensions, ACL 2026.
    https://aclanthology.org/2026.acl-long.712.pdf
89. FairLens — 100K+ image-question pairs, 99% unwarranted inference, Sep 2026.
    https://arxiv.org/abs/2609.01691
90. UKJT Legal Statement — AI liability in English law, Jul 2026.
91. FedDP-STECAR — Federated DP for video, 70.2% higher accuracy under privacy, Mar 2026.
    https://arxiv.org/abs/2603.21305

## Multimodal Agents & Integration
92. EVA — Planning-before-perception video agent, CVPR 2026.
93. SAGE — Any-horizon agents for long video, CVPR 2026.
94. VITAL — Tool-augmented GRPO for video, CVPR 2026.
95. Agent-X — 828 agentic tasks across 6 environments, ICLR 2026.
    https://arxiv.org/abs/2505.24876
96. M* — Modular multimodal serving, 20% lower latency than vLLM-Omni.
    https://arxiv.org/abs/2606.12688
97. RAPID — Perception-reasoning disentanglement, ICLR 2026.
     https://arxiv.org/abs/2506.04559

## Architecture & Scheduling
98. NVIDIA RT-VLM — Real-time VLM on DeepStream, 51 concurrent streams on H100. VSS v3.2.1.
     https://docs.nvidia.com/vss/3.2.1/real-time-vlm.html
99. NVIDIA Patent US20250292557A1 — VLM inference scheduler for vehicles, 30/60 FPS, safety-prioritized.
100. UnifiedServe — MPS-based multi-worker, 4.4× throughput, vision+LLM GPU sharing. Dec 2025.
      https://arxiv.org/abs/2512.17574
101. HeteroServe — Phase-aware multimodal scheduler, two GPU pools, cross-type work stealing. Mar 2026.
      https://arxiv.org/abs/2603.12707
102. TCM-Serve — Modality-aware scheduling, 54% TTFT reduction, video/image/text triage. May 2026.
      https://arxiv.org/abs/2603.26498
103. ReaLB — Real-time load balancing for multimodal MoE, 1.32× throughput, 8× RTX 5090. May 2026.
      https://arxiv.org/abs/2604.19503
104. Co-VStream — Edge-cloud collaboration, 2.99s E2E latency. Jun 2026.
      https://arxiv.org/abs/2606.22804
105. MOSS-Video-Preview — Two-channel architecture, 5× faster TTFT, 2.7× decoding throughput. Jun 2026.
      https://arxiv.org/abs/2606.07639
106. FlashCodec — GOP-based parallel decoding, 2.8-9.1× speedup. Dec 2025.
      https://arxiv.org/abs/2512.17574
107. NVIDIA NvSchema — Production protobuf schema for video analytics, Kafka messaging.

## Inference Optimization
108. FlashAttention-4 — 1,613 TFLOPs/s on B200, 22× faster compilation, CuTe-DSL. Mar 2026.
      https://arxiv.org/abs/2603.05451
109. FlashFormer — Fuses entire transformer forward pass into single kernel.
      https://arxiv.org/abs/2505.22758
110. TensorRT-LLM v1.3 — NVFP4 on Blackwell, FP8 production-ready, KV cache compression.
111. TensorRT v11.2 — NVFP4 dual-GEMM fusion, global perf tuner, multi-device inference.
112. SpecVLM — 2.5-2.9× VLM speedup via speculative decoding with elastic visual compressor.
      https://arxiv.org/abs/2505.22758
113. ViSpec — Vision-aware speculative decoding, first meaningful VLM acceleration. NeurIPS 2025.
114. HCSpec — Two-tier horizontal cascade speculative decoding. ACL 2026.

## Token Compression
115. KVCapsule — 2.4× memory, 2× TPS via learned mask + PCA projection. May 2026.
      https://arxiv.org/abs/2605.16439
116. HybridKV — 7.9× memory, 1.52× decode via head classification + hierarchical budget. ACL 2026.
      https://aclanthology.org/2026.acl-long.13018/
117. MixKV — Importance + diversity for head-wise adaptive compression. ICLR 2026.
118. SelKV — 3.3× decode speedup at 100k tokens, 25% retention. Jul 2026.
      https://arxiv.org/abs/2607.16213
119. EarlyTom — 6.8× FLOPs in vision encoder, early-stage token compression. CVPR 2026.
      https://arxiv.org/abs/2605.30010
120. PruneSID — 88.9% tokens compressed, 96.3% accuracy retained. ICLR 2026.
121. DynaKV — 6% KV cache retained, 94% LongBench performance. Mar 2026.
      https://arxiv.org/abs/2603.04411

## Calibration & Uncertainty
122. VL-Calibration — ECE 0.098 on Qwen3-VL-4B (from 0.421), GRPO-based. ACL 2026.
      https://arxiv.org/abs/2604.09529
123. EACL 2026 Conformal — 18 VLMs, 21K questions, instruction-guided likelihood proxies.
      https://aclanthology.org/2026.findings-eacl.274/
124. Empirical Bayes CP — CP_r-value reduces set sizes 7-44% at α=0.05. May 2026.
      https://arxiv.org/abs/2605.23189
125. Proof-of-Perception — Conformal DAG for multimodal reasoning, +4.2% DocVQA. CVPR 2026.
      https://arxiv.org/abs/2603.00324
126. VLM-UQBench — 9 UQ methods, 4 VLMs, visual uncertainty near-random. Feb 2026.
      https://arxiv.org/abs/2602.09214
127. STREAM-OOD — OOD false alarms 3.4→1.6/hr on NYC traffic. CVPR 2026W.
      https://openaccess.thecvf.com/content/CVPR2026W/papers/stream-ood
128. EDL Warning — Evidential signal is misclassification bias, not true uncertainty. 2023.
      https://arxiv.org/abs/2310.12663

## Video Understanding
129. Video-MME-v2 — 800 videos, 3,200 QA, group-based non-linear scoring. Apr 2026.
      https://arxiv.org/abs/2604.05015
130. OmniVChall — 823 videos, 9,027 QA, camera-based hallucination type. ICML 2026.
      https://arxiv.org/abs/2602.00559
131. CrossVid — First cross-video reasoning benchmark, 5,331 videos. AAAI 2026.
132. EgoCross — Cross-domain challenge, OmniEgo-R2 66.35%. CVPR 2026.
133. RD-MLDG — Reasoning chains 58.6% lower cross-domain divergence than visual features.

## Claim Taxonomy
134. ClaimFlow — 5-relation taxonomy, 1,617 papers, 5,689 claims. 2026.
      https://arxiv.org/abs/2603.16073
135. SciLens — Multimodal scientific claim verification, atom decomposition. KDD 2026.
      https://arxiv.org/abs/2606.20873
136. Claim Verification Patterns — 6 patterns, 5 error types. Apr 2026.
      https://arxiv.org/abs/2604.01657

## Hardware
137. Jetson Thor — 2,070 FP4 TFLOPS, 128GB, 40-130W. 20B at 52 tok/s. GTC 2026.
138. Jetson Orin Nano Super — 7B at 21.75 tok/s, 15W. Jetson AI Lab.
139. RTX 5090 — 32GB, 209.5 TFLOPS FP16, 140 tok/s Llama 8B. $2K MSRP.

## Research policy
Prioritize original papers, official documentation, benchmark papers and primary datasets. Verify publication date, benchmark setup, hardware and task definition before comparing numbers.
