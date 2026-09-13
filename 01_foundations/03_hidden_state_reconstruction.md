# 3. Hidden-state reconstruction

## Objective
Infer unobserved or occluded state using temporal, structural and physical constraints.

## Research areas
- Occluded 2D/3D pose estimation
- Temporal pose completion
- Motion interpolation and extrapolation
- Object trajectory reconstruction
- Occlusion-aware tracking
- Depth and scene geometry completion
- Body kinematics and joint constraints
- Diffusion/transformer motion completion

## Critical experiment
Given frames before and after an occlusion, predict the distribution of hidden hand/foot positions and compare against ground truth.

## Required outputs
Not a single coordinate only; return a distribution, uncertainty region, and alternative trajectories.

## Deliverable
A hidden-state inference module with measurable error and calibration.

---

## REALITY CHECK 2026

### ✅ PROVEN methods, 30-FPS capable:
- **Temporal pose completion via Kalman + joint-limit clamping:** Works online, <1 ms overhead. MPJPE increase ~20% for moderately occluded joints.
- **HiPART (CVPR 2025):** **396 FPS** single frame, **577 FPS** seq-243, MPJPE = 42.0mm on H36M. Fastest published 3D pose method >60 FPS with <50mm MPJPE.
- **MoRo (3DV 2026):** **70 FPS on H200**, MPJPE visible = 37.83mm, occluded = 48.53mm (only 28% gap). Cross-modality learning: trajectory-aware motion prior + image-conditioned pose prior + video-conditioned masked transformer.
- **RAM (CVPR 2026):** **10.32 FPS**, MPJPE = 53.0mm on 3DPW, reduces ID switches from 349 to 15. PoseTrack21 MOTA **74.4** (vs. CoMotion 66.4, 4DHumans 57.7). Zero-shot generalization.
- **BMP (ICCV 2025):** Iterative mutual consistency of bounding boxes, instance masks, and poses. +39% detection improvement in scenes with large instance overlap. SOTA on OCHuman.
- **VisOR (WACV 2026):** Visibility-guided self-supervised occlusion-resilient estimation. +7% improvement over SOTA on occluded benchmarks. Introduces BOW benchmark.
- **Motion interpolation via spline or linear blending:** Deterministic, fast, reliable for short occlusions (<0.5 s).
- **Occlusion-aware tracking (ByteTrack with occlusion handling):** Production-grade, maintains IDs during 1–2 s occlusions in many scenarios.
- **Depth/scene geometry completion (for visible regions only):** Many models exist; depth-from-mono is 5–10 ms.

### ⚠️ PLAUSIBLE, but tradeoffs:
- **OFFLINE diffusion/transformer motion completion for heavy occlusions (1–5 s gaps):** ViDiHand (Jun 2026): **21.67mm MPJPE** on ARCTIC egocentric hands, **0.997 frame accuracy**, 5.5 FPS on 4× A100 (offline clip-level). These methods are clip-level, require lookahead (bidirectional). They belong to the ASYNC deep reasoning queue (0.1–0.3 Hz), not the 30 FPS path.
- **Motion de-occlusion + completion (MoPO, 2026):** Detects occluded joints via spatial confidence + temporal info from past frames, then completes motion from past joint sequence. 64.9mm MPJPE on 3DPW-OC (8.5% reduction from DPMesh's 70.9mm). First to introduce motion complement-based de-occlusion for occluded human mesh recovery.
- **Amodal completion (PHAC, CVPR 2026):** Completes occluded human images with user-specified pose/region prompts using ControlNet modules + inpainting refinement.
- **Object trajectory reconstruction during 1–2 s occlusion:** Linear + ballistic extrapolation works for balls; humans are harder. Use Gaussian mixtures (multiple possible trajectories) not single best path.

### ❌ SET ASIDE (impossible or unwise for this system):
- **"Perfect" hidden pose reconstruction with <5% error increase vs. visible:** No method in 2026 achieves this, even offline. Visible vs. occluded MPJPE gaps of 20–50% are the reality even for SOTA. Accept and surface uncertainty.
- **Returning a single coordinate ONLY for hidden state:** This is malpractice. Always return distribution + alternatives. A point estimate for an occluded joint is a lie calibrated to look confident.

### REQUIRED OUTPUT CONTRACT for every hidden-state variable:
1. Posterior distribution: covariance matrix OR Gaussian mixture (2–3 components).
2. Uncertainty region: 95% confidence ellipse/ball.
3. Alternative trajectories (top-2 plausible paths, not just MAP).
4. Provenance: flag whether this estimate is from temporal_extrapolation / offline_diffusion / physics_prior / tracker_propagation.
5. Expected error growth curve per millisecond of continued occlusion (so scheduler can plan a re-detect or deep-reasoning call).
