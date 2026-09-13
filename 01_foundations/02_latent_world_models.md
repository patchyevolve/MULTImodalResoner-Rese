# 2. Latent world models

## Objective
Determine how the reasoner should maintain an internal representation of the world that changes over time.

## Research questions
- What state variables are sufficient: objects, pose, velocity, relations, scene, intent, physical state, uncertainty?
- Should the state be object-centric, token-centric, graph-centric, latent-vector, or hybrid?
- How do learned latent dynamics compare with explicit kinematic/physics models?
- Can the model predict state transitions without reconstructing pixels?
- How should appearance be separated from task-relevant dynamics?
- Can the world model support counterfactual simulation?

## Candidate families
- Video world models
- State-space models
- Recurrent latent models
- Predictive self-supervised models
- Object-centric world models
- Graph neural dynamics models
- Hybrid learned + analytical dynamics

## Deliverable
A recommended latent state design and transition model.

---

## REALITY CHECK 2026

### ✅ PROVEN state-variable choices:
- Objects (bbox, class, ID, track confidence): production-tested. RF-DETR-2XL achieves **60.1 mAP** on COCO (ICLR 2026). SAM 3-Deep-EIoU achieves **87.2 HOTA** on SportsMOT (Jun 2026).
- Pose + velocity + acceleration (per-joint or per-object): standard in tracking. HiPART (CVPR 2025): **396 FPS**, 42.0mm MPJPE. MoRo (3DV 2026): **70 FPS**, 37.83mm visible / 48.53mm occluded.
- Scene (class, depth estimate, camera intrinsics estimate): off-the-shelf.
- Relations (spatial: above/near/contact; simple): Graph neural networks for scene graphs work at ~1 Hz.
- **Uncertainty per variable (covariance or histogram):** mandatory and proven. VL-Calibration (ACL 2026): ECE 0.421→0.098 on Qwen3-VL-4B.

### ⚠️ PLAUSIBLE but DO NOT expect miracles:
- **Intent or "planning state" for humans:** In 2026, learned intent prediction from video is ~50–70% top-1 accuracy on closed datasets (sports actions, kitchen tasks) at 0.5–2 s horizons. It's a hypothesis, never a fact. Store intent as posterior over K=3–5 intents, never a single label.
- **Object-centric + graph state representations:** Object-centric models (Slot Attention, etc.) work in lab settings but are ~10× slower than detection-based pipelines for the same accuracy. Use detection-tracker for 30 FPS path; consider object-centric latent features ONLY for async deep reasoning path if compute allows.
- **Video world models for prediction:** Seedance 2.0 (ByteDance) leads T2V Arena (Elo 1213), 15s max. Veo 3.1 (Google) leads physics realism. V-JEPA 2.1 outperforms VideoMAE on corruption robustness. But these are generation models, not state-space predictors — they generate pixels, not typed state variables.

### ❌ DO NOT WASTE TIME ON (set aside):
- Pure learned latent vector (no typed state variables) for the 30 FPS path: Uninterpretable, calibration is essentially impossible, and learned dynamics drift catastrophically after ~1 s for in-the-wild humans.
- Predicting state transitions WITHOUT reconstructing pixels, "general" across all object categories: Physics-based world models only generalize reliably for rigid bodies (balls, rigid objects). Articulated humans + deformable objects do not have stable learned pixel-free dynamics across scenes in 2026.
- General counterfactual simulation ("what would happen if the player moved left?"): Not reliably doable from single monocular video without a 3D scene model + dynamics engine + intervention label.

### RECOMMENDED (for buildability 2026):
**Hybrid learned + analytical dynamics (NOT neural-only):**
- Rigid bodies (ball, equipment): analytical kinematics + Kalman. This is 95% accurate for ballistics.
- Humans: per-joint velocity + acceleration + SMPL/GHUM joint-limit projection + per-joint covariance. No need for "learned human dynamics model" end-to-end; simple linear + joint-limit constraints work well for ≤0.5 s forecasts.
- Scene: static (furniture, field lines) vs dynamic (players, ball) separation.
- Async deep reasoning path: Can use a video world model (e.g., VideoWorld 2 style) for richer predictions, but outputs are hypotheses only, not state.
