# 12. Physics and biomechanics

## Objective
Use physically plausible dynamics to constrain hidden-state inference.

## Research
- Human kinematics
- Joint limits
- Velocity/acceleration continuity
- Contact constraints
- Rigid-body dynamics
- Collision dynamics
- Ball/object trajectories
- Differentiable physics
- Learned physics models

## Target behavior
A visually plausible but physically impossible interpretation should be rejected or assigned low confidence.

## Deliverable
A library of domain-independent physical constraints plus optional domain-specific models.

---

## REALITY CHECK 2026

### ✅ PROVEN, implement with confidence:
- **Joint limits (SMPL / GHUM / SMPL-X):** Well-defined, millions of meshes validated. Project pose to feasible manifold via inverse kinematics. <1 ms overhead.
- **Velocity/acceleration continuity:** Simple state checks; catches teleportation and impossible jerks.
- **Rigid-body dynamics + projectile motion (balls):** Analytical solution trivial. 99% accurate for ball trajectories once mass/drag are estimated.
- **Contact constraints (penetration rejection):** Bounding box + mesh interpenetration check. Fast for small entity counts (<50).

### ⚠️ PLAUSIBLE, engineering effort:
- **Human kinematics (full body):** HybrIK-style hybrid analytical-neural IK works at ~10 ms per frame. Do NOT attempt "learned physics model end-to-end" for humans; analytical joint limits + linear smoothing is faster, more accurate, and interpretable.
- **Collision dynamics (impulse-based):** Ball-on-ball, ball-on-rigid-object collisions are reliable. Human-human collisions are approximate (use bounding volumes, don't attempt muscle-level simulation).
- **Differentiable physics for gradient-based state fitting:** If you need to fit a hidden state to observations (e.g., ball trajectory during occlusion), tiny differentiable physics sims (tinygrad, Brax, Nimble, PyTorch3D) work at 10–100 Hz for small scenes. Use in async path.

### ❌ IMPOSSIBLE or OVERKILL for 2026 streaming:
- **Full-body muscle/biomechanical simulation at 30 FPS for 20+ humans:** Too compute heavy (minutes per frame for SOTA models). Pointless for this system — you don't need muscle activation to reject an impossible knee angle.
- **Accurate cloth/deformable simulation:** Don't do it unless it's a core domain requirement. Simple joint + shape limits are 95% as effective at rejecting physically implausible poses.
- **General learned physics model that transfers across all objects:** Dreamer-style world models fail on in-the-wild deformables + humans; keep them out of the critical path.

### PRACTICAL CONSTRAINT LIBRARY (build this, 2026 roadmap):
| Constraint | Compute | Accuracy | Path |
|---|---|---|---|
| SMPL joint-angle limits + bone length | <0.5 ms | 99% | 30 FPS critical path |
| Velocity/acceleration human-max bounds (e.g., ≤12 m/s sprint) | <0.1 ms | 98% | 30 FPS critical path |
| Penetration (AABB + sphere) | O(N²), N<50 → <1 ms | 90%+ | 30 FPS critical path |
| Projectile motion (ball, rigid) | <0.01 ms | 95% | 30 FPS critical path |
| Human IK projection to feasible (HybrIK-like) | 5–15 ms | 90–95% | 5–10 Hz periodic |
| Collision response, rigid bodies | 1–5 ms small scene | 90% | 1–5 Hz async |
| Differentiable physics fitting hidden state | 50–500 ms | case-dependent | Event-triggered only |

### TARGET BEHAVIOR (enforceable TODAY):
A visually plausible but physically impossible interpretation (e.g., knee bending backward, ball teleporting) must:
1. Be caught by the 30 FPS joint-limit checker or velocity checker.
2. Trigger a state re-estimate (e.g., fall back to previous known-good pose + linear motion).
3. Inflate state-estimation uncertainty proportionally.
4. If no valid interpretation fits, emit `inconclusive` rather than hallucinate.
This check is cheap and high-value. No excuses.
