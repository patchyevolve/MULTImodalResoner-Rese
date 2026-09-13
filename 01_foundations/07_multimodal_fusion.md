# 7. Multimodal fusion

## Objective
Combine vision, audio, text and metadata while accounting for source reliability.

## Research questions
- Early vs late vs cross-attention fusion?
- How should modality reliability be learned?
- How should missing modalities be handled?
- How should contradictory modalities be represented?
- Can the reasoner assign different evidence weights dynamically?

## Target behavior
If vision is strong and audio is corrupted, audio should not dominate simply because its classifier output has high numerical confidence.

## Deliverable
Reliability-aware multimodal fusion layer and conflict protocol.

---

## REALITY CHECK 2026

### ✅ PROVEN fusion strategies:
- **Late fusion with per-modality reliability weights (tracked over time):** This is the workhorse; simple, auditable, and production-grade.
- **Cross-modal conflict detection:** Easy — compare cross-modal predictions; if KL divergence / disagreement above threshold → escalate uncertainty. CLASH (CVPR 2026 Findings): best models achieve only 69–75% contradiction detection on cross-modal conflicts.
- **Missing modality handling:** Weight of missing modality = 0; re-normalize remaining weights. This works acceptably if at least 2 modalities remain.
- **Perception-reasoning disentanglement:** RAPID (ICLR 2026) modularizes MLLM into perception (VLM) + reasoning (LLM). Perception outputs serve as universal interface. Replacing images with text descriptions raises performance by **20+ points** on Claude models — perception (not reasoning) is the binding constraint.

### ⚠️ PLAUSIBLE, implement:
- **Dynamic reliability estimation per-claim, not global:** A given audio classifier may be unreliable for gunshot detection in loud stadiums but reliable for speech. Track per-event reliability on a calibration set. Engineering work; conceptually trivial.
- **Conflict protocol (what to do when modalities disagree):**
  1. Tag claim as `cross_modal_conflict`.
  2. Increase hypothesis set size (top-3 not top-1).
  3. Do NOT emit as `supported`; remain `candidate` or `inconclusive`.
  4. Trigger async VLM review.

### ❌ AVOID (speculative or overkill):
- **Cross-attention / early fusion across all modalities end-to-end for the 30 FPS path:** Heavy, slower, calibration terrible, no easy way to assign blame when modalities conflict. Keep late fusion for 30 FPS path; cross-attention only inside the async VLM deep reasoning.
- **Expecting audio to "dominate" vision when both are present:** In real data, audio is frequently corrupted (background noise, compression). Learn per-context reliability; don't hard-code modality ranking.

### TARGET BEHAVIOR verified in 2026:
If vision strongly shows "player kicked ball" (bbox overlap + pose match) but audio classifier says "no kick sound" (stadium echo), vision's higher reliability (for this task) must win, but claim confidence is reduced and the audio contradiction is preserved in evidence. Conversely: if audio has clear speech "GOAL" but vision is 100% occluded, weight audio's text claim high but keep epistemic tag = `audio_only_claim` with elevated unknown on spatial details.

### FUSION WEIGHTS LEARNING:
- Use a calibration set split with known accuracy per modality per task type.
- Simple logistic regression to learn weights (fast, interpretable).
- Update online via EMA if a VLM "ground truth" adjudication comes in later (async).
- Do NOT use a 10-layer neural fusion network; opacity hurts debuggability and calibration.
