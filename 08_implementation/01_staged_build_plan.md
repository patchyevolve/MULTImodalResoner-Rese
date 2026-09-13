# Staged implementation plan

## Stage 0 — Formal specification
Freeze ontology, claim types, evidence schema and evaluation definitions. Evidence schema: `{id, produced_at_ms, source_modality, source_window, source_spatial, producing_model, relation, weight_llr, weight_estimator}`. Timeline: 1–2 weeks.

## Stage 1 — Fast perception backbone
Implement decode → detector → tracker → pose → world-state representation. RF-DETR-N/S + ByteTrack + DETRPose-S. Critical path: ~13–15ms/frame. Compute: pretrained, no training needed. Timeline: 2–3 weeks.

## Stage 2 — Temporal state
Add motion prediction, short-term memory and hidden-state reconstruction. Kalman filter + joint-limit clamping + HiPART for 3D lifting. Compute: <1 GPU day for integration. Timeline: 2–3 weeks.

## Stage 3 — Evidence graph
Connect observations and state estimates to explicit claims and hypotheses. Schema from Stage 0 + DSFlash-style scene graph construction (56 FPS proven). Compute: <1 GPU day. Timeline: 2–3 weeks.

## Stage 4 — Hypothesis engine
Generate alternatives, rank them and detect contradictions. GBDT ranker + SPIKE-RL-style Bayesian surprise. Compute: <1 GPU day for ranker training. Timeline: 2–3 weeks.

## Stage 5 — Calibration
Train/evaluate confidence decomposition and calibration. VL-Calibration + SCP + isotonic regression. Compute: 0 additional training. Timeline: 1–2 weeks.

## Stage 6 — Asynchronous deep reasoning
Integrate a general VLM/LLM behind an event-driven scheduler. Qwen3-VL-30B-A3B on RTX 4090 (90 tok/s). M* serving for multi-model. Compute: QLoRA 4–16 A100-days if fine-tuning. Timeline: 3–4 weeks.

## Stage 7 — Sports specialization
Add domain rules, trajectories, player/ball reasoning and anticipation. SoccerNet 2026 baseline: 24.08% avg mAP. Compute: 1–8 A100-days for LoRA. Timeline: 2–3 weeks.

## Stage 8 — Synthetic-media branch
Add provenance and forensic evidence models. TRIDENT ensemble + C2PA verification. Compute: <1 GPU day for meta-classifier. Timeline: 2–3 weeks.

## Stage 9 — 30 FPS optimization
Profile end-to-end pipeline and optimize bottlenecks. SpecVLM (2.5–2.9× speedup) + Jetson Thor deployment (218 tok/s). Timeline: 2–3 weeks.

## Stage 10 — Cross-domain evaluation
Run the full benchmark (E1-E10) and perform ablations. VLM-RobustBench (49 augmentations) + STREAM-OOD. Timeline: 2–3 weeks.

**Total estimated timeline: 22–30 weeks (5–7 months) for a small team.**
**Total compute: ~500–1,000 RTX 4090 hours or ~100–200 A100 hours.**
