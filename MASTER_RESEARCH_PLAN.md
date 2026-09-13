# Master research plan

## Phase 1 — Scientific foundation
Study tracks 1–5 first. Establish the mathematical model of partial observability, latent state, hidden-state inference, temporal memory and predictive dynamics.

## Phase 2 — Evidence and reasoning
Study tracks 7–10. Define multimodal evidence fusion, evidence graphs, hypothesis management and confidence calibration.

## Phase 3 — Prediction and causality
Study tracks 11–13. Determine how future prediction, physics and causal/counterfactual reasoning fit the state model.

## Phase 4 — Real-world domains
Study tracks 14, 15 and domain files. Integrate synthetic-media forensics and sports/general multimedia constraints.

## Phase 5 — Systems feasibility
Study tracks 16–18 and all runtime files. Build a quantified compute/latency model.

## Phase 6 — Benchmark and prototype
Implement the staged build and run the ablation matrix.

## Final output expected from research
1. Formal system definition.
2. State/evidence/hypothesis data model.
3. Architecture with module boundaries.
4. Candidate model families with benchmark evidence.
5. Confidence/calibration methodology.
6. Real-time compute budget.
7. Dataset/benchmark plan.
8. Prototype implementation sequence.
9. Failure-mode analysis.
10. Clear list of genuinely novel research contributions versus established techniques.

## Central thesis

A practical general multimedia reasoner should behave less like a repeatedly invoked VLM and more like a continuously updated belief system:

observation -> latent state -> prediction -> prediction error -> hypothesis -> verification -> calibrated belief

The 30 FPS requirement applies to maintaining this state continuously; expensive semantic reasoning is scheduled according to information value.
