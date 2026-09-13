# 1. Fundamental theory of multimodal reasoning

## Objective
Define exactly what the system means by observing, inferring, predicting, hypothesizing and explaining.

## Research questions
- How should observation and inference be represented so the model cannot silently upgrade an inference into a fact?
- Which formalism best represents partial observability: Bayesian filtering, POMDP belief states, factor graphs, latent-state neural models, or hybrid neuro-symbolic systems?
- How should abductive reasoning be implemented: candidate generation, Bayesian model selection, program search, or verifier-guided search?
- How should causal claims be separated from temporal correlation?
- Can a single state representation support image, video, audio and metadata?
- What does calibrated uncertainty mean at the level of an individual claim?

## Core mathematical objects
- Latent state S_t
- Observation O_t
- Belief b_t(S)=P(S_t|O_1:t)
- Hypothesis H_i
- Evidence E_j
- Posterior P(H_i|E_1:n)
- Predictive distribution P(S_t+k|S_1:t)

## Deliverable
A formal ontology and inference semantics for the whole system.

---

## REALITY CHECK 2026: What is actually achievable

### ✅ PROVEN / OFF-THE-SHELF:
- **Belief state formalism:** Bayesian filtering / POMDP belief states for tracked entities are textbook-standard in robotics and self-driving. Kalman/extended Kalman + particle filter libraries are production-grade.
- **Factor graphs:** GTSAM, Ceres, PyTorch3D all have working, optimized implementations.

### ⚠️ PLAUSIBLE, ENGINEERING REQUIRED:
- **Single state representation unifying image/video/audio/metadata:** Design the state schema; this is systems work. Do NOT attempt a "learned universal state embedding" end-to-end from scratch — use typed objects (entities/trajectories/events) with embeddings attached as fields.
- **Abductive reasoning at scale:** Candidate generation = top-K from a small closed ontology (not "all possible hypotheses"). Bayesian model selection = compute posterior over K=3–5 hypotheses. This is computationally trivial; the hard part is feature engineering for the likelihood.

### ❌ NOT FEASIBLE IN 2026 (set aside, do not overpromise):
- **"General" abductive reasoning over open-world combinatorial hypothesis spaces** (program search, hypothesis enumeration over all concepts): intractable for any real-time system; LLMs hallucinate candidates; hypothesis pruning heuristics break under distribution shift.
- **"Pure" learned latent state that generalizes to unseen domains without schema**: World models (Dreamer, etc.) work in closed simulators; for in-the-wild video they collapse to memorization or blur. Hybrid is mandatory.

### Recommended practical choice:
Use a HYBRID neuro-symbolic formalism:
- **Typed symbolic state** (entity objects with fields, trajectory structs, struct-like Events) for the fast 30 FPS path.
- **Learned embeddings and likelihoods** (detector scores, VLM consistency checks) attached to symbolic nodes.
- **Bayesian posterior updates** over a bounded (≤5) hypothesis set.
This avoids the worst pitfalls of both pure-symbolic (too brittle) and pure-neural (too hallucinatory) approaches.
