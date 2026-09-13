# 13. Causal and counterfactual reasoning

## Objective
Go beyond temporal association to explanations and alternative outcomes.

## Questions
- How can causal graphs be induced from video?
- How can event chains be grounded in temporal evidence?
- How should interventions be represented?
- Can latent world models simulate counterfactual outcomes?
- How do we prevent causal overclaiming?

## Deliverable
Causal-claim protocol and counterfactual reasoning benchmark.

---

## REALITY CHECK 2026

### ✅ PROVEN / DOABLE WITHIN CONSTRAINTS:
- **Causal claims grounded in domain rules:** Sports rulebooks, physics laws, contracts — if the domain explicitly encodes "A causes B under conditions C", then verify conditions C and emit causal claim. This is reliable and interpretable.
- **Causal claims grounded in physical discontinuity coincidence:** State of A changes discontinuously at t, state of B changes discontinuously at same t±Δt (well within temporal jitter of detectors), physics rules out alternative causes → emit causal claim with reduced confidence.
- **Event chains grounded in temporal evidence:** Can build chains of temporal_correlation; tag separately from causal. ENTER (Apr 2026) achieves **96.5% node precision** and **87.7% edge precision** on event graphs for VideoQA, with **92.0% graph sufficiency**.
- **Counterfactual reasoning with causal graphs:** CFGPT (CVPR 2026) uses two-stage post-training: cross-modal distillation from LLM causal reasoning to VLM, then RL with causal graph reward. DMC-CF benchmark (May 2026): **1,614 videos, 5,317 counterfactual questions** across 3 difficulty levels.

### ⚠️ PLAUSIBLE, with heavy caveats:
- **Inducing limited causal graphs from video in closed domains:** In soccer, with 22 players + ball, and a known event ontology, you can mine statistical associations + use domain rules to orient edges. This works for common patterns. Accuracy ~60–80% on top-1 graph; treat outputs as hypotheses.
- **Counterfactual simulation with differentiable physics:** If you have a 3D scene + entities + physics engine, you CAN do "what if" by changing parameters and re-simulating. This works for ballistics and simple interactions. It does NOT work for "what if this player had jumped higher" without a full biomechanical model and intent model (which we don't have).
- **Prevent causal overclaiming:** Protocol with strict rules + elevation checklist. This is a systems problem; enforcement works if you don't let the VLM emit causal claims unchecked.

### ❌ IMPOSSIBLE IN 2026 (SET ASIDE):
- **General causal graph induction from arbitrary open-world video:** Unsolved. Pure video observation without interventions, without domain priors → correlation is the ceiling.
- **Counterfactuals about human mental states ("he would have passed if he'd seen the defender"):** Unverifiable, hallucination-prone. VLMs invent plausible-sounding but incorrect counterfactuals 40–60% of the time for open-domain mental state reasoning.
- **Using LLM/VLM self-reported causal reasoning as ground truth:** LLMs are known for post-hoc rationalization. Their causal explanations feel right but are often wrong. Require independent evidence.

### CAUSAL-CLAIM PROTOCOL (mandatory for 2026 deployment):
A claim can be tagged `causal` ONLY if ALL of the following are true:
1. **Temporal precedence:** Cause event timestamp < effect event timestamp (within tolerance).
2. **Proximity:** Entities involved in cause/effect are in physical proximity OR domain rule explicitly allows action-at-a-distance.
3. **At least ONE grounding from:**
   - (a) Domain rule library has explicit causal rule "X → Y in context C" and C is satisfied, OR
   - (b) Physical discontinuity test: cause entity state changes > 2σ, effect entity changes > 2σ, within ±50–200 ms window, physics rejects ≥ 2 alternative explanations, OR
   - (c) Explicit intervention data exists (A/B test, replay).
4. **Counterfactual check not required, but if present:** Physics sim of "remove cause" produces a different outcome from "keep cause".
5. **Cross-verification:** Rule-grounded + physics-grounded agreement (if both apply) adds bonus.

### BENCHMARK DESIGN:
- **Do NOT benchmark "general causal reasoning" on open-world datasets.** It will be noisy and LLM-hallucination dominated.
- **Benchmark in a closed rule-based domain (sports):** Events with known rule-causal structure. Measure: precision@causal_claim_emitted, false_causal_rate, coverage (how many true causal events the system identifies, even if as temporal correlation).
- **Counterfactual benchmark:** Only for simple physics interactions that are simulable. Require ground-truth simulator + 3D scene labels.

### DELIVERABLE (2026 REALISTIC):
Causal-claim elevation protocol with checklist + domain-rule + physical-discontinuity grounding. Benchmark in sports. Do NOT over-promise open-world causal graphs.
