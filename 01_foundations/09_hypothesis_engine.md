# 9. Hypothesis generation and verification

## Objective
Move from answer generation to explicit competing explanations.

## Research questions
- How many hypotheses should be maintained?
- How should candidates be generated efficiently?
- What constitutes evidence against a hypothesis?
- How should hypotheses be merged, split, or retired?
- Should the verifier be a VLM, symbolic engine, learned critic, or ensemble?

## Lifecycle
candidate -> supported -> strongly supported -> contradicted -> rejected -> unresolved

## Key capability
The system should actively try to falsify the current best explanation.

## Deliverable
Hypothesis API, verifier, contradiction detector, and ranking policy.

---

## REALITY CHECK 2026

### ✅ PROVEN / DIRECTLY IMPLEMENTABLE:
- Bounded hypothesis set: top-K=3 hypotheses per event. This is the only computationally tractable option.
- Ranking via calibrated posterior: Weight each hypothesis using log-likelihood ratios from evidence (see evidence schema). Normalize → posterior.
- Lifecycle state machine: candidate → supported → strongly supported → contradicted → rejected → unresolved. Straightforward.

### ⚠️ PLAUSIBLE, implement heuristically:
- **Candidate generation:** Closed-vocabulary ontology (e.g., soccer events = pass, shot, dribble, foul, throw_in, ..., other) → candidate hypotheses are top ontology classes. For open-domain, generate candidates via async VLM call (0.3 Hz) only.
- **Hypothesis scoring with Bayesian surprise:** SPIKE-RL (ICLR 2026) generates belief hypotheses using Video-LLM, scores via softmax over negative log-likelihoods, uses KL divergence as surprise signal. Achieves **68.2%** on FunQA surprise localization, **40.3%** hypothesis diversity (vs 33.5% baseline).
- **Hypothesis verification with graph reasoning:** MM-GoT (CVPR 2026W) uses multimodal Graph-of-Thoughts with verification signals (semantic consistency, spatial validity, attentional grounding). Achieves **+3.1 pp** over GoT baselines, up to **+6.9 pp** under high visual ambiguity; **22–24% fewer tokens**.
- **What constitutes evidence against a hypothesis (falsification):**
  1. Any direct evidence with relation = contradicts AND weight_llr > threshold.
  2. Leading hypothesis fails to explain ≥1 key observation that a competing hypothesis explains.
  3. Async VLM verifier pass explicitly flags incompatibility (use VLM only as an additional signal, not sole decider).
- **Merging/splitting hypotheses:** Merge if semantic similarity ≥ threshold AND supporting evidence sets overlap ≥ 60%. Split if a single hypothesis's evidence set naturally clusters into 2+ distinct time windows or modalities.
- **Verifier architecture:** Use lightweight heuristic checks (physics, rules) in 30 FPS path + async VLM-based critic on top-2 hypotheses only for event-triggered cases. Heuristic checks catch 60–80% of obvious contradictions.

### ❌ SPECULATIVE / SET ASIDE:
- **Unbounded, open-world hypothesis generation:** LLM-based candidate generation is hallucination-prone; candidates frequently have P(H)=0 under the dataset but look plausible to the LM. Closed-vocab ontology is safer for 2026.
- **"Active" falsification via agent-like search for counter-evidence:** Too slow, too unbounded. Top-2 contradiction checks + domain rule engine achieve 80% of the value at 1% of compute.

### RANKING POLICY for 2026:
1. **Bayesian posterior** (product of LLR evidence weights × prior) = primary score.
2. **Coverage penalty:** Hypothesis that explains only 30% of evidence gets penalized relative to one explaining 90%.
3. **Simplicity penalty (Occam):** If H1 requires 3 unobserved sub-hypotheses and H2 requires 0, prefer H2 by heuristic bonus.
4. **Contradiction count:** Number of contradictions subtracts multiplicative penalty.
5. **Freshness:** Evidence older than T seconds contributes with decay.

### HYPOTHESIS SET MANAGEMENT RULES:
- Max active hypotheses per event: 5.
- Minimum posterior for retention: 5% (drop anything below).
- If posterior spread < 10% (H1=38%, H2=32%, H3=30%): tag entire claim as `inconclusive` even if top-1 exists.
- A hypothesis elevates from candidate → supported only when: (a) posterior ≥ 60% AND (b) no contradiction weight_llr > +1.0 AND (c) posterior of H2 is ≤ half of H1.
- Do NOT emit claims that are `candidate`; only `supported`+ go to UI/API by default.
