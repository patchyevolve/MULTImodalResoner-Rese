# Claim taxonomy

Every output should be tagged:

- `observed`
- `inferred`
- `predicted`
- `hypothesis`
- `causal`
- `counterfactual`
- `unknown`
- `inconclusive`

The UI and API should make this distinction visible. A claim's epistemic status is part of the result, not explanatory decoration.

---

## REALITY CHECK 2026: CLAIM TAXONOMY RESEARCH

### ClaimFlow (arXiv:2603.16073, 2026):
- 5-relation taxonomy grounded in argumentation theory (Toulmin, 2003).
- 1,617 NLP papers, 5,689 claims, 4,871 relations.
- Relations: BACKGROUND (58.2%), SUPPORT (~18%), EXTEND (13.4%), QUALIFY (5.4%), REFUTE (2.1%).
- 63.5% of claims never reused; only 11.1% ever challenged.

### SciLens — Multimodal Scientific Claim Verification (KDD 2026):
- Decomposes claims into central empirical atoms: comparison, number, scope, rank, arithmetic, visual, trend.
- Grounds atoms to modality-specific evidence witnesses.
- Entailment rules: soft qualifiers ("around"), hard logical ("all", "every"), statistical (p-values).
- Outperforms vanilla Qwen3-VL-30B by 3.2-8.3 points on claim verification.

### Claim Verification Reasoning Patterns (arXiv:2604.01657):
- 6 patterns: direct evidence, nuance/implication, absence of evidence, synthesis, scope mismatch, step-by-step.
- 5 error types: Lexical Overlap Bias, Overcautiousness (41.4%), Negation/Temporal (32.8%), Scope mismatch, Reasoning-chain errors.

### Epistemic Logic for Multimodal Reasoning:
- Standard S5: K_a φ = "agent a knows that φ". Possible worlds semantics.
- **Gap:** No 2026 paper bridges epistemic modal logic with VLM uncertainty. Open research problem.
- **Gap:** Claim taxonomy for multimodal reasoning remains nascent — SciLens and ClaimFlow address text-heavy claims; visual claim verification taxonomy is undeveloped.

### Practical Implementation:
- Each claim carries: `type` (observed/inferred/predicted/hypothesis/causal/counterfactual/unknown/inconclusive), `epistemic_status`, `calibration_method`, `prediction_set`, `staleness_ms`.
- Confidence decomposition: perception, temporal, motion, cross_modal_agreement, reasoning, calibrated — stored as components, not a single number.
