# 8. Evidence representation

## Objective
Represent what supports and contradicts each claim.

## Candidate representations
- Scene graphs
- Temporal graphs
- Knowledge graphs
- Factor graphs
- Evidence graphs
- Hybrid neural-symbolic memory

## Each evidence item should store
- source modality
- timestamp or time range
- spatial region/entity
- producing model
- raw observation reference
- confidence/reliability
- support/contradiction relationship

## Deliverable
A canonical Evidence object and graph schema shared by perception, reasoners and UI.

---

## REALITY CHECK 2026

### ✅ PROVEN / ENGINEERING-ONLY:
- Scene graphs / temporal graphs with typed nodes and edges: Straightforward schema design. DSFlash (CVPR 2026) achieves **56 FPS** (18ms latency) on RTX 3090 for panoptic scene graph generation — proven real-time.
- Source provenance (modality, timestamp, model, spatial region): Mandatory fields; zero ambiguity.
- Evidence graph size = bounded, not global. Keep only evidence for the current top-K hypotheses + current event window. No unbounded memory growth.
- Event-level causal graphs: MECD+ (TPAMI 2026) constructs event-level causal graphs from video using Granger causality, achieving **3.94 false causal relations** per video (avg 12.31 total). Outperforms GPT-4o by 5.77%.
- Structured evidence extraction: Graph-to-Frame RAG (CVPR 2026) extracts entity triples `{id, name, role, attributes, confidence, frame_idx}`, action triples `{verb, subject, objects, preconditions, postconditions}`, and event triples `{title, actions, summary}` from video via MLLM.

### ⚠️ THINK CAREFULLY ABOUT SCHEMA:
- **Factor graphs vs. plain evidence graphs:** Factor graphs (which explicitly represent factors and can run BP inference) are mathematically nicer IF you plan to run loopy belief propagation for posterior updates. Plain evidence edges (supports/contradicts) are simpler for the 30 FPS path. Recommended hybrid: use plain support/contradict edges in the fast path; optionally build a factor graph snapshot only for deep reasoning events that need it.
- **Evidence weight semantics:** What does `weight: 0.82` MEAN? In 2026, if it's not calibrated to a real probability or log-likelihood ratio, it's a marketing number. Define `weight` explicitly as log-likelihood ratio from a calibration set: `log P(E|H) / P(E|not H)`. This lets you do actual Bayesian updates, AND it gives you audit semantics.

### ❌ AVOID UNTIL PROVEN:
- Knowledge graphs with 10M+ entity open-world linking as the base representation: Too slow, too much noise, entity linking error dominates any benefit. Use a small closed ontology for the domain (sports vocabulary) + open embeddings for unknowns.
- Storing the raw media bytes in the evidence object: Store references/offsets only. Raw media goes to disk ring buffer.

### CANONICAL SCHEMA for 2026 deployment:
```
Evidence {
  id: str (uuid)
  produced_at_ms: int (monotonic)
  source_modality: enum[vision, audio, text, metadata, cross_modal, rule_engine, physics, vlm_reasoning]
  source_window: {start_ms, end_ms}
  source_spatial: {entity_id, bbox_norm, joint_id?}  // can be null
  producing_model: str (model_name + version + checkpoint hash)
  producing_model_latency_ms: float
  target_claim_or_hypothesis_id: str
  relation: enum [supports, contradicts, irrelevant, undetermined]
  weight_llr: float (calibrated log-likelihood ratio, not arbitrary 0-1)
  weight_estimator: enum[calibration_set_estimate, conformal_quantile, heuristic, human_label]
  raw_value: any (e.g., detector score, ASR text, rule name)
  input_quality_flags: {occluded_frac, blur_score, compressed, missing_modality_triggered}
}
```
The `weight_llr` + `weight_estimator` fields make the graph auditable and mathematically usable — not just a pretty UI artifact.
