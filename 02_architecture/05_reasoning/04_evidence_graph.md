# Evidence Graph Architecture

## Purpose

Maintain a typed graph of evidence nodes and their relationships to hypotheses. Tracks which evidence supports or contradicts which hypotheses, enabling transparent reasoning and audit trails.

---

## Interfaces

### Input/Output

```
EvidenceGraph {
  nodes:           EvidenceNode[]
  edges:           EvidenceEdge[]
  last_updated:    uint64
}

EvidenceNode {
  id:              string
  type:            Enum                     # OBSERVATION | INFERENCE | RULE | VLM_OUTPUT
  claim:           string
  confidence:      float
  source:          string[]                 # model IDs that produced this
  timestamp_ns:    uint64
  metadata:        Map[string, Any]
}

EvidenceEdge {
  source:          string                   # evidence node ID
  target:          string                   # hypothesis node ID
  relation:        Enum                     # SUPPORTS | CONTRADICTS | ENABLES | WEAKENS
  weight:          float                    # log-likelihood ratio
  weight_estimator: string                  # how weight was computed
}
```

### API

```
add_node(node: EvidenceNode) -> string
add_edge(edge: EvidenceEdge) -> void
remove_node(node_id: string) -> bool
get_support(hypothesis_id: string) -> EvidenceNode[]
get_contradictions(hypothesis_id: string) -> EvidenceNode[]
get_graph(hypothesis_id: string, depth: int) -> EvidenceGraph
compute_posterior(hypothesis_id: string) -> float
```

---

## Data Contracts

### Edge Weight Calculation

```
weight_llr = log(P(evidence | hypothesis) / P(evidence | ¬hypothesis))

Where:
  P(evidence | hypothesis): true positive rate of evidence
  P(evidence | ¬hypothesis): false positive rate of evidence

Calibrated via: split conformal on held-out data
```

### Graph Size Limits

```
EvidenceGraphConfig {
  max_nodes:       int                      # default 1000
  max_edges_per_hypothesis: int             # default 50
  node_ttl_ms:     int                      # default 60000 (1 minute)
  garbage_collection_interval_ms: int       # default 10000
}
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Add node | 0.1–0.5ms | Hash map insert |
| Add edge | 0.05–0.1ms | |
| Get support | 0.5–2ms | Graph traversal |
| Compute posterior | 1–5ms | Weighted sum |
| **Total** | **1–5ms** | |

---

## Dependencies

### Upstream
- `05_reasoning/01_hypothesis_engine.md` — Hypothesis nodes
- `05_reasoning/02_fast_verifier.md` — Verification evidence
- `05_reasoning/03_deep_vlm_reasoner.md` — VLM evidence

### Downstream
- `06_calibration/04_claim_output.md` — Evidence graph for claims
- `04_memory/04_episodic_memory.md` — Graph snapshots for episodes

---

## Reality Check 2026

### DSFlash (CVPR 2026):
- 56 FPS (18ms latency) for panoptic scene graph generation on RTX 3090.
- Scene graphs as production-ready representations.

### MECD+ (CAUSEAL 2026):
- Causal graph with 3.94 mean false relations — best precision/recall.
- Graph-to-Frame RAG retrieves evidence by traversing graph edges, not raw frame similarity.

### Practical Notes:
- Evidence graph is a **working data structure**, not a persistent store.
- Periodically snapshot to long-term memory for audit trails.
- Graph traversal depth should be bounded (max 3 hops) for latency.
