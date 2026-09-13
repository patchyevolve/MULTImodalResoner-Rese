# Working Memory Architecture

## Purpose

Maintain the active hypothesis context — what the system is currently investigating, what evidence it has gathered, and what the current best explanations are. This is the "working set" for the reasoning engine.

---

## Interfaces

### Input/Output

```
WorkingMemory {
  active_hypotheses: HypothesisSet          # current top-K hypotheses
  evidence_buffer:   Evidence[]             # recently gathered evidence
  query_context:     QueryContext           # current user query or event context
  reasoning_trace:   ReasoningStep[]        # chain of reasoning so far
}

HypothesisSet {
  hypotheses:       Hypothesis[]
  ranking_timestamp: uint64
  conformal_sets:   Map[float, string[]]    # alpha → prediction set
}

Hypothesis {
  id:               string
  claim:            string
  status:           Enum                     # CANDIDATE | SUPPORTED | REFUTED | INCONCLUSIVE
  support:          Evidence[]
  contradictions:   Evidence[]
  confidence:       ConfidenceDecomposition
  posterior:        float
  rank:             int
  age_ms:           float                    # time since creation
}

ReasoningStep {
  step_id:          int
  method:           string                   # "rule_check" | "vlm_query" | "trajectory_check"
  input:            string                   # what was asked
  output:           string                   # what was found
  confidence:       float
  timestamp_ns:     uint64
  latency_ms:       float
}
```

### API

```
add_hypothesis(hyp: Hypothesis) -> void
update_hypothesis(id: string, update: HypothesisUpdate) -> void
add_evidence(evidence: Evidence) -> void
get_active_set() -> HypothesisSet
get_hypothesis(id: string) -> Hypothesis
add_reasoning_step(step: ReasoningStep) -> void
clear() -> void
```

---

## Data Contracts

### Working Memory Size

```
WorkingMemoryConfig {
  max_hypotheses:  int                      # default 10
  max_evidence:    int                      # default 100
  max_reasoning_steps: int                  # default 50
  hypothesis_ttl_ms: int                    # default 30000 (30s), then promote or discard
  evidence_ttl_ms: int                      # default 10000 (10s)
}
```

### Lifecycle

1. **Event triggers hypothesis generation:** Top-3 to top-5 candidate hypotheses created.
2. **Evidence accumulates:** Each new observation/evidence added to buffer, linked to hypotheses.
3. **Ranking updates:** Posterior updated as evidence accumulates.
4. **Decision:** If top hypothesis posterior > threshold → promote to "supported" claim. If all < threshold → "inconclusive."
5. **Cleanup:** TTL-based eviction. Old hypotheses demoted, old evidence archived.

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Add hypothesis | 0.1ms | Hash map insert |
| Add evidence | 0.05ms | Array append |
| Update posterior | 0.5ms | Bayesian update |
| Get active set | 0.01ms | Read pointer |
| **Total** | **<1ms** | |

---

## Dependencies

### Upstream
- `05_reasoning/01_hypothesis_engine.md` — Creates hypotheses
- `03_state/04_event_detection.md` — Triggers hypothesis generation

### Downstream
- `05_reasoning/02_fast_verifier.md` — Reads hypotheses for verification
- `05_reasoning/03_deep_vlm_reasoner.md` — Reads hypotheses for deep reasoning
- `06_calibration/04_claim_output.md` — Reads for final claim generation

---

## Reality Check 2026

### Hypothesis Set Size:
- SoccerNet 2026: best action anticipation achieves only 24.08% avg mAP at 5s.
- Even SOTA models are uncertain about future events.
- Maintain top-3 to top-5 hypotheses, not top-1. "Active falsification" = check competing hypotheses.

### Conformal Prediction Sets:
- At α=0.05: typically 1-2 hypotheses in prediction set.
- At α=0.10: typically just the top hypothesis.
- Report `conformal_prediction_set` in every claim output.
