# Hypothesis Engine Architecture

## Purpose

Generate, rank, and maintain competing hypotheses about what is happening. Converts observations into claims that can be verified, supported, or refuted. This is the bridge between perception and understanding.

---

## Interfaces

### Input

```
HypothesisInput {
  world_state:     WorldStateSnapshot
  event:           EventTrigger             # triggering event (if any)
  working_memory:  WorkingMemory            # current hypothesis context
  domain_rules:    DomainRule[]             # domain-specific generation rules
}
```

### Output

```
HypothesisSet {
  hypotheses:      Hypothesis[]
  generated_at:    uint64
  trigger_event:   EventTrigger
  ranking_method:  string                   # "bayesian" | "rule_based" | "vlm_assisted"
}

Hypothesis {
  id:              string
  claim:           string                   # "player_A_kicked_ball"
  type:            Enum                     # OBSERVED | INFERRED | PREDICTED | CAUSAL | COUNTERFACTUAL
  epistemic_status: Enum                    # CONFIRMED | PROBABLE | POSSIBLE | SPECULATIVE | UNKNOWN
  support:         Evidence[]
  contradictions:  Evidence[]
  confidence:      ConfidenceDecomposition
  posterior:       float
  rank:            int
  generation_method: string                 # "rule" | "pattern" | "vlm"
}
```

### API

```
generate_hypotheses(input: HypothesisInput) -> HypothesisSet
update_posterior(hyp_id: string, evidence: Evidence) -> Hypothesis
rank_hypotheses(set: HypothesisSet) -> HypothesisSet
merge_hypotheses(primary_id: string, secondary_id: string) -> Hypothesis
```

---

## Data Contracts

### Hypothesis Generation Methods

| Method | Latency | Coverage | Notes |
|---|---|---|---|
| Rule-based | 1–5ms | Domain-specific | Fast, high precision |
| Pattern matching | 5–10ms | Learned patterns | Moderate speed |
| VLM-assisted | 800–3500ms | Open-ended | Slow, broadest |

### Generation Rules (Sports Example)

```
Rule: "player_X_kicked_ball"
  Trigger: person proximity to ball + pose change (leg swing)
  Evidence: trajectory change of ball + person velocity
  Conflicting: "player_X_headed_ball" (different pose)

Rule: "goal_scored"
  Trigger: ball trajectory crosses goal line
  Evidence: ball position + goal dimensions
  Conflicting: "ball_hit_post" (ball rebounds)
```

### Ranking

```
Posterior = P(hypothesis | evidence) ∝ P(evidence | hypothesis) × P(hypothesis)

P(hypothesis): prior from rule frequency
P(evidence | hypothesis): likelihood from perception confidence
Updated via Bayesian inference as evidence accumulates
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Rule-based generation | 1–5ms | 10-20 rules |
| Pattern matching | 5–10ms | |
| Posterior update | 0.5ms | Per hypothesis |
| Ranking | 0.1ms | Sort by posterior |
| **Total (rule-based)** | **5–15ms** | Fast path |
| **Total (with VLM)** | **800–3500ms** | Async path |

---

## Dependencies

### Upstream
- `03_state/04_event_detection.md` — Event triggers
- `03_state/01_world_state.md` — Current state for hypothesis generation
- `04_memory/02_working_memory.md` — Active hypothesis context

### Downstream
- `05_reasoning/02_fast_verifier.md` — Quick verification
- `05_reasoning/03_deep_vlm_reasoner.md` — Deep verification
- `06_calibration/04_claim_output.md` — Final claim generation

---

## Reality Check 2026

### Hypothesis Set Size:
- Top-3 to top-5 is the computational sweet spot.
- Below minimum posterior threshold → drop immediately.
- SoccerNet 2026: 24.08% avg mAP at 5s — even SOTA models are uncertain about future events.
- "Active falsification" = check whether ANY competing hypothesis explains the data equally well.

### Claim Types (from ClaimFlow 2026):
- OBSERVED: directly visible, high confidence.
- INFERRED: derived from evidence, moderate confidence.
- PREDICTED: future state, lower confidence.
- CAUSAL: cause-effect relationship, requires strong evidence.
- COUNTERFACTUAL: "what if" — speculative, lowest confidence.
