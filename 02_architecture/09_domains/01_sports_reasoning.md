# Sports Reasoning Architecture

## Purpose

Domain-specific reasoning for sports video — player actions, game events, tactical analysis, and rules enforcement. Combines generic perception with sport-specific knowledge to produce actionable claims.

---

## Interfaces

### Input/Output

```
SportsReasoningInput {
  world_state:     WorldStateSnapshot
  domain:          string                   # "soccer" | "basketball" | "tennis" | "cricket"
  game_state:      GameState
  rules:           SportRules
}

GameState {
  score:           [2]int                   # home, away
  period:          int
  time_remaining:  float                    # seconds
  possession:      string                   # team with ball
  field_position:  [2]float                 # normalized position on field
  active_players:  int                      # per team
}

SportRules {
  ruleset:         string                   # "FIFA" | "NBA" | "ITF" | "ICC"
  foul_conditions: Rule[]
  scoring_conditions: Rule[]
  offside_conditions: Rule[]
}
```

### Output

```
SportsClaim {
  claim:           string                   # "goal_scored_by_player_A"
  type:            string                   # "goal" | "foul" | "offside" | "pass" | "dribble"
  confidence:      float
  period:          int
  timestamp_in_game: float                  # seconds from start
  entities:        string[]                 # involved players
  replay_frames:   uint64[]                 # key frames for replay
  evidence:        Evidence[]
}
```

---

## Data Contracts

### Soccer Events (from SoccerNet 2026):

| Event | Trigger | Priority | VLM Required |
|---|---|---|---|
| Goal | Ball crosses line + trajectory change | High | Yes |
| Card | Referee pose + player action | High | Yes |
| Foul | Contact + fall + referee reaction | High | Yes |
| Offside | Trajectory + defensive line + timing | High | Yes |
| Corner/Kick | Ball trajectory + field position | Medium | No |
| Pass | Ball trajectory + player proximity | Low | No |

### SoccerNet 2026 Baseline:
- Action anticipation: 24.08% avg mAP at 5s (vs 16.76% baseline).
- At 2s: 18.18% (vs 13.00% baseline).
- **Key insight:** Anticipation is hard. Use for hypothesis generation, not claims.

### Basketball Events:

| Event | Trigger | Notes |
|---|---|---|
| Shot | Player pose + ball trajectory | High priority |
| Score | Ball through hoop + score change | High priority |
| Foul | Contact + referee signal | High priority |
| Turnover | Ball possession change | Medium priority |
| Substitution | Player swap | Low priority |

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Rule evaluation | 1–5ms | Domain rules |
| Event detection | 5–10ms | Sports-specific |
| Claim generation | 5–15ms | |
| VLM verification | 800–3500ms | Async, high-priority events |
| **Fast path total** | **10–30ms** | Rule-based |
| **Full path total** | **800–3500ms** | With VLM |

---

## Dependencies

### Upstream
- `01_perception/` — All perception components
- `03_state/` — World state, events
- `05_reasoning/` — Hypothesis engine

### Downstream
- `06_calibration/04_claim_output.md` — Sports claims
- `04_memory/04_episodic_memory.md` — Match episodes

---

## Reality Check 2026

### SoccerNet 2025/2026:
- **2025:** Team MJP — 21.23% avg mAP (winner).
- **2026:** FAANTRA-WS — 24.08% avg mAP (winner).
- **Improvement:** 3.85% absolute gain year-over-year.
- **Gap to human:** Still far from reliable autonomous officiating.

### NBA-AWS Partnership:
- Real-time player tracking and analytics.
- Hawk-Eye optical tracking in NBA arenas.
- Computer Vision Suite for automated game analytics.

### Practical Strategy:
- **Rule-based events** for 80% of situations (fast, deterministic).
- **VLM verification** for ambiguous/high-stakes events only.
- **Never auto-call fouls/offside** — flag for human review.
- **Report confidence explicitly** — sports claims need audit trails.
