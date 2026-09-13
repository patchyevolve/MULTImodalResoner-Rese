# Episodic Memory Architecture

## Purpose

Store and retrieve event summaries as coherent episodes — sequences of related events that form a narrative. Provides the input for: long-term summarization, cross-episode queries, and report generation.

---

## Interfaces

### Input/Output

```
Episode {
  episode_id:      string
  start_ns:        uint64
  end_ns:          uint64
  event_ids:       string[]
  summary:         string                   # natural language summary
  entities:        string[]                 # involved entity IDs
  domain:          string                   # "sports" | "news" | "general"
  key_frames:      string[]                 # frame IDs for visual summary
  embedding:       [256]float               # episode embedding for retrieval
}

EpisodeQuery {
  query:           string                   # "show all goals in second half"
  time_range:      [uint64, uint64]
  entities:        string[]
  event_types:     string[]
  domain:          string
}
```

### API

```
create_episode(events: EventSummary[]) -> Episode
store_episode(episode: Episode) -> void
search_episodes(query: EpisodeQuery, top_k: int) -> Episode[]
summarize_episode(episode_id: string) -> string
```

---

## Data Contracts

### Episode Creation Rules

- **Temporal grouping:** Events within 30 seconds are candidates for same episode.
- **Entity grouping:** Events involving the same entities are grouped.
- **Domain grouping:** Same domain events are grouped.
- **Episode boundary:** Scene cut, domain switch, or >30s gap triggers new episode.

### Storage

```
EpisodicConfig {
  max_episodes:    int                      # default 1000
  episode_ttl_ms:  int                      # default 3600000 (1 hour)
  auto_summarize:  bool                     # auto-generate summaries
  summary_model:   string                   # "Qwen3-VL-30B" or rule-based
}
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Episode creation | 1–5ms | Event grouping logic |
| Episode store | 5–10ms | Async |
| Episode search | 10–30ms | FAISS retrieval |
| Episode summarization | 500–2000ms | VLM-based, async |
| **Total (non-summary)** | **5–30ms** | |

---

## Dependencies

### Upstream
- `04_memory/03_long_term.md` — Event summaries
- `03_state/04_event_detection.md` — Events

### Downstream
- User-facing queries and reports
- `05_reasoning/03_deep_vlm_reasoner.md` — Episode context

---

## Reality Check 2026

### Sports Episodic Memory:
- A "goal episode" includes: buildup (3-5 events), the goal event, celebration (1-2 events).
- A "foul episode": foul event, referee reaction, player reactions, free kick/penalty.
- Episodes enable queries like "show all counter-attack goals this match."

### General Episodic Memory:
- Scene-based segmentation: each scene is roughly one episode.
- Cross-episode queries: "all clips showing this person speaking."
- Summarization: VLM generates natural language summary of episode.
