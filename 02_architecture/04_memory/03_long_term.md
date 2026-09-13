# Long-Term Memory Architecture

## Purpose

Persist processed state, entity profiles, and event summaries beyond the real-time window. Provides retrieval for: cross-session entity re-identification, historical queries ("show all goals from this player"), and episodic summarization.

---

## Interfaces

### Input/Output

```
LongTermStore {
  entity_profiles: VectorDB                  # entity embeddings for retrieval
  event_summaries: EventDB                  # event summaries by time/type
  video_segments:  SegmentDB                # processed video chunks with metadata
}

EntityProfile {
  canonical_id:    string
  name:            string                   # if known
  attributes:      Map[string, Any]         # persistent attributes
  embedding:       [512]float               # average appearance embedding
  event_count:     int
  first_seen_ns:   uint64
  last_seen_ns:    uint64
  sessions:        SessionSummary[]
}

EventSummary {
  event_id:        string
  timestamp_ns:    uint64
  event_type:      string
  description:     string
  entities:        string[]
  confidence:      float
  video_ref:       string                   # pointer to video segment
  embedding:       [256]float               # for semantic retrieval
}
```

### API

```
store_entity(profile: EntityProfile) -> void
retrieve_entity(canonical_id: string) -> EntityProfile
search_entities(query_embedding: [512]float, top_k: int) -> EntityProfile[]
store_event(summary: EventSummary) -> void
search_events(query: string, time_range: [uint64, uint64]) -> EventSummary[]
store_segment(ref: string, metadata: Map[string, Any]) -> void
```

---

## Data Contracts

### Storage Backend

| Store | Backend | Latency | Notes |
|---|---|---|---|
| Entity profiles | SQLite + FAISS | 10–50ms | Small dataset, fast queries |
| Event summaries | SQLite | 5–20ms | Structured queries |
| Video segments | S3/minio + metadata DB | 50–200ms | Large objects, async |
| Embeddings | FAISS (IVF-PQ) | 10–30ms | Approximate NN search |

### Flush Policy

```
LongTermConfig {
  flush_interval_ms: int                    # default 5000 (5s)
  entity_flush_threshold: int               # flush after N entity updates, default 50
  event_flush_threshold: int                # flush after N events, default 20
  max_entity_profiles: int                  # default 10000
  max_event_summaries: int                  # default 100000
}
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Entity store/retrieve | 10–50ms | Async, not in critical path |
| Event store/retrieve | 5–20ms | Async |
| Embedding search | 10–30ms | FAISS NN |
| **Total (async)** | **10–50ms** | Never blocks real-time path |

---

## Dependencies

### Upstream
- `04_memory/01_short_term.md` — Periodic flush source
- `03_state/02_entity_tracker.md` — Entity data to persist

### Downstream
- `05_reasoning/03_deep_vlm_reasoner.md` — Historical context for reasoning
- `09_domains/01_sports_reasoning.md` — Player/team history

---

## Reality Check 2026

### Design Rules:
1. **Async only.** Never block the 30 FPS path for long-term writes/reads.
2. **Periodic flush.** Every 5 seconds, flush accumulated state to persistent store.
3. **Bounded size.** LRU eviction when limits are exceeded.
4. **Embedding quality.** Use OSNet for entity embeddings, CLIP for event embeddings.
