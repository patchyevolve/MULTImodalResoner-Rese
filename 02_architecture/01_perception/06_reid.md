# Re-Identification Architecture

## Purpose

Match detected persons/objects across non-overlapping camera views or after long occlusion. Provides appearance-based features for re-linking tracks that were lost. Event-triggered, not per-frame.

---

## Interfaces

### Input

```
ReIDInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  crop:            Tensor[H, W, 3] uint8    # person crop from detection
  entity_id:       string                   # current track ID
  query_type:      Enum                     # GALLERY_MATCH | ENROLL | VERIFY
}
```

### Output

```
ReIDOutput {
  entity_id:       string
  embedding:       [512]float               # appearance feature vector
  matched_id:      string                   # best gallery match (if GALLERY_MATCH)
  match_score:     float                    # similarity score
  rank:            int                      # rank in gallery sorted by similarity
  candidates:      ReIDCandidate[]          # top-K matches
}

ReIDCandidate {
  gallery_id:      string
  score:           float
  last_seen_ns:    uint64
}
```

### API

```
extract_embedding(input: ReIDInput, config: ReIDConfig) -> ReIDOutput
match_gallery(embedding: [512]float, top_k: int) -> ReIDCandidate[]
enroll(entity_id: string, embedding: [512]float) -> bool
```

---

## Data Contracts

### Model Selection

| Model | Rank-1 (Market1501) | Latency | Notes |
|---|---|---|---|
| OSNet | 95.6% | 2–5ms | Lightweight, good |
| TransReID | 97.0% | 20–50ms | Transformer, slow |
| CLIP-based | 94.0% | 10–20ms | Cross-modal capable |

### Runtime Configuration

```
ReIDConfig {
  model:           string                   # "OSNet" | "TransReID"
  gallery_size:    int                      # max gallery entries, default 10000
  similarity_threshold: float               # default 0.7
  top_k:           int                      # default 10
  device:          string
}
```

---

## Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Crop preprocess | 0.2–0.5 ms | Resize to model input |
| Feature extraction | 2–5 ms (OSNet) | GPU inference |
| Gallery search | 1–5 ms | Approximate nearest neighbor |
| **Total** | **3–10 ms** | Event-triggered |

---

## Dependencies

### Upstream
- `01_perception/01_detection.md` — Person crops
- `01_perception/03_tracking.md` — Entity IDs, track re-initiation

### Downstream
- `04_memory/03_long_term.md` — Gallery storage
- `03_state/01_world_state.md` — Entity identity updates

---

## Reality Check 2026

### Production Considerations:
- OSNet is sufficient for most single-camera scenarios.
- TransReID needed for cross-camera, but 10× slower.
- Gallery must be periodically pruned (old entries, low-confidence).
- Appearance features degrade under significant lighting/viewpoint change.
- Use in combination with kinematic features (velocity, trajectory) for robust matching.
