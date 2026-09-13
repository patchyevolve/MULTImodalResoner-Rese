# General Multimedia Architecture

## Purpose

Domain-agnostic reasoning for heterogeneous video sources — consumer videos, news clips, documentaries, UGC, screenshots, audio-video. Maintains the same epistemic architecture when domain-specific priors are weak or unavailable.

---

## Interfaces

### Input/Output

```
GeneralMultimediaInput {
  source_type:     Enum                     # CONSUMER | NEWS | DOCUMENTARY | UGC | SCREENSHOT | AUDIO_VIDEO
  world_state:     WorldStateSnapshot
  metadata:        MediaMetadata            # codec, resolution, duration, source
  domain_hints:    string[]                 # detected domain tags
}

MediaMetadata {
  codec:           string                   # "h264" | "h265" | "vp9" | "av1"
  resolution:      [2]int
  duration_ms:     int
  fps:             float
  audio_channels:  int
  source_url:      string
  upload_date:     uint64
}
```

---

## Data Contracts

### Domain Detection Rules

| Source Type | Indicators | Strategy |
|---|---|---|
| News | Chyron, ticker, anchor shot, lower-third | OCR-heavy, structured |
| Documentary | Narration, B-roll, slow pacing | Audio-heavy, temporal |
| UGC | Handheld, vertical, short | Camera motion, quick events |
| Consumer | Mixed, family events | General detection |
| Screenshot | Static, text-heavy | OCR-heavy, no motion |
| Audio-Video | Speech-dominant | Audio-heavy, AV sync |

### Video-MME-v2 Findings (2026):

| Dimension | Best Model | Human | Gap |
|---|---|---|---|
| Frames & Audio | Gemini-3-Pro | 91.1 | 41.1 |
| Temporal Understanding | Gemini-3-Pro | 87.9 | 42.5 |
| Complex Reasoning | Gemini-3-Pro | 88.9 | 48.3 |
| Action & Motion | All <30 | — | — |

- **41.3-point gap** between best model and human experts.
- All models score below 30 on Action & Motion and Physical World Reasoning.

### RD-MLDG Finding:
- Reasoning chains: 58.6% lower cross-domain divergence than visual features.
- MMD: 0.239 (visual) → 0.099 (reasoning chains).
- **Key insight:** Use reasoning chains for domain invariance, not visual features.

---

## Latency Budget

Same as base pipeline — no additional overhead for general domain.

---

## Dependencies

### Upstream
- All perception components
- `09_domains/03_synthetic_media.md` — If content may be synthetic

### Downstream
- `06_calibration/04_claim_output.md` — General claims

---

## Reality Check 2026

### OmniVChall (ICML 2026):
- Performance drops 5.71% to 9.32% from single-factor to compositional hallucination.
- Camera-based reasoning: ~38.57% accuracy drop (worst category).
- AI-generated videos: 9.31% larger accuracy drop than real-world.

### CrossVid (AAAI 2026):
- First cross-video reasoning benchmark.
- Gemini-2.5-Pro best at 50.4% — most MLLMs struggle with multi-video evidence.

### Design Rules:
1. **Detect domain first**, then apply domain-specific rules if available.
2. **Fall back to generic pipeline** if domain is unknown or mixed.
3. **Report domain confidence** — weak domain priors reduce claim confidence.
4. **Never assume domain.** Let the system discover it from content.
