# Synthetic Media Domain Architecture

## Purpose

Domain-specific handling for synthetic/AI-generated content — detection, provenance verification, and graceful degradation. Combines forensic detection with provenance checking to assess content authenticity.

---

## Interfaces

### Input/Output

```
SyntheticMediaInput {
  media_type:      Enum                     # IMAGE | VIDEO | AUDIO
  content:         Tensor | bytes
  metadata:        MediaMetadata
  suspected:       bool                     # user-suspected synthetic
}

SyntheticMediaOutput {
  authenticity:    Enum                     # AUTHENTIC | SYNTHETIC | MANIPULATED | INCONCLUSIVE
  confidence:      float
  detection_method: string                  # "ensemble" | "provenance" | "combined"
  provenance:      ProvenanceOutput
  forensic:        DeepfakeOutput
  evidence:        string[]
  recommendation:  string                   # "trust" | "flag_for_review" | "reject"
}
```

---

## Data Contracts

### Decision Tree

```
1. Check provenance (C2PA/SynthID)
   - Valid signature + trusted issuer → AUTHENTIC (high confidence)
   - No manifest → continue to detection
   - Invalid manifest → SYNTHETIC (medium confidence)

2. Run forensic ensemble
   - Ensemble consensus > threshold → SYNTHETIC/AUTHENTIC
   - No consensus → INCONCLUSIVE

3. Combine signals
   - Provenance + forensic agree → high confidence
   - Provenance + forensic disagree → INCONCLUSIVE, flag for review
```

### C2PA 2.4 Adoption (2026):

| Property | Value |
|---|---|
| CAI members | 4,000+ |
| C2PA members | 500+ |
| Native recognition | 22% of platforms |
| Credential preservation | 60% on upload |
| Security issues | 7 found (IACR 2026) |

### SynthID Scale:

| Property | Value |
|---|---|
| Images watermarked | 100B+ |
| Audio watermarked | 60,000 years |
| Verification checks | 50M in Gemini |
| Robustness | Survives re-encoding, screenshots, ~20% cropping |
| Cross-provider | Fragmented (no universal detection) |

---

## Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Provenance check | 50–200ms | C2PA + SynthID |
| Forensic ensemble | 200–2000ms | Multiple detectors |
| Decision fusion | 5–10ms | |
| **Total** | **250–2200ms** | Async |

---

## Dependencies

### Upstream
- `08_forensics/01_deepfake_detection.md` — Ensemble detection
- `08_forensics/02_provenance_c2pa.md` — Provenance verification
- `08_forensics/03_audio_forensics.md` — Audio detection

### Downstream
- `06_calibration/04_claim_output.md` — Authenticity claims

---

## Reality Check 2026

### RA-Bench Central Finding:
- **Automated detection is triage, not adjudication.**
- No single detector generalizes across all generators.
- Public-reference rankings do NOT transfer.
- HumanProof subset: detectors average 47.5% AUC (worse than coin flip).

### Social Dissemination Impact:
- Fine-tuned MLLMs: 46.0% → 1.4% FakeR after H.264 + spatial downsampling + 8fps.
- **Compression chain destroys detection signals.**

### Design Rules:
1. **Provenance first** — cheapest, most reliable signal.
2. **Ensemble detection** — 4-6 diverse detectors, logit-space fusion.
3. **Degrade gracefully** — if inconclusive, report uncertainty.
4. **Test under real-world conditions** — compression, resizing, social media upload.
5. **Never auto-reject** — flag for human review.
