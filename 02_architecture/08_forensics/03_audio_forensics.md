# Audio Forensics Architecture

## Purpose

Detect audio deepfakes — voice clones, synthetic speech, manipulated audio. Complements visual deepfake detection for audio-visual content. Event-triggered when speech is detected.

---

## Interfaces

### Input

```
AudioForensicsInput {
  audio_chunk:     Tensor[T] float32        # raw audio
  sample_rate:     int                      # default 16000
  speaker_id:      string                   # known speaker (if available)
  context:         string                   # "phone" | "video_call" | "broadcast" | "recording"
}
```

### Output

```
AudioForensicsOutput {
  is_synthetic:    bool
  confidence:      float
  detector_scores: Map[string, float]
  manipulation_type: string                 # "voice_clone" | "tts" | "splicing" | "enhancement"
  evidence:        string[]                 # detected artifacts
  latency_ms:      float
}
```

### API

```
detect_audio(input: AudioForensicsInput, config: AudioForensicsConfig) -> AudioForensicsOutput
```

---

## Data Contracts

### Detector Selection (from Podonos 2026)

| Detector | Accuracy | RTF | Notes |
|---|---|---|---|
| Resemble AI | 98.1% | 0.33 | Best in-domain |
| Teffic-Audio | 98.5% (EER 1.45%) | — | #1 Speech-DF-Arena |
| FlowFake (34K params) | 75.3% cross-domain | Low | Lightweight |
| XLSR+AASIST | 94.6% | — | Open-source |
| Wav2Vec2 (open) | 48–63% | — | Weak cross-domain |

### Cascaded Distortion Warning:

```
AASIST: clean EER 0.83% → after full AFE pipeline: 47.11%
XLSR+AASIST: clean EER 0.23% → after full AFE pipeline: 22.20%

Individual AFE modules cause limited degradation;
cascaded processing amplifies far beyond sum of individual effects.
```

### Configuration

```
AudioForensicsConfig {
  detectors:       string[]                 # ["Teffic-Audio", "FlowFake", "XLSR+AASIST"]
  context:         string                   # affects expected distortion level
  confidence_threshold: float               # default 0.7
  max_latency_ms:  int                      # default 500
}
```

---

## Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Audio preprocessing | 5–10ms | Feature extraction |
| Per-detector inference | 20–100ms each | |
| Ensemble fusion | 5–10ms | |
| **Total** | **100–500ms** | |

---

## Reality Check 2026

### Teffic-Audio (Jul 2026):
- 1.454% pooled EER across 14 test sets (#1 Speech-DF-Arena).
- Conformer encoder + multi-head attentive statistics pooling.
- Trained only on open-source data.

### FlowFake (ICML 2026 Workshop):
- 34K parameters (vs 300M for SSL Wav2vec2).
- Bimodal timescales: fast (0.1-0.3s spectral) + slow (1.5-4.5s prosodic).
- 75.29% cross-domain accuracy.

### RTCFake (ACL 2026):
- Online EER 13.79% vs offline 5.42% — real-time processing degrades detection.
- Video conferencing black-box processing distorts audio.
- Phoneme-level representations maintain better stability than frame-level.

### Design Rules:
1. **Context matters.** Phone audio has different distortion than broadcast.
2. **Cascaded distortion is the real problem.** Test with full AFE pipeline, not clean audio.
3. **Use phoneme-level features** for better robustness to processing.
4. **Report with audio forensics confidence** in AV claims.
