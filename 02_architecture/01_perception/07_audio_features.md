# Audio Feature Extraction Architecture

## Purpose

Extract audio features from video streams — speech, music, environmental sounds, audio-visual synchronization. Provides audio modality for multimodal reasoning. Runs in parallel with vision pipeline.

---

## Interfaces

### Input

```
AudioInput {
  stream_id:       string
  timestamp_ns:    uint64
  audio_chunk:     Tensor[T] float32        # raw audio samples (16kHz mono)
  sample_rate:     int                      # default 16000
  duration_ms:     int                      # chunk duration, default 1000
}
```

### Output

```
AudioOutput {
  stream_id:       string
  timestamp_ns:    uint64
  features:        AudioFeatures
  inference_ms:    float
}

AudioFeatures {
  speech_text:     string                   # ASR transcription
  speech_confidence: float
  speaker_id:      string                   # speaker diarization
  energy_level:    float                    # 0-1, audio energy
  is_speech:       bool                     # speech activity detection
  environmental:   string[]                 # ["crowd", "whistle", "music"]
  sync_score:      float                    # audio-visual sync quality
  embedding:       [256]float               # audio embedding for retrieval
}
```

### API

```
extract_audio(input: AudioInput, config: AudioConfig) -> AudioOutput
detect_events(audio: AudioOutput) -> AudioEvent[]
```

---

## Data Contracts

### Component Selection

| Component | Model | Latency | Notes |
|---|---|---|---|
| ASR | Whisper-small | 50–100ms/chunk | Good accuracy |
| ASR (fast) | Whisper-tiny | 20–40ms/chunk | Lower accuracy |
| VAD | Silero-VAD | 1–2ms | Speech activity detection |
| Speaker ID | ECAPA-TDNN | 5–10ms | Speaker verification |
| Sound classification | YAMNet | 5–10ms | 521 classes |
| Audio embedding | CLAP | 20–40ms | Audio-language alignment |

---

## Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Audio decode | 1–2 ms | PCM from compressed |
| VAD | 1–2 ms | Speech/non-speech |
| ASR (if speech) | 50–100ms | Event-triggered |
| Sound classification | 5–10ms | Periodic |
| **Total (non-speech)** | **2–5 ms** | |
| **Total (speech)** | **50–100ms** | Async |

---

## Dependencies

### Upstream
- `10_infrastructure/01_data_schemas.md` — Audio format specs

### Downstream
- `02_fusion/01_multimodal_fusion.md` — Audio-visual fusion
- `02_fusion/03_cross_modal_alignment.md` — AV sync
- `03_state/01_world_state.md` — Audio events in world state

---

## Reality Check 2026

### Audio-Visual Sync:
- Video-MME-v2 shows text modality is critical for reasoning (+3.8 to +5.8 with subtitles).
- Audio features complement vision for: crowd noise (sports), speech (news), environmental (general).
- Cross-modal alignment score should be >0.8 for reliable fusion.

### Speech Deepfake Detection:
- Resemble AI: 98.1% accuracy, 0.33 RTF (Podonos 2026).
- RTCFake: online EER 13.79% vs offline 5.42% — real-time processing degrades detection.
- Cascaded distortion: AASIST clean EER 0.83% → after AFE pipeline: 47.11%.
