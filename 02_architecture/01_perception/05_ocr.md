# OCR Pipeline Architecture

## Purpose

Extract text from video frames — jerseys, scoreboards, signs, captions, overlays. Event-triggered, not per-frame. Used for sports analysis (player names, scores), news (chyron reading), and general scene understanding.

---

## Interfaces

### Input

```
OCRInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  image:           Tensor[H, W, 3] uint8
  roi:             [4]float                 # region of interest (normalized), optional
  trigger_reason:  string                   # "scene_cut" | "new_text_detected" | "periodic"
}
```

### Output

```
OCRBatch {
  frame_id:        uint64
  results:         OCRResult[]
  inference_ms:    float
}

OCRResult {
  text:            string
  bbox_norm:       [4]float
  confidence:      float
  language:        string
  text_type:       Enum                     # JERSEY | SCOREBOARD | SIGN | CAPTION | OVERLAY
}
```

### API

```
extract_text(input: OCRInput, config: OCRConfig) -> OCRBatch
```

---

## Data Contracts

### Model Selection

| Model | WER | Latency | Notes |
|---|---|---|---|
| PaddleOCR v4 | ~3% | 15–30ms | Best general-purpose |
| EasyOCR | ~5% | 20–40ms | Good multilingual |
| Tesseract 5 | ~8% | 10–20ms | Fast, less accurate |
| TrOCR (HuggingFace) | ~2% | 50–100ms | Transformer-based, slow |

### Runtime Configuration

```
OCRConfig {
  model:           string                   # "PaddleOCR-v4" | "EasyOCR"
  languages:       string[]                 # ["en"] or ["en", "ar"]
  min_confidence:  float                    # default 0.6
  max_text_regions: int                     # default 20
  use_gpu:         bool                     # default true
}
```

---

## Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Text region detection | 3–5 ms | EAST/DB detector |
| Text recognition | 10–20 ms | Per region |
| Postprocess (NMS, decode) | 1–2 ms | |
| **Total (2–5 regions)** | **15–40 ms** | Event-triggered only |

---

## Dependencies

### Upstream
- `01_perception/01_detection.md` — Scene change detection triggers OCR
- `03_state/04_event_detection.md` — Event-based triggering

### Downstream
- `03_state/01_world_state.md` — Text as entity attributes
- `05_reasoning/03_deep_vlm_reasoner.md` — Text context for reasoning

---

## Failure Modes

| Failure | Detection | Response |
|---|---|---|
| No text detected | Empty results | Log, continue |
| Low confidence | score < threshold | Report as uncertain |
| OCR on blurry frame | blur_score > 0.7 | Skip, wait for keyframe |
| Wrong language | Low confidence | Try next language model |

---

## Reality Check 2026

### PaddleOCR v4:
- ~3% WER on standard benchmarks.
- 15–30ms per frame on GPU.
- Handles rotation, perspective, curved text.
- Best choice for general-purpose video OCR.

### Sports-Specific OCR:
- Jersey numbers: high contrast, large, simpler — accuracy >95%.
- Scoreboards: structured format, can use template matching as fallback.
- Lower-third chyrons: consistent design, easy to detect.

### VLM as OCR Alternative:
- Qwen3-VL-30B-A3B can read text directly from frames.
- Higher latency (100ms+) but handles complex layouts, rotated text, partial occlusion.
- Use VLM for hard cases, traditional OCR for fast/easy cases.
