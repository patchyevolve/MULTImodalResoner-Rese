# Instance Segmentation Architecture

## Purpose

Produce pixel-level masks for detected objects. Used when precise object boundaries matter — occlusion reasoning, spatial relationship analysis, and fine-grained visual understanding.

---

## Interfaces

### Input

```
SegmentationInput {
  frame_id:        uint64
  timestamp_ns:    uint64
  image:           Tensor[H, W, 3] uint8
  detections:      DetectionBatch            # optional: prompt-based segmentation
  prompt_type:     Enum                     # BOX | POINT | TEXT | EVERYTHING
}
```

### Output

```
SegmentationBatch {
  frame_id:        uint64
  masks:           MaskResult[]
  model_id:        string
  inference_ms:    float
}

MaskResult {
  entity_id:       string
  mask_rle:        string                   # run-length encoded binary mask
  mask_area:       int                      # pixel count
  mask_iou_score:  float                    # quality score
  bbox_overlap:    float                    # IoU with input bbox
}
```

### API

```
segment_objects(input: SegmentationInput, config: SegConfig) -> SegmentationBatch
segment_everything(input: SegmentationInput, config: SegConfig) -> SegmentationBatch
```

---

## Data Contracts

### Model Selection

| Model | AP50:95 (Mask) | Latency T4 | Params | Use Case |
|---|---|---|---|---|
| RF-DETR-Seg-S | 43.1 | 4.4ms | 33.7M | Fast, closed-set |
| RF-DETR-Seg-M | 45.3 | 5.9ms | 35.7M | Balanced |
| RF-DETR-Seg-L | 47.1 | 8.8ms | 36.2M | High accuracy |
| YOLO26-M-Seg | 44.0 | 6.32ms | 23.6M | CPU-friendly |
| SAM 3 | 47.0 (LVIS) | 30ms (H200) | 848M | Open-vocabulary |
| SAM 3.1 | 47.0+ | ~16fps (H100) | 848M | Multi-object (16/pass) |

### Runtime Configuration

```
SegConfig {
  model:           string                   # "RF-DETR-Seg-S" | "SAM3" | ...
  prompt_type:     Enum                     # BOX (from detections)
  mask_threshold:  float                    # default 0.5
  use_sam3:        bool                     # true for open-vocabulary
  max_masks:       int                      # default 100
  device:          string
}
```

---

## Latency Budget

| Stage | Budget | Notes |
|---|---|---|
| Crop + preprocess | 0.3–0.5 ms | |
| RF-DETR-Seg inference | 4.4–8.8 ms | Depends on model |
| Mask decode | 0.5–1 ms | RLE encoding |
| **Total (RF-DETR-Seg)** | **5–10 ms** | Async path, 5–10 Hz |
| SAM 3 prompt-based | 30ms (H200) | Async, event-triggered |
| SAM 3 everything | 2921ms (3090) | NOT real-time, batch only |

---

## Dependencies

### Upstream
- `01_perception/01_detection.md` — Bounding boxes for prompt-based segmentation
- `03_state/01_world_state.md` — Entity regions

### Downstream
- `03_state/01_world_state.md` — Pixel-level entity masks
- `05_reasoning/03_deep_vlm_reasoner.md` — Visual context
- `08_forensics/01_deepfake_detection.md` — Spatial artifact analysis

---

## Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Mask quality low | score < threshold | Fall back to bbox only |
| SAM 3 OOM | CUDA OOM | Reduce prompts, process batch |
| Overlapping masks | IoU > threshold | Keep higher-score mask |
| Slow SAM 3 | Latency >100ms | Use RF-DETR-Seg for speed |

---

## Reality Check 2026

### RF-DETR-Seg vs YOLO26-Seg (from benchmark_report_2026.md):
- RF-DETR-Seg leads at every matched latency tier.
- RF-DETR-Seg-M (45.3 AP, 5.9ms) beats YOLO26-M-Seg (44.0, 6.32ms).
- RF-DETR-Seg-2XL reaches 49.9 AP — highest accuracy available.

### SAM 3 (open-vocabulary, heavy):
- 30ms per image with 100+ objects on H200 (prompt-based).
- "Everything mode" is 100× slower — batch processing only.
- SAM 3.1 adds Object Multiplex: 16 objects per pass, 7× faster throughput.
- Use SAM 3 only when open-vocabulary segmentation is required.

### Decision:
- **Real-time (30 FPS):** RF-DETR-Seg-S or skip segmentation entirely.
- **Async (5–10 Hz):** RF-DETR-Seg-M or L.
- **Open-vocabulary:** SAM 3 (prompt-based, event-triggered).
- **Batch processing:** SAM 3 everything mode.
