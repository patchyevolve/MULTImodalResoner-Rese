# 2026 Benchmark Report: RF-DETR, YOLO26, SAM 3/3.1, DETRPose

> **Date:** September 2026
> **Important Caveat:** Two different benchmarking methodologies exist:
> - **Roboflow SAB (Single Artifact Benchmarking):** Re-evaluates ALL models in-house with pycocotools on full COCO val2017 (5000 images). Latency measured on T4, TensorRT 10.4, CUDA 12.4, FP16, batch 1, with 200ms thermal buffer. Includes total latency (NMS included where applicable).
> - **Ultralytics Official:** Reports published benchmark numbers from the model authors. Uses TensorRT10 FP16, batch 1 on Amazon EC2 P4d. Provides both standard mAP (with NMS) and e2e mAP (NMS-free).
>
> **Both sets of numbers are presented below where they differ. They are NOT directly comparable without understanding the methodology.**

---

## 1. RF-DETR Detection Variants (N/S/M/L/XL/2XL)

**Source:** https://rfdetr.roboflow.com/latest/learn/benchmarks/ | Paper: arXiv:2511.09554 | GitHub: https://github.com/roboflow/rf-detr
**Latency Hardware:** NVIDIA T4, TensorRT 10.4, CUDA 12.4, FP16, batch size 1, 200ms thermal buffer
**Accuracy:** COCO val2017, pycocotools (Roboflow SAB re-evaluation)

| Model | COCO AP50 | COCO AP50:95 | RF100-VL AP50 | RF100-VL AP50:95 | Latency T4 TRT FP16 (ms) | Params (M) | Resolution | License |
|-------|-----------|-------------|---------------|-----------------|--------------------------|-----------|------------|---------|
| RF-DETR-N | 67.6 | 48.4 | 85.0 | 57.7 | 2.3 | 30.5 | 384x384 | Apache 2.0 |
| RF-DETR-S | 72.1 | 53.0 | 86.7 | 60.2 | 3.5 | 32.1 | 512x512 | Apache 2.0 |
| RF-DETR-M | 73.6 | 54.7 | 87.4 | 61.2 | 4.4 | 33.7 | 576x576 | Apache 2.0 |
| RF-DETR-L | 75.1 | 56.5 | 88.2 | 62.2 | 6.8 | 33.9 | 704x704 | Apache 2.0 |
| RF-DETR-XL | 77.4 | 58.6 | 88.5 | 62.9 | 11.5 | 126.4 | 700x700 | PML 1.0 |
| RF-DETR-2XL | 78.5 | 60.1 | 89.0 | 63.2 | 17.2 | 126.9 | 880x880 | PML 1.0 |

### RF-DETR on RTX 4090

**Source:** https://github.com/A-SHOJAEI/rf-detr-detection (third-party benchmark)
**Hardware:** NVIDIA RTX 4090 (24GB VRAM), CUDA 12.4, PyTorch 2.5.1

| Configuration | RF-DETR-Large (704px) Latency | FPS |
|--------------|-------------------------------|-----|
| Default (FP32) | 9.5 ms | 106 |
| JIT Optimized (FP32) | 7.8 ms | 128 |
| JIT Optimized (FP16) | 6.1 ms | 163 |

**Source:** https://github.com/khatami-mehrdad/rf-detr-trt-opti (TensorRT optimization)
**Hardware:** NVIDIA RTX 5090, TensorRT FP16

| Configuration | RF-DETR-Large (704px) | FPS |
|--------------|----------------------|-----|
| PyTorch FP32 | 14.5 ms | 69 |
| TRT FP32 | 3.25 ms | 308 |
| TRT FP16 | 1.33 ms | 753 |
| TRT FP16 + GPU Preprocess | 1.56 ms (end-to-end) | 641 |

### RF-DETR on A100

**No official A100 latency numbers are published by Roboflow.** The official benchmarks are T4-only. Third-party reports exist but are not from Roboflow's standardized methodology.

---

## 2. RF-DETR Keypoint Variants

**Source:** https://rfdetr.roboflow.com/latest/learn/run/keypoints/ | https://blog.roboflow.com/real-time-keypoint-detection-with-rf-detr/
**Status:** Preview release (early-access, `RFDETRKeypointPreview`)
**Accuracy:** COCO val2017 person keypoints, AP50:95 (OKS-based)
**Latency Hardware:** NVIDIA T4, TensorRT FP16, batch size 1

### Default Preview Checkpoint (576x576)

| Model | COCO AP50:95 | Latency T4 TRT FP16 (ms) | Params (M) | Resolution | License |
|-------|-------------|--------------------------|-----------|------------|---------|
| RF-DETR Keypoint (Preview) | 71.8 | 9.7 | 40.7 | 576x576 | Apache 2.0 |

### RF-DETR Keypoint at Different Resolutions (same checkpoint, NAS-trained)

**Source:** https://blog.roboflow.com/real-time-keypoint-detection-with-rf-detr/

| Input Resolution | Keypoint AP | Latency T4 TRT FP16 (ms) |
|-----------------|-------------|--------------------------|
| 312x312 | 61.1 | 4.5 |
| 432x432 | 68.1 | 6.2 |
| 576x576 | 71.8 | 9.8 |
| 816x816 | 74.0 | 20.4 |
| 888x888 | 74.2 | 25.9 |

> Note: All resolution variants use the SAME checkpoint (weight-sharing NAS). Only input resolution changes, no retraining required.

---

## 3. YOLO26 Detection Variants (N/S/M/L/X)

### Ultralytics Official Benchmarks

**Source:** https://docs.ultralytics.com/models/yolo26 | Paper: arXiv:2606.03748
**Hardware:** Amazon EC2 P4d, TensorRT10 FP16, batch 1
**Accuracy:** COCO val2017

| Model | Size (px) | mAP val 50-95 | mAP val 50-95 (e2e) | CPU ONNX (ms) | T4 TensorRT10 (ms) | Params (M) | FLOPs (B) |
|-------|-----------|--------------|--------------------|--------------|--------------------|-----------|----------| 
| YOLO26n | 640 | 40.9 | 40.1 | 38.9 +/- 0.7 | 1.7 +/- 0.0 | 2.4 | 5.4 |
| YOLO26s | 640 | 48.6 | 47.8 | 87.2 +/- 0.9 | 2.5 +/- 0.0 | 9.5 | 20.7 |
| YOLO26m | 640 | 53.1 | 52.5 | 220.0 +/- 1.4 | 4.7 +/- 0.1 | 20.4 | 68.2 |
| YOLO26l | 640 | 55.0 | 54.4 | 286.2 +/- 2.0 | 6.2 +/- 0.2 | 24.8 | 86.4 |
| YOLO26x | 640 | 57.5 | 56.9 | 525.8 +/- 4.0 | 11.8 +/- 0.2 | 55.7 | 193.9 |

> CPU ONNX measured on Intel Xeon CPU @ 2.00 GHz. "e2e" = end-to-end NMS-free inference mode.

### Roboflow SAB Re-evaluation (Directly Comparable with RF-DETR)

**Source:** https://rfdetr.roboflow.com/latest/learn/benchmarks/
**Methodology:** Re-evaluated in-house, same protocol as RF-DETR (pycocotools, full val2017)

| Model | COCO AP50 | COCO AP50:95 | RF100-VL AP50 | RF100-VL AP50:95 | Latency T4 TRT (ms) | Params (M) | Resolution |
|-------|-----------|-------------|---------------|-----------------|--------------------|-----------|-----------| 
| YOLO26-N | 55.8 | 40.3 | 76.7 | 52.0 | 1.7 | 2.6 | 640x640 |
| YOLO26-S | 64.3 | 47.7 | 82.7 | 57.0 | 2.6 | 9.4 | 640x640 |
| YOLO26-M | 69.7 | 52.5 | 84.4 | 58.7 | 4.4 | 20.1 | 640x640 |
| YOLO26-L | 71.1 | 54.1 | 85.0 | 59.3 | 5.7 | 25.3 | 640x640 |
| YOLO26-X | 74.0 | 56.9 | 85.6 | 60.0 | 9.6 | 56.9 | 640x640 |

> **Note the mAP differences:** Roboflow SAB numbers differ from Ultralytics official because Roboflow re-evaluates all models using their standardized protocol. The AP50 column (NMS results) is higher; AP50:95 is the primary comparison metric.

---

## 4. YOLO26-Pose Variants

### Ultralytics Official Benchmarks

**Source:** https://docs.ultralytics.com/tasks/pose
**Hardware:** Amazon EC2 P4d, TensorRT10 FP16, batch 1
**Accuracy:** COCO Keypoints val2017

| Model | Size (px) | mAP pose 50-95 (e2e) | mAP pose 50 (e2e) | CPU ONNX (ms) | T4 TensorRT10 (ms) | Params (M) | FLOPs (B) |
|-------|-----------|---------------------|-------------------|--------------|--------------------|-----------|----------| 
| YOLO26n-pose | 640 | 57.2 | 83.3 | 40.3 +/- 0.5 | 1.8 +/- 0.0 | 2.9 | 7.5 |
| YOLO26s-pose | 640 | 63.0 | 86.6 | 85.3 +/- 0.9 | 2.7 +/- 0.0 | 10.4 | 23.9 |
| YOLO26m-pose | 640 | 68.8 | 89.6 | 218.0 +/- 1.5 | 5.0 +/- 0.1 | 21.5 | 73.1 |
| YOLO26l-pose | 640 | 70.4 | 90.5 | 275.4 +/- 2.4 | 6.5 +/- 0.1 | 25.9 | 91.3 |
| YOLO26x-pose | 640 | 71.6 | 91.6 | 565.4 +/- 3.0 | 12.2 +/- 0.2 | 57.6 | 201.7 |

### Roboflow SAB Re-evaluation (Directly Comparable with RF-DETR Keypoint)

**Source:** https://rfdetr.roboflow.com/latest/learn/benchmarks/

| Model | COCO AP50:95 | Latency T4 TRT FP16 (ms) | Params (M) |
|-------|-------------|--------------------------|-----------| 
| YOLO26-pose N | 55.9 | 1.9 | 2.9 |
| YOLO26-pose S | 62.0 | 2.7 | 10.4 |
| YOLO26-pose M | 68.0 | 4.6 | 21.5 |
| YOLO26-pose L | 69.2 | 5.9 | 25.9 |
| YOLO26-pose X | 71.0 | 9.8 | 57.6 |

### YOLO26-pose on Roboflow Inference API

**Source:** https://docs.roboflow.com/models/supported-models/yolo26

| Alias | Input Size | mAP50-95 | ONNX latency (ms) | TensorRT FP16 (ms) |
|-------|-----------|---------|-------------------|-------------------| 
| yolo26n-pose-640 | 640x640 | 57.2 | 3.8 | 2.3 |
| yolo26s-pose-640 | 640x640 | 63.0 | 5.1 | 3.3 |
| yolo26m-pose-640 | 640x640 | 68.8 | 9.0 | 4.6 |
| yolo26l-pose-640 | 640x640 | 70.4 | 11.2 | 5.8 |
| yolo26x-pose-640 | 640x640 | 71.6 | 19.1 | 8.3 |

---

## 5. RF-DETR-Seg Variants

**Source:** https://rfdetr.roboflow.com/latest/learn/benchmarks/
**Latency Hardware:** NVIDIA T4, TensorRT 10.4, CUDA 12.4, FP16, batch size 1
**Accuracy:** COCO val2017 instance segmentation

| Model | COCO AP50 | COCO AP50:95 (Mask) | Latency T4 TRT FP16 (ms) | Params (M) | Resolution | License |
|-------|-----------|--------------------|--------------------------|-----------|------------|---------| 
| RF-DETR-Seg-N | 63.0 | 40.3 | 3.4 | 33.6 | 312x312 | Apache 2.0 |
| RF-DETR-Seg-S | 66.2 | 43.1 | 4.4 | 33.7 | 384x384 | Apache 2.0 |
| RF-DETR-Seg-M | 68.4 | 45.3 | 5.9 | 35.7 | 432x432 | Apache 2.0 |
| RF-DETR-Seg-L | 70.5 | 47.1 | 8.8 | 36.2 | 504x504 | Apache 2.0 |
| RF-DETR-Seg-XL | 72.2 | 48.8 | 13.5 | 38.1 | 624x624 | Apache 2.0 |
| RF-DETR-Seg-2XL | 73.1 | 49.9 | 21.8 | 38.6 | 768x768 | Apache 2.0 |

### YOLO26-Seg (for comparison)

**Source:** https://rfdetr.roboflow.com/latest/learn/benchmarks/ (Roboflow SAB re-evaluation)

| Model | COCO AP50 | COCO AP50:95 (Mask) | Latency T4 TRT (ms) | Params (M) | Resolution |
|-------|-----------|--------------------|--------------------|-----------|-----------| 
| YOLO26-N-Seg | 54.3 | 34.7 | 2.31 | 2.7 | 640x640 |
| YOLO26-S-Seg | 62.4 | 40.2 | 3.47 | 10.4 | 640x640 |
| YOLO26-M-Seg | 67.8 | 44.0 | 6.32 | 23.6 | 640x640 |
| YOLO26-L-Seg | 69.8 | 45.5 | 7.58 | 28.0 | 640x640 |
| YOLO26-X-Seg | 71.6 | 46.8 | 12.92 | 62.8 | 640x640 |

---

## 6. Head-to-Head: RF-DETR vs YOLO26

**Source:** https://rfdetr.roboflow.com/latest/learn/benchmarks/ (all numbers re-evaluated by Roboflow SAB)

### Detection (COCO val2017)

| Latency Tier | RF-DETR Model | RF-DETR AP50:95 | RF-DETR T4 (ms) | YOLO26 Model | YOLO26 AP50:95 | YOLO26 T4 (ms) |
|-------------|--------------|----------------|----------------|-------------|--------------|--------------|
| ~2.3 ms | RF-DETR-N | 48.4 | 2.3 | YOLO26-N | 40.3 | 1.7 |
| ~3.5 ms | RF-DETR-S | 53.0 | 3.5 | YOLO26-S | 47.7 | 2.6 |
| ~4.5 ms | RF-DETR-M | 54.7 | 4.4 | YOLO26-M | 52.5 | 4.4 |
| ~6.5 ms | RF-DETR-L | 56.5 | 6.8 | YOLO26-L | 54.1 | 5.7 |
| ~11 ms | RF-DETR-XL | 58.6 | 11.5 | YOLO26-X | 56.9 | 9.6 |

### Keypoint Detection (COCO Keypoints val2017)

| Model | AP50:95 | Latency T4 TRT FP16 (ms) | Params (M) |
|-------|---------|--------------------------|-----------| 
| RF-DETR Keypoint (Preview) | 71.8 | 9.7 | 40.7 |
| YOLO26-pose X | 71.0 | 9.8 | 57.6 |

### Instance Segmentation (COCO val2017)

| Latency Tier | RF-DETR-Seg Model | RF-DETR-Seg AP50:95 | YOLO26-Seg Model | YOLO26-Seg AP50:95 |
|-------------|------------------|--------------------|-----------------|-------------------|
| ~3.5 ms | Seg-N | 40.3 | N-Seg | 34.7 |
| ~4.5 ms | Seg-S | 43.1 | S-Seg | 40.2 |
| ~6 ms | Seg-M | 45.3 | M-Seg | 44.0 |
| ~8 ms | Seg-L | 47.1 | L-Seg | 45.5 |
| ~13 ms | Seg-XL | 48.8 | X-Seg | 46.8 |

### Key Takeaways from Head-to-Head

1. **On GPU (T4):** RF-DETR achieves higher AP50:95 at every matched latency tier. RF-DETR-M (54.7, 4.4ms) beats YOLO26-M (52.5, 4.7ms). RF-DETR-L (56.5, 6.8ms) beats YOLO26-L (54.1, 5.7ms).
2. **At the smallest tier:** YOLO26-N is faster (1.7ms vs 2.3ms) but gives up 8.1 AP to RF-DETR-N.
3. **Peak accuracy:** RF-DETR-2XL reaches 60.1 AP50:95 — no YOLO variant comes close.
4. **On CPU:** YOLO26 wins decisively. RF-DETR has 30M+ params and no meaningful CPU path. YOLO26n at 38.9ms CPU ONNX is the only real-time option.
5. **RF100-VL (domain transfer):** RF-DETR leads across all tiers due to DINOv2 pretraining.

---

## 7. SAM 3 and SAM 3.1

**Source:** https://ai.meta.com/blog/segment-anything-model-3/ | https://github.com/facebookresearch/sam3
**Paper:** SAM 3 (November 2025), SAM 3.1 (March 27, 2026)

### SAM 3 Core Specs

| Property | Value |
|----------|-------|
| Parameters | 848M (some sources say 473.6M) |
| Model size | 3.45 GB |
| Architecture | Meta Perception Encoder + DETR detector + memory tracker |
| License | SAM 3 research license (gated on HuggingFace) |
| Python requirement | 3.12+ |
| PyTorch requirement | 2.7+ |

### SAM 3 Inference Speed

| Hardware | Latency | Notes |
|----------|---------|-------|
| H200 GPU | 30 ms | Single image with 100+ detected objects |
| RTX 4090 (TensorRT EP) | ~1.1s | ~600x500 image, FP32, ONNX Runtime + TensorRT EP |
| RTX 4090 (CUDA EP) | ~1.4s | Same setup, CUDA backend |
| RTX 3090 (PyTorch) | 3912 ms | Everything mode, FP16 (from third-party) |
| RTX PRO 6000 | 2921 ms | Ultralytics benchmark, torch 2.9.1 |

> **Important:** SAM 3's speed varies dramatically by prompt mode. The 30ms H200 figure is for a single image with text/point prompts. "Everything mode" (automatic mask generation) is orders of magnitude slower.

### SAM 3 Accuracy Benchmarks

| Benchmark | Metric | SAM 3 | Previous Best | Improvement |
|-----------|--------|-------|---------------|-------------|
| LVIS (zero-shot) | Mask AP | 47.0 | 38.5 | +22.1% |
| SA-Co/Gold | CGF1 | 65.0 | 34.3 (OWLv2) | +89.5% |
| COCO (zero-shot) | Box AP | 53.5 | 52.2 (T-Rex2) | +2.5% |
| MOSEv2 VOS | J&F | 60.1 | 47.9 (SAM 2.1) | +25.5% |
| DAVIS 2017 VOS | J&F | 92.0 | 90.7 (SAM 2.1) | +1.4% |
| LVOSv2 VOS | J&F | 88.2 | 79.6 (SAM 2.1) | +10.8% |
| SA-V VOS | J&F | 84.6 | 78.4 (SAM 2.1) | +7.9% |

### SAM 3.1 Improvements (March 2026)

| Feature | SAM 3 | SAM 3.1 |
|---------|-------|---------|
| Release date | November 2025 | March 27, 2026 |
| Objects per forward pass | 1 | Up to 16 (Object Multiplex) |
| Speed (medium objects, H100) | ~16 fps | ~32 fps |
| Speed (128 objects, H100) | Baseline | ~7x faster |
| Throughput (128 objects, H100) | 1.57 FPS | 11.46 FPS |
| torch.compile support | Basic | Enhanced, better op fusion |

### SAM 3.1 Benchmark Numbers

**VOS Benchmarks:**

| Benchmark | SAM 3 | SAM 3.1 | Delta |
|-----------|-------|---------|-------|
| MOSEv1 | 78.4 | 79.6 | +1.2 |
| DAVIS17 | 92.2 | 92.7 | +0.5 |
| LVOSv2 | 88.5 | 89.2 | +0.7 |
| SA-V val | 83.5 | 83.8 | +0.3 |
| SA-V test | 84.4 | 85.1 | +0.7 |
| YTVOS19 | 89.7 | 89.3 | -0.4 |
| MOSEv2 | 60.3 | 62.3 | +2.0 |

**SA-Co/VEval Benchmarks:**

| Benchmark | SAM 3 | SAM 3.1 | Delta |
|-----------|-------|---------|-------|
| SA-V cgF1 | 30.3 | 30.5 | +0.2 |
| SA-V pHOTA | 58.0 | 58.7 | +0.7 |
| YT-Temporal-1B cgF1 | 50.8 | 52.9 | +2.1 |
| YT-Temporal-1B pHOTA | 69.9 | 70.7 | +0.8 |

### SAM 3 Hardware Requirements

**Source:** https://www.spheron.network/blog/deploy-sam-3-gpu-cloud/

| Mode | Resolution | Prompts | Min VRAM | Recommended GPU |
|------|-----------|---------|----------|----------------|
| Image (single) | 1024px | 1-5 points | 12 GB | A100 80GB, H100 |
| Image (batch 8) | 1024px | per image | 22 GB | A100 80GB |
| Image (batch 32) | 1024px | per image | 40+ GB | H100 80GB |
| Video (5 min, 24fps) | 1080p | 1 mask | 28 GB | H100 80GB |
| Video (30 min, 24fps) | 1080p | 3 masks | 55 GB | H100 80GB |

### SAM 3 vs YOLO26-seg Speed Comparison

**Source:** https://docs.ultralytics.com/models/sam-3 (Ultralytics benchmark on RTX PRO 6000)

| Model | Size (MB) | Params (M) | Speed GPU (ms/im) |
|-------|----------|-----------|-------------------| 
| Meta SAM3 | 3450 | 473.6 | 2921 |
| Ultralytics YOLO26n-seg | 6.4 | 2.7 | 8.4 |

> SAM 3 is 347x slower than YOLO26n-seg. Different use cases entirely: SAM 3 is for open-vocabulary concept segmentation; YOLO26-seg is for closed-set real-time instance segmentation.

---

## 8. DETRPose

**Source:** arXiv:2506.13027 | GitHub: https://github.com/SebastianJanampa/DETRPose
**Status:** First real-time end-to-end transformer for multi-person pose estimation

### DETRPose Variants — COCO val2017

**Source:** GitHub README (two different hardware setups reported)

#### Setup 1: TensorRT 8.6.4 FP16, single Tesla V100

| Model | Backbone | Params | Latency V100 TRT8 FP16 (ms) | GFLOPs | AP | AP50 | AP75 |
|-------|----------|--------|---------------------------|--------|-----|------|------| 
| DETRPose-N | HGNetv2-B0 | 4.1M | 2.80 | 9.3 | 57.2 | 81.7 | 61.4 |
| DETRPose-S | HGNetv2-B0 | 11.5M | 4.99 | 33.1 | 67.0 | 87.6 | 72.8 |
| DETRPose-M | HGNetv2-B2 | 20.8M | 7.01 | 67.3 | 69.4 | 89.2 | 75.4 |
| DETRPose-L | HGNetv2-B4 | 32.8M | 9.50 | 107.1 | 72.5 | 90.6 | 79.0 |
| DETRPose-X | HGNetv2-B5 | 73.3M | 13.31 | 239.5 | 73.3 | 90.5 | 79.4 |

#### Setup 2: TensorRT 10.11 FP16, NVIDIA A10 (Lambda.ai)

**Source:** Paper Table 5 / Appendix

| Model | Params | Latency A10 TRT10 FP16 (ms) | GFLOPs | AP | AP50 | AP75 | APM | APL | AR |
|-------|--------|---------------------------|--------|-----|------|------|-----|-----|----| 
| DETRPose-N | 4.1M | 1.55 | 9 | 57.2 | 81.7 | 61.4 | 48.2 | 70.5 | 64.4 |
| DETRPose-S | 11.9M | 2.39 | 33 | 67.0 | 88.1 | 72.9 | 60.4 | 77.3 | 73.5 |
| DETRPose-M | 23.5M | 3.67 | 67 | 69.4 | 89.8 | 75.5 | 63.1 | 79.1 | 75.5 |
| DETRPose-L | 36.8M | 5.08 | 107 | 72.5 | 90.6 | 79.0 | 66.3 | 82.2 | 78.7 |
| DETRPose-X | 82.3M | 8.59 | 240 | 73.3 | 90.5 | 79.4 | 67.5 | 82.7 | 79.4 |

#### Setup 3: COCO test-dev, TensorRT 8.6.4 FP16, Tesla V100

| Model | Params | Latency V100 (ms) | AP | AP50 | AP75 | APM | APL | AR |
|-------|--------|-------------------|-----|------|------|-----|-----|----| 
| DETRPose-S | 11.5M | 2.39 | 66.5 | 87.4 | 72.5 | 59.8 | 76.8 | 73.0 |
| DETRPose-M | 20.8M | 3.47 | 68.5 | 88.6 | 74.6 | 62.1 | 78.3 | 74.6 |
| DETRPose-L | 32.8M | 4.66 | 71.2 | 91.2 | 78.1 | 65.8 | 79.9 | 78.1 |
| DETRPose-X | 73.3M | 7.36 | 72.2 | 91.4 | 79.3 | 67.0 | 80.6 | 78.8 |

### DETRPose on CrowdPose test

| Model | Params | Latency (ms) | AP | AP50 | AP75 | APE | APM | APH |
|-------|--------|-------------|-----|------|------|-----|-----|-----| 
| DETRPose-S | 11.5M | 2.35 | 67.4 | 88.6 | 72.9 | 74.7 | 68.1 | 59.3 |
| DETRPose-M | 20.7M | 3.37 | 72.0 | 91.0 | 77.8 | 78.6 | 72.6 | 64.5 |
| DETRPose-L | 32.7M | 4.52 | 73.3 | 91.6 | 79.4 | 79.5 | 74.0 | 66.1 |
| DETRPose-X | 73.3M | 7.11 | 75.1 | 92.1 | 81.3 | 81.3 | 75.7 | 68.1 |

### DETRPose vs YOLO-pose Comparison (COCO val, A10 TRT10 FP16)

**Source:** Paper Table 5

| Model | Params | Latency A10 (ms) | AP | AP50 | AP75 |
|-------|--------|-----------------|-----|------|------| 
| YOLOv8-N | 3.2M | 1.18 | 50.2 | 81.0 | 53.5 |
| YOLOv8-S | 11.2M | 2.24 | 63.2 | 87.8 | 69.5 |
| YOLOv8-M | 25.9M | 3.45 | 68.6 | 90.7 | 75.8 |
| YOLOv8-L | 43.7M | 4.82 | 70.2 | 91.1 | 77.8 |
| YOLOv8-X | 69.4M | 5.23 | 67.3 | 87.5 | 75.0 |
| YOLO11-N | 2.9M | 1.31 | 48.9 | 81.4 | 52.5 |
| YOLO11-S | 9.9M | 1.84 | 57.5 | 86.0 | 61.5 |
| YOLO11-M | 20.9M | 2.94 | 64.2 | 89.4 | 69.8 |
| YOLO11-L | 26.2M | 3.86 | 63.8 | 86.2 | 71.4 |
| YOLO11-X | 58.8M | 4.93 | 67.2 | 87.5 | 74.8 |
| **DETRPose-N** | **4.1M** | **1.55** | **57.2** | **81.7** | **61.4** |
| **DETRPose-S** | **11.9M** | **2.39** | **67.0** | **88.1** | **72.9** |
| **DETRPose-M** | **23.5M** | **3.67** | **69.4** | **89.8** | **75.5** |
| **DETRPose-L** | **36.8M** | **5.08** | **72.5** | **90.6** | **79.0** |
| **DETRPose-X** | **82.3M** | **8.59** | **73.3** | **90.5** | **79.4** |

> DETRPose-S matches YOLOv8-X and YOLO11-X accuracy (67.0 vs 67.3/67.2 AP) with 81% fewer parameters (11.5M vs 69.4M/58.8M) and 52% faster inference (2.39ms vs 5.23ms/4.93ms).

---

## Summary of Sources

| Model Family | Primary Source | Paper/ArXiv | GitHub |
|-------------|---------------|-------------|--------|
| RF-DETR | https://rfdetr.roboflow.com/latest/learn/benchmarks/ | arXiv:2511.09554 | https://github.com/roboflow/rf-detr |
| RF-DETR Keypoint | https://rfdetr.roboflow.com/latest/learn/run/keypoints/ | (same paper) | (same repo) |
| YOLO26 | https://docs.ultralytics.com/models/yolo26 | arXiv:2606.03748 | https://github.com/ultralytics/ultralytics |
| YOLO26-pose | https://docs.ultralytics.com/tasks/pose | (same paper) | (same repo) |
| SAM 3 | https://ai.meta.com/blog/segment-anything-model-3/ | SAM 3 paper | https://github.com/facebookresearch/sam3 |
| SAM 3.1 | https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md | (same paper) | (same repo) |
| DETRPose | arXiv:2506.13027 | arXiv:2506.13027 | https://github.com/SebastianJanampa/DETRPose |
