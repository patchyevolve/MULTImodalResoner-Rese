# Benchmarking Plan

> How we measure everything. Every metric, every tool, every comparison. This is what makes the difference between "we built something" and "we built something impressive."

---

## 1. Benchmark Categories

| Category | What We Measure | Tools | Target |
|---|---|---|---|
| Component Latency | Per-component inference time | PyTorch Profiler, nsys | Per budget |
| End-to-End Latency | Frame-in to claim-out | Custom timer | <40ms fast path |
| Accuracy | Detection, tracking, reasoning | Standard metrics | Match/exceed baselines |
| Calibration | ECE, coverage, set size | Custom scripts | ECE <0.05 |
| Throughput | Sustained FPS | Custom counter | ≥30 FPS |
| Resource Usage | GPU/CPU/Memory | nvidia-smi, psutil | Stable, no leaks |
| Stress Test | High load behavior | Multi-stream, many entities | Graceful degradation |

---

## 2. Component Latency Benchmarks

### Detection (RF-DETR-S)

```python
# scripts/benchmark/detection_latency.py

import torch
import time
import numpy as np

def benchmark_detection(engine_path, input_sizes, num_iterations=1000):
    """Benchmark detection latency at different input sizes."""
    context = load_trt_engine(engine_path)

    results = {}
    for size in input_sizes:
        input_tensor = torch.randn(1, 3, size, size).cuda()

        # Warmup
        for _ in range(100):
            run_trt(context, input_tensor)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter_ns()
            run_trt(context, input_tensor)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            times.append((end - start) / 1e6)  # ms

        times = np.array(times)
        results[size] = {
            "p50": np.percentile(times, 50),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
            "mean": times.mean(),
            "std": times.std(),
        }
        print(f"  Input {size}x{size}: p50={results[size]['p50']:.2f}ms, "
              f"p95={results[size]['p95']:.2f}ms, p99={results[size]['p99']:.2f}ms")

    return results

# Run
results = benchmark_detection(
    "models/trt_engines/rf_detr_s_fp16.engine",
    input_sizes=[512, 640, 800],
    num_iterations=1000
)
```

**Target:**
| Input Size | p50 | p95 | p99 |
|---|---|---|---|
| 512x512 | <2.0ms | <2.5ms | <3.0ms |
| 640x640 | <2.5ms | <3.0ms | <3.5ms |
| 800x800 | <3.5ms | <4.5ms | <5.0ms |

### Pose Estimation (DETRPose-S)

```python
# Same structure, different model
# Target: p50 <2.0ms on 640x640 person crops
```

### Tracking (ByteTrack)

```python
def benchmark_tracking(num_objects, num_frames=1000):
    """Benchmark tracking with varying object counts."""
    tracker = ByteTrack()

    times = []
    for frame_id in range(num_frames):
        # Generate synthetic detections
        detections = generate_synthetic_detections(num_objects)

        start = time.perf_counter_ns()
        tracks = tracker.update(detections, frame_id)
        end = time.perf_counter_ns()

        times.append((end - start) / 1e6)

    times = np.array(times)
    print(f"Tracking ({num_objects} objects): p50={np.percentile(times, 50):.3f}ms")
    return times

# Test with different object counts
for n in [10, 25, 50, 100]:
    benchmark_tracking(n)
```

**Target:**
| Objects | p50 | p95 |
|---|---|---|
| 10 | <0.1ms | <0.2ms |
| 25 | <0.2ms | <0.3ms |
| 50 | <0.3ms | <0.5ms |
| 100 | <0.5ms | <1.0ms |

---

## 3. End-to-End Latency Benchmark

```python
# scripts/benchmark/e2e_latency.py

class E2EBenchmark:
    """Measure complete pipeline latency from frame input to claim output."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.frame_times = []

    def run(self, video_path, duration_seconds=60):
        """Run benchmark on video file."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(duration_seconds * fps)

        latencies = {
            "perception": [],
            "state_update": [],
            "event_detection": [],
            "reasoning_fast": [],
            "reasoning_deep": [],
            "calibration": [],
            "total": [],
        }

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_input = FrameInput(
                frame_id=i,
                timestamp_ns=time.time_ns(),
                image=frame,
                stream_id="benchmark",
            )

            # Measure each stage
            t0 = time.perf_counter_ns()
            perception = self.pipeline.perception.process_frame(frame_input)
            t1 = time.perf_counter_ns()

            snapshot = self.pipeline.state.update(i, perception["tracks"])
            t2 = time.perf_counter_ns()

            events = self.pipeline._detect_events(snapshot)
            t3 = time.perf_counter_ns()

            for event in events:
                t4 = time.perf_counter_ns()
                fast_result = self.pipeline.reasoning.fast_verifier.verify(event, snapshot)
                t5 = time.perf_counter_ns()

                if fast_result.status == "INCONCLUSIVE":
                    deep_result = self.pipeline.reasoning.vlm_reasoner.reason(event, snapshot)
                    t6 = time.perf_counter_ns()
                    latencies["reasoning_deep"].append((t6 - t5) / 1e6)
                else:
                    t6 = t5

                calibrated = self.pipeline.calibration.calibrate(fast_result, snapshot)
                t7 = time.perf_counter_ns()
                latencies["calibration"].append((t7 - t6) / 1e6)
                latencies["reasoning_fast"].append((t5 - t4) / 1e6)

            latencies["perception"].append((t1 - t0) / 1e6)
            latencies["state_update"].append((t2 - t1) / 1e6)
            latencies["event_detection"].append((t3 - t2) / 1e6)
            latencies["total"].append((t3 - t0) / 1e6)

        cap.release()
        return self._summarize(latencies)

    def _summarize(self, latencies):
        summary = {}
        for key, values in latencies.items():
            if values:
                arr = np.array(values)
                summary[key] = {
                    "p50": np.percentile(arr, 50),
                    "p95": np.percentile(arr, 95),
                    "p99": np.percentile(arr, 99),
                    "mean": arr.mean(),
                    "max": arr.max(),
                    "count": len(arr),
                }
        return summary
```

**Target:**
| Stage | p50 | p95 | Budget |
|---|---|---|---|
| Perception | <10ms | <15ms | <20ms |
| State Update | <0.5ms | <1ms | <1ms |
| Event Detection | <0.3ms | <0.5ms | <0.5ms |
| Fast Reasoning | <10ms | <20ms | <20ms |
| Calibration | <5ms | <8ms | <8ms |
| **Total (fast path)** | **<20ms** | **<35ms** | **<40ms** |

---

## 4. Accuracy Benchmarks

### Detection Accuracy

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_detection(predictions_file, gt_file):
    """Evaluate detection mAP on COCO format."""
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(predictions_file)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Returns: AP, AP50, AP75, APs, APm, APl
    return coco_eval.stats
```

**Target: mAP ≥ 53.0** (RF-DETR-S baseline on COCO)

### Tracking Accuracy

```python
# Use TrackEval or motmetrics
import motmetrics as mm

def evaluate_tracking(gt_file, pred_file):
    """Evaluate tracking MOTA, IDF1, HOTA."""
    acc = mm.MOTChallengeEvaluator.load_file(gt_file, pred_file)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'idf1', 'hota'], name='acc')
    return summary
```

**Target:**
| Metric | Target |
|---|---|
| MOTA | >75% |
| IDF1 | >79% |
| HOTA | >67% |

### Re-ID Accuracy

```python
def evaluate_reid(model, test_loader):
    """Evaluate Rank-1, Rank-5, mAP."""
    all_features = []
    all_labels = []

    for images, labels in test_loader:
        features = model(images.cuda())
        all_features.append(features.cpu())
        all_labels.append(labels)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    # Compute similarity matrix
    similarity = torch.mm(all_features, all_features.t())

    # Rank-1, Rank-5, mAP
    rank1 = compute_rank_k(similarity, all_labels, k=1)
    rank5 = compute_rank_k(similarity, all_labels, k=5)
    mAP = compute_map(similarity, all_labels)

    return {"rank1": rank1, "rank5": rank5, "mAP": mAP}
```

**Target: Rank-1 > 95%, mAP > 85%** (Market1501)

---

## 5. Calibration Benchmarks

```python
# scripts/benchmark/calibration_quality.py

def evaluate_calibration(predictions, labels, confidences):
    """Evaluate calibration quality."""
    results = {}

    # Expected Calibration Error (ECE)
    results["ece"] = compute_ece(confidences, labels, n_bins=15)

    # Maximum Calibration Error (MCE)
    results["mce"] = compute_mce(confidences, labels, n_bins=15)

    # Brier Score
    results["brier"] = compute_brier(confidences, labels)

    # Conformal Coverage
    results["coverage_alpha_05"] = compute_coverage(
        predictions, labels, alpha=0.05
    )
    results["coverage_alpha_10"] = compute_coverage(
        predictions, labels, alpha=0.10
    )

    # Average Prediction Set Size
    results["avg_set_size_alpha_05"] = compute_avg_set_size(
        predictions, alpha=0.05
    )
    results["avg_set_size_alpha_10"] = compute_avg_set_size(
        predictions, alpha=0.10
    )

    # Reliability Diagram
    plot_reliability_diagram(confidences, labels, "reliability_diagram.png")

    return results
```

**Target:**
| Metric | Target |
|---|---|
| ECE | <0.05 |
| MCE | <0.10 |
| Brier Score | <0.15 |
| Coverage (α=0.05) | ≥95% |
| Coverage (α=0.10) | ≥90% |
| Avg Set Size (α=0.05) | 1-3 |

---

## 6. Throughput Benchmark

```python
# scripts/benchmark/throughput.py

def measure_sustained_fps(pipeline, video_path, duration=300):
    """Measure sustained FPS over 5 minutes."""
    cap = cv2.VideoCapture(video_path)
    fps_counter = 0
    start_time = time.time()
    fps_history = []

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        pipeline.process_frame(FrameInput(
            frame_id=fps_counter,
            timestamp_ns=time.time_ns(),
            image=frame,
            stream_id="throughput_test",
        ))
        fps_counter += 1

        if fps_counter % 30 == 0:
            current_fps = 30 / (time.time() - start_time + 0.001)
            fps_history.append(current_fps)

    cap.release()
    return {
        "avg_fps": np.mean(fps_history),
        "min_fps": np.min(fps_history),
        "max_fps": np.max(fps_history),
        "p50_fps": np.percentile(fps_history, 50),
        "total_frames": fps_counter,
    }
```

**Target: Sustained ≥30 FPS** on test video

---

## 7. Resource Usage Benchmark

```python
# scripts/benchmark/resource_usage.py

import psutil
import subprocess

def monitor_resources(duration_seconds=300, interval=1.0):
    """Monitor GPU/CPU/Memory usage over time."""
    gpu_history = []
    cpu_history = []
    mem_history = []

    start = time.time()
    while time.time() - start < duration_seconds:
        # GPU
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"]
        ).decode().strip().split(", ")
        gpu_history.append({
            "timestamp": time.time(),
            "utilization": float(gpu_info[0]),
            "memory_mb": float(gpu_info[1]),
            "temperature": float(gpu_info[2]),
        })

        # CPU
        cpu_history.append(psutil.cpu_percent())

        # Memory
        mem = psutil.virtual_memory()
        mem_history.append(mem.used / 1e9)

        time.sleep(interval)

    return {
        "gpu_avg_utilization": np.mean([g["utilization"] for g in gpu_history]),
        "gpu_max_memory_mb": max(g["memory_mb"] for g in gpu_history),
        "gpu_max_temperature": max(g["temperature"] for g in gpu_history),
        "cpu_avg": np.mean(cpu_history),
        "mem_avg_gb": np.mean(mem_history),
        "mem_peak_gb": max(mem_history),
    }
```

**Target:**
| Metric | Target |
|---|---|
| GPU utilization (inference) | >70% |
| GPU memory peak | <20GB (RTX 4090) |
| GPU temperature | <80°C |
| CPU average | <50% |
| Memory peak | <16GB |

---

## 8. Stress Test

```python
# scripts/benchmark/stress_test.py

def stress_test_multi_stream(pipeline, num_streams=4, duration=60):
    """Test with multiple concurrent video streams."""
    import threading

    results = {}
    for n in range(1, num_streams + 1):
        fps_per_stream = measure_throughput_with_n_streams(pipeline, n, duration)
        results[n] = fps_per_stream
        print(f"{n} streams: {fps_per_stream:.1f} FPS each")

    return results

def stress_test_many_entities(pipeline, entity_counts=[10, 25, 50, 100]):
    """Test with increasing entity counts."""
    results = {}
    for count in entity_counts:
        fps = measure_fps_with_n_entities(pipeline, count)
        results[count] = fps
        print(f"{count} entities: {fps:.1f} FPS")

    return results
```

**Target:**
| Streams | FPS per Stream |
|---|---|
| 1 | ≥30 |
| 2 | ≥25 |
| 4 | ≥15 |

---

## 9. Comparison Against Baselines

We compare our system against:

| Baseline | What It Does | Why We Beat It |
|---|---|---|
| Naive VLM (GPT-4o every frame) | Send every frame to VLM | Our fast path resolves 80% without VLM |
| Detection only | No reasoning | We add reasoning + calibration |
| YOLO + ByteTrack | Baseline detection+tracking | RF-DETR beats YOLO at same latency |
| No calibration | Raw confidence scores | Our ECE <0.05 vs typical 0.15-0.25 |

### Comparison Script

```python
def compare_systems(video_path):
    """Run comparison between our system and baselines."""
    results = {}

    # Our system
    results["our_system"] = benchmark_system(multimodal_reasoner, video_path)

    # Naive VLM every frame
    results["naive_vlm"] = benchmark_naive_vlm(video_path)

    # Detection only (no reasoning)
    results["detection_only"] = benchmark_detection_only(video_path)

    # Generate comparison table
    print_comparison_table(results)

    # Generate comparison plot
    plot_comparison(results, "comparison_results.png")
```

---

## 10. Benchmark Output Format

All benchmarks write results to:

```
experiments/benchmarks/
├── latency/
│   ├── detection_latency.json
│   ├── pose_latency.json
│   ├── tracking_latency.json
│   └── e2e_latency.json
├── accuracy/
│   ├── detection_map.json
│   ├── tracking_mot17.json
│   └── reid_market1501.json
├── calibration/
│   ├── calibration_quality.json
│   └── reliability_diagram.png
├── throughput/
│   ├── sustained_fps.json
│   └── stress_test.json
├── resources/
│   └── resource_usage.json
├── comparison/
│   ├── comparison_results.json
│   └── comparison_plot.png
└── summary.json  ← Combined results for paper
```

---

## 11. Benchmark Timeline

| Day | Benchmark |
|---|---|
| Day 15 | Component latency (detection, pose, tracking) |
| Day 16 | Component latency (VLM, calibration) |
| Day 17 | End-to-end latency |
| Day 18 | Accuracy (detection mAP, tracking MOTA) |
| Day 19 | Calibration quality (ECE, coverage) |
| Day 20 | Throughput (sustained FPS) |
| Day 21 | Stress test (multi-stream, many entities) |
| Day 22 | Comparison against baselines |
| Day 23 | Resource usage monitoring |
| Day 24 | Final summary + paper figures |
