# Infrastructure Development Plan

> The code, the plumbing, the pipeline, the servers — everything that makes the trained models actually work as a system.

---

## 1. What Is "Infrastructure"

Infrastructure is everything that is NOT a model. It is:

| Layer | What | Purpose |
|---|---|---|
| Ingestor | Video stream reader | Get frames from camera/file/stream into GPU memory |
| Preprocessing | Resize, normalize, batch | Transform raw frames into model-ready tensors |
| Model Server | TensorRT engines + batch scheduling | Run inference at 30 FPS |
| Post-Processing | Tracking, state, events | Turn model outputs into structured world state |
| Reasoning | Hypothesis, VLM, evidence | Think about what the world state means |
| Calibration | Confidence, conformal, claims | Make uncertainty honest and structured |
| Output | API, WebSocket, Kafka | Deliver claims to consumers |
| Monitoring | Metrics, logs, traces | Know what is happening and when it breaks |
| Deployment | Docker, K8s, configs | Run everything reliably |

---

## 2. System Architecture — What Runs Where

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXTERNAL INPUTS                             │
│  RTSP Stream · HTTP Video · File Upload · Audio Stream          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  INGESTOR LAYER (Thread)                                        │
│  RTSP Reader (FFmpeg)  ·  File Reader  ·  Audio Splitter        │
│         └─────────────────┼───────────────────┘                 │
│                           ▼                                     │
│              ┌─────────────────────────┐                       │
│              │  Frame Buffer Pool      │                       │
│              │  (Pre-allocated CUDA)   │                       │
│              └────────────┬────────────┘                       │
└───────────────────────────┼────────────────────────────────────┘
                            │  Zero-copy GPU tensors
┌───────────────────────────▼────────────────────────────────────┐
│  PERCEPTION (Process 2, GPU 0, 30 FPS)                         │
│  Decode → Detect(TRT) → Pose(TRT) → Track(ByteTrack)          │
│  + Async: Segmentation, OCR, ReID, Audio, Camera Motion        │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│  STATE + REASONING (Process 3, CPU + GPU 1)                    │
│  World State → Event Detection → Hypothesis Engine             │
│  → Fast Verify (1-5ms) → Deep VLM (800ms+ async)              │
│  → Evidence Graph → Calibration → Claim Output                 │
└───────────────────────────┬────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────┐
│  OUTPUT (Process 4)                                            │
│  REST API · WebSocket · Kafka · File Writer                    │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. How to Build It — Step by Step

### Step 1: Get a single frame from video file into GPU memory

```python
# src/ingestor/file_ingestor.py

import av
import numpy as np
import torch

class FileIngestor:
    """Read video files frame by frame with GPU decode."""

    def __init__(self, video_path: str, target_fps: int = 30):
        self.container = av.open(video_path)
        self.stream = self.container.streams.video[0]
        self.target_fps = target_fps
        self.frame_interval = self.stream.average_rate / target_fps
        self.frame_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Yield next frame as GPU tensor."""
        for frame in self.container.decode(video=0):
            self.frame_count += 1
            # Skip frames to match target FPS
            if self.frame_count % int(self.frame_interval) != 0:
                continue

            # Convert to numpy (HWC, RGB)
            arr = frame.to_ndarray(format="rgb24")

            # Transfer to GPU (pinned memory for speed)
            tensor = torch.from_numpy(arr).cuda(non_blocking=True)
            return {
                "frame_id": self.frame_count,
                "timestamp_ns": int(frame.time * 1e9) if frame.time else 0,
                "image": tensor,  # [H, W, 3] uint8 on GPU
                "source": "file",
            }
        raise StopIteration

    def close(self):
        self.container.close()
```

### Step 2: Preprocess the frame for detection model

```python
# src/preprocessing/preprocessor.py

import torch
import cv2
import numpy as np

class Preprocessor:
    """Preprocess frames for model input. All operations on GPU."""

    def __init__(self, input_size=(640, 640)):
        self.input_size = input_size
        # ImageNet normalization constants (on GPU)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)

    def preprocess(self, frame_tensor):
        """
        Input: [H, W, 3] uint8 GPU tensor (RGB)
        Output: [1, 3, 640, 640] float16 GPU tensor (NCHW, normalized)
        """
        # HWC -> CHW
        tensor = frame_tensor.permute(2, 0, 1).float()  # [3, H, W]
        tensor = tensor / 255.0  # Normalize to [0, 1]

        # Resize to model input size using GPU
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),  # [1, 3, H, W]
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )  # [1, 3, 640, 640]

        # ImageNet normalization
        tensor = (tensor - self.mean) / self.std

        # Convert to FP16 for TensorRT
        return tensor.half()
```

### Step 3: Run detection model

```python
# src/perception/detector.py

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch

class TensorRTDetector:
    """TensorRT inference for RF-DETR detection."""

    def __init__(self, engine_path: str, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate GPU buffers
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Pre-allocate GPU input/output buffers (once, at startup)."""
        self.inputs = []
        self.outputs = []
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(shape)
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            if self.engine.binding_is_input(binding):
                self.inputs.append({"device": device_mem, "shape": shape, "dtype": dtype})
            else:
                self.outputs.append({"device": device_mem, "shape": shape, "dtype": dtype})

    def detect(self, input_tensor: torch.Tensor):
        """
        Run detection on preprocessed tensor.

        Input: [1, 3, 640, 640] float16 GPU tensor
        Output: dict with boxes, scores, labels
        """
        # Copy input to TRT buffer
        cuda.memcpy_dtod(
            self.inputs[0]["device"],
            input_tensor.data_ptr(),
            input_tensor.nelement() * input_tensor.element_size(),
        )

        # Run inference
        self.context.execute_v2(
            bindings=[int(inp["device"]) for inp in self.inputs] +
                      [int(out["device"]) for out in self.outputs]
        )

        # Copy output back
        output = torch.empty(
            self.outputs[0]["shape"], dtype=torch.float16, device="cuda"
        )
        cuda.memcpy_dtod(
            output.data_ptr(),
            self.outputs[0]["device"],
            output.nelement() * output.element_size(),
        )

        # Parse output
        return self._parse_output(output)

    def _parse_output(self, output):
        """Parse TRT output into detection results."""
        # RF-DETR outputs: [batch, num_queries, 4+num_classes]
        boxes = output[0, :, :4]   # [N, 4] normalized xywh
        scores = output[0, :, 4:]  # [N, num_classes]

        # Filter by confidence
        max_scores = scores.max(dim=-1)
        mask = max_scores.values > self.confidence_threshold

        return {
            "boxes": boxes[mask],      # [K, 4]
            "scores": max_scores.values[mask],  # [K]
            "labels": max_scores.indices[mask],  # [K]
        }
```

### Step 4: Run tracking on detections

```python
# src/perception/tracker.py

from byte_track import ByteTracker

class Tracker:
    """ByteTrack wrapper for object tracking."""

    def __init__(self, config):
        self.tracker = ByteTracker(
            track_thresh=config.get("track_thresh", 0.5),
            track_buffer=config.get("track_buffer", 30),
            match_thresh=config.get("match_thresh", 0.8),
            min_box_area=config.get("min_box_area", 10),
        )

    def update(self, detections, frame_id):
        """
        Update tracker with new detections.

        Input: dict with boxes, scores, labels from detector
        Output: list of Track objects with persistent IDs
        """
        # Convert to ByteTrack format: [x1, y1, x2, y2, score, class]
        dets = self._to_bytetrack_format(detections)

        # Run tracking
        online_targets = self.tracker.update(dets, [img_info], [img_size])

        tracks = []
        for t in online_targets:
            tlwh = t.tlwh
            tracks.append({
                "track_id": t.track_id,
                "bbox": [tlwh[0], tlwh[1], tlwh[2], tlwh[3]],  # xywh
                "score": t.score,
                "class_id": t.class_id if hasattr(t, "class_id") else 0,
            })

        return tracks
```

### Step 5: Update world state

```python
# src/state/world_state.py

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

@dataclass
class Entity:
    id: str
    bbox: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    confidence: float = 0.0
    tracking_state: str = "TENTATIVE"
    first_frame: int = 0
    last_frame: int = 0
    frame_count: int = 0

class WorldState:
    """Lock-free ring buffer world state."""

    def __init__(self, capacity: int = 30):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.entities: Dict[str, Entity] = {}

    def update(self, frame_id: int, tracks: list) -> dict:
        """Update with new tracking data. Returns snapshot."""
        # Update entities
        for track in tracks:
            tid = str(track["track_id"])
            bbox = np.array(track["bbox"])

            if tid in self.entities:
                e = self.entities[tid]
                e.velocity = bbox[:2] - e.bbox[:2]  # position delta
                e.bbox = bbox
                e.confidence = track["score"]
                e.last_frame = frame_id
                e.frame_count += 1
                if e.frame_count >= 3:
                    e.tracking_state = "CONFIRMED"
            else:
                self.entities[tid] = Entity(
                    id=tid,
                    bbox=bbox,
                    confidence=track["score"],
                    first_frame=frame_id,
                    last_frame=frame_id,
                    frame_count=1,
                )

        # Clean up lost entities
        for tid, e in list(self.entities.items()):
            if frame_id - e.last_frame > 30:
                del self.entities[tid]

        # Snapshot
        snapshot = {
            "frame_id": frame_id,
            "entities": {k: vars(v) for k, v in self.entities.items()},
            "entity_count": len(self.entities),
        }
        self.buffer.append(snapshot)
        return snapshot
```

### Step 6: Detect events and score priority

```python
# src/state/event_detector.py

import numpy as np

class EventDetector:
    """Rule-based event detection with R-score priority."""

    def __init__(self, rules: list):
        self.rules = rules

    def detect(self, snapshot: dict, prev_snapshot: dict = None) -> list:
        """Detect events from world state."""
        events = []

        for rule in self.rules:
            if rule.check(snapshot, prev_snapshot):
                events.append({
                    "event_type": rule.name,
                    "frame_id": snapshot["frame_id"],
                    "entity_ids": rule.involved_entities(snapshot),
                    "r_score": rule.compute_r_score(snapshot),
                    "confidence": rule.confidence,
                })

        return events

# Example rules:
VELOCITY_SPIKE_RULE = {
    "name": "velocity_spike",
    "check": lambda snap, prev: any(
        np.linalg.norm(e["velocity"]) > 5.0
        for e in snap["entities"].values()
        if e.get("tracking_state") == "CONFIRMED"
    ),
    "r_score": 0.7,
}

NEW_ENTITY_RULE = {
    "name": "new_entity",
    "check": lambda snap, prev: (
        snap["entity_count"] > (prev["entity_count"] if prev else 0)
    ),
    "r_score": 0.5,
}
```

### Step 7: Reasoning pipeline (orchestration)

```python
# src/reasoning/pipeline.py

class ReasoningPipeline:
    """Orchestrates hypothesis generation, verification, and VLM reasoning."""

    def __init__(self, config):
        self.hypothesis_engine = HypothesisEngine(config)
        self.fast_verifier = FastVerifier(config)
        self.vlm_reasoner = VLMReasoner(config)
        self.evidence_graph = EvidenceGraph()

    def process(self, event: dict, world_state, memory):
        """Process event through reasoning pipeline."""
        # Generate hypotheses
        hypotheses = self.hypothesis_engine.generate(event, world_state)

        results = []
        for hyp in hypotheses:
            # Fast verification (1-5ms)
            verdict = self.fast_verifier.verify(hyp, world_state)

            if verdict["status"] in ["SUPPORTED", "REFUTED"]:
                results.append(verdict)
            else:
                # Deep VLM (async, 800ms+)
                deep_result = self.vlm_reasoner.reason(hyp, world_state, memory)
                results.append(deep_result)

            # Update evidence graph
            self.evidence_graph.add(hyp, verdict)

        return results
```

### Step 8: Full pipeline — connect everything

```python
# src/pipeline.py

import cv2
import time

class MultimodalReasoner:
    """Complete end-to-end pipeline."""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)

        # Initialize all components
        self.preprocessor = Preprocessor(input_size=(640, 640))
        self.detector = TensorRTDetector(
            self.config.detection_engine, confidence_threshold=0.5
        )
        self.tracker = Tracker(self.config.tracker)
        self.world_state = WorldState(capacity=30)
        self.event_detector = EventDetector(self.config.events)
        self.reasoning = ReasoningPipeline(self.config.reasoning)
        self.calibration = CalibrationPipeline(self.config.calibration)
        self.output = ClaimPublisher(self.config.output)

        # Performance tracking
        self.frame_count = 0
        self.latency_log = []

    def process_video(self, source: str):
        """Main processing loop."""
        ingestor = FileIngestor(source, target_fps=30)

        for frame in ingestor:
            t_start = time.perf_counter()

            # 1. Preprocess (GPU)
            input_tensor = self.preprocessor.preprocess(frame["image"])

            # 2. Detect (TensorRT, GPU)
            detections = self.detector.detect(input_tensor)

            # 3. Track (ByteTrack, CPU)
            tracks = self.tracker.update(detections, frame["frame_id"])

            # 4. Update world state
            snapshot = self.world_state.update(frame["frame_id"], tracks)

            # 5. Detect events
            events = self.event_detector.detect(snapshot)

            # 6. Reason about events
            for event in events:
                if event["r_score"] >= 0.5:
                    results = self.reasoning.process(event, self.world_state, None)

                    # 7. Calibrate
                    for result in results:
                        claim = self.calibration.calibrate(result)

                        # 8. Output
                        self.output.publish(claim)

            # Track latency
            latency_ms = (time.perf_counter() - t_start) * 1000
            self.latency_log.append(latency_ms)
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                avg = sum(self.latency_log[-30:]) / 30
                print(f"Frame {self.frame_count}: avg latency {avg:.1f}ms "
                      f"({1000/avg:.0f} FPS)")

        ingestor.close()
```

---

## 4. Process Architecture

### Option A: Single Process (Development / Demo)

```
Main Thread:
  for frame in video:
      preprocess → detect → track → state → event → reason → calibrate → output

优点: Simple, debuggable, no IPC overhead
缺点: VLM blocks pipeline, no true async
Use for: Initial development, demos, single-stream
```

### Option B: Multi-Thread (Production)

```
Thread 1 (Decode):      Ingest frames → Frame Buffer Pool
Thread 2 (Perception):  Preprocess → Detect → Pose → Track → State
Thread 3 (Events):      Event Detection → Scheduler → Queue
Thread 4 (Fast Reason): Hypothesis → Fast Verify → Claim
Thread 5 (Deep VLM):    VLM Inference (async, blocks only on GPU)
Thread 6 (Output):      API Server + WebSocket + Kafka

Communication: Lock-free ring buffers between threads
```

### Option C: Multi-Process (Full Production)

```
Process 1: Ingestor (CPU, reads video)
Process 2: Perception (GPU 0, TensorRT inference)
Process 3: State + Reasoning (CPU + GPU 1)
Process 4: API Server (CPU, serves claims)
Process 5: Monitoring (CPU, collects metrics)

Communication: Shared memory or Kafka between processes
```

**Start with Option A. Move to Option B when single-stream works.**

---

## 5. Configuration System

```yaml
# configs/production.yaml

system:
  name: "multimodal-reasoner"
  log_level: "INFO"
  gpu_ids: [0]

ingestor:
  source_type: "file"          # file | rtsp | http
  source_path: "data/videos/test.mp4"
  target_fps: 30
  buffer_size: 8               # frame buffer depth
  decode_threads: 2

preprocessing:
  input_size: [640, 640]
  normalization: "imagenet"
  precision: "fp16"

detection:
  engine_path: "models/trt_engines/rf_detr_s_fp16.engine"
  confidence_threshold: 0.5
  nms_threshold: 0.7
  max_detections: 100
  input_size: [640, 640]

pose:
  engine_path: "models/trt_engines/detrpose_s_fp16.engine"
  enabled: true
  confidence_threshold: 0.3
  crop_padding: 0.1

tracking:
  tracker: "bytetrack"
  track_thresh: 0.5
  match_thresh: 0.8
  track_buffer: 30
  min_box_area: 10

state:
  ring_buffer_capacity: 30
  entity_timeout_frames: 30
  velocity_window: 5

events:
  rules:
    - name: "velocity_spike"
      threshold: 5.0
      r_score: 0.7
    - name: "new_entity"
      r_score: 0.5
    - name: "entity_occluded"
      duration_frames: 10
      r_score: 0.6

reasoning:
  hypothesis_max: 10
  fast_verify_timeout_ms: 5
  vlm_mode: "api"              # api | local
  vlm_model: "gpt-4o"         # or qwen2.5-vl-7b
  vlm_api_key_env: "OPENAI_API_KEY"
  vlm_queue_depth: 4
  vlm_timeout_ms: 5000

calibration:
  temperature_model: "models/calibrator/temperature.json"
  conformal_model: "models/calibrator/conformal.json"
  decomposition_model: "models/calibrator/decomposition.json"

output:
  rest_api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
  websocket:
    enabled: true
    port: 8001
  kafka:
    enabled: false
    brokers: ["localhost:9092"]
    topic_claims: "claims.published"
  file:
    enabled: true
    output_dir: "output/claims/"

monitoring:
  prometheus:
    enabled: true
    port: 9090
  logging:
    format: "json"
    level: "INFO"
```

---

## 6. Docker Deployment

```dockerfile
# deployment/docker/Dockerfile

# Stage 1: Build
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip \
    libavcodec-dev libavformat-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY models/trt_engines/ /app/models/trt_engines/
COPY models/calibrator/ /app/models/calibrator/

WORKDIR /app
EXPOSE 8000 8001 9090

CMD ["python3.11", "-m", "src.server"]
```

```yaml
# deployment/docker/docker-compose.yaml

version: "3.8"

services:
  reasoner:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    ports:
      - "8000:8000"   # REST API
      - "8001:8001"   # WebSocket
      - "9090:9090"   # Prometheus metrics
    volumes:
      - ../../configs:/app/configs
      - ../../data/videos:/app/data/videos
      - ../../output:/app/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 10s
      timeout: 5s
      retries: 3
```

---

## 7. Monitoring Setup

```python
# src/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
FRAMES_PROCESSED = Counter("frames_processed_total", "Total frames processed")
DETECTIONS_COUNT = Counter("detections_total", "Total detections made")
CLAIMS_PUBLISHED = Counter("claims_published_total", "Total claims published")

FRAME_LATENCY = Histogram(
    "frame_latency_ms",
    "Frame processing latency in milliseconds",
    buckets=[1, 2, 5, 10, 20, 30, 50, 100],
)

GPU_UTILIZATION = Gauge("gpu_utilization_percent", "GPU utilization percentage")
GPU_MEMORY_USED = Gauge("gpu_memory_used_gb", "GPU memory used in GB")
QUEUE_DEPTH = Gauge("queue_depth", "Current queue depth", ["queue_name"])
ACTIVE_ENTITIES = Gauge("active_entities", "Number of tracked entities")
ACTIVE_TRACKS = Gauge("active_tracks", "Number of active tracks")

def start_monitoring(port: int = 9090):
    """Start Prometheus metrics server."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")

# Usage in pipeline:
# with FRAME_LATENCY.time():
#     result = process_frame(frame)
# FRAME_LATENCY.observe(latency_ms)
```

---

## 8. Testing Strategy

### Unit Tests (per component)

```python
# tests/test_detector.py

def test_detector_loads_engine():
    detector = TensorRTDetector("models/trt_engines/rf_detr_s_fp16.engine")
    assert detector.engine is not None

def test_detector_returns_boxes():
    detector = TensorRTDetector("models/trt_engines/rf_detr_s_fp16.engine")
    dummy_input = torch.randn(1, 3, 640, 640).half().cuda()
    result = detector.detect(dummy_input)
    assert "boxes" in result
    assert "scores" in result
    assert "labels" in result

def test_detector_latency():
    detector = TensorRTDetector("models/trt_engines/rf_detr_s_fp16.engine")
    dummy_input = torch.randn(1, 3, 640, 640).half().cuda()
    # Warmup
    for _ in range(10):
        detector.detect(dummy_input)
    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        detector.detect(dummy_input)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    assert np.percentile(times, 50) < 5.0  # p50 < 5ms
```

### Integration Tests (pipeline)

```python
# tests/test_pipeline.py

def test_pipeline_runs_on_video():
    pipeline = MultimodalReasoner("configs/test.yaml")
    claims = pipeline.process_video("data/videos/test_5s.mp4")
    assert len(claims) > 0
    assert all("confidence" in c for c in claims)

def test_pipeline_latency():
    pipeline = MultimodalReasoner("configs/test.yaml")
    claims = pipeline.process_video("data/videos/test_5s.mp4")
    avg_latency = np.mean(pipeline.latency_log)
    assert avg_latency < 33.3  # Must be faster than 30 FPS
```

### Stress Tests

```python
# tests/test_stress.py

def test_sustained_30fps():
    pipeline = MultimodalReasoner("configs/production.yaml")
    # Run for 5 minutes on test video
    start = time.time()
    frame_count = 0
    while time.time() - start < 300:
        pipeline.process_next_frame()
        frame_count += 1
    fps = frame_count / (time.time() - start)
    assert fps >= 30.0
```

---

## 9. Build Order (What to Code First)

| Priority | Component | Days | Depends On |
|---|---|---|---|
| 1 | `ingestor/file_ingestor.py` | 0.5 | Nothing |
| 2 | `preprocessing/preprocessor.py` | 0.5 | Nothing |
| 3 | `perception/detector.py` | 1 | TensorRT engine |
| 4 | `perception/tracker.py` | 0.5 | ByteTrack |
| 5 | `state/world_state.py` | 0.5 | Nothing |
| 6 | `state/event_detector.py` | 0.5 | World State |
| 7 | `pipeline.py` (single stream) | 1 | All above |
| 8 | `perception/pose_estimator.py` | 1 | TensorRT engine |
| 9 | `reasoning/hypothesis_engine.py` | 1 | Event Detector |
| 10 | `reasoning/fast_verifier.py` | 1 | Hypothesis Engine |
| 11 | `reasoning/vlm_reasoner.py` | 1 | API keys or local model |
| 12 | `calibration/` (all files) | 2 | Trained calibrator |
| 13 | `output/rest_api.py` | 1 | Pipeline |
| 14 | `output/websocket_server.py` | 1 | Pipeline |
| 15 | `monitoring/metrics.py` | 0.5 | Pipeline |
| 16 | `Docker setup` | 1 | Everything |

**Total: ~13-14 days of coding** (after models are trained)

---

## 10. Key Engineering Principles

1. **GPU-first**: Do everything on GPU until you absolutely must move to CPU. No CPU↔GPU copies in the hot path.

2. **Pre-allocate everything**: Frame buffers, TRT buffers, ring buffers — allocate once at startup, reuse forever.

3. **Zero-copy where possible**: Use CUDA tensors directly, don't convert to numpy unless you must.

4. **Lock-free communication**: Ring buffers between threads, never mutexes in the critical path.

5. **Async everything slow**: VLM reasoning, forensics, memory writes — all on background threads with bounded queues.

6. **Degrade gracefully**: If VLM is slow, skip it. If GPU is overloaded, downgrade model. Never crash.

7. **Measure everything**: Latency histograms, throughput counters, queue depths, GPU utilization. If you can't measure it, you can't optimize it.

8. **Fail loud**: Every error gets logged, every component has a health check, every pipeline stage has a timeout.
