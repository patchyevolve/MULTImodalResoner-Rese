# Inference Integration

> How every trained model connects into the pipeline. Data flows, API contracts, error handling. This is where the system comes alive.

---

## 1. Inference Architecture

```
Video Frame (numpy array)
        │
        ▼
┌───────────────────────────────────┐
│  TensorRT Inference Engine        │
│                                   │
│  ┌─────────┐  ┌─────────┐       │
│  │ RF-DETR │→ │ DETRPose│       │
│  │ (FP16)  │  │ (FP16)  │       │
│  └────┬────┘  └────┬────┘       │
│       │            │             │
│       ▼            ▼             │
│  ┌─────────────────────┐        │
│  │  ByteTrack (CPU)    │        │
│  └─────────┬───────────┘        │
│            │                     │
│            ▼                     │
│  ┌─────────────────────┐        │
│  │  World State Update  │        │
│  │  (Ring Buffer)       │        │
│  └─────────┬───────────┘        │
│            │                     │
└────────────┼─────────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  Reasoning Pipeline (async)     │
│  Hypothesis → Fast Verify →    │
│  (VLM if inconclusive)         │
│  → Evidence Graph              │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  Calibration Layer              │
│  Confidence → Conformal →      │
│  Temperature → Claim Output     │
└────────────────────────────────┘
```

---

## 2. Component Integration Code

### 2.1 Perception Pipeline

```python
# src/perception/pipeline.py

import tensorrt as trt
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FrameInput:
    frame_id: int
    timestamp_ns: int
    image: np.ndarray          # HWC, uint8, BGR
    stream_id: str

@dataclass
class DetectionResult:
    bbox: np.ndarray           # [N, 4] normalized
    scores: np.ndarray         # [N]
    class_ids: np.ndarray      # [N]
    embeddings: Optional[np.ndarray]  # [N, 512] if Re-ID enabled

@dataclass
class PoseResult:
    keypoints: np.ndarray      # [N, 17, 2]
    scores: np.ndarray         # [N, 17]
    body_parts: np.ndarray     # [N, 17] visibility

class PerceptionPipeline:
    """Complete perception pipeline with TensorRT inference."""

    def __init__(self, config):
        self.config = config

        # Load TensorRT engines
        self.detector = self._load_trt_engine("rf_detr_s_fp16.engine")
        self.pose_estimator = self._load_trt_engine("dettrpose_s_fp16.engine")
        self.tracker = self._init_tracker(config.tracker)

        # Pre-processing
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_trt_engine(self, engine_path):
        """Load TensorRT engine."""
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(f"models/trt_engines/{engine_path}", "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return context

    def process_frame(self, frame: FrameInput) -> dict:
        """Process a single frame through the complete perception pipeline.

        Returns:
            dict with keys: detections, tracks, entities, events
        """
        # Step 1: Preprocess
        input_tensor = self._preprocess(frame.image)

        # Step 2: Detection (TensorRT)
        detections = self._detect(input_tensor, frame)

        # Step 3: Pose Estimation (TensorRT, crops from detections)
        poses = self._estimate_pose(input_tensor, detections)

        # Step 4: Tracking (ByteTrack, CPU)
        tracks = self._track(detections, frame)

        # Step 5: State Update (Ring Buffer)
        entities = self._update_state(tracks, poses)

        return {
            "frame_id": frame.frame_id,
            "detections": detections,
            "poses": poses,
            "tracks": tracks,
            "entities": entities,
        }

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to model input size
        import cv2
        resized = cv2.resize(image, (640, 640))
        # Normalize
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        # HWC -> CHW -> NCHW
        tensor = tensor.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, 0)
        return torch.from_numpy(tensor).cuda()

    def _detect(self, input_tensor, frame) -> DetectionResult:
        """Run RF-DETR detection via TensorRT."""
        # Run inference
        output = self._run_trt(self.detector, input_tensor)

        # Parse output
        bbox = output["boxes"]
        scores = output["scores"]
        class_ids = output["labels"]

        # Filter by confidence
        mask = scores > self.config.detection_threshold
        return DetectionResult(
            bbox=bbox[mask],
            scores=scores[mask],
            class_ids=class_ids[mask],
        )

    def _estimate_pose(self, input_tensor, detections) -> PoseResult:
        """Run DETRPose on detected person crops."""
        person_mask = detections.class_ids == 0  # person class
        person_detections = detections[person_mask]

        if len(person_detections.bbox) == 0:
            return PoseResult(
                keypoints=np.array([]),
                scores=np.array([]),
                body_parts=np.array([]),
            )

        # Crop and resize person regions
        crops = self._crop_persons(input_tensor, person_detections.bbox)
        # Run pose estimation
        output = self._run_trt(self.pose_estimator, crops)

        return PoseResult(
            keypoints=output["keypoints"],
            scores=output["keypoint_scores"],
            body_parts=output["body_parts"],
        )

    def _track(self, detections, frame) -> list:
        """Run ByteTrack on detections."""
        return self.tracker.update(detections, frame)

    def _update_state(self, tracks, poses) -> list:
        """Update world state ring buffer."""
        # Implementation depends on state management module
        pass

    def _run_trt(self, context, input_tensor):
        """Run TensorRT inference."""
        # Allocate output buffers
        # Copy input to GPU
        # Execute
        # Copy output from GPU
        # Parse output
        pass
```

### 2.2 State Management

```python
# src/state/world_state.py

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Entity:
    id: str
    class_name: str
    bbox: np.ndarray
    pose: Optional[np.ndarray]
    velocity: np.ndarray
    acceleration: np.ndarray
    confidence: float
    occlusion_level: float
    tracking_state: str  # TENTATIVE, CONFIRMED, LOST
    first_seen: int      # frame_id
    last_seen: int       # frame_id
    attributes: Dict = field(default_factory=dict)

class WorldState:
    """Lock-free ring buffer world state."""

    def __init__(self, capacity=30):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.entities: Dict[str, Entity] = {}

    def update(self, frame_id: int, tracks: list, poses=None):
        """Update world state with new frame data."""
        snapshot = {
            "frame_id": frame_id,
            "entities": {},
            "timestamp_ns": self._get_timestamp(),
        }

        for track in tracks:
            entity_id = track.entity_id

            if entity_id in self.entities:
                # Update existing entity
                entity = self.entities[entity_id]
                entity.bbox = track.bbox
                entity.velocity = self._compute_velocity(entity, track)
                entity.last_seen = frame_id
                entity.confidence = track.score
            else:
                # Create new entity
                entity = Entity(
                    id=entity_id,
                    class_name=track.class_name,
                    bbox=track.bbox,
                    pose=poses.get(entity_id) if poses else None,
                    velocity=np.zeros(3),
                    acceleration=np.zeros(3),
                    confidence=track.score,
                    occlusion_level=0.0,
                    tracking_state="TENTATIVE",
                    first_seen=frame_id,
                    last_seen=frame_id,
                )
                self.entities[entity_id] = entity

            snapshot["entities"][entity_id] = entity

        self.buffer.append(snapshot)
        return snapshot

    def get_recent(self, n=5):
        """Get last n snapshots."""
        return list(self.buffer)[-n:]

    def get_entity_history(self, entity_id, max_frames=30):
        """Get entity history across frames."""
        history = []
        for snapshot in self.buffer:
            if entity_id in snapshot["entities"]:
                history.append(snapshot["entities"][entity_id])
        return history[-max_frames:]
```

### 2.3 Reasoning Pipeline

```python
# src/reasoning/pipeline.py

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Hypothesis:
    id: str
    claim: str
    status: str          # CANDIDATE, SUPPORTED, REFUTED, INCONCLUSIVE
    confidence: dict     # 5-component decomposition
    evidence: list
    priority: float

class ReasoningPipeline:
    """Event-triggered reasoning pipeline."""

    def __init__(self, config):
        self.hypothesis_engine = HypothesisEngine(config)
        self.fast_verifier = FastVerifier(config)
        self.vlm_reasoner = VLMReasoner(config)
        self.evidence_graph = EvidenceGraph()

    def process_event(self, event, world_state, memory):
        """Process an event through the reasoning pipeline."""
        # Step 1: Generate hypotheses
        hypotheses = self.hypothesis_engine.generate(event, world_state, memory)

        results = []
        for hyp in hypotheses:
            # Step 2: Fast verification (1-5ms)
            verdict = self.fast_verifier.verify(hyp, world_state)

            if verdict.status in ["SUPPORTED", "REFUTED"]:
                # Fast path resolved it
                results.append(verdict)
            else:
                # Step 3: Deep VLM reasoning (async, 800-3500ms)
                deep_result = self.vlm_reasoner.reason(
                    hyp, world_state.get_recent(10), memory
                )
                results.append(deep_result)

            # Step 4: Update evidence graph
            self.evidence_graph.add_evidence(hyp, verdict)

        return results
```

### 2.4 VLM Reasoning

```python
# src/reasoning/vlm_reasoner.py

import openai
import google.generativeai as genai
from typing import List, Optional

class VLMReasoner:
    """Vision-Language Model reasoning with API fallback."""

    def __init__(self, config):
        self.config = config
        self.mode = config.vlm_mode  # "local" or "api"

        if self.mode == "api":
            self.client = openai.OpenAI()
            self.model = "gpt-4o"  # or gemini
        else:
            # Local model (Qwen3-VL-30B-A3B)
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.vlm_model_path,
                torch_dtype="auto",
                device_map="auto",
            )

    def reason(self, hypothesis, state_snapshots, memory):
        """Generate VLM reasoning for a hypothesis."""
        # Build prompt
        prompt = self._build_prompt(hypothesis, state_snapshots, memory)

        if self.mode == "api":
            response = self._call_api(prompt)
        else:
            response = self._call_local(prompt)

        # Parse structured output
        result = self._parse_response(response, hypothesis)
        return result

    def _build_prompt(self, hypothesis, snapshots, memory):
        """Build VLM prompt with context."""
        prompt = f"""You are a video reasoning assistant. Analyze the following hypothesis about a video clip.

HYPOTHESIS: {hypothesis.claim}

CURRENT WORLD STATE:
{self._format_state(snapshots[-1])}

RECENT HISTORY (last 5 frames):
{self._format_history(snapshots[-5:])}

SIMILAR PAST EPISODES:
{self._format_episodes(memory.get_similar(hypothesis, k=3))}

TASK: Evaluate whether the hypothesis is SUPPORTED, REFUTED, or INCONCLUSIVE based on the evidence.
Provide your response as JSON:
{{
    "verdict": "supported|refuted|inconclusive",
    "confidence": {{"perception": 0.0-1.0, "temporal": 0.0-1.0, "reasoning": 0.0-1.0}},
    "evidence_refs": ["list", "of", "evidence", "ids"],
    "reasoning": "your explanation",
    "alternative_hypotheses": [{{"claim": "...", "probability": 0.0-1.0}}]
}}"""
        return prompt

    def _call_api(self, prompt):
        """Call VLM API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise video reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1,
        )
        return response.choices[0].message.content
```

### 2.5 Calibration Layer

```python
# src/calibration/pipeline.py

import json
import numpy as np

class CalibrationPipeline:
    """Confidence decomposition + conformal prediction + temperature scaling."""

    def __init__(self, config):
        self.config = config

        # Load trained calibrators
        with open("models/calibrator/temperature.json") as f:
            temp_data = json.load(f)
            self.temperature = temp_data["temperature"]

        with open("models/calibrator/conformal.json") as f:
            conformal_data = json.load(f)
            self.conformal_thresholds = conformal_data["thresholds"]

        with open("models/calibrator/decomposition.json") as f:
            decomp_data = json.load(f)
            self.decomposition_weights = decomp_data["weights"]

    def calibrate(self, hypothesis, evidence_graph):
        """Apply full calibration pipeline to a hypothesis."""
        # Step 1: Decompose confidence
        decomposed = self._decompose_confidence(hypothesis)

        # Step 2: Temperature scaling
        calibrated = self._temperature_scale(decomposed)

        # Step 3: Conformal prediction set
        prediction_set = self._conformal_set(hypothesis, evidence_graph)

        # Step 4: Staleness check
        staleness = self._check_staleness(hypothesis)

        return {
            "hypothesis_id": hypothesis.id,
            "claim_text": hypothesis.claim,
            "confidence": decomposed,
            "calibrated_confidence": calibrated,
            "prediction_set_alpha_05": prediction_set["alpha_05"],
            "prediction_set_alpha_10": prediction_set["alpha_10"],
            "staleness_ms": staleness["ms"],
            "stale": staleness["stale"],
            "epistemic_status": self._determine_epistemic(hypothesis),
        }

    def _decompose_confidence(self, hypothesis):
        """Decompose into 5 confidence components."""
        w = self.decomposition_weights
        return {
            "perception": hypothesis.confidence.get("perception", 0.5),
            "temporal": hypothesis.confidence.get("temporal", 0.5),
            "motion": hypothesis.confidence.get("motion", 0.5),
            "cross_modal_agreement": hypothesis.confidence.get("cross_modal", 0.5),
            "reasoning": hypothesis.confidence.get("reasoning", 0.5),
            "overall": (
                w["perception"] * hypothesis.confidence.get("perception", 0.5) +
                w["temporal"] * hypothesis.confidence.get("temporal", 0.5) +
                w["motion"] * hypothesis.confidence.get("motion", 0.5) +
                w["cross_modal"] * hypothesis.confidence.get("cross_modal", 0.5) +
                w["reasoning"] * hypothesis.confidence.get("reasoning", 0.5)
            ),
        }

    def _temperature_scale(self, decomposed):
        """Apply temperature scaling."""
        overall = decomposed["overall"]
        # Convert to logits, scale, convert back
        logit = np.log(overall / (1 - overall + 1e-10) + 1e-10)
        scaled_logit = logit / self.temperature
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        return calibrated

    def _conformal_set(self, hypothesis, evidence_graph):
        """Compute conformal prediction sets."""
        # Get all alternative hypotheses from evidence graph
        alternatives = evidence_graph.get_alternatives(hypothesis.id)

        # Compute nonconformity scores
        scores = []
        for alt in alternatives:
            score = self._nonconformity_score(hypothesis, alt)
            scores.append((alt.id, score))

        # Apply thresholds
        alpha_05 = [h.id for h, s in scores if s <= self.conformal_thresholds["0.05"]]
        alpha_10 = [h.id for h, s in scores if s <= self.conformal_thresholds["0.10"]]

        return {"alpha_05": alpha_05, "alpha_10": alpha_10}
```

---

## 3. End-to-End Pipeline

```python
# src/pipeline.py

class MultimodalReasoner:
    """Complete end-to-end pipeline."""

    def __init__(self, config):
        self.perception = PerceptionPipeline(config)
        self.state = WorldState(capacity=30)
        self.memory = MemoryManager(config)
        self.reasoning = ReasoningPipeline(config)
        self.calibration = CalibrationPipeline(config)
        self.scheduler = MultiRateScheduler(config)

    def process_video_stream(self, video_source):
        """Main processing loop."""
        cap = cv2.VideoCapture(video_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Create frame input
            frame_input = FrameInput(
                frame_id=self._next_frame_id(),
                timestamp_ns=time.time_ns(),
                image=frame,
                stream_id="main",
            )

            # Step 1: Perception (30 FPS, <12ms)
            perception_result = self.perception.process_frame(frame_input)

            # Step 2: State Update (<1ms)
            snapshot = self.state.update(
                frame_input.frame_id,
                perception_result["tracks"],
                perception_result["poses"],
            )

            # Step 3: Event Detection (<1ms)
            events = self._detect_events(snapshot)

            # Step 4: Scheduler decides priority
            for event in events:
                priority = self.scheduler.compute_priority(event)

                if priority >= 0.85:
                    # High priority: immediate reasoning
                    self._reason_and_output(event, snapshot)
                elif priority >= 0.5:
                    # Normal priority: queue for processing
                    self._queue_reasoning(event, snapshot)
                # Low priority: skip or background process

            # Step 5: Memory update
            self.memory.update(snapshot)

            # Report FPS
            self._report_metrics()

    def _reason_and_output(self, event, snapshot):
        """Full reasoning + calibration + output."""
        # Reasoning
        results = self.reasoning.process_event(event, snapshot, self.memory)

        # Calibration
        for result in results:
            calibrated = self.calibration.calibrate(result, self.reasoning.evidence_graph)

            # Output claim
            self._publish_claim(calibrated)

    def _publish_claim(self, calibrated_result):
        """Publish structured claim."""
        claim = {
            "id": f"claim_{uuid.uuid4().hex[:8]}",
            "schema_version": "1.0.0",
            "timestamp_ns": time.time_ns(),
            "claim_text": calibrated_result["claim_text"],
            "confidence": calibrated_result["confidence"],
            "calibrated_confidence": calibrated_result["calibrated_confidence"],
            "prediction_set_alpha_05": calibrated_result["prediction_set_alpha_05"],
            "prediction_set_alpha_10": calibrated_result["prediction_set_alpha_10"],
            "staleness_ms": calibrated_result["staleness_ms"],
            "stale": calibrated_result["stale"],
            "epistemic_status": calibrated_result["epistemic_status"],
        }

        # Publish to Kafka / REST API / WebSocket
        self.claim_publisher.publish(claim)
```

---

## 4. Deployment Configuration

### Docker

```dockerfile
# deployment/docker/Dockerfile

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Install TensorRT
RUN pip install tensorrt

# Copy our code
COPY src/ /app/src/
COPY models/trt_engines/ /app/models/trt_engines/
COPY models/calibrator/ /app/models/calibrator/
COPY configs/ /app/configs/

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Expose API port
EXPOSE 8000

# Run inference server
CMD ["python", "-m", "src.server"]
```

### REST API

```python
# src/server.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
reasoner = MultimodalReasoner.load("configs/production.yaml")

@app.get("/api/v1/state/current")
async def get_current_state():
    return reasoner.state.get_current_snapshot()

@app.get("/api/v1/claims")
async def get_claims(from_time=None, to_time=None):
    return reasoner.get_claims(from_time, to_time)

@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "gpu_memory_used_gb": torch.cuda.memory_allocated() / 1e9,
        "fps": reasoner.current_fps,
        "queue_depths": reasoner.scheduler.get_queue_depths(),
    }
```

---

## 5. Error Handling

Every component must handle errors gracefully:

```python
class PerceptionError(Exception):
    pass

class ReasoningTimeout(Exception):
    pass

class ModelNotLoaded(Exception):
    pass

# In pipeline:
try:
    result = self.perception.process_frame(frame)
except PerceptionError as e:
    # Log error, skip frame, continue with tracker prediction
    logger.warning(f"Perception failed on frame {frame.frame_id}: {e}")
    result = self._fallback_to_tracker_prediction(frame)

try:
    result = self.reasoning.process_event(event, snapshot, memory)
except ReasoningTimeout:
    # VLM timed out, use fast path result only
    result = self.fast_verifier.verify(hyp, snapshot)
```
