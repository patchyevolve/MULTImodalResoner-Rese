# Deployment Architecture

## Purpose

Define how the system is deployed — container configurations, orchestration, monitoring, and operational procedures. From single-GPU development to multi-node production clusters.

---

## Deployment Configurations

### Docker Compose (Development)

```yaml
version: "3.8"
services:
  perception:
    image: multimodal-reasoner/perception:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./config:/config
    ports:
      - "8080:8080"  # REST API

  reasoning:
    image: multimodal-reasoner/reasoning:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - perception

  scheduler:
    image: multimodal-reasoner/scheduler:latest
    depends_on:
      - perception
      - reasoning
```

### Kubernetes (Production)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multimodal-reasoner
spec:
  replicas: 2
  selector:
    matchLabels:
      app: multimodal-reasoner
  template:
    spec:
      containers:
      - name: perception
        image: multimodal-reasoner/perception:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
        volumeMounts:
        - name: config
          mountPath: /config
      - name: reasoning
        image: multimodal-reasoner/reasoning:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "24Gi"
      volumes:
      - name: config
        configMap:
          name: reasoning-config
```

---

## Monitoring Metrics

### Critical Path Metrics

| Metric | Alert Threshold | Notes |
|---|---|---|
| Frame latency p50 | >15ms | Perception too slow |
| Frame latency p99 | >28ms | Approaching 33ms deadline |
| GPU utilization | >95% sustained | Need downgrade |
| GPU memory | >90% | Approaching OOM |
| VLM queue depth | >3 for >30s | Backpressure needed |
| Dropped frames | >0 per minute | Perception failure |
| Dropped VLM jobs | >0 per minute | VLM overload |

### System Health Metrics

| Metric | Check Interval | Notes |
|---|---|---|
| Model freshness | 1 hour | Are models loaded correctly? |
| Schema version | On startup | Version mismatch detection |
| Calibration freshness | 24 hours | Need recalibration? |
| Storage capacity | 1 hour | Long-term memory full? |

---

## Operational Procedures

### Startup Sequence

```
1. Load configuration
2. Initialize GPU context + CUDA streams
3. Load models (detection → pose → VLM → forensic)
4. Initialize ring buffers (pre-allocate)
5. Start perception pipeline
6. Start scheduler
7. Start async workers (VLM, forensic, summary)
8. Start API server
9. Health check
10. Ready for streams
```

### Shutdown Sequence

```
1. Stop accepting new streams
2. Drain VLM queue (finish pending, don't start new)
3. Flush long-term memory
4. Save calibration state
5. Release GPU resources
6. Exit
```

### Failure Recovery

```
Perception failure:
  1. Log error
  2. Restart perception component
  3. Re-initialize tracker (lose tracks, restart fresh)
  4. Resume

VLM failure:
  1. Log error
  2. Restart VLM component
  3. VLM queue survives (ring buffer)
  4. Resume processing queue

GPU OOM:
  1. Emergency: downgrade perception to N model
  2. Reduce VLM queue depth to 2
  3. Log critical error
  4. If persistent: restart with reduced model sizes
```

---

## Configuration Management

### Environment Variables

```
# GPU
CUDA_VISIBLE_DEVICES=0,1
GPU_PERCEPTION=0
GPU_REASONING=1

# Models
DETECTION_MODEL=RF-DETR-S
POSE_MODEL=DETRPose-S
VLM_MODEL=Qwen3-VL-30B-A3B

# Scheduler
MAX_VLM_QUEUE=4
MAX_PERCEPTION_FPS=30
BACKPRESSURE_THRESHOLD=0.85

# Storage
LONG_TERM_DB_PATH=/data/longterm
EPISODIC_DB_PATH=/data/episodic
VECTOR_DB_PATH=/data/vectors

# API
API_PORT=8080
API_HOST=0.0.0.0
```

---

## Reality Check 2026

### NVIDIA VSS Deployment:
- Docker Compose for development.
- Helm charts on K8s for production.
- DeepStream SDK 9.1 with GStreamer-based pipelines.
- Service Maker: C++/Python pipeline composition.

### Production Considerations:
- **Health checks** must verify GPU is responsive, not just process alive.
- **Graceful degradation** over crash — downgrade models, reduce queue, never block 30 FPS.
- **Observability** — log every dropped frame, every VLM timeout, every backpressure event.
- **Rolling updates** — perception must be updated without stopping video streams.
