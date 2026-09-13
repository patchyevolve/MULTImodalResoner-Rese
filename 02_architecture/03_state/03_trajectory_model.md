# Trajectory Model Architecture

## Purpose

Model and predict entity trajectories — position, velocity, acceleration over time. Provides motion prediction for event detection (e.g., "ball is heading toward goal"), occlusion extrapolation (predicting where an occluded entity will reappear), and physics constraint validation.

---

## Interfaces

### Input

```
TrajectoryInput {
  entity_id:       string
  positions:       [][2]float               # normalized positions over time
  timestamps:      uint64[]                 # corresponding timestamps
  is_occluded:     bool                     # current occlusion state
  camera_motion:   CameraMotionOutput       # ego-motion to subtract
}
```

### Output

```
Trajectory {
  entity_id:       string
  predicted_positions: [H][2]float          # H-step ahead predictions
  predicted_timestamps: uint64[]
  velocity:        [2]float                 # current velocity (normalized/frame)
  acceleration:    [2]float                 # current acceleration
  trajectory_type: Enum                     # LINEAR | CURVED | RANDOM | STATIONARY | OSCILLATING
  prediction_confidence: float
  physics_valid:   bool                     # passes physics constraints
  inference_ms:    float
}
```

### API

```
predict_trajectory(entity_id: string, horizon: int) -> Trajectory
update_trajectory(entity_id: string, position: [2]float, timestamp: uint64) -> Trajectory
validate_physics(trajectory: Trajectory) -> bool
```

---

## Data Contracts

### Prediction Methods

| Method | Latency | Accuracy | Notes |
|---|---|---|---|
| Constant velocity | 0.01ms | Good (short-term) | Default, simplest |
| Constant acceleration | 0.02ms | Better (curved) | For balls, vehicles |
| Kalman filter | 0.1ms | Good | Handles noise |
| Polynomial fit | 0.2ms | Good (smooth) | For predictable motion |
| Learned predictor | 2–5ms | Best | For complex patterns |

### Physics Constraints

```
Constraints:
  - max_velocity: float          # e.g., 2.0 normalized units/frame
  - max_acceleration: float      # e.g., 0.5 normalized units/frame²
  - min_turn_radius: float       # e.g., 0.1 normalized units
  - gravity: float               # for ball trajectories
  - surface_friction: float      # for ground-plane motion
```

### Configuration

```
TrajectoryConfig {
  method:          Enum                     # CONSTANT_V | KALMAN | POLYNOMIAL
  prediction_horizon: int                   # frames ahead, default 10
  max_history:     int                      # frames for fitting, default 15
  physics_enforcement: bool                 # default true
}
```

---

## Latency Budget

| Operation | Budget | Notes |
|---|---|---|
| Single prediction | 0.01–0.1ms | Constant velocity / Kalman |
| Multi-step prediction | 0.1–0.5ms | Polynomial fit |
| Physics validation | 0.01ms | Simple bounds check |
| **Total** | **0.1–0.5ms** | |

---

## Dependencies

### Upstream
- `03_state/02_entity_tracker.md` — Entity position history
- `01_perception/08_camera_motion.md` — Ego-motion subtraction

### Downstream
- `03_state/04_event_detection.md` — Trajectory-based events
- `05_reasoning/05_prediction_model.md` — Future state prediction
- `05_reasoning/01_hypothesis_engine.md` — Trajectory evidence

---

## Reality Check 2026

### Ball Trajectory:
- Soccer ball: parabolic trajectory with gravity, measurable from 2D pose.
- Basketball: bounces, spin effects — polynomial fit works for short-term.
- Tennis: high-speed, predictable arcs — constant acceleration model sufficient.

### Player Trajectory:
- Linear motion: running, walking — constant velocity.
- Curved motion: cutting, turning — Kalman filter with acceleration.
- Random: fakes, stutter steps — short prediction horizon only.

### Key Insight:
- Trajectory prediction confidence degrades rapidly beyond 10 frames (333ms).
- Use trajectory for **event triggering** (threshold-based), not long-term prediction.
- Physics validation catches impossible states (teleportation, through-wall).
