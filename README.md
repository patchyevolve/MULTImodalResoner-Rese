# Real-Time Multi-Factor Multimodal Reasoner

## Research workspace

Central objective:

> Design a general multimedia reasoner that continuously maintains a calibrated latent belief about a changing world, can infer partially or fully unobserved states, generate and falsify competing hypotheses, combine evidence across modalities, reason about natural and artificial media, and sustain a 30 FPS perceptual/state-update loop while expensive reasoning runs asynchronously.

## Core thesis

The target is **not** a monolithic 30-FPS Video-Language Model.

The target is a **multi-rate streaming reasoning system**:

```text
video / image / audio / metadata
            |
            v
     fast perception
            |
            v
     continuous world state  <---- 30 Hz target
            |
       +----+----+----------------+
       |         |                |
       v         v                v
   temporal    motion         evidence
    memory     model           graph
       |         |                |
       +---------+----------------+
                 |
                 v
       hypothesis generation
                 |
                 v
       evidence / contradiction tests
                 |
                 v
       calibrated uncertainty
                 |
          +------+------+
          |             |
          v             v
      fast reasoner   deep reasoner
          |             |
          +------+------+
                 v
     claims + evidence + confidence
```

## Primary research question

**What is the minimum computational architecture capable of maintaining a calibrated latent belief about a continuously changing multimodal world, reconstructing unobserved states, generating and falsifying hypotheses, and updating that belief at 30 Hz?**

## Target domains

1. General images and videos
2. Natural multimodal media (video + audio + text + metadata)
3. Fast media such as sports
4. Artificial/synthetic/generated images and videos
5. Partially observable and occluded scenes

## Important distinction

The 30 FPS constraint applies primarily to:

- input/decode pipeline
- tracking
- state estimation
- lightweight perception
- event/anomaly detection

It does **not** require a large VLM/LLM reasoning pass on every frame.

## Folder map

- `00_core/` — thesis, terminology, design principles
- `01_foundations/` — 18 research tracks with detailed questions
- `02_architecture/` — proposed system architecture and data contracts
- `03_models/` — model-family research and selection criteria
- `04_uncertainty/` — confidence, calibration, evidence semantics
- `05_realtime/` — 30 FPS execution, scheduling, optimization
- `06_domains/` — sports, general media, artificial media
- `07_evaluation/` — benchmarks, metrics, experiment matrix
- `08_implementation/` — staged engineering plan
- `09_sources/` — source index and literature map

## Design principles

- Separate observation from inference.
- Preserve alternative hypotheses rather than collapsing early.
- Make evidence traceable to timestamps, regions, modalities and models.
- Propagate uncertainty through the reasoning chain.
- Treat prediction error as a trigger for expensive computation.
- Prefer object/event/state representations over raw frame accumulation.
- Use multi-rate computation instead of forcing every component to run at 30 FPS.
- Treat synthetic-media detection as an evidence problem, not a binary classifier.
- Evaluate calibration and temporal consistency, not only answer accuracy.
