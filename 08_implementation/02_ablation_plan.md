# Ablation plan

The research should isolate where gains originate. Each ablation records BOTH quality metrics AND compute/latency.

| Ablation | Expected Impact | Measurement |
|---|---|---|
| frame-only VLM vs streaming state model | Streaming state should improve temporal consistency by 15–30% | Event ordering accuracy, temporal consistency score |
| no hidden-state model vs hidden-state model | Hidden-state should reduce occlusion error by 20–40% | MPJPE during occlusion (target: <40% degradation at 50% occlusion) |
| no temporal memory vs bounded memory | Bounded memory should improve long-form retrieval by 20–40% | State continuity across 30+ minute sessions |
| single hypothesis vs multiple hypotheses | Multi-hypothesis should improve recall@3 by 15–25% | Correct hypothesis in top-3 on ambiguous clips |
| no contradiction checking vs falsification | Falsification should reduce false positive claims by 20–30% | False causal claim rate |
| raw confidence vs calibrated confidence | Calibration should reduce ECE from 10–42% to <5% | ECE, Brier score, risk-coverage AUC |
| always-on deep reasoning vs event-triggered reasoning | Event-triggered should match accuracy at 30–50% compute cost | Accuracy vs GPU util % Pareto curve |
| pixels-only reasoning vs evidence-graph reasoning | Evidence graph should improve hypothesis precision by 10–20% | Precision@1 on hypothesis ranking |
| no physics prior vs physics-aware inference | Physics should reject 80%+ of physically impossible poses | Physical plausibility rejection rate |
| single modality vs reliability-aware multimodal fusion | Fusion should improve accuracy by 5–15% on multimodal clips | Accuracy under modality corruption |

Record both quality and compute for every ablation. Show Pareto curves: X-axis = compute budget (GPU util %, dollar cost), Y-axis = accuracy/calibration.
