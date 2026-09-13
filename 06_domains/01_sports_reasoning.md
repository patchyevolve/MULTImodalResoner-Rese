# Sports reasoning domain

## Why sports is useful
Sports provides:
- rapid temporal events
- defined entities
- repeated event patterns
- known rules
- trajectories and physics
- available annotations
- strong before/after signals

## Research targets
- action recognition
- action anticipation
- player tracking
- ball tracking
- pose/biomechanics
- tactical relations
- foul/violation hypotheses
- event localization
- commentary generation grounded in evidence

## Example
Observed foot approach + ball trajectory change + compatible pose transition -> hypothesis of contact. If direct contact is occluded, posterior depends on temporal and physics evidence.

---

## REALITY CHECK 2026 (Verified Sports AI SOTA)

### ACTION SPOTTING
- SoccerNet 2025 Team Ball Action Spotting: Best = **60.03% Team-mAP@1** (dudekTBAS-1), 2nd = 56.78% (Intellindust-AI-Lab), 3rd = 56.08% (KIST-BAS)
- T-DEED baseline: 51.72% Team-mAP@1
- COMEDIAN (WACV 2024): SOTA on SoccerNet-v2 tight-tolerance

### ACTION ANTICIPATION (5s future window, Football)
| Tolerance | Baseline | Best 2026 Challenge (FAANTRA-WS) |
|-----------|----------|-----------------------------------|
| @1s | 5.70% | **9.02%** |
| @2s | 13.00% | **18.18%** |
| @3s | 16.30% | **23.72%** |
| @4s | 19.18% | **27.92%** |
| @5s | 21.02% | **30.18%** |
| @infinity | 22.86% | **31.78%** |
| **Avg mAP** | 16.76% | **24.08%** |

**Reality:** Top-1 accuracy for next action at 2s horizon is ~18%. Strategic predictions are hypotheses with huge uncertainty.

### PLAYER/BALL TRACKING
- SAM 3-Deep-EIoU: HOTA = **87.2** on SportsMOT (new SOTA, Jun 2026)
- McByte++ (training-free, 2026): HOTA = **85.0** on SoccerNet-Tracking, MOTA = 96.8
- SAMIDARE (CVPR 2026W): HOTA = **85.6** on SportsMOT
- Sports-LiteDet (Nature 2026): **68 FPS** on Jetson Orin Nano, 5.8 MB model, 71.8% mAP@0.5

### SOCCERNET 2025 TASK RANKINGS
- Team Ball Action Spotting: 1st = dudekTBAS-1 **60.03%** Team-mAP@1, 2nd = Intellindust-AI-Lab 56.78%, 3rd = KIST-BAS 56.08%
- Monocular Depth Estimation (NEW): 1st = Hands-On Computer Vision RMSE **2.418×10⁻³** (vs ZoeDepth baseline 3.757×10⁻³)
- Multi-View Foul Recognition: 1st = UniBW Munich **52.22%** balanced accuracy (vs baseline 36.99%)
- Game State Reconstruction: 1st = KIST-GSR **63.90** GS-HOTA (vs baseline 29.01)

### SOCCERNET 2026 TASK RANKINGS (NEW tasks)
- Ball Action Anticipation: 1st = FAANTRA-WS **24.08** mAPavg (vs baseline 16.76)
- Player-Centric Ball Action Spotting: 1st = FSITAHAKOM **58.94** F1@0.15
- Novel View Synthesis: 1st = Sarthi-GameChanger **29.89** PSNR
- Visual Question Answering: 1st = vitomeme **98.0%** accuracy (76 submissions)

### SPORTS-SPECIFIC VLMs
- SoccerMaster (CVPR 2026 Oral): Athlete detection AP@50 = **92.3**, Event classification = **73.8%**, Jersey number accuracy = **80.0%**, Role accuracy = **99.2%**, GS-HOTA = **64.1** (beats KIST-GSR 63.9)
- DeepSport (Nov 2025): First end-to-end MLLM for 12 sports, **37.67 overall** (beats GPT-5 at 35.70, Qwen3-VL-235B at 35.36), uses only **9.81 frames** on average
- SportsGrounder (ACM MM 2026): **51.8%** on SoccerNet VQA, **53.6%** on FineSports VQA with only 2B params

### PRODUCTION DEPLOYMENTS
- FIFA World Cup 2026: Advanced SAOT with **10 cm offside threshold** (upgraded from 50 cm), **16 cameras per match**, **50 fps tracking**, **>150 million data points per match**, **1,248 player avatars**, review time **25–70 seconds**, **500 Hz IMU** in Trionda ball, audio alerts to assistant referees
- NBA "Inside the Game" (AWS, 2025-26): **2,500 events/second** peak, ~10 simultaneous games, **29 body points × 60 Hz** per player, new metrics: Leverage Score, Shot Difficulty (xFG%), Player Gravity (transformer-based)
- Hawk-Eye (AWS, Oct 2025): Migrated to Amazon MSK + Managed Flink, **480 messages/second**, 15–125 MB/s per game, 50% storage cost reduction, 60% TCO reduction
- Tennis (Wimbledon 2025-2026): Fully electronic line calling, ~12 cameras per court at **340 fps**, SkeleTRACK **29-point body tracking**, HawkAR augmented reality overlays (2026)
- IPL Cricket (2026): Hawk-Eye ball tracking with 3D trajectory reconstruction, smart stumps with LED circuits + pressure sensors
- Neuron Systems: 104 matches, **42ms glass-to-glass latency**, 5.14M events on Final day
- MLB Scout Insights: Sub-2-second latency with Gemini 2.5 Flash

### CVPR 2025/2026 SPORTS PAPERS
- CVPR 2025: MatchVision, Towards Universal Soccer Video Understanding, MANTA (65× speedup for anticipation)
- CVPR 2026: SoccerMaster (Oral), SAMIDARE, SportMamba, SportR Benchmark
