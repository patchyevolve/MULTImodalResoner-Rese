# Timeline & Milestones

> Day-by-day execution plan. Every day has a clear deliverable. No ambiguity.

> **Hardware note:** Days 1-10 run on RTX 3070 8GB (temporary). Day 10+ migrate to H100. Full hardware strategy in `12_training_system.md`. Batch sizes auto-adapt — no config changes needed at migration.

---

## Phase 0: Pre-Training Preparation (Days 1-3)

| Day | Task | Duration | Deliverable | Done? |
|---|---|---|---|---|
| D1 | GPU verification + CUDA/PyTorch install | 2h | `nvidia-smi` output, PyTorch CUDA test passing | [ ] |
| D1 | Install all pip dependencies | 1h | `pip list` shows all packages | [ ] |
| D1 | Download pre-trained model weights | 2h | All models in `models/pretrained/` | [ ] |
| D1 | Set up W&B account + logging | 30m | W&B project created | [ ] |
| D2 | Run GPU benchmark verification | 1h | Benchmarks match published numbers | [ ] |
| D2 | Create repository code structure | 1h | All directories exist | [ ] |
| D2 | Write configuration templates | 1h | All YAML configs in `configs/` | [ ] |
| D2 | Smoke test: load each model, run dummy inference | 2h | All models load and run | [ ] |
| D3 | Download COCO 2017 | 2-4h | `data/raw/coco/` populated | [ ] |
| D3 | Download MOT17 | 1-2h | `data/raw/mot/` populated | [ ] |
| D3 | Download Market1501 | 10m | `data/raw/market1501/` populated | [ ] |
| D3 | Download SoccerNet v2 | 2-4h | `data/raw/soccernet/` populated | [ ] |
| D3 | Collect custom video clips | 2-3h | 10-20 clips in `data/raw/custom/` | [ ] |

**Phase 0 Exit Criteria:**
- [ ] All models load and run on GPU
- [ ] All datasets downloaded
- [ ] GPU benchmarks match published numbers (±10%)
- [ ] W&B logging working
- [ ] Code structure complete

---

## Phase 1: Dataset Preprocessing (Days 3-5)

| Day | Task | Duration | Deliverable | Done? |
|---|---|---|---|---|
| D3-D4 | Preprocess COCO (verify, split, extract crops) | 3h | `data/processed/coco/` with splits | [ ] |
| D4 | Preprocess MOT17 | 2h | `data/processed/mot/` | [ ] |
| D4 | Preprocess Market1501 | 30m | `data/processed/market1501/` | [ ] |
| D4-D5 | Preprocess SoccerNet v2 | 3h | `data/processed/soccernet/` | [ ] |
| D5 | Annotate custom video clips | 4-8h | `data/processed/custom/` with annotations | [ ] |
| D5 | Run data quality checks | 1h | All checks passing | [ ] |
| D5 | Set up DVC versioning | 30m | DVC tracking all datasets | [ ] |

**Phase 1 Exit Criteria:**
- [ ] All datasets preprocessed and split
- [ ] Data quality checks passing
- [ ] DVC versioning working
- [ ] Custom annotations complete

---

## Phase 2: Model Fine-Tuning (Days 5-14)

| Day | Task | Duration | Deliverable | Done? |
|---|---|---|---|---|
| D5-D7 | Fine-tune RF-DETR-S (detection) | 2-3 days | `models/checkpoints/rf_detr_s/best.pth` | [ ] |
| D7 | Export RF-DETR-S to TensorRT | 2h | `models/trt_engines/rf_detr_s_fp16.engine` | [ ] |
| D7 | Benchmark RF-DETR-S latency | 1h | Benchmark results in W&B | [ ] |
| D7 | Verify RF-DETR-S mAP ≥ 53.0 | 30m | COCO val evaluation passing | [ ] |
| D8-D9 | Fine-tune OSNet (Re-ID) | 1-2 days | `models/checkpoints/osnet/best.pth` | [ ] |
| D9 | Export OSNet to TensorRT | 1h | `models/trt_engines/osnet_fp16.engine` | [ ] |
| D9 | Verify OSNet Rank-1 > 95% | 30m | Market1501 evaluation passing | [ ] |
| D10-D12 | Fine-tune RF-DETR-Seg-S (segmentation) | 2-3 days | `models/checkpoints/rf_detr_seg_s/best.pth` | [ ] |
| D12 | Export RF-DETR-Seg-S to TensorRT | 2h | `models/trt_engines/rf_detr_seg_s_fp16.engine` | [ ] |
| D13 | Train GBDT hypothesis ranker | 2-4h | `models/hypothesis_ranker/ranker.json` | [ ] |
| D13 | Train calibrator (temperature + conformal) | 4h | `models/calibrator/` files | [ ] |
| D13 | Verify calibration ECE < 0.05 | 30m | Calibration quality report | [ ] |

**Phase 2 Exit Criteria:**
- [ ] RF-DETR-S: mAP ≥ 53.0, latency p50 < 2.5ms (TensorRT)
- [ ] OSNet: Rank-1 > 95%, latency p50 < 1.5ms
- [ ] RF-DETR-Seg-S: mAP ≥ 43.0 (optional, skip if behind)
- [ ] Hypothesis Ranker: NDCG@3 ≥ 0.85, latency < 1ms
- [ ] Calibrator: ECE < 0.05, coverage ≥ 95%
- [ ] All models exported to TensorRT (GPU models)
- [ ] All benchmarks logged to W&B

---

## Phase 3: Integration (Days 14-21)

| Day | Task | Duration | Deliverable | Done? |
|---|---|---|---|---|
| D14 | Integrate detection + pose + tracking | 3h | Perception pipeline running | [ ] |
| D14 | Integrate state management (ring buffer) | 2h | World state updating at 30 FPS | [ ] |
| D15 | Integrate event detection + R score | 2h | Events triggering on sample video | [ ] |
| D15 | Integrate hypothesis engine + fast verifier | 3h | Fast reasoning resolving 80% of events | [ ] |
| D16 | Integrate VLM reasoning (API first) | 3h | Deep reasoning producing claims | [ ] |
| D16 | Integrate evidence graph | 2h | Evidence accumulating across frames | [ ] |
| D17 | Integrate calibration layer | 3h | Claims with calibrated confidence | [ ] |
| D17 | Integrate memory (short/long/episodic) | 2h | Historical context feeding VLM | [ ] |
| D18 | Integrate scheduler (multi-rate + backpressure) | 3h | Priority scheduling working | [ ] |
| D18 | End-to-end pipeline test on sample video | 2h | Full pipeline running | [ ] |
| D19 | Fix integration bugs | 4h | No crashes on 5-minute test | [ ] |
| D19 | Set up REST API (FastAPI) | 2h | API endpoints responding | [ ] |
| D20 | Docker container build + test | 3h | Container running pipeline | [ ] |
| D20 | Performance tuning (CUDA graphs, pinned memory) | 2h | FPS improved by 20%+ | [ ] |
| D21 | Final integration testing | 3h | Pipeline stable for 30 minutes | [ ] |

**Phase 3 Exit Criteria:**
- [ ] Full pipeline runs on sample video without crashes
- [ ] Claims output as structured JSON
- [ ] REST API responding
- [ ] Docker container working
- [ ] Pipeline stable for 30+ minutes continuous

---

## Phase 4: Benchmarking & Demo (Days 21-28)

| Day | Task | Duration | Deliverable | Done? |
|---|---|---|---|---|
| D21-D22 | Component latency benchmarks | 4h | All component latencies measured | [ ] |
| D22-D23 | End-to-end latency benchmark | 3h | E2E latency measured | [ ] |
| D23 | Detection accuracy benchmark (mAP) | 2h | mAP results on COCO val | [ ] |
| D23 | Tracking accuracy benchmark (MOTA) | 2h | MOTA/IDF1/HOTA on MOT17 | [ ] |
| D24 | Re-ID accuracy benchmark | 2h | Rank-1/mAP on Market1501 | [ ] |
| D24 | Calibration quality benchmark | 2h | ECE, coverage, set size | [ ] |
| D25 | Throughput benchmark (sustained FPS) | 3h | Sustained ≥30 FPS confirmed | [ ] |
| D25 | Stress test (multi-stream) | 2h | Results for 1/2/4 streams | [ ] |
| D26 | Comparison against baselines | 4h | Comparison table + plot | [ ] |
| D26 | Resource usage monitoring | 2h | GPU/CPU/Memory usage over time | [ ] |
| D27 | Demo video recording | 3h | Polished demo video | [ ] |
| D27 | Final paper writing | 4h | Paper with all results | [ ] |
| D28 | Final presentation preparation | 3h | Slide deck + live demo | [ ] |
| D28 | Buffer day (contingency) | 8h | Handle unexpected issues | [ ] |

**Phase 4 Exit Criteria:**
- [ ] All benchmarks complete and documented
- [ ] Demo video recorded
- [ ] Paper written with results
- [ ] Presentation ready
- [ ] All code committed to GitHub

---

## Milestone Summary

| Milestone | Day | What |
|---|---|---|
| **M0: Environment Ready** | D1-D2 | GPU working, models loaded, deps installed |
| **M1: Data Ready** | D3-D5 | All datasets preprocessed and versioned |
| **M2: Detection Trained** | D7 | RF-DETR-S fine-tuned, benchmarked, exported |
| **M3: All Models Trained** | D13 | All 5 models done (incl. ranker + calibrator) |
| **M4: Pipeline Integrated** | D21 | Full pipeline running end-to-end |
| **M5: Benchmarks Complete** | D26 | All metrics measured and documented |
| **M6: Demo Ready** | D28 | Demo video, paper, presentation |

---

## Risk Contingency

| Risk | Impact | Contingency |
|---|---|---|
| GPU not available on time | Delays everything | Start with API-based VLM, CPU perception (YOLO26-N) |
| Training takes longer than expected | Fewer models trained | Skip segmentation (Phase 2, Model 3) |
| Integration bugs take too long | Demo not ready | Simplify: detection + tracking + VLM API only |
| VLM quality insufficient | Poor reasoning | Fine-tune with few-shot examples, add more context |
| TensorRT export fails | Inference too slow | Use PyTorch inference with torch.compile |

---

## Daily Standup Format

Every day, answer:

1. **What did I do yesterday?**
2. **What will I do today?**
3. **What's blocking me?**
4. **Am I on track for the next milestone?**

---

## Communication

| Channel | Purpose | Frequency |
|---|---|---|
| GitHub Issues | Bug tracking, task assignment | As needed |
| Group Chat | Quick questions, status updates | Daily |
| Weekly Sync | Progress review, planning | Weekly |
| W&B Dashboard | Training metrics, benchmarks | Real-time |
| Demo Recording | Progress demonstrations | After each milestone |
