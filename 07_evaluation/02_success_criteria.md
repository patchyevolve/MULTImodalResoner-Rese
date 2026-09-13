# Success criteria

The project should not declare success from a single VLM QA score.

A credible milestone should demonstrate:

1. **30 FPS continuous state tracking** on a defined hardware target (RTX 4090 or T4). Critical path latency <15ms. Sustained p99 <33ms.
2. **Hidden-state reconstruction with calibrated uncertainty.** MoRo-level accuracy (37.83mm visible, 48.53mm occluded) with 90% coverage on predicted confidence ellipses.
3. **Multiple-hypothesis reasoning with evidence traceability.** Top-3 hypothesis recall ≥85% on ambiguous clips. Evidence graph with typed edges (supports/contradicts).
4. **Better confidence calibration** than uncalibrated baseline. Target: final claim ECE ≤5%. VL-Calibration-level improvement (0.421→0.098).
5. **Robustness to occlusion and missing modalities.** <40% degradation at 50% occlusion. Correct inconclusive output when audio dropped.
6. **No blocking of the streaming path** by deep reasoning. Queue depth bounded. Staleness <1500ms for VLM-derived claims, <50ms for perception-derived.
7. **Measurable benefit from prediction-error/uncertainty-triggered compute.** ≥15% accuracy improvement over fixed-rate at same compute budget.
8. **Generalization** across at least general video + a fast domain such as sports. <10pp degradation across 3+ domain shifts.
9. **A nontrivial synthetic-media forensic branch.** Outputs inconclusive (not confident-wrong) on >80% of adversarially attacked samples. Multi-detector corroboration.
10. **Reproducible latency and accuracy measurements.** Full metric breakdowns per experiment. Hardware target specified (GPU model, CUDA version, driver version).
