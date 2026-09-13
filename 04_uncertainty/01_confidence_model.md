# Confidence model

## Recommended decomposition

```text
raw observation quality
        |
perception confidence
        |
state-estimation confidence
        |
temporal consistency
        |
cross-modal agreement
        |
hypothesis support / contradiction
        |
reasoning confidence
        |
calibration
        v
final claim confidence
```

## Never expose only one number internally
Store the components so downstream systems can determine why confidence is low.

## Candidate methods to compare
- temperature scaling
- isotonic calibration
- conformal prediction
- ensemble disagreement
- evidential approaches
- learned calibration network

## Required tests
- reliability diagrams
- ECE
- Brier score
- coverage at fixed risk
- confidence under occlusion
- confidence under modality failure
- confidence under synthetic manipulation

---

## REALITY CHECK 2026: CALIBRATION STATE OF THE ART

### VL-Calibration (ACL 2026, arXiv:2604.09529):
- RL-based framework (GRPO) that decouples verbalized confidence into visual + reasoning confidence.
- Intrinsic visual certainty: KL-divergence under image perturbation + token entropy.

| Model | Base ECE | VL-Calibration ECE | Reduction | AUROC (hallucination) |
|---|---|---|---|---|
| Qwen3-VL-4B | 0.421 | **0.098** | 76.7% | 0.763 |
| Qwen3-VL-8B | 0.401 | **0.071** | 82.3% | 0.764 |
| Qwen3-VL-30B | 0.388 | **0.082** | 78.9% | 0.767 |

- Training overhead: 11% (adds 15s to 140s step for 8B). Fast convergence: ECE → 0.1 in <100 steps.

### Conformal Prediction for VLMs (EACL 2026):
- 18 VLMs, 6 datasets, 21,000 questions. Larger models = better uncertainty.
- Mathematical/reasoning tasks elicit poorest uncertainty across ALL models.
- Closed-source models: "instruction-guided likelihood proxies" developed for logprob-free CP.

### Empirical Bayes Conformal (arXiv:2605.23189, May 2026):
- CP_r-value reduces prediction set sizes by 7-44% vs standard CP at α=0.05.
- ImageNet: 5.7 vs 10.1 (43.6% reduction). CIFAR-100: 7.8 vs 11.3 (31% reduction).
- Gains shrink as model accuracy increases (33.5% at 65.6% acc → 4.0% at 79.5% acc).

### Proof-of-Perception (CVPR 2026):
- Multimodal reasoning as executable DAG with conformal sets at each node.
- +4.2% DocVQA, +3.6% ChartQA over CoT/ReAct/PoT baselines.
- Adaptive controller uses conformal set size to decide accept/retry/expand.

### VLM-UQBench (arXiv:2602.09214, Feb 2026):
- 9 UQ methods, 4 VLMs, 3 datasets. Visual uncertainty is most challenging modality.
- LLaVA degrades to near-random 0.55 AUROC on visual uncertainty.
- UQ methods are highly model-dependent — optimal method shifts per architecture.
- Hallucination rate under visual blur: 64.7% (Qwen-VL, attribute tasks).

### EDL Warning (arXiv:2310.12663):
- EDL's "evidential signal" from Dirichlet strength is due to misclassification bias, not true uncertainty.
- EDL couples aleatoric and epistemic uncertainty through KL regularization.
- Use evidential approaches with caution; prefer conformal or temperature scaling.

### Real-Time Uncertainty Overhead:
- **No method achieves 1-5ms overhead for full VLMs in 2026.**
- HARMONY (113M proxy model): ~23ms — closest to real-time but requires training a separate model.
- VAUQ (LLaVA-1.5-7B): 0.73s/sample — comparable to inference but not real-time.
- VL-Calibration: 11% overhead — acceptable for async (0.3-0.5 Hz) but not critical path.

### Distribution Shift Impact:
- VLM-RobustBench: geometric distortions cause up to 34pp accuracy drops. Low-severity often worse than severe.
- STREAM-OOD (CVPR 2026W): reduces OOD false alarms from 3.4→1.6/hr on NYC traffic streams.
- Worst-class coverage falls to ≈0 under domain shift despite marginal coverage of ~0.86 (ImageNet-Sketch).
