# 18. Training and adaptation strategy

## Objective
Determine whether to train the entire system end-to-end or compose specialized pretrained modules.

## Investigate
- Pretrained perception + learned reasoner
- Fine-tuning
- LoRA/adapters
- Distillation
- Self-supervised learning
- Synthetic data
- Reinforcement learning
- Preference/critic training
- Hypothesis-verification training
- Calibration-aware training
- Continual learning
- Domain adaptation

## Central question
Which capabilities need joint optimization and which benefit from modularity?

## Deliverable
Training plan with compute estimates, data requirements and ablation plan.

---

## REALITY CHECK 2026

### ✅ DEFAULT, NO-BRAINER STRATEGY (90% of systems will use this):
**Compose pretrained modules.** Do NOT train end-to-end. Training everything together is prohibitively expensive, destroys calibration, and gains are marginal for this architecture because:
- Perception models are already trained on billions of images (RF-DETR / YOLO are pretrained on COCO + Object365 etc.).
- VLMs are already trained on trillion-token multimodal corpora.
- Fine-tuning just the connectors + calibration heads + small hypothesis ranking model gives 95% of the value at 5% of the cost.

### ⚠️ COMPONENT-WISE TRAINING STRATEGY, concrete:
| Component | Strategy | Approximate Compute | Data Requirements |
|---|---|---|---|
| **Detector / tracker / pose** | Use pretrained RF-DETR + ByteTrack. Fine-tune on domain dataset (sports) with LoRA if domain gap is large. | 1–8 A100-days for LoRA fine-tune. | 5K–50K domain frames with boxes/keypoints. |
| **Perception→state embeddings connector** | Small MLP or linear map. Train from scratch with backprop from downstream. | <1 GPU day. | Same as above. |
| **Hidden-state inference module** | Fine-tune HybrIK-style IK + temporal smoothing on 3D pose datasets (3DPW, Fit3D, ARCTIC for hands). Kalman parameters are hand-tuned or grid-searched on calibration set. | 2–10 GPU days. | 3DPW + Human3.6M + Fit3D (~1M frames total). |
| **Hypothesis ranking / verifier** | Small transformer or GBDT (XGBoost/LightGBM) on top of structured evidence graph features. Train with LTR (learning-to-rank) loss. | <1 GPU day for GBDT; 1–5 GPU days for small transformer. | 10K–100K labeled (event, hypotheses, correct rank) clips. Can be semi-automated via VLM adjudication. |
| **Confidence calibration transforms** | Temperature scaling / isotonic regression / SCP — all post-hoc, NO retraining of backbone. | 0 additional training; just calibration set eval. | ≥1000 samples per claim type in calibration split. |
| **VLM deep reasoning** | Prompt engineering + few-shot + tool use first. If quality insufficient: QLoRA fine-tuning of 7–32B VLM on domain QA pairs. GRPO variants for video reasoning (Temporal-R1, VIPO-R1, GRPO-CARE). | Prompt eng: 0 compute. QLoRA 7B: 1–4 A100-days; QLoRA 32B: 8–32 A100-days. GRPO: 4–16 A100-days. | 1K–100K domain QA examples. SynRL (CVPR 2026): 7.7K synthetic CoT samples outperform 165K real samples (21× data efficiency). |
| **Synthetic forensic ensemble** | Freeze pretrained detectors (use 3+ APIs or open-source). Train a logistic meta-classifier / corroborator on top of their outputs + provenance. | <1 GPU day for meta-classifier. | 10K+ authentic/synthetic compressed samples. RA-Bench dataset is starting point. |

### ❌ SPECULATIVE / NOT RECOMMENDED FOR FIRST SYSTEM:
- End-to-end training: perception → state → reasoning → claims. Destroys per-component interpretability and calibration; compute cost 100–1000× higher; no evidence it beats pretrained composition.
- RL / preference training for hypothesis ranking: Overkill until heuristic + LTR baseline is exhausted.
- "Self-supervised learning everything": Waste of cycles; all perception components are already self-supervised or trained on massive data.
- Continual learning at the perception level: Use LoRA fine-tunes on new domains as separate model swappable weights; simpler.

### CENTRAL QUESTION ANSWERED (2026 reality):
**"Which capabilities need joint optimization and which benefit from modularity?"**
- **Benefit from modular composition (DO NOT jointly train):**
  - Perception ↔ state ↔ reasoning. Each is independently calibrated. Composition preserves error bars.
  - Synthetic forensic detectors: Each has blind spots; ensembling independently-trained models gives disproportionate gains (abductive corroboration 2026 result).
- **May benefit from light joint fine-tuning (LoRA or small connector only):**
  - Evidence graph features → hypothesis ranking head.
  - State embeddings → motion prediction residual.
- **Full joint training is justified only if:**
  - The end-to-end task is a single, very well-defined metric (e.g., event recall in soccer), AND
  - The modular baseline has been exhausted and you have 100+ GPU-weeks budget.
  - Otherwise you will just break calibration for marginal top-1 gains.

### COMPUTE ESTIMATES REFERENCE (2026):
| Task | RTX 4090 hours | A100 (80 GB) hours |
|---|---|---|
| LoRA fine-tune 7B VLM on 100K samples | 200–400 h | 50–100 h |
| LoRA fine-tune RF-DETR on custom dataset | 50–150 h | 10–30 h |
| Train GBDT hypothesis ranker | <1 h | <0.5 h |
| Post-hoc calibration (SCP/isotonic) | <0.1 h | <0.1 h |
| TOTAL SYSTEM (from scratch to working prototype, compute only) | **~500–1,000 h single 4090** | **~100–200 h single A100** |

Compute for PERPETUAL RUNNING (inference) of the final system:
- Single RTX 4090 24GB: 30 FPS perception + 0.3–0.5 Hz Qwen 2.5-VL 32B AWQ. Fits in VRAM.
- Single RTX 4090D: 30 FPS + **90 tok/s** with Qwen3-VL-30B-A3B FP8 (best consumer GPU VLM).
- Single A100 80GB: 30 FPS + 0.5–1 Hz 32B + forensic ensemble + **312 tok/s** with Qwen2.5-VL-32B AWQ (vLLM 0.7.3).
- Multi-GPU (2× 4090): perception GPU + reasoning GPU = 1 Hz VLM is comfortable.

### DATA REQUIREMENTS SUMMARY (2026 buildable):
- **Perception fine-tune (if needed):** 5–50K frames.
- **Hidden state evaluation + calibration:** 1K–10K clips with occlusion.
- **Hypothesis ranking / calibration:** 10K–100K structured claim labels.
- **Forensic meta-classifier:** 10K+ compressed authentic/synthetic.
- **TOTAL: ~50–200K labeled samples.** This is doable for a small team with semi-automated labeling (VLM adjudicator + human spot-check).

### DELIVERABLE (2026 practical):
- Module-by-module training plan with compute + data numbers as above.
- Priority order: Pretrained composition first → LoRA fine-tunes → GBDT ranker → calibration. Abandon E2E training until modular baseline is exhausted.
- Ablation plan linked to training choices (modular composite vs. LoRA fine-tune vs. hypothetical E2E) with measured cost vs. quality curves.
