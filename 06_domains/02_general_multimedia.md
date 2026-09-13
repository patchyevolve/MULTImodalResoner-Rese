# General multimedia reasoning

Target heterogeneous inputs:

- consumer videos
- news clips
- documentaries
- user-generated content
- photographs
- screenshots
- audio-video
- text-overlaid media

Research challenge: maintain the same epistemic architecture even when domain-specific priors are weak or unavailable.

---

## REALITY CHECK 2026: CROSS-DOMAIN STATE OF THE ART

### Video-MME-v2 (Apr 2026, arXiv:2604.05015):
- 800 videos, 3,200 QA pairs, 3,300 human annotation hours.
- Group-based non-linear scoring: Capability Consistency + Reasoning Coherence.

| Model | Non-Lin Score | Avg Acc | Non-Lin/Acc Ratio |
|---|---|---|---|
| Human Expert | **90.7** | 94.9 | 95.6% |
| Gemini-3-Pro | 49.4 | 66.1 | 74.7% |
| Doubao-Seed-2.0-Pro | 43.3 | 60.5 | 71.6% |
| GPT-5 | 37.0 | 55.6 | 66.5% |

- **41.3-point gap** between best model and human experts.
- Level 1 (Retrieval): Gemini-3-Pro 50.0 vs Human 91.1.
- Level 2 (Temporal): Gemini-3-Pro 45.4 vs Human 87.9.
- Level 3 (Complex Reasoning): Gemini-3-Pro 40.6 vs Human 88.9.
- Thinking mode: +3.8 to +5.8 with subtitles, but can cause regression without text cues.
- All models score below 30 on Action & Motion and Physical World Reasoning.

### OmniVChall (ICML 2026, arXiv:2602.00559):
- 823 videos, 9,027 QA pairs. 8 hallucination types including camera dynamics.
- Performance drops 5.71% to 9.32% from single-factor to compositional hallucination.
- Camera-based reasoning: ~38.57% accuracy drop (worst category).
- AI-generated videos: 9.31% larger accuracy drop than real-world under compositional queries.

### CrossVid Benchmark (AAAI 2026):
- First cross-video reasoning: 5,331 videos, 9,015 QA pairs.
- Gemini-2.5-Pro best at 50.4% average. Most MLLMs struggle with multi-video evidence integration.

### EgoCross Challenge (CVPR 2026):
- OmniEgo-R2: 66.35% (Source-Limited), 66.77% (Open-Source).
- Cross-domain: surgery, industry, extreme sports, animal perspective.
- Same abstract capabilities require different visual grammar per domain.

### RD-MLDG (Domain Generalization):
- Reasoning chains exhibit 58.6% lower cross-domain divergence than visual features.
- MMD: 0.239 (visual) → 0.099 (reasoning chains).
- **Key insight:** Reasoning provides more domain-invariant representations than raw visual features.

### Practical Implications:
1. **Reasoning chains > visual features** for domain invariance. Maintain explicit reasoning chains.
2. **Subtitle/text modality is critical.** Thinking mode regresses without text cues.
3. **Camera motion is hardest category.** Models confuse lens motion with object motion.
4. **Compositional queries degrade 5-9%** beyond single-factor queries.
5. **No single model dominates all domains.** Design for graceful degradation.
