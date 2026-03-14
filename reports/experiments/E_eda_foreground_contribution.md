# E0X (Q0) Foreground vs Background Contribution Analysis

**Experiment Group:** Robustness and diagnostic experiments

## Main Research Question
Is identity information preserved more in the jaguar region or in the background?

*Does Jaguar Re-ID retrieval depend primarily on jaguar appearance or does background context retain substantial identity information?*
*How much identity information is retained in jaguar-only versus background-only query views, and does retrieval rely more on foreground identity cues or on background context?*

Secondary RQs
- Classification sensitivity: does true-class confidence drop more when jaguar is removed or when background is removed?
- Are background effects stronger for cases the model already gets wrong? / Is background dependence mostly a problem in already difficult cases?
- Embeddings stability: Does the embedding remain closer to the original image when only the jaguar is retained or when only the background is retained?




From the code, you keep the gallery fixed and compare three query variants:
- original
- jaguar-only
- background-only

Then you measure:
- retrieval behavior (rank1, rank5, gold rank, margin)
- embedding stability (cosine similarity to original embedding)
- classification sensitivity (logit/log-prob changes)
- qualitative CAMs for suspicious background-dominant cases


So the core intervention is:
- Remove either foreground or background and see what remains predictive.

- jaguar-only / background-only query variants
- retrieval comparison
- embedding stability
- logit sensitivity
- qualitative heatmaps for suspicious cases



### Intervention
We decomposed each query into three variants: `orig`, `jaguar_only`, and `bg_only`. All variants were evaluated against the same fixed gallery. We compared retrieval outcomes, embedding stability, and classification sensitivity across the three views. In addition, we repeated the analysis separately for queries that were originally correct and originally wrong.

### Main Findings
- `jaguar_only` achieved [x] Rank-1, while `bg_only` achieved [y], indicating [foreground-dominant / mixed / background-sensitive] behavior.
- The jaguar-minus-background gap was [positive / near zero / negative] across [settings].
- Background outperformed jaguar in [z%] of cases overall and in [w%] of originally wrong cases, showing that background dependence is [rare / concentrated in difficult cases / substantial].

### Main Results Table
| condition | orig Rank-1 | jaguar_only Rank-1 | bg_only Rank-1 | jaguar-bg gap | share bg better than jaguar |
|---|---:|---:|---:|---:|---:|
| [setting 1] | [ ] | [ ] | [ ] | [ ] | [ ] |
| [setting 2] | [ ] | [ ] | [ ] | [ ] | [ ] |

### Error Split Table
| group | jaguar_only Rank-1 | bg_only Rank-1 | jaguar-bg gap | share bg better than jaguar |
|---|---:|---:|---:|---:|
| all | [ ] | [ ] | [ ] | [ ] |
| orig_rank1_correct | [ ] | [ ] | [ ] | [ ] |
| orig_rank1_wrong | [ ] | [ ] | [ ] | [ ] |

### Supporting Evidence
- **Classification sensitivity:** [removing jaguar / removing background] caused the larger drop in true-class confidence.
- **Embedding stability:** the original embedding was closer to **[jaguar_only / bg_only]**.

### Plots
- Absolute performance for `orig`, `jaguar_only`, `bg_only`
- Jaguar-minus-background gap
- Share of cases where `bg_only` beats `jaguar_only`


### Interpretation
High `jaguar_only` performance together with low `bg_only` performance indicates that identity is mainly encoded in the animal region. Conversely, strong `bg_only` performance or frequent background-dominant cases suggests that contextual cues contribute materially to retrieval.

### Limitation
The analysis relies on masked query variants, so differences may partly reflect artifacts introduced by masking in addition to true foreground/background dependence.