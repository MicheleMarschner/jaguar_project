# E10 Statistical Stability (Data - Round 2)

**Experiment Group:** Robustness and diagnostic experiments

## Main Research Question

How robust is the jaguar re-identification pipeline to stochastic variation, and to what extent are the observed performance gains dependent on favorable random initializations?

---

## Setup / Intervention

This experiment evaluates the stability of the current best-performing training recipe under different random seeds. Following the backbone and loss-selection experiments, we fix the full model configuration and vary only the pseudo-random initialization.

The evaluated setup uses:

- **EVA-02** as backbone
- **Triplet-based training with hard mining**
- the same optimizer, scheduler, augmentation suite, and evaluation protocol used in the main leaderboard pipeline

For implementation details and the full configuration, see **`E00_leaderboard-kaggle_backbone.md`** and the corresponding base config.

We evaluate the following seeds:

- **42**
- **123**
- **256**
- **512**
- **1024**

This selection mixes commonly used benchmark seeds with larger powers of two, in order to test whether the results remain consistent across different pseudo-random generator states.

---

## Method / Procedure

All hyperparameters are kept fixed across runs. Only the random seed is changed.

The goal is not only to compare final validation scores, but also to inspect whether convergence remains stable across seeds. We therefore analyze:

- validation **identity mAP**
- validation **Pairwise AP**
- validation **Rank-1**
- validation **Similarity Gap**
- validation **Silhouette Score**
- training and validation trajectories over epochs

In addition to aggregate mean and standard deviation, we report the **coefficient of variation (CV)** to quantify relative instability across metrics.

---

## Main Results

**Table 1. Stability summary across five seeds on the Round 2 dataset.**

| Metric | Mean (μ) | Std Dev (σ) | CV (%) |
|---|---:|---:|---:|
| Identity mAP | 0.6509 | 0.0214 | 3.29 |
| Pairwise AP | 0.9005 | 0.0284 | 3.15 |
| Rank-1 Accuracy | 0.9367 | 0.0085 | 0.91 |
| Similarity Gap | 0.7863 | 0.0558 | 7.09 |
| Silhouette Score | 0.6479 | 0.0676 | 10.40 |

The main retrieval metrics are overall quite stable. **Rank-1** shows the lowest relative variability, while **identity mAP** and **Pairwise AP** remain within a moderate range across seeds. By contrast, **Similarity Gap** and especially **Silhouette Score** vary more strongly, indicating that the internal geometry of the embedding space is more sensitive to initialization than the final top-ranked retrieval itself.

---

## Training-Curve Analysis

<p align="center"><img src="../../results/round_2/kaggle_stat_stability/Bildschirmfoto 2026-03-17 um 23.11.30.png" width="80%" /></p>
<p align="center"><em>Figure 1. Validation trajectories across seeds for the fixed EVA-02 pipeline. The runs follow a highly similar overall pattern, with one visibly weaker seed showing slower convergence and lower final metrics.</em></p>

<p align="center"><img src="../../results/round_2/kaggle_stat_stability/W&B Chart 17.3.2026, 23_11_12.png" width="70%" /></p>
<p align="center"><em>Figure 2. Training-loss curves across seeds. Most runs converge closely, while one seed shows slower late-stage refinement and a visibly higher final loss.</em></p>

The validation curves show that the overall training dynamics are largely reproducible. Most seeds follow a very similar trajectory and converge to closely related solutions. The main deviation comes from **seed 256**, which converges more slowly and reaches visibly weaker final values on several validation metrics.

This weaker run is also reflected in the summary statistics: the overall mean remains strong, but the standard deviation in **mAP**, **Similarity Gap**, and **Silhouette** is partly driven by this single underperforming seed.

The most important observation is therefore not that the pipeline is unstable in general, but that it is **mostly stable with one notable outlier**.

---

## Interpretation

### Why does one seed deviate more strongly?

A plausible explanation is the interaction between **hard triplet mining**, **class imbalance**, and the comparatively small **batch size of 32**.

The training set contains **31 jaguar identities**, but the dataset is imbalanced and many images come from burst-heavy capture sessions. As a result, batches often do not contain a balanced representation of identities. For rare jaguars in particular, some mini-batches may contain only a small number of useful positive examples. Under hard mining, this can make the sampled triplets especially sensitive to the random sequence of batches.

This likely explains the temporary irregularities and the slower convergence of seed 256. The issue is not random initialization alone, but the fact that initialization interacts with **which hard positives and negatives are seen early in training**.

### What do the stability statistics imply?

The stability profile differs by metric.

- **Rank-1** is very stable (**CV = 0.91%**), indicating that the model almost always retrieves the correct identity at the top of the ranking.
- **Identity mAP** and **Pairwise AP** are slightly more variable, but still remain in a relatively narrow band (**CV ≈ 3%**).
- **Similarity Gap** and **Silhouette** are noticeably more variable (**CV = 7.09%** and **10.40%**), suggesting that embedding-space structure is more sensitive to stochastic effects than top-level retrieval accuracy.

This distinction is important. The system can be reliably good at retrieving the correct jaguar near the top, even when the underlying cluster geometry differs somewhat across runs.

### Are the reported gains likely to be real or just lucky?

The results argue against the idea that the main gains come only from a lucky seed.

Although one seed underperforms, the majority of runs converge to very similar validation scores. In particular, the low variance of **Rank-1** and the moderate variance of **mAP** indicate that the overall pipeline is reasonably robust. The improvements observed in the broader study therefore appear to reflect the training recipe itself rather than a single favorable initialization.

At the same time, the experiment also shows that fine-grained jaguar Re-ID remains somewhat sensitive to stochasticity, especially when looking beyond top-1 accuracy to the detailed structure of the learned embedding space.

---

## Additional Stability Checks

We also explored two related stabilization strategies during development:

- **larger batch sizes** (e.g. 128), which expose the model to many more identities per update and make triplet formation more stable
- **Exponential Moving Average (EMA)** of model weights

Both interventions improved training smoothness in internal tests, but neither translated into better leaderboard performance in the final Kaggle setting. For that reason, they were not adopted in the final default pipeline.

This is consistent with the broader pattern of the experiment: smoothing training dynamics does not automatically improve final identity-level retrieval.

---

## Main Findings

- The best EVA-02 jaguar Re-ID pipeline is **reasonably stable across random seeds**.
- **Rank-1** is the most robust metric, with very low variability.
- **Identity mAP** and **Pairwise AP** show moderate but still limited variation.
- **Similarity Gap** and **Silhouette** vary more strongly, indicating sensitivity in embedding-space structure.
- One seed (**256**) behaves as a clear weaker run, but the other runs cluster closely together.
- The observed gains are therefore **not plausibly explained only by lucky initialization**, although stochastic effects remain visible.

---

## Limitation

This stability analysis varies only the **random seed** under a fixed training recipe. It does not isolate which stochastic component is most responsible for the observed variance, such as initialization, batch ordering, or hard-triplet composition. In addition, the interpretation of the outlier run is necessarily indirect, since the exact batch-level triplets are not analyzed here.

---

## Conclusion

The statistical stability experiment shows that the current jaguar re-identification pipeline is **not perfectly deterministic**, but it is **robust enough that the main conclusions do not depend on a single favorable run**.

The strongest practical result is that **top-level retrieval performance remains stable across seeds**, especially for **Rank-1**, while embedding-space diagnostics are more sensitive to stochastic variation. The pipeline can therefore be considered reliable for model comparison and selection, but variance should still be acknowledged when interpreting smaller mAP differences or more geometric embedding metrics.

Overall, this experiment supports the broader claim that the reported performance gains are primarily driven by the pipeline design rather than by lucky random initialization.
