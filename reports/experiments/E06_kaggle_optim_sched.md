# Optimizer and Scheduler Grid Study for Jaguar Re-ID

## Main Research Question

Which combination of optimizer (**AdamW**, **Muon**, or **Adam**) and learning-rate scheduler (**JaguarId**, **OneCycle**, **Cosine**, or **ReduceLROnPlateau**) yields the highest identity-balanced retrieval performance and the most robust embedding structure for jaguar re-identification?

This experiment presents a systematic grid study of optimization strategies for jaguar re-identification. While the backbone (**EVA-02**) and loss formulation (**Triplet + Focal/CE**) define the model family, the optimizer and scheduler determine how the model moves through the loss landscape. In a fine-grained Re-ID setting with limited curated data, the optimization recipe strongly affects both convergence behavior and the geometry of the resulting embedding space.

## Setup / Intervention

The pipeline follows the configuration defined in `config/base/kaggle_base.toml` and the backbone setup described in `E01_kaggle-backbone.md`, using the **EVA-02** backbone and **Triplet Loss with semi-hard mining**. In this grid search, only the **optimizer** and **scheduler** are varied; all other hyperparameters, augmentation settings, and evaluation procedures remain fixed.

Each configuration is evaluated across multiple random seeds (see `E10_kaggle_stat_stability.md`) to assess both central tendency and variance. In the following, **PW-AP** denotes **Pairwise Average Precision**.

### Optimizers

- **AdamW**: decoupled weight decay with weight decay `1e-3` and betas `[0.9, 0.999]`; a standard and stable choice for Transformer fine-tuning.
- **Muon**: an orthogonalization-based optimizer designed for Transformer hidden layers, intended to accelerate convergence and improve optimization in high-dimensional spaces.
- **Adam**: standard adaptive optimizer used here as an additional comparison point.

The base learning rate is `1e-5` for AdamW. For Muon, a split learning-rate setup is used (`LR = 3e-4`, `Muon_LR = 2e-3`).

### Schedulers

- **JaguarIdScheduler**: adapted from MiewID; uses a 5-epoch linear warm-up to `lr_max`, followed by a sustain phase and exponential decay (`γ = 0.8`).
- **OneCycleLR**: increases the learning rate to a peak and then anneals toward near-zero.
- **CosineAnnealingLR**: uses smooth cosine decay.
- **ReduceLROnPlateau**: reduces the learning rate by a factor of `0.1` if validation mAP plateaus for 3 epochs.

## Method / Procedure

The comparison is based on retrieval performance, embedding diagnostics, and convergence behavior. For each optimizer–scheduler combination, we summarize:

- **mean validation mAP across 5 seeds** (`42, 123, 256, 512, 1024`),
- **standard deviation of validation mAP** across those seeds,
- the **best** observed values for **PW-AP**, **silhouette**, **Rank-1**, and **validation loss**.

This yields two complementary perspectives:

1. **average performance across seeds**, which reflects robustness, and
2. **best single-run performance**, which reflects the strongest result attained by a configuration.

These two views do not necessarily rank the methods identically.

<p align="center"><img src="../../results/round_2/kaggle_optim_sched/Bildschirmfoto 2026-03-17 um 23.49.32.png" width="95%" /></p>
<p align="center"><em>Figure 1. Validation trajectories for the optimizer–scheduler grid, including similarity gap, silhouette, Rank-1, PW-AP, mAP, and validation loss.</em></p>

<p align="center"><img src="../../results/round_2/kaggle_optim_sched/W&B Chart 17.3.2026, 23_49_05.png" width="80%" /></p>
<p align="center"><em>Figure 2. Training-loss trajectories for the optimizer–scheduler grid.</em></p>

## Main Results

### Table 1 — AdamW: Scheduler Comparison

| Scheduler           | Mean mAP | Std Dev mAP | PW-AP (Best) | Silhouette (Best) | Rank-1 (Best) | Loss (Best) |
|---------------------|----------|-------------|--------------|-------------------|---------------|-------------|
| JaguarId            | 0.5843   | 0.0392      | 0.8831       | 0.5816            | 0.9455        | 1.0985      |
| ReduceOnPlateau     | 0.5715   | 0.0213      | 0.8251       | 0.5885            | 0.9353        | 1.0805      |
| OneCycle            | 0.5303   | 0.0130      | 0.6928       | 0.3036            | 0.9115        | 1.2434      |
| Cosine              | 0.5408   | 0.0051      | 0.6983       | 0.3047            | 0.9081        | 1.2283      |

### Table 2 — Muon: Scheduler Comparison

| Scheduler | Mean mAP | Std Dev mAP | PW-AP (Best) | Silhouette (Best) | Rank-1 (Best) | Loss (Best) |
|----------|----------|-------------|--------------|-------------------|---------------|-------------|
| OneCycle | 0.6432   | 0.0172      | 0.9136       | 0.6219            | 0.9523        | 0.8565      |
| Cosine   | 0.6519   | 0.0079      | 0.9326       | 0.6228            | 0.9523        | 0.7675      |
| JaguarId | 0.6374   | 0.0055      | 0.9043       | 0.5736            | 0.9421        | 0.8628      |

### Table 3 — Adam: Scheduler Comparison

| Scheduler | Mean mAP | Std Dev mAP | PW-AP (Best) | Silhouette (Best) | Rank-1 (Best) | Loss (Best) |
|----------|----------|-------------|--------------|-------------------|---------------|-------------|
| JaguarId | 0.6432   | 0.0179      | 0.9456       | 0.2717            | 0.9456        | 1.6317      |
| Cosine   | 0.4999   | 0.0145      | 0.6378       | 0.2744            | 0.8979        | 1.3910      |
| OneCycle | 0.5047   | 0.0153      | 0.6358       | 0.2305            | 0.8877        | 1.6323      |

### Final Leaderboard

| Rank | Optimizer | Scheduler       | Best mAP | PW-AP  | Silhouette | Rank-1 | Loss   |
|------|----------|-----------------|----------|--------|------------|--------|--------|
| 1    | Muon     | OneCycle        | 0.6658   | 0.9136 | 0.6219     | 0.9523 | 0.8565 |
| 2    | Muon     | Cosine          | 0.6651   | 0.9326 | 0.6228     | 0.9523 | 0.7675 |
| 3    | AdamW    | JaguarId        | 0.6595   | 0.8831 | 0.5816     | 0.9455 | 1.0985 |
| 4    | Adam     | JaguarId        | 0.6574   | 0.9456 | 0.2717     | 0.9456 | 1.6317 |
| 5    | Muon     | JaguarId        | 0.6516   | 0.9043 | 0.5736     | 0.9421 | 0.8628 |
| 6    | AdamW    | ReduceOnPlateau | 0.5986   | 0.8251 | 0.5885     | 0.9353 | 1.0805 |
| 7    | AdamW    | OneCycle        | 0.5465   | 0.6928 | 0.3036     | 0.9115 | 1.2434 |
| 8    | AdamW    | Cosine          | 0.5411   | 0.6983 | 0.3047     | 0.9081 | 1.2283 |
| 9    | Adam     | Cosine          | 0.5144   | 0.6378 | 0.2744     | 0.8979 | 1.3910 |
| 10   | Adam     | OneCycle        | 0.5137   | 0.6358 | 0.2305     | 0.8877 | 1.6323 |

## Analysis

### Average performance across seeds

The clearest pattern in the seed-averaged results is that **Muon** dominates the grid overall. Among all averaged configurations, **Muon + Cosine** yields the highest mean mAP (**0.6519**) with low variance (**± 0.0079**), followed by **Muon + OneCycle** (**0.6432 ± 0.0172**) and **Muon + JaguarId** (**0.6374 ± 0.0055**).

Within the **AdamW** family, **JaguarId** performs best on average (**0.5843 ± 0.0392**), while **OneCycle** and **Cosine** remain clearly lower. For plain **Adam**, only **JaguarId** is competitive on mean mAP (**0.6432 ± 0.0179**); its Cosine and OneCycle variants lag substantially behind.

Thus, when robustness across seeds is prioritized, the strongest region of the grid is the **Muon** family, especially with **Cosine** or **OneCycle** scheduling.

### Best single-run performance

The ranking changes slightly when the comparison is based on the strongest single run rather than the seed average. In the final leaderboard, **Muon + OneCycle** achieves the best observed validation mAP (**0.6658**), narrowly ahead of **Muon + Cosine** (**0.6651**). The strongest non-Muon configuration is **AdamW + JaguarId** (**0.6595**), followed by **Adam + JaguarId** (**0.6574**).

This difference between the averaged and best-run views is important. The strongest single run is not necessarily the most robust configuration, and vice versa. In this study, **Muon + Cosine** is the strongest seed-averaged recipe, whereas **Muon + OneCycle** attains the highest individual run.

### Convergence behavior

The validation trajectories in **Figure 1** and training-loss curves in **Figure 2** suggest three broad optimization regimes.

First, the **Muon-based recipes** converge very quickly. In the user-provided analysis, they reach roughly **0.60+ mAP** already within the first 10 epochs. Their low final validation loss and comparatively high silhouette values indicate very strong and early structuring of the embedding space.

Second, **Adam + JaguarId** follows a slower but more gradual trajectory. Rather than peaking very early, it continues to improve later in training. The comparison reported for **seed 42** illustrates this contrast:

| Optimizer + Scheduler | Epoch 10 mAP | Epoch 30 mAP | Δ (Growth) | Final val/loss |
|----------------------|-------------|--------------|------------|----------------|
| Muon + Cosine        | 0.6229      | 0.6519       | +0.0290    | 0.7675         |
| Adam + JaguarId      | 0.5368      | 0.6555       | +0.1187    | 1.6317         |

This makes the trade-off explicit: **Muon + Cosine** starts much stronger and stays strong, whereas **Adam + JaguarId** improves more slowly but keeps gaining later.

Third, **ReduceLROnPlateau** appears less effective in this low-data setting. As stated in the original experiment description, the concern is that noisy validation signals can trigger premature learning-rate reductions and lead to suboptimal solutions.

### Embedding-space diagnostics

The auxiliary diagnostics point in the same general direction as the retrieval metrics. The best **Muon** configurations also achieve the strongest silhouette values (around **0.62**) and low validation loss, while **AdamW + JaguarId** occupies an intermediate regime and the weaker Adam/AdamW combinations score much lower on silhouette.

At the same time, the experiment description explicitly cautions against equating tighter clustering with better downstream generalization. In particular, the draft interpretation argues that the very low loss and high silhouette values of Muon may correspond to a more rigid embedding structure, whereas **Adam + JaguarId** preserves more flexibility even though its validation loss is substantially higher.

## Interpretation

### Which optimizer–scheduler combinations are strongest on internal validation?

On internal validation, the strongest combinations are clearly **Muon + OneCycle** and **Muon + Cosine**. They dominate both the final leaderboard and the multi-seed averages, and they also achieve the strongest silhouette and low-loss profiles.

The best non-Muon alternatives are **AdamW + JaguarId** and **Adam + JaguarId**. Among AdamW schedules, JaguarId is the only one that remains clearly competitive; among Adam schedules, JaguarId is again the only strong option.

### Do faster convergence and tighter clusters necessarily imply better generalization?

Not necessarily. This is the central tension in the experiment.

The Muon-based configurations produce the best internal validation metrics and the most compact embedding diagnostics. However, the original experiment text argues that **Adam + JaguarId**, despite having a much higher validation loss and much lower silhouette, contributed to the best Kaggle submission (about **90.08% mAP**). The intended interpretation is therefore not that Muon is weak, but that **internal compactness and fast convergence do not automatically translate into the best external generalization**.

### What is the practical optimization recommendation?

If the goal is to maximize **internal validation performance**, the most compelling recipes in this grid are the **Muon** variants, especially **Muon + Cosine** and **Muon + OneCycle**.

If the goal is to prioritize the configuration that ultimately supported the strongest **Kaggle submission**, the experiment text points to **Adam + JaguarId** as the practically favored recipe, precisely because its slower and less rigid optimization trajectory may generalize better to unseen leaderboard data.

## Key Result / Takeaway

The optimizer and scheduler have a large effect on both convergence and retrieval quality.

Internally, the best-performing region of the grid is the **Muon** family. **Muon + Cosine** gives the highest mean mAP across seeds, while **Muon + OneCycle** achieves the best individual run. These recipes also produce the strongest silhouette scores and lowest validation losses.

At the same time, the experiment highlights an important generalization caveat: the configuration that appears strongest on internal validation is not necessarily the one that transfers best to the external Kaggle test distribution. In the original interpretation, **Adam + JaguarId** remains important because its slower, more flexible optimization path was associated with the best leaderboard submission.

## Limitation

This study isolates the optimizer and scheduler within one fixed model family and one fixed loss setup. The conclusions therefore apply to the **EVA-02 + Triplet/Focal/CE** pipeline studied here and should not be automatically generalized to other backbones or objective functions. In addition, the interpretation of external generalization relies on comparison with Kaggle outcomes rather than a controlled held-out test distribution within the same experiment.

## Conclusion

The grid study shows that optimization dynamics are a major design choice in fine-grained jaguar re-identification. **Muon** achieves the strongest internal validation performance, both in terms of mean mAP and best single-run mAP, especially when paired with **Cosine** or **OneCycle** scheduling. However, the study also suggests that **faster convergence and tighter clustering are not the whole story**: a slower and more flexible recipe such as **Adam + JaguarId** may still generalize better to the hidden test distribution.

Overall, the experiment supports a nuanced conclusion. For internal validation, **Muon-based optimization is strongest**. For end-to-end competition performance, a more conservative schedule such as **Adam + JaguarId** may remain preferable.
