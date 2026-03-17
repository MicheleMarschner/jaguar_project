This experiment presents a systematic grid study of optimization strategies for jaguar re-identification. While the model architecture (EVA-02) and the loss function (Triplet + Focal/CE) provide the foundation, the optimizer and learning rate scheduler determine the model’s trajectory through the loss landscape. Given the fine-grained nature of rosette patterns and the sparsity of our curated dataset, selecting an optimization recipe that avoids sharp local minima and ensures stable identity clustering is critical. The **Research Question** we aim to answer is: *Which combination of optimizer (AdamW vs. Muon) and learning rate scheduler (JaguarId, OneCycle, Cosine, or ReduceOnPlateau) yields the highest identity-balanced retrieval accuracy and the most robust embedding manifold for jaguar re-identification?*

The pipeline and optimization setup follow the configuration defined in `config/base/kaggle_base.toml` and described in `E01_kaggle-backbone.md`, using the EVA-02 backbone and Triplet Loss with semi-hard mining. In this grid search, we ablate only the optimizer and scheduler configurations. Each combination is evaluated across multiple seeds (see `E10_kaggle_stat_stability.md`) to assess variance and robustness. In the following, we refer to Pairwise AP as PW-AP.

Optimization Suite
- AdamW (Decoupled Weight Decay): Standard adaptive optimizer with weight decay $1e^{-3}$ and betas $[0.9, 0.999]$. Known for stable Transformer fine-tuning.
- Muon (Orthogonalization Optimizer): Designed for Transformer hidden layers, performing Newton-based orthogonalization updates. It can accelerate convergence and improve generalization in high-dimensional embedding spaces.
- Learning Rate: Base LR of $1e^{-5}$ for AdamW; split LR for Muon ($LR=3e^{-4}$, $Muon\_LR=2e^{-3}$).

Schedulers
- JaguarIdScheduler: Adapted from MiewID. Uses a 5-epoch linear warm-up to $lr_{max}$, followed by a sustain phase and exponential decay ($\gamma=0.8$). Designed to stabilize early backbone adaptation.
- OneCycleLR: Follows the super-convergence principle, increasing LR to a peak and then annealing to near-zero.
- CosineAnnealingLR: Smooth decay that helps avoid premature convergence to sharp minima.
- ReduceLROnPlateau: Reduces LR by a factor of 0.1 if validation mAP plateaus for 3 epochs.

The following tables summarize performance averaged across 5 seeds $[42, 123, 256, 512, 1024]$, capturing both central tendency and variability.

Table 1 — AdamW: Scheduler Comparison
| Scheduler           | Mean mAP | Std Dev mAP | PW-AP (Best) | Silhouette (Best) | Rank-1 (Best) | Loss (Best) |
|---------------------|----------|-------------|--------------|-------------------|---------------|-------------|
| JaguarId            | 0.5843   | 0.0392      | 0.8831       | 0.5816            | 0.9455        | 1.0985      |
| ReduceOnPlateau     | 0.5715   | 0.0213      | 0.8251       | 0.5885            | 0.9353        | 1.0805      |
| OneCycle            | 0.5303   | 0.0130      | 0.6928       | 0.3036            | 0.9115        | 1.2434      |
| Cosine              | 0.5408   | 0.0051      | 0.6983       | 0.3047            | 0.9081        | 1.2283      |

Table 2 — Muon: Scheduler Comparison
| Scheduler | Mean mAP | Std Dev mAP | PW-AP (Best) | Silhouette (Best) | Rank-1 (Best) | Loss (Best) |
|----------|----------|-------------|--------------|-------------------|---------------|-------------|
| OneCycle | 0.6432   | 0.0172      | 0.9136       | 0.6219            | 0.9523        | 0.8565      |
| Cosine   | 0.6519   | 0.0079      | 0.9326       | 0.6228            | 0.9523        | 0.7675      |
| JaguarId | 0.6374   | 0.0055      | 0.9043       | 0.5736            | 0.9421        | 0.8628      |

Table 3 — Adam: Scheduler Comparison
| Scheduler | Mean mAP | Std Dev mAP | PW-AP (Best) | Silhouette (Best) | Rank-1 (Best) | Loss (Best) |
|----------|----------|-------------|--------------|-------------------|---------------|-------------|
| JaguarId | 0.6432   | 0.0179      | 0.9456       | 0.2717            | 0.9456        | 1.6317      |
| Cosine   | 0.4999   | 0.0145      | 0.6378       | 0.2744            | 0.8979        | 1.3910      |
| OneCycle | 0.5047   | 0.0153      | 0.6358       | 0.2305            | 0.8877        | 1.6323      |

Final Leaderboard
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

By examining validation trajectories over 30 epochs, we observe three distinct behaviors:

- Muon-based recipes (Cosine, OneCycle) converge very rapidly, reaching >0.60 mAP within the first 10 epochs. The orthogonalization updates allow efficient navigation of the high-dimensional embedding space. However, their very low loss (~0.76) and high Silhouette (~0.62) suggest a tendency toward overly rigid clustering, which may limit generalization.
- Adam + JaguarId produces a slower but more consistent trajectory. The warm-up phase prevents early collapse, and the gradual decay maintains learning capacity throughout training. Unlike OneCycle, which often peaks early and slightly degrades, this configuration continues improving in later epochs.
- ReduceLROnPlateau performs poorly due to noisy validation signals. In this low-data regime, small fluctuations trigger premature LR reductions, often pushing the model into suboptimal minima.

A detailed comparison (Seed 42) highlights the trade-off:

| Optimizer + Scheduler | Epoch 10 mAP | Epoch 30 mAP | Δ (Growth) | Final val/loss |
|----------------------|-------------|--------------|------------|----------------|
| Muon + Cosine        | 0.6229      | 0.6519       | +0.0290    | 0.7675         |
| Adam + JaguarId      | 0.5368      | 0.6555       | +0.1187    | 1.6317         |

Although Muon achieves lower loss and higher Silhouette scores, Adam + JaguarId exhibits greater late-stage improvement and a more flexible embedding structure. This flexibility appears beneficial for generalization: despite less “ideal” validation metrics, this configuration contributed to the best Kaggle submission (~90.08% mAP).

In conclusion, optimization dynamics play a critical role in fine-grained Re-ID. Fast convergence and tight clustering (Muon) do not necessarily translate to better generalization. Instead, a controlled optimization trajectory (Adam + JaguarId) that preserves embedding flexibility can yield more robust performance on unseen data.