# Triplet Mining Strategy Ablation (Data - Round 2)

**Experiment Group:** Kaggle leaderboard experiments / optimization studies

## Main Research Question

How does the choice of triplet mining strategy — from purely stochastic to difficulty-based selection — affect convergence and final retrieval performance for a species-specific Jaguar Re-ID model?

## Setup / Intervention

This experiment isolates one core component of the metric-learning pipeline: the **Triplet Loss mining strategy**. The backbone, augmentation suite, optimization schedule, and general training setup follow the best-performing EVA-02 configuration used in the second round of the Kaggle Jaguar Re-ID challenge. The goal is therefore not to compare full model families, but to measure how the way positive and negative pairs are selected within a batch changes the learned embedding space and downstream retrieval quality.

All runs use the **EVA-02** backbone and the same `JaguarIDModel` implementation. The model is trained with the **Bag of Tricks (BoT)**-style triplet head, consisting of:

- global feature extraction from the backbone,
- a **BatchNorm neck** before loss computation,
- **Triplet Loss** on the embedding,
- and an auxiliary classification loss (**Cross Entropy** or **Focal Loss**) on the logits.

This follows the general design used in person Re-ID and OpenAnimals-style baselines, while restricting the representation to global features only.

## Mining Strategies Compared

As implemented in `src/utils/utils_losses.py`, four mining strategies are compared:

- **Hard mining (`hard`)**: uses the hardest positive and hardest negative in the batch.
- **Weighted mining (`weighted`)**: uses a softmax-weighted average over positives and negatives, so harder pairs contribute more strongly but not exclusively.
- **Random mining (`random`)**: samples one positive and one negative pair at random.
- **Semi-hard mining (`semi-hard`)**: selects negatives that are farther from the anchor than the positive but still within the margin; if none exist, it falls back to the hardest negative.

## Method / Procedure

To make the comparison stable and competitive, the configuration differs slightly from the default leaderboard setup.

- **Batch size** was increased to **128** for both training and validation so that nearly all **31 identities** can appear within a batch, making informative triplets more likely.
- The scheduler learning-rate start value was adjusted to **3.5e-5** to account for the larger batch size.
- The triplet **margin** was fixed at **0.7**, encouraging stronger separation between visually similar jaguars.
- The triplet formulation uses **`F.margin_ranking_loss`** (hinge-style loss). Preliminary tests with **Softplus** did not converge reliably and are omitted here.
- **Early stopping was disabled** so that all runs could be observed for the full **30 epochs**, including possible late-stage instability.

The experiment is run on the **Round 2** dataset under the same preprocessing and evaluation bundle used in the main leaderboard studies.

## Main Findings

The ablation shows that the mining strategy materially affects both convergence behavior and final retrieval quality.

Three patterns are most important.

First, **difficulty-aware strategies outperform purely random sampling on the main retrieval metric mAP**. The strongest result in this study is achieved by **semi-hard mining**, which reaches **0.6720 val/mAP**, followed by **hard mining** at **0.6598**. **Random mining** is clearly weaker on mAP (**0.6262**), and **weighted mining** performs worst in this setup (**0.6001**).

Second, the picture is more nuanced when looking beyond mAP. **Random mining** attains the highest **Rank-1** (**0.9524**), even though it underperforms on mAP and on the embedding diagnostics. This indicates that Rank-1 alone would give an incomplete picture of mining quality here. By contrast, **semi-hard mining** combines the best mAP with the highest **similarity gap** (**0.7311**) and **silhouette score** (**0.5780**), suggesting a more globally structured embedding space.

Third, the mining strategies differ clearly in their training dynamics. As shown in **Figure 2**, **semi-hard mining** becomes unstable after roughly **epoch 14**, with training loss rising again despite strong validation results. **Hard**, **weighted**, and **random** mining all show more monotonic loss reduction. This means the strongest final retrieval result does not necessarily come from the visually smoothest training trajectory.

## Main Results

**Table 1. Peak validation performance by mining strategy.**

| Mining strategy | Best val/mAP | Best val/Rank-1 | Best silhouette | Best sim gap | Final val/loss |
|---|---:|---:|---:|---:|---:|
| Semi-hard | 0.6720 | 0.9490 | 0.5780 | 0.7311 | 0.5938 |
| Hard | 0.6598 | 0.9456 | 0.5600 | 0.6967 | 0.7466 |
| Random | 0.6262 | 0.9524 | 0.4601 | 0.6386 | 0.4884 |
| Weighted | 0.6001 | 0.9286 | 0.4519 | 0.6594 | 0.5915 |

<p align="center"><img src="../../results/round_2/kaggle_mining/Bildschirmfoto 2026-03-17 um 23.06.19.png" width="80%" /></p>
<p align="center"><em>Figure 1. Validation metrics across epochs for the four triplet mining strategies. The panels show similarity gap, silhouette, Rank-1, pairwise AP, mAP, and validation loss.</em></p>

<p align="center"><img src="../../results/round_2/kaggle_mining/W&B Chart 17.3.2026, 23_05_49.png" width="80%" /></p>
<p align="center"><em>Figure 2. Training loss across epochs for the same mining strategies.</em></p>

## Analysis

### Retrieval performance

The most important ranking metric in this study is **validation mAP**, because it better reflects overall retrieval quality across all relevant matches than Rank-1 alone. Under this criterion, the strategies order themselves as:

**semi-hard > hard > random > weighted**.

This ordering is also broadly supported by the embedding diagnostics. **Semi-hard mining** produces the strongest overall separation in the feature space, with the highest **silhouette** and **similarity gap**, while **hard mining** remains close behind. **Random mining** is competitive on Rank-1 but weaker on the more global metrics, suggesting that it can still place a correct match first in many cases without organizing the embedding space as cleanly overall. **Weighted mining** appears smoother, but in this configuration it does not translate that stability into the strongest retrieval results.

### Training dynamics

The training curves show that raw loss should be interpreted cautiously for triplet-based objectives. The mined pairs change as the embedding space changes, so a noisier or non-monotonic loss curve is not unusual. This is especially visible for **semi-hard mining**, whose training loss begins to rise again after the middle of training. Despite that instability, the corresponding validation metrics remain strong and ultimately yield the best mAP in this ablation.

By contrast, **hard mining** shows a more stable training trajectory and still reaches strong validation performance. This makes it an attractive and robust option, even if it does not quite surpass semi-hard mining in this specific comparison.

### What the diagnostics suggest

The combination of **mAP**, **silhouette**, and **similarity gap** is particularly informative here. It suggests that the most effective strategies are not simply those that minimize validation loss fastest, but those that shape the embedding space most usefully for retrieval.

Under that view, **semi-hard mining** is the strongest setting in this experiment, because it gives the best global retrieval quality and the clearest cluster structure. **Hard mining** remains a strong alternative. **Random mining** is clearly not catastrophic, but it is less reliable when judged beyond Rank-1. **Weighted mining** offers smoother optimization but is weakest in final retrieval quality under the tested setup.

## Interpretation

This ablation indicates that **triplet mining strategy is a consequential design choice** in Jaguar Re-ID. The model does not behave the same under different sample-selection rules, even when all other components are fixed.

The results do **not** support the conclusion that purely stochastic triplet selection is sufficient. Although **random mining** reaches a strong Rank-1 score, its weaker mAP and weaker embedding diagnostics indicate that difficult-example selection remains important when the goal is robust identity retrieval rather than only top-1 success.

At the same time, the results also show that the best mining strategy is not necessarily the one with the smoothest optimization behavior. In this study, **semi-hard mining** yields the best final retrieval quality despite visible instability in the training loss. Thus, difficulty-aware mining appears beneficial, but its effect must be judged by retrieval metrics rather than by loss curves alone.

## Conclusion

The mining-strategy ablation shows that **difficulty-aware triplet selection improves Jaguar Re-ID performance under the EVA-02 backbone setup**. Among the tested strategies, **semi-hard mining** achieves the best overall validation result, combining the highest **mAP**, **similarity gap**, and **silhouette**. **Hard mining** is close behind and remains a strong, stable alternative. **Random mining** can produce competitive Rank-1, but does not organize the embedding space as effectively overall, while **weighted mining** underperforms in this configuration.

Overall, the experiment supports the broader conclusion that **how triplets are selected matters substantially for fine-grained wildlife Re-ID**, and that this choice should be treated as a first-order component of the training pipeline rather than a minor implementation detail.

## Limitation

This study is restricted to **one backbone (EVA-02)**, **one dataset version (Round 2)**, and **one triplet-margin configuration**. The conclusions are therefore specific to this training regime. In addition, the comparison is based on internal validation metrics rather than the hidden Kaggle test set, so the results should be interpreted as controlled model-selection evidence rather than as a definitive leaderboard ranking of mining strategies.
