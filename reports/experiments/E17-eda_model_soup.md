# Model Soup (Data - Round 2)

**Experiment Group:** Robustness and diagnostic experiments

## Main Research Question

Can we improve generalization and refine error patterns by interpolating between weights of models that have converged to different local minima within the same loss basin?

---

## Setup / Intervention

This experiment is closely connected to the seed-stability analysis and asks whether the variation across seeds can be exploited constructively through **model souping**.

Model Souping averages the weights of multiple checkpoints trained with the **same architecture and training recipe**, but with different random seeds. The underlying idea is that if these models converge to compatible regions of parameter space, averaging their weights may retain robust shared structure while smoothing out seed-specific idiosyncrasies.

Following the earlier backbone and stability experiments, we use the same best-performing base configuration:

- **EVA-02** backbone
- **Triplet-based training with hard mining**
- the same optimizer, scheduler, augmentation suite, and evaluation protocol as in the main leaderboard pipeline

For implementation details, see the corresponding backbone and stability experiment pages as well as the base configuration file.

The evaluated seeds are:

- **42**
- **123**
- **256**
- **512**
- **1024**

All hyperparameters remain fixed. Only the random seed changes.

---

## Method / Procedure

We first attempted a **uniform soup over all five seeds**. However, this full soup performed substantially worse than the stronger individual runs. In particular, including the clearly weaker **seed 256** led to a marked drop in retrieval performance, with mAP decreasing by roughly **5 percentage points** relative to the stronger reference runs.

We therefore refined the soup through a more selective procedure. Based on the validation metrics, we retained only the two strongest checkpoints:

- **seed 42**
- **seed 123**

The final reported soup is thus the **two-model soup (42 + 123)**.

In addition to the aggregate metrics, the broader analysis pipeline also supports error analysis over the top failed retrieval cases. Here, we compare the soup qualitatively against a representative individual model and inspect whether weight averaging changes the type of retrieval errors rather than only the final scores.

---

## Main Results

**Table 1. Comparison between the best individual model, the selected model soup, and the mean across seeds.**

| Model | mAP | Δ vs Mean | Δ vs Best | Rank-1 | Δ vs Mean | Δ vs Best |
|---|---:|---:|---:|---:|---:|---:|
| Best Individual (Seed 42) | 0.6639 | +0.0130 | 0.0000 | 0.9422 | +0.0055 | 0.0000 |
| Model Soup (42 + 123) | 0.6547 | +0.0038 | -0.0092 | 0.9286 | -0.0081 | -0.0136 |
| Mean Individual | 0.6509 | 0.0000 | -0.0130 | 0.9367 | 0.0000 | -0.0055 |

The selected soup reaches **mAP = 0.6547**, which is slightly **above the mean performance across seeds** but still **below the best individual model**. The same pattern appears for **Rank-1**: the soup is worse than both the best run and the seed average.

This means that, in this setting, weight averaging provides only a **small stabilizing benefit over the average seed**, but it does **not outperform selecting the strongest individual checkpoint**.

The analysis also indicates that the soup's diagnostic embedding-space metrics, such as **Similarity Gap** and **Silhouette Score**, are lower than those of the stronger individual models. This suggests that although the soup can preserve reasonably good retrieval performance, weight averaging may also make the embedding geometry somewhat less compact or less sharply separated.

---

## Interpretation

### Does model souping help in this pipeline?

Only to a limited extent.

The two-model soup improves slightly over the **mean** individual performance in **mAP**, but not by a large margin, and it does not surpass the **best single seed**. In other words, souping reduces some of the downside of weaker runs, but it also dilutes some of the strongest properties of the best run.

This is consistent with the failed five-model soup: once clearly weaker checkpoints are included, averaging can move the parameters away from the best-performing region rather than toward a better shared solution.

### What does this imply about the loss basin?

The results suggest that at least some seeds are compatible enough to be averaged without catastrophic failure, since the **42 + 123** soup remains competitive. However, the poor result of the full five-seed soup indicates that not all checkpoints lie in equally compatible parts of the landscape.

Thus, the evidence supports a **partially shared but not uniformly safe basin structure**: selective souping can work, but indiscriminate averaging across all seeds is harmful.

### Why might the soup underperform the best individual model?

A likely reason is that the best individual model contains sharper seed-specific discriminative structure that helps retrieval in difficult cases. Averaging weights can smooth these specialized decision boundaries. This may make the model slightly more generic, but also slightly less precise for the hardest identity distinctions.

That interpretation fits the lower embedding diagnostics: the soup appears somewhat smoother, but not clearly better structured.

---

## Error Analysis

We additionally compared the retrieval errors of the soup against a representative individual model (**seed 512**) by logging the top failed cases to Weights & Biases.

The qualitative comparison suggests that the two models fail in somewhat different ways.

For the **individual seed-512 model**, errors often occur when the same jaguar appears under very different **pose**, **lighting**, or **partial visibility** conditions. In these cases, the model sometimes appears to rely too strongly on global body configuration rather than sufficiently robust local spot evidence.

For the **soup model**, the error profile shifts somewhat toward very hard identity confusions, especially among jaguars with highly similar rosette configurations. Several mistakes involve identities that appear visually extremely close even under manual inspection.

Across both models, the dominant hard cases on the **Round 2** background-masked dataset are:

- **head-only crops**, where flank rosette patterns are largely absent
- **strongly distorted poses**, where the visible spot configuration is warped
- **cross-identity similarity**, where some jaguars occupy very nearby regions of the embedding space
- **blurred or partially occluded images**, often due to vegetation or camera-trap artifacts

Because the Round 2 dataset removes most background cues, these errors are especially informative: they show that the remaining difficulty comes primarily from the animal evidence itself rather than from environmental confounds.

---

## Relation to Other Experiments

This experiment also connects back to the augmentation findings. The observed failure modes help explain why some perturbations are more harmful than others.

If many hard cases already involve partial visibility, blur, or unusual pose, then aggressive augmentations such as strong blur, large random crops, or heavy color distortion can remove or weaken the very evidence that remains available. By contrast, **horizontal flipping** does not appear to damage retrieval in the same way and may even improve robustness by discouraging overly pose-specific matching.

Thus, the model-soup error analysis does not only evaluate ensembling at the weight level; it also helps interpret the broader design choices of the final pipeline.

---

## Main Findings

- A **uniform soup over all five seeds** is harmful, mainly because weaker checkpoints degrade the average.
- A **selective two-model soup (42 + 123)** performs **slightly above the mean seed performance** in mAP.
- The soup still remains **below the best individual checkpoint**, both in mAP and Rank-1.
- Embedding-space diagnostics suggest that the soup is somewhat **less compact or less sharply separated** than the stronger individuals.
- The dominant retrieval errors remain biologically difficult cases: head-only views, distorted poses, blurred images, and near-indistinguishable cross-identity patterns.
- For final model selection, **choosing the best individual seed** is preferable to using the soup.

---

## Limitation

This experiment studies only a small number of seeds and one model family under one fixed training recipe. It therefore does not establish a general conclusion about model souping in animal Re-ID. In addition, the interpretation of the error profile is qualitative and based on the top failed retrievals rather than a full quantitative decomposition of error types.

---

## Conclusion

The model-soup experiment shows that **weight averaging is possible but not especially beneficial in this jaguar Re-ID pipeline**.

A selective soup over the two strongest seeds is slightly better than the average seed in **mAP**, but it does not surpass the best individual model and slightly reduces **Rank-1**. The results therefore suggest that the pipeline is already stable enough that **best-checkpoint selection** remains the most effective strategy.

More broadly, the experiment supports a nuanced view of model souping: it can smooth performance relative to weaker runs, but in this setting it does not create a stronger model than the best seed. For the final Kaggle submission, **seed 42** remains the preferred choice, while the soup serves mainly as a robustness check rather than as the final deployed model.
