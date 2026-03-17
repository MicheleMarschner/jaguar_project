# Augmentation Ablation Experiment (Data - Round 2)

**Experiment Group:** Robustness and diagnostic experiments

## Main Research Question

How do different augmentation strategies affect fine-grained jaguar re-identification, and can biologically implausible transformations such as horizontal flipping still act as useful regularizers in a low-data wildlife Re-ID setting?

## Motivation

This experiment studies the effect of data augmentation on jaguar re-identification under a fixed training pipeline. In this setting, augmentation is not a minor implementation detail: jaguar identities are defined by fine-grained rosette patterns, which are spatially complex and asymmetric across the two flanks, while the dataset is comparatively small and visually heterogeneous. The central trade-off is therefore between stronger regularization and preservation of identity-defining detail.

In particular, we ask whether augmentations that are biologically incorrect or structurally disruptive—most notably horizontal flipping and strong erasing—can still improve generalization by reducing pose dependence and increasing robustness to camera-trap noise and occlusion.

## Setup / Intervention

All runs follow the same base pipeline defined in `config/base/kaggle_base.toml` and the same general setup used in the leaderboard experiments, with the **EVA-02** backbone and **Triplet Loss with Semi-Hard mining**. Images are resized to the backbone-native high resolution to preserve rosette detail.

Only the augmentation recipe is changed.

We compare eight controlled augmentation settings:

- `aug_none`: no augmentation
- `aug_flip_only`: horizontal flipping only
- `aug_geom_noflip_nocol_noerase`: geometric augmentation without flip, color jitter, or erasing
- `aug_geom_nocol_noerase`: geometric augmentation plus horizontal flipping, but without color jitter or erasing
- `aug_geom_noerase`: geometric augmentation plus flipping and color jitter, but without erasing
- `aug_full_pipeline`: broader augmentation suite including Gaussian blur and random erasing (`p = 0.1`)
- `aug_full_pipeline_erase025`: same as above, but with stronger random erasing (`p = 0.25`)
- `aug_curr_baseline`: the final optimized recipe used in later experiments, combining geometry, color augmentation, and stronger random erasing while excluding Gaussian blur and Random Resized Crop

## Method / Procedure

The goal of the experiment is to isolate which augmentation components are beneficial and which are harmful for jaguar Re-ID.

The comparison is structured so that individual changes can be interpreted incrementally. In particular, the sequence from `aug_none` to the geometry-based variants makes it possible to assess the effect of flipping, color perturbation, and erasing separately, while the comparison between the full pipeline variants and `aug_curr_baseline` tests whether Gaussian blur and Random Resized Crop help or instead remove too much fine-grained identity information.

Evaluation is based on the same internal validation metrics used throughout the retrieval pipeline, with a primary focus on identity-balanced **mAP**, and complementary inspection of **Rank-1**, **Silhouette**, **Similarity Gap**, and the training/validation curves over epochs.

## Main Results

**Table 1** summarizes the best validation performance of each augmentation suite.

| Run Name | Best val/mAP | Δ mAP vs Baseline | Best val/Rank-1 | Best Silhouette | Best Sim Gap |
|---|---:|---:|---:|---:|---:|
| aug_curr_baseline | 0.7318 | +0.0000 | 0.9456 | 0.5960 | 0.7318 |
| aug_full_pipeline_erase025 | 0.6905 | -0.0413 | 0.9354 | 0.5427 | 0.6905 |
| aug_full_pipeline | 0.6781 | -0.0537 | 0.9252 | 0.5316 | 0.6781 |
| aug_geom_nocol_noerase | 0.6384 | -0.0934 | 0.9456 | 0.5937 | 0.6384 |
| aug_geom_noerase | 0.6359 | -0.0959 | 0.9422 | 0.5522 | 0.6359 |
| aug_flip_only | 0.6358 | -0.0960 | 0.9388 | 0.5968 | 0.6358 |
| aug_geom_noflip_nocol_noerase | 0.6329 | -0.0989 | 0.9150 | 0.5305 | 0.6330 |
| aug_none | 0.6149 | -0.1169 | 0.9150 | 0.5305 | 0.6149 |

<p align="center"><img src="../../results/round_2/baseline/Bildschirmfoto 2026-03-17 um 23.17.29.png" width="90%" /></p>
<p align="center"><em>Figure 1. Validation trajectories for the augmentation ablation, including similarity gap, silhouette, Rank-1, pairwise AP, mAP, and validation loss.</em></p>

<p align="center"><img src="../../results/round_2/baseline/W&B Chart 17.3.2026, 23_17_14.png" width="80%" /></p>
<p align="center"><em>Figure 2. Training loss across augmentation settings.</em></p>

## Analysis

### Overall ranking of augmentation strategies

The best-performing setup is clearly **`aug_curr_baseline`**, which reaches the highest validation **mAP (0.7318)** and also the highest **Similarity Gap (0.7318)**, while matching the strongest **Rank-1** level observed in the experiment (**0.9456**). The two broader full-pipeline variants follow behind, with `aug_full_pipeline_erase025` outperforming `aug_full_pipeline`, but both remain clearly below the optimized baseline.

The weakest settings are the minimal or no-augmentation variants. `aug_none` performs worst overall, and the geometry-only variants improve on it only modestly unless additional regularization is added. This indicates that augmentation is not optional in this setting; some level of perturbation is necessary to obtain strong generalization on the curated jaguar split.

### Horizontal flipping

One of the central questions in jaguar Re-ID is whether horizontal flipping should be avoided because left and right flanks are biologically different. The results here point in the opposite direction: flipping is still beneficial as a regularizer.

This is most visible in the comparison between `aug_geom_noflip_nocol_noerase` and `aug_geom_nocol_noerase`. Adding flipping raises validation **mAP** from **0.6329** to **0.6384** and **Rank-1** from **0.9150** to **0.9456**. The gain is not enormous, but it is consistent in the favorable direction.

A plausible interpretation is that flipping reduces over-reliance on global orientation and encourages the model to learn more local or pose-robust rosette evidence. In a low-data regime, it also effectively increases the variety of seen configurations. This does not mean that flipping is biologically correct; rather, it appears to function as a useful inductive bias for generalization.

### Color and occlusion robustness

Adding color perturbation without erasing (`aug_geom_noerase`) does not improve performance over the flip-plus-geometry setting. In fact, it is slightly lower on both **mAP** and **Silhouette**. On its own, color augmentation is therefore not the main source of the final improvement.

By contrast, stronger occlusion-style regularization appears more useful when applied as part of the final recipe. The optimized baseline, which combines geometry, color augmentation, and heavier random erasing while excluding blur and Random Resized Crop, outperforms all alternatives by a clear margin. This is consistent with the intended role of erasing: it simulates partial occlusion and encourages the model to rely on multiple identity cues instead of a single best patch.

At the same time, the comparison between `aug_full_pipeline` and `aug_full_pipeline_erase025` suggests that stronger erasing can help within the larger pipeline as well, since increasing the erasing probability from **0.1** to **0.25** improves **mAP** from **0.6781** to **0.6905**.

### Why blur and Random Resized Crop were removed

The full-pipeline variants, which include Gaussian blur and Random Resized Crop, are clearly worse than `aug_curr_baseline`. Since the baseline keeps the other regularization components but excludes blur and RRC, the most direct interpretation is that these two operations remove too much useful identity information for this task.

This is plausible for jaguar Re-ID. Gaussian blur weakens the high-frequency boundaries and local contrasts that define rosette structure, while Random Resized Crop may exclude critical flank regions entirely. In camera-trap imagery, where animals are often already partially visible, further cropping can make the remaining evidence too incomplete for reliable identity matching.

### Training dynamics

The learning curves are broadly consistent with the ranking in Table 1. The experiment shows the characteristic transition around the mid-training phase that also appears in other leaderboard experiments, after which the stronger configurations separate more clearly. `aug_curr_baseline` maintains the strongest late-stage validation trajectory and ends with the best mAP and similarity-gap values.

The training-loss curves, by contrast, are much less informative for ranking augmentation quality. All runs converge to similarly low values, and some weaker augmentation settings even show comparably smooth or slightly lower training loss. This is expected: augmentation is primarily intended to improve generalization, not necessarily to minimize training loss fastest.

## Key Result / Takeaway

The most effective augmentation strategy for Jaguar Re-ID is **not** the most aggressive or most ImageNet-like pipeline. Instead, the best results come from a more targeted recipe: geometry, color variation, and strong random erasing, while excluding Gaussian blur and Random Resized Crop.

Two conclusions stand out.

First, **horizontal flipping is beneficial**, despite biological flank asymmetry. In this setting, it acts as a regularizer that improves pose robustness rather than corrupting the task.

Second, **not all standard augmentations transfer well to wildlife Re-ID**. Blur and RRC appear to remove too much identity-specific structure, whereas stronger erasing is useful when it mimics real partial occlusion.

Overall, the results support the final choice of `aug_curr_baseline` as the default augmentation suite in the later leaderboard pipeline.

## Conclusion

This augmentation ablation shows that data augmentation matters substantially for jaguar re-identification, but only when it is aligned with the structure of the task.

The strongest configuration is the final baseline recipe used in the later pipeline: it outperforms all alternative augmentation suites on validation mAP and similarity gap, while remaining among the best on Rank-1. The results suggest that the best trade-off is achieved by combining geometric and color perturbation with strong occlusion-style regularization, while avoiding blur and aggressive cropping.

Perhaps most importantly, the experiment shows that biologically imperfect transformations such as horizontal flipping can still improve performance when they reduce pose dependence and help the model focus on transferable local appearance cues.

## Limitation

The conclusions here are based on one backbone and one fixed optimization setup. They therefore identify the most effective augmentation recipe for this pipeline, but they do not prove that the same ordering must hold for all backbones or all training objectives. In addition, some components are only tested in bundled configurations, so the interpretation of blur and RRC remains partly indirect through comparison to the final optimized baseline.
