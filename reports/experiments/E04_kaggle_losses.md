# Loss and Head Ablation (Data - Round 2)

**Experiment Group:** Backbone and training ablations

## Main Research Question

How do angular margin-based classification losses compare against stabilized metric-learning objectives in shaping a discriminative embedding space for fine-grained jaguar re-identification in a low-data regime, and which loss/head configuration performs best in the EVA-02 pipeline?

## Motivation

While the backbone determines which visual features are available, the loss function and head architecture determine how these features are organized in the embedding space. This is especially important in animal re-identification, where the model must separate visually similar individuals from relatively few labeled examples.

In this setting, the main design choice is whether to use a **classification-style objective** that encourages separation between identity prototypes, or a **metric-learning objective** that directly optimizes image-to-image distances. The present experiment therefore compares standard softmax training, angular-margin classification losses, and stabilized triplet-based objectives within the same jaguar-specific pipeline.

## Model and Shared Experimental Setup

All runs use the same `JaguarIDModel` implementation (`src/jaguar/models/jaguarid_models.py`) with **EVA-02** as the backbone. The augmentation pipeline, optimizer family, scheduler, evaluation bundle, and general training protocol follow the strongest EVA-02 setup used elsewhere in the project. For the backbone-level context, see the dedicated backbone ablation experiment.

The purpose here is not to change the backbone, but to isolate the effect of the **loss function and head design**.

### Stabilized triplet setup

As noted in prior re-identification literature and in our own stability experiments, plain triplet training can be difficult to optimize. To keep the triplet-based variants competitive and fair in comparison to the classification losses, we use the following stabilized setup:

- **batch size = 128** for both training and validation, increasing the number of valid positives and hard negatives within a batch
- a **Bag-of-Tricks-style head**, where the embedding passes through a neck with `BatchNorm` and a classifier head
- **Triplet Loss** applied on the embedding, combined with either **Cross-Entropy** or **Focal Loss** on the logits

This hybrid formulation is intended to stabilize optimization by combining class-level separation with local metric refinement.

## Loss Configurations Under Comparison

The following loss/head configurations are evaluated:

- **Softmax**: standard Cross-Entropy baseline
- **ArcFace**: additive angular margin, using `cos(θ + m)` on the target class
- **CosFace**: additive cosine margin, using `cos(θ) - m`
- **SphereFace**: multiplicative angular margin, using `cos(m · θ)`
- **Triplet + Focal (Hard Mining)**
- **Triplet + CE (Hard Mining)**

The comparison is designed to test whether angular-margin classification losses or stabilized metric-learning objectives produce the stronger retrieval space for jaguar identification.

## Main Results

### Peak validation results

The peak validation results are summarized in **Table 1**.

**Table 1. Peak validation performance across loss/head configurations.**

| Loss Configuration | Best val/mAP | Best val/Rank-1 | Best Sim Gap | Convergence Epoch |
|---|---:|---:|---:|---:|
| Triplet + Focal (Hard) | 0.6705 | 0.9456 | 0.7818 | 20 |
| Triplet + CE (Hard) | 0.6598 | 0.9456 | 0.6967 | 18 |
| ArcFace | 0.6482 | 0.9524 | 0.7257 | 25 |
| CosFace | 0.6356 | 0.9422 | 0.6860 | 22 |
| Softmax | 0.5304 | 0.9150 | 0.4352 | 15 |
| SphereFace | 0.3956 | 0.8844 | 0.3286 | 30+ |

A clear ranking emerges. The strongest configuration under the primary validation metric is **Triplet + Focal (Hard Mining)**, followed by **Triplet + CE (Hard Mining)** and **ArcFace**. **Softmax** is clearly weaker, and **SphereFace** performs worst by a large margin.

The results also show that the best loss depends somewhat on the metric used. **ArcFace** achieves the highest **Rank-1** (**0.9524**), but it does not reach the best **mAP**. By contrast, **Triplet + Focal (Hard)** yields the strongest **mAP** (**0.6705**) and also the largest **similarity gap** (**0.7818**), indicating the best overall retrieval quality and embedding separation among the tested options.

### Training and validation dynamics

The Weights & Biases curves make these differences visible over training.

<p align="center"><img src="../../results/round_2/baseline/Bildschirmfoto 2026-03-17 um 23.03.34.png" width="90%" /></p>
<p align="center"><em>Figure 1. Validation curves across epochs for the different loss/head configurations with EVA-02.</em></p>

<p align="center"><img src="../../results/round_2/baseline/W&B Chart 17.3.2026, 23_03_09.png" width="90%" /></p>
<p align="center"><em>Figure 2. Training-loss curves across epochs for the different loss/head configurations with EVA-02.</em></p>

Several patterns are consistent across the plots:

- The two **triplet-based** configurations rise to the strongest **validation mAP** trajectories.
- **ArcFace** and **CosFace** remain competitive, but plateau slightly below the best triplet setup on mAP.
- **Softmax** improves steadily but saturates much earlier and at a clearly lower level.
- **SphereFace** shows the clearest optimization problems, with extremely large training loss values and much weaker validation metrics throughout training.

Looking specifically at **validation loss**, the triplet-based runs and the stronger angular-margin losses converge to low and stable values, whereas **SphereFace** remains on a completely different scale. This supports the interpretation that its multiplicative margin is too difficult to optimize robustly in the present jaguar setting.

The **pairwise AP**, **mAP**, and **sim-gap** plots tell a consistent story: the best triplet configuration does not only improve one metric in isolation, but produces the strongest global retrieval space across several diagnostics.

### Interpretation of the loss comparison

The main outcome of this experiment is that **stabilized metric learning outperforms purely classification-based objectives on mAP** in this low-data wildlife setting.

A likely explanation is structural. Angular-margin classification losses such as **ArcFace**, **CosFace**, and **SphereFace** rely on learning stable class prototypes. When only relatively few non-duplicate images are available per jaguar, those prototypes may be harder to estimate reliably. Triplet-based training instead focuses directly on **image-to-image relationships**, which may be better suited to capturing subtle local differences in coat pattern and body appearance.

At the same time, the comparison is not one-sided. **ArcFace** remains very strong on **Rank-1**, and the gap between ArcFace and triplet-based training is much smaller there than for mAP. This suggests that ArcFace can still learn a strong top-match signal, but is somewhat weaker at preserving the broader ranking quality needed to retrieve all relevant positives consistently.

The **similarity-gap** results reinforce this reading. The highest sim-gap is achieved by **Triplet + Focal (Hard)**, which indicates the clearest separation between same-identity and different-identity pairs under the tested losses.

## Practical Outcome for the Kaggle Pipeline

Within this experiment, **Triplet + Focal (Hard Mining)** is the strongest overall configuration and therefore becomes the preferred loss/head setup for the later pipeline.

According to the project notes, this same configuration was also the one used in the broader Kaggle system variant that reached **90.08% mAP on the public Kaggle test set**. In that sense, the internal validation results and the external leaderboard behavior were aligned for this experiment.

A more detailed analysis of the mining component itself is provided separately in the dedicated mining experiment, where the sample-selection strategy inside the triplet framework is isolated explicitly.

## Notation and Parameter Note

Let `f(x) ∈ R^d` denote the L2-normalized embedding for image `x`, and let `W_j ∈ R^d` denote the L2-normalized class weight for class `j`. The angle between them is `θ_j`, so that `cos(θ_j) = W_j^T f(x)`. For the ground-truth class `y`, we write `θ = θ_y`.

The main parameters are:

- `m`: margin parameter controlling the enforced separation between classes or pairs
- `s`: scale factor applied to logits to stabilize optimization

This gives the following loss-specific forms:

- **Softmax / CE**: logits use `s · cos(θ_j)`
- **ArcFace**: target logit uses `s · cos(θ + m)`
- **CosFace**: target logit uses `s · (cos(θ) - m)`
- **SphereFace**: target logit uses `s · cos(m · θ)`

For the metric-learning configurations:

- **Triplet Loss** operates on triplets `(a, p, n)` and enforces `d(a, p) + m < d(a, n)`
- **Hard Mining** selects the hardest positive and hardest negative inside the batch
- **Focal Loss** adds a modulating factor to emphasize hard classification examples

## Conclusion

This experiment shows that the choice of loss function materially affects Jaguar Re-ID performance, even when the backbone and general training pipeline are held fixed.

The strongest overall result is obtained with **Triplet + Focal (Hard Mining)**, which achieves the best validation **mAP** and **similarity gap**, while **ArcFace** achieves the best **Rank-1**. In practice, the results favor **carefully stabilized metric learning** over purely classification-based angular-margin losses for this fine-grained, low-data wildlife setting.

The broader conclusion is therefore not that margin-based classification is ineffective, but that for jaguar identification with limited per-identity data, **image-to-image metric structure appears more beneficial than prototype-centric separation alone**.
