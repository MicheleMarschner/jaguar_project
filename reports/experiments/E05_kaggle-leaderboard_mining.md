This experiment focuses on one of the core metric-learning components of our pipeline: the Triplet Loss mining strategy. Following the architecture established in our backbone ablation study, we now evaluate how different ways of selecting positive and negative samples within a batch influence the model's ability to cluster jaguar identities effectively. Therefore the **Research Question** we try to answer is: *How does the choice of triplet mining strategy - from purely stochastic to difficulty-based selection - affect the convergence and final retrieval performance of a Vision Transformer backbone in a species-specific re-identification task?* 

Despite having previously shown results for convolution-based models (i.e., ConvNeXt-V2 and EfficientNet-B4 from the Timm libraries) in `E00_leaderboard-eda_stability.md`, this ablation is performed exclusively on the EVA-02 backbone. This allows us to generalize the findings specifically to Vision Transformer backbones. All other hyperparameters, augmentation strategies, and optimization details follow the best-performing configuration from the second round of the Kaggle Jaguar Re-ID challenge.

The model architecture remains the `JaguarIDModel` (see `src/models/jaguarid-models.py` and `E00_leaderboard-eda_stability.md` for a full discussion) with `head_type` set to triplet. This implementation follows the Bag of Tricks (BoT) approach for person Re-ID and the OpenAnimals framework, which promotes a strong baseline combining:

- Global feature extraction from the EVA-02 backbone (while OpenAnimals combines global and local descriptors, here we rely on global features only)
- BatchNorm neck to normalize features before loss computation
- Triplet Loss + Cross Entropy/Focal Loss: the classification loss handles closed-set identity prediction, while Triplet Loss refines the embedding space for metric similarity. Focal Loss additionally addresses class imbalance. While OpenAnimals reports benefits from label smoothing in some datasets, we did not observe improvements in this setting, and therefore disabled the `label_smoothing` parameter (results omitted for brevity).

As implemented in `src/utils/utils_losses.py`, we ablate four distinct mining strategies:

- Hard Mining (hard): selects the hardest positive (maximum distance) and hardest negative (minimum distance) in the batch, forcing the model to learn from the most difficult examples
- Weighted Mining (weighted): computes a softmax-weighted average over all positive and negative pairs, where harder samples receive higher weights, providing smoother gradients than pure hard mining
- Random Sampling (random): randomly selects one positive and one negative sample, serving as a baseline to evaluate whether structured mining is necessary
- Semi-Hard Mining (semi-hard): selects negatives hat are further from the anchor than the positive but still within the margin; if none are available, it falls back to the hardest negative

To ensure stability and enable a fair comparison, we modified the configuration from `E00_leaderboard-kaggle_backbone.md` as follows:

- Training and validation `batch_size`: increased to 128. This allows the model to observe nearly all 31 identities in each batch, ensuring the presence of meaningful hard triplets at every step
- Learning rate `lr_start`: adjusted to 3.5e-5 using the JaguarIDScheduler to account for the larger batch size
- Margin `m`: set to 0.7. A higher margin is required for fine-grained identification, where identities appear visually similar. This forces the embedding space to separate identities more aggressively
- Loss formulation: we used `F.margin_ranking_loss` (hinge loss). Preliminary tests with Softplus (ln(1 + e^x)) did not converge. Our intuition is that Softplus provided insufficient gradient for already well-separated triplets, while hinge loss maintains a constant gradient until the margin is satisfied
- No `early_stopping`: training was run for the full 30 epochs to observe whether hard mining leads to late-stage overfitting

<insert Weights & Biases dashboar screenshot here> 

Looking at the val/mAP and val/rank1 plots, a clear performance gap emerges:
- Hard Mining and Weighted Mining are the top-performing strategies, both reaching validation mAP above 0.65 and Rank-1 accuracy around 95%
- Random and Semi-Hard mining lag significantly behind. Random sampling performs the worst, confirming that the model requires exposure to difficult examples to learn discriminative spot patterns

The following table summarizes the peak validation performance reached by each mining strategy across 30 epochs on the Round 2 dataset.

| Mining Strategy | Best val/mAP | Best val/Rank-1 | Best Silhouette | Best Sim Gap | Final val/Loss |
|----------------|-------------|-----------------|------------------|--------------|----------------|
| Semi-Hard      | 0.6720      | 0.9490          | 0.5780           | 0.7311       | 0.5938         |
| Hard           | 0.6598      | 0.9456          | 0.5600           | 0.6967       | 0.7466         |
| Random         | 0.6262      | 0.9524          | 0.4601           | 0.6386       | 0.4884         |
| Weighted       | 0.6001      | 0.9286          | 0.4519           | 0.6594       | 0.5915         |

The most informative metrics are val/silhouette and val/sim_gap:

- Hard Mining produces the most discriminative embedding space, achieving the highest silhouette score (approximately 0.58) and the largest similarity gap (approximately 0.7). This indicates strong separation between identities and validates the use of Triplet Loss in the final configuration
- Weighted Mining achieves similar results, with slightly more stable behavior in early epochs (5–15), as soft weighting reduces oscillations caused by extremely hard triplets

TThe val/loss curve is noticeably noisier than accuracy metrics. This is expected in Triplet Loss training, since the mined triplets change dynamically as the embedding space evolves. Therefore, mAP and Pairwise AP are more reliable indicators of performance than the raw loss value. Semi-hard mining shows instability around epoch 15, which likely occurs when the model exhausts useful semi-hard triplets and must transition toward harder samples.

Overall, the results show that Hard Mining is the most effective strategy for Jaguar Re-ID. While Weighted Mining offers a smoother alternative, Hard Mining achieves the best separation between identities and the strongest retrieval performance. The combination of a large batch size (128) and a high margin (0.7) provides the necessary conditions for these strategies to succeed, effectively adapting a general-purpose Vision Transformer into a specialized jaguar identification model.

