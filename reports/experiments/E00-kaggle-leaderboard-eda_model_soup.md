In the model soup experiment, strongly connected to `E00_leaderboard-eda_stability.md`, we aim to answer the following **Research Question**: *Can we improve generalization and refine error patterns by interpolating between weights of models that have converged to different local minima within the same loss basin?*

Model Souping involves averaging the weights of multiple models trained with the same configuration but different seeds. Following the findings from the backbone ablation study (see `E00_leaderboard-kaggle_backbone.md`), we use the best-performing configuration: an EVA-02 backbone with a Triplet Loss head with Hard mining (please refer to `E00_leaderboard-kaggle_backbone.md` and `config/base/kaggle-base.toml` for further details on hyperparameters, configuration, training scheme and evaluation procedure). To evaluate robustness, we fixed all hyperparameters and varied only the random seed. We evaluated the following seeds: [42, 123, 256, 512, 1024].

First, we attempted a uniform soup using all five seeds. However, including the underperforming Seed 256 led to a significant drop in metrics (approximately 5% lower mAP). We therefore refined the soup after a greedy search over the evaluation metrics, selecting only the two best-performing checkpoints (Seed 42 and Seed 123). This approach aims to combine the most discriminative features learned by the two strongest models.

The search and the entire analysis are performed within a subfolder contained in `retrieval`, which runs a dedicated stability analysis computing the mean and variance of performance metrics across models trained with different seeds, optimizers, and schedulers. Optionally, it also reports an error analysis of the top 20 errors for query images together with the first image in the gallery predicted for that query.

The following table compares the model soup against the best individual model and the mean performance across seeds.

Comparing the Model Soup against the mean of individuals:

| Model | mAP | Rank-1 | Pairwise AP | Sim Gap | Silhouette |
|------|------|------|------|------|------|
| Best Individual (Seed 42) | 0.6639 | 0.9422 | 0.9025 | 0.7972 | 0.6539 |
| Model Soup (Seeds 42 + 123) | 0.6547 | 0.9286 | 0.8772 | 0.6361 | 0.5896 |
| Mean Individual | 0.6509 | 0.9367 | 0.9005 | 0.7863 | 0.6479 |

The Model Soup achieved an mAP of 0.6547, which is slightly above the average performance of the individual models but below the single best seed. Interestingly, the soup's diagnostic metrics (similarity gap and silhouette score) are lower than those of the individual models. This suggests that while weight averaging can help stabilize retrieval ranking performance (mAP), it may also transform the embedding clusters into less compact structures in the latent space.

We additionally compared the errors of the Model Soup against a representative individual model (Seed 512) by logging a table of the top 20 errors into the Weights & Biases dashboard.
- For the EVA-02 + Triplet Loss model trained with Seed 512, errors often occurred on queries where the jaguar appeared in significantly different poses or lighting conditions (for example, Query train_0057.png vs Pred train_0108.png). The individual model sometimes hyper-focuses on global shape rather than spot patterns, confirming issues previously observed in the animal re-identification literature.
- For the Soup model, the error profile appears slightly different. Many mistakes involve identities such as jaguar 18 and 19, which likely have very similar spot configurations. These cases are difficult not only for the model but also for human annotators when inspecting the images.

Since backgrounds were removed in the round_2 dataset, the models are forced to focus primarily on the animal itself. The top-20 error tables show that most mistakes involve:
- head-only crops, where rosette patterns are missing  
- highly distorted poses, where the spot pattern is warped  
- cross-identity similarity, where certain jaguars appear extremely close in the embedding space (for example jaguar 18 frequently confused with 5 or 19)

Finally, some identities are consistently represented by blurred or partially occluded images due to vegetation crossing the animal body. This helps explain why augmentations such as random crop size, gaussian blur, and heavy color alterations may affect performance. In contrast, horizontal flip does not appear to hurt performance and may even help the model generalize rather than rely too strongly on pose-specific cues. 

Overall, the pipeline appears robust but sensitive to sampling stochasticity. The high Rank-1 stability confirms that the EVA-02 + Triplet architecture is a solid choice for this task. While the Model Soup did not provide a large improvement in mAP, it smoothed the performance relative to the average across seeds. For the final Kaggle submission, using the best seed (42) remains the optimal strategy, while the Model Soup provides a representation that is less likely to suffer from failures such as those observed with Seed 256.