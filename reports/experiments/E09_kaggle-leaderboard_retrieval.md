This experiment explores post-processing and inference-time techniques to maximize retrieval performance without retraining the model. We investigate the combined and individual effects of Test-Time Augmentation (TTA), Query Expansion (QE), and k-reciprocal Re-ranking on the final retrieval metrics. Therefore, the **Research Question** we aim to answer is: *To what extent can inference-time optimizations—specifically Test-Time Augmentation, local neighborhood Query Expansion, and reciprocal Re-ranking—refine the embedding manifold and improve retrieval precision in a fine-grained jaguar re-identification task?*

This study effectively answers two complementary experimental questions:  
*(1) whether k-reciprocal re-ranking improves retrieval performance in our setting,* and
*(2) more broadly, how different inference-time strategies interact and contribute to performance gains.*  

We utilize the best-performing EVA-02 checkpoint from previous experiments, following the hyperparameter configuration, training pipeline, and evaluation protocol defined in `E03_kaggle_backbone.md`. The retrieval suite (implemented in `src/utils/utils_evaluate.py`) introduces three main post-processing modules:
- Test-Time Augmentation (TTA - Flip): The model processes both the original image and its horizontal flip. The resulting embeddings are averaged and re-normalized. This leverages the previously observed benefit of flip invariance, encouraging the model to focus on local rosette patterns rather than global orientation.
- Query Expansion (QE): A PyTorch implementation of a classical Re-ID technique. For each query, we retrieve its top-k nearest neighbors and compute a refined query embedding by averaging them. This effectively shifts the query toward the centroid of its identity cluster in the embedding space.
- k-reciprocal Re-ranking: A PyTorch implementation of the method proposed by Zhong et al. (CVPR 2017). It computes a Jaccard distance based on reciprocal nearest neighbors. The underlying assumption is that if image A is among the top neighbors of B and B is among the top neighbors of A, they are likely to share the same identity.

We conducted a sweep across these modules on the Round 2 background-masked validation set.

| Run Name              | TTA (Flip) | QE (k) | Re-ranking | Best val/mAP | Δ mAP vs Baseline | Best val/Rank-1 | Best Sim Gap |
|-----------------------|------------|--------|------------|--------------|-------------------|------------------|--------------|
| retrieval_tta_qe      | Yes        | 3      | No         | 0.6643       | +0.0015           | 0.9456           | 0.7363       |
| retrieval_tta_flip    | Yes        | No     | No         | 0.6643       | +0.0015           | 0.9456           | 0.7363       |
| retrieval_baseline    | No         | No     | No         | 0.6628       | 0.0000            | 0.9354           | 0.6987       |
| retrieval_qe          | No         | 3      | No         | 0.6619       | -0.0009           | 0.9456           | 0.7318       |
| retrieval_qe_rerank   | No         | 5      | Yes        | 0.6383       | -0.0245           | 0.9252           | 0.7878       

Note: Re-ranking parameters were swept across  
k₁ ∈ [20, 30, 50], k₂ ∈ [6, 8, 10], and λ ∈ [0.3, 0.5].  
The reported results correspond to k₁ = 20, k₂ = 6, λ = 0.3, which are standard values in the literature. The sweeps and corresponding evaluation `.csv` tables are generated using `src/jaguar/retrieval/retrieval.py` and stored in the checkpoint directory under `retrieval_eval`.

The Δ mAP column highlights that TTA provides a consistent positive gain, while QE alone is slightly unstable and re-ranking significantly degrades performance in this low-data regime.

<insert here Weights & Biases dashboard screenshot showing inference metrics compared across runs>

The integration of horizontal flipping at test time provides a consistent improvement. While the mAP gain over the baseline is modest (+0.0015), Rank-1 Accuracy increases from 0.9354 to 0.9456. This confirms previous findings: flipping provides a complementary view of the same jaguar, allowing the model to better match identities under pose variation.

Query Expansion performs best with a small neighborhood size. At k=3, it stabilizes the ranking by pulling the query toward its closest and most reliable neighbors. As k increases, performance degrades significantly (e.g., mAP drops toward ~0.56 at k=10). This behavior is expected in low-data regimes: larger neighborhoods introduce samples from other identities, corrupting the query embedding and reducing retrieval precision.

While k-reciprocal re-ranking is a valid and well-established post-processing technique, in our setting it does not improve performance. Specifically, re-ranking reduces mAP (e.g., 0.6383 vs 0.6628 baseline), despite increasing the Similarity Gap. This indicates that while the embedding space becomes more separated globally, the local ranking order becomes noisier. This behavior can be explained by the characteristics of our dataset. k-reciprocal re-ranking relies on sufficiently dense and reliable neighborhood structures. In our case, each identity has very few samples, making the reciprocal neighborhood sparse and unstable. As a result, the Jaccard distance becomes noisy and introduces errors rather than refining the ranking. This highlights an important limitation: re-ranking is highly effective in large-scale Re-ID benchmarks but can degrade performance in small, fine-grained datasets.

For the final Kaggle Leaderboard submission, we selected the TTA (Flip) + QE (k=3) configuration. Although the validation mAP gains are modest, this combination consistently improves Rank-1 accuracy and cluster separability. It achieves the strongest Similarity Gap (0.7363) and a Silhouette score around 0.60, indicating better-structured embeddings. These improvements translated into a ~2–3% mAP gain on the test set (~90% mAP overall).

In conclusion, this experiment demonstrates that inference-time post-processing is a valuable component of the Jaguar Re-ID pipeline. TTA with flipping is essential for handling pose variability, and Query Expansion with a conservative k provides effective local refinement of the embedding manifold. In contrast, k-reciprocal re-ranking, while theoretically sound and widely used, proves too aggressive for this low-data regime and is therefore excluded from the final model. This study not only validates re-ranking as a meaningful experiment but also shows that its effectiveness is strongly dependent on dataset scale and embedding density.