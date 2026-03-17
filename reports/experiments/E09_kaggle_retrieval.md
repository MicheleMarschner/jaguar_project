# Inference-Time Retrieval Refinement Experiment (Data - Round 2)

**Experiment Group:** Retrieval and post-processing experiments

## Main Research Question

To what extent can inference-time optimizations—specifically **Test-Time Augmentation (TTA)**, **local Query Expansion (QE)**, and **k-reciprocal Re-ranking**—refine the embedding manifold and improve retrieval precision in a fine-grained jaguar re-identification task?

This experiment also addresses two more specific questions:

1. Does **k-reciprocal re-ranking** improve retrieval in our setting?
2. More broadly, how do different inference-time strategies interact, and which combinations are useful in the final pipeline?

## Setup / Intervention

This study uses the best-performing **EVA-02** checkpoint from the preceding training experiments and keeps the training pipeline fixed. No retraining is performed. Instead, we evaluate post-processing modules that act only at inference time.

The retrieval suite, implemented in `src/utils/utils_evaluate.py`, includes three main components:

- **Test-Time Augmentation (TTA - Flip):** the model processes both the original image and its horizontal flip. The resulting embeddings are averaged and L2-normalized.
- **Query Expansion (QE):** for each query, the top-k nearest gallery neighbors are averaged with the original query embedding to obtain a refined query representation.
- **k-reciprocal Re-ranking:** a PyTorch implementation of the method of Zhong et al. (CVPR 2017), which augments the original distance with a Jaccard-distance term based on reciprocal nearest neighbors.

All evaluations were performed on the **Round 2 background-masked validation set**.

## Method / Procedure

We ran a controlled sweep over the three post-processing modules. The main reported runs are summarized in Table 1.

**Table 1. Retrieval refinement sweep on the Round 2 validation set.**

| Run Name | TTA (Flip) | QE (k) | Re-ranking | Best val/mAP | Δ mAP vs Baseline | Best val/Rank-1 | Best Sim Gap |
|---|---|---:|---|---:|---:|---:|---:|
| retrieval_tta_qe | Yes | 3 | No | 0.6643 | +0.0015 | 0.9456 | 0.7363 |
| retrieval_tta_flip | Yes | No | No | 0.6643 | +0.0015 | 0.9456 | 0.7363 |
| retrieval_baseline | No | No | No | 0.6628 | 0.0000 | 0.9354 | 0.6987 |
| retrieval_qe | No | 3 | No | 0.6619 | -0.0009 | 0.9456 | 0.7318 |
| retrieval_qe_rerank | No | 5 | Yes | 0.6383 | -0.0245 | 0.9252 | 0.7878 |

For re-ranking, the parameters were swept over

- `k1 ∈ {20, 30, 50}`
- `k2 ∈ {6, 8, 10}`
- `λ ∈ {0.3, 0.5}`

The best reported re-ranking result in this study corresponds to **`k1 = 20`, `k2 = 6`, `λ = 0.3`**, i.e. a standard setting from the literature. The sweeps and corresponding evaluation tables are generated via `src/jaguar/retrieval/retrieval.py` and stored in the checkpoint directory under `retrieval_eval`.

<p align="center"><img src="../../results/round_2/kaggle_retrieval/Bildschirmfoto 2026-03-17 um 23.25.22.png" width="80%" /></p>
<p align="center"><em>Figure 1. Weights & Biases comparison of retrieval post-processing runs.</em></p>

<p align="center"><img src="../../results/round_2/kaggle_retrieval/Bildschirmfoto 2026-03-17 um 23.25.41.png" width="65%" /></p>
<p align="center"><em>Figure 2. Retrieval post-processing summary view for the same sweep.</em></p>

## Results and Analysis

### Test-Time Augmentation

The most consistent positive effect comes from **TTA with horizontal flipping**. Compared with the raw retrieval baseline, TTA improves **mAP** from **0.6628** to **0.6643** and **Rank-1** from **0.9354** to **0.9456**. The mAP gain is modest (**+0.0015**), but the Rank-1 gain is clearer, and the **Similarity Gap** also increases from **0.6987** to **0.7363**.

This is consistent with the broader findings from the augmentation study: flip invariance appears useful in this task even though jaguar flanks are biologically asymmetric. At inference time, the flipped view provides a complementary representation and can reduce sensitivity to pose and orientation.

### Query Expansion

**Query Expansion** shows a more mixed pattern. With **`k = 3`**, QE alone leaves **Rank-1** at **0.9456**, matching the TTA runs, and increases the **Similarity Gap** to **0.7318**. However, its **mAP** is slightly below the baseline (**0.6619** vs. **0.6628**).

This suggests that a very small neighborhood can help stabilize the top matches without necessarily improving the full ranking. In a low-data setting such as this one, larger neighborhoods are risky because the query can easily be pulled toward nearby but incorrect identities. The project sweeps indicated that performance deteriorates substantially as `k` increases, which is consistent with this interpretation.

An important practical observation is that **`retrieval_tta_qe` and `retrieval_tta_flip` reach identical reported metrics** in this sweep. In other words, once TTA is enabled, the additional QE step with `k = 3` does not provide a measurable validation gain beyond what TTA already achieves.

### k-reciprocal Re-ranking

The clearest negative result concerns **k-reciprocal re-ranking**. In this study, adding re-ranking reduces **mAP** from **0.6628** to **0.6383** and **Rank-1** from **0.9354** to **0.9252**, even though the **Similarity Gap** increases markedly to **0.7878**.

This is a useful reminder that a stronger global separation metric does not automatically imply a better retrieval order. Re-ranking modifies local neighborhood relations quite aggressively, and in this dataset the reciprocal-neighborhood structure appears too sparse and unstable to support that refinement. Because each identity has only a small number of samples, reciprocal sets can become noisy, so the Jaccard-distance term introduces ranking errors instead of resolving them.

Thus, while k-reciprocal re-ranking is well established in large-scale person Re-ID benchmarks, it is not beneficial in this small, fine-grained jaguar setting.

## Interpretation

### Which inference-time strategy is most useful?

The most reliable inference-time improvement comes from **TTA with horizontal flipping**. Its gain is not large in absolute mAP terms, but it is consistent and improves both **Rank-1** and **Similarity Gap**.

A conservative form of **QE** can be tolerated, but in this sweep it does not clearly improve over TTA alone. The results therefore do not justify treating QE as the primary driver of the final validation improvement.

By contrast, **re-ranking should be excluded** from the final pipeline in this regime. Although it increases the Similarity Gap, it worsens the retrieval metrics that matter most.

### Why does re-ranking fail here?

The likely reason is dataset scale and density. Re-ranking assumes that the local neighborhood structure is sufficiently reliable that reciprocal matches are informative. In this dataset, the identity neighborhoods are much smaller and more fragile. Under those conditions, reciprocal nearest-neighbor sets become noisy, and the Jaccard-distance correction can distort the ranking rather than refine it.

### Final pipeline choice

For the final Kaggle submission workflow, the project selected the **TTA (Flip) + QE (`k = 3`)** configuration. On the validation set, its gains over TTA alone are negligible, but the combination was retained because it consistently preserved the strongest validation profile among the tested conservative post-processing choices and was associated with a reported **~2–3% mAP improvement on the Kaggle test set** in the corresponding leaderboard experiments.

This should still be interpreted carefully: the validation gains are small, so the main validated contribution of the post-processing stack in this experiment is the **flip-based TTA** rather than a broad synergistic effect of all three modules.

## Notation / Procedure Note

Let `q ∈ R^d` be a query embedding and `{g_i}_{i=1}^N` the gallery embeddings, all L2-normalized. Similarity is computed via cosine similarity,

`sim(q, g_i) = q^T g_i`

and gallery images are ranked in descending similarity.

For **TTA**, if `q^(0)` is the embedding of the original image and `q^(1)` the embedding of its horizontal flip, the final query embedding is

`q_hat = (q^(0) + q^(1)) / ||q^(0) + q^(1)||_2`

For **Query Expansion**, let `N_k(q)` denote the top-k nearest neighbors of `q`. The expanded query is

`q_tilde = ( q + sum_{g_j in N_k(q)} g_j ) / (k+1)`

followed by L2 normalization.

For **k-reciprocal re-ranking**, let `R_{k1}(q)` be the reciprocal-neighbor set of `q`. The final distance combines the original distance with a Jaccard-distance term:

`d_final = (1 - λ) d_orig + λ d_J`

where `k1`, `k2`, and `λ` control the size of the reciprocal neighborhood, the local expansion within re-ranking, and the weighting between original and Jaccard distances.

## Main Findings

- **TTA with horizontal flipping** provides the most consistent inference-time improvement.
- The validation gain in **mAP** is small (**+0.0015**), but **Rank-1** improves more clearly (**0.9354 → 0.9456**).
- **QE with `k = 3`** is approximately neutral in mAP and does not outperform TTA when both are combined in this sweep.
- **k-reciprocal re-ranking** substantially degrades retrieval performance in this low-data setting, despite increasing the Similarity Gap.
- The results suggest that conservative local refinement can be helpful, but aggressive neighborhood-based re-ordering is too unstable when each identity has only a few samples.

## Limitation

This experiment is based on a small validation set and one fixed backbone checkpoint. The post-processing conclusions are therefore specific to this embedding space and data regime. In particular, the negative re-ranking result should not be interpreted as a general rejection of k-reciprocal re-ranking; rather, it indicates that the method is not well matched to this dataset scale and neighborhood density.

## Conclusion

Inference-time post-processing is useful in the Jaguar Re-ID pipeline, but its benefits are selective.

The main positive result is that **flip-based TTA** improves retrieval consistently and therefore deserves a place in the final pipeline. **QE** can be used conservatively, but its contribution appears limited in the validation sweep reported here. **k-reciprocal re-ranking**, in contrast, is too aggressive for this small and fine-grained dataset and should be excluded.

Overall, the experiment shows that retrieval refinement is worthwhile, but only when the post-processing method respects the density and reliability of the local embedding neighborhoods.
