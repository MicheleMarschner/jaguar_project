This experiment investigates the most effective loss function and head architecture for Jaguar Re-ID. While the backbone provides the features, the objective function defines how these features are organized in the embedding space. We aim to find the best balance between retrieval precision (mAP) and the geometric quality of the learned clusters. Hence, the **Research Question** we aim to answer is: *How do angular margin-based classification losses compare against stabilized metric learning objectives in shaping a discriminative embedding space for fine-grained animal Re-ID in low-data regimes?*

TThe model architecture used is the `JaguarIDModel` (detailed in `src/jaguar/models/jaguarid_models.py`). All experiments use the EVA-02 backbone, with the optimized augmentation suite, hyperparameters, training scheme, and evaluation protocol that achieve the highest performance across our experiments on both validation mAP and the public Kaggle test set. For more details, please refer to `E00_kaggle-leaderboard_backbone`.

As noted in stability and mining studies in person and vehicle re-identification, Triplet Loss can be difficult to converge. To ensure a fair and competitive comparison, we modify its configuration as follows:
- Training and validation `batch_size`: increased to 128, ensuring the model observes a high diversity of identities (up to 31) in each step, providing sufficient hard negatives.
- The "Bag of Tricks" (BoT) head: following the OpenAnimals and BoT frameworks, our Triplet implementation uses an `EmbeddingHead` (neck) with `BatchNorm`, followed by a fully connected classifier. The Triplet loss is computed on embeddings, while Cross-Entropy (or Focal Loss for class imbalance) is computed on logits. This hybrid formulation stabilizes training by providing global class separation while refining local distances.

The loss functions and corresponding heads evaluated are:
- Softmax (Baseline): standard Cross-Entropy, focusing on inter-class separation without enforcing intra-class compactness  
- ArcFace ($\theta + m$): additive angular margin enforcing separation on a hypersphere  
- CosFace ($\cos(\theta) - m$): margin applied directly to cosine similarity  
- SphereFace ($\cos(m \cdot \theta)$): multiplicative angular margin, known to be difficult to optimize  
- Triplet + Focal (Hard Mining): metric learning with class imbalance handling  
- Triplet + CE (Hard Mining): metric learning with standard classification regularization  

The following table reports peak validation performance:

| Loss Configuration        | Best val/mAP | Best val/Rank-1 | Best Sim Gap | Convergence Epoch |
|--------------------------|--------------|------------------|--------------|-------------------|
| Triplet + Focal (Hard)   | 0.6705       | 0.9456           | 0.7818       | 20                |
| Triplet + CE (Hard)      | 0.6598       | 0.9456           | 0.6967       | 18                |
| ArcFace                  | 0.6482       | 0.9524           | 0.7257       | 25                |
| CosFace                  | 0.6356       | 0.9422           | 0.6860       | 22                |
| Softmax                  | 0.5304       | 0.9150           | 0.4352       | 15                |
| SphereFace               | 0.3956       | 0.8844           | 0.3286       | 30+ (Slow)        |

The results confirm that Triplet Loss with Hard Mining, especially when paired with Focal Loss, outperforms ArcFace in mAP. In our dataset (1.3k training / 370 validation images), there are very few samples per identity once most of the duplicates are removed. Classification losses such as ArcFace rely on learning stable class prototypes in the embedding space. With limited data, these prototypes are harder to estimate reliably. In contrast, Triplet Loss focuses on image-to-image relationships, making it more effective at capturing fine-grained local differences between jaguar patterns.

<insert here Wights & Biases dashboard>

Looking at the val/loss curves, Triplet-based runs are noisier than Softmax. This is expected, as mining dynamically selects hard samples as the embedding space evolves. SphereFace shows severe convergence issues, likely due to its aggressive multiplicative margin, which is too strong for subtle jaguars differences. While ArcFace achieves the highest Rank-1 accuracy (0.9524), it struggles to retrieve all relevant instances of the same identity across the dataset. In contrast, Triplet + Focal (Hard) achieves the highest mAP (0.6705) and Similarity Gap (0.7818), indicating superior global ranking quality. This improved embedding structure enabled the model to reach 90.08% mAP on the public Kaggle test set.

Overall, these results suggest that for fine-grained wildlife Re-ID in low-data regimes, carefully stabilized metric learning objectives outperform purely classification-based angular margin losses. A more detailed analysis of mining strategies is provided in `E00_kaggle-leaderboard_mining.md`, where we isolate the effect of sample selection within the Triplet framework.
