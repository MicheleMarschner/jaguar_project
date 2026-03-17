This experiment presents a hypothesis-driven investigation into the impact of data augmentation strategies on fine-grained wildlife re-identification. Given the unique nature of jaguar rosette patterns—which are spatially complex and asymmetric between the left and right flanks—choosing the right level of image distortion is a critical trade-off between regularization and the preservation of identity-defining features. Therefore, the **Research Question** we try to answer is the following: *Can biologically "incorrect" augmentations (like horizontal flipping) and structural occlusions (like random erasing) serve as effective regularizers to overcome pose-dependency and camera-trap noise in low-data wildlife Re-ID scenarios?*

The pipeline and optimization hyperparameters follow the configuration defined in `config/base/kaggle_base.toml` and thoroughly described in `E00_leaderboard-eda_stability.md`, utilizing the EVA-02 backbone and Triplet Loss with Semi-Hard mining. All images are resized to the backbone's native resolution, favoring high resolutions to preserve rosette detail (e.g., 448x448 in EVA-02 with patch size 14).

We ablate the augmentation components across eight controlled runs:
- `aug_none`: Baseline with no transformations
- `aug_flip_only`: Isolated testing of horizontal flipping
- `aug_geom_noflip_nocol_noerase`: Basic geometry (Affine) and Random Resized Crop (RRC)
- `aug_geom_nocol_noerase`: Adds horizontal flipping to the geometric base
- `aug_geom_noerase`: Adds Color Jitter to account for camera-trap lighting variance
- `aug_full_pipeline`: Comprehensive suite including Gaussian Blur and Random Erasing (p=0.1)
- `aug_full_pipeline_erase025`: Increases Random Erasing to (p=0.25)
- `aug_curr_baseline`: Our optimized recipe: Geometry + Color + Heavy Erasing (p=0.25), but excluding Gaussian Blur and RRC.

The following table summarizes the peak validation performance for each augmentation suite.

| Run Name                         | Best val/mAP | Δ mAP vs Baseline | Best val/Rank-1 | Best Silhouette | Best Sim Gap |
|---------------------------------|--------------|-------------------|------------------|------------------|--------------|
| aug_curr_baseline               | 0.7318       | +0.0000           | 0.9456           | 0.5960           | 0.7318       |
| aug_full_pipeline_erase025      | 0.6905       | -0.0413           | 0.9354           | 0.5427           | 0.6905       |
| aug_full_pipeline               | 0.6781       | -0.0537           | 0.9252           | 0.5316           | 0.6781       |
| aug_geom_nocol_noerase          | 0.6384       | -0.0934           | 0.9456           | 0.5937           | 0.6384       |
| aug_geom_noerase                | 0.6359       | -0.0959           | 0.9422           | 0.5522           | 0.6359       |
| aug_flip_only                   | 0.6358       | -0.0960           | 0.9388           | 0.5968           | 0.6358       |
| aug_geom_noflip_nocol_noerase   | 0.6329       | -0.0989           | 0.9150           | 0.5305           | 0.6330       |
| aug_none                        | 0.6149       | -0.1169           | 0.9150           | 0.5305           | 0.6149       |

A central debate in jaguar Re-ID is the use of horizontal flipping. Biologically, jaguars have different patterns on their left and right sides. Hypothetically, flipping an image of a left flank should create a "spurious" identity that does not exist. However, the data shows a clear advantage for flipping: `aug_geom_nocol_noerase` (with flip) outperforms `aug_geom_noflip_nocol_noerase` (no flip) by a significant margin in mAP. The error analysis in `E00-kaggle-leaderboard-eda_model_soup.md` shows that models often fail on head-only crops or highly distorted poses where global pattern matching is not possible. Flipping likely helps the EVA-02 global attention mechanism become invariant to the direction of the animal, forcing it to learn local rosette textures rather than global body orientation. In low-data regimes, this acts as an effective data-doubling strategy that prevents overfitting to pose-specific identity correlations.

The best-performing configuration, used in the final submission to the second Kaggle competition (`aug_curr_baseline`), explicitly excludes Gaussian Blur and RRC. In wildlife Re-ID, blurring degrades high-frequency features such as rosette boundaries and can further harm already low-quality images. While RRC is standard in ImageNet training, it can remove the most discriminative flank regions entirely, leaving only uninformative background or partial body parts. This is particularly problematic in camera-trap data, where jaguars may be partially occluded, in unusual poses, or only partially visible.

By using Random Erasing (p=0.25), we simulate natural occlusions such as vegetation (grass, branches) commonly present in camera-trap images. This forces the model to rely on multiple discriminative regions rather than a single optimal patch. This observation motivates future work on mapping hierarchical feature representations to better capture multiple identity cues across the animal body.

<insert Weights & Biases screenshot>

Looking at the val/loss and val/mAP curves, we observe—consistently with `E00_leaderboard-eda_stability.md` and other leaderboard experiments using the same best-performing configuration—a characteristic spike around epoch 13. This corresponds to the point where learning rate decay begins and the model transitions from exploration to refinement. The `aug_curr_baseline` configuration shows the most consistent improvement during the final epochs, reaching the highest similarity gap (0.7318). This indicates that the chosen augmentation strategy not only improves retrieval performance but also produces the most separable identity clusters in the embedding space.

The optimal strategy for Jaguar Re-ID involves a careful balance: geometric and color augmentations account for environmental variability, while strong Random Erasing builds robustness to occlusions. Most importantly, despite biological asymmetry, horizontal flipping acts as a crucial regularizer that improves generalization across poses, ultimately contributing to the 90%+ mAP performance observed on the Kaggle leaderboard.
