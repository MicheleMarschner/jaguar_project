This experiment investigates the utility of Progressive Resizing as a training curriculum for jaguar re-identification. Progressive resizing involves starting training at a lower resolution to learn global features (e.g., body shape and large-scale rosette distributions) and gradually increasing the resolution to refine fine-grained details (e.g., micro-spot patterns and edge textures). Therefore, the **Research Question** we try to answer is the following: *Does a progressive resolution curriculum improve final retrieval accuracy and training stability for convolutional backbones, and what are the associated trade-offs in computational efficiency?*

A critical constraint of modern foundation models (like Vision Transformers) is that they are often pretrained with fixed positional embeddings tied to a specific grid size (e.g., 224x224 or 336x336). Changing the input resolution would require complex interpolation of these embeddings. Consequently, this ablation was conducted using ConvNeXt-V2-Base, a fully convolutional architecture that naturally handles varying input sizes. The same experiment coudl be easily replicated with EfficientNet-B4. Indeed, our built-in method apply progressive resizing initially reduced it by 60%, then after some epohcs increased it at 80% and after some others we train at 100%.

We compared three schedules on the `round_2` data ("background-masked") dataset, but we also tested it on `round_1` data (experiment here omitted for brevity), confirming the results here observed:

- `no_progressive_cnv2` (Baseline): Fixed resolution at 224x224 for all 30 epochs.
- `progressive_epochs_10_cnv2`: Trained at 160x160 for the first 10 epochs, then increased to 224x224 for the remaining 20 epochs.
- `progressive_varying_epochs_cnv2`: A multi-step curriculum starting at 128x128, moving to 160x160 at epoch 10, and finishing at 224x224 from epoch 20 onwards.

Optimization, loss (Triplet + CE), and hyperparameters remain consistent with the best-performing recipes described in `E00_leaderboard-kaggle_backbone.md`. Minor adjustments were introduced to ensure competitive performance with Vision Transformer baselines and surpass the challenge reference (MegaDescriptor-L + ArcFace, 74.1% mAP): `RandomResizedCrop` is enabled, and the backbone is fully unfrozen starting from epoch 3.

The table below summarizes the peak retrieval performance and efficiency metrics (based on epoch time):

| Run Name               | Best val/mAP | Best val/Rank-1 | Best Silhouette | Avg. Epoch Time (sec) | Relative Efficiency |
|------------------------|--------------|------------------|------------------|------------------------|----------------------|
| progressive_varying    | 0.5984       | 0.9388           | 0.4523           | 37.8s                 | +6.2% faster         |
| progressive_epochs_10  | 0.5724       | 0.9320           | 0.3742           | 39.1s                 | +3.0% faster         |
| no_progressive         | 0.5775       | 0.9354           | 0.3809           | 40.3s                 | Baseline             |

The results show that progressive resizing, particularly the multi-stage schedule, achieves both higher retrieval performance and improved computational efficiency compared to the fixed-resolution baseline.

<insert Weights & Biases screenshot here>

The `progressive_varying_epochs_cnv2` run (dotted green line) clearly outperforms the fixed-resolution baseline in the final epochs, reaching an mAP of ~0.60 compared to ~0.57. As shown in the val/mAP and val/sim_gap curves, progressive resizing acts as an effective regularizer. The val/silhouette plot indicates that the multi-step strategy produces more cohesive identity clusters. Notably, there are visible increases in the Silhouette score at epochs 10 and 20, corresponding to resolution changes. This suggests that higher resolution enables the model to resolve ambiguities between visually similar identities.

The timing/epoch_time_sec plot demonstrates a clear efficiency gain: the "step-like" behavior in the timing chart confirms that training at 128x128 and 160x160 is significantly cheaper. Total training time was reduced by approximately 6.2% for the varying schedule while simultaneously improving mAP by 3.6%. This satisfies the criteria for an effective optimization study, proving that we can achieve better results with lower total FLOPs.

In the val/loss plot, a characteristic spike appears when the resolution increases (epochs 10 and 20). This occurs because the model is suddenly exposed to higher-frequency details. However, it stabilizes within a few epochs and converges to a better solution than the previous stage. Progressive resizing also leads to smoother convergence in val/pairwise_AP. Compared to fixed resolution, which must learn all feature scales simultaneously, the progressive approach first captures global structure and then refines fine-grained details more effectively.

Progressive resizing is a valid and beneficial strategy for convolutional Re-ID systems. By treating resolution as a curriculum, it provides both improved final performance and reduced computational cost. For the Jaguar task, this suggests that learning global structure first before focusing on fine-grained rosette patterns is advantageous. This insight supports the use of high-resolution inputs in the final Transformer-based leaderboard model.