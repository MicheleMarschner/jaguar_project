# Progressive Resizing Ablation (Data - Round 2)

## Main Research Question

Does a progressive resolution curriculum improve final retrieval accuracy and training stability for convolutional backbones, and what are the associated trade-offs in computational efficiency?

## Setup / Motivation

This experiment investigates the utility of **progressive resizing** as a training curriculum for jaguar re-identification. The basic idea is to start training at a lower input resolution, where the model first learns coarse global structure, and then gradually increase the resolution so that finer identity cues can be incorporated later. In the jaguar setting, this corresponds to a possible progression from broad body shape and large-scale rosette layout toward finer local spot patterns and edge details.

A practical constraint is that many modern Vision Transformers are pretrained with fixed positional embeddings tied to a particular patch grid. Changing the input resolution can therefore require additional handling of positional embeddings. For this reason, the ablation was conducted with **ConvNeXt-V2**, a fully convolutional backbone that naturally supports variable input sizes. The same idea could in principle also be applied to other convolutional backbones such as EfficientNet-B4.

The experiment is run on the **Round 2 background-masked dataset**. The text and plots below focus on the Round 2 results; additional Round 1 tests were also run, but are omitted here for brevity.

## Intervention

We compare three main training schedules:

- **`no_progressive_cnv2`**: fixed resolution at **224×224** for all **30 epochs**
- **`progressive_epochs_10_cnv2`**: **160×160** for the first **10 epochs**, then **224×224** for the remaining **20 epochs**
- **`progressive_varying_epochs_cnv2`**: a multi-step curriculum with **128×128** initially, **160×160** from **epoch 10**, and **224×224** from **epoch 20**

These schedules correspond to the built-in progressive-resizing mechanism in the training pipeline, which begins at a reduced resolution and then increases it in stages until full resolution is reached.

All other optimization, loss, and training settings are kept aligned with the ConvNeXt-V2 leaderboard recipe. In particular, the experiment uses **Triplet + CE**, enables **RandomResizedCrop**, and fully unfreezes the backbone from **epoch 3** onward.

## Main Findings

Three main results emerge.

First, **progressive resizing is competitive and can be beneficial**, but the benefit depends on the schedule. The strongest result comes from the **multi-stage schedule** (`progressive_varying_epochs_cnv2`), which reaches the highest validation mAP and the highest silhouette score among the three main schedules.

Second, the gains are **modest rather than dramatic**. Compared with the fixed-resolution baseline, the best progressive schedule improves validation mAP from **0.5775** to **0.5984** while also reducing average epoch time from **40.3 s** to **37.8 s**. This is a useful improvement, but it should be interpreted as an incremental training optimization rather than a fundamental change in model behavior.

Third, the effect appears to be consistent with the intended curriculum interpretation. Lower-resolution early training is cheaper and may encourage the model to first organize coarse structure before higher-resolution details are introduced. The resolution-change points are visible in the training curves and coincide with short adaptation phases, after which the model stabilizes again.

## Main Results

**Table 1. Peak retrieval performance and average epoch time for the three main progressive-resizing schedules.**

| Run Name | Best val/mAP | Best val/Rank-1 | Best Silhouette | Avg. Epoch Time (sec) | Relative Efficiency |
|---|---:|---:|---:|---:|---:|
| progressive_varying | 0.5984 | 0.9388 | 0.4523 | 37.8 | +6.2% faster |
| progressive_epochs_10 | 0.5724 | 0.9320 | 0.3742 | 39.1 | +3.0% faster |
| no_progressive | 0.5775 | 0.9354 | 0.3809 | 40.3 | Baseline |

The table shows that the **multi-stage progressive schedule** achieves the best overall trade-off in this comparison: it is both the strongest on validation mAP and the fastest on average per epoch. By contrast, the simpler two-stage schedule (`progressive_epochs_10_cnv2`) is slightly faster than the baseline, but underperforms it on mAP and Rank-1.

## Training-Curve Analysis

The Weights & Biases curves support the same interpretation.

<p align="center"><img src="../../results/round_2/kaggle_progressive_resizing/wandb_dashboard_progressive_resizing.png" width="88%" /></p>
<p align="center"><em>Figure 1. Validation curves for the progressive-resizing study. The three main schedules of interest are the progressive and non-progressive ConvNeXt-V2 runs; the dashboard also contains additional fixed-resolution control runs that are not part of the main comparison in Table 1.</em></p>

<p align="center"><img src="../../results/round_2/kaggle_progressive_resizing/wandb_train_loss_progressive_resizing.png" width="70%" /></p>
<p align="center"><em>Figure 2. Training-loss trajectories for the progressive-resizing study.</em></p>

### Aggregate performance behavior

In the **val/mAP** and **val/pairwise_AP** curves, `progressive_varying_epochs_cnv2` finishes above the fixed-resolution baseline, while `progressive_epochs_10_cnv2` remains slightly below it. Thus, progressive resizing is not automatically beneficial; the more gradual curriculum appears to matter.

The **val/rank1** curves show a similar picture. All three schedules eventually converge into a relatively narrow range, but the best final value again comes from the more gradual progressive schedule.

### Silhouette and similarity-gap behavior

The **val/silhouette** curves suggest that the multi-stage progressive curriculum produces the most coherent identity clusters among the three main schedules. Its final silhouette score (**0.4523**) is clearly above both the fixed baseline (**0.3809**) and the two-stage schedule (**0.3742**).

The **val/sim_gap** curves are directionally consistent with this reading: the progressive schedules, especially the varying schedule, end with stronger separation than the fixed baseline.

Because the silhouette metric is shown at a lower temporal resolution than the main retrieval curves, it is most useful here as a final summary signal rather than as evidence for fine-grained epoch-by-epoch effects.

### Loss and adaptation phases

The **val/loss** plot shows brief disruptions around the resolution-change points, which is expected: when the input resolution increases, the model is suddenly exposed to higher-frequency information that changes the effective difficulty of the task. Importantly, these disruptions are temporary and the curves stabilize again within a few epochs.

The **train/loss** plot is consistent with the same interpretation. The progressive schedules do not show pathological instability; rather, they appear to undergo short adjustment phases before continuing to improve.

## Interpretation

### Does progressive resizing help?

**Yes, but only in the stronger schedule tested here.** The multi-stage curriculum (`progressive_varying_epochs_cnv2`) improves both validation accuracy and average epoch-time efficiency relative to the fixed-resolution baseline. The simpler two-stage schedule does not.

This means the experiment supports the usefulness of progressive resizing in principle, but not every curriculum is equally effective.

### Why might the varying schedule help?

A plausible interpretation is that the curriculum separates two learning phases:

- at low resolution, the model first learns broad global structure at lower computational cost
- at later stages, higher resolution allows refinement of finer identity cues

In this experiment, that curriculum seems to be more effective when introduced gradually over multiple stages rather than with a single early jump.

### What are the efficiency trade-offs?

The efficiency gain is real but moderate. The best schedule reduces average epoch time by about **6.2%** relative to the fixed baseline while also improving validation mAP. This is a favorable trade-off, but it does not change the broader conclusion that ConvNeXt-V2 remains below the strongest Transformer-based leaderboard models overall.

## Limitation

This ablation was performed on **ConvNeXt-V2** rather than on the final Transformer-based leaderboard backbone. The conclusions therefore apply most directly to convolutional backbones that can natively accommodate varying input sizes. In addition, the study compares only a small set of progressive schedules, so it does not establish a fully optimized curriculum.

The Weights & Biases dashboard also includes several additional fixed-resolution control runs with different unfreezing choices. These are useful for qualitative context, but the main comparison in this report remains the three schedules summarized in Table 1.

## Conclusion

Progressive resizing is a **valid and potentially useful curriculum strategy** for convolutional jaguar Re-ID models. In this study, the strongest variant is the **multi-stage schedule** that moves from **128×128** to **160×160** and finally to **224×224**. It achieves the best validation mAP (**0.5984**), the best silhouette score (**0.4523**), and the lowest average epoch time (**37.8 s**) among the three main schedules.

The main conclusion is therefore not that progressive resizing always helps, but that a **carefully staged resolution curriculum** can improve both performance and efficiency for convolutional backbones. This supports the broader intuition that, in jaguar Re-ID, learning coarse structure first and refining high-frequency details later can be beneficial.
