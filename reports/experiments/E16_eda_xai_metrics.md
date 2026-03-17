# E16 Interpretability sanity / faithfulness experiment (Data - Round 1)
==========================================

## Sub-experiment 3 — Explanation Evaluation for Pairwise-Similarity: Sanity, Faithfulness, and Complexity
---------------------------------------------------

### Research Question
Are the pairwise similarity explanations tied to the learned model and causally related to the similarity score?

### Intervention
Explanation quality was evaluated with randomization-based sanity checks, masking-based faithfulness tests, and complexity comparisons.

### Method / Procedure
For each pair type and explainer, saliency maps from the trained model were compared to maps from a randomized model. Faithfulness was assessed by masking top-salient regions and comparing the resulting deletion AUC against random masking. Complexity was used to characterize how concentrated or diffuse the explanations were.

### Evaluation
Main outputs to show:
- sanity score by explainer and pair type
- deletion AUC under random vs top-salient masking
- faithfulness gap distribution
- complexity by explainer and pair type

### Results and Analysis
The explanation evaluation refines the qualitative interpretation from the previous sections and shows clear differences between the two explainers.

**Sanity.**  
IG remains essentially at zero across all pair types, with very small spread. In this setup, IG therefore shows almost no change under model randomization and does not provide convincing sanity evidence. GradCAM varies more and produces broader distributions, but medians still remain close to zero. Thus, GradCAM shows a somewhat stronger sanity signal than IG, but the effect is still weak overall.

**Faithfulness.**  
The faithfulness results are mixed and depend strongly on pair type. For **easy positives**, both explainers show a small positive faithfulness gap, meaning salient masking hurts similarity slightly more than random masking. This is the clearest faithfulness signal and fits the qualitative examples, where matching evidence is strong and redundant. For **hard negatives**, the pattern reverses: the gap becomes negative for both explainers, especially for GradCAM, indicating that top-salient masking does not outperform random masking in the error-prone regime. **Hard positives** lie between these extremes: IG stays slightly positive, whereas GradCAM becomes slightly negative.

This is also visible in the deletion AUC summaries. For easy positives, top-k masking yields lower AUC than random masking for both explainers, consistent with more meaningful attribution. For hard negatives and hard positives, however, top-k masking is equal to or worse than random masking, especially for GradCAM.

**Complexity.**  
IG is stable across pair types, with mean complexity values around **0.61** throughout. GradCAM behaves very differently. It is simpler than IG on **easy positives** (**0.525** vs **0.612**), but substantially more complex on **hard negatives** (**0.827**) and **hard positives** (**0.724**). This matches the qualitative impression that GradCAM becomes broader and more diffuse in difficult cases.

Overall, the quantitative results support a cautious interpretation of the qualitative maps. The explanations are visually plausible, but they are not equally reliable across settings. Their strongest support appears in easy positives, whereas explanation quality is weakest in hard negatives, i.e. exactly where interpretability would matter most for understanding retrieval errors.

### Key Result / Takeaway
The explanation evaluation distinguishes the two explainers, but neither is uniformly strong across all checks. IG is more stable but shows almost no sanity effect. GradCAM is more responsive to randomization, yet becomes substantially more diffuse in difficult cases. Faithfulness is most convincing for easy positives and weakest for hard negatives. Thus, pairwise explanations are most trustworthy in easy matching settings and less reliable for the difficult cases that matter most for retrieval failures.

**[Insert Figure: sanity by explainer and pair type]**  
**[Insert Figure: faithfulness gap by explainer and pair type]**  
**[Insert Figure: random vs top-salient masking barplots]**  
**[Insert Figure: complexity by explainer and pair type]**



**[Insert Table: summary of sanity / faithfulness / complexity metrics]**
**[Insert Figure: sanity boxplots]**  
**[Insert Figure: faithfulness gap boxplots]**  
**[Insert Figure: salient vs random masking barplots]**  
**[Insert Figure: complexity boxplots]**




# E16 (Q2a) Explainer Comparison for Class Attributions (GradCAM vs IG)
*(Subexperiment of # E14 (Q2) Class Attributions)*

**Experiment Group:** Interpretability analyses

## Main Research Question
---------------------------------------------------
How do **GradCAM** and **Integrated Gradients (IG)** compare for EVA-02 class attributions, quantitatively and qualitatively, and does explainer choice materially affect the interpretation?

## Relation to the Main Class-Attribution Experiment
---------------------------------------------------
This document is a methodological follow-up to **[E# E14 (Q2) Class Attributions](E14_eda_xai_class_attribution.md)**. It revisits the same quantitative summaries and qualitative overlays, but with a narrower goal: not to ask what the explanations suggest about the model, but rather **which explainer is more useful and how much the overall interpretation depends on explainer choice**.

## Setup
---------------------------------------------------
We compare **GradCAM** and **Integrated Gradients (IG)** on the same class-attribution outputs used in the main experiment. The comparison is based on:

- **sanity** under randomization,
- **faithfulness gap** and masking effects,
- **complexity**,
- and **qualitative plausibility** of the resulting overlays.

The figures shown below are re-used from the main class-attribution analysis and are included again here for clarity. Their interpretation, however, is now explicitly methodological.

## Main Findings
---------------------------------------------------
The broad interpretation is stable across explainer choice: both methods support the same overall conclusion already established in **[# E14 (Q2) Class Attributions](E14_eda_xai_class_attribution.md)**, namely that sanity is acceptable but masking-based faithfulness is weak.

At the same time, the two explainers differ materially in usability:

- **GradCAM** yields slightly stronger quantitative results on the masking-based metrics.
- **IG** is consistently more diffuse, both numerically and visually.
- As a result, **GradCAM is the more useful explainer in this setting**, even though neither method provides strong faithfulness evidence.

## Quantitative Comparison
---------------------------------------------------
The aggregate metric plots from the main experiment are reproduced below because they already contain the key evidence for the explainer comparison.

<p align="center"><img src="../../results/round_1/eda_xai_class_attribution/eva02_val_n332_all_groups/class_xai_analysis/xai_class_metric_means.png" width="72%" /></p>
<p align="center"><em>Figure 1. Mean class-attribution metrics by group and explainer, reproduced from the main class-attribution experiment.</em></p>

<p align="center"><img src="../../results/round_1/eda_xai_class_attribution/eva02_val_n332_all_groups/class_xai_analysis/xai_class_metric_boxplots.png" width="92%" /></p>
<p align="center"><em>Figure 2. Distribution of sanity, faithfulness gap, and complexity by group and explainer, reproduced from the main class-attribution experiment.</em></p>

Figure 1 already shows the main quantitative pattern. The two explainers are very similar on **sanity**, but differ more clearly on **faithfulness gap** and **complexity**. GradCAM has the larger mean faithfulness gap in the two main groups, whereas IG has higher complexity.

Figure 2 shows that these differences are not just artifacts of the mean. The sanity distributions overlap substantially, while the faithfulness-gap and complexity distributions are more clearly separated.

The masking comparison from the main experiment is also directly relevant here, because it shows whether one explainer identifies more decision-relevant regions than the other.

<p align="center"><img src="../../results/round_1/eda_xai_class_attribution/eva02_val_n332_all_groups/class_xai_analysis/xai_class_faithfulness_barplot.png" width="50%" /></p>
<p align="center"><em>Figure 3. Mean target-score drop under top-k salient masking versus random masking, reproduced from the main class-attribution experiment.</em></p>

<p align="center"><img src="../../results/round_1/eda_xai_class_attribution/eva02_val_n332_all_groups/class_xai_analysis/xai_class_faithfulness_gap_distribution.png" width="52%" /></p>
<p align="center"><em>Figure 4. Distribution of faithfulness gaps by group and explainer, reproduced from the main class-attribution experiment.</em></p>

Figures 3 and 4 show that **GradCAM performs somewhat better than IG on the same weak-faithfulness landscape**. In `all` and `orig_rank1_correct`, GradCAM produces a slightly larger positive gap than IG, but the absolute magnitudes remain small. This means the correct interpretation is not that GradCAM is strongly faithful, but rather that it is **relatively better under otherwise weak masking-based evidence**.

**Table 1. Mann–Whitney comparison of GradCAM and IG on the main class-attribution metrics.**

| group | metric | mean GradCAM | mean IG | p-value | significance |
|---|---|---:|---:|---:|---|
| all | sanity | -0.0085 | 0.0001 | 0.4582 | ns |
| orig_rank1_correct | sanity | -0.0293 | -0.0001 | 0.2397 | ns |
| all | faith_gap | 0.0019 | 0.0005 | 0.0007 | *** |
| orig_rank1_correct | faith_gap | 0.0019 | 0.0005 | 0.0009 | *** |
| all | faith_topk | 0.0006 | -0.0008 | 0.0020 | ** |
| orig_rank1_correct | faith_topk | 0.0007 | -0.0007 | 0.0019 | ** |
| all | complexity | 0.5785 | 0.6044 | 0.0017 | ** |
| orig_rank1_correct | complexity | 0.5790 | 0.6041 | 0.0022 | ** |

Table 1 clarifies the same picture statistically. The sanity difference is **not significant**, so sanity does not clearly favor either method. By contrast, GradCAM has a significantly larger **faithfulness gap** and significantly less **complexity** in the two main groups. These differences are meaningful for explainer comparison, even if they do not change the overall conclusion that faithfulness remains weak in absolute terms.

## Qualitative Comparison
---------------------------------------------------
The qualitative overlays from the main experiment are reproduced below because the difference between the two explainers is even clearer visually than numerically.

<p align="center"><img src="../../results/round_1/eda_xai_class_attribution/eva02_val_n332_all_groups/class_xai_analysis/qualitative/qualitative_grid__all.png" width="86%" /></p>
<p align="center"><em>Figure 5. Qualitative class-attribution overlays for the `all` group, reproduced from the main class-attribution experiment.</em></p>

<p align="center"><img src="../../results/round_1/eda_xai_class_attribution/eva02_val_n332_all_groups/class_xai_analysis/qualitative/qualitative_grid__orig_rank1_wrong.png" width="86%" /></p>
<p align="center"><em>Figure 6. Qualitative class-attribution overlays for the `orig_rank1_wrong` group, reproduced from the main class-attribution experiment.</em></p>

As seen in Figure 5, **GradCAM** more often produces a coarse but semantically plausible jaguar-centered emphasis, especially on the **head, torso, flank, and coat pattern**. **IG**, by contrast, is typically much more **diffuse, texture-like, and globally noisy**. The jaguar silhouette is sometimes still visible, but the map is less selective and harder to interpret as a compact class-attribution explanation.

Figure 6 points in the same direction for the hard-case subset. Even there, GradCAM remains more visually interpretable than IG, although neither explainer isolates the class evidence especially cleanly.

## Overall Answer
---------------------------------------------------
The overall interpretation does **not** depend strongly on the explainer: both methods support the same broad conclusion from the main class-attribution experiment, namely that sanity is acceptable but masking-based faithfulness is weak.

However, the comparison does matter in practice. **GradCAM is preferable to IG in this setting**, because it yields slightly stronger quantitative results and substantially more interpretable qualitative overlays.

## Concise Conclusion
---------------------------------------------------
This methodological follow-up shows that the class-attribution conclusions are broadly stable across explainer choice, but the **quality of interpretation is not**. GradCAM is the more useful explainer for EVA-02 class attributions: it performs somewhat better on the masking-based metrics and produces more semantically plausible overlays than IG. That said, neither explainer provides strong faithfulness evidence in absolute terms.