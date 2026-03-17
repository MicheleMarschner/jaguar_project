# E0X (Q2) Class Attributions (Data - Round 1)
==========================================

**Experiment Group:** Interpretability analyses

## Main Research Question
---------------------------------------------------
Which regions drive the model’s identity-class predictions, and are these explanations meaningful?

## Sub-Questions
---------------------------------------------------
- Do the explanations satisfy the **sanity check**, i.e. do they change under randomization?
- Do **top-k salient masks** hurt the model more than same-sized **random masks**?
- Are the findings consistent across **all samples**, **originally correct rank-1 cases**, and **originally wrong rank-1 cases**?
- Do **GradCAM** and **IG** tell a consistent story qualitatively?


## Setup
---------------------------------------------------
We generated **class-target explanations** for the EVA-02 model using **GradCAM** and **Integrated Gradients (IG)** on three groups: `all`, `orig_rank1_correct`, and `orig_rank1_wrong`. We then evaluated the saved explanations with:

- a **sanity check** based on randomization,
- a **faithfulness test** comparing top-k salient masking against a same-sized random-mask control,
- and a **complexity** metric.

Qualitative overlays were additionally inspected to assess whether the highlighted regions plausibly align with jaguar identity cues. The qualitative analysis was generated for the groups `all`, `orig_rank1_correct`, and `orig_rank1_wrong`. :contentReference[oaicite:0]{index=0}


## Main Findings
---------------------------------------------------

### 1. Sanity: both explainers show near-zero randomization agreement, which is encouraging
Across the `all` group, the mean sanity score is **-0.008** for GradCAM and **0.000** for IG; for `orig_rank1_correct`, it is **-0.029** for GradCAM and **-0.0001** for IG. These values are all close to zero, which is consistent with the expected behavior that explanations should substantially change under randomization rather than remain stable.

At the same time, the difference between GradCAM and IG is **not statistically significant** for sanity in the `all` group (**p = 0.458**) or the `orig_rank1_correct` group (**p = 0.240**). This suggests that, on the sanity criterion alone, neither method clearly dominates.

**Interpretation:** the explanations are **not obviously failing the sanity check**. This is the strongest positive result of the experiment.

**[Insert Table: `xai_class_main_table.csv`]**  
**[Insert Figure: `xai_class_metric_means.png`]**  
**[Insert Figure: `xai_class_metric_boxplots.png`]**


### 2. Faithfulness: the results do **not** support strong faithfulness
The faithfulness evidence is weak. In the `all` group:

- **GradCAM:** top-k = **0.0006**, random = **-0.0012**
- **IG:** top-k = **-0.0008**, random = **-0.0012**

and in `orig_rank1_correct`:

- **GradCAM:** top-k = **0.0007**, random = **-0.0012**
- **IG:** top-k = **-0.0007**, random = **-0.0012**

In both groups, the absolute effects are very small, and top-k masking does **not** produce a clearly stronger score drop than random masking. This means the maps are **not strongly identifying regions whose removal reliably hurts the class prediction more than a random control**.

For `orig_rank1_wrong`, the score changes are larger:

- **GradCAM:** top-k = **-0.0098**, random = **-0.0243**
- **IG:** top-k = **-0.0204**, random = **-0.0243**

However, this still does **not** indicate convincing faithfulness, because **random masking hurts at least as much, and in fact more strongly, than top-k masking**. So even where the effects are larger, the result does not support the claim that the most salient regions are uniquely important.

The corresponding gap values are also small overall:

- `all`: **0.0019** (GradCAM), **0.0005** (IG)
- `orig_rank1_correct`: **0.0019** (GradCAM), **0.0005** (IG)
- `orig_rank1_wrong`: **0.0144** (GradCAM), **0.0038** (IG)

These larger gap values in the wrong group should be interpreted carefully, because they are driven by a setting where the random-mask baseline is already highly disruptive.

**Interpretation:** the experiment provides **little evidence that the highlighted regions are strongly faithful** in the masking sense.

**[Insert Figure: `xai_class_faithfulness_barplot.png`]**  
**[Insert Figure: `xai_class_faithfulness_gap_distribution.png`]**


### 3. GradCAM is more plausible qualitatively than IG
The qualitative overlays show a clear difference between the two explainers.

- **GradCAM** often highlights **coarser regions on the jaguar**, especially the **head, face, torso, and spotted coat**, but it frequently also includes substantial **background context** such as vegetation, tree trunks, or water.
- **IG** is much more **diffuse and noisy**, often spreading fine-grained activations across large parts of the image, including the background. In many examples it looks texture-heavy rather than semantically localized.

This is especially visible in the `all` and `orig_rank1_correct` grids: GradCAM usually provides the more interpretable jaguar-centered map, whereas IG often appears dominated by widespread high-frequency patterns. In the `orig_rank1_wrong` example, GradCAM still highlights the face/head region relatively plausibly, but the map also spills into surrounding context; IG remains visually noisy.

**Interpretation:** if one had to choose between the two explainers for qualitative inspection, **GradCAM is the more convincing method**, even though its quantitative faithfulness remains weak.

**[Insert Figure: `qualitative_grid__all.png`]**  
**[Insert Figure: `qualitative_grid__orig_rank1_correct.png`]**  
**[Insert Figure: `qualitative_grid__orig_rank1_wrong.png`]**


### 4. Complexity differs between the methods, but this should be interpreted cautiously
The mean complexity values are:

- `all`: **0.579** (GradCAM) vs **0.604** (IG)
- `orig_rank1_correct`: **0.579** (GradCAM) vs **0.604** (IG)
- `orig_rank1_wrong`: **0.418** (GradCAM) vs **0.681** (IG)

The GradCAM–IG difference is statistically significant for complexity in both the `all` group (**p = 0.0017**) and the `orig_rank1_correct` group (**p = 0.0022**). However, the qualitative overlays suggest that a higher complexity score here should **not automatically be read as “better localization”**: despite higher complexity values, IG still appears visually diffuse and noisy.

**Interpretation:** the complexity metric captures a systematic difference between the explainers, but the qualitative inspection is necessary to interpret what that difference actually means.

**[Insert Table: `xai_class_significance_tests_mannwhitney.csv`]**


## Overall Answer to the Research Questions
---------------------------------------------------

### Which regions drive the model’s identity-class predictions?
The overlays suggest that the model uses a **mixture of jaguar appearance and background context**. GradCAM often focuses on plausible jaguar regions such as the head, face, torso, and coat pattern, but it also repeatedly highlights contextual regions. IG is even less spatially specific and often spreads importance widely across the image.

### Are these explanations meaningful?
**Partly, but not strongly.**
- On the positive side, the explanations appear to **pass the sanity requirement** in the sense that they do not remain stable under randomization.
- On the negative side, the **faithfulness results are weak**: masking the top-k salient regions does not consistently hurt the prediction more than random masking.

So the maps are **not meaningless**, but they also do **not provide strong quantitative evidence of faithful localization**.


## Do we have enough to answer the questions?
---------------------------------------------------
**Mostly yes, but with one important limitation.**

You have enough material to answer the **main question** at a useful level:

- sanity behavior: **yes**
- masking-based faithfulness: **yes**
- qualitative localization patterns: **yes**
- GradCAM vs IG comparison: **yes**

However, the comparison involving **`orig_rank1_wrong`** is **not robust enough for strong conclusions**. In the plots, this group behaves like a very small subset, and there are no corresponding significance-test rows for it. That means you can discuss it **qualitatively and cautiously**, but you should avoid strong claims about “correct vs wrong” differences from this experiment alone.


## Concise Conclusion
---------------------------------------------------
The class-attribution analysis shows that the explanations are **not trivially stable under randomization**, which is a good sign, but they also **fail to provide strong masking-based faithfulness evidence**. Qualitatively, **GradCAM is more interpretable than IG**, because it more often highlights jaguar-centered regions, whereas IG is typically much noisier and more diffuse. Overall, the experiment suggests that the model’s class predictions rely on **both jaguar appearance and contextual cues**, but the current post-hoc explanations are only **partially meaningful** and should be interpreted with caution.
