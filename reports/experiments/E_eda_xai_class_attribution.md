# E0X (Q2) Class Attributions

**Experiment Group:** Interpretability analyses

## Main Research Question
Which regions drive the model’s identity-class predictions, and are these explanations meaningful?

---

### Setup
We generated class-target explanations using **[Grad-CAM / LRP / IG / attention-based]**. We evaluated them with a randomization sanity check and a masking faithfulness test against a same-sized random-mask control.

### Main Findings
- Explanations **[did / did not]** degrade under randomization, satisfying the sanity requirement.
- Masking top-k salient regions caused a larger confidence drop than masking random regions by **[x.xxx]**.
- The effect was **[consistent / mixed]** across correct and incorrect examples.

### Main Results Table
| explainer | sanity passed | top-k mask drop | random mask drop | faithfulness gap |
|---|---|---:|---:|---:|
| [method 1] | [ ] | [ ] | [ ] | [ ] |
| [method 2] | [ ] | [ ] | [ ] | [ ] |

### Plots
- Salient-mask drop vs random-mask drop
- Faithfulness over different `k` values [if available]
- Qualitative examples: trained vs randomized, correct vs incorrect

### Interpretation
An explainer is only useful here if it passes the sanity check and salient-region masking hurts the model more than random masking.



### Limitation
Attribution maps remain post-hoc explanations and do not by themselves establish causal sufficiency of the highlighted regions.



Sanity

randomize weights or labels

explanations should degrade

Faithfulness

mask top-k salient regions

measure drop in:

class logit / log-prob

and ideally retrieval or mAP too

compare against random same-sized masks

What to report
Main table

Per explainer / setting:

sanity score or qualitative degradation result

mean drop after top-k masking

mean drop after random masking

faithfulness gap = top-k masking drop − random masking drop

Main plots

bar plot: top-k masking drop vs random masking drop

maybe line plot over k values if you tested multiple ks

qualitative panel:

correct example

incorrect example

trained vs randomized explanation

What to say

The main point is not “pretty heatmaps.”
It is:

do explanations degrade under sanity check?

do salient regions matter more than random regions?