Possible Sub RQs:
Which query background manipulations are most harmful to jaguar Re-ID retrieval?
Do background manipulations induce systematic rank-1 failures or only mild score degradation?
“Performance degradation under query-side background manipulations indicates that retrieval is not fully invariant to background context.”



# E0X (Q1) Background Intervention

**Experiment Group:** Robustness and diagnostic experiments

## Main Research Question
How robust is retrieval performance to different query-side background interventions?

---

## Intervention
We kept the trained model and the gallery fixed in the original condition and modified only the query background. We evaluated four query settings: `original`, `gray_bg`, `black_bg`, `blur_bg`, `random_bg` and `mixed_original_random_bg`. Retrieval performance was measured with mAP and Rank-1, and all manipulated settings were compared against the original query condition.

[PLACEHOLDER: Figure showing example query background variants]

Main Findings
-------------

*   **Best condition:** **original**, with **mAP = 0.996** and **Rank-1 = 0.997**.
    
*   Among the manipulated settings, **blur\_bg** was by far the least harmful intervention, with only **ΔmAP = -0.006** and **ΔRank-1 = -0.003** relative to the original queries.
    
*   The strongest degradation was observed under **random\_bg**, with **ΔmAP = -0.466** and **ΔRank-1 = -0.512**.
    
*   **black\_bg** also caused a large drop (**ΔmAP = -0.368**, **ΔRank-1 = -0.425**), followed by **gray\_bg** (**ΔmAP = -0.274**, **ΔRank-1 = -0.322**).
    
*   The **mixed\_original\_random\_bg** condition was less harmful than fully random replacement, but still clearly degraded retrieval (**ΔmAP = -0.227**, **ΔRank-1 = -0.253**).
    
*   Overall, the effect of background manipulation is **substantial**, indicating that retrieval is **not fully invariant** to background context in the query image.
    

Main Results Table
------------------

| setting | mAP | Rank-1 | ΔmAP vs original | ΔRank-1 vs original |
|----------|-----:|-------:|-----------------:|--------------------:|
| original | 0.996 | 0.997 | 0.000 | 0.000 |
| gray_bg | 0.722 | 0.675 | -0.274 | -0.322 |
| black_bg | 0.628 | 0.572 | -0.368 | -0.425 |
| blur_bg | 0.990 | 0.994 | -0.006 | -0.003 |
| random_bg | 0.530 | 0.485 | -0.466 | -0.512 |
| mixed_original_random_bg | 0.769 | 0.744 | -0.227 | -0.253 |


\[PLACEHOLDER: Bar plot of aggregate mAP and Rank-1 across settings\]

Per-Query Diagnostics
---------------------

To assess whether the degradation was broad or concentrated in a few difficult cases, we also analyzed per-query changes relative to the original condition.

| setting | mean ΔAP vs original | median ΔAP vs original | rank-1 flip rate | mean Δ first positive rank |
|----------|---------------------:|-----------------------:|-----------------:|---------------------------:|
| gray_bg | -0.274 | -0.011 | 0.322 | 39.410 |
| black_bg | -0.368 | -0.057 | 0.425 | 68.422 |
| blur_bg | -0.006 | 0.000 | 0.003 | 0.217 |
| random_bg | -0.466 | -0.502 | 0.512 | 96.386 |
| mixed_original_random_bg | -0.227 | 0.000 | 0.253 | 50.364 |

### Rank-1 stability

The per-query analysis shows that the interventions do **not** all affect retrieval in the same way.

*   **blur\_bg** is almost perfectly stable. The median ΔAP is **0.000**, the mean ΔAP is only **\-0.006**, and the rank-1 flip rate is **0.3%**. This suggests that mild background blurring leaves identity-relevant retrieval cues largely intact.
    
*   **random\_bg** produces the most systematic failures. Its median ΔAP is **\-0.502**, meaning that the degradation is not just driven by a few outliers but affects a large share of queries. The rank-1 flip rate reaches **51.2%**, indicating that more than half of the original rank-1 successes are lost under random background replacement.
    
*   **black\_bg** and **gray\_bg** show an intermediate pattern. Their mean ΔAP values are strongly negative, but the medians are much closer to zero (**\-0.057** for black, **\-0.011** for gray). This indicates a mixed failure mode: many queries remain relatively stable, while another subset collapses substantially.
    
*   **mixed\_original\_random\_bg** behaves similarly to a softened version of random replacement: the median ΔAP remains **0.000**, but the mean ΔAP is clearly negative (**\-0.227**) and the rank-1 flip rate is still high (**25.3%**). This suggests that partial retention of the original background helps, but does not remove the dependency on contextual cues.
    

\[PLACEHOLDER: Boxplot of per-query ΔAP vs original\]

Interpretation
--------------

These findings answer the sub-questions quite clearly.

### Which query background manipulations are most harmful to jaguar Re-ID retrieval?

The most harmful manipulation is **random\_bg**, followed by **black\_bg**, then **gray\_bg**, with **mixed\_original\_random\_bg** somewhat less harmful and **blur\_bg** largely harmless. The ordering is consistent across both aggregate metrics and per-query diagnostics. This suggests that the model is especially sensitive when the original contextual structure is replaced by a semantically different or distribution-shifted background.

A useful pattern emerges here: **preserving coarse scene statistics seems much less damaging than replacing the background content entirely**. Blurring keeps rough color and layout information from the original scene, whereas black, gray, or random replacement introduce a more artificial change. This likely explains why blur causes almost no degradation while the more disruptive manipulations cause large drops.

### Do background manipulations induce systematic rank-1 failures or only mild score degradation?

The answer depends on the manipulation.

*   For **blur\_bg**, the effect is only a **mild score degradation**, and even that is almost negligible.
    
*   For **random\_bg**, the effect is clearly a **systematic rank-1 failure mode**, not just a slight confidence reduction. The large negative median ΔAP and the **51.2%** rank-1 flip rate show that many queries move from correct top-1 retrieval to incorrect retrieval.
    
*   **black\_bg** also induces many failures, though less systematically than random replacement.
    
*   **gray\_bg** and **mixed\_original\_random\_bg** produce a mixed pattern: many queries remain stable, but a sizeable subset experiences strong degradation.
    

Thus, background interventions do not merely lower similarity scores slightly. In the more severe settings, they change the ranking outcome itself and often displace the correct identity far down the list, as reflected by the large increases in first-positive rank.

### Does performance degradation under query-side background manipulations indicate that retrieval is not fully invariant to background context?

Yes. The results provide strong evidence that the retrieval system is **not fully background-invariant**.

If retrieval relied only on jaguar appearance, changing the background while keeping the foreground jaguar unchanged should have had little effect. Instead, several manipulations cause large drops in both **mAP** and **Rank-1**, and they do so at a magnitude that cannot be dismissed as noise. In particular, **random\_bg**, **black\_bg**, and **gray\_bg** substantially alter retrieval behavior, which implies that the learned representation is at least partly entangled with background context.

At the same time, the near-zero degradation under **blur\_bg** is important. It suggests that the model is not simply fragile to any pixel-level change in the background. Rather, it appears sensitive specifically to interventions that **replace or destroy the original context**, while remaining comparatively robust to interventions that **retain coarse contextual structure**. This points to a contextual dependency rather than a generic lack of robustness.

Limitation
----------

This experiment tests query-side robustness to background manipulation, but it does not fully separate true background reliance from artifacts introduced by the manipulation procedure itself. In particular, black and gray backgrounds are somewhat artificial, and random replacement may introduce distribution shift beyond background removal alone. Therefore, the observed degradation should be interpreted as evidence of **sensitivity to altered context**, not as a perfectly clean estimate of the exact amount of identity information stored in the background.

A second limitation is that this analysis is retrieval-focused. It shows that rankings change, but it does not yet reveal _why_ they change at the feature level or whether some identities, poses, or viewpoints are especially vulnerable to background interventions.

Conclusion
----------

The background intervention experiment shows that jaguar Re-ID retrieval is **not fully robust to query-side background changes**. The model performs almost identically to the original condition when the background is merely blurred, but performance drops sharply when the background is replaced by black, gray, or especially random content. The strongest degradation occurs for **random\_bg**, which causes both the largest aggregate performance loss and the clearest evidence of systematic rank-1 failure.

Overall, these results support the conclusion that retrieval depends not only on jaguar appearance but also, to a meaningful extent, on background context. The system is therefore **partially background-dependent rather than fully background-invariant**.





### Main Findings
- **Best condition:** [original / other] with **mAP = [x.xxx]** and **Rank-1 = [x.xxx]**.
- The strongest degradation was observed under **[setting]**, with **ΔmAP = [x.xxx]** and **ΔRank-1 = [x.xxx]** relative to the original queries.
- Across manipulations, the effect was **[small / moderate / substantial]**, indicating that retrieval is **[largely robust / somewhat sensitive / strongly sensitive]** to background changes in the query image.

### Main Results Table

| setting  | mAP | Rank-1 | ΔmAP vs original | ΔRank-1 vs original |
|----------|-----|--------|------------------|---------------------|
| original | [ ] | [ ]    | [ ]              | [ ]                 |
| gray_bg  | [ ] | [ ]    | [ ]              | [ ]                 |
| blur_bg  | [ ] | [ ]    | [ ]              | [ ]                 |
| black_bg | [ ] | [ ]    | [ ]              | [ ]                 |

### Per-Query Diagnostics
To assess whether the degradation was broad or concentrated in a few difficult cases, we also analyzed per-query changes relative to the original condition.

| setting  | mean ΔAP vs original | median ΔAP vs original | rank-1 flip rate | mean Δ first positive rank |
|----------|----------------------|------------------------|------------------|----------------------------|
| gray_bg  | [ ]                  | [ ]                    | [ ]              | [ ]                        |
| blur_bg  | [ ]                  | [ ]                    | [ ]              | [ ]                        |
| black_bg | [ ]                  | [ ]                    | [ ]              | [ ]                        |

**Rank-1 stability.** 

### Plots
1. **Aggregate performance by setting** (`plot_background_main_metrics.png`)  
   Shows mAP and Rank-1 for each background condition.

2. **Performance drop relative to original** (`plot_background_deltas.png`)  
   Highlights which manipulations caused the largest degradation.

3. **Per-query AP change vs original** (`plot_background_delta_ap_boxplot.png`)  
   Shows whether the effect is consistent across queries or driven by a subset of cases.

### Interpretation
The results show that retrieval performance is **[largely invariant / not fully invariant / clearly sensitive]** to query-side background manipulation. Because the gallery remains unchanged and only the query background is altered, any drop relative to the original condition indicates that the model is not perfectly robust to background changes at inference time.

### Limitation
This experiment tests query-side robustness to background manipulation, but it does not fully separate true background intervention from artifacts introduced by the manipulation itself.

### Conclusion
Background manipulation caused the largest performance drop under **[setting]**, while **[setting]** remained closest to the original condition. Overall, this experiment provides evidence that the model is **[robust / somewhat sensitive / strongly sensitive]** to query background changes, suggesting **[limited / partial / substantial]** dependence on contextual information beyond the jaguar itself.




Main table

One row per intervention:

setting

mAP

Rank-1

ΔmAP vs original

ΔRank-1 vs original

Per-query diagnostics

For each intervention:

mean ΔAP vs original

median ΔAP vs original

rank-1 flip rate

mean Δ first positive rank

Main plots

grouped bar plot: mAP and Rank-1 by setting

delta bar plot: ΔmAP and ΔRank-1 vs original

boxplot: per-query ΔAP vs original