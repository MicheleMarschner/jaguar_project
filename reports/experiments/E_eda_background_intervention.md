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
We kept the trained model and the gallery fixed in the original condition and modified only the query background. We evaluated four query settings: `original`, `gray_bg`, `blur_bg`, and `black_bg`. Retrieval performance was measured with mAP and Rank-1, and all manipulated settings were compared against the original query condition.

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

**Rank-1 stability.** [Paste the auto-generated sentence here.]

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