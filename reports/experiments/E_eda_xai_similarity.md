
RQ1: In top-1 retrieval failures, what evidence makes a wrong gallery image outrank the best true match? (qualitative / semi-quantitative)
    get_top1_failures_for_model(...)
    failure summary across models
    triptych panels: query | wrong top-1 | best true match
    overlay visualizations for wrong vs right pairs

RQ2: Do explanation properties differ between easy positives, hard positives, and hard negatives?
    your plots are grouped by pair_type
    your table is grouped by pair_type
    your tests run within (model, pair_type, metric)


Missing
For RQ1: Add a pairwise delta analysis for failures. For each failure query, compare:
- wrong top-1 pair
- best true-match pair
Then compute per-query deltas:
- faith_wrong - faith_right
- complexity_wrong - complexity_right
- sim_wrong - sim_right
That directly answers the failure RQ better than just showing overlays.





# E0X (Q31) Pairwise Similarity Explanations

**Experiment Group:** Interpretability analyses

## Main Research Question
Which image regions drive pairwise similarity scores in Jaguar Re-ID, and how do these regions differ across correct, difficult, and incorrect retrieval pairs?

---

## Sub-experiment 1 — Pairwise Similarity Explanations

### Research Question
Which visual evidence supports high pairwise similarity scores for different query–gallery pair types?

### Intervention
Post-hoc pairwise similarity explanations using GradCAM and Integrated Gradients for selected query–gallery pairs.

### Method / Procedure
A reproducible subset of validation queries was selected and matched against a curated gallery. From the resulting rankings, representative query–gallery pairs were mined for easy positives, hard positives, and hard negatives. For each pair, similarity-target explanations were generated to identify which query regions most increased the similarity score with the reference image.

### Evaluation
Explanations were compared qualitatively across pair types.

Main outputs to show:
- query–reference example panels
- attribution overlays for easy positives, hard positives, and hard negatives
- side-by-side comparison across explainers

### Key Result / Takeaway
[leave for now]

---

## Sub-experiment 2 — Failure Analysis

### Research Question (RQX.2)
In top-1 retrieval failures, what evidence makes a wrong gallery image outrank the best true match?

### Intervention
Pairwise explanation of failure cases by contrasting the wrong top-1 match with the best true match for the same query.

### Method / Procedure
For each failure query, two comparison pairs were selected: the wrong top-1 hard negative and the best available true-match positive. Pairwise similarity explanations were then generated for both pairs under the same model, enabling a direct within-query comparison between incorrect and correct retrieval evidence.


### Evaluation
Failure cases were evaluated through qualitative comparison panels and simple failure summaries.

Main outputs to show:
- query | wrong top-1 | best true match
- similarity gap between wrong and correct pair
- optional explanation differences between wrong and correct pair

### Key Result / Takeaway
[leave for now]

---

## Sub-experiment 3 — Sanity and Faithfulness Checks

### Research Question (RQX.3)
Are the pairwise similarity explanations tied to the learned model and causally related to the similarity score?

### Intervention
Validation of explanations using randomization sanity checks and masking-based faithfulness tests.

### Method / Procedure
A representative subset of explained pairs was used for explanation validation. In the sanity analysis, model parameters were randomized and the resulting explanations were compared with those from the trained model. In the faithfulness analysis, highly relevant regions were masked and the resulting similarity-score drop was compared against random or weaker masking baselines.

### Evaluation
Main validation outputs:
- sanity degradation after randomization
- similarity-score drop under relevance-based masking
- comparison against random or low-relevance masking

### Key Result / Takeaway
[leave for now]

---

## Overall Conclusion
[leave for now]


### Main Findings
- Similarity explanations **[did / did not]** degrade under randomization.
- Masking salient regions reduced similarity more than random masking by **[x.xxx]**.
- The effect was strongest for **[easy positives / hard positives / hard negatives]**.

### Main Results Table
| explainer / pair type | sanity passed | salient-mask similarity drop | random-mask drop | faithfulness gap |
|---|---|---:|---:|---:|
| [method / pair type 1] | [ ] | [ ] | [ ] | [ ] |
| [method / pair type 2] | [ ] | [ ] | [ ] | [ ] |


### Plots
- Salient-mask vs random-mask similarity drop
- Distribution of similarity drop by pair type
- Qualitative pairwise examples: trained vs randomized




Limitation
Similarity-based attribution remains dependent on the chosen explanation method and masking protocol, and different explainers may emphasize different visual correspondences.