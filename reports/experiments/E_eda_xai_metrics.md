# Experiment: Interpretability sanity / faithfulness experiment (Data - Round 1)
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