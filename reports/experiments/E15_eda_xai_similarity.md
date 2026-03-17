# E15 (Q31) Pairwise Similarity Explanations (Data - Round 1)
==========================================

**Experiment Group:** Interpretability analyses

## Main Research Question
----------------------

Which image regions drive pairwise similarity scores in Jaguar Re-ID, and how do these regions differ across correct, difficult, and incorrect retrieval pairs?

## Sub-experiment 1 — Pairwise Similarity Explanations
---------------------------------------------------

### Research Question

Which visual evidence supports high pairwise similarity scores for different query–gallery pair types?

### Intervention

Post-hoc pairwise similarity explanations using GradCAM and Integrated Gradients for selected query–gallery pairs.

### Method / Procedure

A reproducible subset of validation queries was selected and matched against a curated gallery. Representative pairs were mined for easy positives, hard positives, and hard negatives. For each pair, similarity-target explanations were generated to identify which query regions most increased the similarity score with the reference image.

### Evaluation

Explanations were compared qualitatively across pair types.

Main outputs to show:

*   query–reference example panels
*   attribution overlays for easy positives, hard positives, and hard negatives
*   side-by-side comparison across explainers
    

### Key Result / Takeaway

Similarity is driven mainly by jaguar appearance, especially coat pattern, body contour, and facial structure, but the specificity of this evidence decreases from easy positives to hard negatives.

## Sub-experiment 2 — Failure Analysis
-----------------------------------

### Research Question (RQX.2)

In top-1 retrieval failures, what evidence makes a wrong gallery image outrank the best true match?

### Intervention

Pairwise explanation of failure cases by contrasting the wrong top-1 match with the best true match for the same query.

### Method / Procedure

For each failure query, two comparison pairs were selected: the wrong top-1 hard negative and the best available true-match positive. Pairwise similarity explanations were then generated for both pairs under the same model.

### Evaluation

Failure cases were evaluated through qualitative comparison panels and simple failure summaries.

Main outputs to show:

*   query | wrong top-1 | best true match
*   similarity gap between wrong and correct pair
*   optional explanation differences between wrong and correct pair
    

### Key Result / Takeaway

The shown failure suggests that a wrong gallery image can outrank the true match when it presents stronger overall texture and shape correspondence, even if it is the wrong identity.

### Pair-type differences

**Easy positives** show the clearest and most stable evidence. In the displayed examples, query and reference are visually very close, sometimes almost near-duplicate, and the maps align with the same visible body regions and contours. These cases mainly show that the model can exploit strong visual redundancy.

**Hard positives** are more informative. Here the same identity is matched across more variation in pose and scene context. The maps remain jaguar-centered, but relevance is spread more broadly across flank pattern, torso, legs, and body outline. This suggests that difficult true matches are supported by distributed appearance cues rather than one sharply localized identity marker.

**Hard negatives** are the most diagnostic. In these cases, the maps still emphasize plausible jaguar regions, especially coat texture, facial structure, and body shape, but this evidence is not specific enough to prevent confusion. The model appears to rely on visually convincing animal evidence that is still compatible with the wrong identity.

**\[Insert Figure: easy / hard positive / hard negative example panels\]**

### Failure analysis

The EVA-02 failure summary suggests that top-1 errors are rare in the evaluated subset, but the shown failure is still meaningful because it is not a close tie. The wrong top-1 image clearly scores higher than the best true match, while the true match is ranked far lower.

Qualitatively, the wrong top-1 provides a strong, clean pattern-rich match, whereas the best true match is darker, softer, and less directly aligned to the query. This suggests that the model prefers stronger overall visual correspondence over identity-correct evidence when the latter is weaker or less clearly exposed.

**\[Insert Table: failure summary across models\]****\[Insert Figure: query | wrong top-1 | best true match\]****\[Insert Figure: failure overlay comparison\]**

### Overall interpretation

Across pair types, the main shift is not from animal to background, but from more discriminative to less discriminative animal evidence. Easy positives are supported by highly redundant overlap, hard positives by broader but still correct appearance cues, and hard negatives by plausible yet too generic jaguar evidence. Thus, the qualitative results suggest that retrieval errors arise less from obvious background reliance and more from insufficient identity specificity in the learned similarity signal.

## Overall Conclusion
------------------

The qualitative pairwise explanations indicate that EVA-02 mainly bases similarity on jaguar-centered evidence, especially coat pattern, body contour, and facial regions. Easy positives are driven by strong visual redundancy, while hard positives and hard negatives depend on broader, less specific evidence. The failure case shows that a wrong image can outrank the true match when it offers stronger overall visual correspondence. Overall, the results suggest that the model usually attends to plausible animal evidence, but that this evidence is not always identity-specific enough to separate difficult positives from difficult negatives.

## Main Findings
-------------

*   Similarity maps are mostly centered on the jaguar rather than the background.
*   Easy positives show the strongest and most stable correspondence.
*   Hard positives rely on broader distributed cues across the body.
*   Hard negatives reveal that plausible jaguar evidence can still support incorrect retrieval.
*   The shown failure indicates a preference for strong visual similarity over identity-correct matching when the true match is weaker.
    

## Main Results Table
------------------

sectionresultEasy positivesstrongest and most stable overlapHard positivesbroader, distributed matching cuesHard negativesplausible but insufficiently specific animal evidenceFailure casewrong top-1 clearly outranks best true matchValidity of explanationsrequires separate quantitative evaluation

**\[Insert Table: per-pair-type summary\]**


**\[Insert Figure: pairwise examples by pair type\]****
\[Insert Figure: failure triptych\]****\
[Insert Figure: trained vs randomized examples\]**

Limitation
----------

This section is qualitative and based on a small number of representative examples. The failure analysis is especially limited because it relies on very few failure cases, so the conclusions should be treated as case-based rather than general.

For


## Overall Conclusion
The qualitative pairwise explanations suggest that EVA-02 mainly bases similarity on jaguar-centered evidence, especially coat pattern, body contour, and facial structure. Easy positives are driven by strong visual redundancy, while hard positives and hard negatives rely on broader and less specific evidence. The failure case shows that a wrong image can outrank the true match when it offers stronger overall visual correspondence.

The explanation evaluation tempers this interpretation. Easy positives show the clearest faithfulness signal, but hard negatives do not: in that regime, salient masking is not better than random masking on average. In addition, IG shows almost no sanity effect under randomization, while GradCAM is somewhat more responsive but also much more diffuse in difficult cases. Taken together, the results suggest that pairwise explanations are most interpretable in easy settings, but substantially less reliable for the hard cases that matter most for retrieval errors.

---

## Main Findings
- Similarity maps are mostly centered on the jaguar rather than the background.
- Easy positives show the strongest and most stable correspondence.
- Hard positives rely on broader distributed cues across the body.
- Hard negatives reveal that plausible jaguar evidence can still support incorrect retrieval.
- The shown failure indicates a preference for strong visual similarity over identity-correct matching when the true match is weaker.
- IG shows almost no sanity effect in this setup.
- GradCAM shows somewhat stronger sensitivity to randomization, but becomes much more complex in difficult cases.
- Faithfulness is strongest for easy positives and weakest for hard negatives.

---

## Main Results Table

| section | result |
|---|---|
| Easy positives | strongest and most stable overlap; small positive faithfulness gap |
| Hard positives | broader distributed matching cues; mixed faithfulness evidence |
| Hard negatives | plausible but insufficiently specific animal evidence; negative faithfulness gap |
| Failure case | wrong top-1 clearly outranks best true match |
| IG sanity | near zero across all pair types |
| GradCAM complexity | low for easy positives, much higher for hard negatives and hard positives |

**[Insert Table: per-pair-type summary]**  
**[Insert Figure: pairwise examples by pair type]**  
**[Insert Figure: failure triptych]**  

---

## Limitation
The qualitative part is based on a small set of representative examples, and the failure analysis relies on very few failure cases. The quantitative validation is more informative, but several effects remain modest. In particular, explanation quality is weakest in the hard-negative regime that is most relevant for understanding retrieval errors.