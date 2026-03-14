# Experiment: Interpretability sanity / faithfulness experiment

similarity / pairwise explanation explains why two embeddings match for Re-ID, the second is usually more aligned with the task with IG and GradCam with a similarity target

"""
Since Jaguar Re-ID is evaluated by pair similarity rather than class logits, we used explanation methods targeted at the similarity score between query and reference images. 
"""

RQ: Do attribution maps reflect meaningful model evidence?

RQ: Which explanation method provides more faithful and more stable pair-similarity explanations for Jaguar Re-ID: GradCAM or Integrated Gradients?
    You aggregate faithfulness, sanity/randomization, and complexity/sparseness.
    You compare IG vs GradCAM across models and pair types.
    You create summary tables, boxplots, and significance tests.


For significance can I rather use: Wilcoxon signed-rank test?

