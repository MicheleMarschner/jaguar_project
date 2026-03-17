# Jaguar Re-Identification — Experiment Overview

## 1. Project Scope and Goal

This repository documents the experiments conducted for the Jaguar Re-Identification Challenge 2026. The goal of the project is to develop and evaluate retrieval-based models that match images of the same jaguar across different observations.

Beyond model comparison, the project includes targeted experiments on data curation, robustness, interpretability, and ensemble methods. Together, these experiments address three core aims: improving retrieval performance, understanding failure modes and sensitivity factors, and assessing whether complementary models can be combined for stronger final results.

## 2. How to Read This Repository

This README is the central index of the experimental project. It is designed to help readers quickly find what they are looking for: which experiments were run, what each experiment was meant to show, and where the corresponding code, artifacts, and tracked runs are located.

The documentation is organized on two levels:
* Overview README: summarizes the full experimental program and maps experiments to their implementation and outputs.
* Per-experiment READMEs: document each experiment in a consistent format: question → intervention → evaluation → results → analysis.

For each experiment, the overview will point to the relevant:
* code entry point or module,
* Kaggle outputs or saved artifacts,
* Weights & Biases runs / groups / tags,
* individual experiment README.

The purpose of this document is not to repeat all methodological details, but to provide a clear and efficient navigation layer between the eda questions, the implementation, and the results.


## 3. Conducted Experiments at a Glance

The experimental work is organized into **five main areas**:
1. *Data-centric studies:* analyses of dataset quality, duplicate structure, and split design.
2. *Core model development:* backbone, training, and embedding experiments aimed at improving retrieval performance.
3. *Robustness and diagnostic experiments:* targeted tests of how retrieval behaves under controlled changes such as background manipulations or alternative evaluation protocols.
4. *Interpretability analyses:* experiments that examine which image regions influence similarity decisions and how reliable these explanations are.
5. *Ensemble experiments:* final experiments that combine complementary models to test whether fusion improves retrieval beyond the best individual model.

Together, these experiment groups move from baseline model development to deeper analysis of robustness, explanation quality, and model complementarity.


## 4. Final Set of Executed Experiments

### 4.1 Experiment overview

## EDA Experiments

| ID | Experiment | Research question | README |
|---|---|---|---|
| E01 | Baseline retrieval model and EDA | Which baseline retrieval setup provides the reference point for later experiments? | `reports/experiments/E01_baseline_and_eda.md` |
| E12; Q1 | Background intervention | How sensitive is retrieval performance to background changes in the query image? | `reports/experiments/E12_eda_background_intervention.md` |
| E13; Q0 | Foreground contribution | Which image regions contribute most to similarity-based match decisions, and how stable are these explanations? | `reports/experiments/E13_eda_foreground_contribution` |
| E14; Q2 | XAI class attribution |  | `reports/experiments/E14_eda_class_attribution` |
| E15; Q31 | XAI similarity |  | `reports/experiments/E15_eda_xai_similarity` |
| E16; | XAI metrics |  | `reports/experiments/E16_eda_xai_similarity` |


## Leaderboard Experiments

| ID | Experiment | Research question | README |
|---|---|---|---|
| E02; Q14 | Deduplication analysis | How does duplicate removal affect retrieval quality and training behavior? | `reports/experiments/E02_kaggle_deduplication.md` |
| E03; Q5 | Backbone comparison | Which backbone yields the strongest retrieval performance under the shared protocol? | `reports/experiments/E03_kaggle_backbone.md` |
| E04; Q11 | Loss comparison | How do different loss functions and head architectures shape embedding quality and retrieval performance in low-data jaguar Re-ID? | `reports/experiments/E04_kaggle_losses.md` |
| E05 | Mining strategy | How does the choice of triplet mining strategy affect convergence and final retrieval performance? | `reports/experiments/E05_kaggle_mining.md` |
| E06 | Scheduler and optimizer | Which optimizer and learning-rate scheduler combination yields the strongest and most robust retrieval performance? | `reports/experiments/E06_kaggle_optim_sched.md` |
| E07 | Augmentation | Which augmentation strategy best regularizes jaguar Re-ID without destroying identity-defining pattern information? | `reports/experiments/E07_kaggle_augmentation.md` |
| E08 | Progressive resizing | Does a progressive resolution curriculum improve retrieval performance and efficiency for convolutional backbones? | `reports/experiments/E08_kaggle_resizing.md` |
| E09 | Retrieval post-processing | To what extent can inference-time techniques such as TTA, query expansion, and re-ranking improve retrieval precision? | `reports/experiments/E09_kaggle_retrieval.md` |
| E10 | Statistical stability | Are the observed gains robust across random seeds, or are they dependent on favorable initializations? | `reports/experiments/E10_kaggle_stability.md` |
| E11 | Ensemble | Can complementary models be combined to outperform the best single model, and are the gains coherent across gallery protocols? | `reports/experiments/E11_kaggle_ensemble.md` |

