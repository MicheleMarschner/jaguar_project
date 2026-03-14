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
| E0X; Q1 | Background intervention | How sensitive is retrieval performance to background changes in the query image? | `reports/experiments/E04_eda_background_intervention.md` |
| E0X; Q0 | Foreground contribution | Which image regions contribute most to similarity-based match decisions, and how stable are these explanations? | `reports/experiments/E_eda_foreground_contribution` |
| E0X; Q2 | XAI class attribution |  | `reports/experiments/E_eda_class_attribution` |
| E0X; Q31 | XAI similarity |  | `reports/experiments/E_eda_xai_similarity` |

## Leaderboard Experiments

| E02; Q14 | Deduplication analysis | How does duplicate removal affect retrieval quality and training behavior?  | `reports/experiments/E02_kaggle_deduplication.md` |
| E03; Q5 | Backbone comparison | Which backbone yields the strongest retrieval performance under the shared protocol? (X models) | `reports/experiments/E03_backbone_comparison.md` |
| E04; Q11 | Loss comparison | (X loss types + X combinations) | `reports/experiments/.md` |
|  | Augmentation  |  | `reports/experiments/.md` |
|  | Scheduler and Optimizer |  | `reports/experiments/.md` |
|  | Resizing |  | `reports/experiments/.md` |

| E0X | Ensemble | Can complementary models be combined to outperform the best single model? | `reports/experiments/E0X_kaggle_ensemble.md` |

### 4.2 Traceability map

| ID | Code | Kaggle | W&B |
|---|---|---|---|
| E01 | `src/jaguar/...` | `kaggle/outputs/...` | `project / group / tag` |
| E02 | `src/jaguar/preprocessing/` | `kaggle/outputs/...` | `project / group / tag` |
| E03 | `src/jaguar/...` | `kaggle/outputs/...` | `project / group / tag` |
| E04 | `src/jaguar/...` | `kaggle/outputs/...` | `project / group / tag` |
| E05 | `src/jaguar/...` | `kaggle/outputs/...` | `project / group / tag` |
| E06 | `src/jaguar/...` | `kaggle/outputs/...` | `project / group / tag` |

## 5. Experimental Logic and Dependency Structure

The experiments followed a staged structure rather than a set of isolated runs. The project began with baseline development and exploratory data analysis (EDA) to establish an initial retrieval setup and understand the dataset. It then moved to data and protocol experiments, including split design, burst analysis, curation, and duplicate handling, since these choices define the basis for all later comparisons.

On this basis, the project proceeded to model and training experiments, followed by robustness and interpretability analyses to study failure modes, sensitivity factors, and model decision behavior. The final stage was ensemble construction, which used the earlier results to combine complementary models.

Overall, the dependency structure was:
baseline and EDA → data and protocol experiments → model and training experiments → robustness and interpretability analyses → ensemble experiments

## 6. Evaluation Protocols
## 7. Where to Find the Evidence / Reproducibility Notes
## 8. Key Findings Summary