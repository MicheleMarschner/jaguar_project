# Jaguar Re-ID Project — SpotItLikeItsHot

SpotItLikeItsHot is a reproducible research project on jaguar re-identification for the Kaggle Jaguar Re-ID challenges. It provides a structured codebase for running experiments, tracking results, and reproducing the main findings from shared configurations and public logs.

This README is intentionally limited to setup and reproducibility. Detailed descriptions of the project experiments, research questions, interventions, and findings are documented separately in **[E00_experiments_outline](reports/E00_experiments_outline.md)**.

> Disclaimer
> AI tools were used exclusively for bug fixing and improving readability, including class and function documentation, visualization styling, and docstrings. They were not used for full code generation or for modeling decisions. 
> Parts of the implementation and experimental setup were further informed by publicly available notebooks and code shared in the active Jaguar Re-ID Kaggle challenge. Such materials served as inspiration and reference points only; all adaptation, integration, and evaluation for this project were carried out independently.

---

## Repository Structure

The project follows a `src` layout to keep source code, configuration, and outputs clearly separated and to support reproducible execution.

!TODO ADAPT

```text
JAGUAR_PROJECT
├── src/
│   └── jaguar/
├── configs/
├── checkpoints/
├── data/
├── documentation/
├── experiments/
├── results/
├── slurm_scripts/
├── tests/
├── pyproject.toml
├── requirements-captum-nodeps.txt
└── README.md
```
---

## Setup Guide

The project was developed in a virtual environment called `jaguar` using **micromamba**. Using the same environment is the easiest way to reproduce the original results. If you prefer `conda`, a very similar setup should also work as long as the same Python version and package versions are used.

### 1. Install Micromamba

Micromamba is a lightweight, standalone package manager that works similarly to conda.

#### macOS / Linux

Download the micromamba binary for your system architecture.

Example for **macOS Apple Silicon (`arm64`)**:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj bin/micromamba
```

Verify the binary architecture:

```bash
file bin/micromamba
```

Move the binary to a location on your `PATH`:

```bash
sudo mv bin/micromamba /usr/local/bin/micromamba
```

Initialize micromamba:

```bash
micromamba shell init -s zsh -r ~/micromamba
source ~/.zshrc
```

For **Linux x86_64**, use the Linux binary instead:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
```

The shell initialization works the same way as on macOS. Restart the terminal after installation if needed.

#### Windows (PowerShell)

```powershell
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1).Content)
```

---

## Create and Activate the Environment

Create the environment with Python 3.11:

```bash
micromamba create -n jaguar python=3.11 -c conda-forge
micromamba activate jaguar
```

Install the project in editable mode:

```bash
pip install -e .
```

Install the additional Captum-related requirements:

```bash
pip install -r requirements-captum-nodeps.txt
```

This setup installs the project dependencies defined in `pyproject.toml` together with the additional requirements used in the interpretability pipeline.

---

## Verify the Installation

Check that the package and key dependencies are available:

```bash
python -m pip show jaguar-project
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); import fiftyone as fo; print('FiftyOne installed')"
```

If you use VS Code or Jupyter, make sure the `jaguar` kernel is selected. If it does not appear automatically, register it manually:

```bash
python -m ipykernel install --user --name jaguar --display-name "Python 3.11 (jaguar)"
```

---

## Data Download

To reproduce the project analyses, download the Kaggle challenge data from:

- [Kaggle Competition Data - Round 1](https://www.kaggle.com/competitions/jaguar-re-id/data)
- [Kaggle Competition Data - Round 2](https://www.kaggle.com/competitions/round-2-jaguar-reidentification-challenge/data)

Even if you only work with one round, we recommend placing the data in the directory structure below so that the existing configuration files can be used without modification.

```text
data/
├── round_1/
│   └── raw/
│       └── jaguar-re-id/
│           ├── test/
│           │   └── test/
│           ├── train/
│           │   └── train/
│           ├── sample_submission.csv
│           ├── test.csv
│           └── train.csv
└── round_2/
    └── raw/
        └── jaguar-re-id/
            ├── test/
            │   └── test/
            ├── train/
            │   └── train/
            ├── sample_submission.csv
            ├── test.csv
            └── train.csv
```

---

## Project Configuration

Before running experiments, check the local path configuration and environment variables.

### 1. Adapt Local Paths

If your local folder structure differs from the expected one, update the relevant paths in the project configuration seen in `config`.

### 2. Create an Environment File

A sample environment file `.env.sample` is included in the repository. Copy it and fill in the required values.

```bash
cp .env.example .env
```

```env
Weights & Biases
WANDB_API_KEY=your_wandb_api_key_here
WANDB_ENTITY=your_wandb_username_or_team
WANDB_PROJECT=jaguar-reid
```

---

## Running Experiments

The project uses a layered configuration setup:

- a **base configuration** for shared default settings in `configs/base`
- an **experiment configuration** in `configs/experiments` for one complete experiment series, containing several ablations and overriding only the parameters relevant to that study

In practice, the final run configuration is created by merging the files in this order. This keeps experiments consistent while making changes traceable and reproducible.

Example command:

```bash
PYTHONPATH=src python -m jaguar.experiments.experiment_runner \
  --base_config base/kaggle_base \
  --experiment_config experiments/kaggle_deduplication
```

This command generates an executable configuration for each run defined in the experiment configuration under `configs/_generated/<experiment_group>` and executes them sequentially.
The commands need to be started from the project root directory

The base config defines the common defaults for training, model setup, preprocessing, and evaluation, while the experiment config specifies the actual intervention to be tested.

---

## Running Analyses

After training or evaluation runs are complete, the corresponding analysis scripts can be executed to recreate summary tables, comparisons, and visualizations.

```bash
PYTHONPATH=src python -m jaguar.analysis.analysis_runner --experiment_group eda_background_intervention --run_name eva02_triplet_bg_orig_gray_black_rdn_blur_mixed
```

!TODO Anpassung!
PYTHONPATH=src python -m jaguar.analysis.analysis_runner --experiment_group eda_foreground_contribution

PYTHONPATH=src python -m jaguar.analysis.analysis_runner --experiment_group baseline
PYTHONPATH=src python -m jaguar.analysis.analysis_runner --experiment_group eda_xai_similarity
jaguar_project % PYTHONPATH=src python -m jaguar.analysis.analysis_runner \
   --experiment_group eda_xai_class_attribution \
  --run_name eva02_val_n332_all_groups

PYTHONPATH=src python -m jaguar.analysis.analysis_runner --experiment_group kaggle_deduplication

---

## Workflow Overview

The general workflow is:

1. set up the environment
2. download the Kaggle data into the expected folder structure
3. adapt local paths and environment variables if needed
4. run an experiment series from a base config plus an experiment override
5. inspect outputs in the saved run directories `experiments/<experiment_group>` and in W&B grouped by `<experiment_group>`
6. run the corresponding analysis scripts to aggregate results and recreate figures or tables

---

### Typical Outputs

Depending on the experiment, runs store artifacts such as:
- full configuration snapshots
- training and validation metrics
- training histories
- saved predictions or embeddings
- aggregated summaries for comparison across runs

---

## Reproducing Results

To reproduce the reported project results, use the same:

- Python version (`3.11`)
- package environment
- dataset folder structure
- base and experiment configuration files
- evaluation protocol
- public W&B runs as reference for expected outputs

The W&B logs are especially useful for verifying whether a reproduced run matches the original trends in training curves, validation metrics, and configuration values.

---

## Weights & Biases

The runs associated with the experiments are available in public W&B projects at:

**[W&B PROJECT LINK](https://wandb.ai/michele-marschner-university-of-potsdam/jaguar-reid-SpotItLikeItsHot?nw=nwusermichelemarschner)**

These projects contain the logged training runs, configurations, metrics, and artifacts used during the project.

> project_name = jaguar-reid-SpotItLikeItsHot

---

## Limitations

- Reproducibility depends on matching the original environment as closely as possible.
- Some external services, paths, or credentials may require local adaptation.
- Hardware differences can affect runtime and, in rare cases, small numerical details.

!TODO make it meaningful