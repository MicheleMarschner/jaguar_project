from __future__ import annotations

import torch
import quantus
from pathlib import Path
import gc
import numpy as np

from typing import Any, Dict, Literal, Optional

from src.configs.global_config import PATHS, DEVICE, IG_STEPS
from src.utils import cpu, as_np_int64_1d, collect_x_from_loader
from src.data import get_clean_data, get_corrupted_data
from src.experiment_stages.helper import save_quantus_metrics
from src.metrics import build_quantus_metrics
from src.explainers import mask_invariant, mask_correct



def build_quantus_metrics() -> Dict[str, Any]:
    # Adapted from the Quantus climate tutorial defaults. :contentReference[oaicite:5]{index=5}
    metrics = {
        "complexity__sparseness": quantus.Sparseness(
            abs=True,
            normalise=True,
            normalise_func=quantus.normalise_func.normalise_by_negative,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=True,
            display_progressbar=True, 
        ),
        "robustness__avg_sensitivity": quantus.AvgSensitivity(
            nr_samples=20,  # maybe later 50
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=quantus.functions.perturb_func.batch_uniform_noise,
            perturb_func_kwargs={"perturb_mean":0.0, "perturb_std":0.15},
            similarity_func=quantus.similarity_func.difference,
            abs=True, # im tut false
            return_nan_when_prediction_changes=True, # sollte ich das lassen
            normalise=True, # in einem false in einem true
            normalise_func=quantus.normalise_func.normalise_by_negative,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=False,
            display_progressbar=True, 
        ),
        "faithfulness__corr": quantus.FaithfulnessCorrelation(
            nr_runs=10, # maybe later around 100
            subset_size=224,  # for CIFAR (32*32=1024), 224 is okay-ish; tune later
            perturb_baseline="mean", # or black
            perturb_func=quantus.functions.perturb_func.batch_baseline_replacement_by_indices,
            similarity_func=quantus.similarity_func.correlation_pearson,
            abs=True, # im tut false
            normalise=True,    # in einem true in einem false
            normalise_func=quantus.normalise_func.normalise_by_negative,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=False,
            display_progressbar=True, 
        ),
    }
    return metrics



def to_scalar(x):
    print(type(x))

    return (float(x[0]))


def run_quantus_metrics(
    pair_idx,
    corruption,
    severity,
    clean_path: Path,
    artifact_path: Optional[Path],
    save_path: Path,
    model,
    transform,
    mode: Literal["clean", "corr"] = "corr",
) -> Dict[str, Any]:
    """
    Computes Quantus metrics for either:
      - mode='clean': uses clean data + sal_clean
      - mode='corr' : uses corrupted data + sal_corr (from artifact)

    Protocol:
      y_batch = pred_clean (fixed target)
      x_batch = domain inputs (clean or corrupted)
      a_batch = domain attribution maps (sal_clean or sal_corr)

    Saves:
      out_path (.pt): payload {row, metrics, meta}
      03__quantus_results.csv: upsert by (corruption,severity)

    Note:
    Quantus should see inputs exactly as the model sees inputs during inference.
    a_batch shape/type is what Quantus expects. / Use Quantus’ explain only if you want a quick sanity run or if you’re struggling with shape/device quirks.
    s_batch is segmentation masks only needed for Localization
    abs = absolute relevance (treat negative and positive relevance as “importance magnitude”). -> abs=True: importance = magnitude, ignores sign
    normalise = Whether Quantus should normalize the attribution map before computing the metric.
    return_nan_when_prediction_changes = used in metrics where the protocol assumes you evaluate explanations for a fixed decision. If prediction changes during perturbations
    """

    # Load stage00 reference
    ref = torch.load(clean_path, map_location="cpu", weights_only=False)
    cr = ref["clean_reference"]

    pred_clean = cr["pred_clean"].long()       # torch tensor [N]
    y_batch = as_np_int64_1d(pred_clean)


    # Decide domain inputs + attributions
    if mode == "clean":
        clean_loader, _, _ = get_clean_data(path=PATHS.data_clean, idx=pair_idx, transform=transform)
        X_clean_t = collect_x_from_loader(clean_loader)     # torch [N,C,H,W]
        x_batch = cpu(X_clean_t).numpy()

        #sal_clean = cr["sal_clean"].float()
        #a_batch = cpu(sal_clean.unsqueeze(1)).numpy()  # numpy [N,1,H,W]

        masks = {
            "all": np.ones_like(pred_clean, dtype=bool)
        }

    else:
        corruption = str(corruption)
        severity = int(severity)

        corr_loader, _, _ = get_corrupted_data(
            idx=pair_idx,
            path=PATHS.data_corr,
            transform=transform,
            corruption=corruption,
            severity=severity,
        )
        X_corr_t = collect_x_from_loader(corr_loader)
        x_batch = cpu(X_corr_t).numpy()

        y_true = cr["y_clean"]

        art = torch.load(artifact_path, map_location="cpu", weights_only=False)
        cc = art["corrupt_reference"]
        pred_corr = cc["pred_corr"]

        #sal_corr = cc["sal_corr"].float()
        #a_batch = cpu(sal_corr.unsqueeze(1)).numpy()  # numpy [N,1,H,W]

        masks = {
            "all": np.ones_like(pred_clean, dtype=bool),
            "inv": mask_invariant(pred_clean, pred_corr).bool(),
            "both_corr": mask_correct(pred_clean, pred_corr, y_true).bool()
        }

        n_invariant = masks['inv'].sum().item()
        print(f" labels corresponding both domains {n_invariant} from {len(pred_clean)}")
        n_both_correct = masks['both_corr'].sum().item()
        print(f" labels correct in both domains {n_both_correct} from {len(pred_clean)}")

    assert x_batch.shape[0] == y_batch.shape[0]
    assert y_batch.ndim == 1
    #assert a_batch.shape[0] == x_batch.shape[0] == y_batch.shape[0]
    #assert a_batch.shape[-2:] == x_batch.shape[-2:]  # (H,W)

    # -----------------------
    # Quantus metrics
    # -----------------------
    metrics = build_quantus_metrics()

    # Quantus runs forward passes 
    model.eval()

    results = {}
    for slice_name, m in masks.items():
        idx = np.where(m)[0]

        # always record slice size
        results[f"n__{slice_name}"] = int(idx.size)

        # handle empty slice
        if idx.size == 0:
            # store NaN if slice is empty
            for metric_name in metrics:
                results[f"{metric_name}__{slice_name}"] = float("nan")
            continue

        x_s = x_batch[idx]
        y_s = y_batch[idx]

        for metric, metric_func in metrics.items():
            print(f"Evaluating {metric}.")
            gc.collect()
            torch.cuda.empty_cache()

            scores = metric_func(
                model=model, 
                x_batch=x_s, 
                y_batch=y_s, 
                a_batch=None,
                s_batch=None,
                device=DEVICE,
                explain_func=quantus.explain,   
                explain_func_kwargs={
                    "method": "IntegratedGradients",
                    "n_steps": int(IG_STEPS),
                    "device": DEVICE,
                },
            )
            results[f"{metric}__{slice_name}"] = to_scalar(scores)

    row = {"corruption": corruption, "severity": severity, **results}
    
    save_quantus_metrics(save_path, row, mode)

    # Empty cache.
    gc.collect()
    torch.cuda.empty_cache()

    return row