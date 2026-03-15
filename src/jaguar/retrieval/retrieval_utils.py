import json
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

from jaguar.utils.utils_evaluate import (
    extract_embeddings,
    query_expansion,
    k_reciprocal_rerank
)

from jaguar.evaluation.metrics import ReIDEvalBundle


# ------------------Embedding Cache------------------

def get_cached_embeddings(model, loader, tta_mode, cache_dir):

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = cache_dir / f"val_embeddings_{tta_mode}.npy"

    if path.exists():
        print(f"Loading cached embeddings {path}")
        return np.load(path)

    print(f"Extracting embeddings (tta={tta_mode})")

    emb = extract_embeddings(model, loader, tta_mode)

    np.save(path, emb)

    return emb


# ------------------Parameter grid------------------

def build_parameter_grid(run_cfg):

    tta_modes = ["none"]

    if run_cfg.get("apply_tta", False):
        tta_modes = [run_cfg.get("tta_modality", "flip")]

    qe_enabled = [False]
    qe_topk = [None]

    if run_cfg.get("apply_qe", False):
        qe_enabled = [True]
        qe_topk = run_cfg.get("top_k_expansion", [5])

    rerank_enabled = [False]

    if run_cfg.get("apply_rerank", False):
        rerank_enabled = [True]

    k1 = run_cfg.get("k1", [20])
    k2 = run_cfg.get("k2", [6])
    lambda_v = run_cfg.get("lambda_value", [0.3])

    grid = []

    for tta in tta_modes:

        for qe_flag, qe_k in product(qe_enabled, qe_topk):

            for rr in rerank_enabled:

                if rr:

                    for a, b, l in product(k1, k2, lambda_v):

                        grid.append(
                            dict(
                                tta=tta,
                                qe=qe_flag,
                                qe_k=qe_k,
                                rerank=True,
                                k1=a,
                                k2=b,
                                lambda_value=l
                            )
                        )

                else:

                    grid.append(
                        dict(
                            tta=tta,
                            qe=qe_flag,
                            qe_k=qe_k,
                            rerank=False
                        )
                    )

    return grid


# ------------------Evaluation------------------

def evaluate_retrieval(
    embeddings,
    labels,
    qe=False,
    qe_k=None,
    rerank=False,
    k1=20,
    k2=6,
    lambda_value=0.3,
):

    emb = embeddings.copy()

    if qe:
        emb = query_expansion(emb, qe_k)

    sim = emb @ emb.T

    if rerank:
        sim = k_reciprocal_rerank(
            sim,
            k1=k1,
            k2=k2,
            lambda_value=lambda_value
        )

    bundle = ReIDEvalBundle(
        embeddings=emb,
        labels=labels
    )

    metrics = bundle.compute_all()

    return metrics, sim


# ------------------Full Sweep------------------

def run_retrieval_sweep(
    model,
    val_loader,
    labels,
    run_cfg,
    output_dir
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = output_dir / "embeddings"
    runs_dir = output_dir / "runs"

    embeddings_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)

    grid = build_parameter_grid(run_cfg)

    print(f"Parameter combinations: {len(grid)}")

    leaderboard = []

    for params in grid:

        emb = get_cached_embeddings(
            model,
            val_loader,
            params["tta"],
            embeddings_dir
        )

        metrics, sim = evaluate_retrieval(
            emb,
            labels,
            qe=params.get("qe", False),
            qe_k=params.get("qe_k"),
            rerank=params.get("rerank", False),
            k1=params.get("k1", 20),
            k2=params.get("k2", 6),
            lambda_value=params.get("lambda_value", 0.3)
        )

        run_name = "_".join([f"{k}{v}" for k, v in params.items()])

        run_dir = runs_dir / run_name
        run_dir.mkdir(exist_ok=True)

        np.save(run_dir / "similarity_matrix.npy", sim)

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        leaderboard.append({**params, **metrics})

        print(f"{run_name} | pairAP={metrics['pairwise_AP']:.4f}")

    df = pd.DataFrame(leaderboard)

    df = df.sort_values("pairwise_AP", ascending=False)

    df.to_csv(output_dir / "leaderboard.csv", index=False)

    print("\nTOP CONFIGS")
    print(df.head(10))
