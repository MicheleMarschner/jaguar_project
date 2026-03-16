import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jaguar.config import DATA_ROOT, EXPERIMENTS_STORE, PATHS, DEVICE
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.utils.utils_models import load_or_extract_jaguarid_embeddings
from jaguar.utils.utils_experiments import read_toml_from_path
from jaguar.utils.utils_evaluation import build_local_to_emb_row
from jaguar.utils.utils_datasets import build_processing_fn, load_split_jaguar_from_FO_export
from jaguar.utils.utils import ensure_dir, resolve_path


def get_embedding_cache_path(
    config: dict,
    member_name: str,
    split_name: str,
    cache_prefix: str | None = None,
) -> Path:
    """Build cache path for one ensemble member and split."""
    cache_dir = DATA_ROOT / "embeddings" / "ensemble" / member_name
    ensure_dir(cache_dir)

    tta_tag = "tta" if config["inference"]["use_tta"] else "no_tta"
    prefix = f"{cache_prefix}__" if cache_prefix else ""

    return cache_dir / f"{prefix}{split_name}_{tta_tag}_embeddings.npy"


def load_or_extract_jaguarid_embeddings_cached(
    model,
    torch_ds,
    config: dict,
    member_name: str,
    split_name: str,
    cache_prefix: str | None = None,
) -> np.ndarray:
    """Load cached embeddings or extract them and let the extractor save them."""
    cache_path = get_embedding_cache_path(
        config=config,
        member_name=member_name,
        split_name=split_name,
        cache_prefix=cache_prefix,
    )

    if cache_path.exists():
        print(f"[Cache] Loading embeddings from {cache_path}")
        emb = np.load(cache_path)
        print(f"[Cache] Loaded shape: {emb.shape}")
        return emb

    return load_or_extract_jaguarid_embeddings(
        model=model,
        torch_ds=torch_ds,
        split=split_name,
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix=cache_prefix,
        folder=cache_path.parent,
    )


def _resolve_from_project_root(path_str: str) -> Path:
    """Resolve absolute paths directly and relative paths against PATHS.checkpoints."""
    p = Path(path_str)
    return p if p.is_absolute() else PATHS.checkpoints / p


def load_model(member_cfg: dict, num_classes: int):
    """Load one trained JaguarID ensemble member."""
    model_cfg = read_toml_from_path(_resolve_from_project_root(member_cfg["config_path"]))
    checkpoint_dir = _resolve_from_project_root(member_cfg["checkpoint_path"])

    print(f"Loading model '{member_cfg['name']}' from {checkpoint_dir}...")

    model = JaguarIDModel(
        backbone_name=model_cfg["model"]["backbone_name"],
        num_classes=num_classes,
        head_type=model_cfg["model"]["head_type"],
        device=DEVICE,
        emb_dim=model_cfg["model"]["emb_dim"],
        freeze_backbone=model_cfg["model"]["freeze_backbone"],
        loss_s=model_cfg["model"].get("s", 30.0),
        loss_m=model_cfg["model"].get("m", 0.5),
        use_projection=model_cfg["model"]["use_projection"],
        use_forward_features=model_cfg["model"]["use_forward_features"],
    )

    checkpoint = torch.load(
        checkpoint_dir / "best_model.pth",
        map_location=DEVICE,
        weights_only=False,
    )
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def cosine_similarity_matrix_rect(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute rectangular query-gallery cosine similarity matrix."""
    return query_embeddings @ gallery_embeddings.T


def create_simple_ensemble(config: dict, save_dir=None) -> dict:
    """Load ensemble members and return raw per-member outputs for later fusion."""
    parquet_root = resolve_path(config["data"]["split_data_path"], EXPERIMENTS_STORE)
    data_path = PATHS.data_export / "splits_curated"
    split_df = pd.read_parquet(parquet_root)

    members = config["members"]
    weights = config["fusion"]["weights"]

    gallery_protocol = config["ensemble"]["gallery_protocol"]
    if gallery_protocol not in {"trainval_gallery", "valonly_gallery"}:
        raise ValueError(
            "ensemble.gallery_protocol must be one of {'trainval_gallery', 'valonly_gallery'}"
        )

    train_processing_fn = build_processing_fn(config, split="train")
    val_processing_fn = build_processing_fn(config, split="val")

    _, train_ds, val_ds = load_split_jaguar_from_FO_export(
        data_path,
        overwrite_db=False,
        parquet_path=parquet_root,
        train_processing_fn=train_processing_fn,
        val_processing_fn=val_processing_fn,
        include_duplicates=config["split"]["include_duplicates"],
        use_fiftyone=config["data"]["use_fiftyone"],
    )

    num_classes = len(train_ds.label_to_idx)

    train_local_to_emb_row = build_local_to_emb_row(train_ds, split_df, split="train")
    val_local_to_emb_row = build_local_to_emb_row(val_ds, split_df, split="val")

    member_outputs = {}

    for i, member in enumerate(members):
        print(f"\n========== Model {i + 1}/{len(members)}: {member['name']} ==========")

        model = load_model(member, num_classes=num_classes)

        train_ds.transform = model.backbone_wrapper.transform
        val_ds.transform = model.backbone_wrapper.transform

        cache_prefix = member.get("cache_prefix", member["name"])

        train_embeddings = load_or_extract_jaguarid_embeddings_cached(
            model=model,
            torch_ds=train_ds,
            config=config,
            member_name=member["name"],
            split_name="train",
            cache_prefix=cache_prefix,
        )

        val_embeddings = load_or_extract_jaguarid_embeddings_cached(
            model=model,
            torch_ds=val_ds,
            config=config,
            member_name=member["name"],
            split_name="val",
            cache_prefix=cache_prefix,
        )

        query_embeddings = val_embeddings
        query_labels = np.asarray(val_ds.labels)
        query_global_indices = val_local_to_emb_row

        if gallery_protocol == "trainval_gallery":
            gallery_embeddings = np.concatenate([train_embeddings, val_embeddings], axis=0)
            gallery_labels = np.concatenate(
                [np.asarray(train_ds.labels), np.asarray(val_ds.labels)],
                axis=0,
            )
            gallery_global_indices = np.concatenate(
                [train_local_to_emb_row, val_local_to_emb_row],
                axis=0,
            )
        else:
            gallery_embeddings = val_embeddings
            gallery_labels = np.asarray(val_ds.labels)
            gallery_global_indices = val_local_to_emb_row

        sim_matrix = cosine_similarity_matrix_rect(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
        )

        print(f"  Min:  {sim_matrix.min():.4f}")
        print(f"  Max:  {sim_matrix.max():.4f}")
        print(f"  Mean: {sim_matrix.mean():.4f}")
        print(f"  Std:  {sim_matrix.std():.4f}")

        member_outputs[member["name"]] = {
            "query_embeddings": query_embeddings,
            "gallery_embeddings": gallery_embeddings,
            "sim_matrix": sim_matrix,
            "weight": float(weights[i]),
            "embedding_dim": int(query_embeddings.shape[1]),
            "query_count": int(query_embeddings.shape[0]),
            "gallery_count": int(gallery_embeddings.shape[0]),
        }

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "member_outputs": member_outputs,
        "query_labels": query_labels,
        "gallery_labels": gallery_labels,
        "query_global_indices": query_global_indices,
        "gallery_global_indices": gallery_global_indices,
        "split_df": split_df,
    }