import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from jaguar.config import PATHS
from jaguar.experiments.experiment_output import save_requested_outputs
from jaguar.utils.utils import ensure_dir, save_parquet, to_rel_path
from jaguar.utils.utils_evaluation import build_eval_context
from jaguar.utils.utils_experiments import load_toml_config, deep_update, load_toml_from_path
from jaguar.xai.xai_metrics import compute_saliency_ig_class, compute_saliency_gradcam_class


@dataclass
class ClassAttributionConfig:
    explainer_names: tuple[str, ...]
    groups: tuple[str, ...]
    ig_steps: int
    ig_internal_bs: int
    ig_batch_size: int


def parse_args():
    parser = argparse.ArgumentParser(description="Run class attribution generation")
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    return parser.parse_args()


def _find_single_parquet(run_dir: Path, prefix: str) -> Path:
    matches = sorted(run_dir.glob(f"{prefix}__*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching '{prefix}__*.parquet' in {run_dir}, "
            f"found {len(matches)}: {[m.name for m in matches]}"
        )
    return matches[0]


def load_stage1_source_manifest(stage1_run_dir: Path) -> pd.DataFrame:
    """
    Load and merge the Stage-1 per-sample tables needed for class attribution generation.

    Required inputs from Stage 1:
    - classification_sensitivity__*.parquet
    - analysis_merged__*.parquet
    """
    logits_path = _find_single_parquet(stage1_run_dir, "classification_sensitivity")
    analysis_path = _find_single_parquet(stage1_run_dir, "analysis_merged")

    logits_df = pd.read_parquet(logits_path)
    analysis_df = pd.read_parquet(analysis_path)

    required_logits = {"id", "filepath", "gold_idx", "pred_orig", "is_correct_orig"}
    required_analysis = {"id", "filepath", "query_idx", "is_rank1_orig", "is_rank5_orig", "gold_rank_orig"}

    missing_logits = required_logits - set(logits_df.columns)
    missing_analysis = required_analysis - set(analysis_df.columns)

    if missing_logits:
        raise ValueError(f"Missing required columns in logits parquet: {sorted(missing_logits)}")
    if missing_analysis:
        raise ValueError(f"Missing required columns in analysis parquet: {sorted(missing_analysis)}")

    manifest = analysis_df.merge(
        logits_df,
        on=["id", "filepath"],
        how="inner",
        validate="one_to_one",
    )

    if len(manifest) != len(analysis_df):
        raise ValueError(
            f"Stage-1 merge mismatch: analysis rows={len(analysis_df)}, merged rows={len(manifest)}"
        )

    manifest["group_all"] = True
    manifest["group_orig_rank1_correct"] = manifest["is_rank1_orig"].fillna(False).astype(bool)
    manifest["group_orig_rank1_wrong"] = ~manifest["is_rank1_orig"].fillna(False).astype(bool)

    return manifest


def build_val_resolver(ctx):
    """
    Resolve a global val emb_row back to the corresponding validation dataset sample.
    """
    emb_row_to_local = {
        int(emb_row): int(local_idx)
        for local_idx, emb_row in enumerate(ctx.val_local_to_emb_row)
    }

    def resolve_sample(sample_idx: int):
        sample_idx = int(sample_idx)
        if sample_idx not in emb_row_to_local:
            raise KeyError(f"emb_row {sample_idx} not found in validation mapping")
        local_idx = emb_row_to_local[sample_idx]
        return ctx.val_ds, local_idx, "val"

    return resolve_sample


def build_group_artifact(manifest: pd.DataFrame, group_name: str) -> dict:
    """
    Build the minimal artifact input expected by the class saliency functions.
    """
    if group_name == "all":
        sub = manifest[manifest["group_all"]].copy()
    elif group_name == "orig_rank1_correct":
        sub = manifest[manifest["group_orig_rank1_correct"]].copy()
    elif group_name == "orig_rank1_wrong":
        sub = manifest[manifest["group_orig_rank1_wrong"]].copy()
    else:
        raise ValueError(f"Unknown group: {group_name}")

    sub = sub.sort_values("query_idx").reset_index(drop=True)

    return {
        "meta": {"group": group_name},
        "sample_indices": torch.tensor(sub["query_idx"].astype("int64").to_numpy(), dtype=torch.long),
        "manifest_df": sub,
    }


def save_group_manifest(run_dir: Path, manifest: pd.DataFrame) -> Path:
    out_path = run_dir / "class_attr_source_manifest.parquet"
    save_parquet(out_path, manifest)
    return out_path


def save_group_subsets(run_dir: Path, groups: dict[str, pd.DataFrame]) -> dict[str, str]:
    out = {}
    group_dir = run_dir / "groups"
    ensure_dir(group_dir)

    for group_name, df_group in groups.items():
        path = group_dir / f"group__{group_name}.parquet"
        save_parquet(path, df_group)
        out[group_name] = str(path.name)

    return out


def run_class_attribution_generation(
    config: dict,
    train_config: dict,
    checkpoint_dir: Path,
    stage1_run_dir: Path,
    run_dir: Path,
) -> dict:
    """
    Stage 2:
    - load Stage-1 tables
    - build source manifest
    - form groups
    - compute class saliency maps
    - save artifacts for Stage 3
    """
    cfg = ClassAttributionConfig(
        explainer_names=tuple(config["xai"]["explainer_names"]),
        groups=tuple(config["xai"]["groups"]),
        ig_steps=int(config["xai"].get("ig_steps", 32)),
        ig_internal_bs=int(config["xai"].get("ig_internal_bs", 8)),
        ig_batch_size=int(config["xai"].get("ig_batch_size", 1)),
    )

    ctx = build_eval_context(
        config=config,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        eval_val_setting="original",
    )
    resolve_sample = build_val_resolver(ctx)

    manifest = load_stage1_source_manifest(stage1_run_dir)
    manifest_path = save_group_manifest(run_dir, manifest)

    groups = {}
    artifacts_saved = []

    explanations_root = run_dir / "explanations"
    ensure_dir(explanations_root)

    for group_name in cfg.groups:
        base_artifact = build_group_artifact(manifest, group_name)
        groups[group_name] = base_artifact["manifest_df"]

        if len(base_artifact["manifest_df"]) == 0:
            print(f"[Skip] Empty group: {group_name}")
            continue

        for explainer_name in cfg.explainer_names:
            if explainer_name == "IG":
                art = compute_saliency_ig_class(
                    resolve_sample=resolve_sample,
                    model=ctx.model,
                    artifact=base_artifact,
                    cfg=cfg,
                )
            elif explainer_name == "GradCAM":
                art = compute_saliency_gradcam_class(
                    resolve_sample=resolve_sample,
                    model=ctx.model,
                    artifact=base_artifact,
                    cfg=cfg,
                )
            else:
                raise NotImplementedError(f"Unsupported explainer: {explainer_name}")

            out_dir = explanations_root / explainer_name
            ensure_dir(out_dir)
            out_path = out_dir / f"sal__{group_name}.pt"
            torch.save(art, out_path)

            artifacts_saved.append(
                {
                    "explainer": explainer_name,
                    "group": group_name,
                    "n_samples": int(len(base_artifact["manifest_df"])),
                    "path": str(out_path),
                }
            )

    group_paths = save_group_subsets(run_dir, groups)

    meta = {
        "stage1_run_dir": to_rel_path(stage1_run_dir),
        "manifest_path": to_rel_path(manifest_path),
        "groups": list(cfg.groups),
        "explainers": list(cfg.explainer_names),
        "group_files": group_paths,
    }

    with open(run_dir / "class_attr_generation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "manifest": manifest,
        "artifacts_saved": artifacts_saved,
        "meta": meta,
    }


def main():
    args = parse_args()

    base_config = load_toml_config(args.base_config)
    experiment_config = load_toml_config(args.experiment_config)
    config = deep_update(base_config, experiment_config)
    config.setdefault("evaluation", {})
    config["evaluation"]["experiment_name"] = args.experiment_name

    checkpoint_dir = PATHS.checkpoints / config["evaluation"]["checkpoint_dir"]
    train_config = load_toml_from_path(checkpoint_dir / "config_leaderboard_exp.toml")

    experiment_group = config.get("output", {}).get("experiment_group")
    exp_name = config["evaluation"]["experiment_name"]

    if experiment_group:
        suffix = Path(experiment_group) / exp_name
    else:
        suffix = Path(exp_name)

    run_dir = PATHS.runs / suffix
    ensure_dir(run_dir)

    stage1_rel = config["xai_metrics"]["source_run_dir"]
    stage1_run_dir = PATHS.runs / stage1_rel

    result = run_class_attribution_generation(
        config=config,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        stage1_run_dir=stage1_run_dir,
        run_dir=run_dir,
    )

    artifacts = {
        "run_dir": run_dir,
        "config": config,
    }
    save_requested_outputs(config, artifacts)

    print(f"[Done] class attribution generation: {args.experiment_name}")
    print(f"[Stage1] {stage1_run_dir}")
    print(f"[Saved] {run_dir}")
    print(f"[Artifacts] {len(result['artifacts_saved'])}")


if __name__ == "__main__":
    main()