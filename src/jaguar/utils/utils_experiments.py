from pathlib import Path
import tomllib

from jaguar.config import EXPERIMENTS_STORE, PATHS
from jaguar.utils.utils import ensure_dir, resolve_path
from jaguar.utils.utils_setup import build_split_stem


def load_toml_config(config_name: str) -> dict:
    with open(PATHS.configs / f"{config_name}.toml", "rb") as f:
        return tomllib.load(f)
    
def load_toml_from_path(path: str | Path) -> dict:
    """Load a TOML file from an arbitrary path."""
    path = Path(path)
    with open(path, "rb") as f:
        return tomllib.load(f)
    
def read_toml_from_path(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result



def to_toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(to_toml_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported type: {type(value)}")


def dict_to_toml(data: dict) -> str:
    lines: list[str] = []

    scalar_items: list[tuple[str, object]] = []
    table_items: list[tuple[str, dict]] = []
    array_table_items: list[tuple[str, list[dict]]] = []

    for key, value in data.items():
        if isinstance(value, dict):
            table_items.append((key, value))
        elif isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            array_table_items.append((key, value))
        else:
            scalar_items.append((key, value))

    for key, value in scalar_items:
        lines.append(f"{key} = {to_toml_value(value)}")

    if scalar_items:
        lines.append("")

    for section, values in table_items:
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {to_toml_value(value)}")
        lines.append("")

    for section, rows in array_table_items:
        for row in rows:
            lines.append(f"[[{section}]]")
            for key, value in row.items():
                lines.append(f"{key} = {to_toml_value(value)}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"



def _pick_value(run_cfg: dict, experiment_meta: dict, base_config: dict, *path, default=None):
    key = path[-1]

    if key in run_cfg:
        return run_cfg[key]
    if key in experiment_meta:
        return experiment_meta[key]

    cur = base_config
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def build_ensemble_override(
    run_cfg: dict,
    experiment_meta: dict,
    base_config: dict,
) -> dict:
    override = {
        "ensemble": {
            "name": run_cfg["experiment_name"],
        }
    }

    if "gallery_protocol" in run_cfg:
        override["ensemble"]["gallery_protocol"] = run_cfg["gallery_protocol"]

    if "members" in run_cfg:
        override["members"] = run_cfg["members"]

    field_to_section = {
        "weights": ("fusion", "weights"),
        "normalize_per_model": ("fusion", "normalize_per_model"),
        "square_before_fusion": ("fusion", "square_before_fusion"),
        "use_tta": ("inference", "use_tta"),
        "use_qe": ("inference", "use_qe"),
        "use_rerank": ("inference", "use_rerank"),
        "batch_size": ("inference", "batch_size"),
        "split_data_path": ("data", "split_data_path"),
        "num_workers": ("data", "num_workers"),
    }

    for key, value in run_cfg.items():
        if key in {"experiment_name", "gallery_protocol", "members"}:
            continue
        if key not in field_to_section:
            continue

        section, target_key = field_to_section[key]
        override.setdefault(section, {})
        override[section][target_key] = value

    output_profile = experiment_meta.get("output_profile")
    experiment_group = experiment_meta.get("name")

    if output_profile is not None or experiment_group is not None:
        override.setdefault("output", {})
        if output_profile is not None:
            override["output"]["profile"] = output_profile
        if experiment_group is not None:
            override["output"]["experiment_group"] = experiment_group

    fusion_suite_cfg = experiment_meta.get("fusion_suite")
    if fusion_suite_cfg is not None:
        override["fusion_suite"] = fusion_suite_cfg

    return override



def resolve_xai_metrics_paths(config: dict):
    source_run_dir = str(config["xai_metrics"]["source_run_dir"])

    run_root = resolve_path(source_run_dir, EXPERIMENTS_STORE)
    if not run_root.exists():
        print(f"[Skip] Missing source XAI run: {run_root}")
        raise SystemExit(0)

    run_root_write = Path(EXPERIMENTS_STORE.write_root) / source_run_dir
    metrics_path = run_root_write / "xai_metrics"
    randomized_root = metrics_path / "explanations_randomized"

    ensure_dir(metrics_path)
    ensure_dir(randomized_root)

    return run_root, run_root_write, metrics_path, randomized_root


def build_xai_override(
    run_cfg: dict,
    experiment_meta: dict,
    base_config: dict,
) -> dict:
    override = {
        "evaluation": {
            "experiment_name": run_cfg["experiment_name"],
        },
    }

    field_to_section = {
        "batch_size": ("inference", "batch_size"),
        "use_tta": ("inference", "use_tta"),
        "use_qe": ("inference", "use_qe"),
        "use_rerank": ("inference", "use_rerank"),

        "checkpoint_dir": ("evaluation", "checkpoint_dir"),

        "num_workers": ("data", "num_workers"),
        "include_duplicates": ("split", "include_duplicates"),

        "dataset_name": ("xai", "dataset_name"),
        "split_name": ("xai", "split_name"),
        "n_samples": ("xai", "n_samples"),
        "seed": ("xai", "seed"),
        "explainer_names": ("xai", "explainer_names"),
        "pair_types": ("xai", "pair_types"),
        "ig_steps": ("xai", "ig_steps"),
        "ig_internal_bs": ("xai", "ig_internal_bs"),
        "ig_batch_size": ("xai", "ig_batch_size"),

        "output_profile": ("output", "profile"),
        "experiment_group": ("output", "experiment_group"),

        "logging_enabled": ("logging", "enabled"),
        "logging_project": ("logging", "project"),
        "logging_mode": ("logging", "mode"),

        "faithfulness_steps": ("xai_metrics.faithfulness", "steps"),
        "faithfulness_baseline": ("xai_metrics.faithfulness", "baseline"),
        "faithfulness_use_abs": ("xai_metrics.faithfulness", "use_abs"),

        "complexity_abs": ("xai_metrics.complexity", "abs"),
        "complexity_normalise": ("xai_metrics.complexity", "normalise"),
        "complexity_display_progressbar": ("xai_metrics.complexity", "display_progressbar"),

        "source_type": ("xai_metrics", "source_type"),
        "source_run_dir": ("xai_metrics", "source_run_dir"),
    }

    for key, value in run_cfg.items():
        if key == "experiment_name":
            continue
        if key not in field_to_section:
            continue

        section, target_key = field_to_section[key]

        if "." in section:
            outer, inner = section.split(".", 1)
            override.setdefault(outer, {})
            override[outer].setdefault(inner, {})
            override[outer][inner][target_key] = value
        else:
            override.setdefault(section, {})
            override[section][target_key] = value

    output_profile = experiment_meta.get("output_profile")
    experiment_group = experiment_meta.get("name")

    if output_profile is not None or experiment_group is not None:
        override.setdefault("output", {})
        if output_profile is not None:
            override["output"]["profile"] = output_profile
        if experiment_group is not None:
            override["output"]["experiment_group"] = experiment_group

    return override


def build_standard_override(
    run_cfg: dict,
    experiment_meta: dict,
    base_config: dict,
) -> dict:
    override = {
        "training": {
            "experiment_name": run_cfg["experiment_name"],
        },
    }

    field_to_section = {
        "train_background": ("preprocessing", "train_background"),
        "val_background": ("preprocessing", "val_background"),

        "backbone_name": ("model", "backbone_name"),
        "head_type": ("model", "head_type"),
        "s": ("model", "s"),
        "m": ("model", "m"),

        "optimizer_type": ("optimizer", "type"),
        "optimizer_lr": ("optimizer", "lr"),
        "optimizer_weight_decay": ("optimizer", "weight_decay"),
        "optimizer_momentum": ("optimizer", "momentum"),
        "optimizer_betas": ("optimizer", "betas"),

        "scheduler_type": ("scheduler", "type"),
        "scheduler_T_max": ("scheduler", "T_max"),
        "scheduler_lr_start": ("scheduler", "lr_start"),
        "scheduler_lr_min": ("scheduler", "lr_min"),
        "scheduler_lr_max": ("scheduler", "lr_max"),
        "scheduler_lr_ramp_ep": ("scheduler", "lr_ramp_ep"),
        "scheduler_lr_sus_ep": ("scheduler", "lr_sus_ep"),
        "scheduler_lr_decay": ("scheduler", "lr_decay"),
        "scheduler_factor": ("scheduler", "factor"),
        "scheduler_patience": ("scheduler", "patience"),
        "scheduler_epochs": ("scheduler", "epochs"),
        "scheduler_total_steps": ("scheduler", "total_steps"),

        "apply_augmentations": ("augmentation", "apply_augmentations"),
        "horizontal_flip": ("augmentation", "horizontal_flip"),
        "affine_degrees": ("augmentation", "affine_degrees"),
        "affine_translate": ("augmentation", "affine_translate"),
        "affine_scale": ("augmentation", "affine_scale"),
        "color_jitter_brightness": ("augmentation", "color_jitter_brightness"),
        "color_jitter_contrast": ("augmentation", "color_jitter_contrast"),
        "random_erasing_p": ("augmentation", "random_erasing_p"),

        "pr_enabled": ("progressive_resizing", "enabled"),
        "pr_sizes": ("progressive_resizing", "sizes"),
        "pr_stage_epochs": ("progressive_resizing", "stage_epochs"),

        "train_k": ("curation", "train_k"),
        "val_k": ("curation", "val_k"),
        "phash_threshold": ("curation", "phash_threshold"),
        "split_strategy": ("split", "strategy"),
        "val_split_size": ("split", "val_split_size"),
        "include_duplicates": ("split", "include_duplicates"),

        "seed": ("training", "seed"),

        "model_name": ("xai", "model_name"),
        "n_samples": ("xai", "n_samples"),
        "split_name": ("xai", "split_name"),
        "pair_types": ("xai", "pair_types"),
        "explainer_names": ("xai", "explainer_names"),
        "ig_steps": ("xai", "ig_steps"),
        "ig_internal_bs": ("xai", "ig_internal_bs"),
        "ig_batch_size": ("xai", "ig_batch_size"),

        "output_profile": ("output", "profile"),
    }

    for key, value in run_cfg.items():
        if key == "experiment_name":
            continue
        if key not in field_to_section:
            continue

        section, target_key = field_to_section[key]
        override.setdefault(section, {})
        override[section][target_key] = value

    output_profile = experiment_meta.get("output_profile")
    experiment_group = experiment_meta.get("name")

    if output_profile is not None or experiment_group is not None:
        override.setdefault("output", {})
        if output_profile is not None:
            override["output"]["profile"] = output_profile
        if experiment_group is not None:
            override["output"]["experiment_group"] = experiment_group

    existing_split_path = _pick_value(run_cfg, experiment_meta, base_config, "data", "split_data_path")

    has_split_override = any(
        key in run_cfg or key in experiment_meta
        for key in ["split_strategy", "include_duplicates", "train_k", "val_k", "phash_threshold"]
    )

    if existing_split_path is not None and not has_split_override:
        override.setdefault("data", {})
        override["data"]["split_data_path"] = existing_split_path
    elif has_split_override:
        split_strategy = _pick_value(run_cfg, experiment_meta, base_config, "split", "strategy")
        include_duplicates = _pick_value(run_cfg, experiment_meta, base_config, "split", "include_duplicates")
        train_k = _pick_value(run_cfg, experiment_meta, base_config, "curation", "train_k")
        val_k = _pick_value(run_cfg, experiment_meta, base_config, "curation", "val_k")
        phash_threshold = _pick_value(run_cfg, experiment_meta, base_config, "curation", "phash_threshold")

        if (
            split_strategy is not None
            and include_duplicates is not None
            and train_k is not None
            and val_k is not None
            and phash_threshold is not None
        ):
            split_relpath = build_split_relpath(
                split_strategy=split_strategy,
                include_duplicates=include_duplicates,
                train_k_per_dedup=train_k,
                val_k_per_dedup=val_k,
                phash_thresh_dedup=phash_threshold,
            )
            override.setdefault("data", {})
            override["data"]["split_data_path"] = split_relpath

    return override


def build_split_relpath(
    split_strategy: str,
    include_duplicates: bool,
    train_k_per_dedup: int,
    val_k_per_dedup: int,
    phash_thresh_dedup: int,
) -> str:
    stem = build_split_stem(
        split_strategy=split_strategy,
        include_duplicates=include_duplicates,
        train_k_per_dedup=train_k_per_dedup,
        val_k_per_dedup=val_k_per_dedup,
        phash_thresh_dedup=phash_thresh_dedup,
    )
    return f"splits/{stem}/full_split.parquet"

