import torch
import tomli_w

def average_checkpoints(checkpoints):
    avg_state = None
    for ckpt_dir in checkpoints:
        state = torch.load(
            ckpt_dir / "best_model.pth",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]

        if avg_state is None:
            avg_state = {
                k: v.clone().float()
                for k, v in state.items()
            }

        else:
            for k in avg_state:
                avg_state[k] += state[k].float()

    for k in avg_state:
        avg_state[k] /= len(checkpoints)
    return avg_state

def build_soup_override(run_cfg, experiment_meta):

    override = {}

    field_to_section = {
        "apply_tta": ("evaluation", "apply_tta"),
        "tta_modality": ("evaluation", "tta_modality"),
        "apply_qe": ("evaluation", "apply_qe"),
        "top_k_expansion": ("evaluation", "top_k_expansion"),
        "apply_rerank": ("evaluation", "apply_rerank"),
        "group_by": ("evaluation", "group_by"),
        "build_model_soup": ("evaluation", "build_model_soup"),
    }

    for key, value in run_cfg.items():

        if key == "experiment_name":
            continue
        if key not in field_to_section:
            continue

        section, target = field_to_section[key]

        override.setdefault(section, {})
        override[section][target] = value

    override.setdefault("evaluation", {})
    override["evaluation"]["experiment_group"] = experiment_meta["name"]
    return override


def generate_soup_experiments(experiment_cfg, output_root):
    meta = experiment_cfg["experiment"]
    runs = meta["runs"]

    output_dir = output_root / meta["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []

    for run in runs:
        run_name = run["experiment_name"]
        override = build_soup_override(run, meta)
        out_path = output_dir / f"{run_name}.toml"

        with open(out_path, "wb") as f:
            tomli_w.dump(override, f)

        generated_paths.append(out_path)
        print(f"Generated {out_path}")
    return generated_paths
