import argparse
from pathlib import Path

from jaguar.utils.utils_evaluate import (
    SUPPORTED_MULTISCALE_BACKBONES,
    load_model_from_toml,
    build_dataset,
    extract_embeddings,
    compute_similarity,
    generate_submission,
)
    

def evaluate_experiment(exp_dir, test_csv, tta_mode, use_qe, use_rerank):
    """Load one experiment folder, run inference on the test set, and write the submission file."""
    exp_dir = Path(exp_dir)

    if not exp_dir.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Experiment folder does not exist:\n{exp_dir}\n"
            "Please recheck the folder path."
        )

    toml_files = list(exp_dir.glob("*.toml"))
    ckpt_files = list(exp_dir.glob("*.pth"))

    if len(toml_files) == 0:
        raise RuntimeError(
            f"\n[ERROR] No TOML config found in {exp_dir}"
        )

    if len(ckpt_files) == 0:
        raise RuntimeError(
            f"\n[ERROR] No checkpoint (.pth) found in {exp_dir}"
        )

    toml_file = toml_files[0]
    checkpoint = ckpt_files[0]

    model = load_model_from_toml(toml_file, checkpoint)

    # Restrict multiscale TTA backbones
    if tta_mode == "multiscale":
        backbone_name = model.backbone_wrapper.name.lower()
        if not any(b in backbone_name for b in SUPPORTED_MULTISCALE_BACKBONES):
            raise ValueError(
                f"\n[ERROR] Multiscale TTA not supported for backbone: {backbone_name}\n"
                f"Supported backbones: {SUPPORTED_MULTISCALE_BACKBONES}\n"
                "Use '--tta flip' or '--tta none'."
            )

    test_df, loader, filenames = build_dataset(
        test_csv,
        model.backbone_wrapper.transform,
    )

    embeddings = extract_embeddings(model, loader, tta_mode)
    
    sim_matrix = compute_similarity(
        embeddings,
        use_qe,
        use_rerank,
    )
    
    generate_submission(
        sim_matrix,
        test_df,
        filenames,
        exp_dir,
    )


def main():
    """Parse CLI arguments and evaluate either one experiment folder or all experiment folders in a directory."""
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiments_dir",
        type=str,
        help="Directory containing multiple experiment folders",
    )
    group.add_argument(
        "--experiment",
        type=str,
        help="Single experiment folder to evaluate",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tta",
        type=str,
        default="none",
        choices=["none", "flip", "multiscale"],
    )
    parser.add_argument(
        "--qe",
        action="store_true",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
    )
    args = parser.parse_args()

    # Single experiment mode
    if args.experiment:

        exp_dir = Path(args.experiment)

        if not exp_dir.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Experiment directory does not exist:\n{exp_dir}\n"
                "Please recheck the folder path."
            )

        print(f"\nEvaluating single experiment: {exp_dir.name}")

        evaluate_experiment(
            exp_dir,
            args.test_csv,
            args.tta,
            args.qe,
            args.rerank,
        )

    # Multiple experiments mode
    else:
        exp_root = Path(args.experiments_dir)

        if not exp_root.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Experiments directory does not exist:\n{exp_root}\n"
                "Please recheck the folder path."
            )

        if not exp_root.is_dir():
            raise NotADirectoryError(
                f"\n[ERROR] Provided path is not a directory:\n{exp_root}"
            )

        exp_dirs = sorted([d for d in exp_root.iterdir() if d.is_dir()])

        if len(exp_dirs) == 0:
            raise RuntimeError(
                f"\n[ERROR] No experiment folders found inside:\n{exp_root}"
            )

        print(f"\nFound {len(exp_dirs)} experiments")

        for exp_dir in exp_dirs:

            print(f"\nEvaluating {exp_dir.name}")

            evaluate_experiment(
                exp_dir,
                args.test_csv,
                args.tta,
                args.qe,
                args.rerank,
            )

if __name__ == "__main__":
    main()