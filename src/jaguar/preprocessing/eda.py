from pathlib import Path
import pandas as pd

from jaguar.config import PATHS
from jaguar.datasets.FiftyOneDataset import FODataset
from jaguar.utils.utils_eda import (
    analyze_images, 
    basic_integrity_report, 
    check_filename_and_folder_consistency, 
    class_distribution, 
    identity_filter_summary
)
from jaguar.utils.utils_visualization import plot_identity_distribution, plot_image_dimensions

FO_DATASET_NAME = "jaguar_init"


# ----------------------------
# EDA Analysis
# ----------------------------

def run_eda(data_path: Path, train_file: Path, test_file: Path) -> None:

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    save_dir = PATHS.results / "eda"

    integrity = basic_integrity_report(train_df, test_df)

    counts, class_summary = class_distribution(train_df)
    plot_identity_distribution(
        counts,
        save_path=(save_dir / "identity_distribution.png")
    )

    img_stats_df = analyze_images(train_df, PATHS.data_train)
    
    if not img_stats_df.empty:
        plot_image_dimensions(
            img_stats_df,
            save_path=(save_dir / "image_dimensions.png")
        )

    check_filename_and_folder_consistency(train_df, data_path=PATHS.data_train)

    counts, class_summary = class_distribution(train_df)

    thresholds = [10, 20, 30, 40, 50]
    summary_df = identity_filter_summary(
        identity_counts=counts,
        thresholds=thresholds,
        out_dir=save_dir,
    )

    # Log to W&B
    #wandb.log({"identity_distribution_full": wandb.Image(fig)})
    


def build_from_csv_labels(
    dataset_name: str,
    train_dir: Path,
    csv_path: Path,
    overwrite_db: bool = True,
) -> FODataset:
    df = pd.read_csv(csv_path)
    
    print(train_dir, csv_path)

    # basic validation
    assert {"filename", "ground_truth"}.issubset(df.columns), f"CSV columns are {list(df.columns)}"
    assert df["filename"].nunique() == len(df), "Duplicate filenames in CSV"

    fo_ds = FODataset(dataset_name=dataset_name, overwrite=overwrite_db)

    samples = []
    missing = 0

    for _, r in df.iterrows():
        p = train_dir / str(r["filename"])
        if not p.exists():
            missing += 1
            continue
        
        label = str(r["ground_truth"])
        s = fo_ds.create_sample(filepath=p, label=label, tags=["train"])
        print(s)
        s["split"] = "train"
        s["filename"] = p.name
        samples.append(s)

    if not samples:
        raise RuntimeError("No samples created. Check train_dir and csv filenames.")

    fo_ds.add_samples(samples)
    print(f"Built FO dataset with {len(samples)} samples. Missing files: {missing}")
    return fo_ds


def main():
    manifest_dir = PATHS.data_export / "init"
    csv_file = PATHS.data / "raw/jaguar-re-id/train.csv"
    test_csv_file = PATHS.data / "raw/jaguar-re-id/test.csv"
    
    '''
    ### add labels to fiftyOne
    def build_fn():
        return build_from_csv_labels(
            dataset_name=FO_DATASET_NAME,
            train_dir=PATHS.data_train,
            csv_path=csv_file,
            overwrite_db=True,
        )

    
    fo_ds = get_or_create_manifest_dataset(
        dataset_name=FO_DATASET_NAME,
        manifest_dir=manifest_dir,
        build_fn=build_fn,
        overwrite_load=False,
    )
    '''
    run_eda(PATHS.data_train, train_file=csv_file, test_file=test_csv_file)

    

if __name__ == "__main__":
    main()