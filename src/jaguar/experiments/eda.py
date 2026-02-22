from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jaguar.config import PATHS
from jaguar.FiftyOne import (
    FODataset, get_or_create_manifest_dataset
)

FO_DATASET_NAME = "jaguar_stage0"


def run_eda(csv_file: Path, data_path: Path) -> None:

    df = pd.read_csv(csv_file)

    # --- basic EDA ---
    print("=== CSV OVERVIEW ===")
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print("unique filenames:", df["filename"].nunique())
    print("unique IDs:", df["ground_truth"].nunique())

    missing_files = df["filename"].isna().sum() + (df["filename"].astype(str).str.strip() == "").sum()
    missing_ids   = df["ground_truth"].isna().sum() + (df["ground_truth"].astype(str).str.strip() == "").sum()
    print("missing/empty filenames:", int(missing_files))
    print("missing/empty IDs:", int(missing_ids))

    # filename format + index coverage
    pat = re.compile(r"^train_(\d{4})\.png$", re.IGNORECASE)
    ok = df["filename"].astype(str).apply(lambda s: bool(pat.match(s)))
    print("filename format train_####.png valid:", f"{ok.mean()*100:.1f}%")
    if (~ok).any():
        print("bad filename examples:", df.loc[~ok, "filename"].head(10).tolist())

    nums = df["filename"].astype(str).str.extract(r"train_(\d{4})\.png", expand=False).dropna().astype(int)
    mn, mx = int(nums.min()), int(nums.max())
    missing_idx = sorted(set(range(mn, mx + 1)) - set(nums.tolist()))
    print(f"index coverage: {mn}..{mx} | missing indices:", len(missing_idx))
    if missing_idx[:10]:
        print("missing index examples:", missing_idx[:10])

    # --- CSV vs folder check ---
    print("\n=== CSV ↔ TRAIN FOLDER CONSISTENCY ===")
    if not data_path.exists():
        print("TRAIN_DIR does not exist:", data_path)
    else:
        csv_files = set(df["filename"].astype(str))
        disk_files = set(p.name for p in data_path.glob("*") if p.is_file())

        missing_on_disk = sorted(csv_files - disk_files)
        extra_on_disk   = sorted(disk_files - csv_files)

        print("train folder files:", len(disk_files))
        print("missing on disk (in CSV, not in folder):", len(missing_on_disk))
        print("extra on disk (in folder, not in CSV):", len(extra_on_disk))

        if missing_on_disk[:10]:
            print("example missing:", missing_on_disk[:10])
        if extra_on_disk[:10]:
            print("example extra:", extra_on_disk[:10])


    # ReID balance
    print(f"\nSample rows:")
    print(df.head())

    # Analyze identity distribution
    identity_counts = df["ground_truth"].value_counts()
    arr = identity_counts.values  # numpy array of counts per ID

    print("\nIdentity distribution:")
    print("IDs:", len(identity_counts))
    print(f"  Min images per identity: {identity_counts.min()} ({identity_counts.idxmin()})")
    print(f"  Max images per identity: {identity_counts.max()} ({identity_counts.idxmax()})")
    print(f"  Mean images per identity: {identity_counts.mean():.1f}")
    print(f"  Median images per identity: {np.median(arr):.1f}")

    print("\nTop-10 IDs by count:")
    print(identity_counts.head(10))


    # Visualize identity distribution and log to W&B
    fig, ax = plt.subplots(figsize=(14, 5))
    identity_counts.plot(kind='bar', ax=ax, color='steelblue')
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{int(h)}",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
            xytext=(0, 3),
            textcoords="offset points",
        )
    ax.set_xlabel('Jaguar Identity')
    ax.set_ylabel('Number of Images')
    ax.set_title('Training Data: Images per Jaguar Identity')
    ax.axhline(y=identity_counts.mean(), color='red', linestyle='--', label=f'Mean: {identity_counts.mean():.1f}')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Log to W&B
    #wandb.log({"identity_distribution_full": wandb.Image(fig)})
    plt.show()

    # Identify identities that may need careful handling (few samples)
    min_samples_for_split = 2  # Need at least 2 to split
    low_sample_identities = identity_counts[identity_counts < min_samples_for_split]

    if len(low_sample_identities) > 0:   
        print(f"\nWarning: {len(low_sample_identities)} identities have fewer than {min_samples_for_split} images")

    out_dir = PATHS.results / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "identity_distribution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved:", out_path)

    # ---- thresholds to compare ----
    thresholds = [10, 20, 30, 40, 50]  # 1 = baseline (no filtering)

    # ---- summary table ----
    rows = []
    for x in thresholds:
        keep = identity_counts[identity_counts >= x]
        kept_ids = int(len(keep))
        kept_samples = int(keep.sum())
        removed_ids = int((identity_counts < x).sum())
        removed_samples = int(identity_counts[identity_counts < x].sum())

        if kept_ids > 0:
            arr = keep.values
            row = dict(
                min_imgs_per_id=int(arr.min()),
                median_imgs_per_id=float(np.median(arr)),
                mean_imgs_per_id=float(arr.mean()),
                max_imgs_per_id=int(arr.max()),
            )
        else:
            row = dict(min_imgs_per_id=np.nan, median_imgs_per_id=np.nan, mean_imgs_per_id=np.nan, max_imgs_per_id=np.nan)

        rows.append(dict(
            threshold=x,
            kept_ids=kept_ids,
            kept_samples=kept_samples,
            removed_ids=removed_ids,
            removed_samples=removed_samples,
            **row
        ))

    summary = pd.DataFrame(rows)
    print("\n=== FILTER SUMMARY (keep IDs with >= x images) ===")
    print(summary.to_string(index=False))

    summary.to_csv(out_dir / "identity_filter_summary.csv", index=False)
    print("Saved:", out_dir / "identity_filter_summary.csv")



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
    manifest_dir = PATHS.data_export
    csv_file = PATHS.data / "raw/jaguar-re-id/train.csv"
    
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
    # run_eda(csv_file, PATHS.data_train)


if __name__ == "__main__":
    main()