from pathlib import Path
import random
from PIL import Image
import numpy as np
import pandas as pd

from jaguar.config import DATA_STORE, EXPERIMENTS_STORE
from jaguar.datasets.FiftyOneDataset import FODataset, get_or_create_manifest_dataset
from jaguar.utils.utils import resolve_path



def build_habitat_backgrounds(
    raw_dir: Path,
    cutout_dir: Path,
    out_dir: Path,
    n_patches: int = 100,
    patch_size: int = 224,
    max_tries_per_patch: int = 30,
    max_fg_frac: float = 0.02,   # allow <=2% foreground pixels in patch
    seed: int = 51,
):
    """
    Build a pool of mostly jaguar-free background patches from raw images and cutout masks.
    """
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    if not raw_files:
        raise ValueError(f"No images found in {raw_dir}")

    saved = 0
    attempts = 0

    while saved < n_patches and attempts < n_patches * max_tries_per_patch:
        attempts += 1
        raw_path = random.choice(raw_files)
        cut_path = cutout_dir / raw_path.name
        if not cut_path.exists():
            continue

        raw = Image.open(raw_path).convert("RGB")
        cut = Image.open(cut_path).convert("RGBA")  # Use cutout alpha as a foreground mask proxy to avoid sampling patches that contain the jaguar.

        if raw.size != cut.size:
            # if misaligned, skip (or resize cut alpha to raw)
            continue

        W, H = raw.size
        if W < patch_size or H < patch_size:
            continue

        alpha = np.array(cut.getchannel("A"))  # H,W
        # sample a random crop location
        x0 = random.randint(0, W - patch_size)
        y0 = random.randint(0, H - patch_size)

        a_crop = alpha[y0:y0+patch_size, x0:x0+patch_size]
        fg_frac = (a_crop > 0).mean()

        if fg_frac > max_fg_frac:       # Keep only near-background patches; small tolerance handles imperfect cutout masks/edges.
            continue

        patch = raw.crop((x0, y0, x0+patch_size, y0+patch_size))
        out_path = out_dir / f"bg_{saved:06d}.jpg"
        patch.save(out_path, quality=90)
        saved += 1

        if saved % 200 == 0:
            print(f"Saved {saved}/{n_patches} backgrounds...")

    print(f"Done. Saved {saved} patches to {out_dir}. Attempts={attempts}")


def _build_from_csv_labels(
    dataset_name: str,
    train_dir: Path,
    csv_path: Path,
    overwrite_db: bool = True,
) -> FODataset:
    """
    Build a FiftyOne dataset from the training CSV and attach labels and split metadata.
    """
    df = pd.read_csv(csv_path)
    
    print(train_dir, csv_path)

    # basic validation
    assert {"filename", "ground_truth"}.issubset(df.columns), f"CSV columns are {list(df.columns)}"
    assert df["filename"].nunique() == len(df), "Duplicate filenames in CSV"

    fo_wrapper = FODataset(dataset_name=dataset_name, overwrite=overwrite_db)

    samples = []
    missing = 0

    for _, r in df.iterrows():
        p = train_dir / str(r["filename"])
        if not p.exists():
            missing += 1
            continue
        
        label = str(r["ground_truth"])
        s = fo_wrapper.create_sample(filepath=p, label=label, tags=["train"])
        print(s)
        s["split"] = "train"
        s["filename"] = p.name
        samples.append(s)

    if not samples:
        raise RuntimeError("No samples created. Check train_dir and csv filenames.")

    fo_wrapper.add_samples(samples)
    print(f"Built FO dataset with {len(samples)} samples. Missing files: {missing}")
    return fo_wrapper


def init_fiftyone_dataset(
        FO_dataset_name: str, 
        manifest_dir: Path, 
        csv_file: Path,
        train_dir: Path
) -> None:
    """
    Create or load the project FiftyOne dataset and persist its manifest-backed version.
    """
    
    ### add labels to fiftyOne
    def build_fn():
        return _build_from_csv_labels(
            dataset_name=FO_dataset_name,
            train_dir=train_dir,
            csv_path=csv_file,
            overwrite_db=True,
        )
    
    get_or_create_manifest_dataset(
        dataset_name=FO_dataset_name,
        manifest_dir=manifest_dir,
        build_fn=build_fn,
        overwrite_load=False,
    )


def build_split_stem(
    split_strategy: str,
    include_duplicates: bool,
    train_k_per_dedup: int,
    val_k_per_dedup: int,
    phash_thresh_dedup: int,
) -> str:
    """
    Build the canonical filename stem that identifies one split configuration.
    """
    return (
        f"str{split_strategy}"
        f"__dup{include_duplicates}"
        f"__kTrain{train_k_per_dedup}"
        f"__kVal{val_k_per_dedup}"
        f"__p{phash_thresh_dedup}"
    )


def get_split_paths(
    split_strategy: str,
    include_duplicates: bool,
    train_k: int,
    val_k: int,
    phash_threshold: int,
) -> dict[str, Path | str]:
    """
    Return all key paths and relative locations for one configured split bundle.
    """
    stem = build_split_stem(
        split_strategy=split_strategy,
        include_duplicates=include_duplicates,
        train_k_per_dedup=train_k,
        val_k_per_dedup=val_k,
        phash_thresh_dedup=phash_threshold,
    )

    rel_root = Path("splits") / stem

    return {
        "stem": stem,
        "rel_root": rel_root,
        "write_root": EXPERIMENTS_STORE.write_root / rel_root,
        "full_split_relpath": rel_root / "full_split.parquet",
        "full_split_file": EXPERIMENTS_STORE.write_root / rel_root / "full_split.parquet",
        "config_file": EXPERIMENTS_STORE.write_root / rel_root / "config.json",
        "manifest_dir": resolve_path("fiftyone/burst", DATA_STORE),
        "meta_img_file": resolve_path("bursts/meta_img_features.parquet", EXPERIMENTS_STORE),
        "export_dir": DATA_STORE.write_root / "fiftyone" / "splits_curated",
    }


def get_burst_paths() -> dict[str, Path]:
    """
    Return the standard project paths used for burst-analysis artifacts.
    """
    write_root = EXPERIMENTS_STORE.write_root / "bursts"
    return {
        "write_root": write_root,
        "meta_img_file": write_root / "meta_img_features.parquet",
        "cand_file": write_root / "candidate_edges_all_pairs.parquet",
        "diagnostics_cache_dir": write_root / "diagnostics_cache",
    }