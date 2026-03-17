from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
try:
    import fiftyone as fo
    HAS_FIFTYONE = True
except ImportError:
    fo = None
    HAS_FIFTYONE = False
from PIL import Image


FNAME_RE = re.compile(r"^(train|test)_(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)

def _require_fiftyone() -> None:
    """Raise an error when FiftyOne-dependent functionality is used without FiftyOne installed."""
    if not HAS_FIFTYONE:
        raise ImportError(
            "FiftyOne is not installed. Use manifest-only loading on cluster "
            "or install FiftyOne locally."
        )

class FODataset:
    """
    Thin FiftyOne dataset wrapper for creating, loading, and exporting sample collections.
    """

    def __init__(self, dataset_name: str, overwrite: bool = False, persistent: bool = True):
        """Create or load a FiftyOne dataset and optionally overwrite an existing one."""
        _require_fiftyone()
        
        self.dataset_name = dataset_name

        if dataset_name in fo.list_datasets():
            if overwrite:
                fo.delete_dataset(dataset_name)
                self.dataset = fo.Dataset(dataset_name)
            else:
                self.dataset = fo.load_dataset(dataset_name)
        else:
            self.dataset = fo.Dataset(dataset_name)

        self.dataset.persistent = bool(persistent)

    def get_dataset(self):
        """Return the underlying FiftyOne dataset object."""
        return self.dataset

    # ----------------------------
    # Sample creation
    # ----------------------------
    @staticmethod
    def normalize_bbox(box, w, h):
        """Convert an absolute [x1, y1, x2, y2] box to FiftyOne's normalized [x, y, w, h] format."""
        x1, y1, x2, y2 = box
        return [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

    def create_sample(
        self,
        filepath: Union[str, Path],
        label: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        detections: Optional[List[Dict]] = None,
    ):
        """Build a FiftyOne sample from an image path with optional label, tags, metadata, and detections."""
        _require_fiftyone()
        abs_path = str(Path(filepath).absolute())
        sample = fo.Sample(filepath=abs_path)

        if label is not None:
            sample["ground_truth"] = fo.Classification(label=str(label))

        if tags:
            sample.tags = tags

        if metadata:
            for k, v in metadata.items():
                if v is not None:
                    sample[k] = v

        if detections:
            with Image.open(filepath) as img:
                w, h = img.size

            fo_dets = []
            for det in detections:
                norm_box = self.normalize_bbox(det["box"], w, h)
                fo_dets.append(
                    fo.Detection(
                        label=det.get("label", "object"),
                        bounding_box=norm_box,
                        confidence=det.get("score", None),
                    )
                )
            sample["predictions"] = fo.Detections(detections=fo_dets)

        return sample

    def add_samples(self, samples) -> None:
        """Add a batch of samples to the dataset and persist the changes."""
        self.dataset.add_samples(samples)
        self.dataset.save()


    @staticmethod
    def _manifest_exists(dir_: Union[str, Path]) -> bool:
        """Return whether a FiftyOne manifest directory contains expected dataset files."""
        d = Path(dir_)
        return d.exists() and any((d / f).exists() for f in ["samples.json", "metadata.json", "dataset.json"])


    def export_manifest(
        self,
        export_dir: Union[str, Path],
        label_field: str = "ground_truth",
    ) -> Path:
        """Export the dataset as a manifest without copying media files."""
        _require_fiftyone()
        export_root = Path(export_dir)
        export_root.mkdir(parents=True, exist_ok=True)

        view = self.dataset.clone()
        view.export(
            export_dir=str(export_root),
            dataset_type=fo.types.FiftyOneDataset,
            export_media=False,
            label_field=label_field,
        )
        view.delete()
        return export_root

    
    @classmethod
    def load_manifest(
        cls,
        export_dir: Union[str, Path],
        dataset_name: str,
        overwrite_db: bool = False,
        persistent: bool = True,
    ) -> "FODataset":
        """Load a FiftyOne dataset from an exported manifest directory."""
        _require_fiftyone()
        export_dir = Path(export_dir)
        if not cls._manifest_exists(export_dir):
            raise FileNotFoundError(f"Manifest missing in: {export_dir}")

        ds = "fo.Dataset".from_dir(
            dataset_dir=str(export_dir),
            dataset_type=fo.types.FiftyOneDataset,
            name=dataset_name,
            overwrite=overwrite_db,
        )

        inst = cls(dataset_name, overwrite=False, persistent=persistent)
        inst.dataset = ds
        return inst


    def launch(self):
        """Launch the FiftyOne app for the current dataset."""
        _require_fiftyone()
        return fo.launch_app(self.dataset, auto=False)


def rewrite_samples_json_to_data_relative(export_dir: Path, data_root: Path) -> Path:
    """Rewrite manifest filepaths from absolute paths to paths relative to the data root."""
    export_dir = Path(export_dir)
    data_root = Path(data_root).resolve()
    fp = export_dir / "samples.json"

    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    changed = 0

    for s in samples:
        p = Path(s.get("filepath", ""))
        # normalize only if absolute and under data_root
        try:
            rel = p.resolve().relative_to(data_root).as_posix()
        except Exception:
            # if it's not under data_root, leave it as-is
            continue

        s["filepath"] = rel              # now portable
        s["path_root"] = "data"          # optional but useful
        s.setdefault("filename", Path(rel).name)
        changed += 1

    # write back in the same structure
    if isinstance(data, dict) and "samples" in data:
        data["samples"] = samples
        out_obj = data
    else:
        out_obj = samples

    with fp.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[FO] Rewrote {changed} sample filepaths to data-relative in {fp}")
    return fp

def build_from_raw_filename_ids(
    dataset_name: str,
    train_dir: Union[str, Path],
    test_dir: Union[str, Path],
    overwrite_db: bool = True,
) -> FODataset:
    """Build a FiftyOne dataset from raw train/test image folders using filename-based IDs."""
    _require_fiftyone()
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    fo_ds = FODataset(dataset_name=dataset_name, overwrite=overwrite_db)

    samples = []

    # train labeled
    for p in sorted(train_dir.glob("*")):
        if not p.is_file():
            continue
        m = FNAME_RE.match(p.name)
        if not m or m.group(1).lower() != "train":
            continue
        jag_id = m.group(2)  # "0001"
        s = fo_ds.create_sample(filepath=p, label=jag_id, tags=["train"])
        s["split"] = "train"
        samples.append(s)

    # test usually unlabeled
    for p in sorted(test_dir.glob("*")):
        if not p.is_file():
            continue
        m = FNAME_RE.match(p.name)
        if not m or m.group(1).lower() != "test":
            continue
        s = fo_ds.create_sample(filepath=p, tags=["test"])
        s["split"] = "test"
        samples.append(s)

    if not samples:
        raise RuntimeError(
            f"No samples found. Check dirs + filenames.\ntrain_dir={train_dir}\ntest_dir={test_dir}"
        )

    fo_ds.add_samples(samples)
    return fo_ds


def manifest_exists(manifest_dir: Union[str, Path]) -> bool:
    """Return whether a manifest directory contains a valid exported dataset manifest."""
    return FODataset._manifest_exists(manifest_dir)


def get_or_create_manifest_dataset(
    dataset_name: str,
    manifest_dir: Union[str, Path],
    build_fn: Callable[[], FODataset],
    overwrite_load: bool = False,
) -> FODataset:
    """Load a dataset from a manifest when available, otherwise build it and export a new manifest."""
    _require_fiftyone()
    manifest_dir = Path(manifest_dir)

    if manifest_exists(manifest_dir):
        print(f"[FO] Loading manifest from {manifest_dir}")
        return FODataset.load_manifest(
            export_dir=manifest_dir,
            dataset_name=dataset_name,
            overwrite_db=overwrite_load,
        )

    print(f"[FO] No manifest at {manifest_dir}. Building dataset...")
    fo_ds = build_fn()

    print(f"[FO] Exporting manifest to {manifest_dir} (no media copy)")
    fo_ds.export_manifest(manifest_dir)
    return fo_ds



class ManifestDataset:
    """Minimal manifest-backed dataset reader that does not require FiftyOne."""

    def __init__(self, manifest_dir: Union[str, Path]):
        """Load samples from a manifest directory containing a samples.json file."""
        manifest_dir = Path(manifest_dir)
        fp = manifest_dir / "samples.json"
        if not fp.exists():
            raise FileNotFoundError(f"Missing samples.json in: {manifest_dir}")

        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.manifest_dir = manifest_dir
        self.samples = data["samples"] if isinstance(data, dict) and "samples" in data else data

    def __len__(self) -> int:
        """Return the number of samples in the manifest."""
        return len(self.samples)

    def get_samples(self):
        """Return the loaded manifest sample records."""
        return self.samples