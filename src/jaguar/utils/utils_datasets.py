from pathlib import Path
import fiftyone as fo

from jaguar.datasets.FiftyOneDataset import FODataset
from jaguar.datasets.JaguarDataset import JaguarDataset 


def load_jaguar_from_FO_export(
    manifest_dir,
    dataset_name="jaguar_stage0",
    transform=None,
    processing_fn=None,
    overwrite_db=False,
):
    """
    - If dataset exists in FiftyOne DB: load it
    - Else: import from manifest_dir/samples.json into DB via load_manifest()
    Returns: (fo_wrapper, fo_dataset, torch_dataset)
    """
    manifest_dir = Path(manifest_dir)

    if dataset_name in fo.list_datasets() and not overwrite_db:
        fo_ds = FODataset(dataset_name, overwrite=False)
    else:
        fo_ds = FODataset.load_manifest(
            export_dir=manifest_dir,
            dataset_name=dataset_name,
            overwrite_db=overwrite_db,
        )

    # Torch dataset reads the same samples.json and uses absolute paths inside it
    torch_ds = JaguarDataset(
        base_root=manifest_dir,
        filepath_key="filepath",
        transform=transform,
        processing_fn=processing_fn,
    )

    return fo_ds, torch_ds 