import fiftyone as fo
from PIL import Image
from pathlib import Path
from typing import List, Dict, Union
import shutil
import os


class FODataset:
    """
    Wrapper for standard FiftyOne dataset operations:
    Creation, Sample ingestion and Export.
    """
    def __init__(self, dataset_name: str, overwrite: bool = False):
        self.dataset_name = dataset_name

        if dataset_name in fo.list_datasets():
            if overwrite:
                fo.delete_dataset(dataset_name)
                self.dataset = fo.Dataset(dataset_name)
            else:
                self.dataset = fo.load_dataset(dataset_name)
        else:
            self.dataset = fo.Dataset(dataset_name)

        self.dataset.persistent = True

    def get_dataset(self):
        return self.dataset

    @staticmethod
    def rebind_dataset_paths(dataset, new_root: Union[str, Path]):
        """
        Heals paths by pointing them to a new media root.
        Works even if the current paths are broken, relative, or wrongly absolute.
        """
        new_root = Path(new_root)
        print(f"Rebinding paths to new root: {new_root}")

        # 1. Identify all fields that store filepaths
        schema = dataset.get_field_schema()
        path_fields = ["filepath"] + [f for f in schema if f.endswith("_path")]

        for field in path_fields:
            print(f"Fixing field: {field}...")
            
            # 2. Get the current values from the DB
            current_values = dataset.values(field)
            
            # 3. Reconstruct paths: Take the filename and prepend the new root
            # We assume files are in a subfolder like 'data/' or 'masks/' 
            # based on your previous export logic.
            
            # Identify which subfolder this field belongs to
            subfolder = "data"
            if "mask" in field: subfolder = "masks"
            elif "crop" in field: subfolder = "crops"
            elif "fg_rgba" in field: subfolder = "foreground_rgba"

            # The Magic: Prepend new_root / subfolder / filename
            new_values = [
                str(new_root / subfolder / os.path.basename(v)) if v else None 
                for v in current_values
            ]
            
            # 4. Push back to the database in one single batch operation
            dataset.set_values(field, new_values)
        
        print("✅ Paths successfully updated and synchronized.")

    @classmethod
    def load_from_export(cls, export_dir: Union[str, Path], dataset_name: str, overwrite: bool = False, stage_dir=None):
        """
        Factory method: Creates a manager instance by loading an exported folder 
        and automatically rebinding paths to the absolute location.
        """
        if stage_dir == None:
            stage_dir = export_dir 

        stage_dir = Path(stage_dir)
        
        # 1. Load the dataset using FiftyOne's native from_dir
        # This reads the manifest and sets up the initial relative paths
        ds = fo.Dataset.from_dir(
            dataset_dir=str(stage_dir),
            dataset_type=fo.types.FiftyOneDataset,
            name=dataset_name,
            overwrite=overwrite
        )

        # 2. Rebind paths immediately so the App/Notebook can see the files
        cls.rebind_dataset_paths(ds, export_dir)

        # 3. Return an instance of our class wrapping this dataset
        instance = cls(dataset_name)
        instance.dataset = ds

        return instance

    def _get_relative_path(self, filepath: Path, suffix: Path) -> str:
        """Helper: Internal function to strip prefix."""
        if filepath:
            try:
                # Returns "subfolder/image.png"
                return str(suffix.relative_to(filepath))
            except ValueError:
                # Fallback if file isn't inside filepath
                return str(suffix.name)
        return str(suffix.name)

    @staticmethod
    def normalize_bbox(box, w, h):
        x1, y1, x2, y2 = box
        # Convert [x1, y1, x2, y2] absolute -> [x, y, w, h] normalized
        return [
            x1 / w,          # top-left x
            y1 / h,          # top-left y
            (x2 - x1) / w,   # width
            (y2 - y1) / h    # height
        ]
    
    def create_sample(
        self,
        filepath: Union[str, Path],
        label: str = None,
        tags: List[str] = None,
        metadata: Dict = None,
        detections: List[Dict] = None
    ) -> fo.Sample:
        """
        Constructs a single fo.Sample object without adding it to the DB yet.

        Args:
            filepath: Path to image
            label: Ground truth label (optional)
            tags: List of string tags (e.g. ['train', 'blurry'])
            metadata: Dict of scalar fields (e.g. {'view': 'left', 'quality': 0.9})
            detections: List of dicts containing {'label': str, 'box': [x1,y1,x2,y2], 'score': float}
        """
        abs_path = str(Path(filepath).absolute())
        sample = fo.Sample(filepath=abs_path)

        # 1. Add Basic Label
        if label:
            sample["ground_truth"] = fo.Classification(label=label)

        # 2. Add Tags
        if tags:
            sample.tags = tags

        # 3. Add Custom Metadata (Scalars/Strings)
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    sample[key] = value

        # 4. Add Detections (Requires Image Load for Dimensions)
        if detections:
            # Lazy load image only if we need to normalize bboxes
            with Image.open(filepath) as img:
                w, h = img.size

            fo_dets = []
            for det in detections:
                # Expects dict: {'label': 'jaguar', 'box': [x1,y1,x2,y2], 'score': 0.99}
                norm_box = self.normalize_bbox(det['box'], w, h)
                fo_dets.append(
                    fo.Detection(
                        label=det.get('label', 'object'),
                        bounding_box=norm_box,
                        confidence=det.get('score', None)
                    )
                )

            # Store as a Detections field
            sample["predictions"] = fo.Detections(detections=fo_dets)

        return sample

    def add_samples(self, samples: List[fo.Sample]):
        """Batch add samples to the dataset."""
        self.dataset.add_samples(samples)
        self.dataset.save()

    def export_portable_dataset(self, export_dir: Union[str, Path]):
        """
        Organizes all files into a unified schema and exports a FiftyOne manifest.
        Folder structure:
           export_dir/
              data/             (Images)
              masks/
              crops/
              foreground_rgba/
              metadata.json     (The FiftyOne manifest)
        """
        export_root = Path(export_dir)
        export_root.mkdir(parents=True, exist_ok=True)

        # 1. Clone to avoid messing up your current session's absolute paths
        export_view = self.dataset.clone()

        # 2. Map fields to your desired folder names
        field_to_folder = {
            "filepath": "data",
            "mask_path": "masks",
            "crop_path": "crops",
            "fg_rgba_path": "foreground_rgba"
        }

        print(f"Organizing files and exporting to {export_root}...")

        # 3. Manually move files and update the clone with RELATIVE paths
        for sample in export_view:
            for field, folder in field_to_folder.items():
                src_path = sample[field]
                
                if src_path and Path(src_path).exists():
                    src_path = Path(src_path)
                    
                    # Create the subfolder (e.g., export_dir/masks)
                    dest_folder = export_root / folder
                    dest_folder.mkdir(exist_ok=True)
                    
                    # Copy the file
                    dest_file = dest_folder / src_path.name
                    shutil.copy(src_path, dest_file)
                    
                    # UPDATE SAMPLE: Set path to be relative for the export
                    # e.g., "masks/jaguar_mask.png"
                    sample[field] = f"{folder}/{src_path.name}"
            
            sample.save()

        # 4. Export the metadata only (since we already moved the media)
        # This writes the 'metadata.json' or 'samples.json' that FiftyOne needs
        export_view.export(
            export_dir=str(export_root),
            dataset_type=fo.types.FiftyOneDataset,
            export_media=False, # Important: we already moved them manually
            label_field="ground_truth"
        )

        # 5. Cleanup
        export_view.delete()
        print(f"Export complete. Portable dataset ready at: {export_root}")

    def export_metadata_snapshot(self, export_dir: Union[str, Path]):
        """
        Saves ONLY the FiftyOne manifest (JSON) to the directory.
        Does NOT copy images. Use this for versioning/checkpoints.
        """
        export_root = Path(export_dir)
        export_root.mkdir(parents=True, exist_ok=True)

        # 1. Clone to avoid messing with current session paths
        export_view = self.dataset.clone()

        # 2. Crucial: Ensure the JSON will point to the relative 'data/' folder
        # even if we aren't copying the images right now.
        for sample in export_view:
            # We assume images live in the 'data' subfolder of the Stage 0 export
            sample.filepath = f"data/{Path(sample.filepath).name}"
            sample.save()

        # 3. Export ONLY the JSON
        export_view.export(
            export_dir=str(export_root),
            dataset_type=fo.types.FiftyOneDataset,
            export_media=False, # <--- The key difference
            # label_field="ground_truth"
        )

        export_view.delete()
        print(f"Metadata snapshot saved to {export_root}")
        
    def launch(self):
        """Launch the app in the notebook."""
        session = fo.launch_app(self.dataset, auto=False)
        return session
    