import argparse

from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.config import PATHS, SEED
from jaguar.utils.utils import set_seeds, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train"])
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.action == "train": 

        set_seeds(SEED) 
        ensure_dirs()
        # intialize wandb
        # choose experiment  

        # get data from fifty One for that experiment
        fo_ds, torch_ds = load_jaguar_from_FO_export(
            PATHS.data_export,
            dataset_name="jaguar_stage0",
            overwrite_db=False,
        )
        

if __name__ == "__main__":
    main()