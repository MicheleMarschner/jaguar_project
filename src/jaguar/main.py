import argparse
from pathlib import Path
import yaml
from dotenv import load_dotenv

from jaguar.config import PROJECT_ROOT, SEED
from jaguar.utils.utils import set_seeds, ensure_dirs
from jaguar.experiments.gradcam import run_gradcam

def load_experiment_template(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train"])
    parser.add_argument("template_path", nargs="?", type=Path, help="Meaning depends on action")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.action == "train": 
        set_seeds(SEED) 
        ensure_dirs()

        #exp_config = load_experiment_template(args.template_path)

        # intialize wandb if on colab
        #run = init_wandb(exp_config)

        run_gradcam()
        

if __name__ == "__main__":
    main()