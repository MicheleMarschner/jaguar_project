import re
from pathlib import Path
from collections import defaultdict

SEED_PATTERN = re.compile(r"seed_(\d+)")

def discover_seed_models(root_dir):
    root = Path(root_dir)
    models = []

    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = SEED_PATTERN.search(d.name)

        if not m:
            continue
        seed = int(m.group(1))

        models.append({
            "path": d,
            "name": d.name,
            "seed": seed
        })

    return models

def parse_training_signature(name):
    parts = name.split("_")

    # Remove prefix: stability_seed_<seed>
    if parts[0] == "stability" and parts[1] == "seed":
        parts = parts[3:]

    # Remaining: e.g. EVA-02_Adam_JaguardIdScheduler
    backbone = parts[0]
    optimizer = parts[1]
    scheduler = parts[2]
    return backbone, optimizer, scheduler

def group_models(models, mode):
    groups = defaultdict(list)
    
    for m in models:
        backbone, optimizer, scheduler = parse_training_signature(m["name"])
        
        if mode == "seed":
            key = m["seed"]
        elif mode == "training":
            key = f"{backbone}_{optimizer}_{scheduler}"
        elif mode == "optimizer":
            key = optimizer
        elif mode == "scheduler":
            key = scheduler
        elif mode == "grid":
            key = f"{optimizer}_{scheduler}"
        else:
            raise ValueError(mode)

        groups[key].append(m)
    return groups