from pathlib import Path

from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir


def resolve_analysis_paths(
    root_dir: Path | None = None,
    run_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve the analysis input root and corresponding output directory."""
    if run_dir is not None:
        run_root = run_dir
        save_root = PATHS.results / "analysis" / run_dir.parent.name / run_dir.name
    elif root_dir is not None:
        run_root = root_dir
        save_root = PATHS.results / "analysis" / root_dir.name
    else:
        raise ValueError("Expected either run_dir or root_dir")

    ensure_dir(save_root)
    return run_root, save_root