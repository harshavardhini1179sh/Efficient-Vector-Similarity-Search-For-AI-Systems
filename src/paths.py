from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
DATASET_DIR = WORKSPACE / "Dataset"
OUTPUT_ROOT = WORKSPACE / "Output"


def output_dir_for(dataset_slug: str) -> Path:
    """Per-dataset folder: Output/<slug>/"""
    p = OUTPUT_ROOT / dataset_slug
    p.mkdir(parents=True, exist_ok=True)
    return p
