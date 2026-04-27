"""Safe output directory handling and sweep table writers."""
from __future__ import annotations

import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import ExperimentResult, result_to_jsonable


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def check_output_conflict(out_dir: Path, overwrite: bool, key_files: tuple[str, ...]) -> None:
    """
    If key outputs already exist and overwrite is False, back them up once and continue,
    so reruns do not silently lose prior results. Prints a clear warning.
    """
    existing = [out_dir / name for name in key_files if (out_dir / name).exists()]
    if not existing:
        return
    if overwrite:
        return
    backup = out_dir / f"backup_before_run_{_timestamp()}"
    backup.mkdir(parents=True, exist_ok=True)
    for p in existing:
        dest = backup / p.name
        shutil.move(str(p), str(dest))
    print(
        f"[output] Existing files in {out_dir.name} were moved to {backup.name} "
        f"(use --overwrite to replace in place next time)."
    )


def write_sweep_csv(path: Path, results: list[ExperimentResult]) -> None:
    """Flatten ExperimentResult rows to a wide CSV (same schema as hnsw_sweep.csv)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    header: list[str] | None = None
    for r in results:
        flat = result_to_jsonable(r)
        lat = flat.pop("latency")
        rec = flat.pop("recall_at_k")
        extra = flat.pop("extra")
        row_dict = {**flat, **lat, **rec, **{f"extra_{k}": v for k, v in extra.items()}}
        if header is None:
            header = list(row_dict.keys())
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerow([row_dict[c] for c in header])
        else:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([row_dict.get(c, "") for c in header])


 