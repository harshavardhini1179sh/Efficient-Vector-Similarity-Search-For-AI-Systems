"""Merge ANN method runs into tables (NSG vs HNSW vs IVF)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .metrics import ExperimentResult, result_to_jsonable


def _tuning(r: ExperimentResult) -> tuple[str, int | str | None]:
    if "IVF" in r.method:
        return ("nprobe", r.ivf_nprobe)
    if "HNSW" in r.method:
        return ("ef_search", r.hnsw_ef_search)
    if "NSG" in r.method:
        # NSG is sensitive to both graph degree (GK) and search list size (search_L).
        gk = r.nsg_GK if r.nsg_GK is not None else 0
        L = r.nsg_search_L if r.nsg_search_L is not None else 0
        return ("nsg_GK_search_L", f"GK{gk}_L{L}")
    return ("param", None)


def comparison_dict(r: ExperimentResult) -> dict[str, Any]:
    """Single flat row for CSV/JSON comparison."""
    d = result_to_jsonable(r)
    lat = d.pop("latency")
    rec = d.pop("recall_at_k")
    extra = d.pop("extra")
    mrr = d.pop("mrr")
    tune_name, tune_val = _tuning(r)
    row: dict[str, Any] = {
        **d,
        "tuning_name": tune_name,
        "tuning_value": tune_val,
        **{f"latency_{k}": v for k, v in lat.items()},
        **rec,
        "mrr": mrr,
    }
    row.update({f"extra_{k}": v for k, v in extra.items()})
    return row


def write_comparison_bundle(
    out_dir: Path,
    dataset_slug: str,
    data_meta: dict[str, Any],
    hnsw_runs: list[ExperimentResult],
    ivf_runs: list[ExperimentResult],
    nsg_runs: list[ExperimentResult] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nsg_runs = nsg_runs or []
    rows = [comparison_dict(r) for r in hnsw_runs + ivf_runs + nsg_runs]
    # Stable sort: method family then tuning value
    def _sort_key(x: dict[str, Any]) -> tuple[Any, ...]:
        tv = x.get("tuning_value")
        if isinstance(tv, (int, float)):
            tv_key: Any = (0, float(tv))
        else:
            tv_key = (1, str(tv or ""))
        gk = x.get("nsg_GK")
        gk_key = int(gk) if str(gk).isdigit() else -1
        return (x.get("method", ""), gk_key, tv_key)

    rows.sort(key=_sort_key)

    csv_path = out_dir / "comparison.csv"
    if csv_path.exists():
        csv_path.unlink()
    if not rows:
        return
    header = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def write_cross_dataset_summary(output_root: Path, slugs: list[str]) -> None:
    """One CSV aggregating comparison.csv from each dataset folder."""
    all_rows: list[dict[str, Any]] = []
    for slug in slugs:
        p = output_root / slug / "comparison.csv"
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                row = dict(row)
                row["dataset_slug"] = slug
                all_rows.append(row)
    if not all_rows:
        return
    key_union: set[str] = set()
    for row in all_rows:
        key_union.update(row.keys())
    keys = ["dataset_slug"] + sorted(k for k in key_union if k != "dataset_slug")
    out = output_root / "all_datasets_comparison.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
