"""Pick best ANN configuration per method from sweep results (recall–latency tradeoff)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .comparison import comparison_dict
from .metrics import ExperimentResult


def _recall10(row: dict[str, Any]) -> float:
    v = row.get("recall@10")
    if v is None or v == "":
        return 0.0
    return float(v)


def _latency_mean(row: dict[str, Any]) -> float:
    v = row.get("latency_mean_ms")
    if v is None or v == "":
        return float("inf")
    return float(v)


def _index_bytes(row: dict[str, Any]) -> float:
    v = row.get("index_file_bytes")
    if v is None or v == "":
        return float("inf")
    return float(v)


def _method_family(method: str) -> str:
    if "HNSW" in method:
        return "hnsw"
    if "IVF" in method:
        return "ivf"
    if "NSG" in method:
        return "nsg"
    return "other"


def select_best_per_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    """
    For each family (hnsw, ivf, nsg), pick one row:
      1) maximize recall@10
      2) tie-break: minimize mean latency
      3) tie-break: minimize index_file_bytes (proxy for disk footprint)
    """
    buckets: dict[str, list[dict[str, Any]]] = {"hnsw": [], "ivf": [], "nsg": []}
    for row in rows:
        fam = _method_family(str(row.get("method", "")))
        if fam in buckets:
            buckets[fam].append(row)

    out: dict[str, dict[str, Any] | None] = {}
    for fam, rs in buckets.items():
        if not rs:
            out[f"best_{fam}"] = None
            continue
        best = max(
            rs,
            key=lambda r: (
                _recall10(r),
                -_latency_mean(r),
                -_index_bytes(r),
            ),
        )
        out[f"best_{fam}"] = dict(best)
    return out


def best_from_results(
    hnsw_runs: list[ExperimentResult],
    ivf_runs: list[ExperimentResult],
    nsg_runs: list[ExperimentResult],
) -> dict[str, Any]:
    rows = [comparison_dict(r) for r in hnsw_runs + ivf_runs + nsg_runs]
    picked = select_best_per_family(rows)
    return {
        "selection_rule": (
            "Maximize recall@10; tie-break lower latency_mean_ms; "
            "then lower index_file_bytes."
        ),
        **picked,
    }


def write_best_configs_bundle(out_dir: Path, bundle: dict[str, Any], _data_meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    for key in ("best_hnsw", "best_ivf", "best_nsg"):
        row = bundle.get(key)
        if row is None:
            continue
        r = dict(row)
        r["config_key"] = key
        rows_out.append(r)
    if not rows_out:
        return
    p = out_dir / "best_configs.csv"
    if p.exists():
        p.unlink()
    keys: set[str] = set()
    for r in rows_out:
        keys.update(r.keys())
    header = ["config_key"] + sorted(k for k in keys if k != "config_key")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)


def read_comparison_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_scalability_bundle(
    out_dir: Path,
    rows: list[dict[str, Any]],
    data_meta: dict[str, Any],
) -> None:
    """Per-dataset scalability table (one row per corpus size × best method config)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "scalability.csv"
    if p.exists():
        p.unlink()
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_final_summary(output_root: Path, slugs: list[str]) -> None:
    """Cross-dataset summary of best configs (one row per dataset × method family)."""
    rows: list[dict[str, Any]] = []
    for slug in slugs:
        p = output_root / slug / "best_configs.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            dataset_rows = list(csv.DictReader(f))
        for row in dataset_rows:
            flat = dict(row)
            flat["dataset_slug"] = slug
            flat["method_bucket"] = row.get("config_key", "")
            rows.append(flat)
    if not rows:
        return
    out = output_root / "final_summary.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

