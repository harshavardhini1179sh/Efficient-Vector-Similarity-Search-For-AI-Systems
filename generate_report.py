#!/usr/bin/env python3
"""Build Output/final_report.md from experiment artifacts (after run_experiments.py)."""
from __future__ import annotations

import csv
from pathlib import Path

from src.paths import OUTPUT_ROOT

DEFAULT_SLUGS = ("QQP", "CIFAR10", "UCR_TimeSeries")


def _read_csv_rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: str | None) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def _fmt_row(r: dict) -> str:
    keys = [
        "method",
        "tuning_name",
        "tuning_value",
        "recall@10",
        "latency_mean_ms",
        "latency_qps",
        "build_time_s",
        "index_file_bytes",
        "memory_rss_after_build_mb",
    ]
    parts = [f"{k}={r.get(k, '')}" for k in keys if r.get(k, "") != ""]
    return "; ".join(parts)


def main() -> None:
    slugs = [s for s in DEFAULT_SLUGS if (OUTPUT_ROOT / s).exists()]
    if not slugs:
        raise FileNotFoundError(
            f"No dataset folders under {OUTPUT_ROOT}. Run python run_experiments.py first."
        )

    lines: list[str] = []
    lines.append("# ANN benchmark — final report\n")
    lines.append("## Goal and motivation\n")
    lines.append(
        "This project implements and compares **HNSW** (hnswlib), **IVF** (FAISS `IndexIVFFlat`), "
        "and **NSG** (FAISS `IndexNSGFlat`) on real embeddings from **QQP** (Sentence-BERT), "
        "**CIFAR-10** (CNN GAP features), and **UCR** time series (z-normalized series vectors). "
        "The objective is to measure accuracy (recall vs exact `IndexFlatL2` neighbors), latency, "
        "memory/index size, build time, and scalability across corpus sizes when `--scalability` is used.\n"
    )

    lines.append("## Datasets\n")
    for slug in slugs:
        best_rows = _read_csv_rows(OUTPUT_ROOT / slug / "best_configs.csv")
        meta = best_rows[0] if best_rows else {}
        tier = meta.get("experiment_tier", "")
        lines.append(f"- **{slug}**")
        if tier:
            lines.append(f" — run tier: `{tier}`")
        if slug == "UCR_TimeSeries" and str(meta.get("n_index", "")) == "50":
            lines.append(
                "\n  - *Warning:* UCR split is small (n_index=50). Treat as a sanity check, not a scalability claim."
            )
        lines.append("\n")

    lines.append("## Methods\n")
    lines.append(
        "- **HNSW**: hierarchical navigable small-world graph (`hnswlib`), `ef_search` sweep.\n"
        "- **IVF**: inverted file with k-means cells (`IndexIVFFlat`), `nprobe` sweep.\n"
        "- **NSG**: navigating spreading-out graph (`IndexNSGFlat`), **GK** (graph degree) × **search_L** sweep.\n"
    )

    lines.append("## Metrics\n")
    lines.append(
        "- **Recall@k**: overlap between predicted top-k and exact `IndexFlatL2` top-k, averaged over queries.\n"
        "- **MRR**: mean reciprocal rank of the true nearest neighbor (column 0 of ground truth).\n"
        "- **Latency**: mean / percentiles of per-query time; **QPS** = queries / total query seconds.\n"
        "- **Memory**: RSS after index build (process-level) and **serialized index size** (`index_file_bytes`).\n"
        "- **Build time**: wall time to construct the index (training/add/graph build included per method).\n"
    )

    lines.append("## Experiment setup\n")
    lines.append(
        "Ground truth neighbors are computed with **FAISS `IndexFlatL2`** on the same corpus as each method. "
        "Each sweep varies one tuning knob per family (HNSW `ef_search`, IVF `nprobe`, NSG `GK`×`search_L`). "
        "`best_configs.*` picks the best row per method by **max recall@10**, then **lower mean latency**, "
        "then **lower index bytes**.\n"
    )

    lines.append("## Result tables (best configs)\n")
    for slug in slugs:
        rows = _read_csv_rows(OUTPUT_ROOT / slug / "best_configs.csv")
        by_key = {r.get("config_key", ""): r for r in rows}
        lines.append(f"### {slug}\n")
        for key in ("best_hnsw", "best_ivf", "best_nsg"):
            row = by_key.get(key)
            if not row:
                lines.append(f"- **{key}**: *no run / skipped*\n")
                continue
            lines.append(f"- **{key}**: {_fmt_row(row)}\n")
        lines.append("\n")

    lines.append("## Scalability\n")
    any_scal = False
    for slug in slugs:
        rows = _read_csv_rows(OUTPUT_ROOT / slug / "scalability.csv")
        if not rows:
            continue
        any_scal = True
        lines.append(f"### {slug} (`scalability.csv`)\n")
        lines.append("| size tag | family | n_index | recall@10 | mean ms | build s | index bytes |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            lines.append(
                f"| {r.get('scalability_size_tag','')} | {r.get('method_family','')} | "
                f"{r.get('n_index','')} | {r.get('recall@10','')} | {r.get('latency_mean_ms','')} | "
                f"{r.get('build_time_s','')} | {r.get('index_file_bytes','')} |\n"
            )
        lines.append("\n")
    if not any_scal:
        lines.append(
            "_No `scalability.csv` files found. Re-run with `python run_experiments.py --scalability`._\n\n"
        )

    lines.append("## NSG discussion\n")
    nsg_notes: list[str] = []
    for slug in slugs:
        rows = _read_csv_rows(OUTPUT_ROOT / slug / "best_configs.csv")
        by_key = {r.get("config_key", ""): r for r in rows}
        bh = by_key.get("best_hnsw") or {}
        bn = by_key.get("best_nsg") or {}
        if not bn:
            continue
        rh = _f(str(bh.get("recall@10", "nan")))
        rn = _f(str(bn.get("recall@10", "nan")))
        if rh == rh and rn == rn and (rh - rn) > 0.05:
            nsg_notes.append(
                f"- **{slug}**: best NSG recall@10 ({rn:.4f}) trails best HNSW ({rh:.4f}). "
                "NSG is **parameter-sensitive** (GK, search_L, graph build mode); with the FAISS "
                "implementation and sweeps here it may need wider tuning or more search budget to match HNSW."
            )
    if nsg_notes:
        lines.extend([n + "\n" for n in nsg_notes])
    else:
        lines.append(
            "NSG is included with multi-GK sweeps; if recall matches HNSW at similar latency, graph quality "
            "and search budgets aligned for this corpus. When NSG lags, treat it as evidence that **tuning "
            "and implementation details matter**, not that NSG is universally inferior.\n"
        )

    lines.append("\n## Conclusions\n")
    lines.append(
        "- **HNSW** typically offers a strong **recall–latency** tradeoff on dense embeddings with modest tuning.\n"
        "- **IVF** is simple and scales with corpus size, but **recall depends heavily on `nprobe`** (search breadth).\n"
        "- **NSG** can work well but is **sensitive to GK / search_L / graph construction**; report low recall honestly when observed.\n"
        "- **UCR**: if only a tiny archive split (e.g., classic GunPoint) is available, use it as a **sanity check**, "
        "not as evidence about behavior at thousands+ of series.\n"
    )

    lines.append("\n## Artifacts\n")
    lines.append(
        "- Per dataset: `hnsw_sweep.*`, `ivf_sweep.*`, `nsg_sweep.*`, `comparison.*`, `best_configs.*`, optional `scalability.*`.\n"
        "- Cross-run: `all_datasets_comparison.csv`, `final_summary.*`, `Plots/*.png`, this `final_report.md`.\n"
    )

    out = OUTPUT_ROOT / "final_report.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
