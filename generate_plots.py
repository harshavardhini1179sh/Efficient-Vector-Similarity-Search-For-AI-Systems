#!/usr/bin/env python3
"""Generate PNG plots from experiment outputs under Output/Plots/."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paths import OUTPUT_ROOT

PLOTS_DIR = OUTPUT_ROOT / "Plots"
DEFAULT_SLUGS = ("QQP", "CIFAR10", "UCR_TimeSeries")
MAIN_COMPARE_SLUGS = ("QQP", "CIFAR10")


def _read_comparison(slug: str) -> list[dict]:
    p = OUTPUT_ROOT / slug / "comparison.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_scalability(slug: str) -> list[dict]:
    p = OUTPUT_ROOT / slug / "scalability.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_best(slug: str) -> dict:
    p = OUTPUT_ROOT / slug / "best_configs.csv"
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("config_key", "")
            if key:
                out[key] = row
    return out


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _method_color(method: str) -> str:
    if "HNSW" in method:
        return "#1f77b4"
    if "IVF" in method:
        return "#ff7f0e"
    if "NSG" in method:
        return "#2ca02c"
    return "#7f7f7f"


def plot_recall_vs_latency(slugs: list[str]) -> None:
    fig, axes = plt.subplots(1, len(slugs), figsize=(5 * len(slugs), 4), squeeze=False)
    for ax, slug in zip(axes[0], slugs):
        rows = _read_comparison(slug)
        for r in rows:
            m = r.get("method", "")
            ax.scatter(
                _f(r.get("latency_mean_ms", "nan")),
                _f(r.get("recall@10", "nan")),
                s=22,
                alpha=0.75,
                color=_method_color(m),
                label=m,
            )
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower right")
        ax.set_xlabel("Mean latency (ms)")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"{slug}: recall vs latency")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "com_recall10_vs_latency.png", dpi=150)
    plt.close(fig)


def plot_recall_vs_latency_separate(slugs: list[str]) -> None:
    for slug in slugs:
        rows = _read_comparison(slug)
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rows:
            m = r.get("method", "")
            ax.scatter(
                _f(r.get("latency_mean_ms", "nan")),
                _f(r.get("recall@10", "nan")),
                s=22,
                alpha=0.8,
                color=_method_color(m),
                label=m,
            )
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower right")
        ax.set_xlabel("Mean latency (ms)")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"{slug}: recall vs latency")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"recall10_vs_latency_{slug}.png", dpi=150)
        plt.close(fig)


def plot_recall_vs_index_size(slugs: list[str]) -> None:
    fig, axes = plt.subplots(1, len(slugs), figsize=(5 * len(slugs), 4), squeeze=False)
    for ax, slug in zip(axes[0], slugs):
        rows = _read_comparison(slug)
        for r in rows:
            m = r.get("method", "")
            ax.scatter(
                _f(r.get("index_file_bytes", "nan")),
                _f(r.get("recall@10", "nan")),
                s=22,
                alpha=0.75,
                color=_method_color(m),
                label=m,
            )
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower right")
        ax.set_xlabel("Index file size (bytes)")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"{slug}: recall vs index size")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "com_recall10_vs_index_size.png", dpi=150)
    plt.close(fig)


def plot_recall_vs_index_size_separate(slugs: list[str]) -> None:
    for slug in slugs:
        rows = _read_comparison(slug)
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rows:
            m = r.get("method", "")
            ax.scatter(
                _f(r.get("index_file_bytes", "nan")),
                _f(r.get("recall@10", "nan")),
                s=22,
                alpha=0.8,
                color=_method_color(m),
                label=m,
            )
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower right")
        ax.set_xlabel("Index file size (bytes)")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"{slug}: recall vs index size")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"recall10_vs_index_size_{slug}.png", dpi=150)
        plt.close(fig)


def plot_build_time_vs_size(slugs: list[str]) -> None:
    any_data = False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for slug in slugs:
        rows = _read_scalability(slug)
        if not rows:
            continue
        for fam in ("best_hnsw", "best_ivf", "best_nsg"):
            sub = [r for r in rows if r.get("method_family") == fam]
            if not sub:
                continue
            sub.sort(key=lambda r: _f(str(r.get("n_index", "0"))))
            xs = [_f(str(r.get("n_index", "0"))) for r in sub]
            ys = [_f(str(r.get("build_time_s", "0"))) for r in sub]
            if not any(x == x for x in xs) or not any(y == y for y in ys):
                continue
            any_data = True
            ax.plot(xs, ys, marker="o", label=f"{slug} / {fam}")
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("Index size (n_index)")
    ax.set_ylabel("Build time (s)")
    ax.set_title("Build time vs corpus size (scalability runs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "com_build_time_vs_corpus_size.png", dpi=150)
    plt.close(fig)


def plot_qps_best(slugs: list[str]) -> None:
    qps_vals = []
    labels = []
    for slug in slugs:
        doc = _read_best(slug)
        for key in ("best_hnsw", "best_ivf", "best_nsg"):
            row = doc.get(key)
            if not row:
                continue
            qps_vals.append(_f(str(row.get("latency_qps", "0"))))
            labels.append(f"{slug}\n{key.replace('best_', '')}")
    if not qps_vals:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 4))
    x = np.arange(len(labels))
    ax.bar(x, qps_vals, color="#4c72b0")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("QPS (best config)")
    ax.set_title("Best-configuration QPS by dataset / method")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "com_qps_best_configs.png", dpi=150)
    plt.close(fig)


def plot_best_recall_bar(slugs: list[str]) -> None:
    series: dict[str, list[float]] = {"hnsw": [], "ivf": [], "nsg": []}
    for slug in slugs:
        doc = _read_best(slug)
        for fam, key in (("hnsw", "best_hnsw"), ("ivf", "best_ivf"), ("nsg", "best_nsg")):
            row = doc.get(key)
            series[fam].append(_f(str(row.get("recall@10", "0"))) if row else 0.0)
    if not any(series[k] for k in series):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(slugs))
    w = 0.25
    ax.bar(x - w, series["hnsw"], width=w, label="HNSW")
    ax.bar(x, series["ivf"], width=w, label="IVF")
    ax.bar(x + w, series["nsg"], width=w, label="NSG")
    ax.set_xticks(x, slugs)
    ax.set_ylabel("Recall@10 (best config)")
    ax.set_title("Best-method recall@10 by dataset")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "com_best_recall10_by_dataset.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    slugs = [s for s in DEFAULT_SLUGS if (OUTPUT_ROOT / s).exists()]
    if not slugs:
        raise FileNotFoundError(
            f"No dataset folders under {OUTPUT_ROOT}. Run python run_experiments.py first."
        )
    main_slugs = [s for s in MAIN_COMPARE_SLUGS if s in slugs]
    if not main_slugs:
        main_slugs = slugs

    plot_recall_vs_latency(main_slugs)
    plot_recall_vs_latency_separate(slugs)
    plot_recall_vs_index_size(main_slugs)
    plot_recall_vs_index_size_separate(slugs)
    plot_build_time_vs_size(main_slugs)
    plot_qps_best(main_slugs)
    plot_best_recall_bar(main_slugs)
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
