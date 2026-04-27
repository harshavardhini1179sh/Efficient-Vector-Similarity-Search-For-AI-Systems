#!/usr/bin/env python3
"""Run ANN experiments and write results under Output/<dataset_slug>/."""
from __future__ import annotations

import os

# Keep BLAS/OpenMP single-threaded for stability on macOS.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
from typing import Any

import numpy as np

from src.best_configs import (
    best_from_results,
    select_best_per_family,
    write_best_configs_bundle,
    write_final_summary,
    write_scalability_bundle,
)
from src.comparison import comparison_dict, write_comparison_bundle, write_cross_dataset_summary
from src.hnsw_eval import exact_neighbors, sweep_ef_search
from src.ivf_eval import sweep_ivf_nprobe
from src.nsg_eval import sweep_nsg_GK_and_search_L
from src.output_utils import check_output_conflict, write_sweep_csv
from src.paths import OUTPUT_ROOT, output_dir_for

KEY_OUTPUT_FILES = (
    "comparison.csv",
    "hnsw_sweep.csv",
    "ivf_sweep.csv",
    "nsg_sweep.csv",
    "best_configs.csv",
)


def proposal_alignment_block() -> dict:
    return {
        "summary": "ANN benchmark results on QQP, CIFAR10, and UCR time-series.",
        "algorithms": [
            "NSG (GK sweep × search_L sweep)",
            "HNSW (ef_search sweep)",
            "IVF (nprobe sweep)",
        ],
        "datasets": [
            "QQP — 768-d Sentence-BERT (all-mpnet-base-v2) on Dataset/questions.csv",
            "CIFAR-10 — 512-d ImageNet CNN embeddings via torchvision",
            "UCR time series — local UCR dataset from Dataset/ucr_cache ('*_TRAIN.txt', '*_TEST.txt'); auto-picks largest TRAIN split when --ucr-name is omitted",
        ],
        "evaluation_axes": [
            "Accuracy — recall@K and MRR vs exact FAISS IndexFlatL2 neighbors",
            "Query speed — mean/p95 latency and QPS",
            "Memory — RSS after build; serialized index size (bytes)",
            "Scalability — index build_time_s vs corpus size n_index (scalability.csv when --scalability)",
        ],
        "metrics_emitted_per_dataset": [
            "recall@1, @5, @10, @20, @50, @K_max (subset depending on K)",
            "MRR (rank of first exact neighbor)",
            "latency_ms: mean, std, p50, p95, p99; QPS",
            "Per method: hnsw_sweep.csv, ivf_sweep.csv, nsg_sweep.csv",
            "comparison.csv — method comparison on the same queries",
            "best_configs.* — best point per method under recall@10-first selection",
            "Output/all_datasets_comparison.csv — stacked results across datasets from one run",
            "Output/final_summary.* — best rows across datasets",
        ],
    }


def run_one_dataset(
    slug: str,
    X_index: np.ndarray,
    X_query: np.ndarray,
    data_meta: dict,
    ef_list: list[int],
    nprobe_list: list[int],
    nsg_search_L_list: list[int],
    nsg_gk_list: list[int],
    k_neighbors: int,
    M: int,
    ef_construction: int,
    query_batch_size: int,
    run_ivf: bool,
    run_nsg: bool,
    overwrite: bool,
    *,
    emit_files: bool = True,
) -> tuple[list, list, list, dict]:
    """Run one dataset and optionally write files."""
    out = output_dir_for(slug)
    if emit_files:
        check_output_conflict(out, overwrite, KEY_OUTPUT_FILES)

    k = int(min(k_neighbors, X_index.shape[0]))
    neighbors_true = exact_neighbors(X_index, X_query, k=k)

    results = sweep_ef_search(
        slug,
        X_index,
        X_query,
        neighbors_true,
        ef_list=ef_list,
        M=M,
        ef_construction=ef_construction,
        query_batch_size=query_batch_size,
    )

    ivf_results: list = []
    if run_ivf:
        ivf_results = sweep_ivf_nprobe(
            slug,
            X_index,
            X_query,
            neighbors_true,
            nprobe_list=nprobe_list,
            query_batch_size=query_batch_size,
        )

    nsg_results: list = []
    if run_nsg:
        gks = [int(g) for g in nsg_gk_list]
        nsg_results = sweep_nsg_GK_and_search_L(
            slug,
            X_index,
            X_query,
            neighbors_true,
            search_L_list=nsg_search_L_list,
            gk_list=gks,
            query_batch_size=query_batch_size,
        )

    if emit_files:
        write_sweep_csv(out / "hnsw_sweep.csv", results)
        if ivf_results:
            write_sweep_csv(out / "ivf_sweep.csv", ivf_results)
        if nsg_results:
            write_sweep_csv(out / "nsg_sweep.csv", nsg_results)

        write_comparison_bundle(out, slug, data_meta, results, ivf_results, nsg_results)

        bc = best_from_results(results, ivf_results, nsg_results)
        write_best_configs_bundle(out, bc, data_meta)

        print(f"Wrote results to {out}")

    return results, ivf_results, nsg_results, data_meta


def _parse_scalability_sizes(spec: str, quick: bool) -> list[str]:
    parts = [p.strip().lower() for p in spec.split(",") if p.strip()]
    if not parts:
        parts = ["5000", "10000", "25000", "max"]
    if quick and spec.strip() in ("", "5000,10000,25000,max"):
        parts = ["5000"]
    return parts


def _qqp_cap_for_tag(tag: str, quick: bool) -> int | None:
    if tag == "max":
        return None if not quick else 5000
    return int(tag)


def _cifar_cap_for_tag(tag: str, quick: bool) -> int | None:
    if tag == "max":
        return None if not quick else 5000
    return int(tag)


def _ucr_cap_for_tag(tag: str, quick: bool) -> int | None:
    if tag == "max":
        return None if not quick else None
    return int(tag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick tier. Caps QQP to 5000 and CIFAR train to 5000 (unless overridden).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs in place.")
    parser.add_argument(
        "--scalability",
        action="store_true",
        help="Run multiple corpus sizes per dataset.",
    )
    parser.add_argument(
        "--scalability-sizes",
        type=str,
        default="5000,10000,25000,max",
        help='Comma-separated sizes (ints or "max").',
    )
    parser.add_argument(
        "--qqp-max-unique",
        type=int,
        default=None,
        help="QQP unique cap for non-scalability runs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["QQP", "CIFAR10", "UCR_TimeSeries"],
        choices=["QQP", "CIFAR10", "UCR_TimeSeries"],
    )
    parser.add_argument("--k-neighbors", type=int, default=100)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-list", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument(
        "--cifar-backbone",
        choices=("resnet18", "resnet50"),
        default="resnet18",
        help="Backbone for CIFAR embeddings.",
    )
    parser.add_argument(
        "--cifar-max-train",
        type=int,
        default=None,
        metavar="N",
        help="Use at most N CIFAR train images for embeddings.",
    )
    parser.add_argument(
        "--nprobe-list",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="IVF nprobe sweep list.",
    )
    parser.add_argument("--skip-ivf", action="store_true", help="Skip IVF.")
    parser.add_argument(
        "--nsg-search-L-list",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="NSG search_L sweep list.",
    )
    parser.add_argument(
        "--nsg-gk-list",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="NSG GK sweep list.",
    )
    parser.add_argument("--skip-nsg", action="store_true", help="Skip NSG.")
    parser.add_argument(
        "--ucr-name",
        type=str,
        default="",
        help="UCR dataset name from Dataset/ucr_cache; empty = largest local TRAIN split.",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    max_q_default = 5000 if args.quick else 25_000
    max_q = args.qqp_max_unique if args.qqp_max_unique is not None else max_q_default

    def cifar_train_cap_default() -> int | None:
        if args.cifar_max_train is not None:
            return int(args.cifar_max_train)
        return 5000 if args.quick else None

    tier = "quick_not_final" if args.quick else "full"
    if args.quick:
        print(
            "[experiment] --quick enabled: results are labeled quick_not_final and are not the course-grade final run."
        )

    size_tags = _parse_scalability_sizes(args.scalability_sizes, args.quick)
    if args.scalability:
        print(f"[experiment] scalability sizes: {size_tags}")

    def enrich_meta(base: dict, *, corpus_tag: str, n_index: int) -> dict:
        return {
            **base,
            "experiment_tier": tier,
            "corpus_size_target": corpus_tag,
            "n_index_effective": int(n_index),
        }

    def run_qqp(corpus_tag: str, cap: int | None, emit: bool) -> tuple[list, list, list, dict]:
        from src.data_qqp import load_qqp_embeddings

        Xi, Xq, meta = load_qqp_embeddings(
            max_unique_questions=cap,
            model_name="sentence-transformers/all-mpnet-base-v2",
        )
        meta = enrich_meta(meta, corpus_tag=corpus_tag, n_index=Xi.shape[0])
        return run_one_dataset(
            "QQP",
            Xi,
            Xq,
            meta,
            ef_list=args.ef_list,
            nprobe_list=args.nprobe_list,
            nsg_search_L_list=args.nsg_search_L_list,
            nsg_gk_list=args.nsg_gk_list,
            k_neighbors=args.k_neighbors,
            M=args.M,
            ef_construction=args.ef_construction,
            query_batch_size=args.query_batch_size,
            run_ivf=not args.skip_ivf,
            run_nsg=not args.skip_nsg,
            overwrite=args.overwrite,
            emit_files=emit,
        )

    def run_cifar(corpus_tag: str, train_cap: int | None, emit: bool) -> tuple[list, list, list, dict]:
        from src.data_cifar import load_cifar10_embeddings_512

        Xi, Xq, meta = load_cifar10_embeddings_512(
            backbone=args.cifar_backbone,
            max_train_samples=train_cap,
        )
        meta = enrich_meta(meta, corpus_tag=corpus_tag, n_index=Xi.shape[0])
        return run_one_dataset(
            "CIFAR10",
            Xi,
            Xq,
            meta,
            ef_list=args.ef_list,
            nprobe_list=args.nprobe_list,
            nsg_search_L_list=args.nsg_search_L_list,
            nsg_gk_list=args.nsg_gk_list,
            k_neighbors=min(args.k_neighbors, Xi.shape[0]),
            M=args.M,
            ef_construction=args.ef_construction,
            query_batch_size=max(args.query_batch_size, 64),
            run_ivf=not args.skip_ivf,
            run_nsg=not args.skip_nsg,
            overwrite=args.overwrite,
            emit_files=emit,
        )

    def run_ucr(corpus_tag: str, train_cap: int | None, emit: bool) -> tuple[list, list, list, dict]:
        from src.data_ucr import load_ucr_local_vectors

        Xi, Xq, meta = load_ucr_local_vectors(
            dataset_name=args.ucr_name or None,
            max_train_samples=train_cap,
        )
        meta = enrich_meta(meta, corpus_tag=corpus_tag, n_index=Xi.shape[0])
        gks = [min(g, max(4, Xi.shape[0] // 3)) for g in args.nsg_gk_list]
        return run_one_dataset(
            "UCR_TimeSeries",
            Xi,
            Xq,
            meta,
            ef_list=args.ef_list,
            nprobe_list=args.nprobe_list,
            nsg_search_L_list=args.nsg_search_L_list,
            nsg_gk_list=gks,
            k_neighbors=min(args.k_neighbors, Xi.shape[0]),
            M=args.M,
            ef_construction=args.ef_construction,
            query_batch_size=8,
            run_ivf=not args.skip_ivf,
            run_nsg=not args.skip_nsg,
            overwrite=args.overwrite,
            emit_files=emit,
        )

    for ds in args.datasets:
        if not args.scalability:
            if ds == "QQP":
                run_qqp(corpus_tag=str(max_q), cap=max_q, emit=True)
            elif ds == "CIFAR10":
                tr_cifar = cifar_train_cap_default()
                lbl_cifar = "50000_full_train" if tr_cifar is None else str(tr_cifar)
                run_cifar(corpus_tag=lbl_cifar, train_cap=tr_cifar, emit=True)
            else:
                run_ucr(corpus_tag="full_train", train_cap=None, emit=True)
            continue

        scal_rows: list[dict[str, Any]] = []
        last_meta: dict[str, Any] = {}
        last_tag = size_tags[-1]

        for tag in size_tags:
            is_last = tag == last_tag
            emit = is_last
            if ds == "QQP":
                cap = _qqp_cap_for_tag(tag, args.quick)
                h, iv, ns, meta_last = run_qqp(corpus_tag=tag, cap=cap, emit=emit)
            elif ds == "CIFAR10":
                tr = _cifar_cap_for_tag(tag, args.quick)
                lbl = "50000_full_train" if tr is None else str(tr)
                h, iv, ns, meta_last = run_cifar(corpus_tag=lbl, train_cap=tr, emit=emit)
            else:
                tr = _ucr_cap_for_tag(tag, args.quick)
                lbl = "full_train" if tr is None else str(tr)
                h, iv, ns, meta_last = run_ucr(corpus_tag=lbl, train_cap=tr, emit=emit)

            rows = [comparison_dict(r) for r in h + iv + ns]
            picked = select_best_per_family(rows)
            for fam_key, row in picked.items():
                if row is None:
                    continue
                scal_rows.append(
                    {
                        "dataset_slug": ds,
                        "scalability_size_tag": tag,
                        "method_family": fam_key,
                        "n_index": row.get("n_index", ""),
                        **{k: row[k] for k in row if k not in ("dataset",)},
                    }
                )
            if is_last:
                last_meta = {
                    **meta_last,
                    "dataset_slug": ds,
                    "scalability_ran": True,
                    "scalability_sizes": size_tags,
                }

        out = output_dir_for(ds)
        write_scalability_bundle(out, scal_rows, last_meta)

    write_cross_dataset_summary(OUTPUT_ROOT, list(args.datasets))
    write_final_summary(OUTPUT_ROOT, list(args.datasets))


if __name__ == "__main__":
    main()
