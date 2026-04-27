from __future__ import annotations

import os
import tempfile
from typing import Any

import hnswlib
import numpy as np
import psutil

import time

from .metrics import (
    ExperimentResult,
    mean_reciprocal_rank,
    recall_at_k,
    latency_stats,
)


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def exact_neighbors(
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
) -> np.ndarray:
    """Exact L2 top-k via FAISS flat index."""
    import faiss

    try:
        faiss.omp_set_num_threads(1)
    except AttributeError:
        pass
    d = index_vectors.shape[1]
    k = int(min(k, index_vectors.shape[0]))
    if k < 1:
        raise ValueError("k must be >= 1 and index must be non-empty")

    x = np.ascontiguousarray(index_vectors.astype(np.float32))
    q = np.ascontiguousarray(query_vectors.astype(np.float32))
    index = faiss.IndexFlatL2(d)
    index.add(x)
    _distances, neighbors = index.search(q, k)
    return neighbors.astype(np.int64)


def run_hnsw_experiment(
    dataset_name: str,
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    neighbors_true: np.ndarray,
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 64,
    space: str = "l2",
    query_batch_size: int = 32,
) -> ExperimentResult:
    dim = index_vectors.shape[1]
    n_index = index_vectors.shape[0]
    n_queries = query_vectors.shape[0]
    k_gt = neighbors_true.shape[1]

    data = np.ascontiguousarray(index_vectors.astype(np.float32))
    queries = np.ascontiguousarray(query_vectors.astype(np.float32))

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n_index, ef_construction=ef_construction, M=M)
    # Use 1 thread: num_threads=-1 can stall on some macOS + OpenMP stacks.
    index.add_items(data, num_threads=1)
    build_time = time.perf_counter() - t0
    rss_after = _rss_mb()

    index.set_ef(ef_search)
    lat = []
    neighbors_pred = np.empty((n_queries, k_gt), dtype=np.int64)
    bs = max(1, int(query_batch_size))
    for start in range(0, n_queries, bs):
        end = min(start + bs, n_queries)
        tq0 = time.perf_counter()
        labels, _dist = index.knn_query(queries[start:end], k=k_gt)
        dt = time.perf_counter() - tq0
        per = dt / (end - start)
        lat.extend([per] * (end - start))
        neighbors_pred[start:end] = labels

    ls = latency_stats(np.array(lat, dtype=np.float64))
    ks = sorted({1, 5, 10, min(20, k_gt), min(50, k_gt), k_gt})
    ks = [k for k in ks if k <= k_gt and k >= 1]
    rec = recall_at_k(neighbors_pred, neighbors_true, ks)
    mrr = mean_reciprocal_rank(neighbors_pred, neighbors_true, k_gt)

    file_bytes: int | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp_path = tmp.name
        index.save_index(tmp_path)
        file_bytes = os.path.getsize(tmp_path)
        os.unlink(tmp_path)
    except OSError:
        file_bytes = None

    return ExperimentResult(
        dataset=dataset_name,
        method="HNSW (hnswlib)",
        dim=dim,
        n_index=n_index,
        n_queries=n_queries,
        k_ground_truth=k_gt,
        hnsw_M=M,
        hnsw_ef_construction=ef_construction,
        hnsw_ef_search=ef_search,
        build_time_s=build_time,
        index_file_bytes=file_bytes,
        memory_rss_after_build_mb=float(rss_after),
        latency=ls,
        recall_at_k=rec,
        mrr=mrr,
        extra={
            "rss_delta_vs_before_build_mb": float(rss_after - rss_before),
            "query_batch_size": bs,
            "note": "HNSW graph index results.",
        },
    )


def sweep_ef_search(
    dataset_name: str,
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    neighbors_true: np.ndarray,
    ef_list: list[int],
    M: int = 16,
    ef_construction: int = 200,
    query_batch_size: int = 32,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for ef in ef_list:
        results.append(
            run_hnsw_experiment(
                dataset_name,
                index_vectors,
                query_vectors,
                neighbors_true,
                M=M,
                ef_construction=ef_construction,
                ef_search=ef,
                query_batch_size=query_batch_size,
            )
        )
    return results
