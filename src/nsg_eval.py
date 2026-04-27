"""NSG (Navigating Spreading-out Graph) via FAISS IndexNSGFlat."""
from __future__ import annotations

import os
import tempfile
import time

import faiss
import numpy as np
import psutil

from .metrics import ExperimentResult, latency_stats, mean_reciprocal_rank, recall_at_k


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def _knn_graph_matrix(x: np.ndarray, k_graph: int) -> np.ndarray:
    """Build a valid exact k-NN graph for NSG."""
    n = x.shape[0]
    if n < 2:
        raise ValueError("NSG knn graph needs n >= 2")
    k_graph = int(min(k_graph, n - 1))
    k_buf = int(min(max(k_graph + 8, k_graph + 1), n))
    _, I = faiss.knn(x, x, k_buf, faiss.METRIC_L2)
    out = np.zeros((n, k_graph), dtype=np.int64)
    for i in range(n):
        seen: set[int] = set()
        take: list[int] = []
        for nb in I[i]:
            v = int(nb)
            if v < 0 or v >= n or v == i or v in seen:
                continue
            seen.add(v)
            take.append(v)
            if len(take) >= k_graph:
                break
        if len(take) < k_graph:
            for c in range(n):
                if c == i or c in seen:
                    continue
                take.append(c)
                seen.add(c)
                if len(take) >= k_graph:
                    break
        if len(take) < k_graph:
            raise ValueError(f"row {i}: could not fill {k_graph} valid neighbors (n={n})")
        out[i] = np.array(take[:k_graph], dtype=np.int64)
    return out


def build_nsg_index(
    x: np.ndarray,
    GK: int = 32,
    k_graph: int = 40,
    nndescent_threshold: int = 2048,
) -> tuple[faiss.IndexNSGFlat, float, float, float, int | None]:
    """Build NSG index and return timing/memory stats."""
    x = np.ascontiguousarray(x.astype(np.float32))
    n, d = x.shape
    gk_max = max(4, n // 3) if n < 256 else (n - 1)
    GK = int(max(4, min(GK, gk_max)))
    k_graph = int(max(GK, min(k_graph, max(GK, n - 1))))

    index = faiss.IndexNSGFlat(d, GK)
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    if n <= nndescent_threshold:
        index.build_type = 0
        graph = _knn_graph_matrix(x, k_graph)
    else:
        index.build_type = 1
        index.nndescent_R = 64
        index.nndescent_L = 200
        index.nndescent_iter = 12
        index.nndescent_S = 15
        graph = np.zeros((n, k_graph), dtype=np.int64)
    index.build(x, graph)
    build_time = time.perf_counter() - t0
    rss_after = _rss_mb()

    file_bytes: int | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tmp:
            tmp_path = tmp.name
        faiss.write_index(index, tmp_path)
        file_bytes = os.path.getsize(tmp_path)
        os.unlink(tmp_path)
    except OSError:
        file_bytes = None

    return index, build_time, rss_before, rss_after, file_bytes


def run_nsg_experiment(
    dataset_name: str,
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    neighbors_true: np.ndarray,
    search_L: int,
    query_batch_size: int = 32,
    *,
    _index: faiss.IndexNSGFlat | None = None,
    _build_time: float | None = None,
    _rss_before: float | None = None,
    _rss_after: float | None = None,
    _file_bytes: int | None = None,
    _GK: int | None = None,
) -> ExperimentResult:
    dim = index_vectors.shape[1]
    n_index = index_vectors.shape[0]
    n_queries = query_vectors.shape[0]
    k_gt = neighbors_true.shape[1]
    q = np.ascontiguousarray(query_vectors.astype(np.float32))

    if _index is None:
        idx, build_time, rb, ra, fb = build_nsg_index(index_vectors)
        GK = int(idx.GK)
    else:
        idx = _index
        build_time = float(_build_time or 0.0)
        rb = float(_rss_before or _rss_mb())
        ra = float(_rss_after or _rss_mb())
        fb = _file_bytes
        GK = int(_GK or idx.GK)

    idx.nsg.search_L = int(max(1, search_L))

    neighbors_pred = np.empty((n_queries, k_gt), dtype=np.int64)
    lat: list[float] = []
    bs = max(1, int(query_batch_size))
    for start in range(0, n_queries, bs):
        end = min(start + bs, n_queries)
        tq0 = time.perf_counter()
        D, I = idx.search(q[start:end], k_gt)
        dt = time.perf_counter() - tq0
        per = dt / (end - start)
        lat.extend([per] * (end - start))
        neighbors_pred[start:end] = I.astype(np.int64)

    ls = latency_stats(np.array(lat, dtype=np.float64))
    ks = sorted({1, 5, 10, min(20, k_gt), min(50, k_gt), k_gt})
    ks = [k for k in ks if k <= k_gt and k >= 1]
    rec = recall_at_k(neighbors_pred, neighbors_true, ks)
    mrr = mean_reciprocal_rank(neighbors_pred, neighbors_true, k_gt)

    return ExperimentResult(
        dataset=dataset_name,
        method="NSG (FAISS IndexNSGFlat)",
        dim=dim,
        n_index=n_index,
        n_queries=n_queries,
        k_ground_truth=k_gt,
        hnsw_M=0,
        hnsw_ef_construction=0,
        hnsw_ef_search=0,
        build_time_s=build_time,
        index_file_bytes=fb,
        memory_rss_after_build_mb=float(ra),
        latency=ls,
        recall_at_k=rec,
        mrr=mrr,
        ivf_nlist=None,
        ivf_nprobe=None,
        nsg_search_L=idx.nsg.search_L,
        nsg_GK=GK,
        extra={
            "rss_delta_vs_before_build_mb": float(ra - rb),
            "query_batch_size": bs,
            "note": "NSG graph index results.",
        },
    )


def _gk_feasible_for_n(GK: int, n: int) -> bool:
    """FAISS NSG needs GK <= n-1; tiny corpora also cap GK in build_nsg_index."""
    if n < 5:
        return False
    gk_max = max(4, n // 3) if n < 256 else (n - 1)
    gk_use = int(max(4, min(int(GK), gk_max)))
    return gk_use == int(GK)


def sweep_nsg_GK_and_search_L(
    dataset_name: str,
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    neighbors_true: np.ndarray,
    search_L_list: list[int],
    gk_list: list[int],
    query_batch_size: int = 32,
) -> list[ExperimentResult]:
    """Sweep NSG GK and search_L."""
    n = int(index_vectors.shape[0])
    out: list[ExperimentResult] = []
    tried_gk: list[int] = []
    for GK in sorted(set(int(g) for g in gk_list)):
        if not _gk_feasible_for_n(GK, n):
            continue
        tried_gk.append(GK)
        idx, build_time, rb, ra, fb = build_nsg_index(index_vectors, GK=GK)
        for L in search_L_list:
            out.append(
                run_nsg_experiment(
                    dataset_name,
                    index_vectors,
                    query_vectors,
                    neighbors_true,
                    search_L=L,
                    query_batch_size=query_batch_size,
                    _index=idx,
                    _build_time=build_time,
                    _rss_before=rb,
                    _rss_after=ra,
                    _file_bytes=fb,
                    _GK=GK,
                )
            )
    if not tried_gk and gk_list:
        GK0 = int(max(gk_list))
        idx, build_time, rb, ra, fb = build_nsg_index(index_vectors, GK=GK0)
        for L in search_L_list:
            out.append(
                run_nsg_experiment(
                    dataset_name,
                    index_vectors,
                    query_vectors,
                    neighbors_true,
                    search_L=L,
                    query_batch_size=query_batch_size,
                    _index=idx,
                    _build_time=build_time,
                    _rss_before=rb,
                    _rss_after=ra,
                    _file_bytes=fb,
                    _GK=int(idx.GK),
                )
            )
    return out


def sweep_nsg_search_L(
    dataset_name: str,
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    neighbors_true: np.ndarray,
    search_L_list: list[int],
    query_batch_size: int = 32,
    GK: int = 32,
) -> list[ExperimentResult]:
    return sweep_nsg_GK_and_search_L(
        dataset_name,
        index_vectors,
        query_vectors,
        neighbors_true,
        search_L_list,
        [GK],
        query_batch_size=query_batch_size,
    )
