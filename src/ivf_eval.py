"""FAISS IVF (Inverted File + flat quantizer)."""
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


def _pick_nlist(n_index: int) -> int:
    """Heuristic nlist for IVFFlat; FAISS recommends ~39 * nlist training points for k-means."""
    if n_index < 16:
        nl = max(1, n_index // 4)
    else:
        nl = int(min(4096, max(4, round(np.sqrt(n_index)))))
    max_lists = max(1, n_index // 39)
    return max(1, min(nl, max_lists))


def _build_ivf_index(
    x: np.ndarray, nl: int
) -> tuple[faiss.IndexIVFFlat, float, float, float]:
    """Train + add; returns (index, build_time_s, rss_before, rss_after)."""
    dim = x.shape[1]
    n_index = x.shape[0]
    nl = max(1, min(nl, max(1, n_index - 1)))

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nl, faiss.METRIC_L2)

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    if not index.is_trained:
        index.train(x)
    index.add(x)
    build_time = time.perf_counter() - t0
    rss_after = _rss_mb()
    return index, build_time, rss_before, rss_after


def run_ivf_experiment(
    dataset_name: str,
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    neighbors_true: np.ndarray,
    nlist: int | None = None,
    nprobe: int = 8,
    query_batch_size: int = 32,
    *,
    _index: faiss.IndexIVFFlat | None = None,
    _build_time: float | None = None,
    _rss_before: float | None = None,
    _rss_after: float | None = None,
    _index_bytes: int | None = None,
) -> ExperimentResult:
    dim = index_vectors.shape[1]
    n_index = index_vectors.shape[0]
    n_queries = query_vectors.shape[0]
    k_gt = neighbors_true.shape[1]

    x = np.ascontiguousarray(index_vectors.astype(np.float32))
    q = np.ascontiguousarray(query_vectors.astype(np.float32))

    nl = nlist if nlist is not None else _pick_nlist(n_index)
    nl = max(1, min(nl, max(1, n_index - 1)))

    if _index is not None:
        index = _index
        build_time = float(_build_time or 0.0)
        rss_before = float(_rss_before or _rss_mb())
        rss_after = float(_rss_after or _rss_mb())
        file_bytes = _index_bytes
    else:
        index, build_time, rss_before, rss_after = _build_ivf_index(x, nl)
        file_bytes = None

    if file_bytes is None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tmp:
                tmp_path = tmp.name
            faiss.write_index(index, tmp_path)
            file_bytes = os.path.getsize(tmp_path)
            os.unlink(tmp_path)
        except OSError:
            file_bytes = None

    index.nprobe = int(min(max(1, nprobe), index.nlist))

    neighbors_pred = np.empty((n_queries, k_gt), dtype=np.int64)
    lat: list[float] = []
    bs = max(1, int(query_batch_size))
    for start in range(0, n_queries, bs):
        end = min(start + bs, n_queries)
        tq0 = time.perf_counter()
        _d, ids = index.search(q[start:end], k_gt)
        dt = time.perf_counter() - tq0
        per = dt / (end - start)
        lat.extend([per] * (end - start))
        neighbors_pred[start:end] = ids.astype(np.int64)

    ls = latency_stats(np.array(lat, dtype=np.float64))
    ks = sorted({1, 5, 10, min(20, k_gt), min(50, k_gt), k_gt})
    ks = [k for k in ks if k <= k_gt and k >= 1]
    rec = recall_at_k(neighbors_pred, neighbors_true, ks)
    mrr = mean_reciprocal_rank(neighbors_pred, neighbors_true, k_gt)

    return ExperimentResult(
        dataset=dataset_name,
        method="IVF (FAISS IndexIVFFlat)",
        dim=dim,
        n_index=n_index,
        n_queries=n_queries,
        k_ground_truth=k_gt,
        hnsw_M=0,
        hnsw_ef_construction=0,
        hnsw_ef_search=0,
        build_time_s=build_time,
        index_file_bytes=file_bytes,
        memory_rss_after_build_mb=float(rss_after),
        latency=ls,
        recall_at_k=rec,
        mrr=mrr,
        ivf_nlist=index.nlist,
        ivf_nprobe=index.nprobe,
        extra={
            "rss_delta_vs_before_build_mb": float(rss_after - rss_before),
            "query_batch_size": bs,
            "note": "IVF index results.",
        },
    )


def sweep_ivf_nprobe(
    dataset_name: str,
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    neighbors_true: np.ndarray,
    nprobe_list: list[int],
    nlist: int | None = None,
    query_batch_size: int = 32,
) -> list[ExperimentResult]:
    x = np.ascontiguousarray(index_vectors.astype(np.float32))
    nl = nlist if nlist is not None else _pick_nlist(x.shape[0])
    nl = max(1, min(nl, max(1, x.shape[0] - 1)))

    index, build_time, rss_before, rss_after = _build_ivf_index(x, nl)

    file_bytes: int | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tmp:
            tmp_path = tmp.name
        faiss.write_index(index, tmp_path)
        file_bytes = os.path.getsize(tmp_path)
        os.unlink(tmp_path)
    except OSError:
        file_bytes = None

    out: list[ExperimentResult] = []
    for np_ in nprobe_list:
        capped = min(max(1, np_), index.nlist)
        out.append(
            run_ivf_experiment(
                dataset_name,
                index_vectors,
                query_vectors,
                neighbors_true,
                nlist=nl,
                nprobe=capped,
                query_batch_size=query_batch_size,
                _index=index,
                _build_time=build_time,
                _rss_before=rss_before,
                _rss_after=rss_after,
                _index_bytes=file_bytes,
            )
        )
    return out
