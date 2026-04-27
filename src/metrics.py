"""
Evaluation metrics aligned with the project proposal:
- Accuracy: recall@K, mean reciprocal rank (MRR)
- Query speed: latency (mean, std, percentiles), QPS
- Memory: RSS delta / index serialization size where applicable
- Build/index time for scalability reporting
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class LatencyStats:
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float


@dataclass
class ExperimentResult:
    dataset: str
    method: str
    dim: int
    n_index: int
    n_queries: int
    k_ground_truth: int
    hnsw_M: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    build_time_s: float
    index_file_bytes: int | None
    memory_rss_after_build_mb: float | None
    latency: LatencyStats
    recall_at_k: dict[str, float]
    mrr: float
    extra: dict[str, Any] = field(default_factory=dict)
    # IVF (FAISS IndexIVFFlat); None when method is HNSW
    ivf_nlist: int | None = None
    ivf_nprobe: int | None = None
    # NSG (FAISS IndexNSGFlat, Fu et al. VLDB 2019); None when not NSG
    nsg_search_L: int | None = None
    nsg_GK: int | None = None


def latency_stats(latencies_s: np.ndarray) -> LatencyStats:
    ms = latencies_s.astype(np.float64) * 1000.0
    total = float(ms.sum() / 1000.0)
    qps = len(ms) / total if total > 0 else 0.0
    return LatencyStats(
        mean_ms=float(ms.mean()),
        std_ms=float(ms.std()),
        p50_ms=float(np.percentile(ms, 50)),
        p95_ms=float(np.percentile(ms, 95)),
        p99_ms=float(np.percentile(ms, 99)),
        qps=float(qps),
    )


def recall_at_k(
    neighbors_pred: np.ndarray,
    neighbors_true: np.ndarray,
    ks: list[int],
) -> dict[str, float]:
    """
    neighbors_pred, neighbors_true: shape (n_queries, k_max), int indices.
    Recall@k = |pred[:k] ∩ true[:k]| / k  (standard ANN benchmark convention for same k).
    """
    n, k_max = neighbors_true.shape
    out: dict[str, float] = {}
    for k in ks:
        kk = min(k, k_max)
        hits = 0
        for i in range(n):
            true_set = set(neighbors_true[i, :kk].tolist())
            pred_set = set(neighbors_pred[i, :kk].tolist())
            hits += len(true_set & pred_set)
        out[f"recall@{k}"] = hits / (n * kk) if kk else 0.0
    return out


def mean_reciprocal_rank(
    neighbors_pred: np.ndarray,
    neighbors_true: np.ndarray,
    k_first_true: int,
) -> float:
    """MRR for the single nearest neighbor in ground truth (first column)."""
    primary = neighbors_true[:, 0]
    mrr = 0.0
    n = len(primary)
    for i in range(n):
        tgt = primary[i]
        row = neighbors_pred[i]
        rank = np.where(row == tgt)[0]
        if rank.size:
            mrr += 1.0 / (float(rank[0]) + 1.0)
    return mrr / n if n else 0.0


def time_block() -> tuple[float, Any]:
    t0 = time.perf_counter()
    return t0, None


def elapsed_s(t0: float) -> float:
    return time.perf_counter() - t0


def result_to_jsonable(r: ExperimentResult) -> dict[str, Any]:
    d = asdict(r)
    d["latency"] = asdict(r.latency)
    return d


def write_json(path: Any, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv_row(path: Any, header: list[str], row: list[Any], write_header: bool) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
