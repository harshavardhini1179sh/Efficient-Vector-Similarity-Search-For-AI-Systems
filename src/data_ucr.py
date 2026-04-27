"""Load local UCR time-series datasets from Dataset/ucr_cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import DATASET_DIR


def _znorm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return ((x - m) / s).astype(np.float32)


def list_local_ucr_datasets(cache_dir: Path | None = None) -> list[str]:
    cache = (DATASET_DIR / "ucr_cache") if cache_dir is None else cache_dir
    if not cache.exists():
        return []
    names: set[str] = set()
    for p in cache.glob("*_TRAIN.txt"):
        base = p.name[: -len("_TRAIN.txt")]
        test = cache / f"{base}_TEST.txt"
        if test.exists():
            names.add(base)
    return sorted(names)


def _load_local_ucr_txt(dataset_name: str, cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_path = cache / f"{dataset_name}_TRAIN.txt"
    test_path = cache / f"{dataset_name}_TEST.txt"
    train = np.loadtxt(train_path)
    test = np.loadtxt(test_path)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] < 2 or test.shape[1] < 2:
        raise ValueError(f"Unexpected UCR txt format for {dataset_name}")
    y_tr = train[:, 0].astype(np.int64)
    x_tr = train[:, 1:].astype(np.float32)
    y_te = test[:, 0].astype(np.int64)
    x_te = test[:, 1:].astype(np.float32)
    return x_tr, y_tr, x_te, y_te


def _train_row_count(cache: Path, name: str) -> int:
    train_path = cache / f"{name}_TRAIN.txt"
    if not train_path.exists():
        return 0
    t = np.loadtxt(train_path)
    if t.ndim == 1:
        return 1
    return int(t.shape[0])


def pick_largest_ucr_dataset(cache_dir: Path | None = None) -> tuple[str, dict]:
    cache = (DATASET_DIR / "ucr_cache") if cache_dir is None else cache_dir
    names = list_local_ucr_datasets(cache)
    if not names:
        raise FileNotFoundError(
            f"No local UCR datasets found in {cache}. "
            "Expected files like <NAME>_TRAIN.txt and <NAME>_TEST.txt."
        )
    best = max(names, key=lambda n: _train_row_count(cache, n))
    meta = {
        "chosen_reason": "largest_TRAIN_split_in_cache",
        "n_train": _train_row_count(cache, best),
        "available_datasets": names,
    }
    return best, meta


def load_ucr_local_vectors(
    dataset_name: str | None = None,
    *,
    max_train_samples: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load one UCR dataset from local files:
      Dataset/ucr_cache/<NAME>_TRAIN.txt
      Dataset/ucr_cache/<NAME>_TEST.txt

    If dataset_name is empty, auto-pick the largest TRAIN split present in local cache.
    max_train_samples optionally subsamples the TRAIN index (queries unchanged) for scalability curves.
    """
    cache = DATASET_DIR / "ucr_cache"
    cache.mkdir(parents=True, exist_ok=True)
    pick_meta: dict = {}
    if dataset_name is None or dataset_name.strip() == "":
        dataset_name, pick_meta = pick_largest_ucr_dataset(cache)

    train_path = cache / f"{dataset_name}_TRAIN.txt"
    test_path = cache / f"{dataset_name}_TEST.txt"
    if not (train_path.exists() and test_path.exists()):
        available = list_local_ucr_datasets(cache)
        raise FileNotFoundError(
            "Missing local UCR dataset files for "
            f"{dataset_name}. Expected {train_path.name} and {test_path.name} in {cache}. "
            f"Available local UCR datasets: {available}"
        )

    x_tr, y_tr, x_te, y_te = _load_local_ucr_txt(dataset_name, cache)

    if max_train_samples is not None and max_train_samples < x_tr.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(x_tr.shape[0], size=int(max_train_samples), replace=False)
        x_tr = x_tr[idx]
        y_tr = y_tr[idx]

    y_all = np.concatenate([y_tr, y_te])

    x_index = _znorm(x_tr)
    x_query = _znorm(x_te)

    n_train_full = _train_row_count(cache, dataset_name)
    small_sanity = (
        dataset_name == "GunPoint"
        or x_index.shape[0] < 500
        or (n_train_full < 500 and x_index.shape[0] == n_train_full)
    )
    ucr_notes: list[str] = []
    if small_sanity:
        ucr_notes.append(
            "UCR split is small (e.g., classic GunPoint is ~50 train series). "
            "Treat this as a pipeline sanity check, not strong evidence about scalability or recall "
            "tradeoffs at thousands+ of points."
        )

    meta = {
        "dataset": "UCR_TimeSeries",
        "ucr_dataset_name": dataset_name,
        "dim": int(x_index.shape[1]),
        "n_index": int(len(x_index)),
        "n_queries": int(len(x_query)),
        "classes": int(np.max(y_all) + 1) if y_all.size else 0,
        "proposal": (
            "UCR Time Series classification repository; numeric sequences used as fixed-length "
            "embeddings after per-series z-normalization. DataSummary.csv catalogs the archive."
        ),
        "source": "local Dataset/ucr_cache/*_TRAIN.txt,*_TEST.txt",
        "ucr_selection": pick_meta,
        "ucr_warnings": ucr_notes,
    }
    return x_index, x_query, meta
