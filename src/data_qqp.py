from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .paths import DATASET_DIR

def load_qqp_embeddings(
    max_unique_questions: int | None = 25_000,
    random_state: int = 42,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> tuple[np.ndarray, np.ndarray, dict]:
    csv_path = DATASET_DIR / "questions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path, usecols=["question1", "question2"], dtype=str, nrows=None)
    s1 = df["question1"].dropna().astype(str)
    s2 = df["question2"].dropna().astype(str)
    texts = pd.unique(pd.concat([s1, s2], ignore_index=True))
    texts = np.array([t for t in texts if isinstance(t, str) and len(t) > 0], dtype=object)
    rng = np.random.default_rng(random_state)
    n_unique_total = int(len(texts))
    if max_unique_questions is not None and len(texts) > int(max_unique_questions):
        idx = rng.choice(len(texts), size=int(max_unique_questions), replace=False)
        texts = texts[idx]

    train_texts, test_texts = train_test_split(
        texts, test_size=0.15, random_state=random_state
    )
    # Hold-out queries from disjoint pool
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_name)

    def encode_batch(strings: list[str], batch_size: int = 64) -> np.ndarray:
        return np.asarray(
            embedder.encode(
                strings,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    X_index = encode_batch(train_texts.tolist())
    X_query = encode_batch(test_texts[: min(2000, len(test_texts))].tolist())
    meta = {
        "model_name": model_name,
        "dim": X_index.shape[1],
        "n_index": len(X_index),
        "n_queries": len(X_query),
        "qqp_unique_cap": max_unique_questions,
        "qqp_unique_total_in_csv": n_unique_total,
        "proposal": "QQP ~400k pairs; 768-d Sentence-BERT embeddings (duplicate labels exist in full QQP).",
    }
    return X_index, X_query, meta
