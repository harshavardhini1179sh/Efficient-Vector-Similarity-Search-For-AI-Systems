"""CIFAR-10 → 512-d ImageNet ResNet GAP embeddings (proposal: 512-d ResNet-50 on 60k images, 10 classes)."""
from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from .paths import DATASET_DIR


def load_cifar10_embeddings_512(
    batch_size: int = 256,
    num_workers: int = 0,
    random_state: int = 42,
    backbone: str = "resnet18",
    max_train_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Native 512-d GAP features via ResNet-18 (fc removed). ResNet-50 is heavier on CPU;
    meta documents both the proposal target and the backbone used here.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    root = str(DATASET_DIR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=getattr(np, "VisibleDeprecationWarning", UserWarning))
        warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.datasets\.cifar")
        train_set = datasets.CIFAR10(
            root=root, train=True, download=True, transform=tfm
        )
        test_set = datasets.CIFAR10(
            root=root, train=False, download=True, transform=tfm
        )

    n_train = len(train_set)
    if max_train_samples is not None and max_train_samples < n_train:
        rng_sub = np.random.default_rng(random_state)
        pick_tr = rng_sub.choice(n_train, size=int(max_train_samples), replace=False)
        train_set = Subset(train_set, pick_tr.tolist())

    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(backbone)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()

    def embed_split(ds, shuffle: bool) -> np.ndarray:
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        feats: list[np.ndarray] = []
        with torch.no_grad():
            for x, _y in loader:
                x = x.to(device)
                o = net(x).cpu().numpy().astype(np.float32)
                feats.append(o)
        return np.concatenate(feats, axis=0)

    X_index = embed_split(train_set, shuffle=False)
    Xq_full = embed_split(test_set, shuffle=False)

    rng = np.random.default_rng(random_state)
    max_queries = 2000
    if Xq_full.shape[0] > max_queries:
        pick = rng.choice(Xq_full.shape[0], size=max_queries, replace=False)
        X_query = Xq_full[pick]
    else:
        X_query = Xq_full

    meta = {
        "dim": int(X_index.shape[1]),
        "n_index": len(X_index),
        "n_queries": len(X_query),
        "n_test_full": int(Xq_full.shape[0]),
        "cifar_train_cap": max_train_samples,
        "backbone": backbone,
        "proposal": (
            "CIFAR-10: 60,000 32×32 images, 10 classes; proposal text cites 512-d ResNet-50 embeddings. "
            f"This run uses ImageNet-pretrained {backbone} global-average-pool vectors (512-d) for CPU-friendly "
            "reproducibility; swap backbone='resnet50' for the heavier variant."
        ),
    }
    return X_index, X_query, meta
