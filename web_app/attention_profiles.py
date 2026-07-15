from __future__ import annotations

from typing import Any

import numpy as np


def normalize_attention(values: np.ndarray | None, *, mode: str = "raw") -> np.ndarray | None:
    if values is None:
        return None

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    if not np.isfinite(arr).all():
        raise ValueError("attention values contain NaN or Inf")

    normalized_mode = mode.strip().lower()
    if normalized_mode == "raw":
        return arr
    if normalized_mode in {"global", "per-model", "per_model"}:
        span = float(arr.max() - arr.min())
        if span <= 1e-8:
            return np.full_like(arr, 0.5, dtype=np.float32)
        return (arr - arr.min()) / span
    raise ValueError(f"Unknown normalization mode: {mode}")


def combine_attention(prior: np.ndarray | None, learned: np.ndarray | None) -> np.ndarray | None:
    if prior is None or learned is None:
        return None

    prior_arr = np.asarray(prior, dtype=np.float32).reshape(-1)
    learned_arr = np.asarray(learned, dtype=np.float32).reshape(-1)
    if prior_arr.size != learned_arr.size:
        raise ValueError("prior and learned attention must have the same length")

    fused = prior_arr * learned_arr
    span = float(fused.max() - fused.min())
    if span <= 1e-8:
        return np.full_like(fused, 0.5, dtype=np.float32)
    return (fused - fused.min()) / span


def occlusion_values(visibility: np.ndarray | None) -> np.ndarray | None:
    if visibility is None:
        return None
    vis = np.asarray(visibility, dtype=np.float32).reshape(-1)
    if vis.size == 0:
        return None
    if not np.isfinite(vis).all():
        raise ValueError("visibility values contain NaN or Inf")
    return 1.0 - np.clip(vis, 0.0, 1.0)


def is_available(values: Any) -> bool:
    return values is not None and np.asarray(values).size > 0
