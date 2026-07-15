from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np


MAX_FRAMES = 64


def compute_video_hash(video_bytes: bytes) -> str:
    return hashlib.sha256(video_bytes).hexdigest()


def preprocess_signature(
    *,
    max_frames: int = MAX_FRAMES,
    pose_model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    visibility_threshold: float = 0.3,
) -> str:
    return (
        f"max_frames={max_frames}|pose_model_complexity={pose_model_complexity}|"
        f"min_detection_confidence={min_detection_confidence:.2f}|"
        f"min_tracking_confidence={min_tracking_confidence:.2f}|"
        f"visibility_threshold={visibility_threshold:.2f}"
    )


def build_frame_mapping(original_count: int, max_frames: int = MAX_FRAMES) -> dict[str, object]:
    if original_count <= 0:
        raise ValueError("original_count must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    if original_count >= max_frames:
        sampled_indices = np.linspace(0, original_count - 1, max_frames, dtype=int).tolist()
        resampled_to_source = sampled_indices
        padded_count = 0
        mapping_mode = "linspace_downsample"
    else:
        sampled_indices = list(range(original_count))
        resampled_to_source = sampled_indices + [original_count - 1] * (max_frames - original_count)
        padded_count = max_frames - original_count
        mapping_mode = "repeat_last_padding"

    return {
        "original_count": original_count,
        "max_frames": max_frames,
        "sampled_indices": sampled_indices,
        "resampled_to_source": resampled_to_source,
        "padded_count": padded_count,
        "mapping_mode": mapping_mode,
    }


@dataclass(frozen=True)
class FrameSyncInfo:
    frame_index: int
    source_frame_index: int
    timestamp_ms: float
    source_kind: str


def resolve_frame_sync_info(
    *,
    frame_index: int,
    mapping: dict[str, object],
    source_timestamps_ms: Iterable[float],
) -> FrameSyncInfo:
    resampled_to_source = list(mapping["resampled_to_source"])
    original_count = int(mapping["original_count"])

    if frame_index < 0 or frame_index >= len(resampled_to_source):
        raise ValueError(f"frame_index must be within 0..{len(resampled_to_source) - 1}")

    source_frame_index = int(resampled_to_source[frame_index])
    timestamps = list(source_timestamps_ms)
    if not timestamps:
        timestamp_ms = float("nan")
    else:
        timestamp_ms = float(timestamps[min(source_frame_index, len(timestamps) - 1)])

    source_kind = "padded" if original_count < int(mapping["max_frames"]) and frame_index >= original_count else "sampled"
    return FrameSyncInfo(
        frame_index=frame_index,
        source_frame_index=source_frame_index,
        timestamp_ms=timestamp_ms,
        source_kind=source_kind,
    )


def frame_cache_key(video_hash: str, preprocess_key: str, frame_index: int) -> str:
    return f"{video_hash}:{preprocess_key}:frame={frame_index:02d}"