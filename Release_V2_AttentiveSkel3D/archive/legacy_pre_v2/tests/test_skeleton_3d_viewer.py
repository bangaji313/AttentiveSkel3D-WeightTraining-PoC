from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from web_app.components.skeleton_3d_viewer import (
    LANDMARK_COUNT,
    VIEW_PRESETS,
    create_3d_skeleton_figure,
    prepare_frame_payload,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_SAMPLE = ROOT / "results" / "data_integrity" / "Squat_001_raw_keypoints_tmp.npy"
PROCESSED_SAMPLE = ROOT / "data" / "processed" / "tensors" / "Squat_001.npy"


def test_prepare_frame_payload_extracts_visibility_from_real_raw_tensor():
    raw = np.load(RAW_SAMPLE, allow_pickle=False)
    coords, visibility = prepare_frame_payload(raw[0])

    assert raw.shape == (99, 33, 4)
    assert coords.shape == (33, 3)
    assert visibility.shape == (33,)
    assert np.isfinite(coords).all()
    assert np.isfinite(visibility).all()
    assert visibility.min() >= 0.0


def test_create_3d_skeleton_figure_uses_real_processed_tensor_without_mutation():
    tensor = np.load(PROCESSED_SAMPLE, allow_pickle=False)
    frame = tensor[0].copy()
    original = frame.copy()

    fig = create_3d_skeleton_figure(frame, title="Squat sample", view="isometric", normalize_display=True)

    assert frame.shape == (33, 3)
    assert np.array_equal(frame, original)
    assert len(fig.data) == 4
    assert fig.layout.scene.camera.eye.x == VIEW_PRESETS["isometric"]["eye"]["x"]
    assert fig.layout.scene.camera.eye.y == VIEW_PRESETS["isometric"]["eye"]["y"]
    assert fig.layout.scene.camera.eye.z == VIEW_PRESETS["isometric"]["eye"]["z"]


def test_create_3d_skeleton_figure_accepts_raw_frame_with_visibility_column():
    raw = np.load(RAW_SAMPLE, allow_pickle=False)
    fig = create_3d_skeleton_figure(raw[0], view="front", normalize_display=False)

    marker_trace = fig.data[-1]
    assert marker_trace.name == "Landmarks"
    assert len(marker_trace.x) == LANDMARK_COUNT
    assert marker_trace.hovertemplate is not None


def test_create_3d_skeleton_figure_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="33 landmarks"):
        create_3d_skeleton_figure(np.zeros((32, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="3 or 4 columns"):
        create_3d_skeleton_figure(np.zeros((33, 2), dtype=np.float32))


def test_create_3d_skeleton_figure_rejects_nan_and_attention_mismatch():
    bad = np.zeros((33, 3), dtype=np.float32)
    bad[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        create_3d_skeleton_figure(bad)

    with pytest.raises(ValueError, match="attention must have shape"):
        create_3d_skeleton_figure(np.zeros((33, 3), dtype=np.float32), attention=np.zeros(12, dtype=np.float32))


def test_view_presets_cover_required_orientations():
    assert {"front", "left", "right", "isometric"}.issubset(VIEW_PRESETS)
