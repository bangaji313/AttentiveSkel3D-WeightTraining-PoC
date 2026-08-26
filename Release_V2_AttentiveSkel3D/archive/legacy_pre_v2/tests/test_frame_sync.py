from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np

from web_app.frame_sync import build_frame_mapping, compute_video_hash, frame_cache_key, resolve_frame_sync_info


if "cv2" not in sys.modules:
    fake_cv2 = types.SimpleNamespace(
        VideoCapture=lambda *args, **kwargs: None,
        COLOR_BGR2RGB=1,
        LINE_AA=1,
        FONT_HERSHEY_SIMPLEX=0,
        cvtColor=lambda image, code: image,
        line=lambda *args, **kwargs: None,
        circle=lambda *args, **kwargs: None,
        putText=lambda *args, **kwargs: None,
    )
    sys.modules["cv2"] = fake_cv2

if "mediapipe" not in sys.modules:
    class _FakePose:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def process(self, rgb):
            return types.SimpleNamespace(pose_landmarks=None)

    fake_pose_module = types.SimpleNamespace(
        POSE_CONNECTIONS=[(11, 12), (11, 23), (23, 24), (23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32)],
        Pose=_FakePose,
    )
    fake_mp = types.SimpleNamespace(solutions=types.SimpleNamespace(pose=fake_pose_module))
    sys.modules["mediapipe"] = fake_mp

if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = np.ndarray
    fake_torch.FloatTensor = np.ndarray
    fake_torch.float32 = np.float32
    fake_torch.tensor = lambda data, dtype=None: np.asarray(data, dtype=np.float32 if dtype is not None else None)
    fake_torch.from_numpy = lambda array: array
    fake_torch.ones = lambda *shape, dtype=None: np.ones(shape, dtype=np.float32 if dtype is not None else np.float32)
    fake_torch.randn = lambda *shape, dtype=None: np.random.randn(*shape).astype(np.float32)
    fake_torch.load = lambda *args, **kwargs: {}
    fake_torch.no_grad = lambda: _DummyCtx()
    fake_torch.sigmoid = lambda value: value
    fake_torch.softmax = lambda value, dim=None: value

    fake_nn = types.ModuleType("torch.nn")
    class _Module:
        def __call__(self, *args, **kwargs):
            if hasattr(self, "forward"):
                return self.forward(*args, **kwargs)
            return None

        def load_state_dict(self, *args, **kwargs):
            return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def eval(self):
            return self

    class _Linear:
        def __init__(self, in_features=0, out_features=0, *args, **kwargs):
            self.in_features = in_features
            self.out_features = out_features

    class _Conv3d:
        def __init__(self, in_channels=0, out_channels=0, *args, **kwargs):
            self.in_channels = in_channels
            self.out_channels = out_channels

    class _Identity:
        def __init__(self, *args, **kwargs):
            pass

    class _Sigmoid:
        def __call__(self, value):
            return value

    fake_nn.Module = _Module
    fake_nn.Parameter = lambda value: value
    fake_nn.Sequential = lambda *args, **kwargs: list(args)
    fake_nn.Linear = _Linear
    fake_nn.ReLU = _Identity
    fake_nn.Conv3d = _Conv3d
    fake_nn.Dropout = _Identity
    fake_nn.BatchNorm3d = _Identity
    fake_nn.MaxPool3d = _Identity
    fake_nn.AdaptiveAvgPool3d = _Identity
    fake_nn.Sigmoid = _Sigmoid

    fake_torch.nn = fake_nn
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn

from web_app import app as dashboard_app


ROOT = Path(__file__).resolve().parents[1]
RAW_SAMPLE = ROOT / "results" / "data_integrity" / "Squat_001_raw_keypoints_tmp.npy"
PROCESSED_SAMPLE = ROOT / "data" / "processed" / "tensors" / "Squat_001.npy"


class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def metric(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def divider(self, *args, **kwargs):
        return None


class _FakeLm:
    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def _build_fake_landmarks(frame: np.ndarray) -> list[_FakeLm]:
    return [_FakeLm(float(pt[0]), float(pt[1]), float(vis)) for pt, vis in zip(frame[:, :3], frame[:, 3] if frame.shape[1] == 4 else np.ones(33))]


def _install_streamlit_noops(monkeypatch, slider_value: int = 7):
    def fake_columns(spec):
        count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_DummyCtx() for _ in range(count)]

    monkeypatch.setattr(dashboard_app.st, "columns", fake_columns, raising=False)
    monkeypatch.setattr(dashboard_app.st, "expander", lambda *args, **kwargs: _DummyCtx(), raising=False)
    monkeypatch.setattr(dashboard_app.st, "spinner", lambda *args, **kwargs: _DummyCtx(), raising=False)
    monkeypatch.setattr(dashboard_app.st, "slider", lambda *args, **kwargs: slider_value, raising=False)
    monkeypatch.setattr(dashboard_app.st, "radio", lambda *args, **kwargs: args[1][0], raising=False)
    monkeypatch.setattr(dashboard_app.st, "checkbox", lambda *args, **kwargs: kwargs.get("value", False), raising=False)

    for name in [
        "header", "subheader", "caption", "metric", "warning", "success", "error", "info",
        "write", "dataframe", "pyplot", "divider", "video", "plotly_chart", "progress",
    ]:
        monkeypatch.setattr(dashboard_app.st, name, lambda *args, **kwargs: None, raising=False)


def _build_fake_state() -> SimpleNamespace:
    raw = np.load(RAW_SAMPLE, allow_pickle=False)
    processed = np.load(PROCESSED_SAMPLE, allow_pickle=False)
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(64)]
    lm_frames = [_build_fake_landmarks(raw[min(i, raw.shape[0] - 1)]) for i in range(64)]
    vis_matrix = np.asarray([raw[min(i, raw.shape[0] - 1), :, 3] for i in range(64)], dtype=np.float32)
    resampled_to_source = list(range(64))
    timestamps = [float(i * 33.3333) for i in range(64)]
    interpolated_mask = np.zeros((64, 33), dtype=bool)
    unrecoverable_mask = np.zeros((64, 33), dtype=bool)

    return SimpleNamespace(
        video_bytes=b"fake-video-bytes",
        tensor_np=processed,
        tensor=np.expand_dims(processed, axis=0),
        raw_frames=frames,
        lm_frames=lm_frames,
        exercise_type="Squat",
        validator_df=dashboard_app.run_validator_per_frame(processed, "Squat", dashboard_app.BiomechanicalValidator()),
        video_stats={
            "video_frame_count": 99,
            "pose_frame_count": 64,
            "original_count": 64,
            "padded_count": 0,
            "sampled_indices": list(range(64)),
            "resampled_to_source": resampled_to_source,
            "resampled_source_frame_indices": resampled_to_source,
            "source_frame_indices": resampled_to_source,
            "source_timestamps_ms": timestamps,
            "resampled_timestamps_ms": timestamps,
            "source_interpolated_mask": interpolated_mask,
            "source_unrecoverable_mask": unrecoverable_mask,
            "resampled_interpolated_mask": interpolated_mask,
            "resampled_unrecoverable_mask": unrecoverable_mask,
            "interpolated_landmark_count": 0,
            "unrecoverable_landmark_count": 0,
            "vis_matrix": vis_matrix,
            "low_vis_frames": int((vis_matrix < 0.5).any(axis=1).sum()),
            "low_vis_lm": int((vis_matrix < 0.5).any(axis=0).sum()),
            "tensor_shape": tuple((1, 64, 33, 3)),
            "frame_mapping_mode": "linspace_downsample",
            "preprocess_signature": "max_frames=64|pose_model_complexity=1|min_detection_confidence=0.50|min_tracking_confidence=0.50|visibility_threshold=0.30",
            "video_hash": compute_video_hash(b"fake-video-bytes"),
        },
        processed=True,
    )


def test_frame_mapping_and_sync_info_are_deterministic():
    mapping_long = build_frame_mapping(99, 64)
    mapping_short = build_frame_mapping(40, 64)

    assert len(mapping_long["resampled_to_source"]) == 64
    assert len(mapping_short["resampled_to_source"]) == 64
    assert mapping_long["resampled_to_source"][0] == 0
    assert mapping_long["resampled_to_source"][-1] == 98
    assert mapping_short["resampled_to_source"][-1] == 39

    info = resolve_frame_sync_info(
        frame_index=63,
        mapping=mapping_short,
        source_timestamps_ms=[float(i * 40.0) for i in range(40)],
    )
    assert info.frame_index == 63
    assert info.source_frame_index == 39
    assert info.source_kind == "padded"


def test_frame_cache_keys_do_not_collide_across_videos():
    key_a = frame_cache_key("hash_a", "cfg", 7)
    key_b = frame_cache_key("hash_b", "cfg", 7)

    assert key_a != key_b


def test_frame_overlay_and_viewer_share_same_frame_index(monkeypatch):
    fake_state = _build_fake_state()
    monkeypatch.setattr(dashboard_app.st, "session_state", fake_state, raising=False)
    _install_streamlit_noops(monkeypatch, slider_value=7)

    calls: dict[str, int] = {}

    def fake_draw_frame_skeleton(frame_bgr, landmarks, **kwargs):
        calls["overlay_landmarks"] = len(landmarks)
        return frame_bgr.copy()

    def fake_render_mini_3d_viewer(**kwargs):
        calls["viewer_frame_index"] = kwargs["frame_index"]
        return True

    monkeypatch.setattr(dashboard_app, "draw_frame_skeleton", fake_draw_frame_skeleton, raising=False)
    monkeypatch.setattr(dashboard_app, "_render_mini_3d_viewer", fake_render_mini_3d_viewer, raising=False)
    monkeypatch.setattr(dashboard_app, "extract_spatial_attention", lambda model: np.ones(33, dtype=np.float32), raising=False)
    monkeypatch.setattr(dashboard_app, "extract_temporal_attention", lambda model, tensor: (np.ones(32, dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)), raising=False)
    monkeypatch.setattr(dashboard_app, "load_model_cached", lambda *args, **kwargs: (object(), [], [], ""), raising=False)

    dashboard_app._tab_data_integrity()
    dashboard_app._tab_frame_inspector("Full Model")

    assert calls["viewer_frame_index"] == 7
    assert calls["overlay_landmarks"] == 33


def test_tabs_render_without_crashing(monkeypatch):
    fake_state = _build_fake_state()
    monkeypatch.setattr(dashboard_app.st, "session_state", fake_state, raising=False)
    _install_streamlit_noops(monkeypatch, slider_value=0)
    monkeypatch.setattr(dashboard_app, "_render_mini_3d_viewer", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(dashboard_app, "draw_frame_skeleton", lambda frame_bgr, landmarks, **kwargs: frame_bgr.copy(), raising=False)
    monkeypatch.setattr(dashboard_app, "load_model_cached", lambda *args, **kwargs: (object(), [], [], ""), raising=False)
    monkeypatch.setattr(dashboard_app, "extract_spatial_attention", lambda model: np.ones(33, dtype=np.float32), raising=False)
    monkeypatch.setattr(dashboard_app, "extract_temporal_attention", lambda model, tensor: (np.ones(32, dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)), raising=False)

    dashboard_app._tab_data_integrity()
    dashboard_app._tab_frame_inspector("Full Model")
