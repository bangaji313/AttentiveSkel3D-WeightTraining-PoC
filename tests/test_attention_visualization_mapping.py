from __future__ import annotations

import sys
import types

import numpy as np


try:
    import cv2  # type: ignore
except Exception:
    fake_cv2 = types.SimpleNamespace(
        COLOR_BGR2RGB=0,
        LINE_AA=16,
        cvtColor=lambda frame, code: frame,
        line=lambda *args, **kwargs: None,
        circle=lambda *args, **kwargs: None,
        putText=lambda *args, **kwargs: None,
        FONT_HERSHEY_SIMPLEX=0,
    )
    sys.modules["cv2"] = fake_cv2

try:
    import mediapipe  # type: ignore
except Exception:
    fake_mp = types.SimpleNamespace()
    fake_mp.solutions = types.SimpleNamespace(
        pose=types.SimpleNamespace(Pose=object, POSE_CONNECTIONS=[]),
        drawing_utils=types.SimpleNamespace(),
    )
    sys.modules["mediapipe"] = fake_mp

try:
    import torch  # type: ignore
except Exception:
    class _DummyCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

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
    fake_nn.Sequential = lambda *args, **kwargs: list(args)
    fake_nn.Linear = _Linear
    fake_nn.ReLU = _Identity
    fake_nn.Dropout = _Identity
    fake_nn.Conv3d = _Conv3d
    fake_nn.BatchNorm3d = _Identity
    fake_nn.MaxPool3d = _Identity
    fake_nn.AdaptiveAvgPool3d = _Identity
    fake_nn.Sigmoid = _Sigmoid
    fake_nn.Parameter = lambda value: value
    fake_torch.nn = fake_nn
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn

try:
    import streamlit  # type: ignore
except Exception:
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.set_page_config = lambda **kwargs: None
    fake_streamlit.cache_data = lambda *args, **kwargs: (lambda fn: fn)
    fake_streamlit.cache_resource = lambda *args, **kwargs: (lambda fn: fn)
    fake_streamlit.image = lambda *args, **kwargs: None
    fake_streamlit.progress = lambda *args, **kwargs: types.SimpleNamespace(progress=lambda *a, **k: None, empty=lambda: None)
    fake_streamlit.radio = lambda *args, **kwargs: kwargs.get("index", 0)
    fake_streamlit.slider = lambda *args, **kwargs: kwargs.get("value", 0)
    fake_streamlit.checkbox = lambda *args, **kwargs: kwargs.get("value", False)
    fake_streamlit.columns = lambda *args, **kwargs: [types.SimpleNamespace() for _ in range(args[0] if args else 1)]
    fake_streamlit.expander = lambda *args, **kwargs: types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False)
    fake_streamlit.spinner = lambda *args, **kwargs: types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False)
    fake_streamlit.plotly_chart = lambda *args, **kwargs: None
    fake_streamlit.pyplot = lambda *args, **kwargs: None
    fake_streamlit.divider = lambda *args, **kwargs: None
    fake_streamlit.header = lambda *args, **kwargs: None
    fake_streamlit.subheader = lambda *args, **kwargs: None
    fake_streamlit.caption = lambda *args, **kwargs: None
    fake_streamlit.metric = lambda *args, **kwargs: None
    fake_streamlit.warning = lambda *args, **kwargs: None
    fake_streamlit.success = lambda *args, **kwargs: None
    fake_streamlit.error = lambda *args, **kwargs: None
    fake_streamlit.info = lambda *args, **kwargs: None
    fake_streamlit.dataframe = lambda *args, **kwargs: None
    fake_streamlit.video = lambda *args, **kwargs: None
    fake_streamlit.write = lambda *args, **kwargs: None
    fake_streamlit.code = lambda *args, **kwargs: None
    fake_streamlit.markdown = lambda *args, **kwargs: None
    fake_streamlit.session_state = types.SimpleNamespace()
    sys.modules["streamlit"] = fake_streamlit

from web_app.app import extract_attention_bundle, extract_spatial_attention
from web_app.attention_profiles import combine_attention, occlusion_values
from web_app.visualization_rules import exercise_relevant_landmarks, rule_landmarks, rule_marker_labels, rule_metric_for_frame


def test_rule_landmark_mapping_and_labels_are_consistent() -> None:
    squat_rules = [
        {
            "name": "Hip Flexion Angle",
            "lm_a": 11,
            "lm_b": 23,
            "lm_c": 25,
            "threshold_val": 137.0,
        },
        {
            "name": "Squat Depth (Knee Angle)",
            "lm_a": 23,
            "lm_b": 25,
            "lm_c": 27,
            "threshold_val": 100.0,
        },
    ]

    assert exercise_relevant_landmarks(squat_rules) == [11, 23, 25, 27]
    assert rule_landmarks(squat_rules[0]) == [11, 23, 25]
    assert rule_marker_labels(squat_rules[0]) == {11: "A", 23: "B", 25: "C"}


def test_rule_metric_for_frame_uses_actual_frame_geometry() -> None:
    frame = np.zeros((33, 3), dtype=np.float32)
    frame[25, 0] = 1.0
    frame[26, 0] = 0.0
    frame[27, 0] = 2.0
    frame[28, 0] = 0.0

    rule = {
        "name": "Knee Valgus",
        "lm_a": 25,
        "lm_b": None,
        "lm_c": 27,
        "threshold_val": 0.85,
    }

    result = rule_metric_for_frame("Squat", rule, frame)
    assert result["metric_value"] == 0.5
    assert result["status"] == "INVALID"
    assert result["relevant_landmarks"] == [25, 27]


def test_baseline_attention_stays_unavailable() -> None:
    class DummyModel:
        use_spatial_prior = False
        use_learned_spatial = False

    visibility = np.linspace(0.0, 1.0, 33, dtype=np.float32)
    bundle = extract_attention_bundle(DummyModel(), np.zeros((1, 64, 33, 3), dtype=np.float32), visibility)

    assert extract_spatial_attention(DummyModel()) is None
    assert bundle["prior"] is None
    assert bundle["learned"] is None
    assert bundle["fused"] is None
    np.testing.assert_allclose(bundle["occlusion"], 1.0 - visibility)


def test_attention_helpers_fuse_and_normalize_values() -> None:
    prior = np.array([0.2, 0.6, 0.8], dtype=np.float32)
    learned = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    fused = combine_attention(prior, learned)
    assert fused is not None
    assert fused.shape == (3,)
    assert float(fused.max()) == 1.0
    assert float(fused.min()) == 0.0


def test_occlusion_vector_is_derived_from_visibility_only() -> None:
    visibility = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    occlusion = occlusion_values(visibility)
    np.testing.assert_allclose(occlusion, np.array([1.0, 0.75, 0.0], dtype=np.float32))
