from __future__ import annotations

from pathlib import Path

import numpy as np

from web_app import app as dashboard_app
from web_app import ui_compat


ROOT = Path(__file__).resolve().parents[1]


class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeLm:
    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def test_import_web_app_app_success():
    assert dashboard_app is not None
    assert callable(dashboard_app.main)


def test_display_image_compat_numpy_rgb(monkeypatch):
    captured = {}

    def fake_image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto"):
        captured["image"] = image
        captured["kwargs"] = {
            "caption": caption,
            "width": width,
            "use_column_width": use_column_width,
            "clamp": clamp,
            "channels": channels,
            "output_format": output_format,
        }

    monkeypatch.setattr(ui_compat.st, "image", fake_image)

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    ui_compat.display_image_compat(rgb, caption="demo", stretch=True, channels="RGB")

    assert isinstance(captured["image"], np.ndarray)
    assert captured["image"].shape == (32, 32, 3)
    assert captured["kwargs"]["caption"] == "demo"
    assert captured["kwargs"]["channels"] == "RGB"


def test_display_image_compat_does_not_send_use_container_width_when_not_supported(monkeypatch):
    captured = {}

    # Simulate legacy signature: no use_container_width, has use_column_width.
    def fake_image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto"):
        captured["kwargs"] = {
            "caption": caption,
            "width": width,
            "use_column_width": use_column_width,
            "clamp": clamp,
            "channels": channels,
            "output_format": output_format,
        }

    monkeypatch.setattr(ui_compat.st, "image", fake_image)

    ui_compat.display_image_compat(np.zeros((8, 8, 3), dtype=np.uint8), stretch=True)

    # Ensure compatibility fallback used and no unsupported kw passed.
    assert "use_container_width" not in captured["kwargs"]
    assert captured["kwargs"]["use_column_width"] is True


def test_render_tab_with_debug_catches_and_reports(monkeypatch):
    calls = {"error": 0, "expander": 0, "code": 0}

    def fake_error(msg):
        calls["error"] += 1

    def fake_expander(label):
        calls["expander"] += 1
        return _DummyCtx()

    def fake_code(text):
        calls["code"] += 1

    monkeypatch.setattr(ui_compat.st, "error", fake_error)
    monkeypatch.setattr(ui_compat.st, "expander", fake_expander)
    monkeypatch.setattr(ui_compat.st, "code", fake_code)

    def boom():
        raise RuntimeError("tab crash")

    ui_compat.render_tab_with_debug("Tab Test", boom)

    assert calls["error"] == 1
    assert calls["expander"] == 1
    assert calls["code"] == 1


def test_five_tab_render_functions_exist():
    assert callable(dashboard_app._tab_data_integrity)
    assert callable(dashboard_app._tab_biomechanical)
    assert callable(dashboard_app._tab_classification)
    assert callable(dashboard_app._tab_attention)
    assert callable(dashboard_app._tab_frame_inspector)


def test_overlay_pipeline_no_crash_with_synthetic_landmarks():
    frame_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    lms = [_FakeLm(x=(i % 8) / 8.0, y=(i // 8) / 5.0, visibility=1.0) for i in range(33)]
    attn = np.linspace(0.0, 1.0, 33, dtype=np.float32)

    overlay = dashboard_app.draw_frame_skeleton(
        frame_bgr,
        lms,
        color_per_joint=[(0, 255, 0)] * 33,
        relevant_joints=[11, 23, 25],
        attention_labels=attn,
    )

    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == frame_bgr.shape


def test_run_validator_per_frame_returns_64_rows():
    tensor = np.random.randn(64, 33, 3).astype(np.float32)
    vdf = dashboard_app.run_validator_per_frame(
        tensor,
        "Squat",
        dashboard_app.BiomechanicalValidator(),
    )

    assert len(vdf) == 64
    assert {"frame_index", "validator_status", "metric_value", "threshold"}.issubset(vdf.columns)
