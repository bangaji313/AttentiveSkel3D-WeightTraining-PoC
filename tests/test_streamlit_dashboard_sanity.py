from pathlib import Path

import numpy as np
import torch

from web_app import app as dash
from src.models.model_3dcnn import AttentiveSkel3D


ROOT = Path(__file__).resolve().parents[1]


def _abs_ckpt(name: str) -> str:
    return str(ROOT / dash.CHECKPOINT_PATHS[name])


def _load(name: str):
    cfg = dash.MODEL_CONFIGS[name]
    model, missing, unexpected, err = dash.load_model_cached(
        _abs_ckpt(name),
        cfg["use_spatial_prior"],
        cfg["use_learned_spatial"],
        cfg["use_temporal_attention"],
    )
    assert err == "", f"Failed loading {name}: {err}"
    assert model is not None
    return model, missing, unexpected


def test_five_checkpoints_have_distinct_sha256():
    shas = []
    for name, rel_path in dash.CHECKPOINT_PATHS.items():
        p = ROOT / rel_path
        assert p.exists(), f"Checkpoint missing: {name} -> {p}"
        sha = dash.compute_sha256(p)
        assert not sha.startswith("ERROR:"), sha
        assert len(sha) == 16
        shas.append(sha)

    assert len(set(shas)) == 5, f"SHA collision found: {shas}"


def test_models_not_reused_between_different_scenarios():
    full_model, _, _ = _load("Full Model")
    base_model, _, _ = _load("Baseline 3D-CNN")

    assert id(full_model) != id(base_model)
    assert full_model is not base_model


def test_attention_shape_valid_for_full_model():
    model, _, _ = _load("Full Model")
    x = torch.randn(1, 64, 33, 3, dtype=torch.float32)

    temporal, logits = dash.extract_temporal_attention(model, x)

    assert logits.shape == (2,)
    assert temporal is not None
    assert temporal.ndim == 1
    assert len(temporal) in (32, 64)


def test_baseline_has_no_fake_attention():
    model, _, _ = _load("Baseline 3D-CNN")
    x = torch.randn(1, 64, 33, 3, dtype=torch.float32)

    spatial = dash.extract_spatial_attention(model)
    temporal, logits = dash.extract_temporal_attention(model, x)

    assert spatial is None, "Baseline must not expose fake spatial attention"
    assert temporal is None, "Baseline must not expose temporal attention"
    assert logits.shape == (2,)


def test_timeline_interpolation_always_64():
    src = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    out = dash._interp_to_64(src)

    assert isinstance(out, np.ndarray)
    assert out.shape == (64,)


def test_model_accepts_expected_input_shape():
    m = AttentiveSkel3D(
        num_classes=2,
        use_spatial_prior=True,
        use_learned_spatial=True,
        use_temporal_attention=True,
    )
    x = torch.randn(2, 64, 33, 3, dtype=torch.float32)
    y = m(x)

    assert y.shape == (2, 2)
