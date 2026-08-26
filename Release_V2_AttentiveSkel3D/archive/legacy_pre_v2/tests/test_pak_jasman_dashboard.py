from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAK = ROOT / "web_app" / "app_pak_jasman.py"
FULL = ROOT / "web_app" / "app.py"


PAK_SOURCE = PAK.read_text(encoding="utf-8")
FULL_SOURCE = FULL.read_text(encoding="utf-8")


def test_pak_jasman_exposes_five_tabs_and_class_mapping() -> None:
    assert 'TAB_LABELS = [' in PAK_SOURCE
    assert '"1. Data Integrity"' in PAK_SOURCE
    assert '"2. Biomechanical Validator"' in PAK_SOURCE
    assert '"3. Classification"' in PAK_SOURCE
    assert '"4. Attention Sanity Check"' in PAK_SOURCE
    assert '"5. Frame Inspector"' in PAK_SOURCE
    assert 'FRAME_SLIDER_RANGE = (0, MAX_FRAMES - 1)' in PAK_SOURCE
    assert 'CLASS_MAPPING = {0: "Benar", 1: "Salah"}' in PAK_SOURCE
    assert 'CLASS_DISPLAY = {0: "Form Benar ✔", 1: "Form Salah ✗"}' in PAK_SOURCE
    assert 'SCENARIO_OPTIONS = {' in PAK_SOURCE
    assert '"Baseline": "Baseline 3D-CNN"' in PAK_SOURCE


def test_baseline_scenario_disables_attention_branches() -> None:
    assert '"Baseline 3D-CNN":               dict(use_spatial_prior=False, use_learned_spatial=False, use_temporal_attention=False)' in FULL_SOURCE
    assert '"Baseline": "Baseline 3D-CNN"' in PAK_SOURCE
    assert 'Attention tidak tersedia pada skenario Baseline' in PAK_SOURCE


def test_pak_jasman_does_not_depend_on_3d_viewer() -> None:
    assert 'create_3d_skeleton_figure' not in PAK_SOURCE
    assert 'skeleton_3d_viewer' not in PAK_SOURCE
    assert 'mini 3D' not in PAK_SOURCE


def test_full_dashboard_file_remains_intact() -> None:
    assert 'Scientific Proof Dashboard' in FULL_SOURCE
    assert 'streamlit run web_app/app.py' in FULL_SOURCE
    assert 'web_app/app_pak_jasman.py' not in FULL_SOURCE
