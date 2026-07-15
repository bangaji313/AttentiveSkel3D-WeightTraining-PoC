"""
AttentiveSkel-3D — Scientific Proof Dashboard
==============================================
Dashboard pembuktian ilmiah: integritas tensor, validasi biomekanik per-frame,
klasifikasi sequence-level, analisis atensi 5 skenario, dan inspeksi frame kritis.

Jalankan: streamlit run web_app/app.py
"""
from __future__ import annotations

import hashlib
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

# ── sys.path setup ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
for _p in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from web_app.ui_compat import display_image_compat, render_tab_with_debug
from web_app.attention_profiles import combine_attention, normalize_attention, occlusion_values
from web_app.components import create_3d_skeleton_figure, prepare_frame_payload
from web_app.frame_sync import build_frame_mapping, compute_video_hash, frame_cache_key, preprocess_signature, resolve_frame_sync_info
from web_app.visualization_rules import exercise_relevant_landmarks, rule_marker_labels, rule_metric_for_frame

# ── Page config (WAJIB paling pertama) ───────────────────────────────────────
st.set_page_config(
    page_title="AttentiveSkel-3D | Scientific Proof Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.models.model_3dcnn import AttentiveSkel3D
from src.data.biomechanics_validator import BiomechanicalValidator

log = logging.getLogger("dashboard")

# =============================================================================
# CONSTANTS — validated against results/model_audit/audit_report.md
# =============================================================================

MAX_FRAMES = 64

# Checkpoint paths (relative to ROOT_DIR)
CHECKPOINT_PATHS: dict[str, str] = {
    "Full Model":                    "models/saved_models/AttentiveSkel3D_Final.pth",
    "Baseline 3D-CNN":               "models/saved_models/baseline_3dcnn_model.pth",
    "Ablasi A - No Prior":           "models/saved_models/ablasi_a_no_prior.pth",
    "Ablasi B - No Learned Spatial": "models/saved_models/ablasi_b_no_learned.pth",
    "Ablasi C - No Temporal":        "models/saved_models/ablasi_c_no_temporal.pth",
}

# Architecture flags per scenario — VALIDATED against training config & audit
MODEL_CONFIGS: dict[str, dict] = {
    "Full Model":                    dict(use_spatial_prior=True,  use_learned_spatial=True,  use_temporal_attention=True),
    "Baseline 3D-CNN":               dict(use_spatial_prior=False, use_learned_spatial=False, use_temporal_attention=False),
    "Ablasi A - No Prior":           dict(use_spatial_prior=False, use_learned_spatial=True,  use_temporal_attention=True),
    "Ablasi B - No Learned Spatial": dict(use_spatial_prior=True,  use_learned_spatial=False, use_temporal_attention=True),
    "Ablasi C - No Temporal":        dict(use_spatial_prior=True,  use_learned_spatial=True,  use_temporal_attention=False),
}

# SHA256 prefixes from audit (16 hex chars) — used for integrity verification
AUDIT_SHA256: dict[str, str] = {
    "Full Model":                    "5440164315d6d275",
    "Baseline 3D-CNN":               "3224393f85793286",
    "Ablasi A - No Prior":           "d13e665670e3ad61",
    "Ablasi B - No Learned Spatial": "805634011271f1bd",
    "Ablasi C - No Temporal":        "a739bd17cf24d8bc",
}

CLASS_NAMES = {0: "Form Benar ✔", 1: "Form Salah ✗"}

# Biomechanical rules per exercise (from BiomechanicalValidator source & literature)
EXERCISE_RULES: dict[str, list[dict]] = {
    "Squat": [
        {
            "name": "Knee Valgus",
            "lm_a": 25, "lm_b": None, "lm_c": 27,
            "label": "Left/Right Knee width ÷ Ankle width",
            "threshold_str": "ratio ≥ 0.85",
            "threshold_val": 0.85,
            "description": (
                "Lebar horizontal lutut tidak boleh < 85% lebar pergelangan kaki "
                "pada frame terdalam. Mencegah kolaps medial (knee valgus / ACL injury)."
            ),
            "reference": "Chen et al. (2022)",
        },
        {
            "name": "Hip Flexion Angle",
            "lm_a": 11, "lm_b": 23, "lm_c": 25,
            "label": "Shoulder(11) → Hip(23) → Knee(25)",
            "threshold_str": "≤ 137°",
            "threshold_val": 137.0,
            "description": (
                "Sudut Bahu-Pinggul-Lutut harus ≤ 137° di posisi terdalam. "
                "Berdasarkan Rao et al. (2023): rata-rata 128° ± 9°. "
                "Nilai > 137° = half rep, gluteal tidak teraktivasi optimal."
            ),
            "reference": "Rao et al. (2023), Ko et al. (2024)",
        },
        {
            "name": "Squat Depth (Knee Angle)",
            "lm_a": 23, "lm_b": 25, "lm_c": 27,
            "label": "Hip(23) → Knee(25) → Ankle(27)",
            "threshold_str": "≤ 100°",
            "threshold_val": 100.0,
            "description": (
                "Sudut Pinggul-Lutut-Pergelangan Kaki ≤ 100° = parallel squat. "
                "Batas minimum kedalaman efektif secara biomekanik."
            ),
            "reference": "Hales et al. (2009)",
        },
    ],
    "BenchPress": [
        {
            "name": "Elbow ROM",
            "lm_a": 11, "lm_b": 13, "lm_c": 15,
            "label": "Shoulder(11) → Elbow(13) → Wrist(15)",
            "threshold_str": "≤ 85°",
            "threshold_val": 85.0,
            "description": (
                "Sudut Bahu-Siku-Pergelangan Tangan ≤ 85° saat bar paling dekat ke dada. "
                "Memastikan full ROM untuk aktivasi pectoralis major optimal."
            ),
            "reference": "Chen et al. (2022)",
        },
    ],
    "Deadlift": [
        {
            "name": "Spine Inclination",
            "lm_a": 11, "lm_b": 23, "lm_c": 25,
            "label": "Shoulder(11) → Hip(23) → Knee(25)",
            "threshold_str": "20°–60°",
            "threshold_val": 60.0,
            "description": (
                "Inklinasi tulang belakang 20°–60° dari vertikal sepanjang gerakan. "
                "< 20° = terlalu tegak; > 60° = terlalu condong (risiko cedera lumbar)."
            ),
            "reference": "Ko et al. (2024)",
        },
    ],
}

# Primary metric landmark triplet per exercise (for per-frame angle timeline)
PRIMARY_METRIC: dict[str, tuple[int, int, int]] = {
    "Squat":      (11, 23, 25),  # Hip Flexion Angle: Shoulder-Hip-Knee
    "BenchPress": (11, 13, 15),  # Elbow ROM: Shoulder-Elbow-Wrist
    "Deadlift":   (11, 23, 25),  # Spine proxy: Shoulder-Hip-Knee
}

PRIMARY_THRESHOLD: dict[str, float] = {
    "Squat":      BiomechanicalValidator.SQUAT_HIP_MAX_DEG,
    "BenchPress": BiomechanicalValidator.BENCH_ELBOW_THRESHOLD_DEG,
    "Deadlift":   BiomechanicalValidator.DEADLIFT_SPINE_MAX_DEG,
}

# MediaPipe
_mp_pose   = mp.solutions.pose
_POSE_CONN = _mp_pose.POSE_CONNECTIONS


# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def compute_sha256(path: Path, n: int = 16) -> str:
    """First n hex chars of SHA256; returns 'ERROR:...' on failure."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except OSError as exc:
        return f"ERROR:{exc}"


@st.cache_resource(show_spinner=False)
def load_model_cached(
    abs_path: str,
    use_spatial_prior: bool,
    use_learned_spatial: bool,
    use_temporal_attention: bool,
) -> tuple:
    """
    Load AttentiveSkel3D from disk.
    Cache key = (abs_path, use_spatial_prior, use_learned_spatial, use_temporal_attention).
    NOT a single global cache — each scenario gets its own cached instance.

    Returns: (model | None, missing_keys, unexpected_keys, error_str)
    """
    p = Path(abs_path)
    if not p.exists():
        return None, [], [], f"File not found: {p}"
    try:
        mdl = AttentiveSkel3D(
            num_classes=2,
            use_spatial_prior=use_spatial_prior,
            use_learned_spatial=use_learned_spatial,
            use_temporal_attention=use_temporal_attention,
        )
        try:
            ckpt = torch.load(abs_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(abs_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict):
            state_dict = (
                ckpt.get("model_state_dict")
                or ckpt.get("state_dict")
                or ckpt
            )
        else:
            state_dict = ckpt

        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        if hasattr(mdl, "load_state_dict"):
            incompatible = mdl.load_state_dict(state_dict, strict=False)
            missing_keys = list(getattr(incompatible, "missing_keys", []))
            unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
        if hasattr(mdl, "eval"):
            mdl.eval()
        return mdl, missing_keys, unexpected_keys, ""
    except Exception as exc:
        return None, [], [], str(exc)


def build_inference_tensor(video_path: str, *, video_bytes: bytes | None = None) -> tuple:
    """
    Extract MediaPipe BlazePose skeleton from video.
    Maps extracted pose frames to exactly MAX_FRAMES (64) using deterministic
    linspace downsampling or repeat-last padding.

    Returns:
        tensor      : torch.Tensor (1, 64, 33, 3) | None
        lm_frames   : list[landmarks] length 64
        raw_frames  : list[np.ndarray BGR] length 64
        stats       : dict with provenance metadata
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, [], [], {"error": "Cannot open video file"}

    seqs: list = []
    lms_list: list = []
    raws: list = []
    vis_list: list = []
    source_frame_indices: list[int] = []
    source_timestamps_ms: list[float] = []
    raw_video_frame_count = 0

    with _mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            current_frame_index = raw_video_frame_count
            raw_video_frame_count += 1

            timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if timestamp_ms <= 0.0:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if fps > 0.0:
                    timestamp_ms = (current_frame_index / fps) * 1000.0

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks is None:
                continue
            lms = res.pose_landmarks.landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            vis = np.array([lm.visibility for lm in lms], dtype=np.float32)
            seqs.append(coords)
            lms_list.append(lms)
            raws.append(frame_bgr.copy())
            vis_list.append(vis)
            source_frame_indices.append(current_frame_index)
            source_timestamps_ms.append(timestamp_ms)
    cap.release()

    if not seqs:
        return None, [], [], {"error": "No pose detected in video"}

    source_interpolated_mask, source_unrecoverable_mask = _compute_interpolation_masks(seqs, vis_list)

    original_count = len(seqs)
    mapping = build_frame_mapping(original_count, MAX_FRAMES)
    sampled_indices = list(mapping["sampled_indices"])
    resampled_to_source = list(mapping["resampled_to_source"])
    padded_count = int(mapping["padded_count"])

    if original_count >= MAX_FRAMES:
        indices = np.asarray(sampled_indices, dtype=int)
        seqs = [seqs[i] for i in indices]
        lms_list = [lms_list[i] for i in indices]
        raws = [raws[i] for i in indices]
        vis_list = [vis_list[i] for i in indices]
    else:
        last = original_count - 1
        seqs = seqs + [seqs[last]] * padded_count
        lms_list = lms_list + [lms_list[last]] * padded_count
        raws = raws + [raws[last]] * padded_count
        vis_list = vis_list + [vis_list[last]] * padded_count

    tensor_np = np.stack(seqs, axis=0)   # (64, 33, 3)
    vis_matrix = np.stack(vis_list, axis=0)   # (64, 33)
    tensor = torch.tensor(tensor_np, dtype=torch.float32).unsqueeze(0)  # (1, 64, 33, 3)

    resampled_source_frame_indices = [source_frame_indices[i] for i in resampled_to_source]
    resampled_timestamps_ms = [source_timestamps_ms[i] for i in resampled_to_source]
    resampled_interpolated_mask = source_interpolated_mask[resampled_to_source]
    resampled_unrecoverable_mask = source_unrecoverable_mask[resampled_to_source]

    stats = {
        "video_frame_count": raw_video_frame_count,
        "pose_frame_count": original_count,
        "original_count": original_count,
        "padded_count": padded_count,
        "sampled_indices": sampled_indices,
        "resampled_to_source": resampled_to_source,
        "resampled_source_frame_indices": resampled_source_frame_indices,
        "source_frame_indices": source_frame_indices,
        "source_timestamps_ms": source_timestamps_ms,
        "resampled_timestamps_ms": resampled_timestamps_ms,
        "source_interpolated_mask": source_interpolated_mask,
        "source_unrecoverable_mask": source_unrecoverable_mask,
        "resampled_interpolated_mask": resampled_interpolated_mask,
        "resampled_unrecoverable_mask": resampled_unrecoverable_mask,
        "interpolated_landmark_count": int(source_interpolated_mask.sum()),
        "unrecoverable_landmark_count": int(source_unrecoverable_mask.sum()),
        "tensor_shape": tuple(tensor.shape),
        "vis_matrix": vis_matrix,
        "low_vis_frames": int((vis_matrix < 0.5).any(axis=1).sum()),
        "low_vis_lm": int((vis_matrix < 0.5).any(axis=0).sum()),
        "frame_mapping_mode": str(mapping["mapping_mode"]),
        "preprocess_signature": preprocess_signature(max_frames=MAX_FRAMES),
        "video_hash": compute_video_hash(video_bytes) if video_bytes is not None else None,
    }
    return tensor, lms_list, raws, stats


def _compute_interpolation_masks(
    coords_list: list[np.ndarray],
    visibility_list: list[np.ndarray],
    *,
    visibility_threshold: float = 0.3,
    max_interp_gap: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return landmark-level masks for interpolated and unrecoverable values."""
    if not coords_list:
        return np.zeros((0, 33), dtype=bool), np.zeros((0, 33), dtype=bool)

    coords = np.stack(coords_list, axis=0).astype(np.float32, copy=True)
    visibility = np.stack(visibility_list, axis=0).astype(np.float32, copy=True)
    low_confidence_mask = visibility < visibility_threshold

    data = np.concatenate([coords, visibility[..., None]], axis=2)
    data[low_confidence_mask, :3] = np.nan

    interpolated_mask = np.zeros((data.shape[0], data.shape[1]), dtype=bool)
    unrecoverable_mask = np.zeros((data.shape[0], data.shape[1]), dtype=bool)

    for landmark_idx in range(data.shape[1]):
        for coord_idx in range(3):
            series = pd.Series(data[:, landmark_idx, coord_idx])
            before = series.isna()
            after = series.interpolate(
                method="linear",
                limit=max_interp_gap,
                limit_direction="both",
            )
            filled = before & after.notna()
            interpolated_mask[:, landmark_idx] |= filled.to_numpy()
            remaining = after.isna()
            unrecoverable_mask[:, landmark_idx] |= remaining.to_numpy()

    return interpolated_mask, unrecoverable_mask


def extract_spatial_attention(model: AttentiveSkel3D) -> Optional[np.ndarray]:
    """
    Extract BSP spatial weights (33,) via sigmoid.
    Returns None when use_spatial_prior=False.
    NEVER returns fake uniform values.
    """
    if not getattr(model, "use_spatial_prior", False):
        return None
    if not hasattr(model, "biomechanical_spatial_prior"):
        return None
    with torch.no_grad():
        w = torch.sigmoid(model.biomechanical_spatial_prior).squeeze().cpu().numpy()
    return w.astype(np.float32)


def extract_learned_spatial_attention(
    model: AttentiveSkel3D,
    tensor_input: torch.Tensor,
) -> Optional[np.ndarray]:
    """Derive a landmark-level proxy from the learned spatial branch.

    The model's learned spatial branch is channel attention, so the dashboard
    projects the post-attention feature volume back to 33 landmarks by
    averaging the attention-weighted activations across channels and time.
    Returns None when the branch is disabled.
    """
    if not getattr(model, "use_learned_spatial", False):
        return None

    with torch.no_grad():
        x = tensor_input.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()

        if getattr(model, "use_spatial_prior", False) and hasattr(model, "biomechanical_spatial_prior"):
            x = x * torch.sigmoid(model.biomechanical_spatial_prior)

        x = model.conv_block_1(x)
        x = model.conv_block_2(x)
        x = model.conv_block_3(x)

        gap_feat = x.mean(dim=[2, 3, 4])
        ch_weights = model.learned_spatial_attention(gap_feat)
        weighted = x * ch_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pooled = weighted.abs().mean(dim=1).mean(dim=1).squeeze(-1)
        pooled_np = pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)

    if pooled_np.size == 0:
        return None
    if pooled_np.size == 33:
        return pooled_np

    x_src = np.linspace(0.0, pooled_np.size - 1.0, pooled_np.size, dtype=np.float32)
    x_dst = np.linspace(0.0, pooled_np.size - 1.0, 33, dtype=np.float32)
    return np.interp(x_dst, x_src, pooled_np).astype(np.float32)


def extract_attention_bundle(
    model: AttentiveSkel3D,
    tensor_input: torch.Tensor,
    visibility: np.ndarray | None = None,
) -> dict[str, Optional[np.ndarray]]:
    prior = extract_spatial_attention(model)
    learned = extract_learned_spatial_attention(model, tensor_input)
    fused = combine_attention(prior, learned)
    occlusion = occlusion_values(visibility)
    return {
        "prior": prior,
        "learned": learned,
        "fused": fused,
        "occlusion": occlusion,
    }


def extract_temporal_attention(
    model: AttentiveSkel3D,
    tensor_input: torch.Tensor,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """
    Forward pass with hook on the temporal_attention Conv3d layer.
    Replicates model's own pipeline: mean over spatial dims → softmax over T'.
    ALL hooks are removed after inference (no leak between calls).

    Returns:
        temporal_scores : np.ndarray (T'=32,) softmax weights | None
        logits_np       : np.ndarray (2,)
    """
    if not getattr(model, "use_temporal_attention", False):
        with torch.no_grad():
            logits = model(tensor_input)
        if hasattr(logits, "detach"):
            return None, logits.squeeze(0).detach().cpu().numpy()
        return None, np.asarray(logits).squeeze(0)

    if not hasattr(model, "named_modules"):
        with torch.no_grad():
            logits = model(tensor_input)
        fallback_len = 32 if getattr(tensor_input, "shape", (0, 64))[1] >= 64 else int(getattr(tensor_input, "shape", (0, 32))[1])
        fallback_scores = np.full(fallback_len, 1.0 / max(fallback_len, 1), dtype=np.float32)
        logits_np = np.asarray(logits).squeeze(0) if not hasattr(logits, "detach") else logits.squeeze(0).detach().cpu().numpy()
        return fallback_scores, logits_np

    captured: dict = {}
    handles:  list = []

    def _hook(module, inp, out):
        # out: (B, 1, T', L', 1) from Conv3d(128→1, kernel=1×1×1)
        # Replicate model's forward: mean over [L', W] → softmax over T'
        spatial_mean = out.mean(dim=[3, 4], keepdim=True)   # (B, 1, T', 1, 1)
        weights      = torch.softmax(spatial_mean, dim=2)    # (B, 1, T', 1, 1)
        t_scores     = weights.squeeze()                      # (T',)
        if t_scores.dim() == 0:
            t_scores = t_scores.unsqueeze(0)
        captured["scores"] = t_scores.detach().cpu().numpy()

    # Hook onto the Conv3d inside the temporal_attention Sequential
    for name, module in model.named_modules():
        if "temporal_attention" in name and isinstance(module, torch.nn.Conv3d):
            handles.append(module.register_forward_hook(_hook))
            break

    try:
        with torch.no_grad():
            logits = model(tensor_input)
    finally:
        for h in handles:  # Always remove hooks
            h.remove()

    if captured.get("scores") is None:
        fallback_len = 32 if getattr(tensor_input, "shape", (0, 64))[1] >= 64 else int(getattr(tensor_input, "shape", (0, 32))[1])
        fallback_scores = np.full(fallback_len, 1.0 / max(fallback_len, 1), dtype=np.float32)
        logits_np = np.asarray(logits).squeeze(0) if not hasattr(logits, "detach") else logits.squeeze(0).detach().cpu().numpy()
        return fallback_scores, logits_np

    logits_np = logits.squeeze(0).detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits).squeeze(0)
    return captured.get("scores"), logits_np


def _interp_to_64(scores: np.ndarray) -> np.ndarray:
    """Interpolate temporal scores of length T' to MAX_FRAMES=64."""
    if len(scores) == MAX_FRAMES:
        return scores
    x_src = np.linspace(0, MAX_FRAMES - 1, len(scores))
    return np.interp(np.arange(MAX_FRAMES), x_src, scores)


def run_validator_per_frame(
    tensor_np: np.ndarray,   # (64, 33, 3)
    exercise: str,
    validator: BiomechanicalValidator,
) -> pd.DataFrame:
    """
    Apply BiomechanicalValidator on each frame independently.
    Primary metric angle computed for timeline visualization.
    """
    validate_fn = {
        "Squat":      validator.validate_squat,
        "BenchPress": validator.validate_benchpress,
        "Deadlift":   validator.validate_deadlift,
    }.get(exercise, validator.validate_squat)

    threshold = PRIMARY_THRESHOLD.get(exercise, 137.0)
    records   = []

    for i in range(len(tensor_np)):
        frame       = tensor_np[i]               # (33, 3)
        frame_batch = frame[np.newaxis]           # (1, 33, 3)

        try:
            is_valid, reason = validate_fn(frame_batch)
        except Exception as exc:
            is_valid, reason = False, f"Error: {exc}"

        # Primary angle for timeline
        la, lb, lc = PRIMARY_METRIC.get(exercise, (11, 23, 25))
        metric = BiomechanicalValidator.calculate_angle_3d(
            frame[la], frame[lb], frame[lc]
        )

        records.append({
            "frame_index":      i,
            "is_valid":         bool(is_valid),
            "validator_status": "VALID" if is_valid else "INVALID",
            "metric_value":     round(float(metric), 2),
            "threshold":        threshold,
            "reason":           str(reason)[:150],
        })

    return pd.DataFrame(records)


def run_validator_sequence(
    tensor_np: np.ndarray,
    exercise: str,
    validator: BiomechanicalValidator,
) -> tuple[bool, str]:
    """Run the exercise-level validator on the full 64-frame tensor."""
    validate_fn = {
        "Squat": validator.validate_squat,
        "BenchPress": validator.validate_benchpress,
        "Deadlift": validator.validate_deadlift,
    }.get(exercise, validator.validate_squat)

    return validate_fn(tensor_np)


def format_agreement_status(model_pred: int, validator_is_valid: bool) -> str:
    """Map classifier/validator outcomes to a concise agreement label."""
    model_label = 0 if model_pred == 0 else 1
    validator_label = 0 if validator_is_valid else 1

    if model_label == validator_label:
        return "AGREE_CORRECT" if model_label == 0 else "AGREE_INCORRECT"
    if validator_label == 1 and model_label == 0:
        return "DISAGREE_VALIDATOR_INCORRECT_MODEL_CORRECT"
    return "DISAGREE_VALIDATOR_CORRECT_MODEL_INCORRECT"


def draw_frame_skeleton(
    frame_bgr: np.ndarray,
    landmarks,
    color_per_joint: Optional[list] = None,
    relevant_joints: Optional[list] = None,
    attention_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw MediaPipe skeleton overlay.

    color_per_joint : list[(B,G,R)] per 33 joints; defaults to cyan
    relevant_joints : drawn larger with white ring + label
    attention_labels: np.ndarray (33,) — printed beside each relevant joint
    """
    out  = frame_bgr.copy()
    h, w = out.shape[:2]
    n_lm = min(33, len(landmarks))

    for conn in _POSE_CONN:
        s, e = conn
        if s >= n_lm or e >= n_lm:
            continue
        lms = landmarks[s]; lme = landmarks[e]
        if getattr(lms, "visibility", 1.0) < 0.3:
            continue
        cv2.line(out,
                 (int(lms.x * w), int(lms.y * h)),
                 (int(lme.x * w), int(lme.y * h)),
                 (55, 55, 55), 1, cv2.LINE_AA)

    for i in range(n_lm):
        lm = landmarks[i]
        if getattr(lm, "visibility", 1.0) < 0.25:
            continue
        px, py = int(lm.x * w), int(lm.y * h)
        is_rel = relevant_joints is not None and i in relevant_joints
        color  = color_per_joint[i] if color_per_joint else (0, 200, 200)
        radius = 9 if is_rel else 4
        cv2.circle(out, (px, py), radius, color, -1, cv2.LINE_AA)
        if is_rel:
            cv2.circle(out, (px, py), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
            lbl = (
                f"{i}:{float(attention_labels[i]):.2f}"
                if attention_labels is not None
                else str(i)
            )
            cv2.putText(out, lbl, (px + radius + 3, py - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    return out


@st.cache_data(show_spinner=False)
def _cached_mini_3d_figure(
    cache_key: str,
    frame_points: np.ndarray,
    visibility: np.ndarray,
    display_mode: str,
    view_mode: str,
    show_axes: bool,
    show_labels: bool,
    show_bbox: bool,
    interpolated_indices: tuple[int, ...],
    attention: Optional[np.ndarray] = None,
    extra_hover_data: Optional[dict[str, np.ndarray]] = None,
) -> object:
    return create_3d_skeleton_figure(
        frame_points,
        visibility=visibility,
        attention=attention,
        extra_hover_data=extra_hover_data,
        display_mode=display_mode,
        view=view_mode,
        show_axes=show_axes,
        show_labels=show_labels,
        show_bbox=show_bbox,
        interpolated_landmarks=interpolated_indices,
    )


def _render_mini_3d_viewer(
    *,
    frame_points: np.ndarray,
    visibility: np.ndarray,
    video_hash: str,
    preprocess_key: str,
    frame_index: int,
    display_mode: str,
    view_mode: str,
    show_axes: bool,
    show_labels: bool,
    show_bbox: bool,
    interpolated_indices: tuple[int, ...],
    attention: Optional[np.ndarray] = None,
    extra_hover_data: Optional[dict[str, np.ndarray]] = None,
) -> bool:
    try:
        cache_key = frame_cache_key(video_hash, preprocess_key, frame_index)
        fig = _cached_mini_3d_figure(
            cache_key,
            frame_points,
            visibility,
            display_mode,
            view_mode,
            show_axes,
            show_labels,
            show_bbox,
            interpolated_indices,
            attention,
            extra_hover_data,
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        return True
    except Exception as exc:
        st.error(f"Mini 3D Skeleton Viewer gagal: {exc}")
        return False


def _checkpoint_integrity_df() -> pd.DataFrame:
    rows = []
    for name, rel in CHECKPOINT_PATHS.items():
        p       = ROOT_DIR / rel
        exists  = p.exists()
        live    = compute_sha256(p) if exists else "—"
        audit   = AUDIT_SHA256[name]
        match   = "✓ OK" if live == audit else ("⚠ MISMATCH" if exists else "✗ MISSING")
        cfg     = MODEL_CONFIGS[name]
        rows.append({
            "Scenario":           name,
            "Path":               rel,
            "Exists":             "✓" if exists else "✗",
            "SHA256 (live)":      live,
            "SHA256 (audit)":     audit,
            "Integrity":          match,
            "BSP":                "✓" if cfg["use_spatial_prior"]      else "✗",
            "Learned Spatial":    "✓" if cfg["use_learned_spatial"]    else "✗",
            "Temporal Attention": "✓" if cfg["use_temporal_attention"] else "✗",
        })
    return pd.DataFrame(rows)


# =============================================================================
# TAB RENDERERS
# =============================================================================

def _tab_data_integrity() -> None:
    st.header("📊 Tab 1 — Data Integrity")
    s, stats = st.session_state, st.session_state.video_stats

    frame_idx = st.slider("Frame Preview (0–63):", 0, MAX_FRAMES - 1, 0, 1, key="data_integrity_frame_idx")
    display_mode_label = st.radio(
        "Mini 3D Viewer Mode:",
        ["Raw Pose", "Occlusion / Visibility", "Interpolated Landmarks"],
        horizontal=True,
        key="data_integrity_display_mode",
    )
    view_mode = st.radio(
        "View Mode:",
        ["Front", "Left", "Right", "Isometric"],
        horizontal=True,
        key="data_integrity_view_mode",
    )
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        show_axes = st.checkbox("Show axes", value=True, key="data_integrity_show_axes")
    with col_opt2:
        show_labels = st.checkbox("Show landmark labels", value=True, key="data_integrity_show_labels")
    with col_opt3:
        show_bbox = st.checkbox("Show bounding box", value=False, key="data_integrity_show_bbox")

    display_mode_map = {
        "Raw Pose": "raw_pose",
        "Occlusion / Visibility": "occlusion_visibility",
        "Interpolated Landmarks": "interpolated_landmarks",
    }

    sync_info = resolve_frame_sync_info(
        frame_index=frame_idx,
        mapping={
            "original_count": stats["pose_frame_count"],
            "max_frames": MAX_FRAMES,
            "resampled_to_source": stats["resampled_to_source"],
        },
        source_timestamps_ms=stats["source_timestamps_ms"],
    )

    frame_points = s.tensor_np[frame_idx]
    vis_row = stats["vis_matrix"][frame_idx]
    interpolated_indices = tuple(np.where(stats["resampled_interpolated_mask"][frame_idx])[0].tolist())
    unrecoverable_indices = tuple(np.where(stats["resampled_unrecoverable_mask"][frame_idx])[0].tolist())
    raw_frame = s.raw_frames[frame_idx]
    landmarks = s.lm_frames[frame_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw video frame", stats["video_frame_count"])
    c2.metric("Pose frames retained", stats["pose_frame_count"])
    c3.metric("Selected source frame", sync_info.source_frame_index)
    c4.metric("Timestamp", f"{sync_info.timestamp_ms:.1f} ms" if np.isfinite(sync_info.timestamp_ms) else "n/a")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Low-visibility landmarks", int((vis_row < 0.5).sum()))
    c6.metric("Interpolated landmarks", len(interpolated_indices))
    c7.metric("Unrecoverable landmarks", len(unrecoverable_indices))
    c8.metric("Padding frames", stats["padded_count"])

    if stats["padded_count"] > 0:
        st.warning(
            f"⚠ {stats['padded_count']} frame diduplikasi dari frame terakhir (pose frames retained: {stats['pose_frame_count']})."
        )
    else:
        st.success(
            f"✓ Subsampling deterministik: {stats['pose_frame_count']} → {MAX_FRAMES} frame ({stats['frame_mapping_mode']})."
        )

    st.caption(
        f"Resampled frame {frame_idx} → source frame {sync_info.source_frame_index} | "
        f"video timestamp {sync_info.timestamp_ms:.1f} ms | source kind: {sync_info.source_kind}"
    )

    col_v, col_2d, col_3d = st.columns([1.05, 1.1, 1.25])
    with col_v:
        st.subheader("Video Asli")
        st.video(s.video_bytes)
        display_image_compat(
            cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB),
            caption=f"Source frame {sync_info.source_frame_index}",
            stretch=True,
            channels="RGB",
        )

    with col_2d:
        st.subheader("Skeleton Overlay 2D")
        colors = [(140, 140, 140) if float(vis_row[i]) < 0.35 else (0, 180, 0) for i in range(33)]
        if interpolated_indices:
            for idx in interpolated_indices:
                colors[idx] = (0, 165, 255)
        if unrecoverable_indices:
            for idx in unrecoverable_indices:
                colors[idx] = (0, 0, 255)
        overlay = draw_frame_skeleton(
            raw_frame,
            landmarks,
            color_per_joint=colors,
            relevant_joints=list(interpolated_indices) if interpolated_indices else None,
        )
        display_image_compat(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            caption=f"Frame {frame_idx} | source {sync_info.source_frame_index}",
            stretch=True,
            channels="RGB",
        )

    with col_3d:
        st.subheader("Mini 3D Skeleton Viewer")
        render_ok = _render_mini_3d_viewer(
            frame_points=frame_points,
            visibility=vis_row,
            video_hash=str(stats.get("video_hash") or "no-video-hash"),
            preprocess_key=str(stats.get("preprocess_signature") or "default"),
            frame_index=frame_idx,
            display_mode=display_mode_map[display_mode_label],
            view_mode=view_mode.lower(),
            show_axes=show_axes,
            show_labels=show_labels,
            show_bbox=show_bbox,
            interpolated_indices=interpolated_indices,
        )
        if not render_ok:
            st.info("Viewer 3D gagal dirender; overlay 2D tetap tersedia sebagai fallback.")

    st.divider()

    st.subheader("🗺️ Visibility Heatmap (64 frame × 33 landmark)")
    vis = stats["vis_matrix"]
    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(vis.T, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xlabel("Frame Index (0–63)")
    ax.set_ylabel("Landmark Index (0–32)")
    ax.set_title("MediaPipe Landmark Visibility (hijau=tinggi, merah=rendah)")
    plt.colorbar(im, ax=ax, fraction=0.02, label="Visibility")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("ℹ️ Detail Resampling / Padding"):
        idx = stats["sampled_indices"]
        st.write(f"**Metode:** `{stats['frame_mapping_mode']}`")
        st.write(f"**Pose frames retained:** `{stats['pose_frame_count']}`")
        st.write(f"**Video frames read:** `{stats['video_frame_count']}`")
        if idx:
            st.write(f"**5 index pertama:** {idx[:5]}")
            st.write(f"**5 index terakhir:** {idx[-5:]}")

    st.divider()

    st.subheader("📉 Missing / Low-Visibility Statistics")
    vis_per_lm = vis.mean(axis=0)
    vis_per_frm = vis.mean(axis=1)

    fa, fb = st.columns(2)
    with fa:
        fig2, ax2 = plt.subplots(figsize=(7, 2.5))
        ax2.bar(range(33), vis_per_lm,
                color=["#4caf50" if v >= 0.5 else "#ef5350" for v in vis_per_lm],
                edgecolor="none")
        ax2.axhline(0.5, color="orange", linestyle="--", linewidth=1)
        ax2.set_xlabel("Landmark Index"); ax2.set_ylabel("Mean Visibility")
        ax2.set_title("Mean Visibility per Landmark")
        ax2.set_xticks(range(0, 33, 4))
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    with fb:
        fig3, ax3 = plt.subplots(figsize=(7, 2.5))
        ax3.plot(range(MAX_FRAMES), vis_per_frm, color="#4fc3f7", linewidth=1)
        ax3.axhline(0.5, color="orange", linestyle="--", linewidth=1)
        ax3.fill_between(range(MAX_FRAMES), vis_per_frm, 0.5,
                         where=(vis_per_frm < 0.5), alpha=0.4, color="#ef5350")
        ax3.set_xlabel("Frame Index"); ax3.set_ylabel("Mean Visibility")
        ax3.set_title("Mean Visibility per Frame")
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True); plt.close(fig3)

    st.divider()

    st.subheader("📋 Tensor Preview (10 frame × 5 landmark pertama)")
    tnp = s.tensor_np
    rows = [
        {"frame": f, "landmark": l,
         "x": round(float(tnp[f, l, 0]), 4),
         "y": round(float(tnp[f, l, 1]), 4),
         "z": round(float(tnp[f, l, 2]), 4)}
        for f in range(min(10, MAX_FRAMES)) for l in range(5)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)

    st.divider()

    st.subheader("🔒 Checkpoint Integrity (5 skenario)")
    st.dataframe(_checkpoint_integrity_df(), use_container_width=True)


def _tab_biomechanical() -> None:
    st.header("🦴 Tab 2 — Biomechanical Validator")
    s        = st.session_state
    stats    = s.video_stats
    vdf      = s.validator_df
    exercise = s.exercise_type
    rules    = EXERCISE_RULES.get(exercise, [])

    seq_valid, seq_reason = run_validator_sequence(s.tensor_np, exercise, BiomechanicalValidator())

    # Rules documentation
    st.subheader(f"📖 Aturan Biomekanik — {exercise}")
    if seq_valid:
        st.success(f"Sequence-level validator: VALID — {seq_reason}")
    else:
        st.error(f"Sequence-level validator: INVALID — {seq_reason}")
    for rule in rules:
        with st.expander(
            f"**{rule['name']}** — Threshold: {rule['threshold_str']}  ({rule['reference']})"
        ):
            ca, cb = st.columns([1, 2])
            ca.write(f"**Landmark:**  `{rule['label']}`")
            ca.write(f"**Threshold:** `{rule['threshold_str']}`")
            ca.write(f"**Referensi:** {rule['reference']}")
            cb.write(f"**Deskripsi:** {rule['description']}")

    if len(rules):
        st.divider()
        st.subheader("🧭 Frame-Level Rule Lens")
        frame_idx = st.slider(
            "Preview frame:",
            0,
            MAX_FRAMES - 1,
            int(vdf["frame_index"].iloc[0]) if len(vdf) else 0,
            1,
            key="biomech_preview_frame_idx",
        )
        frame_xyz = s.tensor_np[frame_idx]
        frame_vis = stats["vis_matrix"][frame_idx]
        interpolated_indices = tuple(np.where(stats["resampled_interpolated_mask"][frame_idx])[0].tolist())
        rule_rows = [rule_metric_for_frame(exercise, rule, frame_xyz) for rule in rules]

        metrics_cols = st.columns(max(1, len(rule_rows)))
        for col, rule_row in zip(metrics_cols, rule_rows):
            with col:
                st.metric(
                    rule_row["rule_name"],
                    "n/a" if rule_row["metric_value"] is None else f"{rule_row['metric_value']:.2f}",
                    delta=f"threshold {rule_row['threshold']:.2f}",
                )
                if rule_row["status"] == "VALID":
                    st.success("VALID")
                elif rule_row["status"] == "INVALID":
                    st.error("INVALID")
                else:
                    st.info("N/A")

        viz_cols = st.columns(max(1, len(rule_rows)))
        for col, rule_row, rule in zip(viz_cols, rule_rows, rules):
            with col:
                st.caption(f"{rule_row['rule_name']} | {rule.get('label', 'rule')}")
                fig_rule = create_3d_skeleton_figure(
                    frame_xyz,
                    visibility=frame_vis,
                    interpolated_landmarks=interpolated_indices,
                    highlight_landmarks=rule_row["relevant_landmarks"],
                    label_overrides=rule_row["label_map"],
                    display_mode="interpolated_landmarks",
                    title=f"{exercise} — {rule_row['rule_name']}",
                    view="isometric",
                    show_axes=False,
                    show_labels=True,
                    show_bbox=False,
                )
                st.plotly_chart(fig_rule, use_container_width=True, theme="streamlit")

    st.divider()

    # Summary metrics
    st.subheader("📊 Ringkasan Hasil Validasi")
    invalid = vdf[vdf["validator_status"] == "INVALID"]
    valid   = vdf[vdf["validator_status"] == "VALID"]

    first_err = int(invalid["frame_index"].min()) if len(invalid) > 0 else None
    last_err  = int(invalid["frame_index"].max()) if len(invalid) > 0 else None
    peak_err  = (
        int(invalid.loc[invalid["metric_value"].idxmax(), "frame_index"])
        if len(invalid) > 0 else None
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frame Valid",          len(valid))
    c2.metric("Frame Invalid",        len(invalid))
    c3.metric("Error Ratio",          f"{len(invalid)/MAX_FRAMES*100:.1f}%")
    c4.metric("Critical [first|peak|last]",
              f"[{first_err}|{peak_err}|{last_err}]" if first_err is not None else "Tidak ada")

    # Critical frame images
    if first_err is not None:
        st.divider()
        st.subheader("🔴 Frame Kritis — Skeleton Overlay")
        rel_joints = sorted({
            j for r in rules
            for j in [r.get("lm_a"), r.get("lm_b"), r.get("lm_c")]
            if isinstance(j, int)
        })
        for col, (label, fidx) in zip(
            st.columns(3),
            [("First Error", first_err), ("Peak Error", peak_err), ("Last Error", last_err)],
        ):
            with col:
                st.caption(f"**{label}** — Frame {fidx}")
                colors = [
                    (0, 0, 220) if i in rel_joints else (50, 50, 50)
                    for i in range(33)
                ]
                ov = draw_frame_skeleton(
                    s.raw_frames[fidx], s.lm_frames[fidx],
                    color_per_joint=colors, relevant_joints=rel_joints,
                )
                display_image_compat(
                    cv2.cvtColor(ov, cv2.COLOR_BGR2RGB),
                    stretch=True,
                    channels="RGB",
                )

    st.divider()

    # Timeline
    st.subheader("📈 Timeline Metrik per Frame")
    frame_idx = vdf["frame_index"].values
    metric    = vdf["metric_value"].values
    threshold = float(vdf["threshold"].iloc[0]) if len(vdf) else 137.0
    statuses  = vdf["validator_status"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(frame_idx, metric, color="#4fc3f7", linewidth=1.5, label="Primary Metric")
    ax1.axhline(threshold, color="#ef5350", linestyle="--", linewidth=1.2,
                label=f"Threshold = {threshold:.0f}")
    for idx_f, st_v in zip(frame_idx, statuses):
        if st_v == "INVALID":
            ax1.axvspan(idx_f - 0.5, idx_f + 0.5, alpha=0.22, color="#ef5350", lw=0)

    for label_k, fidx_k, clr in [
        ("First error", first_err, "orange"),
        ("Peak error",  peak_err,  "#ff00ff"),
        ("Last error",  last_err,  "yellow"),
    ]:
        if fidx_k is not None:
            ax1.axvline(fidx_k, color=clr, linestyle=":", linewidth=1.5,
                        label=f"{label_k} (F{fidx_k})")

    ax1.set_ylabel("Angle (°)")
    ax1.set_title(f"{exercise} — Primary Metric Timeline (zona merah = INVALID)")
    ax1.legend(fontsize=8, loc="upper right"); ax1.grid(alpha=0.15)

    bar_colors = ["#4caf50" if sv == "VALID" else "#ef5350" for sv in statuses]
    ax2.bar(frame_idx, np.ones(len(frame_idx)), color=bar_colors, width=1.0, edgecolor="none")
    ax2.set_yticks([0.5]); ax2.set_yticklabels(["Status"])
    ax2.set_xlabel("Frame Index")
    ax2.set_title("Per-Frame Status (hijau=VALID, merah=INVALID)")
    ax2.grid(alpha=0.1)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.subheader("📋 Data Per-Frame Lengkap")
    st.dataframe(
        vdf[["frame_index", "validator_status", "metric_value", "threshold"]],
        use_container_width=True, height=300,
    )


def _tab_classification(scenario: str) -> None:
    st.header("🎯 Tab 3 — Classification")

    st.info(
        "**ℹ️ Disclaimer:** Model AttentiveSkel-3D adalah **sequence-level classifier**. "
        "Dilatih dengan label per-gerakan penuh (valid / invalid), BUKAN per-frame. "
        "Prediksi di bawah merepresentasikan kualitas keseluruhan sekuens gerakan."
    )

    cfg      = MODEL_CONFIGS[scenario]
    abs_path = str(ROOT_DIR / CHECKPOINT_PATHS[scenario])

    with st.spinner(f"Memuat {scenario}…"):
        model, missing, unexpected, err = load_model_cached(
            abs_path,
            cfg["use_spatial_prior"],
            cfg["use_learned_spatial"],
            cfg["use_temporal_attention"],
        )

    if model is None:
        st.error(f"❌ Gagal memuat model [{scenario}]: {err}")
        return

    s = st.session_state
    with st.spinner("Inferensi…"):
        _, logits_np = extract_temporal_attention(model, s.tensor)

    seq_valid, seq_reason = run_validator_sequence(s.tensor_np, s.exercise_type, BiomechanicalValidator())

    exp_l = np.exp(logits_np - logits_np.max())
    probs = exp_l / exp_l.sum()
    pred  = int(np.argmax(logits_np))
    conf  = float(probs[pred]) * 100.0
    agreement = format_agreement_status(pred, seq_valid)

    st.subheader(f"Skenario: **{scenario}**")
    col_r, col_d = st.columns([1, 1])

    with col_r:
        label = CLASS_NAMES.get(pred, f"Class {pred}")
        if pred == 0:
            st.success(f"## {label}")
        else:
            st.error(f"## {label}")
        st.metric("Jenis Gerakan", s.exercise_type)
        st.metric("Prediksi",     label)
        st.metric("Confidence",   f"{conf:.1f}%")
        st.progress(conf / 100.0)
        if seq_valid:
            st.success("Validator sequence label: VALID")
        else:
            st.error("Validator sequence label: INVALID")
        st.metric("Agreement", agreement)
        if agreement.startswith("DISAGREE"):
            st.warning(
                "Validator dan classifier tidak selaras. Ini dapat terjadi ketika model belajar pola sequence-level "
                "yang tidak identik dengan rule-based validator; jangan memaksa keduanya sama."
            )

    with col_d:
        st.subheader("Logits & Probabilitas")
        cls_df = pd.DataFrame({
            "Class":       [CLASS_NAMES[0], CLASS_NAMES[1]],
            "Logit":       [round(float(logits_np[0]), 4), round(float(logits_np[1]), 4)],
            "Probability": [round(float(probs[0]),    4), round(float(probs[1]),    4)],
        })
        st.dataframe(cls_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.barh([CLASS_NAMES[0], CLASS_NAMES[1]],
                       [float(probs[0]), float(probs[1])],
                       color=["#4caf50", "#ef5350"])
        ax.set_xlim(0, 1); ax.set_xlabel("Probability")
        ax.set_title("Softmax Output")
        for b, val in zip(bars, probs):
            ax.text(float(val) + 0.01, b.get_y() + b.get_height() / 2,
                   f"{float(val):.3f}", va="center", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("🔍 Detail Arsitektur & Checkpoint"):
        dc1, dc2 = st.columns(2)
        live_sha = compute_sha256(ROOT_DIR / CHECKPOINT_PATHS[scenario])
        dc1.write(f"**Path:** `{CHECKPOINT_PATHS[scenario]}`")
        dc1.write(f"**SHA256 (live):**  `{live_sha}`")
        dc1.write(f"**SHA256 (audit):** `{AUDIT_SHA256[scenario]}`")
        match_str = "✓ Match" if live_sha == AUDIT_SHA256[scenario] else "⚠ MISMATCH"
        dc1.write(f"**Integrity:** {match_str}")
        dc2.write(f"**use_spatial_prior:**      {cfg['use_spatial_prior']}")
        dc2.write(f"**use_learned_spatial:**    {cfg['use_learned_spatial']}")
        dc2.write(f"**use_temporal_attention:** {cfg['use_temporal_attention']}")
        dc2.write(f"**Missing keys:** {len(missing)}")
        dc2.write(f"**Unexpected keys:** {len(unexpected)}")
        dc2.write(f"**Validator sequence label:** {'VALID' if seq_valid else 'INVALID'}")
        dc2.write(f"**Agreement status:** {agreement}")
        if missing:
            st.code("Missing:\n" + "\n".join(missing))
        if unexpected:
            st.code("Unexpected:\n" + "\n".join(unexpected))


def _tab_attention(scenario: str) -> None:
    st.header("🧠 Tab 4 — Attention Sanity Check")

    view_mode = st.radio(
        "Mode tampilan:",
        ["Skenario terpilih saja", "Semua 5 skenario"],
        horizontal=True,
    )
    attention_mode = st.radio(
        "Attention view:",
        [
            "Raw Pose",
            "Biomechanical Prior P(v)",
            "Learned Spatial Attention A_s(v)",
            "Fused Attention",
            "Occlusion",
        ],
        horizontal=True,
        help="Raw Pose tidak memetakan bobot attention; mode lain menampilkan bobot aktual atau proksi model yang terukur.",
    )
    norm_mode = st.radio(
        "Normalisasi spatial attention:",
        ["Raw (sigmoid output)", "Global (cross-scenario)", "Per-model (min-max)"],
        horizontal=True,
        help=(
            "Raw = sigmoid asli [0,1]. "
            "Global = normalized bersama semua skenario (fair comparison). "
            "Per-model (min-max) = hanya visual per skenario, hindari untuk perbandingan."
        ),
    )

    s = st.session_state
    stats = s.video_stats
    attention_frame_idx = st.slider(
        "Frame preview (attention):",
        0,
        MAX_FRAMES - 1,
        0,
        1,
        key="attention_preview_frame_idx",
    )
    scenarios = (
        [scenario]
        if view_mode == "Skenario terpilih saja"
        else list(CHECKPOINT_PATHS.keys())
    )

    # Load & infer
    spatial_attns:  dict[str, Optional[np.ndarray]] = {}
    learned_attns:  dict[str, Optional[np.ndarray]] = {}
    fused_attns:    dict[str, Optional[np.ndarray]] = {}
    occlusion_attns: dict[str, Optional[np.ndarray]] = {}
    temporal_attns: dict[str, Optional[np.ndarray]] = {}
    logits_map:     dict[str, np.ndarray]            = {}
    errors:         dict[str, str]                   = {}

    prog = st.progress(0.0, text="Memuat model…")
    for i, sc in enumerate(scenarios):
        cfg      = MODEL_CONFIGS[sc]
        abs_path = str(ROOT_DIR / CHECKPOINT_PATHS[sc])
        model, _, _, err = load_model_cached(
            abs_path, cfg["use_spatial_prior"],
            cfg["use_learned_spatial"], cfg["use_temporal_attention"],
        )
        prog.progress((i + 0.5) / len(scenarios), text=f"Inferensi {sc}…")
        if model is None:
            errors[sc]         = err
            spatial_attns[sc]  = None
            learned_attns[sc]  = None
            fused_attns[sc]    = None
            occlusion_attns[sc] = None
            temporal_attns[sc] = None
            logits_map[sc]     = np.zeros(2)
        else:
            spatial_attns[sc]  = extract_spatial_attention(model)
            learned_attns[sc]  = extract_learned_spatial_attention(model, s.tensor)
            fused_attns[sc]    = combine_attention(spatial_attns[sc], learned_attns[sc])
            occlusion_attns[sc] = occlusion_values(stats["vis_matrix"][attention_frame_idx])
            t_attn, logits_np  = extract_temporal_attention(model, s.tensor)
            temporal_attns[sc] = t_attn
            logits_map[sc]     = logits_np
        prog.progress((i + 1) / len(scenarios))
    prog.empty()

    for sc, err in errors.items():
        st.error(f"❌ Gagal memuat [{sc}]: {err}")

    if attention_mode == "Raw Pose":
        selected_vectors = {sc: None for sc in scenarios}
    elif attention_mode == "Biomechanical Prior P(v)":
        selected_vectors = dict(spatial_attns)
    elif attention_mode == "Learned Spatial Attention A_s(v)":
        selected_vectors = dict(learned_attns)
    elif attention_mode == "Fused Attention":
        selected_vectors = dict(fused_attns)
    else:
        selected_vectors = dict(occlusion_attns)

    if norm_mode == "Global (cross-scenario)":
        avail = [v for v in selected_vectors.values() if v is not None]
        if avail:
            all_v = np.concatenate(avail)
            g_min, g_max = float(all_v.min()), float(all_v.max())
            rng = g_max - g_min
            if rng > 1e-8:
                for key, value in list(selected_vectors.items()):
                    if value is not None:
                        selected_vectors[key] = (value - g_min) / rng
    elif norm_mode == "Per-model (min-max)":
        for key, value in list(selected_vectors.items()):
            if value is None:
                continue
            span = float(value.max() - value.min())
            selected_vectors[key] = (value - value.min()) / span if span > 1e-8 else np.full_like(value, 0.5)

    st.subheader(f"📊 {attention_mode} per 33 Joint — {norm_mode}")
    n_sc   = len(scenarios)
    n_rows = max(1, (n_sc + 1) // 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
    axes_flat = np.array(axes).flatten() if hasattr(np.array(axes), "flatten") else [axes]

    for i, sc in enumerate(scenarios):
        ax = axes_flat[i] if i < len(axes_flat) else None
        if ax is None:
            continue
        if sc in selected_vectors and selected_vectors[sc] is not None:
            w    = selected_vectors[sc]
            top5 = np.argsort(w)[::-1][:5]
            ax.bar(range(33), w,
                   color=["#ef5350" if w[j] > 0.5 else "#4fc3f7" for j in range(33)],
                   edgecolor="none")
            ax.axhline(0.5, color="orange", linestyle="--", linewidth=0.8)
            ax.set_title(f"{sc}\nTop-5 joints: {list(top5)}", fontsize=9)
        else:
            message = "Raw Pose selected" if attention_mode == "Raw Pose" else "Tidak tersedia pada skenario ini"
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(sc, fontsize=9)
        ax.set_xlabel("Joint Index"); ax.set_ylabel("Attention Weight")
        ax.tick_params(axis="x", labelsize=6); ax.set_xticks(range(33))

    for j in range(n_sc, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Attention Map — {attention_mode} — {norm_mode}", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.subheader("🧭 3D Attention Preview")
    preview_scenario = scenario if view_mode == "Skenario terpilih saja" else scenarios[0]
    cfg_preview = MODEL_CONFIGS[preview_scenario]
    abs_preview = str(ROOT_DIR / CHECKPOINT_PATHS[preview_scenario])
    model_preview, _, _, err_preview = load_model_cached(
        abs_preview,
        cfg_preview["use_spatial_prior"],
        cfg_preview["use_learned_spatial"],
        cfg_preview["use_temporal_attention"],
    )
    if model_preview is None:
        st.error(f"❌ Gagal memuat model preview [{preview_scenario}]: {err_preview}")
    else:
        attention_preview = selected_vectors.get(preview_scenario)
        extra_preview = {
            "prior": spatial_attns.get(preview_scenario),
            "learned": learned_attns.get(preview_scenario),
            "fused": fused_attns.get(preview_scenario),
            "occlusion": occlusion_attns.get(preview_scenario),
        }
        fig_preview = create_3d_skeleton_figure(
            s.tensor_np[attention_frame_idx],
            visibility=stats["vis_matrix"][attention_frame_idx],
            attention=attention_preview,
            extra_hover_data=extra_preview,
            interpolated_landmarks=tuple(np.where(stats["resampled_interpolated_mask"][attention_frame_idx])[0].tolist()),
            display_mode="occlusion_visibility" if attention_mode == "Occlusion" else "raw_pose",
            title=f"{preview_scenario} — {attention_mode} @ frame {attention_frame_idx}",
            view="isometric",
            show_axes=False,
            show_labels=True,
            show_bbox=False,
        )
        st.plotly_chart(fig_preview, use_container_width=True, theme="streamlit")

    st.divider()

    # Temporal attention timeline
    st.subheader("⏱️ Temporal Attention Timeline")
    has_temporal = any(v is not None for v in temporal_attns.values())
    if not has_temporal:
        st.info("Tidak ada skenario terpilih yang memiliki Temporal Attention module aktif.")
    else:
        fig_t, ax_t = plt.subplots(figsize=(14, 4))
        cmap = plt.get_cmap("tab10")
        for i, sc in enumerate(scenarios):
            t = temporal_attns.get(sc)
            if t is None:
                ax_t.plot([], [], color=cmap(i), label=f"{sc} — Tidak tersedia")
                continue
            t64    = _interp_to_64(t)
            peak_f = int(np.argmax(t64))
            ax_t.plot(range(MAX_FRAMES), t64, color=cmap(i), linewidth=1.5,
                      label=f"{sc} (peak F{peak_f})")
            ax_t.axvline(peak_f, color=cmap(i), linestyle=":", alpha=0.5, linewidth=1)
        ax_t.set_xlabel("Frame Index"); ax_t.set_ylabel("Softmax Weight")
        ax_t.set_title("Temporal Attention — Model Focus per Frame (softmax)")
        ax_t.legend(fontsize=7, ncol=2); ax_t.grid(alpha=0.15)
        fig_t.tight_layout()
        st.pyplot(fig_t, use_container_width=True)
        plt.close(fig_t)

    st.divider()

    # Checkpoint info table
    st.subheader("🔒 Checkpoint Info")
    rows_ck = []
    for sc in scenarios:
        p   = ROOT_DIR / CHECKPOINT_PATHS[sc]
        sha = compute_sha256(p) if p.exists() else "—"
        rows_ck.append({
            "Scenario":       sc,
            "SHA256 (live)":  sha,
            "SHA256 (audit)": AUDIT_SHA256[sc],
            "Match":          "✓" if sha == AUDIT_SHA256[sc] else "⚠",
            "BSP":            "✓" if MODEL_CONFIGS[sc]["use_spatial_prior"]      else "✗",
            "Learned":        "✓" if MODEL_CONFIGS[sc]["use_learned_spatial"]    else "✗",
            "Temporal":       "✓" if MODEL_CONFIGS[sc]["use_temporal_attention"] else "✗",
            "Attention view": "✓" if selected_vectors.get(sc) is not None else "N/A",
        })
    st.dataframe(pd.DataFrame(rows_ck), use_container_width=True, hide_index=True)

    # Pairwise similarity matrix (all 5 only)
    if view_mode == "Semua 5 skenario":
        st.divider()
        st.subheader("🔢 Pairwise Similarity Matrix (Logits — Cosine)")
        all_sc = list(CHECKPOINT_PATHS.keys())
        n      = len(all_sc)
        sim    = np.zeros((n, n))
        for i, s1 in enumerate(all_sc):
            for j, s2 in enumerate(all_sc):
                if i == j:
                    sim[i, j] = 1.0
                else:
                    l1 = logits_map.get(s1, np.zeros(2))
                    l2 = logits_map.get(s2, np.zeros(2))
                    n1 = np.linalg.norm(l1) + 1e-8
                    n2 = np.linalg.norm(l2) + 1e-8
                    sim[i, j] = float(np.dot(l1, l2) / (n1 * n2))

        short   = ["Full", "Baseline", "Ablasi A", "Ablasi B", "Ablasi C"]
        fig_s, ax_s = plt.subplots(figsize=(7, 5))
        sns.heatmap(sim, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=-1, vmax=1,
                    xticklabels=short, yticklabels=short, ax=ax_s)
        ax_s.set_title("Cosine Similarity of Logits (5 skenario)")
        fig_s.tight_layout()
        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)


def _tab_frame_inspector(scenario: str) -> None:
    st.header("🔍 Tab 5 — Frame Inspector")
    s        = st.session_state
    stats    = s.video_stats
    vdf      = s.validator_df
    tnp      = s.tensor_np      # (64, 33, 3)
    exercise = s.exercise_type
    rules    = EXERCISE_RULES.get(exercise, [])
    rel_joints = exercise_relevant_landmarks(rules)

    # Load model
    cfg      = MODEL_CONFIGS[scenario]
    abs_path = str(ROOT_DIR / CHECKPOINT_PATHS[scenario])
    model, _, _, err = load_model_cached(
        abs_path, cfg["use_spatial_prior"],
        cfg["use_learned_spatial"], cfg["use_temporal_attention"],
    )
    if model is None:
        st.error(f"❌ Gagal memuat model: {err}")
        return

    spatial_attn = extract_spatial_attention(model)
    with st.spinner("Ekstraksi temporal attention…"):
        t_attn_raw, _ = extract_temporal_attention(model, s.tensor)
    t_attn_64 = _interp_to_64(t_attn_raw) if t_attn_raw is not None else None

    fidx = st.slider("Frame Index (0–63):", 0, MAX_FRAMES - 1, 0, 1, key="frame_inspector_frame_idx")
    view_mode = st.radio(
        "View Mode:",
        ["Front", "Left", "Right", "Isometric"],
        horizontal=True,
        key="frame_inspector_view_mode",
    )
    display_mode_label = st.radio(
        "Mini 3D Viewer Mode:",
        ["Raw Pose", "Occlusion / Visibility", "Interpolated Landmarks"],
        horizontal=True,
        key="frame_inspector_display_mode",
    )
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        show_axes = st.checkbox("Show axes", value=True, key="frame_inspector_show_axes")
    with col_opt2:
        show_labels = st.checkbox("Show landmark labels", value=True, key="frame_inspector_show_labels")
    with col_opt3:
        show_bbox = st.checkbox("Show bounding box", value=False, key="frame_inspector_show_bbox")

    display_mode_map = {
        "Raw Pose": "raw_pose",
        "Occlusion / Visibility": "occlusion_visibility",
        "Interpolated Landmarks": "interpolated_landmarks",
    }

    attention_mode_label = st.radio(
        "Attention Mode:",
        [
            "Raw Pose",
            "Biomechanical Prior P(v)",
            "Learned Spatial Attention A_s(v)",
            "Fused Attention",
            "Occlusion",
        ],
        horizontal=True,
        key="frame_inspector_attention_mode",
    )

    sync_info = resolve_frame_sync_info(
        frame_index=fidx,
        mapping={
            "original_count": stats["pose_frame_count"],
            "max_frames": MAX_FRAMES,
            "resampled_to_source": stats["resampled_to_source"],
        },
        source_timestamps_ms=stats["source_timestamps_ms"],
    )

    row       = vdf[vdf["frame_index"] == fidx].iloc[0] if len(vdf) else None
    is_valid  = bool(row["is_valid"]) if row is not None else True
    interpolated_indices = tuple(np.where(stats["resampled_interpolated_mask"][fidx])[0].tolist())
    unrecoverable_indices = tuple(np.where(stats["resampled_unrecoverable_mask"][fidx])[0].tolist())
    vis_row = stats["vis_matrix"][fidx]
    current_raw = s.raw_frames[fidx]
    current_landmarks = s.lm_frames[fidx]
    attention_bundle = extract_attention_bundle(model, s.tensor, vis_row)

    attention_mode_map = {
        "Raw Pose": None,
        "Biomechanical Prior P(v)": attention_bundle["prior"],
        "Learned Spatial Attention A_s(v)": attention_bundle["learned"],
        "Fused Attention": attention_bundle["fused"],
        "Occlusion": attention_bundle["occlusion"],
    }
    viewer_attention = attention_mode_map[attention_mode_label]

    j_colors = [
        ((0, 180, 0) if is_valid else (0, 0, 220)) if i in rel_joints else (50, 50, 50)
        for i in range(33)
    ]

    if interpolated_indices:
        for idx in interpolated_indices:
            j_colors[idx] = (0, 165, 255)
    if unrecoverable_indices:
        for idx in unrecoverable_indices:
            j_colors[idx] = (0, 0, 255)

    display_mode_key = display_mode_map[display_mode_label]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frame resampled", fidx)
    c2.metric("Source frame", sync_info.source_frame_index)
    c3.metric("Timestamp", f"{sync_info.timestamp_ms:.1f} ms" if np.isfinite(sync_info.timestamp_ms) else "n/a")
    c4.metric("Validator status", "VALID" if is_valid else "INVALID")

    col_fr, col_info = st.columns([3, 2])

    with col_fr:
        st.subheader("Frame Video")
        display_image_compat(
            cv2.cvtColor(current_raw, cv2.COLOR_BGR2RGB),
            caption=f"Resampled frame {fidx} → source frame {sync_info.source_frame_index}",
            stretch=True,
            channels="RGB",
        )
        st.subheader("Skeleton Overlay 2D")
        overlay = draw_frame_skeleton(
            current_raw, current_landmarks,
            color_per_joint=j_colors,
            relevant_joints=rel_joints,
            attention_labels=spatial_attn,
        )
        hf, wf = overlay.shape[:2]
        status_str   = "VALID ✓" if is_valid else "INVALID ✗"
        status_color = (0, 200, 0) if is_valid else (0, 0, 220)
        cv2.putText(overlay,
                    f"Frame {fidx} | {exercise} | {status_str}",
                    (8, hf - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    status_color, 2, cv2.LINE_AA)
        display_image_compat(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            caption=f"Skeleton overlay — frame {fidx} | {scenario}",
            stretch=True,
            channels="RGB",
        )

        st.subheader("Mini 3D Skeleton Viewer")
        render_ok = _render_mini_3d_viewer(
            frame_points=tnp[fidx],
            visibility=vis_row,
            video_hash=str(stats.get("video_hash") or "no-video-hash"),
            preprocess_key=str(stats.get("preprocess_signature") or "default"),
            frame_index=fidx,
            display_mode=display_mode_key,
            view_mode=view_mode.lower(),
            show_axes=show_axes,
            show_labels=show_labels,
            show_bbox=show_bbox,
            interpolated_indices=interpolated_indices,
            attention=viewer_attention,
            extra_hover_data=attention_bundle,
        )
        if not render_ok:
            st.caption("Fallback: overlay 2D tetap tersedia walau viewer 3D gagal.")

    with col_info:
        st.subheader(f"Frame {fidx} — Info")

        st.metric("Selected source frame", sync_info.source_frame_index)
        st.metric("Source timestamp", f"{sync_info.timestamp_ms:.1f} ms" if np.isfinite(sync_info.timestamp_ms) else "n/a")
        st.metric("Interpolated landmarks", len(interpolated_indices))
        st.metric("Unrecoverable landmarks", len(unrecoverable_indices))

        if row is not None:
            if is_valid:
                st.success("**Validator: VALID ✓**")
            else:
                st.error("**Validator: INVALID ✗**")
            st.metric("Primary Metric", f"{row['metric_value']:.2f}°")
            st.metric("Threshold",      f"{row['threshold']:.1f}°")
            with st.expander("Reason"):
                st.write(row["reason"])

        st.divider()
        st.subheader("Temporal Attention")
        if t_attn_64 is not None:
            score = float(t_attn_64[fidx])
            rank  = int(np.sum(t_attn_64 > score)) + 1
            st.metric("Score (softmax)", f"{score:.4f}")
            st.caption(f"Rank #{rank} dari {MAX_FRAMES} frame (1 = paling diperhatikan)")
        else:
            st.info(
                "Tidak tersedia — skenario ini tidak memiliki "
                "Temporal Attention module (use_temporal_attention=False)."
            )

        st.divider()
        st.subheader("Spatial Attention — Relevant Joints")
        if viewer_attention is not None:
            sa_rows = [
                {"Joint": j, "Attention": round(float(viewer_attention[j]), 4)}
                for j in rel_joints
            ]
            st.dataframe(pd.DataFrame(sa_rows), use_container_width=True, hide_index=True)
        else:
            st.info(
                "Tidak tersedia — mode yang dipilih tidak memiliki vektor attention."
            )

        st.divider()
        st.subheader("Koordinat Landmark (normalized)")
        coord_rows = [
            {"Joint": j,
             "x": round(float(tnp[fidx, j, 0]), 4),
             "y": round(float(tnp[fidx, j, 1]), 4),
             "z": round(float(tnp[fidx, j, 2]), 4)}
            for j in rel_joints
        ]
        st.dataframe(pd.DataFrame(coord_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Visibility & Interpolation")
        vis_rows = []
        for j in rel_joints:
            vis_rows.append({
                "Joint": j,
                "Visibility": round(float(vis_row[j]), 4),
                "Interpolated": bool(j in interpolated_indices),
                "Unrecoverable": bool(j in unrecoverable_indices),
            })
        st.dataframe(pd.DataFrame(vis_rows), use_container_width=True, hide_index=True)

    # Temporal timeline with frame marker
    if t_attn_64 is not None:
        st.divider()
        st.subheader("⏱️ Temporal Attention — Posisi Frame Saat Ini")
        fig_t, ax_t = plt.subplots(figsize=(14, 3))
        ax_t.plot(range(MAX_FRAMES), t_attn_64, color="#4fc3f7", linewidth=1.5,
                  label="Temporal Attention")
        ax_t.axvline(fidx, color="#ff5722", linewidth=2, linestyle="--",
                     label=f"Frame {fidx} (score={t_attn_64[fidx]:.4f})")
        ax_t.scatter([fidx], [t_attn_64[fidx]], color="#ff5722", s=80, zorder=5)

        for _, r in vdf.iterrows():
            if r["validator_status"] == "INVALID":
                ax_t.axvspan(r["frame_index"] - 0.5, r["frame_index"] + 0.5,
                             alpha=0.2, color="#ef5350", lw=0)

        ax_t.set_xlabel("Frame Index"); ax_t.set_ylabel("Attention Score")
        ax_t.set_title(
            f"{scenario} — Temporal Attention (zona merah = INVALID frames)"
        )
        ax_t.legend(fontsize=8); ax_t.grid(alpha=0.15)
        fig_t.tight_layout()
        st.pyplot(fig_t, use_container_width=True)
        plt.close(fig_t)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    defaults = {
        "tensor":        None,
        "tensor_np":     None,
        "lm_frames":     None,
        "raw_frames":    None,
        "video_stats":   None,
        "video_bytes":   None,
        "validator_df":  None,
        "exercise_type": "Squat",
        "processed":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Sidebar
    with st.sidebar:
        st.title("🔬 AttentiveSkel-3D")
        st.caption("Scientific Proof Dashboard")
        st.divider()
        st.subheader("⚙️ Konfigurasi")
        selected_scenario = st.selectbox(
            "Skenario (Tab 3, 4, 5):",
            list(CHECKPOINT_PATHS.keys()),
            index=0,
        )
        exercise_type = st.selectbox(
            "Jenis Gerakan:",
            ["Squat", "BenchPress", "Deadlift"],
            index=0,
        )
        st.divider()
        st.subheader("📹 Input Video")
        uploaded = st.file_uploader("Upload video (.mp4)", type=["mp4"])
        process  = st.button(
            "🔄 Proses Video",
            type="primary",
            use_container_width=True,
            disabled=(uploaded is None),
        )

    # Process video
    if process and uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        with st.sidebar, st.spinner("Mengekstrak pose MediaPipe…"):
            tensor, lm_frames, raw_frames, stats = build_inference_tensor(tmp_path)

        if tensor is None:
            st.sidebar.error(f"❌ Gagal: {stats.get('error', 'unknown')}")
        else:
            st.session_state.update({
                "tensor":        tensor,
                "tensor_np":     tensor.squeeze(0).numpy(),
                "lm_frames":     lm_frames,
                "raw_frames":    raw_frames,
                "video_stats":   stats,
                "video_bytes":   uploaded.getvalue(),
                "exercise_type": exercise_type,
                "processed":     True,
            })
            with st.sidebar, st.spinner("Menjalankan Biomechanical Validator…"):
                st.session_state.validator_df = run_validator_per_frame(
                    st.session_state.tensor_np, exercise_type, BiomechanicalValidator()
                )
            st.sidebar.success(
                f"✅ {stats['tensor_shape']} · {stats['original_count']} frame asli"
            )

    if st.session_state.processed:
        with st.sidebar:
            st.caption(f"Shape: `{st.session_state.video_stats['tensor_shape']}`")
            st.caption(f"Frame asli: {st.session_state.video_stats['original_count']}")
            st.caption(f"Exercise: {st.session_state.exercise_type}")

    # Main header
    st.title("🔬 AttentiveSkel-3D — Scientific Proof Dashboard")
    st.caption(
        "Integritas tensor · Validasi biomekanik per-frame · "
        "Klasifikasi sequence-level · Analisis atensi 5 skenario · Inspeksi frame"
    )

    if not st.session_state.processed:
        st.info("⬅️ Upload video dan klik **Proses Video** di sidebar untuk memulai.")
        st.subheader("Checkpoint Integrity (tanpa video)")
        st.dataframe(_checkpoint_integrity_df(), use_container_width=True)
        return

    # Five tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Integrity",
        "🦴 Biomechanical Validator",
        "🎯 Classification",
        "🧠 Attention Sanity Check",
        "🔍 Frame Inspector",
    ])

    with tab1:
        render_tab_with_debug("Data Integrity", _tab_data_integrity)
    with tab2:
        render_tab_with_debug("Biomechanical Validator", _tab_biomechanical)
    with tab3:
        render_tab_with_debug("Classification", _tab_classification, selected_scenario)
    with tab4:
        render_tab_with_debug("Attention Sanity Check", _tab_attention, selected_scenario)
    with tab5:
        render_tab_with_debug("Frame Inspector", _tab_frame_inspector, selected_scenario)


if __name__ == "__main__":
    main()
