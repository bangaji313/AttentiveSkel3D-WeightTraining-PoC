"""AttentiveSkel-3D Attention Showcase.

Run:
    streamlit run web_app/attention_showcase.py

Goal:
    Dual-panel demo for stage presentation: original video on the left and
    model-driven attention overlay on the right, synchronized inside one
    fixed-size preview. The overlay is derived from the model output, not from
    biomechanical validator rules.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
for _path in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from web_app.attention_profiles import combine_attention, normalize_attention
from web_app.frame_sync import build_frame_mapping
from src.models.model_3dcnn import AttentiveSkel3D


# -----------------------------------------------------------------------------
# Page config and visual style
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AttentiveSkel-3D | Attention Showcase",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .app-hero {
        padding: 1.2rem 1.3rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0b1220 0%, #111c2e 45%, #1b2435 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 50px rgba(0,0,0,0.28);
        margin-bottom: 1rem;
      }
      .app-hero h1 {
        margin: 0;
        font-size: 2.1rem;
        color: #f8fafc;
        letter-spacing: 0.2px;
      }
      .app-hero p {
        margin: 0.35rem 0 0 0;
        color: #b6c2d9;
        font-size: 0.98rem;
      }
      .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.85rem;
      }
      .pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        color: #e5eefb;
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 0.8rem;
      }
      .video-shell {
        padding: 0.75rem;
        border-radius: 20px;
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 42px rgba(0,0,0,0.22);
      }
      .section-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.3rem 0;
      }
      .small-note {
        color: #475569;
        font-size: 0.88rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MAX_FRAMES = 64
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "attention_showcase"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MOVEMENT_OPTIONS = ["Squat", "Deadlift", "Bench Press"]

MODEL_SPECS: dict[str, dict[str, object]] = {
    "Baseline 3D-CNN": {
        "checkpoint": ROOT_DIR / "models" / "saved_models" / "baseline_3dcnn_model.pth",
        "config": dict(use_spatial_prior=False, use_learned_spatial=False, use_temporal_attention=False),
    },
    "Ablasi A - Tanpa Prior": {
        "checkpoint": ROOT_DIR / "models" / "saved_models" / "ablasi_a_no_prior.pth",
        "config": dict(use_spatial_prior=False, use_learned_spatial=True, use_temporal_attention=True),
    },
    "Ablasi B - Tanpa Learned Spatial": {
        "checkpoint": ROOT_DIR / "models" / "saved_models" / "ablasi_b_no_learned.pth",
        "config": dict(use_spatial_prior=True, use_learned_spatial=False, use_temporal_attention=True),
    },
    "Ablasi C - Tanpa Temporal": {
        "checkpoint": ROOT_DIR / "models" / "saved_models" / "ablasi_c_no_temporal.pth",
        "config": dict(use_spatial_prior=True, use_learned_spatial=True, use_temporal_attention=False),
    },
    "Full Model (Final)": {
        "checkpoint": ROOT_DIR / "models" / "saved_models" / "AttentiveSkel3D_Final.pth",
        "config": dict(use_spatial_prior=True, use_learned_spatial=True, use_temporal_attention=True),
    },
}

MOVEMENT_TARGET_JOINTS: dict[str, list[int]] = {
    "Squat": [23, 24, 25, 26],
    "Deadlift": [11, 12, 23, 24, 25, 26],
    "Bench Press": [11, 12, 13, 14, 15, 16],
}

MOVEMENT_HINTS: dict[str, str] = {
    "Squat": "Expected focus zone: hips + knees",
    "Deadlift": "Expected focus zone: shoulders + hips + knees",
    "Bench Press": "Expected focus zone: shoulders + elbows + wrists",
}

MOVEMENT_SOURCE_DIRS: dict[str, list[Path]] = {
    "Squat": [ROOT_DIR / "data" / "test", ROOT_DIR / "data" / "raw" / "Squat"],
    "Deadlift": [ROOT_DIR / "data" / "test", ROOT_DIR / "data" / "raw" / "Deadlift"],
    "Bench Press": [ROOT_DIR / "data" / "test", ROOT_DIR / "data" / "raw" / "BenchPress"],
}

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return slug.strip("_") or "item"


@st.cache_resource(show_spinner=False)
def load_model_cached(scenario_label: str) -> tuple[Optional[AttentiveSkel3D], str]:
    spec = MODEL_SPECS[scenario_label]
    checkpoint = Path(spec["checkpoint"])
    config = dict(spec["config"])

    if not checkpoint.exists():
        return None, f"Checkpoint not found: {checkpoint}"

    try:
        model = AttentiveSkel3D(num_classes=2, **config)
        try:
            ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, ""
    except Exception as exc:
        return None, str(exc)


def extract_spatial_attention(model: AttentiveSkel3D) -> Optional[np.ndarray]:
    if not getattr(model, "use_spatial_prior", False):
        return None
    if not hasattr(model, "biomechanical_spatial_prior"):
        return None
    with torch.no_grad():
        weights = torch.sigmoid(model.biomechanical_spatial_prior).squeeze().cpu().numpy()
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    return weights if weights.size == 33 else None


def extract_learned_spatial_attention(model: AttentiveSkel3D, tensor_input: torch.Tensor) -> Optional[np.ndarray]:
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


def select_attention_vector(model: AttentiveSkel3D, tensor_input: torch.Tensor) -> tuple[np.ndarray, str]:
    prior = extract_spatial_attention(model)
    learned = extract_learned_spatial_attention(model, tensor_input)

    if prior is not None and learned is not None:
        fused = combine_attention(prior, learned)
        if fused is not None:
            return fused.astype(np.float32), "fused"

    if learned is not None:
        learned_norm = normalize_attention(learned, mode="global")
        if learned_norm is not None:
            return learned_norm.astype(np.float32), "learned"

    if prior is not None:
        prior_norm = normalize_attention(prior, mode="global")
        if prior_norm is not None:
            return prior_norm.astype(np.float32), "prior"

    return np.full(33, 0.5, dtype=np.float32), "neutral"


def build_inference_tensor(video_path: Path, max_frames: int = MAX_FRAMES) -> tuple[torch.Tensor, list, list, dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_bgr_seq: list[np.ndarray] = []
    landmark_seq: list = []
    coords_seq: list[np.ndarray] = []
    timestamps_ms: list[float] = []

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_bgr_seq.append(frame_bgr)
            timestamps_ms.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))

            result = pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks is None:
                landmark_seq.append(None)
                if coords_seq:
                    coords_seq.append(coords_seq[-1].copy())
                else:
                    coords_seq.append(np.zeros((33, 3), dtype=np.float32))
                continue

            lms = result.pose_landmarks.landmark
            landmark_seq.append(lms)
            coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            coords_seq.append(coords)

    cap.release()

    if len(frame_bgr_seq) == 0:
        raise RuntimeError("No frames extracted from video.")

    mapping = build_frame_mapping(len(frame_bgr_seq), max_frames=max_frames)
    sampled_indices = list(mapping["resampled_to_source"])

    sampled_frames = [frame_bgr_seq[i] for i in sampled_indices]
    sampled_landmarks = [landmark_seq[i] for i in sampled_indices]
    sampled_coords = [coords_seq[i] for i in sampled_indices]

    tensor = torch.tensor(np.stack(sampled_coords, axis=0), dtype=torch.float32).unsqueeze(0)
    stats = {
        "original_count": len(frame_bgr_seq),
        "sampled_count": len(sampled_indices),
        "mapping_mode": mapping["mapping_mode"],
        "sampled_indices": sampled_indices,
        "timestamps_ms": timestamps_ms,
    }
    return tensor, sampled_frames, sampled_landmarks, stats


def _attention_colors(attn: np.ndarray) -> np.ndarray:
    span = float(attn.max() - attn.min())
    normalized = np.full_like(attn, 0.5, dtype=np.float32) if span <= 1e-8 else (attn - attn.min()) / span
    colors = []
    for value in normalized:
        red = int(120 + 135 * float(value))
        green = int(45 + 90 * (1.0 - float(value)))
        blue = int(40 + 80 * (1.0 - float(value)))
        colors.append((blue, green, red))
    return np.asarray(colors, dtype=np.int32)


def _draw_attention_overlay(frame_bgr: np.ndarray, landmarks, attn: np.ndarray, *, movement: str) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # darkened background for a stage-friendly look
    out = (out.astype(np.float32) * 0.42).astype(np.uint8)

    for s_idx, e_idx in POSE_CONNECTIONS:
        if s_idx >= 33 or e_idx >= 33:
            continue
        if landmarks is None:
            continue
        lms = landmarks[s_idx]
        lme = landmarks[e_idx]
        if getattr(lms, "visibility", 1.0) < 0.3 or getattr(lme, "visibility", 1.0) < 0.3:
            continue
        x1, y1 = int(lms.x * w), int(lms.y * h)
        x2, y2 = int(lme.x * w), int(lme.y * h)
        line_score = float((attn[s_idx] + attn[e_idx]) / 2.0)
        line_color = (60, 60, 60) if line_score < 0.45 else (0, 0, 185)
        cv2.line(out, (x1, y1), (x2, y2), line_color, 2 if line_score >= 0.45 else 1, cv2.LINE_AA)

    glow = np.zeros_like(out)
    colors = _attention_colors(attn)
    top_indices = list(np.argsort(attn)[::-1][:5])

    for idx in range(33):
        if landmarks is None:
            continue
        lm = landmarks[idx]
        if getattr(lm, "visibility", 1.0) < 0.25:
            continue

        px, py = int(lm.x * w), int(lm.y * h)
        score = float((attn[idx] - attn.min()) / (attn.max() - attn.min() + 1e-8))
        radius = int(5 + 12 * score)
        color = tuple(int(v) for v in colors[idx])

        if idx in top_indices:
            cv2.circle(glow, (px, py), radius + 14, color, -1)

        cv2.circle(out, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)

        if idx in top_indices:
            cv2.putText(
                out,
                f"{idx}:{attn[idx]:.2f}",
                (px + radius + 3, py - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    glow = cv2.GaussianBlur(glow, (41, 41), 16)
    out = cv2.addWeighted(out, 1.0, glow, 0.75, 0)

    hint = MOVEMENT_HINTS.get(movement, "Expected focus zone")
    cv2.putText(out, hint, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "Model-driven overlay only", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
    return out


def _video_panel(frame_bgr: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    panel = cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (639, 40), (12, 18, 30), -1)
    cv2.putText(panel, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (14, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def _composite_frame(left_bgr: np.ndarray, right_bgr: np.ndarray, *, movement: str, scenario: str, summary: dict, frame_idx: int, total: int) -> np.ndarray:
    canvas = np.zeros((512, 1280, 3), dtype=np.uint8)
    canvas[:] = (8, 14, 24)

    cv2.rectangle(canvas, (0, 0), (1279, 72), (16, 23, 35), -1)
    cv2.rectangle(canvas, (0, 432), (1279, 511), (12, 18, 30), -1)

    left = _video_panel(left_bgr, "ORIGINAL", f"Frame {frame_idx + 1}/{total}")
    right = _video_panel(right_bgr, "ATTENTION OVERLAY", f"Frame {frame_idx + 1}/{total}")

    canvas[72:432, 0:640] = left
    canvas[72:432, 640:1280] = right

    cv2.putText(
        canvas,
        f"{scenario}  |  {movement}  |  pred: {summary['prediction_label']}  |  conf: {summary['confidence']:.1f}%",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Attention source: {summary['attention_source']}  |  Top joints: {summary['top_joints_text']}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{summary['video_name']}  |  64-frame preview  |  synchronized dual-panel",
        (18, 485),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _reencode_h264(src: Path, dst: Path) -> bool:
    """Re-encode *src* to H.264 MP4 at *dst*. Uses imageio-ffmpeg bundled binary."""
    import subprocess

    # Prefer imageio_ffmpeg bundled binary, fall back to system ffmpeg.
    ffmpeg_exe = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    try:
        proc = subprocess.run(
            [
                ffmpeg_exe, "-y",
                "-i", str(src),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(dst),
            ],
            capture_output=True,
            timeout=300,
        )
        return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _write_showcase_assets(
    video_path: Path,
    movement: str,
    scenario_label: str,
) -> dict:
    video_bytes = video_path.read_bytes()
    video_hash = hashlib.sha256(video_bytes).hexdigest()[:16]
    slug = f"{video_hash}_{_safe_slug(movement)}_{_safe_slug(scenario_label)}"
    out_dir = OUTPUT_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    out_video = out_dir / "dual_panel_showcase.mp4"
    out_poster = out_dir / "dual_panel_poster.jpg"
    out_summary = out_dir / "summary.json"

    if out_video.exists() and out_poster.exists() and out_summary.exists():
        import json

        try:
            with open(out_summary, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
            summary["out_video"] = str(out_video)
            summary["out_poster"] = str(out_poster)
            return summary
        except (json.JSONDecodeError, KeyError):
            # Stale/corrupt cache from a previous failed run — delete and re-render.
            for stale in (out_summary, out_video, out_poster):
                if stale.exists():
                    stale.unlink(missing_ok=True)

    model, error = load_model_cached(scenario_label)
    if model is None:
        raise RuntimeError(error)

    tensor_input, sampled_frames, sampled_landmarks, stats = build_inference_tensor(video_path)
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=-1)
        pred_class = int(logits.argmax(dim=-1).item())
        confidence = float(probs[0, pred_class].item()) * 100.0

    attention_vector, attention_source = select_attention_vector(model, tensor_input)
    top_indices = [int(idx) for idx in np.argsort(attention_vector)[::-1][:5]]
    top_joints_text = ", ".join(f"#{idx}({attention_vector[idx]:.2f})" for idx in top_indices)

    summary = {
        "video_name": video_path.name,
        "movement": movement,
        "scenario": scenario_label,
        "prediction": pred_class,
        "prediction_label": "Form Benar" if pred_class == 0 else "Form Salah",
        "confidence": confidence,
        "attention_source": attention_source,
        "top_joints_text": top_joints_text,
        "top_joints": top_indices,
        "original_frames": stats["original_count"],
        "preview_frames": stats["sampled_count"],
        "mapping_mode": stats["mapping_mode"],
        "out_video": str(out_video),
        "out_poster": str(out_poster),
    }

    # Try H.264 (avc1) directly first; fall back to mp4v if codec unavailable.
    raw_video = out_dir / "dual_panel_raw.mp4"
    for _fcc_str in ("avc1", "mp4v"):
        _fourcc = cv2.VideoWriter_fourcc(*_fcc_str)
        writer = cv2.VideoWriter(str(raw_video), _fourcc, 18.0, (1280, 512))
        if writer.isOpened():
            break
    poster_frame = None

    total = len(sampled_frames)
    for frame_idx, (frame_bgr, landmarks) in enumerate(zip(sampled_frames, sampled_landmarks)):
        overlay_bgr = _draw_attention_overlay(frame_bgr, landmarks, attention_vector, movement=movement)
        composite = _composite_frame(
            frame_bgr,
            overlay_bgr,
            movement=movement,
            scenario=scenario_label,
            summary=summary,
            frame_idx=frame_idx,
            total=total,
        )
        if poster_frame is None:
            poster_frame = composite.copy()
        writer.write(composite)

    writer.release()

    # Re-encode to H.264 for browser / Streamlit compatibility.
    if _reencode_h264(raw_video, out_video):
        raw_video.unlink(missing_ok=True)
    else:
        # ffmpeg not available — fall back to raw mp4v (may not play in all browsers)
        raw_video.replace(out_video)

    if poster_frame is None:
        poster_frame = np.zeros((512, 1280, 3), dtype=np.uint8)
    cv2.imwrite(str(out_poster), poster_frame)

    import json

    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def _collect_videos_for_movement(movement: str) -> list[Path]:
    candidates: list[Path] = []
    for directory in MOVEMENT_SOURCE_DIRS[movement]:
        if not directory.exists():
            continue
        candidates.extend(sorted(directory.glob("*.mp4")))

    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _sidebar() -> tuple[str, str, Optional[Path], Optional[bytes], str, bool]:
    with st.sidebar:
        st.title("\U0001f52c Attention Showcase")
        st.caption("Mini Streamlit demo for stage presentation")
        st.divider()

        movement = st.selectbox("Movement", MOVEMENT_OPTIONS, index=0)
        scenario = st.selectbox("Model scenario", list(MODEL_SPECS.keys()), index=4)
        source_mode = st.radio("Input source", ["Demo sample", "Upload custom MP4"], index=0)

        sample_path: Optional[Path] = None
        uploaded_bytes: Optional[bytes] = None
        uploaded_name = ""

        if source_mode == "Demo sample":
            demos = _collect_videos_for_movement(movement)
            if demos:
                labels = [p.name for p in demos]
                selected_name = st.selectbox("Sample video", labels, index=0)
                sample_path = demos[labels.index(selected_name)]
            else:
                st.warning("No sample video found for this movement. Upload a custom MP4 instead.")
        else:
            uploaded = st.file_uploader("Upload MP4", type=["mp4"])
            if uploaded is not None:
                uploaded_bytes = uploaded.getvalue()
                uploaded_name = uploaded.name

        render = st.button("Render showcase", type="primary", use_container_width=True)

        st.divider()
        st.subheader("Presentation notes")
        st.markdown(
            "- Left: original clip\n"
            "- Right: model-driven attention overlay\n"
            "- No biomechanical validator in the overlay\n"
            "- Use Full Model for strongest demo effect"
        )

    return movement, scenario, sample_path, uploaded_bytes, uploaded_name, render


def _render_hero(movement: str, scenario: str) -> None:
    hint = MOVEMENT_HINTS.get(movement, "Expected focus zone")
    st.markdown(
        f"""
        <div class="app-hero">
          <h1>Attention Showcase</h1>
          <p>Dual-panel demo for stage presentation: original video on the left, model attention overlay on the right.</p>
          <div class="pill-row">
            <span class="pill">Movement: {movement}</span>
            <span class="pill">Scenario: {scenario}</span>
            <span class="pill">{hint}</span>
            <span class="pill">Synchronized 64-frame preview</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------


def main() -> None:
    movement, scenario, sample_path, uploaded_bytes, uploaded_name, render = _sidebar()
    _render_hero(movement, scenario)

    if not render:
        st.info("Choose movement and scenario, then click Render showcase.")
        st.stop()

    if uploaded_bytes is not None:
        suffix = Path(uploaded_name).suffix or ".mp4"
        temp_path = OUTPUT_DIR / f"upload_{_safe_slug(uploaded_name)}{suffix}"
        temp_path.write_bytes(uploaded_bytes)
        video_path = temp_path
    elif sample_path is not None:
        video_path = sample_path
    else:
        st.error("No video source available. Upload a custom MP4 or choose a sample video.")
        st.stop()

    with st.spinner("Rendering synchronized dual-panel demo..."):
        summary = _write_showcase_assets(video_path, movement, scenario)

    st.success("Showcase ready for demo stage.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Prediction", summary["prediction_label"])
    metric_cols[1].metric("Confidence", f"{summary['confidence']:.1f}%")
    metric_cols[2].metric("Attention source", summary["attention_source"])
    metric_cols[3].metric("Preview frames", summary["preview_frames"])

    st.markdown("<div class='video-shell'>", unsafe_allow_html=True)
    st.video(Path(summary["out_video"]).read_bytes())
    st.markdown("</div>", unsafe_allow_html=True)

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("<div class='section-title'>What the audience should read</div>", unsafe_allow_html=True)
        st.markdown(
            f"- Movement: {movement}\n"
            f"- Target zone: {MOVEMENT_HINTS.get(movement, '-') }\n"
            f"- Scenario: {scenario}\n"
            f"- Output is a synchronized split-screen preview, not a rule-based validator overlay."
        )
    with detail_cols[1]:
        st.markdown("<div class='section-title'>Joint ranking</div>", unsafe_allow_html=True)
        st.write(summary["top_joints_text"])
        st.caption("High-saliency joints should dominate the red glow if the model learned a meaningful focus pattern.")

    dl_cols = st.columns(2)
    with dl_cols[0]:
        st.download_button(
            "Download showcase MP4",
            data=Path(summary["out_video"]).read_bytes(),
            file_name=Path(summary["out_video"]).name,
            mime="video/mp4",
            use_container_width=True,
        )
    with dl_cols[1]:
        st.download_button(
            "Download poster JPG",
            data=Path(summary["out_poster"]).read_bytes(),
            file_name=Path(summary["out_poster"]).name,
            mime="image/jpeg",
            use_container_width=True,
        )

    with st.expander("Stage script"):
        st.markdown(
            """
            1. This is a synchronized split-screen preview.
            2. Left side shows the original clip.
            3. Right side shows the model-driven attention overlay.
            4. The overlay is derived from the selected scenario, not from biomechanical rules.
            5. For the strongest wow factor, use Full Model on Squat.
            """
        )


if __name__ == "__main__":
    main()
