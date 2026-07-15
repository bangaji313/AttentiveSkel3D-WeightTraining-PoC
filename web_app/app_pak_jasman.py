"""Streamlit dashboard ringkas untuk bimbingan dosen pembimbing.

Dashboard ini memusatkan lima bukti utama tanpa mengubah dashboard lengkap:
data integrity, biomechanical validator, classification, attention sanity check,
dan frame inspector.
"""
from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
for _path in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# app_pak_jasman reuses helpers from web_app.app, but that module also calls
# set_page_config at import time. Temporarily suppress it so this file can own
# the page config call exactly once.
_orig_set_page_config = st.set_page_config
st.set_page_config = lambda *args, **kwargs: None
try:
    from web_app.app import (
        AUDIT_SHA256,
        CHECKPOINT_PATHS,
        CLASS_NAMES,
        EXERCISE_RULES,
        MAX_FRAMES,
        MODEL_CONFIGS,
        PRIMARY_METRIC,
        PRIMARY_THRESHOLD,
        BiomechanicalValidator,
        _checkpoint_integrity_df,
        build_inference_tensor,
        compute_sha256,
        draw_frame_skeleton,
        extract_attention_bundle,
        extract_learned_spatial_attention,
        extract_spatial_attention,
        extract_temporal_attention,
        format_agreement_status,
        load_model_cached,
        render_tab_with_debug,
        resolve_frame_sync_info,
        run_validator_per_frame,
        run_validator_sequence,
    )
finally:
    st.set_page_config = _orig_set_page_config
from web_app.attention_profiles import combine_attention, occlusion_values
from web_app.ui_compat import display_image_compat
from web_app.visualization_rules import exercise_relevant_landmarks, rule_marker_labels, rule_metric_for_frame


st.set_page_config(
    page_title="AttentiveSkel-3D | Pak Jasman",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


TAB_LABELS = [
    "1. Data Integrity",
    "2. Biomechanical Validator",
    "3. Classification",
    "4. Attention Sanity Check",
    "5. Frame Inspector",
]

FRAME_SLIDER_RANGE = (0, MAX_FRAMES - 1)

MOVEMENT_OPTIONS = ["Squat", "Deadlift", "Bench Press"]
SCENARIO_OPTIONS = {
    "Baseline": "Baseline 3D-CNN",
    "Ablasi A": "Ablasi A - No Prior",
    "Ablasi B": "Ablasi B - No Learned Spatial",
    "Ablasi C": "Ablasi C - No Temporal",
    "Full Model": "Full Model",
}

CLASS_MAPPING = {0: "Benar", 1: "Salah"}
CLASS_DISPLAY = {0: "Form Benar ✔", 1: "Form Salah ✗"}


def _scenario_key(label: str) -> str:
    return SCENARIO_OPTIONS[label]


def _scenario_config(label: str) -> dict:
    return MODEL_CONFIGS[_scenario_key(label)]


def _scenario_checkpoint(label: str) -> str:
    return CHECKPOINT_PATHS[_scenario_key(label)]


def _interp_to_64(scores: np.ndarray) -> np.ndarray:
    if len(scores) == MAX_FRAMES:
        return scores
    x_src = np.linspace(0, MAX_FRAMES - 1, len(scores))
    return np.interp(np.arange(MAX_FRAMES), x_src, scores)


def _frame_overlay(frame_bgr: np.ndarray, landmarks, colors, relevant_joints=None, labels=None, extra_text: Optional[list[str]] = None) -> np.ndarray:
    overlay = draw_frame_skeleton(
        frame_bgr,
        landmarks,
        color_per_joint=colors,
        relevant_joints=relevant_joints,
        attention_labels=labels,
    )
    if extra_text:
        y = 22
        for line in extra_text:
            cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
            y += 22
    return overlay


def _annotate_rule_labels(frame_bgr: np.ndarray, landmarks, rule: dict) -> np.ndarray:
    overlay = frame_bgr.copy()
    label_map = rule_marker_labels(rule)
    for idx, label in label_map.items():
        lm = landmarks[idx]
        px = int(lm.x * overlay.shape[1])
        py = int(lm.y * overlay.shape[0])
        cv2.putText(overlay, label, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def _timeline_figure(frame_indices: np.ndarray, metric: np.ndarray, threshold: float, statuses: np.ndarray, *, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    ax.plot(frame_indices, metric, color="#4fc3f7", linewidth=1.6, label="Metric")
    ax.axhline(threshold, color="#ef5350", linestyle="--", linewidth=1.2, label=f"Threshold = {threshold:.2f}")
    for idx, status in zip(frame_indices, statuses):
        if status == "INVALID":
            ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.18, color="#ef5350", lw=0)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _attention_colors(values: Optional[np.ndarray]) -> list[tuple[int, int, int]]:
    if values is None:
        return [(110, 110, 110)] * 33
    arr = np.asarray(values, dtype=np.float32)
    if arr.size != 33:
        return [(110, 110, 110)] * 33
    span = float(arr.max() - arr.min())
    normalized = np.full(33, 0.5, dtype=np.float32) if span <= 1e-8 else (arr - arr.min()) / span
    colors: list[tuple[int, int, int]] = []
    for value in normalized:
        red = int(255 * float(value))
        blue = int(220 * (1.0 - float(value)))
        green = int(120 * (1.0 - float(value)))
        colors.append((blue, green, red))
    return colors


def _process_video(video_file: st.runtime.uploaded_file_manager.UploadedFile, exercise_type: str, scenario_label: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        with st.spinner("Mengekstrak skeleton dan membangun tensor fitur..."):
            tensor, lm_frames, raw_frames, stats = build_inference_tensor(str(tmp_path), video_bytes=video_file.getvalue())
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    if tensor is None:
        st.session_state.update({
            "processed": False,
            "process_error": stats.get("error", "Unknown processing error"),
            "tensor": None,
            "tensor_np": None,
            "lm_frames": None,
            "raw_frames": None,
            "video_stats": None,
            "validator_df": None,
            "exercise_type": exercise_type,
            "scenario_label": scenario_label,
        })
        return

    tensor_np = tensor.squeeze(0).detach().cpu().numpy()
    validator_df = run_validator_per_frame(tensor_np, exercise_type, BiomechanicalValidator())

    st.session_state.update({
        "processed": True,
        "process_error": None,
        "tensor": tensor,
        "tensor_np": tensor_np,
        "lm_frames": lm_frames,
        "raw_frames": raw_frames,
        "video_stats": stats,
        "validator_df": validator_df,
        "video_bytes": video_file.getvalue(),
        "exercise_type": exercise_type,
        "scenario_label": scenario_label,
        "frame_inspector_idx": 0,
        "biomech_rule_idx": 0,
    })


def _sidebar() -> tuple[Optional[st.runtime.uploaded_file_manager.UploadedFile], str, str, bool]:
    with st.sidebar:
        st.title("🔬 AttentiveSkel-3D")
        st.caption("Pak Jasman Dashboard")
        st.divider()
        uploaded = st.file_uploader("Upload video MP4", type=["mp4"])
        exercise_type = st.selectbox("Jenis gerakan", MOVEMENT_OPTIONS, index=0)
        scenario_label = st.selectbox("Skenario model", list(SCENARIO_OPTIONS.keys()), index=4)
        process = st.button("Proses Video", type="primary", use_container_width=True, disabled=uploaded is None)

        st.divider()
        st.subheader("Ringkasan")
        if st.session_state.get("processed") and st.session_state.get("video_stats"):
            stats = st.session_state.video_stats
            st.metric("Jumlah frame asli", stats.get("original_count", stats.get("pose_frame_count", 0)))
            st.metric("Shape tensor final", str(stats.get("tensor_shape", (0, 0, 0))))
            st.metric("Checkpoint aktif", scenario_label)
            st.metric("Status pemrosesan", "Selesai")
        else:
            st.metric("Jumlah frame asli", "-")
            st.metric("Shape tensor final", "-")
            st.metric("Checkpoint aktif", scenario_label)
            st.metric("Status pemrosesan", "Belum diproses")

    return uploaded, exercise_type, scenario_label, process


def _intro_box() -> None:
    st.info(
        "Dashboard ini disusun untuk menjawab lima kebutuhan pembuktian:\n"
        "1. Apakah data skeleton valid?\n"
        "2. Bagaimana rule biomekanis dihitung?\n"
        "3. Apa hasil klasifikasi sequence-level?\n"
        "4. Pada sendi dan frame mana model fokus?\n"
        "5. Apa yang terjadi pada frame tertentu?"
    )


def _tab1_data_integrity() -> None:
    st.subheader("1. Data Integrity")
    s = st.session_state
    stats = s.video_stats
    tensor_np = s.tensor_np

    if stats is None or tensor_np is None:
        st.error("Data belum diproses.")
        return

    frame_idx = st.slider("Frame preview", 0, MAX_FRAMES - 1, 0, 1, key="pak_data_frame")
    sync_info = resolve_frame_sync_info(
        frame_index=frame_idx,
        mapping={"original_count": stats["pose_frame_count"], "max_frames": MAX_FRAMES, "resampled_to_source": stats["resampled_to_source"]},
        source_timestamps_ms=stats["source_timestamps_ms"],
    )

    vis = stats["vis_matrix"]
    vis_row = vis[frame_idx]
    low_conf = int((vis_row < 0.5).sum())
    missing = int(stats.get("unrecoverable_landmark_count", 0))
    interpolated = int(stats.get("interpolated_landmark_count", 0))
    nan_count = int(np.isnan(tensor_np).sum())
    inf_count = int(np.isinf(tensor_np).sum())
    status = "VALID" if stats.get("pose_frame_count", 0) > 0 and nan_count == 0 and inf_count == 0 else "DIELIMINASI"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frame asli", stats["original_count"])
    c2.metric("Frame final", tensor_np.shape[0])
    c3.metric("Low-confidence landmarks", low_conf)
    c4.metric("Status akhir sampel", status)

    st.markdown(
        "Tensor selalu memiliki 33 slot landmark; landmark tertutup dapat memiliki confidence rendah; interpolasi hanya dilakukan sesuai aturan pipeline."
    )

    col_v, col_2d = st.columns([1, 1.2])
    with col_v:
        st.video(s.video_bytes)
        display_image_compat(cv2.cvtColor(s.raw_frames[frame_idx], cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx}", stretch=True, channels="RGB")
    with col_2d:
        colors = [(140, 140, 140) if float(vis_row[i]) < 0.35 else (0, 180, 0) for i in range(33)]
        for idx in np.where(stats["resampled_interpolated_mask"][frame_idx])[0]:
            colors[int(idx)] = (0, 165, 255)
        for idx in np.where(stats["resampled_unrecoverable_mask"][frame_idx])[0]:
            colors[int(idx)] = (0, 0, 255)
        overlay = draw_frame_skeleton(s.raw_frames[frame_idx], s.lm_frames[frame_idx], color_per_joint=colors, relevant_joints=None)
        display_image_compat(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"Overlay skeleton 2D — source frame {sync_info.source_frame_index}", stretch=True, channels="RGB")

    st.metric("Shape tensor final", str(tensor_np.shape))
    tensor_df = pd.DataFrame(
        [
            {"frame_index": i, "landmark": j, "x": float(tensor_np[i, j, 0]), "y": float(tensor_np[i, j, 1]), "z": float(tensor_np[i, j, 2])}
            for i in range(tensor_np.shape[0]) for j in range(33)
        ]
    )
    st.dataframe(tensor_df, use_container_width=True, height=280)

    st.write(
        {
            "jumlah_frame_asli": stats["original_count"],
            "jumlah_frame_final": tensor_np.shape[0],
            "low_confidence_landmarks": low_conf,
            "missing_landmarks": missing,
            "interpolated_landmarks": interpolated,
            "NaN": nan_count,
            "Inf": inf_count,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(vis.T, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Landmark Index")
    ax.set_title("Visibility Heatmap")
    plt.colorbar(im, ax=ax, fraction=0.02, label="Visibility")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    missing_ratio = 1.0 - vis.mean(axis=0)
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.bar(range(33), missing_ratio, color="#ef5350", edgecolor="none")
    ax2.set_title("Missing ratio per landmark")
    ax2.set_xlabel("Landmark")
    ax2.set_ylabel("Missing ratio")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


def _tab2_biomechanical_validator() -> None:
    st.subheader("2. Biomechanical Validator")
    s = st.session_state
    stats = s.video_stats
    tensor_np = s.tensor_np
    exercise_type = s.exercise_type
    if stats is None or tensor_np is None:
        st.error("Data belum diproses.")
        return

    st.warning("Biomechanical Validator merupakan evaluasi rule-based per frame dan digunakan untuk membentuk atau memeriksa ground truth, bukan prediksi AI.")
    rules = EXERCISE_RULES[exercise_type]
    rule_names = [rule["name"] for rule in rules]
    rule_idx = st.radio("Nama rule aktif", list(range(len(rules))), format_func=lambda i: rule_names[i], horizontal=True, key="pak_biomech_rule")
    rule = rules[rule_idx]

    rule_rows = [rule_metric_for_frame(exercise_type, rule, tensor_np[i]) for i in range(tensor_np.shape[0])]
    active_df = pd.DataFrame(
        {
            "frame_index": list(range(tensor_np.shape[0])),
            "timestamp": stats["resampled_timestamps_ms"],
            "metric_value": [row["metric_value"] for row in rule_rows],
            "threshold": [row["threshold"] for row in rule_rows],
            "validator_status": [row["status"] for row in rule_rows],
        }
    )

    metric_def = "Rasio" if rule["name"] == "Knee Valgus" else "Sudut"
    unit = "ratio" if rule["name"] == "Knee Valgus" else "deg"
    st.markdown(f"**Metric aktif:** {rule['name']}")
    st.markdown(f"**Definisi metric:** {rule['description']}")
    st.markdown(f"**Satuan metric:** {unit}")
    st.markdown(f"**Rumus:** {rule['label']}")
    st.markdown(f"**Threshold:** {rule['threshold_str']}")
    st.markdown(f"**Kondisi INVALID:** {rule['threshold_str']}")

    landmark_names = {11: "Left Shoulder", 12: "Right Shoulder", 13: "Left Elbow", 14: "Right Elbow", 15: "Left Wrist", 16: "Right Wrist", 23: "Left Hip", 24: "Right Hip", 25: "Left Knee", 26: "Right Knee", 27: "Left Ankle", 28: "Right Ankle"}
    a, b, c = rule["lm_a"], rule["lm_b"], rule["lm_c"]
    lm_table = pd.DataFrame([
        {"Point": "A", "name": landmark_names.get(a, "-"), "index": a, "vertex": "-"},
        {"Point": "B", "name": landmark_names.get(b, "-"), "index": b, "vertex": "vertex"},
        {"Point": "C", "name": landmark_names.get(c, "-"), "index": c, "vertex": "-"},
    ])
    st.dataframe(lm_table, use_container_width=True, hide_index=True)
    st.code(rule["description"], language="text")

    invalid = active_df[active_df["validator_status"] == "INVALID"]
    first_invalid = int(invalid["frame_index"].min()) if not invalid.empty else None
    worst_idx = int(active_df["metric_value"].idxmax()) if len(active_df) else None
    worst_frame = int(active_df.loc[worst_idx, "frame_index"]) if worst_idx is not None else None
    last_invalid = int(invalid["frame_index"].max()) if not invalid.empty else None
    invalid_count = int(len(invalid))
    invalid_ratio = float(invalid_count / MAX_FRAMES * 100.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("first invalid frame", first_invalid if first_invalid is not None else "N/A")
    c2.metric("peak/worst frame", worst_frame if worst_frame is not None else "N/A")
    c3.metric("last invalid frame", last_invalid if last_invalid is not None else "N/A")
    c4.metric("invalid frame ratio", f"{invalid_ratio:.1f}%")

    fig = _timeline_figure(active_df["frame_index"].to_numpy(), active_df["metric_value"].to_numpy(), float(active_df["threshold"].iloc[0]), active_df["validator_status"].to_numpy(), title=f"{exercise_type} — {rule['name']}")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.dataframe(active_df, use_container_width=True, height=260)

    frame_idx = st.slider("Frame preview", *FRAME_SLIDER_RANGE, key="pak_biomech_frame")
    frame_xyz = tensor_np[frame_idx]
    vis_row = stats["vis_matrix"][frame_idx]
    row = active_df.iloc[frame_idx]
    colors = [(0, 180, 0) if i in exercise_relevant_landmarks([rule]) else (70, 70, 70) for i in range(33)]
    overlay = draw_frame_skeleton(
        s.raw_frames[frame_idx],
        s.lm_frames[frame_idx],
        color_per_joint=colors,
        relevant_joints=exercise_relevant_landmarks([rule]),
        attention_labels=None,
    )
    overlay = _annotate_rule_labels(overlay, s.lm_frames[frame_idx], rule)
    cv2.putText(overlay, f"{rule['name']} | metric={row['metric_value']:.2f} | status={row['validator_status']}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    display_image_compat(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx}", stretch=True, channels="RGB")
    st.caption(f"Landmark relevan: {exercise_relevant_landmarks([rule])}")


def _tab3_classification() -> None:
    st.subheader("3. Classification")
    s = st.session_state
    if s.tensor is None:
        st.error("Data belum diproses.")
        return

    scenario_key = _scenario_key(s.scenario_label)
    cfg = MODEL_CONFIGS[scenario_key]
    abs_path = str(Path(__file__).resolve().parents[1] / _scenario_checkpoint(s.scenario_label))
    model, missing, unexpected, err = load_model_cached(abs_path, cfg["use_spatial_prior"], cfg["use_learned_spatial"], cfg["use_temporal_attention"])
    if model is None:
        st.error(f"Checkpoint gagal dimuat: {err}")
        return

    _, logits_np = extract_temporal_attention(model, s.tensor)
    exp_l = np.exp(logits_np - logits_np.max())
    probs = exp_l / exp_l.sum()
    pred = int(np.argmax(logits_np))
    conf = float(probs[pred]) * 100.0
    seq_valid, seq_reason = run_validator_sequence(s.tensor_np, s.exercise_type, BiomechanicalValidator())
    agreement_detail = format_agreement_status(pred, seq_valid)
    agreement = agreement_detail if agreement_detail.startswith("AGREE") else "DISAGREEMENT"

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**Jenis gerakan:** {s.exercise_type}")
        st.markdown(f"**Prediksi:** {CLASS_DISPLAY[pred]}")
        st.markdown(f"**Confidence:** {conf:.1f}%")
        st.markdown(f"**Checkpoint aktif:** {scenario_key}")
        st.markdown(f"**Class mapping:** 0 = Benar, 1 = Salah")
        st.markdown(f"**Validator sequence label:** {'VALID' if seq_valid else 'INVALID'}")
        st.markdown(f"**Agreement status:** {agreement}")
        if agreement == "DISAGREEMENT":
            st.warning("Classifier dan validator tidak selaras. Ini normal pada beberapa sekuens karena classifier adalah sequence-level, bukan per-frame.")
    with col2:
        cls_df = pd.DataFrame({"Class": [CLASS_DISPLAY[0], CLASS_DISPLAY[1]], "Logit": [float(logits_np[0]), float(logits_np[1])], "Softmax": [float(probs[0]), float(probs[1])]})
        st.dataframe(cls_df, use_container_width=True, hide_index=True)
        st.write({"logits": [float(x) for x in logits_np], "softmax": [float(x) for x in probs]})
        st.markdown(f"**Missing keys:** {len(missing)} · **Unexpected keys:** {len(unexpected)}")

    st.warning("Classifier menghasilkan satu prediksi untuk keseluruhan sekuens 64 frame. Prediksi ini bukan klasifikasi per frame.")


def _tab4_attention() -> None:
    st.subheader("4. Attention Sanity Check")
    s = st.session_state
    if s.tensor is None:
        st.error("Data belum diproses.")
        return

    scenario_key = _scenario_key(s.scenario_label)
    cfg = MODEL_CONFIGS[scenario_key]
    model, _, _, err = load_model_cached(str(Path(__file__).resolve().parents[1] / _scenario_checkpoint(s.scenario_label)), cfg["use_spatial_prior"], cfg["use_learned_spatial"], cfg["use_temporal_attention"])
    if model is None:
        st.error(f"Checkpoint gagal dimuat: {err}")
        return

    st.markdown(f"**Skenario model aktif:** {s.scenario_label}")
    st.write({
        "spatial_prior": cfg["use_spatial_prior"],
        "learned_spatial": cfg["use_learned_spatial"],
        "temporal_attention": cfg["use_temporal_attention"],
    })

    spatial = extract_spatial_attention(model)
    learned = extract_learned_spatial_attention(model, s.tensor)
    fused = combine_attention(spatial, learned)
    vis_row = s.video_stats["vis_matrix"][st.session_state.get("frame_inspector_idx", 0)] if s.video_stats else None
    occlusion = occlusion_values(vis_row)
    t_attn, logits_np = extract_temporal_attention(model, s.tensor)
    t64 = _interp_to_64(t_attn) if t_attn is not None else None

    if spatial is None:
        st.info("Attention tidak tersedia pada skenario Baseline")
        return

    st.subheader("Spatial attention per 33 landmark")
    top5 = np.argsort(spatial)[::-1][:5]
    top_rows = []
    for idx in top5:
        top_rows.append({"index": int(idx), "name": f"LM {idx}", "raw": float(spatial[idx]), "normalized": float(spatial[idx])})
    st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

    relevant = exercise_relevant_landmarks(EXERCISE_RULES[s.exercise_type])
    overlap = len(set(map(int, top5)).intersection(relevant))
    if overlap >= 3:
        sanity = "fokus relevan"
    elif overlap >= 1:
        sanity = "fokus sebagian relevan"
    else:
        sanity = "fokus tidak konsisten"
    st.metric("Status sanity check", sanity)
    st.caption(f"Relevant joints biomekanis: {relevant}")

    colors = _attention_colors(spatial)
    frame_idx = st.slider("Frame preview attention", *FRAME_SLIDER_RANGE, key="pak_attention_frame")
    overlay = draw_frame_skeleton(
        s.raw_frames[frame_idx],
        s.lm_frames[frame_idx],
        color_per_joint=colors,
        relevant_joints=relevant,
        attention_labels=spatial,
    )
    display_image_compat(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Spatial attention overlay", stretch=True, channels="RGB")

    fig, ax = plt.subplots(figsize=(12.5, 3.5))
    ax.bar(range(33), spatial, color=["#ef5350" if val > 0.5 else "#4fc3f7" for val in spatial], edgecolor="none")
    ax.set_title("Spatial attention per landmark")
    ax.set_xlabel("Landmark")
    ax.set_ylabel("Weight")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if fused is not None:
        st.subheader("Fused attention")
        st.dataframe(pd.DataFrame({"index": list(range(33)), "fused": fused}), use_container_width=True, hide_index=True)
    else:
        st.info("Fused attention tidak tersedia untuk kombinasi modul saat ini.")

    if t64 is not None:
        fig2, ax2 = plt.subplots(figsize=(12.5, 3))
        ax2.plot(range(MAX_FRAMES), t64, color="#4fc3f7", linewidth=1.5)
        ax2.axvline(int(np.argmax(t64)), color="#ff5722", linestyle="--", linewidth=2)
        ax2.set_title(f"Temporal attention — peak frame {int(np.argmax(t64))}")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Weight")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
    else:
        st.info("Temporal attention tidak tersedia pada skenario ini.")


def _tab5_frame_inspector() -> None:
    st.subheader("5. Frame Inspector")
    s = st.session_state
    if s.tensor is None:
        st.error("Data belum diproses.")
        return

    scenario_key = _scenario_key(s.scenario_label)
    cfg = MODEL_CONFIGS[scenario_key]
    model, _, _, err = load_model_cached(str(Path(__file__).resolve().parents[1] / _scenario_checkpoint(s.scenario_label)), cfg["use_spatial_prior"], cfg["use_learned_spatial"], cfg["use_temporal_attention"])
    if model is None:
        st.error(f"Checkpoint gagal dimuat: {err}")
        return

    stats = s.video_stats
    validator_df = s.validator_df
    frame_idx = st.slider("Frame 0–63", *FRAME_SLIDER_RANGE, key="frame_inspector_idx")
    if "first invalid" in st.session_state:
        frame_idx = st.session_state.frame_inspector_idx

    rules = EXERCISE_RULES[s.exercise_type]
    rule = rules[st.session_state.get("biomech_rule_idx", 0) % len(rules)]
    rule_row = rule_metric_for_frame(s.exercise_type, rule, s.tensor_np[frame_idx])
    sync_info = resolve_frame_sync_info(
        frame_index=frame_idx,
        mapping={"original_count": stats["pose_frame_count"], "max_frames": MAX_FRAMES, "resampled_to_source": stats["resampled_to_source"]},
        source_timestamps_ms=stats["source_timestamps_ms"],
    )
    spatial = extract_spatial_attention(model)
    t_attn, _ = extract_temporal_attention(model, s.tensor)
    t64 = _interp_to_64(t_attn) if t_attn is not None else None
    attention_bundle = extract_attention_bundle(model, s.tensor, stats["vis_matrix"][frame_idx])
    relevant = exercise_relevant_landmarks([rule])
    interpolated_indices = tuple(np.where(stats["resampled_interpolated_mask"][frame_idx])[0].tolist())
    unrecoverable_indices = tuple(np.where(stats["resampled_unrecoverable_mask"][frame_idx])[0].tolist())

    btn_cols = st.columns(4)
    if btn_cols[0].button("first invalid"):
        invalid = validator_df[validator_df["validator_status"] == "INVALID"]
        if not invalid.empty:
            st.session_state.frame_inspector_idx = int(invalid["frame_index"].min())
            st.rerun()
    if btn_cols[1].button("worst frame"):
        idx = int(validator_df["metric_value"].idxmax())
        st.session_state.frame_inspector_idx = int(validator_df.loc[idx, "frame_index"])
        st.rerun()
    if btn_cols[2].button("peak temporal attention") and t64 is not None:
        st.session_state.frame_inspector_idx = int(np.argmax(t64))
        st.rerun()
    if btn_cols[3].button("last invalid"):
        invalid = validator_df[validator_df["validator_status"] == "INVALID"]
        if not invalid.empty:
            st.session_state.frame_inspector_idx = int(invalid["frame_index"].max())
            st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original frame index", sync_info.source_frame_index)
    c2.metric("Timestamp", f"{sync_info.timestamp_ms:.1f} ms" if np.isfinite(sync_info.timestamp_ms) else "n/a")
    c3.metric("Spatial attention", "N/A" if spatial is None else f"{float(spatial[relevant[0]] if relevant else spatial[0]):.4f}")
    c4.metric("Temporal attention", "N/A" if t64 is None else f"{float(t64[frame_idx]):.4f}")

    col_img, col_info = st.columns([1.2, 1])
    with col_img:
        st.image(cv2.cvtColor(s.raw_frames[frame_idx], cv2.COLOR_BGR2RGB), caption=f"Original frame {sync_info.source_frame_index}", use_container_width=True)
        colors = [(0, 180, 0) if i in relevant else (80, 80, 80) for i in range(33)]
        if spatial is not None:
            colors = _attention_colors(spatial)
        overlay = draw_frame_skeleton(
            s.raw_frames[frame_idx],
            s.lm_frames[frame_idx],
            color_per_joint=colors,
            relevant_joints=relevant,
            attention_labels=spatial,
        )
        overlay = _annotate_rule_labels(overlay, s.lm_frames[frame_idx], rule)
        cv2.putText(overlay, f"{rule['name']} | metric={rule_row['metric_value']:.2f} | {rule_row['status']}", (10, overlay.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        display_image_compat(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Skeleton overlay 2D", stretch=True, channels="RGB")
    with col_info:
        st.markdown(f"**Rule:** {rule['name']}")
        st.markdown(f"**Metric value:** {rule_row['metric_value'] if rule_row['metric_value'] is not None else 'N/A'}")
        st.markdown(f"**Threshold:** {rule_row['threshold']}")
        st.markdown(f"**Status:** {rule_row['status']}")
        st.markdown(f"**Relevant landmarks:** {relevant}")
        st.markdown(f"**Interpolated relevant:** {sorted(set(relevant).intersection(set(np.where(stats['resampled_interpolated_mask'][frame_idx])[0].tolist())))}")
        st.markdown(f"**Unrecoverable relevant:** {sorted(set(relevant).intersection(set(np.where(stats['resampled_unrecoverable_mask'][frame_idx])[0].tolist())))}")
        if t64 is not None:
            st.markdown(f"**Peak temporal attention frame:** {int(np.argmax(t64))}")
        top_joints = []
        if spatial is not None:
            top_idx = np.argsort(spatial)[::-1][:5]
            for idx in top_idx:
                top_joints.append({"index": int(idx), "name": f"LM {int(idx)}", "raw": float(spatial[idx]), "normalized": float(spatial[idx])})
        st.dataframe(pd.DataFrame(top_joints), use_container_width=True, hide_index=True)

    st.caption(f"Sinkronisasi frame: source frame {sync_info.source_frame_index} | tensor frame {frame_idx}")


def main() -> None:
    if "processed" not in st.session_state:
        st.session_state.processed = False
        st.session_state.process_error = None
        st.session_state.tensor = None
        st.session_state.tensor_np = None
        st.session_state.lm_frames = None
        st.session_state.raw_frames = None
        st.session_state.video_stats = None
        st.session_state.validator_df = None
        st.session_state.exercise_type = MOVEMENT_OPTIONS[0]
        st.session_state.scenario_label = "Full Model"
        st.session_state.frame_inspector_idx = 0
        st.session_state.biomech_rule_idx = 0

    uploaded, exercise_type, scenario_label, process = _sidebar()
    _intro_box()

    if process and uploaded is not None:
        _process_video(uploaded, exercise_type, scenario_label)

    if not st.session_state.processed:
        if st.session_state.process_error:
            st.error(st.session_state.process_error)
        st.dataframe(_checkpoint_integrity_df(), use_container_width=True)
        return

    st.success("Video berhasil diproses.")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_LABELS)
    render_tab_with_debug(TAB_LABELS[0], _tab1_data_integrity)
    render_tab_with_debug(TAB_LABELS[1], _tab2_biomechanical_validator)
    render_tab_with_debug(TAB_LABELS[2], _tab3_classification)
    render_tab_with_debug(TAB_LABELS[3], _tab4_attention)
    render_tab_with_debug(TAB_LABELS[4], _tab5_frame_inspector)


if __name__ == "__main__":
    main()
