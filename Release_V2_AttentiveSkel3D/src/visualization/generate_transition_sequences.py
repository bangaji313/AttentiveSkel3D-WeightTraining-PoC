# Release_V2_AttentiveSkel3D/src/visualization/generate_transition_sequences.py
#
# ==============================================================================
# PEMBANGKIT VISUALISASI TRANSISI SKELETON (BENAR <-> SALAH)
# ==============================================================================
#
# Menghasilkan 3 visualisasi runtun transisi sekuensial biomekanik (11 frame)
# dan file metadata CSV komprehensif untuk:
#   1. Squat (Squat_057)         -> Transisi BENAR -> SALAH (Hip Flexion / Depth)
#   2. Bench Press (BenchPress_034) -> Transisi BENAR -> SALAH (Elbow Flexion ROM)
#   3. Deadlift (Deadlift_002)    -> Transisi SALAH -> BENAR (Spine Inclination)
#
# Menggunakan raw MediaPipe image-space coordinates (x, y, visibility)
# langsung dari video asli, tanpa mengubah dataset / model.

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─── Direktori Proyek ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TENSORS_DIR  = PROJECT_ROOT / "data" / "tensors"
OUTPUT_DIR   = PROJECT_ROOT / "hasil_evaluasi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from src.data.biomechanics_validator import BiomechanicalValidator

validator = BiomechanicalValidator()
mp_pose = mp.solutions.pose

# ─── Koneksi Anatomis Skeleton (Original 33 MediaPipe IDs) ────────────────────
ANATOMICAL_CONNECTIONS = [
    # Kepala & Bahu
    (0, 11), (0, 12), (11, 12),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Lengan Kiri & Kanan
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Kaki Kiri & Kanan
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# ─── Konfigurasi Sampel Terbaik per Latihan ───────────────────────────────────
CONFIGS = [
    {
        "exercise": "Squat",
        "video_id": "Squat_057",
        "raw_rel_path": "Squat/primer_squat_frontal_subjek03_rep7.mp4",
        "temporal_range": list(range(45, 56)),  # 11 frames: t=45..55 (1-idx: 46..56)
        "transition_pair": (51, 52),            # Transisi terjadi antara t=51 dan t=52
        "metric_name": "Hip Flexion Angle",
        "threshold_str": "≤ 137.0°",
        "out_filename": "Sequence_Transition_Squat_Final.png"
    },
    {
        "exercise": "Bench Press",
        "video_id": "BenchPress_034",
        "raw_rel_path": "BenchPress/primer_benchpress_frontal_subjek01_rep4.mp4",
        "temporal_range": list(range(44, 55)),  # 11 frames: t=44..54 (1-idx: 45..55)
        "transition_pair": (50, 51),            # Transisi terjadi antara t=50 dan t=51
        "metric_name": "Elbow Flexion Angle",
        "threshold_str": "≤ 85.0°",
        "out_filename": "Sequence_Transition_BenchPress_Final.png"
    },
    {
        "exercise": "Deadlift",
        "video_id": "Deadlift_002",
        "raw_rel_path": "Deadlift/primer_deadlift_lateral_subjek01_rep10.mp4",
        "temporal_range": list(range(46, 57)),  # 11 frames: t=46..56 (1-idx: 47..57)
        "transition_pair": (52, 53),            # Transisi terjadi antara t=52 dan t=53
        "metric_name": "Spine Inclination Angle",
        "threshold_str": "20.0° ≤ θ ≤ 60.0°",
        "out_filename": "Sequence_Transition_Deadlift_Final.png"
    }
]



def extract_raw_frames_data(video_path: Path, temporal_indices: list[int]) -> list[dict]:
    """Ekstraksi frame mentah dan keypoints MediaPipe untuk indeks temporal tertentu."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Tidak dapat membuka file video: {video_path}")

    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Hitung mapping temporal 0..63 -> source raw frame
    # Formula standar: index = round(t * (total_raw_frames - 1) / 63)
    raw_frame_map = {}
    for t in temporal_indices:
        src_f = int(np.round(t * (total_raw_frames - 1) / 63.0))
        raw_frame_map[t] = src_f

    needed_src_frames = set(raw_frame_map.values())
    
    # Baca frame
    frames_rgb = {}
    curr_f = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if curr_f in needed_src_frames:
            frames_rgb[curr_f] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        curr_f += 1
        if len(frames_rgb) == len(needed_src_frames):
            break
    cap.release()

    # Ekstrak MediaPipe Pose (Heavy model complexity=2)
    results = []
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose_model:
        for t in temporal_indices:
            src_f = raw_frame_map[t]
            rgb = frames_rgb.get(src_f)
            ts_sec = src_f / fps
            ts_str = f"{int(ts_sec//60):02d}:{ts_sec%60:05.2f}"

            lm_array = np.zeros((33, 3), dtype=np.float32)
            has_pose = False
            if rgb is not None:
                res = pose_model.process(rgb)
                if res.pose_landmarks:
                    has_pose = True
                    for i in range(33):
                        lm = res.pose_landmarks.landmark[i]
                        lm_array[i, 0] = lm.x
                        lm_array[i, 1] = lm.y
                        lm_array[i, 2] = lm.visibility

            results.append({
                "temporal_index_0_63": t,
                "temporal_index_1_64": t + 1,
                "source_frame": src_f,
                "timestamp_str": ts_str,
                "timestamp_sec": round(ts_sec, 2),
                "landmarks": lm_array,
                "has_pose": has_pose
            })

    return results


def render_transition_strip(
    cfg: dict,
    frames_data: list[dict],
    tensor_full: np.ndarray,
    vis_threshold: float = 0.5
) -> tuple[Path, list[dict]]:
    """
    Merender figure strip 11 panel sekuensial transisi BENAR <-> SALAH
    dengan metrik biomekanis eksplisit per frame.
    """
    n_panels = len(frames_data)
    ex_name = cfg["exercise"]
    video_id = cfg["video_id"]
    metric_name = cfg["metric_name"]
    threshold_str = cfg["threshold_str"]
    out_path = OUTPUT_DIR / cfg["out_filename"]

    metadata_rows = []

    # Hitung metrik biomekanis aktual dari tensor normalisasi untuk tiap frame
    for item in frames_data:
        t = item["temporal_index_0_63"]
        f_ten = tensor_full[t : t + 1]

        if ex_name == "Squat":
            is_valid, reason = validator.validate_squat(f_ten)
            hip_L = validator._get_per_frame_angles(f_ten, 11, 23, 25)[0]
            hip_R = validator._get_per_frame_angles(f_ten, 12, 24, 26)[0]
            metric_val = float((hip_L + hip_R) / 2.0)
            metric_display = f"Hip: {metric_val:.1f}°"
        elif ex_name == "Bench Press":
            is_valid, reason = validator.validate_benchpress(f_ten)
            elbow_L = validator._get_per_frame_angles(f_ten, 11, 13, 15)[0]
            elbow_R = validator._get_per_frame_angles(f_ten, 12, 14, 16)[0]
            metric_val = float((elbow_L + elbow_R) / 2.0)
            metric_display = f"Elbow: {metric_val:.1f}°"
        elif ex_name == "Deadlift":
            is_valid, reason = validator.validate_deadlift(f_ten)
            mid_shoulder = (f_ten[:, 11, :] + f_ten[:, 12, :]) / 2.0
            spine_vec = mid_shoulder[0]
            norm_s = np.linalg.norm(spine_vec)
            if norm_s > 1e-8:
                dot_p = spine_vec @ np.array([0.0, -1.0, 0.0])
                cos_a = np.clip(dot_p / norm_s, -1.0, 1.0)
                metric_val = float(np.degrees(np.arccos(cos_a)))
            else:
                metric_val = 0.0
            metric_display = f"Spine: {metric_val:.1f}°"

        status_label = "BENAR" if is_valid else "SALAH"
        item["is_valid"] = is_valid
        item["status_label"] = status_label
        item["metric_val"] = round(metric_val, 2)
        item["metric_display"] = metric_display
        item["reason"] = reason

        metadata_rows.append({
            "exercise": ex_name,
            "video_id": video_id,
            "temporal_index": item["temporal_index_1_64"],
            "source_frame": item["source_frame"],
            "timestamp": item["timestamp_str"],
            "label_v2": status_label,
            "metric_name": metric_name,
            "metric_value": round(metric_val, 2),
            "threshold": threshold_str,
            "validity": "VALID" if item["has_pose"] else "INVALID_POSE"
        })

    # ── 1. Hitung Bounding Box Global untuk Skala Seragam ─────────────────────
    all_x = []
    all_y = []
    for item in frames_data:
        lm = item["landmarks"]
        vis_mask = lm[:, 2] >= vis_threshold
        if np.any(vis_mask):
            all_x.extend(lm[vis_mask, 0])
            all_y.extend(-lm[vis_mask, 1])

    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    max_range = max(max_x - min_x, max_y - min_y)
    pad = max_range * 0.12
    mid_x_global = (min_x + max_x) / 2.0
    mid_y_global = (min_y + max_y) / 2.0
    half_w = (max_range / 2.0) + pad

    # ── 2. Buat Figure Matplotlib ─────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(n_panels * 1.65, 4.8),
        facecolor="#ffffff"
    )
    if n_panels == 1:
        axes = [axes]

    fig.subplots_adjust(
        left=0.02, right=0.98,
        bottom=0.18, top=0.76,
        wspace=0.10
    )

    # ── 3. Render Setiap Panel Skeleton ──────────────────────────────────────
    for col, item in enumerate(frames_data):
        ax = axes[col]
        ax.set_facecolor("#ffffff")

        status = item["status_label"]
        lm = item["landmarks"]
        has_pose = item["has_pose"]

        # Warna sesuai status
        if status == "BENAR":
            bone_color = "#1e824c"   # Hijau Zamrud
            joint_color = "#27ae60"
            badge_bg = "#d4edda"
            badge_fg = "#155724"
            border_color = "#28a745"
        elif status == "SALAH":
            bone_color = "#c0392b"   # Merah Crimson
            joint_color = "#e74c3c"
            badge_bg = "#f8d7da"
            badge_fg = "#721c24"
            border_color = "#dc3545"
        else:
            bone_color = "#7f8c8d"   # Abu-abu
            joint_color = "#95a5a6"
            badge_bg = "#e2e3e5"
            badge_fg = "#383d41"
            border_color = "#6c757d"

        # Gambar Edge Tulang
        if has_pose:
            for (i, j) in ANATOMICAL_CONNECTIONS:
                if lm[i, 2] >= vis_threshold and lm[j, 2] >= vis_threshold:
                    ax.plot(
                        [lm[i, 0], lm[j, 0]],
                        [-lm[i, 1], -lm[j, 1]],
                        color=bone_color,
                        linewidth=2.4,
                        solid_capstyle="round",
                        zorder=2
                    )

            # Gambar Joint
            for ji in BODY_JOINTS:
                if lm[ji, 2] >= vis_threshold:
                    ax.scatter(
                        lm[ji, 0], -lm[ji, 1],
                        color=joint_color,
                        s=30,
                        edgecolors="#ffffff",
                        linewidths=0.6,
                        zorder=4
                    )

        # Set Coordinate Limits & Aspect Ratio
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(mid_x_global - half_w * 0.95, mid_x_global + half_w * 0.95)
        ax.set_ylim(mid_y_global - half_w, mid_y_global + half_w)
        ax.set_xticks([])
        ax.set_yticks([])

        # Border Frame
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.8 if col in [4, 5] else 1.0)

        # Header Badge: Status + Metric
        t_idx_display = item["temporal_index_1_64"]
        m_disp = item["metric_display"]
        ax.set_title(
            f"{status}\n{m_disp}",
            fontsize=8.5,
            fontweight="bold",
            color=badge_fg,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_bg, edgecolor=border_color, linewidth=1.0),
            pad=8
        )

        # Footer Info
        src_f = item["source_frame"]
        ts_s = item["timestamp_str"]
        ax.set_xlabel(
            f"t = {t_idx_display}/64\n#{src_f}\n({ts_s})",
            fontsize=8,
            color="#2d3436",
            fontweight="medium",
            labelpad=6
        )

    # ── 4. Header & Caption Fig 13 Academic Format ───────────────────────────
    title_text = (
        f"Visualization of skeletal key-point sequence extracted from raw video frames using MediaPipe BlazePose\n"
        f"Exercise: {ex_name}  |  Video ID: {video_id}  |  Metric Rule: {metric_name} ({threshold_str})"
    )
    fig.text(
        0.5, 0.93, title_text,
        ha="center", va="center",
        fontsize=10.5, color="#1e272c",
        fontweight="bold", fontfamily="sans-serif",
        linespacing=1.35
    )

    # Transition Marker Caption di Bawah
    t_a, t_b = cfg["transition_pair"]
    bottom_desc = (
        f"Transition occurs between t={t_a} and t={t_b} ── "
        f"Green indicates valid biomechanics ({threshold_str}); Red indicates biomechanical violation."
    )
    fig.text(
        0.5, 0.065, bottom_desc,
        ha="center", va="center",
        fontsize=9.0, color="#2d3436",
        fontweight="medium"
    )

    # Note Kecil Sumber Status Label
    note_text = (
        "Note: Frame status is determined by the Biomechanical Validator; "
        "this visualization only displays the extracted skeleton sequence and its frame-level status."
    )
    fig.text(
        0.5, 0.025, note_text,
        ha="center", va="center",
        fontsize=7.5, color="#636e72",
        fontstyle="italic"
    )

    # Simpan Gambar
    fig.savefig(
        str(out_path),
        dpi=220,
        bbox_inches="tight",
        facecolor="#ffffff",
        edgecolor="none"
    )
    plt.close(fig)
    print(f"[SUKSES] Visualisasi transisi tersimpan di: {out_path}")

    return out_path, metadata_rows


def main():
    print("=" * 80)
    print("  MEMPROSES VISUALISASI TRANSISI BIO-MEKANIK SKELETON (JALUR A - FINAL)")
    print("=" * 80)

    all_csv_records = []

    for cfg in CONFIGS:
        ex = cfg["exercise"]
        video_id = cfg["video_id"]
        rel_path = cfg["raw_rel_path"]
        video_path = RAW_DATA_DIR / rel_path
        tensor_path = TENSORS_DIR / f"{video_id}.npy"

        print(f"\n[PROSES] {ex.upper()} ({video_id})")
        print(f"  • Video mentah: {video_path}")
        print(f"  • Tensor: {tensor_path}")

        if not video_path.exists():
            print(f"  [ERROR] Video mentah tidak ditemukan: {video_path}")
            continue
        if not tensor_path.exists():
            print(f"  [ERROR] Tensor tidak ditemukan: {tensor_path}")
            continue

        tensor_full = np.load(str(tensor_path)).astype(np.float32)

        # Ekstrak data raw frames
        frames_data = extract_raw_frames_data(video_path, cfg["temporal_range"])
        
        # Render visualisasi strip & generate metadata
        out_png, meta_rows = render_transition_strip(cfg, frames_data, tensor_full)
        all_csv_records.extend(meta_rows)

    # Simpan metadata CSV gabungan final
    csv_out_path = OUTPUT_DIR / "Sequence_Transition_Metadata_Final.csv"
    df_meta = pd.DataFrame(all_csv_records)
    df_meta.to_csv(csv_out_path, index=False)
    print(f"\n[SUKSES] Metadata CSV tersimpan di: {csv_out_path}")

    print("\n" + "=" * 80)
    print("  RINGKASAN METADATA TRANSISI TERSIMPAN")
    print("=" * 80)
    print(df_meta[["exercise", "video_id", "temporal_index", "source_frame", "timestamp", "label_v2", "metric_name", "metric_value", "threshold"]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
