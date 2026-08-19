# Release_V2_AttentiveSkel3D/src/visualization/visualize_skeleton_sequence_fig13.py
#
# ==============================================================================
# VISUALISASI SKELETON-ONLY SEQUENCE HORIZONTAL (SEPERTI FIG. 13 XU ET AL., 2024)
# ==============================================================================
#
# Memvisualisasikan runtun skeleton bersih (skeleton-only) 10-12 frame horizontal
# dari video mentah menggunakan keypoint MediaPipe BlazePose (x, y, visibility)
# di ruang piksel asli (image-space), BUKAN tensor normalisasi .npy.
#
# Ketentuan Ketat:
#   1. Mempertahankan original 33 MediaPipe landmark IDs (0 s/d 32).
#   2. Tidak menghapus / menggeser elemen array.
#   3. Tidak menggunakan tensor final .npy sebagai sumber koordinat visual.
#   4. Edge hanya digambar jika KEDUA endpoint memiliki visibility >= threshold (>= 0.5).
#   5. Joint dengan visibility rendah (< threshold) tidak digambar.
#   6. Centering dan scaling dilakukan secara global pada level sequence untuk
#      mempertahankan aspek rasio asli (1:1) dan gerak alami tubuh.
#   7. Mengambil 10-12 frame representatif sekuensial.
#   8. Output bergaya publikasi ilmiah (clean paper aesthetic / white background).
#   9. Tanpa mengubah preprocessing / dataset / training pipeline.
#
# Penggunaan:
#   conda run -n attentiveskel python Release_V2_AttentiveSkel3D/src/visualization/visualize_skeleton_sequence_fig13.py --all_samples
#   conda run -n attentiveskel python Release_V2_AttentiveSkel3D/src/visualization/visualize_skeleton_sequence_fig13.py --stem Squat_001 --n_frames 12

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ─── Direktori Proyek ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_DIR   = PROJECT_ROOT / "data" / "v2_labels"
MANIFEST_PATH = PROJECT_ROOT / "src" / "data" / "manifest_v2.csv"
OUTPUT_DIR   = PROJECT_ROOT / "hasil_evaluasi" / "fig13_skeleton_sequence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Inisialisasi MediaPipe ───────────────────────────────────────────────────
mp_pose = mp.solutions.pose

# ─── Koneksi Anatomis Skeleton Standar (Menggunakan Original 33 MediaPipe IDs) ─
# Hanya menghubungkan sendi tubuh utama yang bermakna secara biomekanik
ANATOMICAL_CONNECTIONS = [
    # Kepala & Leher
    (0, 11), (0, 12),           # Nose -> Left/Right Shoulder
    (11, 12),                   # Left Shoulder <-> Right Shoulder
    # Torso
    (11, 23), (12, 24),         # Shoulder -> Hip (kiri & kanan)
    (23, 24),                   # Left Hip <-> Right Hip
    # Lengan Kiri
    (11, 13), (13, 15),         # L.Shoulder -> L.Elbow -> L.Wrist
    (15, 17), (15, 19), (15, 21), # L.Wrist -> L.Pinky / L.Index / L.Thumb
    # Lengan Kanan
    (12, 14), (14, 16),         # R.Shoulder -> R.Elbow -> R.Wrist
    (16, 18), (16, 20), (16, 22), # R.Wrist -> R.Pinky / R.Index / R.Thumb
    # Kaki Kiri
    (23, 25), (25, 27),         # L.Hip -> L.Knee -> L.Ankle
    (27, 29), (27, 31), (29, 31), # L.Ankle -> L.Heel -> L.Foot_Index
    # Kaki Kanan
    (24, 26), (26, 28),         # R.Hip -> R.Knee -> R.Ankle
    (28, 30), (28, 32), (30, 32), # R.Ankle -> R.Heel -> R.Foot_Index
]

# Sendi tubuh penting yang ditampilkan titiknya jika terdeteksi
BODY_JOINTS = [
    0,                      # Nose / Head
    11, 12,                 # Shoulders
    13, 14,                 # Elbows
    15, 16,                 # Wrists
    17, 18, 19, 20, 21, 22, # Hands (Pinky, Index, Thumb)
    23, 24,                 # Hips
    25, 26,                 # Knees
    27, 28,                 # Ankles
    29, 30, 31, 32          # Feet (Heels, Foot Indices)
]


def cari_path_video_mentah(stem: str) -> Path | None:
    """Mencari file video mentah .mp4 berdasarkan stem."""
    bagian = stem.split("_", 1)
    if len(bagian) < 2:
        return None
    nama_latihan = bagian[0]
    try:
        nomor_urut = int(bagian[1])
    except ValueError:
        return None

    folder_latihan = None
    for subfolder in RAW_DATA_DIR.iterdir():
        if subfolder.is_dir() and subfolder.name.lower() == nama_latihan.lower():
            folder_latihan = subfolder
            break
    if folder_latihan is None:
        return None

    daftar_video = sorted(folder_latihan.glob("*.mp4"))
    idx_video = nomor_urut - 1
    if 0 <= idx_video < len(daftar_video):
        return daftar_video[idx_video]
    return None


def ekstrak_raw_mediapipe_sequence(
    video_path: Path,
    n_frames: int = 12
) -> list[dict]:
    """
    Membaca video asli dan mengekstraksi raw image-space landmarks (x, y, visibility)
    untuk n_frames yang terdistribusi merata sepanjang video.
    
    Returns:
        list of dict: [
            {
                'frame_idx': int,
                'source_frame_num': int,
                'progress_pct': float,
                'landmarks_33x3': np.ndarray (33, 3) -> [x_norm, y_norm, visibility],
                'has_pose': bool
            }, ...
        ]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Gagal membuka video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Pilih n_frames yang terdistribusi merata dari awal hingga akhir
    source_frame_indices = np.round(np.linspace(0, total_frames - 1, n_frames)).astype(int).tolist()
    needed_set = set(source_frame_indices)

    # Baca frame secara sekuensial
    frames_rgb = {}
    current_f = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if current_f in needed_set:
            frames_rgb[current_f] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        current_f += 1
        if len(frames_rgb) == len(needed_set):
            break
    cap.release()

    # Ekstraksi MediaPipe Pose (Heavy Model untuk akurasi tertinggi)
    extracted_sequence = []
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose_model:
        for idx, src_f in enumerate(source_frame_indices):
            raw_rgb = frames_rgb.get(src_f)
            if raw_rgb is None:
                continue
            
            res = pose_model.process(raw_rgb)
            lm_array = np.zeros((33, 3), dtype=np.float32)  # [x, y, vis]
            has_pose = False
            
            if res.pose_landmarks:
                has_pose = True
                for i in range(33):
                    lm = res.pose_landmarks.landmark[i]
                    # x, y dalam rentang [0.0, 1.0] image-space
                    lm_array[i, 0] = lm.x
                    lm_array[i, 1] = lm.y
                    lm_array[i, 2] = lm.visibility

            extracted_sequence.append({
                "frame_idx": idx,
                "source_frame_num": src_f,
                "progress_pct": (src_f / max(1, total_frames - 1)) * 100.0,
                "landmarks_33x3": lm_array,
                "has_pose": has_pose,
                "video_w": video_w,
                "video_h": video_h,
            })

    return extracted_sequence


def render_fig13_skeleton_strip(
    extracted_sequence: list[dict],
    exercise_name: str,
    stem: str,
    vis_threshold: float = 0.5,
    theme: str = "paper",
    output_path: Path = None
) -> Path:
    """
    Menggambar deretan horizontal skeleton-only (persis seperti Fig. 13 Xu et al.).
    
    Fitur Geometris & Visual:
      - Aspek rasio 1:1 dipertahankan konsisten untuk seluruh frame.
      - Skala & bounding box dihitung secara global per-video sehingga gerak naik-turun
        (squat depth, elbow flexion, deadlift hinge) terlihat natural sebagai urutan aksi.
      - Edge HANYA digambar jika vis(start) >= vis_threshold DAN vis(end) >= vis_threshold.
      - Joint HANYA digambar jika vis(joint) >= vis_threshold.
    """
    n_frames = len(extracted_sequence)
    
    # ── 1. Hitung Global Bounding Box & Scale untuk Seluruh Sequence ───────────
    # Kumpulkan seluruh titik valid yang memiliki visibility >= vis_threshold
    all_valid_x = []
    all_valid_y = []

    for item in extracted_sequence:
        if not item["has_pose"]:
            continue
        lm = item["landmarks_33x3"]
        vis_mask = lm[:, 2] >= vis_threshold
        if np.any(vis_mask):
            all_valid_x.extend(lm[vis_mask, 0])
            # Catatan: MediaPipe Y bertambah ke bawah; untuk plot Cartesian kita balik (-y)
            all_valid_y.extend(-lm[vis_mask, 1])

    if len(all_valid_x) == 0:
        # Fallback jika tidak ada deteksi
        min_x, max_x = 0.2, 0.8
        min_y, max_y = -0.9, -0.1
    else:
        min_x, max_x = np.min(all_valid_x), np.max(all_valid_x)
        min_y, max_y = np.min(all_valid_y), np.max(all_valid_y)

    # Tambahkan margin padding 10%
    range_x = max(1e-4, max_x - min_x)
    range_y = max(1e-4, max_y - min_y)
    
    # Agar aspek rasio 1:1 seragam di setiap panel, gunakan rentang maksimum
    max_range = max(range_x, range_y)
    pad = max_range * 0.12

    # Pusat global sequence
    mid_x_global = (min_x + max_x) / 2.0
    mid_y_global = (min_y + max_y) / 2.0

    # ── 2. Skema Warna (Paper Style vs Dark Style) ───────────────────────────
    if theme == "paper":
        # Gaya paper akademik persis Fig. 13 (White background, Forest Green bones, Crimson joints)
        fig_bg = "#ffffff"
        panel_bg = "#ffffff"
        bone_color = "#1e824c"       # Deep Emerald / Forest Green
        joint_color = "#d63031"      # Crimson Red
        joint_edge = "#ffffff"
        text_color = "#2d3436"
        subtext_color = "#636e72"
        border_color = "#dfe6e9"
    else:
        # Gaya dark premium
        fig_bg = "#0f111a"
        panel_bg = "#161925"
        bone_color = "#2ecc71"       # Neon Green
        joint_color = "#e74c3c"      # Coral Red
        joint_edge = "#ffffff"
        text_color = "#f5f6fa"
        subtext_color = "#a4b0be"
        border_color = "#2f3640"

    # ── 3. Buat Figure Matplotlib Horizontal Strip ───────────────────────────
    # Rasio per panel ~ 1.6 : 3.2 inch
    fig, axes = plt.subplots(
        1, n_frames,
        figsize=(n_frames * 1.55, 4.2),
        facecolor=fig_bg
    )
    if n_frames == 1:
        axes = [axes]

    fig.subplots_adjust(
        left=0.02, right=0.98,
        bottom=0.18, top=0.82,
        wspace=0.08
    )

    # ── 4. Gambar Setiap Frame Skeleton ──────────────────────────────────────
    for col, item in enumerate(extracted_sequence):
        ax = axes[col]
        ax.set_facecolor(panel_bg)
        
        lm = item["landmarks_33x3"]
        has_pose = item["has_pose"]

        if has_pose:
            # (A) Gambar Edge Tulang (Hanya jika KEDUA endpoint valid)
            for (i, j) in ANATOMICAL_CONNECTIONS:
                vis_i = lm[i, 2]
                vis_j = lm[j, 2]
                if vis_i >= vis_threshold and vis_j >= vis_threshold:
                    x_coords = [lm[i, 0], lm[j, 0]]
                    y_coords = [-lm[i, 1], -lm[j, 1]]
                    ax.plot(
                        x_coords, y_coords,
                        color=bone_color,
                        linewidth=2.2,
                        solid_capstyle="round",
                        zorder=2
                    )

            # (B) Gambar Titik Sendi (Hanya jika visibility >= threshold)
            for ji in BODY_JOINTS:
                vis_ji = lm[ji, 2]
                if vis_ji >= vis_threshold:
                    ax.scatter(
                        lm[ji, 0], -lm[ji, 1],
                        color=joint_color,
                        s=32,
                        edgecolors=joint_edge,
                        linewidths=0.6,
                        zorder=4
                    )

        # Konfigurasi Tampilan Axes
        ax.set_aspect("equal", adjustable="box")
        
        # Gunakan sequence global window dengan margin proporsional agar gerak translasi (hinge/descent)
        # terlihat utuh dan tidak terpotong di tepi panel
        half_w = (max_range / 2.0) + pad
        ax.set_xlim(mid_x_global - half_w * 0.95, mid_x_global + half_w * 0.95)
        ax.set_ylim(mid_y_global - half_w, mid_y_global + half_w)

        ax.set_xticks([])
        ax.set_yticks([])
        
        # Border halus
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.0)

        # Label nomor frame di bawah tiap panel
        src_num = item["source_frame_num"]
        ax.set_xlabel(
            f"t = {col+1}\n(#{src_num})",
            fontsize=8,
            color=subtext_color,
            fontweight="medium",
            labelpad=6
        )

    # ── 5. Judul & Keterangan Gambar Akademik (Style Fig. 13) ─────────────────
    caption_text = (
        f"Fig. 13.  A visualization of 3D skeletal key points for the action of \"{exercise_name.capitalize()}\" "
        f"({stem}, {n_frames} representative frames)."
    )
    fig.text(
        0.5, 0.05, caption_text,
        ha="center", va="center",
        fontsize=10.5, color=text_color,
        fontweight="bold", fontfamily="sans-serif"
    )

    # Sub-keterangan teknis
    sub_caption = (
        f"Extracted directly from raw video frames using MediaPipe BlazePose (image-space) "
        f"with visibility threshold $\\geq {vis_threshold:.1f}$."
    )
    fig.text(
        0.5, 0.94, sub_caption,
        ha="center", va="center",
        fontsize=8.5, color=subtext_color,
        fontstyle="italic"
    )

    # Simpan Output
    if output_path is None:
        output_path = OUTPUT_DIR / f"{stem}_fig13_skeleton_sequence.png"

    fig.savefig(
        str(output_path),
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none"
    )
    plt.close(fig)

    print(f"[SUKSES] Visualisasi Fig. 13 tersimpan di: {output_path}")
    return output_path


def process_single_video(
    stem: str,
    n_frames: int = 12,
    vis_threshold: float = 0.5,
    theme: str = "paper"
) -> Path | None:
    """Pipeline lengkap untuk satu video."""
    video_path = cari_path_video_mentah(stem)
    if video_path is None or not video_path.exists():
        print(f"[ERROR] Video mentah untuk {stem} tidak ditemukan.")
        return None

    exercise_name = stem.split("_")[0].lower()
    print(f"\n[PROSES] Mengekstraksi MediaPipe raw frames untuk {stem} ({n_frames} frame)...")
    
    extracted = ekstrak_raw_mediapipe_sequence(video_path, n_frames=n_frames)
    out_file = render_fig13_skeleton_strip(
        extracted_sequence=extracted,
        exercise_name=exercise_name,
        stem=stem,
        vis_threshold=vis_threshold,
        theme=theme
    )
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Pembangkit Visualisasi Skeleton-Only Sequence (Fig. 13 Style)")
    parser.add_argument("--stem", type=str, default=None, help="Nama stem video (contoh: Squat_001, BenchPress_001)")
    parser.add_argument("--n_frames", type=int, default=12, help="Jumlah frame yang diambil (10-12 frame)")
    parser.add_argument("--vis_threshold", type=float, default=0.5, help="Ambang batas visibilitas MediaPipe (default: 0.5)")
    parser.add_argument("--theme", type=str, default="paper", choices=["paper", "dark"], help="Tema warna (paper / dark)")
    parser.add_argument("--all_samples", action="store_true", help="Generate sampel untuk Squat, Bench Press, dan Deadlift")
    args = parser.parse_args()

    print("=" * 80)
    print("  PEMBANGKIT SKELETON-ONLY SEQUENCE VISUALIZER (FIG. 13 STYLE)")
    print("=" * 80)
    print(f"Direktori Output: {OUTPUT_DIR}\n")

    if args.stem:
        process_single_video(args.stem, n_frames=args.n_frames, vis_threshold=args.vis_threshold, theme=args.theme)
    else:
        # Default: Generate 3 representative samples for Squat, Bench Press, and Deadlift (plus lateral view)
        sample_stems = [
            "Squat_001",
            "BenchPress_001",
            "BenchPress_002",
            "Deadlift_001",
            "Deadlift_002"
        ]
        print(f"[INFO] Memproses sampel representatif: {sample_stems} ...\n")
        outputs = []
        for s in sample_stems:
            out = process_single_video(s, n_frames=args.n_frames, vis_threshold=args.vis_threshold, theme=args.theme)
            if out:
                outputs.append(out)

        print("\n" + "=" * 80)
        print(f"  SELESAI! Berhasil menghasilkan {len(outputs)} gambar strip Fig. 13.")
        print(f"  Semua file tersimpan di: {OUTPUT_DIR}")
        print("=" * 80)


if __name__ == "__main__":
    main()
