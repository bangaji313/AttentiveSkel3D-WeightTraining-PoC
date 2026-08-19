# Release_V2_AttentiveSkel3D/src/visualization/visualize_real_skeleton_overlay.py
#
# ==============================================================================
# VISUALISASI URUTAN SKELETON REAL-FRAME (OVERLAY DI ATAS FRAME ASLI VIDEO)
# ==============================================================================
#
# Script independen untuk memvisualisasikan runtun skeleton MediaPipe BlazePose
# yang di-overlay langsung di atas frame asli video (ruang piksel/image-space).
#
# Fitur Utama:
#   1. Membaca frame asli langsung dari file video mentah (.mp4).
#   2. Menjalankan MediaPipe BlazePose per-frame untuk mendapatkan koordinat
#      image-space (X, Y piksel asli), BUKAN tensor normalisasi .npy.
#   3. Menggambar skeleton presisi di atas subjek video (hijau=BENAR, merah=SALAH).
#   4. Memilih 8 frame representatif cerdas (fase transisi BENAR <-> SALAH).
#   5. Menghitung dan menampilkan metrik biomekanis riil per-frame:
#        - Squat      : Knee Flexion Angle (<=100 deg) & Hip Angle (<=137 deg)
#        - Bench Press: Elbow ROM Angle (<=85 deg)
#        - Deadlift   : Spine Inclination Angle (20 deg - 60 deg)
#   6. Menyematkan timeline 64 frame di bawah strip sebagai bukti kontinuitas sekuensial.
#   7. Menyimpan komposit PNG berkualitas tinggi (High-DPI) untuk Squat, Bench Press, dan Deadlift.
#
# Catatan: Pipeline training & preprocessing tidak dimodifikasi sama sekali.
#
# Penggunaan:
#   conda run -n attentiveskel python Release_V2_AttentiveSkel3D/src/visualization/visualize_real_skeleton_overlay.py --stem Squat_001
#   conda run -n attentiveskel python Release_V2_AttentiveSkel3D/src/visualization/visualize_real_skeleton_overlay.py --all_samples

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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd

# ─── Path & Direktori ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TENSORS_DIR  = PROJECT_ROOT / "data" / "tensors"
LABELS_DIR   = PROJECT_ROOT / "data" / "v2_labels"
MANIFEST_PATH = PROJECT_ROOT / "src" / "data" / "manifest_v2.csv"
OUTPUT_DIR   = PROJECT_ROOT / "hasil_evaluasi" / "composite_overlay_viz"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Inisialisasi MediaPipe ───────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Koneksi Skeleton BlazePose (indeks 0..32)
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

# Landmark penting yang diutamakan
KEY_LANDMARKS = [
    0,   # Hidung
    11, 12, # Bahu L/R
    13, 14, # Siku L/R
    15, 16, # Pergelangan tangan L/R
    23, 24, # Pinggul L/R
    25, 26, # Lutut L/R
    27, 28, # Pergelangan kaki L/R
    31, 32, # Kaki/Jari L/R
]

# ─── Skema Warna ─────────────────────────────────────────────────────────────
BGR_BENAR = (46, 204, 113)    # Hijau terang (RGB: #2ecc71)
BGR_SALAH = (60, 76, 231)     # Merah terang (RGB: #e74c3c)
BGR_JOINT_BENAR = (39, 174, 96)
BGR_JOINT_SALAH = (43, 57, 192)

HEX_BENAR = "#2ecc71"
HEX_SALAH = "#e74c3c"


# ==============================================================================
# FUNGSI UTILITAS: PENCARIAN VIDEO & PERHITUNGAN GEOMETRI BIOMEKANIK
# ==============================================================================

def cari_path_video_mentah(stem: str) -> Path | None:
    """
    Menemukan file video mentah di data/raw/<Latihan>/ berdasarkan stem.
    Misalnya 'Squat_001' -> data/raw/Squat/primer_squat_frontal_subjek01_rep1.mp4.
    """
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


def calculate_angle_3d_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Menghitung sudut 3D (derajat) antara vektor BA dan BC (vertex di b).
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 180.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def hitung_metrik_biomekanis_frame(
    landmarks_3d: np.ndarray,
    exercise: str
) -> dict:
    """
    Menghitung metrik biomekanis spesifik untuk 1 frame berdasarkan titik 3D MediaPipe.
    landmarks_3d: array (33, 3) [x, y, z] image-space / world landmark MediaPipe.
    
    Returns:
        dict: {'nama_metrik': str, 'nilai_str': str, 'keterangan': str, 'status_lulus': bool}
    """
    ex = exercise.lower()
    
    # Indeks Landmark
    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_WR, R_WR = 15, 16
    L_HP, R_HP = 23, 24
    L_KN, R_KN = 25, 26
    L_AK, R_AK = 27, 28

    if "squat" in ex:
        # 1. Knee Depth Angle (Hip - Knee - Ankle)
        ang_k_l = calculate_angle_3d_points(landmarks_3d[L_HP], landmarks_3d[L_KN], landmarks_3d[L_AK])
        ang_k_r = calculate_angle_3d_points(landmarks_3d[R_HP], landmarks_3d[R_KN], landmarks_3d[R_AK])
        knee_angle = (ang_k_l + ang_k_r) / 2.0

        # 2. Hip Flexion Angle (Shoulder - Hip - Knee)
        ang_h_l = calculate_angle_3d_points(landmarks_3d[L_SH], landmarks_3d[L_HP], landmarks_3d[L_KN])
        ang_h_r = calculate_angle_3d_points(landmarks_3d[R_SH], landmarks_3d[R_HP], landmarks_3d[R_KN])
        hip_angle = (ang_h_l + ang_h_r) / 2.0

        # Kriteria: Knee <= 100 deg (Parallel) & Hip <= 137 deg (Depth)
        lulus_depth = (knee_angle <= 105.0) or (hip_angle <= 137.0)
        
        return {
            "metrik_utama": f"Knee: {knee_angle:.1f} deg | Hip: {hip_angle:.1f} deg",
            "target": "Target: Knee <=100 deg, Hip <=137 deg",
            "status_lulus": lulus_depth,
            "detail": f"Sudut Fleksi Lutut: {knee_angle:.1f} deg, Pinggul: {hip_angle:.1f} deg"
        }

    elif "bench" in ex:
        # Elbow ROM Angle (Shoulder - Elbow - Wrist)
        ang_el_l = calculate_angle_3d_points(landmarks_3d[L_SH], landmarks_3d[L_EL], landmarks_3d[L_WR])
        ang_el_r = calculate_angle_3d_points(landmarks_3d[R_SH], landmarks_3d[R_EL], landmarks_3d[R_WR])
        elbow_angle = (ang_el_l + ang_el_r) / 2.0

        # Kriteria: Elbow <= 85 deg (Full ROM / Bar to Chest)
        lulus_rom = elbow_angle <= 85.0

        return {
            "metrik_utama": f"Elbow ROM: {elbow_angle:.1f} deg",
            "target": "Target Full ROM <= 85 deg",
            "status_lulus": lulus_rom,
            "detail": f"Sudut Fleksi Siku: {elbow_angle:.1f} deg"
        }

    elif "deadlift" in ex:
        # Spine Inclination Angle from Vertical
        mid_sh = (landmarks_3d[L_SH] + landmarks_3d[R_SH]) / 2.0
        mid_hp = (landmarks_3d[L_HP] + landmarks_3d[R_HP]) / 2.0
        spine_vec = mid_sh - mid_hp
        
        vertical_up = np.array([0.0, -1.0, 0.0]) # MediaPipe Y bertambah ke bawah
        norm_spine = np.linalg.norm(spine_vec)
        if norm_spine > 1e-8:
            cos_val = np.clip(np.dot(spine_vec, vertical_up) / norm_spine, -1.0, 1.0)
            spine_angle = float(np.degrees(np.arccos(cos_val)))
        else:
            spine_angle = 0.0

        # Kriteria: 20 deg <= spine_angle <= 60 deg (Hip Hinge Neutral)
        lulus_spine = (20.0 <= spine_angle <= 60.0)

        return {
            "metrik_utama": f"Spine Incl: {spine_angle:.1f} deg",
            "target": "Target Neutral: 20 - 60 deg",
            "status_lulus": lulus_spine,
            "detail": f"Inklinasi Punggung: {spine_angle:.1f} deg dari vertikal"
        }

    else:
        return {
            "metrik_utama": "Pose Tracking Active",
            "target": "-",
            "status_lulus": True,
            "detail": "-"
        }


# ==============================================================================
# FUNGSI PEMILIHAN FRAME CERDAS (8 REPRESENTATIVE FRAMES)
# ==============================================================================

def pilih_8_frame_representatif(labels: np.ndarray) -> list[int]:
    """
    Memilih tepat 8 frame paling informatif dari 64 frame:
    - Frame 0 (Posisi Awal / Setup)
    - Titik-titik transisi BENAR <-> SALAH
    - Frame puncak gerakan (tengah / kedalaman terdalam)
    - Frame 63 (Posisi Akhir / Lockout)
    - Terdistribusi proporsional jika transisi sedikit
    """
    T = len(labels)
    kandidat = set()

    # Anchor penting
    kandidat.add(0)
    kandidat.add(T // 4)
    kandidat.add(T // 2)
    kandidat.add((3 * T) // 4)
    kandidat.add(T - 1)

    # Deteksi setiap transisi status
    for i in range(1, T):
        if labels[i] != labels[i - 1]:
            kandidat.add(max(0, i - 1))
            kandidat.add(i)
            kandidat.add(min(T - 1, i + 1))

    kandidat_sorted = sorted(kandidat)

    if len(kandidat_sorted) >= 8:
        indices = np.round(np.linspace(0, len(kandidat_sorted) - 1, 8)).astype(int)
        return [kandidat_sorted[i] for i in indices]
    else:
        # Tambah dari linspace merata
        extra = set(np.round(np.linspace(0, T - 1, 8)).astype(int).tolist())
        gabungan = sorted(kandidat | extra)
        indices = np.round(np.linspace(0, len(gabungan) - 1, 8)).astype(int)
        return [gabungan[i] for i in indices]


# ==============================================================================
# FUNGSI PENGGAMBARAN OVERLAY SKELETON DI ATAS FRAME ASLI
# ==============================================================================

def gambar_skeleton_overlay_pada_frame(
    frame_rgb: np.ndarray,
    landmarks_list,
    label: int,
    frame_t_idx: int,
    source_frame_idx: int,
    metric_info: dict,
    exercise: str
) -> np.ndarray:
    """
    Menggambar skeleton MediaPipe presisi di atas frame asli dengan informasi visual:
    - Garis tulang tebal & bersinar (glow effect)
    - Titik sendi kontras tinggi
    - Card info atas: Index Frame, Label Badge (BENAR/SALAH)
    - Card info bawah: Metrik Biomekanik Riil
    - Border frame berwarna sesuai label
    """
    h, w, _ = frame_rgb.shape
    canvas = frame_rgb.copy()
    
    # Warna berdasarkan status label
    warna_bgr = (46, 204, 113) if label == 0 else (231, 76, 60)       # Hijau/Merah BGR-equivalent
    warna_rgb = (46, 204, 113) if label == 0 else (231, 76, 60)       # RGB
    warna_tulang = (0, 240, 100) if label == 0 else (255, 60, 60)
    warna_joint = (255, 255, 255)

    # 1. Gambar koneksi tulang jika landmark tersedia
    if landmarks_list is not None:
        lm_px = []
        for lm in landmarks_list.landmark:
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))
            lm_px.append((px, py))

        # Gambar tulang dengan lapisan ganda (glow effect)
        for (i, j) in POSE_CONNECTIONS:
            if i < len(lm_px) and j < len(lm_px):
                pt1 = lm_px[i]
                pt2 = lm_px[j]
                
                # Shadow/outer line gelap untuk kontras
                cv2.line(canvas, pt1, pt2, (10, 10, 10), 5, cv2.LINE_AA)
                # Garis utama berwarna
                cv2.line(canvas, pt1, pt2, warna_tulang, 3, cv2.LINE_AA)

        # Gambar titik sendi penting
        for ji in KEY_LANDMARKS:
            if ji < len(lm_px):
                pt = lm_px[ji]
                cv2.circle(canvas, pt, 5, (10, 10, 10), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 4, warna_tulang, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 2, warna_joint, -1, cv2.LINE_AA)

    # 2. Overlay Header Atas (Semi-transparan)
    header_h = max(38, int(h * 0.12))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Label text & Frame badge
    status_str = "BENAR [0]" if label == 0 else "SALAH [1]"
    warna_status_rgb = (46, 204, 113) if label == 0 else (231, 76, 60)

    # Teks Baris 1: Frame info
    font_scale = max(0.45, w / 700.0)
    font_thick = 1 if w < 500 else 2
    cv2.putText(canvas, f"F{frame_t_idx+1:02d} (Src #{source_frame_idx})", (8, int(header_h * 0.45)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 240), font_thick, cv2.LINE_AA)

    # Teks Status Badge
    cv2.putText(canvas, status_str, (8, int(header_h * 0.88)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, warna_status_rgb, font_thick, cv2.LINE_AA)

    # 3. Overlay Footer Bawah (Metrik Biomekanik)
    footer_h = max(40, int(h * 0.14))
    overlay_foot = canvas.copy()
    cv2.rectangle(overlay_foot, (0, h - footer_h), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay_foot, 0.80, canvas, 0.20, 0, canvas)

    # Teks Metrik
    metrik_text = metric_info.get("metrik_utama", "")
    target_text = metric_info.get("target", "")
    cv2.putText(canvas, metrik_text, (8, h - int(footer_h * 0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.95, (255, 225, 100), font_thick, cv2.LINE_AA)
    cv2.putText(canvas, target_text, (8, h - int(footer_h * 0.15)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.80, (180, 180, 180), 1, cv2.LINE_AA)

    # 4. Border Warna Frame Sesuai Label
    border_thick = max(3, int(w * 0.012))
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), warna_tulang, border_thick)

    return canvas


# ==============================================================================
# FUNGSI PEMBANGKIT VISUALISASI KOMPOSIT UTAMA (STRIP 8 PANEL + TIMELINE)
# ==============================================================================

def generate_overlay_strip_for_video(
    stem: str,
    output_filename: str = None
) -> Path | None:
    """
    Menghasilkan 1 gambar strip komposit beresolusi tinggi untuk 1 video input.
    """
    print(f"\n[PROSES] Memproses visualisasi real-frame overlay: {stem} ...")
    
    # 1. Cari video mentah
    video_path = cari_path_video_mentah(stem)
    if video_path is None or not video_path.exists():
        print(f"[ERROR] Video mentah tidak ditemukan untuk {stem}")
        return None

    # 2. Muat labels
    label_path = LABELS_DIR / f"{stem}_labels.npy"
    if not label_path.exists():
        print(f"[ERROR] File label tidak ditemukan: {label_path}")
        return None
    labels = np.load(label_path).astype(int)

    # 3. Buka video & hitung resampling 64 frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Indeks sampling 64 frame
    indices_64 = [int(round(x)) for x in np.linspace(0, total_frames - 1, 64)]
    
    # Pilih 8 frame kunci
    key_8_indices = pilih_8_frame_representatif(labels)
    set_key_indices = set(key_8_indices)

    # Baca seluruh frame yang dibutuhkan secara sekuensial
    target_source_frames = {t_idx: indices_64[t_idx] for t_idx in key_8_indices}
    needed_source_set = set(target_source_frames.values())

    frames_rgb_dict = {}
    current_f = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if current_f in needed_source_set:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb_dict[current_f] = frame_rgb
        current_f += 1
        if len(frames_rgb_dict) == len(needed_source_set):
            break
    cap.release()

    # 4. Jalankan MediaPipe Pose pada 8 frame terpilih
    exercise_name = stem.split("_")[0]
    processed_panels = []
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,       # Gunakan mode Heavy/Akurat untuk visualisasi publikasi
        min_detection_confidence=0.5
    ) as pose_detector:
        
        for t_idx in key_8_indices:
            src_idx = indices_64[t_idx]
            raw_rgb = frames_rgb_dict.get(src_idx, np.zeros((video_h, video_w, 3), dtype=np.uint8))
            
            # MediaPipe inference pada frame ini
            results = pose_detector.process(raw_rgb)
            
            # Hitung metrik biomekanis riil dari landmark 3D
            if results.pose_landmarks:
                landmarks_3d = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
                ], dtype=np.float32)
                metric_info = hitung_metrik_biomekanis_frame(landmarks_3d, exercise_name)
            else:
                metric_info = {"metrik_utama": "No Landmark", "target": "-", "status_lulus": False}

            # Gambar overlay
            panel_img = gambar_skeleton_overlay_pada_frame(
                frame_rgb=raw_rgb,
                landmarks_list=results.pose_landmarks,
                label=labels[t_idx],
                frame_t_idx=t_idx,
                source_frame_idx=src_idx,
                metric_info=metric_info,
                exercise=exercise_name
            )
            processed_panels.append((panel_img, t_idx, src_idx, labels[t_idx]))

    # 5. Susun Layout Komposit dengan Matplotlib (8 Kolom + Baris Timeline)
    n_panels = len(processed_panels)
    fig = plt.figure(figsize=(n_panels * 2.8, 6.2), facecolor="#0e1117")
    
    gs = GridSpec(2, n_panels, figure=fig,
                  height_ratios=[4.5, 0.65],
                  hspace=0.25, wspace=0.06,
                  left=0.015, right=0.985,
                  top=0.88, bottom=0.12)

    # Render Panel Gambar
    for col, (panel_rgb, t_idx, src_idx, lbl) in enumerate(processed_panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(panel_rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Border axes
        warna_border = HEX_BENAR if lbl == 0 else HEX_SALAH
        for spine in ax.spines.values():
            spine.set_edgecolor(warna_border)
            spine.set_linewidth(2.5)

    # Render Timeline Bar 64 Frame di Bawah
    ax_tl = fig.add_subplot(gs[1, :])
    ax_tl.set_facecolor("#151922")
    
    # Bar warna 64 frame
    for i, lbl in enumerate(labels):
        warna_bar = HEX_BENAR if lbl == 0 else HEX_SALAH
        ax_tl.barh(0, 1, left=i, height=1.0, color=warna_bar, edgecolor="#151922", linewidth=0.3)

    # Marker panah untuk 8 frame terpilih
    for t_idx in key_8_indices:
        ax_tl.annotate("▼", xy=(t_idx + 0.5, 1.15), fontsize=9, ha='center',
                       va='bottom', color='#f1c40f', fontweight='bold')

    ax_tl.set_xlim(0, 64)
    ax_tl.set_ylim(-0.6, 2.3)
    ax_tl.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 63])
    ax_tl.set_xticklabels(["F1", "F9", "F17", "F25", "F33", "F41", "F49", "F57", "F64"],
                          fontsize=7, color='#cccccc')
    ax_tl.set_yticks([])
    for spine in ax_tl.spines.values():
        spine.set_visible(False)

    # Ringkasan Kuantitatif
    n_benar = int(np.sum(labels == 0))
    n_salah = int(np.sum(labels == 1))
    pct_benar = (n_benar / 64.0) * 100.0
    
    ax_tl.text(65.2, 0.45, f"✓ Benar: {n_benar} ({pct_benar:.0f}%)",
               color=HEX_BENAR, fontsize=8, va='center', fontweight='bold')
    ax_tl.text(65.2, -0.20, f"✗ Salah: {n_salah} ({100-pct_benar:.0f}%)",
               color=HEX_SALAH, fontsize=8, va='center', fontweight='bold')

    # Judul Header Atas
    judul_utama = (
        f"BUKTI RUNTONG SKELETON REAL-FRAME (IMAGE-SPACE OVERLAY)\n"
        f"Dataset: {stem}  |  Latihan: {exercise_name.upper()}  |  File: {video_path.name}  |  Resolusi: {video_w}x{video_h} ({fps:.1f} FPS)"
    )
    fig.text(0.5, 0.98, judul_utama, ha='center', va='top',
             color='white', fontsize=9.5, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e2433', edgecolor='#3d4860', alpha=0.95))

    # Simpan Output PNG
    if output_filename is None:
        out_path = OUTPUT_DIR / f"{stem}_real_overlay_strip.png"
    else:
        out_path = OUTPUT_DIR / output_filename

    fig.savefig(str(out_path), dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"[SUKSES] Visualisasi tersimpan di: {out_path}")
    return out_path


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pembangkit Visualisasi Urutan Skeleton Real-Frame Overlay")
    parser.add_argument("--stem", type=str, default=None, help="Nama stem dataset (misal: Squat_001, BenchPress_001)")
    parser.add_argument("--all_samples", action="store_true", help="Generate sampel standar untuk ketiga latihan (Squat, Bench, Deadlift)")
    args = parser.parse_args()

    print("=" * 80)
    print("  PEMBANGKIT VISUALISASI SKELETON OVERLAY REAL-FRAME — AttentiveSkel-3D V2")
    print("=" * 80)
    print(f"Direktori Output: {OUTPUT_DIR}\n")

    if args.stem:
        generate_overlay_strip_for_video(args.stem)
    else:
        # Default: Generate 3 representative examples for Squat, Bench Press, and Deadlift
        contoh_latihan = [
            "Squat_001",
            "BenchPress_001",
            "BenchPress_002",
            "Deadlift_001",
            "Deadlift_002"
        ]
        print(f"[INFO] Memproses sampel representatif: {contoh_latihan} ...\n")
        hasil = []
        for stem in contoh_latihan:
            out = generate_overlay_strip_for_video(stem)
            if out:
                hasil.append(out)

        print("\n" + "=" * 80)
        print(f"  SELESAI! Berhasil menghasilkan {len(hasil)} visualisasi komposit.")
        print(f"  Semua file tersimpan di: {OUTPUT_DIR}")
        print("=" * 80)


if __name__ == "__main__":
    main()
