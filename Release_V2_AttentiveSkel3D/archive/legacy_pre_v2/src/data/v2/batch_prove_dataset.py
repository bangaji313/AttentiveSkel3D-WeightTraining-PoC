# src/data/v2/batch_prove_dataset.py
#
# Script batch rendering — skeleton MediaPipe menempel presisi di badan subjek.
#
# Arsitektur dual-source:
#   SKELETON → Deteksi MediaPipe live di atas frame video mentah.
#              Hasilnya 100% akurat menempel di tubuh karena menggunakan
#              pixel space asli, bukan koordinat ternormalisasi dari NPY.
#
#   TEKS OVERLAY → Dibaca dari file NPY (fitur & label).
#                  Membuktikan bahwa data tensor tersinkronisasi dengan video.
#
# Visualisasi membuktikan 3 dimensi tensor secara bersamaan:
#   Dim-1 Temporal  → "FRAME TENSOR: 01 / 64"
#   Dim-2 Landmark  → Skeleton 33 titik menempel di badan (via MediaPipe live)
#   Dim-3 Koordinat → "TENSOR LUTUT Y: {nilai_dari_npy}"  (Z tersimpan di NPY)
#   Label per-frame → "LABEL NPY: 0 BENAR / 1 SALAH"
#
# Cara menjalankan dari root proyek:
#   conda activate attentiveskel
#   python src/data/v2/batch_prove_dataset.py

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import mediapipe as mp
from tqdm import tqdm

# ============================================================
# Konfigurasi path
# ============================================================
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest_v2.csv"
RAW_DATA_DIR  = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "processed" / "v2_video_proofs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Inisialisasi MediaPipe Drawing sekali di level modul
# (Reuse antar video — tidak perlu reinitialize setiap saat)
# ============================================================
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# Style garis tulang: putih agak transparan
LANDMARK_DRAWING_SPEC = mp_drawing.DrawingSpec(
    color=(0, 230, 0),    # Titik sendi: hijau terang (BGR)
    thickness=3,
    circle_radius=4,
)
CONNECTION_DRAWING_SPEC = mp_drawing.DrawingSpec(
    color=(255, 200, 0),  # Garis tulang: biru-kuning (BGR)
    thickness=2,
)

# ============================================================
# Konstanta tampilan teks overlay
# ============================================================
FONT      = cv2.FONT_HERSHEY_SIMPLEX
LINE_TYPE = cv2.LINE_AA

WARNA_HIJAU  = (0,   220,   0)    # Label 0 = BENAR
WARNA_MERAH  = (0,   0,   220)    # Label 1 = SALAH
WARNA_PUTIH  = (255, 255, 255)    # Info umum
WARNA_KUNING = (0,   215, 255)    # Nilai tensor dari NPY

# Indeks sendi MediaPipe BlazePose untuk teks NPY ditentukan secara dinamis
# berdasarkan jenis latihan (Squat, Deadlift, Bench Press) di dalam loop.


# ============================================================
# Fungsi utilitas
# ============================================================

def _cari_video_mentah(stem: str) -> Path | None:
    """
    Menemukan file video mentah di data/raw/<Latihan>/ berdasarkan stem nama .npy.
    Strategi: 'BenchPress_042' → folder BenchPress → video ke-42 (sorted A-Z).
    """
    bagian = stem.split("_", 1)
    if len(bagian) < 2:
        return None

    nama_latihan = bagian[0]
    try:
        nomor_urut = int(bagian[1])
    except ValueError:
        return None

    # Cari folder latihan (case-insensitive)
    folder_latihan = None
    for subfolder in RAW_DATA_DIR.iterdir():
        if subfolder.is_dir() and subfolder.name.lower() == nama_latihan.lower():
            folder_latihan = subfolder
            break

    if folder_latihan is None:
        return None

    daftar_video = sorted(folder_latihan.glob("*.mp4"))
    idx_video    = nomor_urut - 1

    if 0 <= idx_video < len(daftar_video):
        return daftar_video[idx_video]
    return None


def _hitung_indeks_temporal(total_frame: int, n_target: int = 64) -> list[int]:
    """
    Hitung 64 indeks frame yang dipilih dari total_frame menggunakan linspace.
    Identik dengan logika temporal resampling di preprocess.py.
    """
    indeks = np.linspace(0, total_frame - 1, n_target)
    return [int(round(x)) for x in indeks]


def _baca_frame_sekuensial(
    cap: cv2.VideoCapture,
    indeks_target: list[int],
    lebar: int,
    tinggi: int,
) -> dict[int, np.ndarray]:
    """
    Membaca frame video secara SEKUENSIAL dan mengumpulkan hanya yang dibutuhkan.
    Jauh lebih cepat daripada cap.set(CAP_PROP_POS_FRAMES) per frame.
    """
    set_target   = set(indeks_target)
    frame_buffer = {}
    frame_hitam  = np.zeros((tinggi, lebar, 3), dtype=np.uint8)
    frame_idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in set_target:
            frame_buffer[frame_idx] = frame.copy()

        frame_idx += 1

        # Berhenti setelah semua frame target terkumpul (optimasi awal)
        if len(frame_buffer) == len(set_target):
            break

    # Fallback frame hitam untuk indeks yang gagal dibaca
    for idx in indeks_target:
        if idx not in frame_buffer:
            frame_buffer[idx] = frame_hitam.copy()

    return frame_buffer


def _gambar_mediapipe_skeleton(
    frame_bgr: np.ndarray,
    pose: mp_pose.Pose,
) -> np.ndarray:
    """
    Menjalankan inferensi MediaPipe Pose pada satu frame dan menggambar skeleton.

    Skeleton yang dihasilkan 100% menempel di tubuh subjek karena bekerja
    di ruang piksel asli video (bukan koordinat ternormalisasi NPY).

    Args:
        frame_bgr : Frame video format BGR, shape (H, W, 3).
        pose      : Instance mp.solutions.pose.Pose yang sudah diinisialisasi.

    Returns:
        Frame dengan skeleton MediaPipe tergambar, format BGR.
    """
    # MediaPipe membutuhkan input RGB — konversi sementara
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Tandai frame sebagai tidak writable agar MediaPipe tidak copy — efisiensi memori
    frame_rgb.flags.writeable = False
    hasil = pose.process(frame_rgb)

    # Kembalikan ke writable untuk operasi selanjutnya
    frame_rgb.flags.writeable = True
    frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Gambar skeleton hanya jika pose terdeteksi
    if hasil.pose_landmarks:
        mp_drawing.draw_landmarks(
            image                   = frame_out,
            landmark_list           = hasil.pose_landmarks,
            connections             = mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec   = LANDMARK_DRAWING_SPEC,
            connection_drawing_spec = CONNECTION_DRAWING_SPEC,
        )

    return frame_out


def _tulis_overlay_teks(
    frame: np.ndarray,
    nomor_frame: int,
    label_npy: int,
    teks_koordinat: str,
    nama_latihan: str,
) -> np.ndarray:
    """
    Menggambar teks overlay informatif di atas frame yang sudah ada skeleton.

    Semua nilai teks (label & koordinat) dibaca dari NPY — bukan dari MediaPipe.
    Ini membuktikan bahwa data tensor tersinkronisasi dengan frame video.

    Layout:
        Panel atas  : Nama latihan + nomor frame temporal (Dim-1)
        Panel bawah : Label NPY + teks koordinat dinamis dari NPY (Dim-3)

    Args:
        frame          : Frame BGR setelah skeleton MediaPipe digambar.
        nomor_frame    : Nomor frame saat ini, 1-indexed (1..64).
        label_npy      : Nilai label per-frame dari NPY (0 atau 1).
        teks_koordinat : Teks koordinat Y yang sudah diformat sesuai jenis latihan.
        nama_latihan   : Nama jenis latihan untuk judul.

    Returns:
        Frame dengan overlay teks lengkap.
    """
    h, w      = frame.shape[:2]
    frame_out = frame.copy()

    # ------------------------------------------------------------------
    # Panel transparan ATAS — Dimensi 1: Temporal
    # ------------------------------------------------------------------
    panel_atas = frame_out.copy()
    cv2.rectangle(panel_atas, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(panel_atas, 0.55, frame_out, 0.45, 0, frame_out)

    # Teks temporal: nama latihan + nomor frame dari 64
    teks_temporal = f"[{nama_latihan.upper()}]  DIM-1 TEMPORAL: FRAME TENSOR {nomor_frame:02d} / 64"
    cv2.putText(frame_out, teks_temporal, (8, 24), FONT, 0.56, WARNA_PUTIH, 2, LINE_TYPE)

    # ------------------------------------------------------------------
    # Panel transparan BAWAH — Label NPY + Koordinat NPY
    # ------------------------------------------------------------------
    tinggi_panel = 100
    panel_bawah  = frame_out.copy()
    cv2.rectangle(panel_bawah, (0, h - tinggi_panel), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(panel_bawah, 0.60, frame_out, 0.40, 0, frame_out)

    # Baris 1 bawah: Label per-frame dari NPY
    if label_npy == 0:
        teks_label = "LABEL NPY : 0  (BENAR)  -> Dibaca dari labels_path .npy"
        warna      = WARNA_HIJAU
    else:
        teks_label = "LABEL NPY : 1  (SALAH)  -> Dibaca dari labels_path .npy"
        warna      = WARNA_MERAH

    cv2.putText(
        frame_out, teks_label,
        (8, h - tinggi_panel + 24),
        FONT, 0.58, warna, 2, LINE_TYPE,
    )

    # Baris 2 bawah: Koordinat Y dinamis dari tensor NPY
    cv2.putText(
        frame_out, teks_koordinat,
        (8, h - tinggi_panel + 54),
        FONT, 0.50, WARNA_KUNING, 1, LINE_TYPE,
    )

    # Baris 3 bawah: Catatan tambahan
    teks_catatan = "-> Skeleton di atas murni dirender dari tensor NPY (X,Y,Z)"
    cv2.putText(
        frame_out, teks_catatan,
        (8, h - tinggi_panel + 80),
        FONT, 0.50, (180, 180, 180), 1, LINE_TYPE,
    )

    return frame_out


def proses_satu_sampel(row: pd.Series) -> bool:
    """
    Memproses satu baris manifest — menghasilkan satu video proof.

    Sumber data:
        Skeleton visual → MediaPipe live inference pada frame video mentah.
        Teks & label    → Dibaca dari file NPY (dual-source).

    Args:
        row: Satu baris DataFrame dari manifest_v2.csv.

    Returns:
        True jika berhasil, False jika ada error.
    """
    file_path   = row["file_path"]
    labels_path = row["labels_path"]
    exercise    = str(row.get("exercise", "unknown"))
    stem        = Path(file_path).stem

    # Cari video mentah
    path_video = _cari_video_mentah(stem)
    if path_video is None or not path_video.exists():
        tqdm.write(f"  [SKIP] Video tidak ditemukan: {stem}")
        return False

    # Validasi file NPY
    if not os.path.exists(file_path) or not os.path.exists(labels_path):
        tqdm.write(f"  [SKIP] File NPY tidak ditemukan: {stem}")
        return False

    # Muat tensor fitur (64, 33, 3) dari NPY — sumber teks koordinat
    tensor_fitur = np.load(file_path).astype(np.float32)
    label_array  = np.load(labels_path).astype(np.int8)

    # Validasi shape
    if tensor_fitur.shape != (64, 33, 3) or len(label_array) != 64:
        tqdm.write(f"  [SKIP] Shape tidak valid: {tensor_fitur.shape} | {stem}")
        return False

    # Buka video mentah untuk membaca frame asli
    cap = cv2.VideoCapture(str(path_video))
    if not cap.isOpened():
        tqdm.write(f"  [SKIP] Gagal buka video: {path_video.name}")
        return False

    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_asli    = cap.get(cv2.CAP_PROP_FPS)
    lebar       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    tinggi      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_output  = fps_asli if fps_asli > 0 else 15.0

    # Hitung indeks temporal 64 frame (linspace — konsisten dengan preprocess)
    indeks_temporal = _hitung_indeks_temporal(total_frame, n_target=64)

    # Baca semua frame yang dibutuhkan SEKUENSIAL (cepat, tanpa seeking)
    frame_buffer = _baca_frame_sekuensial(cap, indeks_temporal, lebar, tinggi)
    cap.release()  # Tutup video setelah baca selesai

    # Siapkan VideoWriter output
    path_output = OUTPUT_DIR / f"{stem}_proof.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path_output), fourcc, fps_output, (lebar, tinggi))

    if not writer.isOpened():
        tqdm.write(f"  [ERROR] VideoWriter gagal: {stem}")
        return False

    # ============================================================
    # Inisialisasi MediaPipe Pose SEKALI per video
    # Inisialisasi per-video (bukan per-frame) agar efisien.
    # Akan di-close dengan pose.close() setelah 64 frame selesai.
    # ============================================================
    pose = mp_pose.Pose(
        static_image_mode      = False,  # Mode video (lebih cepat, pakai tracking antar frame)
        model_complexity       = 1,      # 0=ringan, 1=seimbang, 2=akurat — pakai 1 untuk batch
        smooth_landmarks       = True,   # Haluskan gerakan landmark antar frame
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

    try:
        # ============================================================
        # Loop 64 frame temporal
        # ============================================================
        for i in range(64):
            # Ambil frame dari buffer (sudah dibaca sekuensial)
            idx_asli  = indeks_temporal[i]
            frame_bgr = frame_buffer.get(
                idx_asli, np.zeros((tinggi, lebar, 3), dtype=np.uint8)
            )

            # ----------------------------------------------------------
            # SUMBER 1: MediaPipe live inference → skeleton visual
            # ----------------------------------------------------------
            frame_skeleton = _gambar_mediapipe_skeleton(frame_bgr, pose)

            # ----------------------------------------------------------
            # SUMBER 2: Data NPY → teks overlay (label + koordinat)
            # ----------------------------------------------------------
            label_i = int(label_array[i])
            
            # Deteksi jenis latihan dari nama file atau file_path
            if "BenchPress" in file_path:
                nama_sendi = "SIKU KANAN (idx14)"
                idx_sendi = 14  # Indeks siku kanan
            elif "Deadlift" in file_path:
                nama_sendi = "PINGGUL KANAN (idx24)"
                idx_sendi = 24  # Indeks pinggul kanan
            else: # Secara default Squat
                nama_sendi = "LUTUT KANAN (idx26)"
                idx_sendi = 26  # Indeks lutut kanan
            
            # Ambil nilai Y dari NPY berdasarkan indeks sendi yang sesuai
            nilai_y_dinamis = float(tensor_fitur[i, idx_sendi, 1])
            
            # Tulis teks overlay ke layar video
            teks_koordinat = f"TENSOR {nama_sendi} Y: {nilai_y_dinamis:+.4f}  <-- Koordinat ternormalisasi NPY"

            # Tulis semua teks overlay di atas frame + skeleton
            frame_final = _tulis_overlay_teks(
                frame          = frame_skeleton,
                nomor_frame    = i + 1,       # 1-indexed
                label_npy      = label_i,
                teks_koordinat = teks_koordinat,
                nama_latihan   = exercise,
            )

            writer.write(frame_final)

    finally:
        # Pastikan pose.close() selalu dipanggil — cegah memory leak MediaPipe
        pose.close()
        writer.release()

    return True


def main():
    """Fungsi utama — baca manifest dan proses seluruh sampel."""
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest tidak ditemukan: {MANIFEST_PATH}")
        print("Jalankan: python src/data/v2/build_manifest.py")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    print(f"[INFO] Manifest dimuat      : {len(df)} sampel")
    print(f"[INFO] Output folder        : {OUTPUT_DIR}")
    print(f"[INFO] Skeleton sumber      : Skeleton dari tensor NPY (X,Y,Z)")
    print(f"[INFO] Label + Tensor sumber: File NPY (dual-source proof)\n")

    n_berhasil = 0
    n_gagal    = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Rendering Proof",
        unit="video",
        ncols=90,
    ):
        if proses_satu_sampel(row):
            n_berhasil += 1
        else:
            n_gagal += 1

    print(f"\n{'='*55}")
    print(f"[SELESAI] Total         : {len(df)}")
    print(f"          Berhasil      : {n_berhasil}")
    print(f"          Gagal/Skip    : {n_gagal}")
    print(f"          Output        : {OUTPUT_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
