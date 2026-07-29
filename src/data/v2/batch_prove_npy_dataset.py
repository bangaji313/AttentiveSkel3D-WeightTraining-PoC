# src/data/v2/batch_prove_npy_dataset.py
#
# Script batch rendering — skeleton murni digambar dari file NPY.
#
# Arsitektur:
#   SKELETON → Digambar manual dari tensor NPY (64 frame, 33 joint, 3 koordinat).
#              Membuktikan bahwa isi tensor NPY (X, Y, Z) valid dan bisa
#              divisualisasikan kembali ke atas frame aslinya.
#
#   TEKS OVERLAY → Dibaca dari file NPY (fitur & label).
#
# Cara menjalankan dari root proyek:
#   conda activate attentiveskel
#   python src/data/v2/batch_prove_npy_dataset.py

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
from mediapipe.framework.formats import landmark_pb2
from tqdm import tqdm

# ============================================================
# Konfigurasi path
# ============================================================
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest_v2.csv"
RAW_DATA_DIR  = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "processed" / "v2_video_proofs_npy"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Inisialisasi MediaPipe Drawing sekali di level modul
# ============================================================
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# Style garis tulang: merah untuk membedakan dengan live MP
LANDMARK_DRAWING_SPEC = mp_drawing.DrawingSpec(
    color=(0, 0, 255),    # Titik sendi: merah (BGR)
    thickness=3,
    circle_radius=4,
)
CONNECTION_DRAWING_SPEC = mp_drawing.DrawingSpec(
    color=(0, 165, 255),  # Garis tulang: oranye (BGR)
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

# ============================================================
# Fungsi utilitas
# ============================================================

def _cari_video_mentah(stem: str) -> Path | None:
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
    idx_video    = nomor_urut - 1

    if 0 <= idx_video < len(daftar_video):
        return daftar_video[idx_video]
    return None


def _hitung_indeks_temporal(total_frame: int, n_target: int = 64) -> list[int]:
    indeks = np.linspace(0, total_frame - 1, n_target)
    return [int(round(x)) for x in indeks]


def _baca_frame_sekuensial(
    cap: cv2.VideoCapture,
    indeks_target: list[int],
    lebar: int,
    tinggi: int,
) -> dict[int, np.ndarray]:
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

        if len(frame_buffer) == len(set_target):
            break

    for idx in indeks_target:
        if idx not in frame_buffer:
            frame_buffer[idx] = frame_hitam.copy()

    return frame_buffer


def _gambar_skeleton_dari_tensor(
    frame_bgr: np.ndarray,
    tensor_frame: np.ndarray,
) -> np.ndarray:
    """
    Menggambar skeleton dari koordinat tensor NPY (33, 3).
    """
    frame_out = frame_bgr.copy()
    
    # Buat objek NormalizedLandmarkList dari protobuf mediapipe
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for j in range(33):
        landmark = landmark_list.landmark.add()
        landmark.x = float(tensor_frame[j, 0])
        landmark.y = float(tensor_frame[j, 1])
        landmark.z = float(tensor_frame[j, 2])
        landmark.visibility = 1.0 # Anggap visibility 1 agar selalu digambar

    mp_drawing.draw_landmarks(
        image                   = frame_out,
        landmark_list           = landmark_list,
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
    h, w      = frame.shape[:2]
    frame_out = frame.copy()

    panel_atas = frame_out.copy()
    cv2.rectangle(panel_atas, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(panel_atas, 0.55, frame_out, 0.45, 0, frame_out)

    teks_temporal = f"[{nama_latihan.upper()}]  DIM-1 TEMPORAL: FRAME TENSOR {nomor_frame:02d} / 64"
    cv2.putText(frame_out, teks_temporal, (8, 24), FONT, 0.56, WARNA_PUTIH, 2, LINE_TYPE)

    tinggi_panel = 100
    panel_bawah  = frame_out.copy()
    cv2.rectangle(panel_bawah, (0, h - tinggi_panel), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(panel_bawah, 0.60, frame_out, 0.40, 0, frame_out)

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

    cv2.putText(
        frame_out, teks_koordinat,
        (8, h - tinggi_panel + 54),
        FONT, 0.50, WARNA_KUNING, 1, LINE_TYPE,
    )

    teks_catatan = "-> Skeleton di atas murni dirender dari tensor NPY (X,Y,Z)"
    cv2.putText(
        frame_out, teks_catatan,
        (8, h - tinggi_panel + 80),
        FONT, 0.50, (180, 180, 180), 1, LINE_TYPE,
    )

    return frame_out


def proses_satu_sampel(row: pd.Series) -> bool:
    file_path   = row["file_path"]
    labels_path = row["labels_path"]
    exercise    = str(row.get("exercise", "unknown"))
    stem        = Path(file_path).stem

    path_video = _cari_video_mentah(stem)
    if path_video is None or not path_video.exists():
        tqdm.write(f"  [SKIP] Video tidak ditemukan: {stem}")
        return False

    if not os.path.exists(file_path) or not os.path.exists(labels_path):
        tqdm.write(f"  [SKIP] File NPY tidak ditemukan: {stem}")
        return False

    tensor_fitur = np.load(file_path).astype(np.float32)
    label_array  = np.load(labels_path).astype(np.int8)

    if tensor_fitur.shape != (64, 33, 3) or len(label_array) != 64:
        tqdm.write(f"  [SKIP] Shape tidak valid: {tensor_fitur.shape} | {stem}")
        return False

    cap = cv2.VideoCapture(str(path_video))
    if not cap.isOpened():
        tqdm.write(f"  [SKIP] Gagal buka video: {path_video.name}")
        return False

    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_asli    = cap.get(cv2.CAP_PROP_FPS)
    lebar       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    tinggi      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_output  = fps_asli if fps_asli > 0 else 15.0

    indeks_temporal = _hitung_indeks_temporal(total_frame, n_target=64)
    frame_buffer = _baca_frame_sekuensial(cap, indeks_temporal, lebar, tinggi)
    cap.release()

    path_output = OUTPUT_DIR / f"{stem}_proof.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path_output), fourcc, fps_output, (lebar, tinggi))

    if not writer.isOpened():
        tqdm.write(f"  [ERROR] VideoWriter gagal: {stem}")
        return False

    try:
        for i in range(64):
            idx_asli  = indeks_temporal[i]
            frame_bgr = frame_buffer.get(
                idx_asli, np.zeros((tinggi, lebar, 3), dtype=np.uint8)
            )

            # SUMBER: Data NPY -> skeleton visual
            tensor_frame = tensor_fitur[i]  # shape: (33, 3)
            frame_skeleton = _gambar_skeleton_dari_tensor(frame_bgr, tensor_frame)

            label_i = int(label_array[i])
            
            if "BenchPress" in file_path:
                nama_sendi = "SIKU KANAN (idx14)"
                idx_sendi = 14
            elif "Deadlift" in file_path:
                nama_sendi = "PINGGUL KANAN (idx24)"
                idx_sendi = 24
            else:
                nama_sendi = "LUTUT KANAN (idx26)"
                idx_sendi = 26
            
            nilai_y_dinamis = float(tensor_fitur[i, idx_sendi, 1])
            teks_koordinat = f"TENSOR {nama_sendi} Y: {nilai_y_dinamis:+.4f}  <-- Koordinat ternormalisasi NPY"

            frame_final = _tulis_overlay_teks(
                frame          = frame_skeleton,
                nomor_frame    = i + 1,
                label_npy      = label_i,
                teks_koordinat = teks_koordinat,
                nama_latihan   = exercise,
            )

            writer.write(frame_final)

    finally:
        writer.release()

    return True


def main():
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest tidak ditemukan: {MANIFEST_PATH}")
        print("Jalankan: python src/data/v2/build_manifest.py")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    print(f"[INFO] Manifest dimuat      : {len(df)} sampel")
    print(f"[INFO] Output folder        : {OUTPUT_DIR}")
    print(f"[INFO] Skeleton sumber      : Murni dari tensor NPY (64x33x3)")
    print(f"[INFO] Label + Tensor sumber: File NPY\n")

    n_berhasil = 0
    n_gagal    = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Rendering NPY Proof",
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
