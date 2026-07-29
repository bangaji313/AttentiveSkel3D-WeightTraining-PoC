# src/data/v2/build_manifest.py
#
# Membangun manifest_v2.csv untuk pelatihan per-frame (64 label per video).
# Setiap baris manifest memetakan satu file .npy ke label array (64,) yang
# dihasilkan oleh BiomechanicalValidator per-frame.
#
# Berbeda dari v1 (satu label global per video), v2 menghasilkan:
#   - Satu baris per file .npy
#   - Kolom 'labels_path': path ke file .npy terpisah berisi array label (64,)
#
# Cara menjalankan dari root proyek:
#   python src/data/v2/build_manifest.py

import sys
from pathlib import Path

# Tambahkan root proyek ke sys.path agar modul src dapat diimport
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.biomechanics_validator import BiomechanicalValidator

# ============================================================
# Konfigurasi path
# ============================================================
TENSORS_DIR     = PROJECT_ROOT / "data" / "processed" / "tensors"
LABELS_DIR      = PROJECT_ROOT / "data" / "processed" / "v2_labels"
MANIFEST_OUT    = PROJECT_ROOT / "data" / "processed" / "manifest_v2.csv"

# Buat folder output untuk label array per-frame
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Peta nama gerakan (dari nama subfolder/file) ke metode validator
# ============================================================
# Nama file .npy diasumsikan mengandung kata kunci gerakan:
# "squat", "benchpress" / "bench_press" / "bench press", "deadlift"
def _detect_exercise(stem: str) -> str:
    """Deteksi jenis gerakan dari nama file (case-insensitive)."""
    s = stem.lower()
    if "squat" in s:
        return "squat"
    if "bench" in s:
        return "benchpress"
    if "dead" in s:
        return "deadlift"
    # Fallback: kembalikan 'unknown' agar baris ini dilewati
    return "unknown"


def _validate_per_frame(
    tensor: np.ndarray,
    exercise: str,
    validator: BiomechanicalValidator,
) -> np.ndarray:
    """
    Menghasilkan array label (64,) dari satu tensor (64, 33, 3).

    Strategi:
    - Setiap frame ke-i dievaluasi menggunakan tensor satu frame bentuk (1, 33, 3).
    - Validator biomekanis dirancang untuk tensor (F, 33, 3), sehingga kita
      mengirimkan irisan satu frame sebagai (1, 33, 3).
    - Jika frame tidak valid (sudut tidak terpenuhi), label = 1 (Salah).
    - Jika frame valid, label = 0 (Benar).

    Args:
        tensor   : Array (64, 33, 3) float32.
        exercise : Nama gerakan: 'squat', 'benchpress', atau 'deadlift'.
        validator: Instance BiomechanicalValidator.

    Returns:
        np.ndarray: Array label (64,) berisi 0 atau 1, dtype int8.
    """
    n_frames = tensor.shape[0]  # Biasanya 64
    labels   = np.zeros(n_frames, dtype=np.int8)

    for i in range(n_frames):
        # Irisan satu frame, tambahkan dimensi batch → (1, 33, 3)
        frame_tensor = tensor[i : i + 1, :, :]  # (1, 33, 3)

        # Pilih metode validator sesuai gerakan
        if exercise == "squat":
            is_valid, _ = validator.validate_squat(frame_tensor)
        elif exercise == "benchpress":
            is_valid, _ = validator.validate_benchpress(frame_tensor)
        elif exercise == "deadlift":
            is_valid, _ = validator.validate_deadlift(frame_tensor)
        else:
            # Gerakan tidak dikenal → anggap salah agar tidak mempengaruhi training
            is_valid = False

        labels[i] = 0 if is_valid else 1

    return labels


def main():
    validator = BiomechanicalValidator()

    # Kumpulkan semua file .npy dari direktori tensors
    npy_files = sorted(TENSORS_DIR.rglob("*.npy"))
    if not npy_files:
        print(f"[ERROR] Tidak ada file .npy di: {TENSORS_DIR}")
        return

    print(f"[INFO] Ditemukan {len(npy_files)} file .npy di {TENSORS_DIR}")
    print(f"[INFO] Menghasilkan label per-frame dan menyimpan ke {LABELS_DIR}")

    records = []

    for npy_path in tqdm(npy_files, desc="Memproses tensor"):
        exercise = _detect_exercise(npy_path.stem)

        # Lewati file yang jenis gerakannya tidak dikenal
        if exercise == "unknown":
            print(f"  [SKIP] Gerakan tidak terdeteksi: {npy_path.name}")
            continue

        # Muat tensor (64, 33, 3)
        tensor = np.load(str(npy_path)).astype(np.float32)

        # Validasi shape: harus (64, 33, 3)
        if tensor.ndim != 3 or tensor.shape[1:] != (33, 3):
            print(f"  [SKIP] Shape tidak valid {tensor.shape}: {npy_path.name}")
            continue

        # Generate array label per-frame (64,)
        per_frame_labels = _validate_per_frame(tensor, exercise, validator)

        # Simpan array label ke file .npy terpisah
        label_save_path = LABELS_DIR / (npy_path.stem + "_labels.npy")
        np.save(str(label_save_path), per_frame_labels)

        # Hitung statistik label untuk laporan
        n_benar = int(np.sum(per_frame_labels == 0))
        n_salah = int(np.sum(per_frame_labels == 1))

        records.append({
            "file_path"    : str(npy_path),
            "labels_path"  : str(label_save_path),
            "exercise"     : exercise,
            "n_frames"     : int(tensor.shape[0]),
            "frames_benar" : n_benar,
            "frames_salah" : n_salah,
        })

    # Simpan manifest ke CSV
    df = pd.DataFrame(records)
    df.to_csv(MANIFEST_OUT, index=False)

    print(f"\n[SELESAI] Manifest v2 disimpan ke: {MANIFEST_OUT}")
    print(f"  Total sampel : {len(df)}")
    print(f"  Distribusi exercise:\n{df['exercise'].value_counts().to_string()}")
    total_frames = df["n_frames"].sum()
    total_benar  = df["frames_benar"].sum()
    total_salah  = df["frames_salah"].sum()
    print(f"\n  Total frame        : {total_frames}")
    print(f"  Frame Benar (0)    : {total_benar} ({100*total_benar/total_frames:.1f}%)")
    print(f"  Frame Salah (1)    : {total_salah} ({100*total_salah/total_frames:.1f}%)")


if __name__ == "__main__":
    main()
