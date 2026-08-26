# src/data/build_dataset.py
#
# Script untuk memproses banyak video sekaligus dari struktur folder yang terorganisir
# dan membangun dataset manifest (CSV) yang siap digunakan oleh PyTorch DataLoader.
#
# ── SISTEM AUTOMATED GROUND TRUTH LABELING ────────────────────────────────────
# Label ditetapkan secara OTOMATIS menggunakan BiomechanicalValidator, bukan
# berdasarkan nama sub-folder. Setiap tensor (64, 33, 3) yang dihasilkan akan
# dievaluasi oleh validator biomekanik:
#   - Validator mengembalikan True  → label = 0 (Benar)
#   - Validator mengembalikan False → label = 1 (Salah)
#
# Struktur folder input yang diharapkan:
#   data/raw/<NamaLatihan>/video.mp4
#   Contoh:
#     data/raw/Deadlift/deadlift_001.mp4
#     data/raw/Squat/squat_002.mp4
#     data/raw/BenchPress/bench_003.mp4
#
# Output:
#   data/processed/tensors/<NamaLatihan>_<nomor>.npy  → tensor (64, 33, 3)
#   data/processed/dataset_manifest.csv               → {file_path, label, exercise, reason}
#
# Peta Validator per Nama Latihan (case-insensitive substring match):
#   "squat"      → BiomechanicalValidator.validate_squat
#   "deadlift"   → BiomechanicalValidator.validate_deadlift
#   "benchpress" → BiomechanicalValidator.validate_benchpress
#   "bench"      → BiomechanicalValidator.validate_benchpress
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Pastikan src/ ada di path saat script dijalankan langsung ──────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.extract_pose import PoseExtractor
from data.preprocess import DataPreprocessor
from data.biomechanics_validator import BiomechanicalValidator

# ── Konfigurasi logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Inisialisasi validator biomekanik (satu instance, digunakan bersama) ───────
_validator = BiomechanicalValidator()

# ── Peta nama latihan (substring lowercase) → fungsi validator ────────────────
EXERCISE_VALIDATOR_MAP: dict[str, callable] = {
    "squat"      : _validator.validate_squat,
    "deadlift"   : _validator.validate_deadlift,
    "benchpress" : _validator.validate_benchpress,
    "bench"      : _validator.validate_benchpress,
}

# ── Konstanta label integer ───────────────────────────────────────────────────
LABEL_BENAR = 0  # Lulus validasi biomekanik → gerakan benar
LABEL_SALAH = 1  # Gagal validasi biomekanik → gerakan salah


def build_dataset(
    raw_root: str | Path,
    processed_root: str | Path,
    extractor: PoseExtractor | None = None,
    preprocessor: DataPreprocessor | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Memproses semua video .mp4 di dalam `raw_root` secara rekursif,
    mengekstrak pose, melakukan pra-pemrosesan, menentukan label secara otomatis
    menggunakan BiomechanicalValidator, lalu menyimpan tensor dan manifest CSV.

    Struktur folder yang diharapkan di `raw_root`:
        <raw_root>/<NamaLatihan>/*.mp4
        Contoh: data/raw/Squat/squat_001.mp4

    Label ditetapkan secara OTOMATIS berdasarkan hasil validator biomekanik:
        BiomechanicalValidator.validate_<exercise>(tensor) → (is_valid, reason)
        - is_valid = True  → label = 0 (Benar)
        - is_valid = False → label = 1 (Salah)

    Args:
        raw_root       : Direktori root data mentah, misal ``data/raw``.
        processed_root : Direktori root output, misal ``data/processed``.
        extractor      : Instance PoseExtractor. Jika None, dibuat dengan default.
        preprocessor   : Instance DataPreprocessor. Jika None, dibuat dengan default.
        overwrite      : Jika False, video yang sudah punya tensor .npy di-skip.
                         Label otomatis tetap dihitung ulang dari tensor yang ada.

    Returns:
        pd.DataFrame dengan kolom ``file_path``, ``label``, ``exercise``, ``reason``.
    """
    raw_root       = Path(raw_root)
    processed_root = Path(processed_root)
    tensor_dir     = processed_root / "tensors"
    manifest_path  = processed_root / "dataset_manifest.csv"

    # Buat direktori output jika belum ada
    tensor_dir.mkdir(parents=True, exist_ok=True)

    # Inisialisasi komponen jika tidak disediakan dari luar
    if extractor is None:
        extractor = PoseExtractor(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    if preprocessor is None:
        preprocessor = DataPreprocessor(
            visibility_threshold=0.3,
            nan_frame_ratio=0.30,
            max_interp_gap=5,
            median_kernel=3,
            target_frames=64,
        )

    manifest_rows: list[dict] = []  # Kumpulan baris untuk manifest CSV

    # Temukan semua sub-folder latihan (Deadlift, Squat, BenchPress, dst.)
    exercise_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])

    if not exercise_dirs:
        logger.warning(
            f"Tidak ada sub-folder latihan yang ditemukan di: {raw_root}. "
            "Pastikan struktur folder sudah benar."
        )
        return pd.DataFrame(columns=["file_path", "label", "exercise", "reason"])

    logger.info(f"Ditemukan {len(exercise_dirs)} jenis latihan: "
                f"{[d.name for d in exercise_dirs]}")

    # ── Loop melalui setiap jenis latihan ──────────────────────────────────────
    for exercise_dir in exercise_dirs:
        exercise_name = exercise_dir.name

        # ── Tentukan fungsi validator berdasarkan nama latihan ─────────────────
        # Pencocokan dilakukan secara case-insensitive dengan substring matching
        validate_fn = None
        for key, fn in EXERCISE_VALIDATOR_MAP.items():
            if key in exercise_name.lower():
                validate_fn = fn
                break

        if validate_fn is None:
            logger.warning(
                f"[{exercise_name}] Tidak ada validator yang cocok. "
                f"Nama latihan harus mengandung salah satu dari: "
                f"{list(EXERCISE_VALIDATOR_MAP.keys())}. Folder ini di-skip."
            )
            continue

        # Kumpulkan semua file video .mp4 langsung di dalam folder latihan
        # (TIDAK ada sub-folder Benar/Salah — struktur baru)
        video_files = sorted(exercise_dir.glob("*.mp4"))

        if not video_files:
            logger.warning(
                f"[{exercise_name}] Tidak ada file .mp4 di: {exercise_dir}. Di-skip."
            )
            continue

        logger.info(
            f"\n[{exercise_name}] Memproses {len(video_files)} video "
            f"dengan auto-labeling biomekanik..."
        )

        # ── Loop melalui setiap file video ─────────────────────────────────────
        for video_idx, video_path in enumerate(video_files, start=1):
            # Buat nama file output yang unik dan deskriptif
            tensor_filename = f"{exercise_name}_{video_idx:03d}.npy"
            tensor_path     = tensor_dir / tensor_filename

            # ── Lewati ekstraksi jika sudah diproses dan overwrite=False ────────
            if tensor_path.exists() and not overwrite:
                logger.info(
                    f"  [{video_idx:03d}/{len(video_files):03d}] "
                    f"Tensor sudah ada, langsung re-label: {tensor_filename}"
                )
            else:
                logger.info(
                    f"  [{video_idx:03d}/{len(video_files):03d}] "
                    f"Memproses: {video_path.name} ..."
                )

                try:
                    # ── Langkah 1: Ekstraksi pose menggunakan MediaPipe BlazePose ──
                    raw_npy_path = tensor_dir / f"_tmp_{tensor_filename}"
                    raw_array = extractor.extract_video(
                        video_path=str(video_path),
                        output_npy_path=str(raw_npy_path),
                        output_video_path=None,
                    )

                    # Pastikan ada frame yang berhasil diekstraksi
                    if raw_array.shape[0] < 2:
                        logger.warning(
                            f"  Video terlalu pendek atau tidak ada pose terdeteksi: "
                            f"{video_path.name}. Di-skip."
                        )
                        if raw_npy_path.exists():
                            raw_npy_path.unlink()
                        continue

                    # ── Langkah 2: Pra-pemrosesan (clean → smooth → normalize → resample) ──
                    preprocessor.process(
                        npy_file_path=str(raw_npy_path),
                        output_npy_path=str(tensor_path),
                    )

                    # Hapus file .npy sementara setelah preprocessing selesai
                    if raw_npy_path.exists():
                        raw_npy_path.unlink()

                    # Verifikasi shape tensor akhir
                    final_tensor = np.load(tensor_path)
                    if final_tensor.shape != (64, 33, 3):
                        logger.error(
                            f"  Shape tensor tidak sesuai untuk {tensor_filename}: "
                            f"{final_tensor.shape} (diharapkan (64, 33, 3)). Di-skip."
                        )
                        tensor_path.unlink()
                        continue

                except Exception as e:
                    logger.error(
                        f"  Gagal memproses {video_path.name}: {e}. Di-skip.",
                        exc_info=True,
                    )
                    # Bersihkan file output yang mungkin terbuat parsial
                    for tmp_path in [tensor_path, raw_npy_path]:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    continue

            # ── Langkah 3: AUTO-LABELING menggunakan BiomechanicalValidator ────
            # Tensor sudah tersedia di disk (baik baru diproses maupun di-skip)
            try:
                tensor_data = np.load(tensor_path)  # (64, 33, 3)
                is_valid, reason = validate_fn(tensor_data)
            except Exception as e:
                logger.error(
                    f"  Gagal menjalankan validator untuk {tensor_filename}: {e}. Di-skip.",
                    exc_info=True,
                )
                continue

            # Tentukan label berdasarkan hasil validator
            auto_label = LABEL_BENAR if is_valid else LABEL_SALAH
            label_str  = "Benar (0)" if is_valid else "Salah (1)"

            logger.info(
                f"  [{video_idx:03d}/{len(video_files):03d}] "
                f"→ {tensor_filename}  |  label={auto_label} [{label_str}]  |  {reason[:80]}"
            )

            # ── Tambahkan ke manifest ──────────────────────────────────────────
            manifest_rows.append({
                "file_path": str(tensor_path),
                "label"    : auto_label,
                "exercise" : exercise_name,
                "reason"   : reason,
            })

    # ── Buat dan simpan manifest CSV ───────────────────────────────────────────
    manifest_df = pd.DataFrame(
        manifest_rows, columns=["file_path", "label", "exercise", "reason"]
    )

    # Acak urutan baris agar distribusi kelas lebih merata saat dipakai DataLoader
    manifest_df = manifest_df.sample(frac=1, random_state=42).reset_index(drop=True)

    manifest_df.to_csv(manifest_path, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset manifest disimpan ke : {manifest_path}")
    logger.info(f"Total sampel                 : {len(manifest_df)}")
    label_names = {LABEL_BENAR: "Benar", LABEL_SALAH: "Salah"}
    for label_val, label_name in label_names.items():
        count = (manifest_df["label"] == label_val).sum()
        logger.info(f"  Kelas {label_val} ({label_name:5s})           : {count} sampel")
    logger.info(f"{'='*60}")

    return manifest_df


# ── Entry point: jalankan langsung dari terminal ───────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build dataset dengan Auto-Labeling: proses semua video → "
            "tensor .npy → label otomatis via BiomechanicalValidator → manifest CSV."
        )
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "raw"),
        help="Path ke direktori data mentah (default: data/raw/)",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "processed"),
        help="Path ke direktori output (default: data/processed/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Timpa tensor .npy yang sudah ada dan proses ulang dari video. "
            "Jika tidak di-set, tensor yang sudah ada di-skip namun label "
            "tetap dihitung ulang secara otomatis."
        ),
    )
    args = parser.parse_args()

    build_dataset(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        overwrite=args.overwrite,
    )
