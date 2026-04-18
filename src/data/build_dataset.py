# src/data/build_dataset.py
#
# Script untuk memproses banyak video sekaligus dari struktur folder yang terorganisir
# dan membangun dataset manifest (CSV) yang siap digunakan oleh PyTorch DataLoader.
#
# Struktur folder input yang diharapkan:
#   data/raw/<NamaLatihan>/<Kelas>/video.mp4
#   Contoh:
#     data/raw/Deadlift/Benar/deadlift_001.mp4
#     data/raw/Deadlift/Salah/deadlift_error_001.mp4
#     data/raw/Squat/Benar/squat_001.mp4
#     data/raw/Squat/Salah/squat_error_001.mp4
#
# Output:
#   data/processed/tensors/<NamaLatihan>_<Kelas>_<nomor>.npy  → tensor (64, 33, 3)
#   data/processed/dataset_manifest.csv                       → {file_path, label}
#
# Peta Label (CLASS_LABEL_MAP):
#   "Benar" → 0
#   "Salah" → 1

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

# ── Konfigurasi logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Peta nama folder kelas → indeks label integer ─────────────────────────────
CLASS_LABEL_MAP: dict[str, int] = {
    "Benar": 0,
    "Salah": 1,
}


def build_dataset(
    raw_root: str | Path,
    processed_root: str | Path,
    extractor: PoseExtractor | None = None,
    preprocessor: DataPreprocessor | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Memproses semua video .mp4 di dalam `raw_root` secara rekursif,
    mengekstrak pose, melakukan pra-pemrosesan, dan menyimpan tensor
    serta file manifest CSV.

    Struktur folder yang diharapkan di `raw_root`:
        <raw_root>/<NamaLatihan>/<Kelas>/*.mp4

    Args:
        raw_root       : Direktori root data mentah, misal ``data/raw``.
        processed_root : Direktori root output, misal ``data/processed``.
        extractor      : Instance PoseExtractor. Jika None, dibuat dengan default.
        preprocessor   : Instance DataPreprocessor. Jika None, dibuat dengan default.
        overwrite      : Jika False, video yang sudah punya tensor .npy di-skip.

    Returns:
        pd.DataFrame dengan kolom ``file_path`` dan ``label``.
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
        return pd.DataFrame(columns=["file_path", "label"])

    logger.info(f"Ditemukan {len(exercise_dirs)} jenis latihan: "
                f"{[d.name for d in exercise_dirs]}")

    # ── Loop melalui setiap jenis latihan ──────────────────────────────────────
    for exercise_dir in exercise_dirs:
        exercise_name = exercise_dir.name

        # Temukan semua sub-folder kelas (Benar, Salah)
        class_dirs = sorted([d for d in exercise_dir.iterdir() if d.is_dir()])

        for class_dir in class_dirs:
            class_name = class_dir.name

            # Validasi: nama folder kelas harus ada di CLASS_LABEL_MAP
            if class_name not in CLASS_LABEL_MAP:
                logger.warning(
                    f"Folder kelas '{class_name}' tidak dikenali "
                    f"(bukan salah satu dari {list(CLASS_LABEL_MAP.keys())}). Di-skip."
                )
                continue

            label = CLASS_LABEL_MAP[class_name]

            # Kumpulkan semua file video .mp4 di folder ini
            video_files = sorted(class_dir.glob("*.mp4"))

            if not video_files:
                logger.warning(
                    f"Tidak ada file .mp4 di: {class_dir}. Di-skip."
                )
                continue

            logger.info(
                f"\n[{exercise_name}/{class_name}] Memproses {len(video_files)} video "
                f"(label={label})..."
            )

            # ── Loop melalui setiap file video ─────────────────────────────────
            for video_idx, video_path in enumerate(video_files, start=1):
                # Buat nama file output yang unik dan deskriptif
                tensor_filename = f"{exercise_name}_{class_name}_{video_idx:03d}.npy"
                tensor_path     = tensor_dir / tensor_filename

                # ── Lewati jika sudah diproses dan overwrite=False ──────────────
                if tensor_path.exists() and not overwrite:
                    logger.info(
                        f"  [{video_idx:03d}/{len(video_files):03d}] "
                        f"Di-skip (sudah ada): {tensor_filename}"
                    )
                    manifest_rows.append({
                        "file_path": str(tensor_path),
                        "label"    : label,
                    })
                    continue

                logger.info(
                    f"  [{video_idx:03d}/{len(video_files):03d}] "
                    f"Memproses: {video_path.name} ..."
                )

                try:
                    # ── Langkah 1: Ekstraksi pose menggunakan MediaPipe BlazePose ──
                    # Simpan hasil mentah ke file .npy sementara
                    raw_npy_path = tensor_dir / f"_tmp_{tensor_filename}"
                    raw_array = extractor.extract_video(
                        video_path=str(video_path),
                        output_npy_path=str(raw_npy_path),
                        output_video_path=None,  # Tidak perlu video visualisasi
                    )

                    # Pastikan ada frame yang berhasil diekstraksi
                    if raw_array.shape[0] < 2:
                        logger.warning(
                            f"  Video terlalu pendek atau tidak ada pose terdeteksi: "
                            f"{video_path.name}. Di-skip."
                        )
                        if raw_npy_path.exists():
                            raw_npy_path.unlink()  # Hapus file sementara
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

                    # ── Tambahkan ke manifest ───────────────────────────────────
                    manifest_rows.append({
                        "file_path": str(tensor_path),
                        "label"    : label,
                    })

                    logger.info(
                        f"  [{video_idx:03d}/{len(video_files):03d}] "
                        f"Selesai → {tensor_filename}  shape={final_tensor.shape}"
                    )

                except Exception as e:
                    logger.error(
                        f"  Gagal memproses {video_path.name}: {e}. Di-skip.",
                        exc_info=True,
                    )
                    # Bersihkan file output yang mungkin terbuat parsial
                    for tmp_path in [tensor_path, raw_npy_path]:
                        if tmp_path.exists():
                            tmp_path.unlink()

    # ── Buat dan simpan manifest CSV ───────────────────────────────────────────
    manifest_df = pd.DataFrame(manifest_rows, columns=["file_path", "label"])

    # Acak urutan baris agar distribusi kelas lebih merata saat dipakai DataLoader
    manifest_df = manifest_df.sample(frac=1, random_state=42).reset_index(drop=True)

    manifest_df.to_csv(manifest_path, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset manifest disimpan ke : {manifest_path}")
    logger.info(f"Total sampel                 : {len(manifest_df)}")
    for label_val, label_name in {v: k for k, v in CLASS_LABEL_MAP.items()}.items():
        count = (manifest_df["label"] == label_val).sum()
        logger.info(f"  Kelas {label_val} ({label_name:5s})           : {count} sampel")
    logger.info(f"{'='*60}")

    return manifest_df


# ── Entry point: jalankan langsung dari terminal ───────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build dataset: proses semua video → tensor .npy + manifest CSV."
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
        help="Timpa tensor .npy yang sudah ada. Jika tidak di-set, video yang sudah diproses di-skip.",
    )
    args = parser.parse_args()

    build_dataset(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        overwrite=args.overwrite,
    )
