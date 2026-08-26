# src/data/v2/dataset.py
#
# PyTorch Dataset untuk pelatihan per-frame (v2).
# Setiap sampel mengembalikan:
#   - tensor_input  : (64, 33, 3) — koordinat pose spasio-temporal
#   - tensor_labels : (64,)       — label per-frame (0=Benar, 1=Salah)
#
# Dataset membaca daftar file dari manifest_v2.csv yang dihasilkan oleh
# src/data/v2/build_manifest.py.

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class PerFrameDataset(Dataset):
    """
    Dataset PyTorch untuk klasifikasi kualitas gerakan per-frame.

    Setiap sampel terdiri dari:
        - tensor_input  (torch.FloatTensor): shape (64, 33, 3)
          → 64 frame, 33 landmark BlazePose, 3 koordinat xyz ternormalisasi.
        - tensor_labels (torch.LongTensor) : shape (64,)
          → Label 0 (Benar) atau 1 (Salah) untuk setiap frame.

    Args:
        csv_file (str | Path): Path ke manifest_v2.csv.

    Raises:
        FileNotFoundError : Jika manifest_v2.csv tidak ditemukan.
        ValueError        : Jika kolom wajib tidak ada di CSV.
    """

    def __init__(self, csv_file: str | Path):
        csv_file = Path(csv_file)

        # Validasi keberadaan manifest
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Manifest v2 tidak ditemukan: '{csv_file}'\n"
                "Jalankan terlebih dahulu: python src/data/v2/build_manifest.py"
            )

        self.manifest = pd.read_csv(csv_file)

        # Validasi kolom wajib
        required = {"file_path", "labels_path"}
        missing  = required - set(self.manifest.columns)
        if missing:
            raise ValueError(
                f"Kolom wajib tidak ditemukan di CSV: {missing}. "
                f"Kolom tersedia: {list(self.manifest.columns)}"
            )

        self.manifest = self.manifest.reset_index(drop=True)

    def __len__(self) -> int:
        """Jumlah total sampel (video) dalam dataset."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Mengambil satu sampel berdasarkan indeks.

        Args:
            idx (int): Indeks sampel (0-based).

        Returns:
            tuple:
                - tensor_input  (torch.FloatTensor): (64, 33, 3)
                - tensor_labels (torch.LongTensor) : (64,)

        Raises:
            FileNotFoundError: Jika file .npy tidak ditemukan di path yang tercatat.
        """
        row         = self.manifest.iloc[idx]
        npy_path    = row["file_path"]
        labels_path = row["labels_path"]

        # Validasi keberadaan kedua file
        for p in (npy_path, labels_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"File tidak ditemukan: '{p}'\n"
                    "Jalankan ulang src/data/v2/build_manifest.py."
                )

        # Muat tensor pose (64, 33, 3) float32
        pose_array    = np.load(npy_path).astype(np.float32)
        tensor_input  = torch.from_numpy(pose_array)          # (64, 33, 3)

        # Muat array label per-frame (64,) int8 → konversi ke LongTensor
        labels_array  = np.load(labels_path).astype(np.int64)
        tensor_labels = torch.from_numpy(labels_array)        # (64,)

        return tensor_input, tensor_labels

    def __repr__(self) -> str:
        return (
            f"PerFrameDataset(\n"
            f"  total_video = {len(self)}\n"
            f"  kolom CSV   = {list(self.manifest.columns)}\n"
            f")"
        )


def create_dataloaders_v2(
    csv_file: str | Path,
    batch_size: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Membuat DataLoader train/val/test untuk dataset per-frame v2.

    Args:
        csv_file    : Path ke manifest_v2.csv.
        batch_size  : Jumlah sampel per batch. Default 16.
        train_ratio : Proporsi data train. Default 70%.
        val_ratio   : Proporsi data validasi. Default 15%.
        num_workers : Worker paralel. Set 0 di Windows.
        random_seed : Seed reproduktibilitas.

    Returns:
        tuple: (train_loader, val_loader, test_loader)

    Raises:
        ValueError: Jika train_ratio + val_ratio >= 1.0.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) harus < 1.0."
        )

    full_dataset = PerFrameDataset(csv_file=csv_file)

    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    n_test  = n_total - n_train - n_val

    # Pembagian acak yang dapat direproduksi dengan seed tetap
    generator = torch.Generator().manual_seed(random_seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # pin_memory=True mempercepat transfer CPU→GPU saat CUDA tersedia
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,           # Acak urutan tiap epoch (hanya training)
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=True,         # Buang batch terakhir jika ukurannya kurang dari batch_size
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )

    print(f"Dataset split selesai (seed={random_seed}):")
    print(f"  Train  : {len(train_set):4d} sampel → {len(train_loader)} batch")
    print(f"  Val    : {len(val_set):4d} sampel → {len(val_loader)} batch")
    print(f"  Test   : {len(test_set):4d} sampel → {len(test_loader)} batch")

    return train_loader, val_loader, test_loader
