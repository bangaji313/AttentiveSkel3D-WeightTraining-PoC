# src/data/dataset.py
#
# Implementasi PyTorch Dataset dan utilitas DataLoader untuk proyek AttentiveSkel-3D.
#
# Class:
#   WeightTrainingDataset  — membaca tensor .npy berdasarkan manifest CSV
#
# Fungsi utilitas:
#   create_dataloaders     — membuat DataLoader train/val/test dari satu manifest

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class WeightTrainingDataset(Dataset):
    """
    Dataset PyTorch untuk data pose skeleton latihan beban.

    Setiap sampel terdiri dari:
        - Tensor pose  : torch.FloatTensor berukuran (64, 33, 3)
          → 64 frame, 33 landmark BlazePose, 3 koordinat (x, y, z) ternormalisasi
        - Label integer: 0 = Gerakan Benar, 1 = Gerakan Salah

    Dataset membaca daftar file dari sebuah file CSV (dataset_manifest.csv)
    yang dihasilkan oleh `build_dataset.py`. Format CSV:
        file_path,label
        /path/to/tensor.npy,0
        /path/to/tensor.npy,1
        ...

    Args:
        csv_file  (str | Path): Path ke file dataset_manifest.csv.
        transform (callable, optional): Transformasi opsional yang diterapkan
                                        pada tensor sebelum dikembalikan.
                                        Berguna untuk augmentasi data.

    Raises:
        FileNotFoundError: Jika csv_file tidak ditemukan.
        ValueError        : Jika CSV tidak memiliki kolom 'file_path' atau 'label'.
    """

    def __init__(self, csv_file: str | Path, transform=None):
        csv_file = Path(csv_file)

        # Validasi keberadaan file manifest
        if not csv_file.exists():
            raise FileNotFoundError(
                f"File manifest CSV tidak ditemukan: '{csv_file}'\n"
                "Jalankan terlebih dahulu: python src/data/build_dataset.py"
            )

        # Muat manifest sebagai DataFrame
        self.manifest = pd.read_csv(csv_file)

        # Validasi kolom yang diperlukan
        required_columns = {"file_path", "label"}
        missing = required_columns - set(self.manifest.columns)
        if missing:
            raise ValueError(
                f"Kolom berikut tidak ditemukan di CSV: {missing}. "
                f"Kolom yang tersedia: {list(self.manifest.columns)}"
            )

        # Reset index untuk memastikan pengindeksan berurutan
        self.manifest = self.manifest.reset_index(drop=True)

        # Transformasi opsional (untuk augmentasi data saat pelatihan)
        self.transform = transform

        # Peta label → nama kelas (untuk kemudahan inspeksi)
        self.label_to_class = {0: "Benar", 1: "Salah"}

    def __len__(self) -> int:
        """Mengembalikan jumlah total sampel dalam dataset."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.FloatTensor, int]:
        """
        Mengambil satu sampel dari dataset berdasarkan indeks.

        Args:
            idx (int): Indeks sampel (0-based).

        Returns:
            tuple: (tensor_data, label)
                - tensor_data (torch.FloatTensor): Tensor pose (64, 33, 3).
                - label (int): Kelas gerakan (0=Benar, 1=Salah).

        Raises:
            FileNotFoundError: Jika file .npy untuk sampel ini tidak ditemukan.
        """
        # Ambil baris ke-idx dari manifest
        row = self.manifest.iloc[idx]
        npy_path = row["file_path"]
        label    = int(row["label"])

        # Validasi keberadaan file .npy
        if not os.path.exists(npy_path):
            raise FileNotFoundError(
                f"File tensor tidak ditemukan: '{npy_path}'\n"
                "Mungkin file telah dihapus atau dipindahkan. "
                "Jalankan ulang build_dataset.py."
            )

        # Muat array NumPy dari disk dan konversi ke FloatTensor PyTorch
        # np.load() mengembalikan (64, 33, 3) float32
        pose_array  = np.load(npy_path)
        tensor_data = torch.from_numpy(pose_array).float()  # (64, 33, 3)

        # Terapkan transformasi jika ada (misalnya augmentasi Gaussian noise)
        if self.transform is not None:
            tensor_data = self.transform(tensor_data)

        return tensor_data, label

    def get_class_distribution(self) -> dict[str, int]:
        """
        Mengembalikan distribusi jumlah sampel per kelas.

        Returns:
            dict: {'Benar': jumlah, 'Salah': jumlah}
        """
        distribution = {}
        for label_val, class_name in self.label_to_class.items():
            count = (self.manifest["label"] == label_val).sum()
            distribution[class_name] = int(count)
        return distribution

    def __repr__(self) -> str:
        dist = self.get_class_distribution()
        return (
            f"WeightTrainingDataset(\n"
            f"  total_samples = {len(self)}\n"
            f"  Benar (label=0) = {dist.get('Benar', 0)}\n"
            f"  Salah (label=1) = {dist.get('Salah', 0)}\n"
            f")"
        )


def create_dataloaders(
    csv_file: str | Path,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    random_seed: int = 42,
    train_transform=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Membuat DataLoader untuk split train / validation / test dari satu manifest CSV.

    Pembagian dilakukan secara acak berdasarkan `random_seed` untuk
    reproduktibilitas eksperimen.

    Args:
        csv_file       : Path ke dataset_manifest.csv.
        batch_size     : Jumlah sampel per batch.
        train_ratio    : Proporsi data untuk pelatihan (default 70%).
        val_ratio      : Proporsi data untuk validasi (default 15%).
                         Sisanya (100% - train - val) digunakan untuk pengujian.
        num_workers    : Jumlah worker proses untuk memuat data paralel.
                         Set 0 di Windows untuk menghindari masalah multiprocessing.
        random_seed    : Seed untuk reproduktibilitas pembagian split.
        train_transform: Transformasi augmentasi yang hanya diterapkan pada data train.

    Returns:
        tuple: (train_loader, val_loader, test_loader)

    Raises:
        ValueError: Jika train_ratio + val_ratio >= 1.0.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) harus < 1.0 "
            "agar ada data untuk split test."
        )

    # Buat dataset lengkap (tanpa augmentasi untuk val & test)
    full_dataset = WeightTrainingDataset(csv_file=csv_file)

    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    n_test  = n_total - n_train - n_val  # Sisa untuk test

    # Pembagian acak yang dapat direproduksi
    generator = torch.Generator().manual_seed(random_seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Terapkan transformasi augmentasi hanya pada training set
    if train_transform is not None:
        train_set.dataset.transform = train_transform

    # Buat DataLoader untuk masing-masing split
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,           # Acak urutan tiap epoch hanya pada data train
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Percepat transfer ke GPU jika ada
        drop_last=True,         # Buang batch terakhir jika kurang dari batch_size
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,          # Tidak perlu diacak untuk validasi
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Dataset split selesai (seed={random_seed}):")
    print(f"  Train  : {len(train_set):4d} sampel → {len(train_loader)} batch")
    print(f"  Val    : {len(val_set):4d} sampel → {len(val_loader)} batch")
    print(f"  Test   : {len(test_set):4d} sampel → {len(test_loader)} batch")

    return train_loader, val_loader, test_loader
