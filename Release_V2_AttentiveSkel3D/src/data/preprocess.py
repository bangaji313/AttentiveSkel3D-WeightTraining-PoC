# src/data/preprocess.py
#
# Modul pra-pemrosesan data pose skeleton untuk proyek AttentiveSkel-3D.
#
# Pipeline yang diimplementasikan (sesuai urutan dalam metode `process`):
#   1. filter_and_clean  — Hapus landmark dengan confidence rendah, interpolasi gap
#   2. smooth_data       — Median filter temporal untuk mengurangi jitter
#   3. spatial_normalize — Translasi ke mid-hip + scaling dengan panjang torso
#   4. temporal_resample — Resample jumlah frame menjadi tepat 64
#
# Format data:
#   Input  : (T, 33, 4)  → T frame, 33 landmark, [x, y, z, visibility]
#   Output : (64, 33, 3) → 64 frame, 33 landmark, [x, y, z] ternormalisasi

import os
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d


class DataPreprocessor:
    """
    Melakukan serangkaian pra-pemrosesan pada array pose skeleton 3D.

    Pipeline lengkap:
        filter_and_clean → smooth_data → spatial_normalize → temporal_resample

    Atribut:
        visibility_threshold (float): Ambang batas confidence; di bawahnya
                                      koordinat dianggap tidak valid (NaN).
        nan_frame_ratio (float)     : Batas rasio landmark NaN per frame;
                                      frame dengan rasio lebih tinggi dihapus.
        max_interp_gap (int)        : Jumlah frame gap maksimum yang boleh
                                      diisi dengan interpolasi linear.
        median_kernel (int)         : Ukuran kernel median filter temporal.
        target_frames (int)         : Jumlah frame target setelah resampling.
    """

    def __init__(
        self,
        visibility_threshold: float = 0.3,
        nan_frame_ratio: float = 0.30,
        max_interp_gap: int = 5,
        median_kernel: int = 3,
        target_frames: int = 64,
    ):
        self.visibility_threshold = visibility_threshold
        self.nan_frame_ratio = nan_frame_ratio
        self.max_interp_gap = max_interp_gap
        self.median_kernel = median_kernel
        self.target_frames = target_frames

    # ------------------------------------------------------------------
    # 2.2  FILTER & CLEAN
    # ------------------------------------------------------------------
    def filter_and_clean(self, data: np.ndarray) -> np.ndarray:
        """
        Membersihkan data pose dengan langkah:
          (a) Tandai koordinat (x,y,z) sebagai NaN jika visibility < threshold.
          (b) Hapus frame yang memiliki lebih dari `nan_frame_ratio` landmark NaN.
          (c) Interpolasi linear untuk mengisi gap NaN (maks. `max_interp_gap` frame).

        Args:
            data: Array bentuk (T, 33, 4) → [x, y, z, visibility].

        Returns:
            Array bentuk (T', 33, 4) setelah pembersihan, di mana T' ≤ T.
        """
        data = data.copy().astype(np.float32)
        T, N_landmarks, _ = data.shape  # T frame, 33 landmark

        # (a) Tandai koordinat (x, y, z) dengan NaN jika visibility di bawah threshold
        low_confidence_mask = data[:, :, 3] < self.visibility_threshold  # (T, 33)
        data[low_confidence_mask, :3] = np.nan

        # (b) Hitung rasio NaN per frame dan hapus frame yang terlalu buruk
        # Suatu landmark dianggap NaN jika minimal satu koordinat-nya NaN
        nan_per_frame = np.isnan(data[:, :, 0]).sum(axis=1)  # (T,)
        nan_ratio_per_frame = nan_per_frame / N_landmarks     # (T,)

        valid_frame_mask = nan_ratio_per_frame <= self.nan_frame_ratio
        data = data[valid_frame_mask]                         # (T', 33, 4)

        n_removed = T - data.shape[0]
        print(f"  [filter_and_clean] Frame dihapus (>{self.nan_frame_ratio*100:.0f}% NaN): {n_removed}")
        print(f"  [filter_and_clean] Frame tersisa: {data.shape[0]}")

        # (c) Interpolasi linear per landmark per koordinat untuk mengisi NaN
        # Gunakan Pandas interpolate dengan limit agar tidak mengisi gap panjang
        T_clean = data.shape[0]
        for landmark_idx in range(N_landmarks):
            for coord_idx in range(3):  # hanya x, y, z (bukan visibility)
                series = pd.Series(data[:, landmark_idx, coord_idx])

                # Interpolasi linear, hanya mengisi gap <= max_interp_gap frame
                series_interp = series.interpolate(
                    method="linear",
                    limit=self.max_interp_gap,
                    limit_direction="both",
                )
                data[:, landmark_idx, coord_idx] = series_interp.to_numpy()

        # Sisa NaN yang tidak dapat diinterpolasi (gap terlalu panjang) → isi dengan 0
        remaining_nan = np.isnan(data[:, :, :3]).sum()
        if remaining_nan > 0:
            warnings.warn(
                f"  [filter_and_clean] {remaining_nan} nilai NaN tersisa setelah interpolasi "
                f"(gap > {self.max_interp_gap} frame). Diganti dengan 0.",
                stacklevel=2,
            )
            data[:, :, :3] = np.nan_to_num(data[:, :, :3], nan=0.0)

        return data

    # ------------------------------------------------------------------
    # 2.3  SMOOTH DATA
    # ------------------------------------------------------------------
    def smooth_data(self, data: np.ndarray) -> np.ndarray:
        """
        Terapkan Median Filter pada dimensi waktu (frame) untuk setiap
        koordinat sendi guna mengurangi noise/jitter.

        Median filter hanya diterapkan pada dimensi frame (axis=0) dengan
        kernel_size = `median_kernel`. Kolom visibility tidak difilter.

        Args:
            data: Array bentuk (T, 33, 4).

        Returns:
            Array bentuk (T, 33, 4) yang telah dihaluskan.
        """
        data = data.copy()

        # Terapkan median filter pada koordinat x, y, z saja (indeks 0–2)
        # scipy.ndimage.median_filter bekerja pada seluruh array;
        # atur size agar filter hanya aktif pada dimensi frame (axis 0)
        kernel_shape = (self.median_kernel, 1, 1)  # filter hanya di dimensi waktu

        data[:, :, :3] = median_filter(data[:, :, :3], size=kernel_shape)

        print(f"  [smooth_data] Median filter diterapkan (kernel={self.median_kernel}) → shape: {data.shape}")
        return data

    # ------------------------------------------------------------------
    # 2.4  SPATIAL NORMALIZE
    # ------------------------------------------------------------------
    def spatial_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalisasi spasial per-frame:
          (a) Translasi: geser semua koordinat sehingga mid-hip menjadi (0,0,0).
          (b) Scaling  : bagi semua koordinat dengan panjang torso (mid-hip → mid-shoulder).
          (c) Buang kolom visibility; output menjadi (T, 33, 3).

        Landmark referensi MediaPipe BlazePose:
          - Left Hip  = 23, Right Hip  = 24  → mid-hip
          - Left Shoulder = 11, Right Shoulder = 12 → mid-shoulder

        Args:
            data: Array bentuk (T, 33, 4).

        Returns:
            Array bentuk (T, 33, 3) yang telah dinormalisasi.
        """
        data = data.copy()
        T = data.shape[0]

        # Indeks landmark referensi (MediaPipe BlazePose)
        IDX_LEFT_HIP       = 23
        IDX_RIGHT_HIP      = 24
        IDX_LEFT_SHOULDER  = 11
        IDX_RIGHT_SHOULDER = 12

        # --- (a) Hitung mid-hip per frame: (T, 3) ---
        mid_hip = (data[:, IDX_LEFT_HIP, :3] + data[:, IDX_RIGHT_HIP, :3]) / 2.0

        # --- (b) Translasi: kurangi semua landmark dengan mid-hip ---
        # Broadcast: (T, 33, 3) - (T, 1, 3)
        coords_xyz = data[:, :, :3] - mid_hip[:, np.newaxis, :]

        # --- (c) Hitung panjang torso per frame sebagai faktor scaling ---
        mid_shoulder = (
            data[:, IDX_LEFT_SHOULDER, :3] + data[:, IDX_RIGHT_SHOULDER, :3]
        ) / 2.0

        # Panjang torso = jarak Euclidean dari mid-hip ke mid-shoulder: (T,)
        torso_length = np.linalg.norm(mid_shoulder - mid_hip, axis=1)

        # Hindari pembagian dengan nol jika torso_length sangat kecil
        torso_length = np.where(torso_length < 1e-6, 1.0, torso_length)

        # --- (d) Scaling: bagi koordinat dengan panjang torso ---
        # Broadcast: (T, 33, 3) / (T, 1, 1)
        coords_xyz = coords_xyz / torso_length[:, np.newaxis, np.newaxis]

        print(f"  [spatial_normalize] Translasi & scaling selesai → shape: {coords_xyz.shape}")
        print(f"  [spatial_normalize] Rata-rata panjang torso: {torso_length.mean():.4f}")

        return coords_xyz.astype(np.float32)  # (T, 33, 3)

    # ------------------------------------------------------------------
    # 2.5  TEMPORAL RESAMPLE
    # ------------------------------------------------------------------
    def temporal_resample(
        self,
        data: np.ndarray,
        target_frames: int = None,
    ) -> np.ndarray:
        """
        Resample jumlah frame secara temporal menjadi `target_frames` yang tepat
        menggunakan interpolasi linear.

        Args:
            data         : Array bentuk (T, 33, 3).
            target_frames: Jumlah frame target. Jika None, gunakan self.target_frames.

        Returns:
            Array bentuk (target_frames, 33, 3).
        """
        if target_frames is None:
            target_frames = self.target_frames

        T_src = data.shape[0]

        # Jika jumlah frame sudah tepat, tidak perlu resampling
        if T_src == target_frames:
            print(f"  [temporal_resample] Jumlah frame sudah {target_frames}, tidak perlu resampling.")
            return data.copy()

        # Buat sumbu waktu asli dan target (dinormalisasi 0–1)
        t_src    = np.linspace(0.0, 1.0, T_src)
        t_target = np.linspace(0.0, 1.0, target_frames)

        # Reshape ke (T, 33*3) untuk memudahkan interpolasi vektor
        N_landmarks, N_coords = data.shape[1], data.shape[2]
        data_flat = data.reshape(T_src, -1)  # (T, 99)

        # Buat fungsi interpolasi linear dan terapkan ke sumbu waktu target
        interp_func   = interp1d(t_src, data_flat, kind="linear", axis=0)
        resampled_flat = interp_func(t_target)  # (target_frames, 99)

        # Kembalikan ke bentuk (target_frames, 33, 3)
        resampled = resampled_flat.reshape(target_frames, N_landmarks, N_coords)

        print(f"  [temporal_resample] {T_src} frame → {target_frames} frame selesai → shape: {resampled.shape}")
        return resampled.astype(np.float32)

    # ------------------------------------------------------------------
    # PIPELINE UTAMA
    # ------------------------------------------------------------------
    def process(self, npy_file_path: str, output_npy_path: str) -> np.ndarray:
        """
        Metode utama pipeline pra-pemrosesan.

        Urutan proses:
            1. Baca file .npy  → (T, 33, 4)
            2. filter_and_clean → (T', 33, 4)
            3. smooth_data      → (T', 33, 4)
            4. spatial_normalize → (T', 33, 3)
            5. temporal_resample → (64, 33, 3)
            6. Simpan hasil ke disk.

        Args:
            npy_file_path  : Path ke file .npy input hasil ekstraksi pose.
            output_npy_path: Path tujuan untuk menyimpan tensor akhir.

        Returns:
            np.ndarray: Tensor akhir dengan bentuk (64, 33, 3).

        Raises:
            FileNotFoundError: Jika npy_file_path tidak ditemukan.
            ValueError        : Jika data terlalu pendek untuk diproses.
        """
        # --- Validasi file input ---
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"File .npy tidak ditemukan: '{npy_file_path}'")

        # --- Langkah 1: Baca file .npy ---
        data = np.load(npy_file_path)
        print(f"[PROSES] File dimuat: {npy_file_path}")
        print(f"         Shape awal: {data.shape}")

        if data.shape[0] < 2:
            raise ValueError(
                f"Data terlalu pendek ({data.shape[0]} frame). "
                "Minimal 2 frame diperlukan untuk interpolasi."
            )

        # --- Langkah 2: Filter & Clean ---
        print("\n[Step 2.2] Filter & Clean ...")
        data = self.filter_and_clean(data)

        # --- Langkah 3: Smooth ---
        print("\n[Step 2.3] Smooth (Median Filter) ...")
        data = self.smooth_data(data)

        # --- Langkah 4: Spatial Normalize ---
        print("\n[Step 2.4] Spatial Normalize ...")
        data = self.spatial_normalize(data)   # output: (T', 33, 3)

        # --- Langkah 5: Temporal Resample ---
        print("\n[Step 2.5] Temporal Resample ...")
        data = self.temporal_resample(data, target_frames=self.target_frames)  # (64, 33, 3)

        # --- Langkah 6: Simpan ke disk ---
        os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
        np.save(output_npy_path, data)

        print(f"\n[SELESAI] Tensor akhir shape: {data.shape}")
        print(f"[SELESAI] Disimpan ke       : {output_npy_path}")

        return data
