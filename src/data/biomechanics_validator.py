# src/data/biomechanics_validator.py
#
# Modul validasi Ground Truth labeling dataset berbasis kriteria biomekanika
# dari literatur ilmiah terbaru (2022-2024).
#
# Referensi yang digunakan:
#   [1] Chen, K.-Y., Shin, J., Hasan, M. A. M., Liaw, J.-J., Yuichi, O., & Tomioka, Y.
#       (2022). "Fitness Movement Types and Completeness Detection Using a
#       Transfer-Learning-Based Deep Neural Network."
#   [2] Rao, P., Asha, C. S., & Rao, R. P. (2023). "Real-time Posture Correction
#       of Squat Exercise: A Deep Learning Approach for Performance Analysis and
#       Error Correction."
#   [3] Ko, Y.-M., Nasridinov, A., & Park, S.-H. (2024). "Real-Time AI Posture
#       Correction for Powerlifting Exercises Using YOLOv5 and MediaPipe."
#
# Tujuan utama:
#   Memvalidasi secara matematis apakah video yang diberi label "Benar"
#   memang memenuhi kriteria biomekanik yang valid secara kuantitatif,
#   sehingga kualitas ground truth dataset dapat dipertanggungjawabkan secara
#   akademis.
#
# Indeks Landmark MediaPipe BlazePose yang digunakan:
#   11 = Left Shoulder   12 = Right Shoulder
#   13 = Left Elbow      14 = Right Elbow
#   15 = Left Wrist      16 = Right Wrist
#   23 = Left Hip        24 = Right Hip
#   25 = Left Knee       26 = Right Knee
#   27 = Left Ankle      28 = Right Ankle
#
# Format tensor input semua metode validate_*:
#   (F, 33, 3) → F frame, 33 landmark, 3 koordinat (x, y, z) ternormalisasi
#   Koordinat telah dinormalisasi terhadap mid-hip (pusat = 0,0,0) dan
#   diskalakan dengan panjang torso.

import numpy as np


class BiomechanicalValidator:
    """
    Kelas validasi biomekanik untuk tiga gerakan latihan beban:
      - Squat
      - Bench Press
      - Deadlift

    Semua nilai threshold didasarkan pada literatur biomekanika olahraga terbaru:
      - Chen et al. (2022) — deteksi kelengkapan gerakan fitness
      - Rao et al. (2023)  — koreksi postur squat berbasis deep learning
      - Ko et al. (2024)   — koreksi postur powerlifting real-time

    Tidak ada state internal; semua metode dapat dipanggil tanpa inisialisasi khusus.
    """

    # ── Konstanta threshold biomekanik ──────────────────────────────────────────

    # Squat
    SQUAT_DEPTH_THRESHOLD_DEG: float = 100.0   # Sudut lutut maks. di posisi terdalam (derajat)
    SQUAT_VALGUS_RATIO_THRESHOLD: float = 0.85  # Rasio lebar lutut/pergelangan kaki minimum

    # Bench Press
    BENCH_ELBOW_THRESHOLD_DEG: float = 85.0    # Sudut siku maks. saat bar paling rendah (derajat)

    # Deadlift
    DEADLIFT_SPINE_MAX_DEG: float = 60.0       # Sudut inklinasi punggung maks. dari vertikal (derajat)
    DEADLIFT_SPINE_MIN_DEG: float = 20.0       # Sudut inklinasi punggung min. (memastikan gerakan terjadi)

    # ── Indeks landmark MediaPipe BlazePose ────────────────────────────────────
    IDX_L_SHOULDER = 11;  IDX_R_SHOULDER = 12
    IDX_L_ELBOW    = 13;  IDX_R_ELBOW    = 14
    IDX_L_WRIST    = 15;  IDX_R_WRIST    = 16
    IDX_L_HIP      = 23;  IDX_R_HIP      = 24
    IDX_L_KNEE     = 25;  IDX_R_KNEE     = 26
    IDX_L_ANKLE    = 27;  IDX_R_ANKLE    = 28

    # ── Metode utilitas ─────────────────────────────────────────────────────────

    @staticmethod
    def calculate_angle_3d(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> float:
        """
        Menghitung sudut (dalam derajat) yang dibentuk oleh tiga titik 3D,
        dengan `b` sebagai titik sudut (vertex).

        Rumus:
            θ = arccos( (BA · BC) / (|BA| × |BC|) )

        Args:
            a: Koordinat 3D titik pertama, bentuk (3,).
            b: Koordinat 3D titik sudut (vertex), bentuk (3,).
            c: Koordinat 3D titik ketiga, bentuk (3,).

        Returns:
            float: Sudut dalam derajat [0°, 180°].
                   Mengembalikan 180.0 jika salah satu vektor memiliki panjang nol
                   (posisi landmark yang identik atau tidak valid).
        """
        # Vektor dari vertex b ke titik a dan c
        ba = a - b
        bc = c - b

        # Panjang masing-masing vektor
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        # Hindari pembagian dengan nol akibat landmark yang bertumpuk
        if norm_ba < 1e-8 or norm_bc < 1e-8:
            return 180.0

        # Hitung cosinus sudut dengan dot product, clamp ke [-1, 1] untuk
        # menghindari domain error pada arccos akibat floating point
        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def _get_per_frame_angles(
        tensor_data: np.ndarray,
        idx_a: int,
        idx_b: int,
        idx_c: int,
    ) -> np.ndarray:
        """
        Menghitung sudut satu sendi untuk seluruh frame sekaligus (vektorisasi NumPy).

        Args:
            tensor_data : Array (F, 33, 3).
            idx_a, idx_b, idx_c: Indeks landmark untuk titik a, b (vertex), c.

        Returns:
            np.ndarray: Array sudut (derajat) dengan panjang F.
        """
        a = tensor_data[:, idx_a, :]  # (F, 3)
        b = tensor_data[:, idx_b, :]  # (F, 3)
        c = tensor_data[:, idx_c, :]  # (F, 3)

        ba = a - b  # (F, 3)
        bc = c - b  # (F, 3)

        # Dot product per frame
        dot_products = np.einsum("fi,fi->f", ba, bc)  # (F,)

        # Norma per frame
        norm_ba = np.linalg.norm(ba, axis=1)  # (F,)
        norm_bc = np.linalg.norm(bc, axis=1)  # (F,)

        # Masker untuk frame dengan landmark valid
        valid = (norm_ba > 1e-8) & (norm_bc > 1e-8)

        cos_angles = np.full(tensor_data.shape[0], 1.0)  # default cos=1 → 0 derajat
        cos_angles[valid] = dot_products[valid] / (norm_ba[valid] * norm_bc[valid])
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        return np.degrees(np.arccos(cos_angles))

    # ── Validator utama ─────────────────────────────────────────────────────────

    def validate_squat(
        self,
        tensor_data: np.ndarray,
    ) -> tuple[bool, str]:
        """
        Memvalidasi eksekusi gerakan Squat berdasarkan dua kriteria biomekanik:

        **Kriteria 1 — Kedalaman Squat (Squat Depth)**
          Referensi: Escamilla et al. (2001), Hales et al. (2009)
          Kriteria  : Sudut Pinggul-Lutut-Pergelangan Kaki (rata-rata kiri & kanan)
                      harus mencapai ≤ 100° pada frame dengan fleksi lutut terdalam.
          Rasional  : Sudut 100° secara klinis setara dengan posisi "parallel squat"
                      (paha sejajar tanah), yang merupakan batas minimum kedalaman
                      yang dianggap efektif secara biomekanik.

        **Kriteria 2 — Knee Valgus (Lutut Mengunci ke Dalam)**
          Referensi: Bell et al. (2008), Chen et al. (2022)
          Kriteria  : Pada posisi terdalam, lebar horizontal lutut (sumbu X) tidak
                      boleh < 85% dari lebar horizontal pergelangan kaki.
          Rasional  : Kolaps medial lutut (valgus) meningkatkan risiko cedera ACL
                      secara signifikan.

        Args:
            tensor_data: Array pose (F, 33, 3) yang telah dinormalisasi.

        Returns:
            tuple[bool, str]: (is_valid, alasan)
                - is_valid : True jika kedua kriteria terpenuhi, False jika ada yang gagal.
                - alasan   : Penjelasan kuantitatif hasil validasi.
        """
        if tensor_data.ndim != 3 or tensor_data.shape[1:] != (33, 3):
            return False, f"Format tensor tidak valid: {tensor_data.shape} (diharapkan (F, 33, 3))"

        # Hitung sudut Pinggul-Lutut-Pergelangan Kaki per frame untuk kiri dan kanan
        angles_left  = self._get_per_frame_angles(
            tensor_data, self.IDX_L_HIP, self.IDX_L_KNEE, self.IDX_L_ANKLE
        )
        angles_right = self._get_per_frame_angles(
            tensor_data, self.IDX_R_HIP, self.IDX_R_KNEE, self.IDX_R_ANKLE
        )
        angles_avg = (angles_left + angles_right) / 2.0  # Rata-rata kiri & kanan

        # Temukan frame dengan sudut lutut minimum (posisi terdalam)
        min_knee_angle = float(np.min(angles_avg))
        deepest_frame  = int(np.argmin(angles_avg))

        # ── Kriteria 1: Kedalaman Squat ───────────────────────────────────────
        if min_knee_angle > self.SQUAT_DEPTH_THRESHOLD_DEG:
            return False, (
                f"Kedalaman squat tidak memadai. "
                f"Sudut lutut minimum = {min_knee_angle:.1f}° "
                f"(threshold ≤ {self.SQUAT_DEPTH_THRESHOLD_DEG}°, "
                f"terjadi pada frame {deepest_frame}). "
                f"Posisi 'parallel squat' belum tercapai."
            )

        # ── Kriteria 2: Knee Valgus ───────────────────────────────────────────
        # Jarak horizontal (sumbu X) antara lutut kiri dan kanan
        knee_width  = abs(
            tensor_data[deepest_frame, self.IDX_L_KNEE,  0]
            - tensor_data[deepest_frame, self.IDX_R_KNEE,  0]
        )
        # Jarak horizontal (sumbu X) antara pergelangan kaki kiri dan kanan
        ankle_width = abs(
            tensor_data[deepest_frame, self.IDX_L_ANKLE, 0]
            - tensor_data[deepest_frame, self.IDX_R_ANKLE, 0]
        )

        if ankle_width < 1e-6:
            # Tidak bisa menghitung rasio — lewati pemeriksaan valgus
            return True, (
                f"Squat valid (kedalaman OK: sudut lutut = {min_knee_angle:.1f}°). "
                f"Pemeriksaan knee valgus dilewati (lebar pergelangan kaki ≈ 0)."
            )

        valgus_ratio = knee_width / ankle_width

        if valgus_ratio < self.SQUAT_VALGUS_RATIO_THRESHOLD:
            return False, (
                f"Terdeteksi Knee Valgus (kolaps medial lutut). "
                f"Lebar lutut = {knee_width:.3f}, "
                f"lebar pergelangan kaki = {ankle_width:.3f}, "
                f"rasio = {valgus_ratio:.2f} "
                f"(threshold ≥ {self.SQUAT_VALGUS_RATIO_THRESHOLD}). "
                f"Lutut kolaps ke dalam pada frame {deepest_frame}."
            )

        return True, (
            f"Squat valid. "
            f"Sudut lutut minimum = {min_knee_angle:.1f}° (≤ {self.SQUAT_DEPTH_THRESHOLD_DEG}°), "
            f"rasio lebar lutut/kaki = {valgus_ratio:.2f} (≥ {self.SQUAT_VALGUS_RATIO_THRESHOLD}). "
            f"Posisi terdalam pada frame {deepest_frame}."
        )

    def validate_benchpress(
        self,
        tensor_data: np.ndarray,
    ) -> tuple[bool, str]:
        """
        Memvalidasi eksekusi gerakan Bench Press berdasarkan satu kriteria biomekanik:

        **Kriteria — Full Range of Motion Siku (Elbow ROM)**
          Referensi: Glass & Armstrong (1997), Barnett et al. (1995), Chen et al. (2022)
          Kriteria  : Sudut Bahu-Siku-Pergelangan Tangan (rata-rata kiri & kanan)
                      harus mencapai ≤ 85° pada frame dengan fleksi siku terdalam
                      (posisi bar paling dekat ke dada).
          Rasional  : Sudut 85° (toleransi ±5° dari batas ideal 80°) memastikan
                      bar diturunkan dengan ROM yang mencukupi untuk mengaktivasi
                      pectoralis major secara optimal. Lebih dari 85° mengindikasikan
                      "half rep" yang tidak efektif.

        Args:
            tensor_data: Array pose (F, 33, 3) yang telah dinormalisasi.

        Returns:
            tuple[bool, str]: (is_valid, alasan)
        """
        if tensor_data.ndim != 3 or tensor_data.shape[1:] != (33, 3):
            return False, f"Format tensor tidak valid: {tensor_data.shape} (diharapkan (F, 33, 3))"

        # Hitung sudut Bahu-Siku-Pergelangan Tangan per frame
        angles_left  = self._get_per_frame_angles(
            tensor_data, self.IDX_L_SHOULDER, self.IDX_L_ELBOW, self.IDX_L_WRIST
        )
        angles_right = self._get_per_frame_angles(
            tensor_data, self.IDX_R_SHOULDER, self.IDX_R_ELBOW, self.IDX_R_WRIST
        )
        angles_avg = (angles_left + angles_right) / 2.0

        # Temukan frame dengan fleksi siku terdalam (sudut siku minimum = bar paling rendah)
        min_elbow_angle = float(np.min(angles_avg))
        lowest_frame    = int(np.argmin(angles_avg))

        if min_elbow_angle > self.BENCH_ELBOW_THRESHOLD_DEG:
            return False, (
                f"Range of Motion siku tidak memadai (half rep). "
                f"Sudut siku minimum = {min_elbow_angle:.1f}° "
                f"(threshold ≤ {self.BENCH_ELBOW_THRESHOLD_DEG}°, "
                f"terjadi pada frame {lowest_frame}). "
                f"Bar tidak diturunkan cukup dekat ke dada."
            )

        return True, (
            f"Bench Press valid. "
            f"Sudut siku minimum = {min_elbow_angle:.1f}° "
            f"(≤ {self.BENCH_ELBOW_THRESHOLD_DEG}°), "
            f"full ROM tercapai pada frame {lowest_frame}."
        )

    def validate_deadlift(
        self,
        tensor_data: np.ndarray,
    ) -> tuple[bool, str]:
        """
        Memvalidasi eksekusi gerakan Deadlift berdasarkan satu kriteria biomekanik:

        **Kriteria — Sudut Inklinasi Punggung / Hip Hinge Pattern**
          Referensi: Escamilla et al. (2000), Hales (2010), Chen et al. (2022)
          Kriteria  : Pada posisi terbawah (torso paling miring ke depan), sudut
                      inklinasi punggung dari sumbu vertikal harus:
                      ≥ 20° (ada gerakan hip hinge yang bermakna), DAN
                      ≤ 60° (punggung netral, tidak membungkuk berlebihan).
          Rasional  : Sudut inklinasi punggung < 20° mengindikasikan tidak ada
                      hip hinge (gerakan salah atau data kurang representatif).
                      Sudut > 60° mengindikasikan lumbar flexion yang berlebihan
                      (punggung bungkuk/rounded back), yang meningkatkan risiko
                      cedera cakram lumbar secara signifikan.

        Metode pengukuran:
          - Vektor tulang belakang (spine vector) = mid_shoulder - mid_hip
            (dalam ruang koordinat ternormalisasi, mid_hip = asal/origin)
          - Vektor vertikal = [0, -1, 0]
            (Y negatif = ke atas, karena Y MediaPipe bertambah ke bawah)
          - Sudut inklinasi = sudut antara spine_vector dan [0, -1, 0]

        Args:
            tensor_data: Array pose (F, 33, 3) yang telah dinormalisasi.

        Returns:
            tuple[bool, str]: (is_valid, alasan)
        """
        if tensor_data.ndim != 3 or tensor_data.shape[1:] != (33, 3):
            return False, f"Format tensor tidak valid: {tensor_data.shape} (diharapkan (F, 33, 3))"

        # Hitung mid-shoulder per frame (dalam ruang ternormalisasi, mid-hip = origin)
        mid_shoulder = (
            tensor_data[:, self.IDX_L_SHOULDER, :] + tensor_data[:, self.IDX_R_SHOULDER, :]
        ) / 2.0  # (F, 3)

        # Vektor tulang belakang: dari mid-hip (origin) ke mid-shoulder
        # Karena mid-hip sudah di (0,0,0) per frame, spine_vector = mid_shoulder
        spine_vector = mid_shoulder  # (F, 3)

        # Vektor vertikal "ke atas" dalam sistem koordinat MediaPipe (Y negatif = atas)
        vertical_up = np.array([0.0, -1.0, 0.0])

        # Hitung sudut inklinasi punggung dari vertikal untuk setiap frame
        norms = np.linalg.norm(spine_vector, axis=1)  # (F,)
        valid_mask = norms > 1e-8

        inclination_angles = np.full(tensor_data.shape[0], 0.0)
        if np.any(valid_mask):
            dot_products = spine_vector[valid_mask] @ vertical_up  # (F_valid,)
            cos_angles   = dot_products / norms[valid_mask]
            cos_angles   = np.clip(cos_angles, -1.0, 1.0)
            inclination_angles[valid_mask] = np.degrees(np.arccos(cos_angles))

        # Temukan frame dengan inklinasi punggung terbesar (posisi terbawah / paling membungkuk)
        max_inclination = float(np.max(inclination_angles))
        bottom_frame    = int(np.argmax(inclination_angles))

        # ── Kriteria 1: Pastikan ada gerakan hip hinge yang bermakna ─────────
        if max_inclination < self.DEADLIFT_SPINE_MIN_DEG:
            return False, (
                f"Hip hinge tidak terdeteksi secara bermakna. "
                f"Inklinasi punggung maksimum = {max_inclination:.1f}° "
                f"(threshold ≥ {self.DEADLIFT_SPINE_MIN_DEG}°). "
                f"Gerakan mungkin tidak merepresentasikan deadlift yang lengkap."
            )

        # ── Kriteria 2: Pastikan tidak ada rounded back berlebihan ────────────
        if max_inclination > self.DEADLIFT_SPINE_MAX_DEG:
            return False, (
                f"Terdeteksi excessive lumbar flexion (rounded back). "
                f"Inklinasi punggung maksimum = {max_inclination:.1f}° "
                f"(threshold ≤ {self.DEADLIFT_SPINE_MAX_DEG}°, "
                f"terjadi pada frame {bottom_frame}). "
                f"Punggung membungkuk berlebihan; risiko cedera lumbar tinggi."
            )

        return True, (
            f"Deadlift valid. "
            f"Inklinasi punggung maksimum = {max_inclination:.1f}° "
            f"({self.DEADLIFT_SPINE_MIN_DEG}° ≤ x ≤ {self.DEADLIFT_SPINE_MAX_DEG}°), "
            f"hip hinge pattern yang aman terdeteksi pada frame {bottom_frame}."
        )
