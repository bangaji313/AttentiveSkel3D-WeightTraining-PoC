# src/data/extract_pose.py
#
# Modul untuk mengekstraksi 33 titik sendi tubuh (landmarks) dari video
# menggunakan MediaPipe BlazePose dan menyimpannya sebagai file .npy.
#
# Struktur output array: (T, 33, 4)
#   T   = jumlah frame yang berhasil diproses
#   33  = jumlah landmark BlazePose
#   4   = [x, y, z, visibility]

import os
import cv2
import numpy as np
import mediapipe as mp


class PoseExtractor:
    """
    Mengekstraksi pose skeleton 3D dari file video menggunakan MediaPipe BlazePose.

    Atribut:
        model_complexity (int): Kompleksitas model BlazePose (0, 1, atau 2).
                                Nilai 2 memberikan akurasi tertinggi.
        min_detection_confidence (float): Ambang batas minimum deteksi pose.
        min_tracking_confidence (float): Ambang batas minimum pelacakan pose.
    """

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Inisialisasi PoseExtractor dengan konfigurasi MediaPipe BlazePose.

        Args:
            model_complexity: Tingkat kompleksitas model (0=Lite, 1=Full, 2=Heavy).
            min_detection_confidence: Ambang batas keyakinan deteksi awal.
            min_tracking_confidence: Ambang batas keyakinan pelacakan antar-frame.
        """
        # Inisialisasi modul MediaPipe yang akan digunakan
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        # Simpan konfigurasi model
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def extract_video(
        self,
        video_path: str,
        output_npy_path: str,
        output_video_path: str = None,
    ) -> np.ndarray:
        """
        Membaca video frame-demi-frame, mengekstraksi pose dari setiap frame,
        dan menyimpan hasilnya sebagai file .npy.

        Args:
            video_path (str): Path ke file video input (.mp4, .avi, dll.).
            output_npy_path (str): Path tujuan untuk menyimpan array NumPy (.npy).
            output_video_path (str, optional): Jika diberikan, video hasil visualisasi
                                               skeleton akan disimpan ke path ini.

        Returns:
            np.ndarray: Array pose dengan bentuk (T, 33, 4), di mana T adalah jumlah
                        frame valid, 33 adalah jumlah landmark, dan 4 adalah
                        [x, y, z, visibility].

        Raises:
            FileNotFoundError: Jika file video tidak ditemukan di video_path.
            IOError: Jika video tidak dapat dibuka oleh OpenCV.
        """
        # --- Validasi file input ---
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File video tidak ditemukan: '{video_path}'")

        # --- Buka video menggunakan OpenCV ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"OpenCV tidak dapat membuka video: '{video_path}'")

        # Ambil properti video untuk digunakan pada output video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Membuka video: {video_path}")
        print(f"[INFO] Resolusi: {frame_width}x{frame_height} | FPS: {fps:.2f} | Total Frame: {total_frames}")

        # --- Siapkan VideoWriter jika output_video_path diberikan ---
        video_writer = None
        if output_video_path is not None:
            # Pastikan direktori tujuan ada sebelum menulis
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (frame_width, frame_height)
            )
            print(f"[INFO] Output video skeleton akan disimpan ke: {output_video_path}")

        # --- List untuk menampung data pose dari setiap frame ---
        all_pose_landmarks = []
        frame_idx = 0
        frames_with_pose = 0
        frames_without_pose = 0

        # --- Inisialisasi model BlazePose dalam context manager ---
        with self._mp_pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as pose_model:

            while cap.isOpened():
                # Baca satu frame dari video
                success, frame_bgr = cap.read()
                if not success:
                    # Akhir dari video atau error pembacaan frame
                    break

                frame_idx += 1

                # Konversi warna dari BGR (OpenCV default) ke RGB (MediaPipe default)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Tandai image sebagai tidak dapat ditulis untuk meningkatkan performa
                frame_rgb.flags.writeable = False

                # --- Jalankan inferensi pose estimasi ---
                results = pose_model.process(frame_rgb)

                # Kembalikan writeable flag sebelum menggambar
                frame_rgb.flags.writeable = True

                if results.pose_landmarks:
                    frames_with_pose += 1

                    # Ekstraksi 33 landmark: [x, y, z, visibility] untuk setiap titik sendi
                    frame_landmarks = np.array(
                        [
                            [lm.x, lm.y, lm.z, lm.visibility]
                            for lm in results.pose_landmarks.landmark
                        ],
                        dtype=np.float32,
                    )  # Bentuk: (33, 4)

                    all_pose_landmarks.append(frame_landmarks)

                    # Gambar skeleton pada frame jika output video diminta
                    if video_writer is not None:
                        # Konversi kembali ke BGR untuk penggambaran OpenCV
                        frame_bgr_draw = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        self._mp_drawing.draw_landmarks(
                            image=frame_bgr_draw,
                            landmark_list=results.pose_landmarks,
                            connections=self._mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
                        )
                        # Tambahkan informasi frame pada video output
                        cv2.putText(
                            frame_bgr_draw,
                            f"Frame: {frame_idx} | Pose: Terdeteksi",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                        video_writer.write(frame_bgr_draw)

                else:
                    frames_without_pose += 1

                    # Jika pose tidak terdeteksi, tulis frame asli (tanpa skeleton)
                    if video_writer is not None:
                        frame_bgr_draw = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            frame_bgr_draw,
                            f"Frame: {frame_idx} | Pose: Tidak Terdeteksi",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                        video_writer.write(frame_bgr_draw)

        # --- Bersihkan resource OpenCV ---
        cap.release()
        if video_writer is not None:
            video_writer.release()

        print(f"[INFO] Pemrosesan selesai.")
        print(f"[INFO] Frame dengan pose terdeteksi : {frames_with_pose}")
        print(f"[INFO] Frame tanpa pose             : {frames_without_pose}")

        # --- Susun semua frame menjadi satu array NumPy ---
        if len(all_pose_landmarks) == 0:
            print("[PERINGATAN] Tidak ada pose yang berhasil diekstraksi dari video ini.")
            pose_array = np.empty((0, 33, 4), dtype=np.float32)
        else:
            # Tumpuk list frame menjadi array (T, 33, 4)
            pose_array = np.stack(all_pose_landmarks, axis=0)
            print(f"[INFO] Bentuk array pose akhir: {pose_array.shape}  (Frame x 33 Landmark x 4 Koordinat)")

        # --- Simpan array ke file .npy ---
        os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
        np.save(output_npy_path, pose_array)
        print(f"[INFO] Array pose disimpan ke: {output_npy_path}")

        return pose_array

