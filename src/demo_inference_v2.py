# src/demo_inference.py
#
# Script demonstrasi inferensi model AttentiveSkel-3D v2.
# 
# Menerima input video mentah, mengekstrak fitur, memproses tensor ke (64, 33, 3),
# melakukan forward pass pada model yang dipilih, lalu merender video output yang
# memuat prediksi per-frame beserta visualisasi heatmap atensi per-sendi (jika ada).
#
# Cara menggunakan:
#   python src/demo_inference.py --video data/raw/Squat/Squat_001.mp4
#   python src/demo_inference.py --video data/raw/Squat/Squat_001.mp4 --model models/saved_models/v2/best_model_ablasi_a.pth

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
# pyrefly: ignore [missing-import]
import mediapipe as mp
from tqdm import tqdm

from src.data.extract_pose import PoseExtractor
from src.data.preprocess import DataPreprocessor
from src.models.v2.model_per_frame import AttentiveSkel3DPerFrame

def main():
    parser = argparse.ArgumentParser(description="Demo Inferensi AttentiveSkel-3D v2")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path ke video input mentah."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/saved_models/v2/best_model_full.pth",
        help="Path ke model .pth (default: models/saved_models/v2/best_model_full.pth)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/demo_outputs",
        help="Direktori penyimpanan hasil (video dan tensor sementara)"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"[ERROR] Video tidak ditemukan: {video_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # =================================================================
    # 1. Tentukan device & Muat Model secara Cerdas
    # =================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Menggunakan device: {device}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Deteksi fitur atensi dari state_dict (penting untuk model ablasi)
    use_spatial_prior = any(k.startswith("biomechanical_spatial_prior") for k in state_dict.keys())
    use_learned_spatial = any(k.startswith("learned_spatial_attention") for k in state_dict.keys())
    use_temporal_attention = any(k.startswith("temporal_attention") for k in state_dict.keys())

    print(f"[INFO] Model dimuat dari: {model_path.name}")
    print(f"       - Biomechanical Spatial Prior: {'Aktif' if use_spatial_prior else 'Nonaktif'}")
    print(f"       - Learned Spatial Attention  : {'Aktif' if use_learned_spatial else 'Nonaktif'}")
    print(f"       - Temporal Attention         : {'Aktif' if use_temporal_attention else 'Nonaktif'}")

    model = AttentiveSkel3DPerFrame(
        num_classes=2,
        use_spatial_prior=use_spatial_prior,
        use_learned_spatial=use_learned_spatial,
        use_temporal_attention=use_temporal_attention
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Ekstrak bobot spatial attention untuk heatmap
    if use_spatial_prior:
        # Sigmoid untuk membawa ke rentang 0-1
        bsp_tensor = torch.sigmoid(model.biomechanical_spatial_prior).detach().cpu().squeeze().numpy()
        # bsp_tensor memiliki panjang 33
        spatial_weights = bsp_tensor
        print("[INFO] Bobot heatmap berhasil diekstraksi dari model.")
    else:
        spatial_weights = np.ones(33) * 0.5
        print("[INFO] Model tidak memiliki Spatial Prior. Heatmap dinonaktifkan (seragam).")

    # =================================================================
    # 2. Ekstraksi Fitur dari Video Mentah (MediaPipe)
    # =================================================================
    raw_npy_path = output_dir / f"{stem}_raw.npy"
    print("\n[EKSTRAKSI POSE]")
    extractor = PoseExtractor(model_complexity=2)
    extractor.extract_video(
        video_path=str(video_path),
        output_npy_path=str(raw_npy_path)
    )

    # =================================================================
    # 3. Preprocessing Data (Cleaning, Normalize, Resample ke 64)
    # =================================================================
    tensor_64_path = output_dir / f"{stem}_64.npy"
    print("\n[PREPROCESSING TENSOR]")
    preprocessor = DataPreprocessor(target_frames=64)
    try:
        tensor_data = preprocessor.process(
            npy_file_path=str(raw_npy_path),
            output_npy_path=str(tensor_64_path)
        )
    except Exception as e:
        print(f"[ERROR] Preprocessing gagal: {e}")
        sys.exit(1)

    # =================================================================
    # 4. Inferensi Model per-frame
    # =================================================================
    print("\n[INFERENSI AI]")
    # tensor_data: (64, 33, 3) -> (1, 64, 33, 3)
    input_tensor = torch.tensor(tensor_data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)         # (1, 64, 2)
        preds  = logits.argmax(dim=2)        # (1, 64)
        probs  = torch.softmax(logits, dim=2) # (1, 64, 2)

    preds_np = preds.squeeze().cpu().numpy()  # (64,)
    probs_np = probs.squeeze().cpu().numpy()  # (64, 2)

    n_benar = (preds_np == 0).sum()
    n_salah = (preds_np == 1).sum()
    print(f"Prediksi Selesai! Frame BENAR: {n_benar}, Frame SALAH: {n_salah}")

    # =================================================================
    # 5. Visualisasi dan Rendering Video Output
    # =================================================================
    print("\n[RENDERING VIDEO OUTPUT]")
    out_video_path = output_dir / f"{stem}_heatmap_demo.mp4"
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka kembali video: {video_path}")
        sys.exit(1)

    total_frames_asli = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Gunakan codec avc1 (H.264) agar video bisa diputar di browser (HTML5)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # Tentukan 64 indeks frame asli yang akan dirender
    target_indices = np.linspace(0, total_frames_asli - 1, 64).astype(int)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=1, 
        min_detection_confidence=0.5
    )

    print("Merender 64 frame sekuensial dengan heatmap...")
    for i in tqdm(range(64)):
        idx_asli = target_indices[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_asli)
        ret, frame = cap.read()
        if not ret:
            # Fallback jika gagal baca
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Proses MediaPipe hanya untuk mendapatkan titik (X,Y) piksel asli pada frame ini
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        pred_label = preds_np[i]
        prob_salah = probs_np[i, 1] * 100

        # Draw Skeleton & Heatmap
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Gambar garis koneksi standar
            for conn in mp_pose.POSE_CONNECTIONS:
                p1, p2 = conn
                lm1, lm2 = landmarks[p1], landmarks[p2]
                if lm1.visibility > 0.3 and lm2.visibility > 0.3:
                    x1, y1 = int(lm1.x * width), int(lm1.y * height)
                    x2, y2 = int(lm2.x * width), int(lm2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)

            # ── Heatmap Atensi (Contrast-Amplified) ──────────────────────────
            # Masalah: bobot BSP sangat berdekatan (misal 0.66–0.77), sehingga
            # normalisasi linear menghasilkan warna yang seragam dan tidak jelas.
            # Solusi: Normalisasi ke [0,1] LALU pangkatkan dengan eksponen tinggi
            # (power amplification) sehingga perbedaan kecil menjadi sangat kontras.

            max_w = spatial_weights.max()
            min_w = spatial_weights.min()
            w_range = max_w - min_w if (max_w - min_w) > 1e-6 else 1.0

            POWER = 4.0  # Semakin besar, semakin dramatis perbedaan warna
            MIN_RADIUS = 4
            MAX_RADIUS = 22

            for j, lm in enumerate(landmarks):
                if lm.visibility > 0.3:
                    x, y = int(lm.x * width), int(lm.y * height)

                    raw_w   = spatial_weights[j]
                    # Step 1: Normalisasi linear ke [0, 1]
                    norm_w  = (raw_w - min_w) / w_range
                    # Step 2: Power amplification — sendi rendah makin mendekati 0
                    amp_w   = norm_w ** POWER

                    # Step 3: Colormap TURBO memberikan kontras lebih baik dari JET
                    # Biru gelap (atensi rendah) → Kuning → Merah terang (atensi tinggi)
                    val = int(amp_w * 255)
                    color = cv2.applyColorMap(np.uint8([[[val]]]), cv2.COLORMAP_TURBO)[0][0]
                    color = (int(color[0]), int(color[1]), int(color[2]))

                    # Step 4: Ukuran lingkaran proporsional dengan atensi
                    radius = MIN_RADIUS + int(amp_w * (MAX_RADIUS - MIN_RADIUS))

                    # Step 5: Glow effect — lingkaran besar transparan di belakang
                    # (mensimulasikan efek cahaya/glow pada sendi penting)
                    if amp_w > 0.3:
                        overlay = frame.copy()
                        glow_r  = radius + int(amp_w * 12)
                        cv2.circle(overlay, (x, y), glow_r, color, -1)
                        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

                    # Step 6: Lingkaran inti solid
                    cv2.circle(frame, (x, y), radius, color, -1)
                    # Outline putih tipis agar terlihat di background apapun
                    cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)

        # Panel atas — diperbesar 40 → 80px
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame {i+1}/64",
                    (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Model: {model_path.name}",
                    (14, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 200, 255), 1, cv2.LINE_AA)

        # Panel bawah — diperbesar 60 → 110px
        cv2.rectangle(frame, (0, height - 110), (width, height), (0, 0, 0), -1)
        if pred_label == 0:
            status = "BENAR  (Label 0)"
            color = (0, 230, 80)
        else:
            status = "SALAH  (Label 1)"
            color = (60, 60, 255)
        
        cv2.putText(frame, f"Prediksi AI: {status}",
                    (14, height - 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        cv2.putText(frame, f"Probabilitas Salah: {prob_salah:.1f}%",
                    (14, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 210, 220), 2, cv2.LINE_AA)
        
        writer.write(frame)

    cap.release()
    writer.release()
    pose.close()

    print(f"\n[SELESAI] Video tersimpan di: {out_video_path}")
    print("="*60)

if __name__ == "__main__":
    main()
