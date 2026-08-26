# src/demo_inference_core.py
#
# Inti logika inferensi untuk digunakan oleh aplikasi web FastAPI (src/app.py).

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import mediapipe as mp

from src.data.extract_pose import PoseExtractor
from src.data.preprocess import DataPreprocessor
from src.models.v2.model_per_frame import AttentiveSkel3DPerFrame

def run_inference_pipeline(
    video_path: str,
    model_path: str,
    output_dir: str,
    progress_callback=None
) -> dict:
    """
    Menjalankan proses penuh dari video mentah hingga video output dengan heatmap.

    Args:
        video_path        : Path absolut ke video mentah.
        model_path        : Path absolut ke model .pth.
        output_dir        : Path absolut ke folder output.
        progress_callback : Opsional. Fungsi (step, total, message, detail) untuk
                            melaporkan progress secara real-time (digunakan SSE).

    Returns:
        dict: {'video_path', 'n_benar', 'n_salah'}
    """
    def report(step, msg, detail=""):
        if progress_callback:
            progress_callback(step, 4, msg, detail)
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # ── 1. Muat Model ─────────────────────────────────────────────────────────
    report(1, "Memuat model AI ke memori...",
           f"Membaca bobot dari: {Path(model_path).name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    use_spatial_prior    = any(k.startswith("biomechanical_spatial_prior") for k in state_dict.keys())
    use_learned_spatial  = any(k.startswith("learned_spatial_attention")   for k in state_dict.keys())
    use_temporal_attention = any(k.startswith("temporal_attention")         for k in state_dict.keys())

    model = AttentiveSkel3DPerFrame(
        num_classes=2,
        use_spatial_prior=use_spatial_prior,
        use_learned_spatial=use_learned_spatial,
        use_temporal_attention=use_temporal_attention
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    bsp_label = "Aktif" if use_spatial_prior else "Nonaktif"
    report(1, "✓ Model dimuat ke memori.",
           f"Device: {device} | BSP: {bsp_label}")

    if use_spatial_prior:
        bsp_tensor = torch.sigmoid(model.biomechanical_spatial_prior).detach().cpu().squeeze().numpy()
        spatial_weights = bsp_tensor
    else:
        spatial_weights = np.ones(33) * 0.5

    # ── 2. Ekstraksi Pose (MediaPipe) ──────────────────────────────────────────
    report(2, "Mengekstrak pose dari video mentah...",
           f"Membuka: {Path(video_path).name}")

    raw_npy_path = output_dir / f"{stem}_raw.npy"
    extractor = PoseExtractor(model_complexity=2)
    extractor.extract_video(
        video_path=str(video_path),
        output_npy_path=str(raw_npy_path)
    )

    import numpy as _np_check
    _raw = _np_check.load(str(raw_npy_path))
    report(2, "✓ Ekstraksi pose selesai.",
           f"Berhasil mengekstrak {_raw.shape[0]} frame × 33 sendi × 4 koordinat")

    # ── 3. Preprocessing & Inferensi AI ───────────────────────────────────────
    report(3, "Preprocessing tensor & Inferensi AI...",
           f"Memampatkan {_raw.shape[0]} frame → 64 frame (shape: 64×33×3)")

    tensor_64_path = output_dir / f"{stem}_64.npy"
    preprocessor = DataPreprocessor(target_frames=64)
    tensor_data = preprocessor.process(
        npy_file_path=str(raw_npy_path),
        output_npy_path=str(tensor_64_path)
    )

    # ── 4. Inferensi ───────────────────────────────────────────────────────────
    input_tensor = torch.tensor(tensor_data).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)          # (1, 64, 2)
        preds  = logits.argmax(dim=2)         # (1, 64)
        probs  = torch.softmax(logits, dim=2) # (1, 64, 2)

    preds_np = preds.squeeze().cpu().numpy()  # (64,)
    probs_np = probs.squeeze().cpu().numpy()  # (64, 2)
    n_benar  = int((preds_np == 0).sum())
    n_salah  = int((preds_np == 1).sum())

    report(3, "✓ Inferensi AI selesai.",
           f"{n_benar} frame BENAR + {n_salah} frame SALAH dari 64 frame total")

    # ── 5. Rendering Video Output ──────────────────────────────────────────────
    report(4, "Merender video output dengan heatmap atensi...",
           "Menggambar visualisasi heatmap per-frame...")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames_asli = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Gunakan codec avc1 (H.264) agar video bisa diputar di browser (HTML5)
    out_video_path = output_dir / f"{stem}_heatmap_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    target_indices = np.linspace(0, total_frames_asli - 1, 64).astype(int)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=1, 
        min_detection_confidence=0.5
    )

    for i in range(64):
        idx_asli = target_indices[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_asli)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        pred_label = preds_np[i]
        prob_salah = probs_np[i, 1] * 100

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            for conn in mp_pose.POSE_CONNECTIONS:
                p1, p2 = conn
                lm1, lm2 = landmarks[p1], landmarks[p2]
                if lm1.visibility > 0.3 and lm2.visibility > 0.3:
                    x1, y1 = int(lm1.x * width), int(lm1.y * height)
                    x2, y2 = int(lm2.x * width), int(lm2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 3)

            # ── Heatmap Atensi (Power Amplification + Glow) ──────────────
            # Masalah: bobot BSP berdekatan (0.66–0.77), normalisasi linear
            # menghasilkan warna seragam. Solusi: pangkatkan dengan POWER tinggi
            # sehingga perbedaan kecil jadi sangat kontras secara visual.
            max_w    = spatial_weights.max()
            min_w    = spatial_weights.min()
            w_range  = max_w - min_w if (max_w - min_w) > 1e-6 else 1.0
            POWER      = 4.0   # Amplifikasi kontras
            MIN_RADIUS = 4
            MAX_RADIUS = 22

            for j, lm in enumerate(landmarks):
                if lm.visibility > 0.3:
                    x, y = int(lm.x * width), int(lm.y * height)

                    norm_w = (spatial_weights[j] - min_w) / w_range
                    amp_w  = norm_w ** POWER  # Amplifikasi

                    val   = int(amp_w * 255)
                    color = cv2.applyColorMap(np.uint8([[[val]]]), cv2.COLORMAP_TURBO)[0][0]
                    color = (int(color[0]), int(color[1]), int(color[2]))

                    radius = MIN_RADIUS + int(amp_w * (MAX_RADIUS - MIN_RADIUS))

                    # Glow effect pada sendi dengan atensi tinggi
                    if amp_w > 0.3:
                        overlay = frame.copy()
                        glow_r  = radius + int(amp_w * 12)
                        cv2.circle(overlay, (x, y), glow_r, color, -1)
                        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

                    cv2.circle(frame, (x, y), radius, color, -1)
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

    report(4, "✓ Video heatmap selesai dirender!",
           f"Tersimpan: {out_video_path.name} | Frame BENAR: {n_benar} | Frame SALAH: {n_salah}")

    return {
        "video_path": str(out_video_path),
        "n_benar"   : n_benar,
        "n_salah"   : n_salah,
    }
