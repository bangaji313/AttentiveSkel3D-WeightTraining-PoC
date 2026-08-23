# web_app/inference_core_v2.py  (updated — Phase 3)
#
# Inti logika inferensi untuk aplikasi web FastAPI.
# Perubahan dari versi sebelumnya:
#   - Heatmap sendi sekarang menggunakan Joint Influence Attribution per-frame
#     (perturbation-based) — BUKAN lagi BSP weights statis.
#   - BSP weights masih ditampilkan di Panel B (Learned Attention), tapi TIDAK
#     lagi menentukan warna heatmap joint.
#   - Baseline yang tidak memiliki BSP tetap bisa menghasilkan attribution
#     (perturbation bekerja tanpa memerlukan BSP).
#   - Output JSON explanation disimpan bersama video.
#
# File yang TIDAK diubah:
#   src/models/arsitektur_v2.py, bobot_model/*.pth, src/data/preprocess.py

import json
import os
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import mediapipe as mp

# Add web_app directory to path so sibling modules can import each other
WEB_APP_DIR = Path(__file__).resolve().parent
if str(WEB_APP_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_APP_DIR))

from src.data.extract_pose import PoseExtractor
from src.data.preprocess import DataPreprocessor
from src.models.arsitektur_v2 import AttentiveSkel3DPerFrame
from explainer_v2 import (
    forward_with_attention,
    joint_influence,
    sequence_joint_score,
    reference_attribution_share,
    perturbation_faithfulness_check,
    build_explanation_json,
    LANDMARK_NAMES,
    BIOMECHANICAL_REFERENCE,
)


def _load_model(model_path: Path, device: torch.device):
    """Load checkpoint dan bangun model dengan arsitektur yang terdeteksi otomatis."""
    checkpoint = torch.load(str(model_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    use_spatial_prior     = any(k.startswith("biomechanical_spatial_prior") for k in state_dict)
    use_learned_spatial   = any(k.startswith("learned_spatial_attention")   for k in state_dict)
    use_temporal_attention= any(k.startswith("temporal_attention")           for k in state_dict)

    model = AttentiveSkel3DPerFrame(
        num_classes=2,
        use_spatial_prior=use_spatial_prior,
        use_learned_spatial=use_learned_spatial,
        use_temporal_attention=use_temporal_attention,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, use_spatial_prior, use_learned_spatial, use_temporal_attention


def run_inference_pipeline(
    video_path: str,
    model_path: str,
    output_dir: str,
    exercise: str = "Squat",
    progress_callback=None,
) -> dict:
    """
    Menjalankan proses penuh dari video mentah hingga:
      - Video heatmap dengan Joint Influence Attribution per-frame
      - File JSON explanation yang seluruh nilainya traceable

    Args:
        video_path        : Path absolut ke video mentah.
        model_path        : Path absolut ke model .pth.
        output_dir        : Path absolut ke folder output.
        exercise          : Jenis latihan ('Squat'/'Bench Press'/'Deadlift').
        progress_callback : Opsional fungsi (step, total, message, detail).

    Returns:
        dict: {video_path, json_path, n_benar, n_salah, explanation}
    """
    TOTAL_STEPS = 5

    def report(step, msg, detail=""):
        if progress_callback:
            progress_callback(step, TOTAL_STEPS, msg, detail)

    video_path = Path(video_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # ── 1. Muat Model ────────────────────────────────────────────────────────
    report(1, "Memuat model AI ke memori...",
           f"Membaca bobot dari: {model_path.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, use_sp, use_ls, use_ta = _load_model(model_path, device)

    module_str = " | ".join(filter(None, [
        "BSP" if use_sp else None,
        "LS" if use_ls else None,
        "Temporal" if use_ta else None,
    ])) or "Baseline (no attention)"
    report(1, "✓ Model dimuat ke memori.", f"Device: {device} | Modul: {module_str}")

    # ── 2. Ekstraksi Pose (MediaPipe) ─────────────────────────────────────────
    report(2, "Mengekstrak pose dari video mentah...",
           f"Membuka: {video_path.name}")

    raw_npy_path = output_dir / f"{stem}_raw.npy"
    extractor = PoseExtractor(model_complexity=2)
    extractor.extract_video(video_path=str(video_path), output_npy_path=str(raw_npy_path))

    _raw = np.load(str(raw_npy_path))
    report(2, "✓ Ekstraksi pose selesai.",
           f"Berhasil mengekstrak {_raw.shape[0]} frame × 33 sendi × 4 koordinat")

    # ── 3. Preprocessing & Inferensi AI ──────────────────────────────────────
    report(3, "Preprocessing & Inferensi AI...",
           f"Memampatkan {_raw.shape[0]} frame → 64 frame (shape: 64×33×3)")

    tensor_64_path = output_dir / f"{stem}_64.npy"
    preprocessor = DataPreprocessor(target_frames=64)
    tensor_data = preprocessor.process(
        npy_file_path=str(raw_npy_path),
        output_npy_path=str(tensor_64_path),
    )

    input_tensor = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0).to(device)

    # Forward pass via explainer (identik hasilnya dengan model(x))
    attn_out = forward_with_attention(model, input_tensor)
    logits   = attn_out["logits"]                        # (1, 64, 2)
    preds    = logits.argmax(dim=2)                       # (1, 64)
    probs    = torch.softmax(logits, dim=2)               # (1, 64, 2)

    preds_np = preds.squeeze().cpu().numpy()              # (64,)
    probs_np = probs.squeeze().cpu().numpy()              # (64, 2)
    n_benar  = int((preds_np == 0).sum())
    n_salah  = int((preds_np == 1).sum())

    report(3, "✓ Inferensi AI selesai.",
           f"{n_benar} frame BENAR + {n_salah} frame SALAH dari 64 frame total")

    # ── 4. Perturbation Joint Influence Attribution ───────────────────────────
    report(4, "Menghitung Joint Influence Attribution...",
           "Perturbation batched: 33 landmark × 64 frame → influence matrix (64×33)")

    influence, delta_prob = joint_influence(model, input_tensor, device)
    S       = sequence_joint_score(influence)
    ras_res = reference_attribution_share(S, influence, exercise)
    faith   = perturbation_faithfulness_check(model, input_tensor, S, device)

    report(4, "✓ Attribution selesai.",
           f"RAS={ras_res['RAS']:.3f} | Attribution Lift={ras_res['attribution_lift']:.2f}× | "
           f"Top joint: {ras_res['top5'][0]['name'] if ras_res['top5'] else 'N/A'}")

    # ── 5. Rendering Video + Export JSON ─────────────────────────────────────
    report(5, "Merender video heatmap dengan Joint Influence Attribution...",
           "Menggambar visualisasi per-frame berdasarkan attribution terkomputasi...")

    cap               = cv2.VideoCapture(str(video_path))
    total_frames_asli = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps               = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width             = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = output_dir / f"{stem}_explain_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    target_indices = np.linspace(0, total_frames_asli - 1, 64).astype(int)
    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                           min_detection_confidence=0.5)

    ref_joints = BIOMECHANICAL_REFERENCE.get(exercise, [])

    for i in range(64):
        idx_asli = target_indices[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_asli)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = pose.process(frame_rgb)

        pred_label = preds_np[i]
        prob_salah = probs_np[i, 1] * 100

        # Per-frame influence values
        inf_frame = influence[i]  # (33,)
        max_abs   = float(np.abs(inf_frame).max())
        if max_abs < 1e-8:
            max_abs = 1.0  # prevent div by zero

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # ── Skeleton edges ───────────────────────────────────────────────
            for conn in mp_pose.POSE_CONNECTIONS:
                p1, p2 = conn
                lm1, lm2 = landmarks[p1], landmarks[p2]
                if lm1.visibility > 0.3 and lm2.visibility > 0.3:
                    x1 = int(lm1.x * width);  y1 = int(lm1.y * height)
                    x2 = int(lm2.x * width);  y2 = int(lm2.y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (120, 120, 120), 2)

            # ── Joint Attribution Heatmap ────────────────────────────────────
            # Radius ∝ |influence[i, v]|
            # Merah  = mendukung SALAH (influence > 0)
            # Biru   = mendukung BENAR (influence < 0)
            # Abu    = netral (influence ≈ 0)
            MIN_R = 4;  MAX_R = 20

            for j, lm in enumerate(landmarks):
                if j >= 33 or lm.visibility < 0.3:
                    continue
                px = int(lm.x * width);  py = int(lm.y * height)

                inf_val = float(inf_frame[j])
                norm_abs = abs(inf_val) / max_abs  # 0..1

                radius = MIN_R + int(norm_abs * (MAX_R - MIN_R))

                # Warna attribution
                if inf_val > 0.01 * max_abs:
                    # Mendukung SALAH → merah
                    intensity = int(norm_abs * 200) + 55
                    color = (30, 30, intensity)          # BGR merah
                elif inf_val < -0.01 * max_abs:
                    # Mendukung BENAR → biru-hijau
                    intensity = int(norm_abs * 200) + 55
                    color = (intensity, intensity // 2, 30)  # BGR biru-hijau
                else:
                    color = (80, 80, 80)  # Netral

                # Glow untuk sendi dengan influence tinggi (>40%)
                if norm_abs > 0.4:
                    overlay = frame.copy()
                    cv2.circle(overlay, (px, py), radius + int(norm_abs * 8), color, -1)
                    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

                cv2.circle(frame, (px, py), radius, color, -1)

                # Biomechanical reference joints → tambahan outline cincin putih
                if j in ref_joints:
                    cv2.circle(frame, (px, py), radius + 4, (255, 255, 255), 1)

                cv2.circle(frame, (px, py), radius, (200, 200, 200), 1)

        # ── Panel atas ───────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (width, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame {i+1}/64",
                    (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{model_path.name}  |  Exercise: {exercise}",
                    (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 200, 255), 1, cv2.LINE_AA)

        # ── Panel bawah ──────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, height - 130), (width, height), (0, 0, 0), -1)
        if pred_label == 0:
            status = "BENAR  (Label 0)";  sc = (0, 230, 80)
        else:
            status = "SALAH  (Label 1)";  sc = (60, 60, 255)
        cv2.putText(frame, f"Prediksi AI: {status}",
                    (14, height - 82), cv2.FONT_HERSHEY_SIMPLEX, 1.1, sc, 3, cv2.LINE_AA)
        cv2.putText(frame, f"P(Salah): {prob_salah:.1f}%",
                    (14, height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (200, 210, 220), 2, cv2.LINE_AA)

        # Top-1 influential joint pada frame ini
        top1_joint = int(np.argmax(np.abs(inf_frame)))
        top1_inf   = float(inf_frame[top1_joint])
        top1_dir   = "-->SALAH" if top1_inf > 0 else "-->BENAR"
        cv2.putText(frame,
                    f"Top joint: {LANDMARK_NAMES[top1_joint]} ({top1_dir}, {abs(top1_inf):.3f})",
                    (14, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (170, 220, 180), 1, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    pose.close()

    # ── Export Explanation JSON ───────────────────────────────────────────────
    explanation = build_explanation_json(
        video_stem=stem,
        model_name=model_path.name,
        exercise=exercise,
        preds_np=preds_np,
        probs_np=probs_np,
        influence=influence,
        delta_prob=delta_prob,
        bsp_weights=attn_out["bsp_weights"],
        ls_weights=attn_out["ls_weights"],
        temporal_weights=attn_out["temporal_weights"],
        ras_result=ras_res,
        faithfulness=faith,
    )

    json_path = output_dir / f"{stem}_explanation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(explanation, f, ensure_ascii=False, indent=2)

    report(5, "✓ Video heatmap & JSON explanation selesai!",
           f"Video: {out_video_path.name} | JSON: {json_path.name} | "
           f"BENAR: {n_benar} | SALAH: {n_salah}")

    return {
        "video_path":  str(out_video_path),
        "json_path":   str(json_path),
        "n_benar":     n_benar,
        "n_salah":     n_salah,
        "explanation": explanation,
    }
