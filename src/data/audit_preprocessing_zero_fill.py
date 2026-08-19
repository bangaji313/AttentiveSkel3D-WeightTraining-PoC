# Release_V2_AttentiveSkel3D/src/data/audit_preprocessing_zero_fill.py
#
# ==============================================================================
# AUDIT PREPROCESSING & ZERO-FILL IMPACT ANALYSIS (487 DATASET VIDEOS)
# ==============================================================================
#
# Skrip audit komprehensif untuk memeriksa:
#   1. Deteksi landmark zero-filled / collapsed phantom coordinates pada 487 tensor.
#   2. Distribusi per jenis latihan (Bench Press, Deadlift, Squat) & camera view (frontal, lateral, sekunder).
#   3. Persentase zero-fill untuk setiap 33 landmark MediaPipe (0-32).
#   4. 10 video dengan persentase zero-fill tertinggi.
#   5. Identifikasi apakah zero-fill mengenai sendi kunci Biomechanical Validator:
#        - Squat: Shoulders (11,12), Hips (23,24), Knees (25,26), Ankles (27,28)
#        - Bench Press: Shoulders (11,12), Elbows (13,14), Wrists (15,16)
#        - Deadlift: Shoulders (11,12), Hips (23,24)
#   6. Jumlah frame label yang perhitungannya terdampak oleh zero-fill.
#   7. Klasifikasi dampak (Negligible / Moderate / Critical).
#
# TIDAK MENGUBAH DATASET, PIPELINE, ATAU MODEL APAPUN.

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

# ─── Konfigurasi Path ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TENSORS_DIR  = PROJECT_ROOT / "data" / "tensors"
LABELS_DIR   = PROJECT_ROOT / "data" / "v2_labels"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "src" / "data" / "manifest_v2.csv"
PEMETAAN_JSON = PROJECT_ROOT / "src" / "data" / "Pemetaan_Detail_Label_Per_Frame_V2.json"
OUTPUT_DIR   = PROJECT_ROOT / "hasil_evaluasi"

mp_pose = mp.solutions.pose
LM_NAMES = {lm.value: lm.name for lm in mp_pose.PoseLandmark}

# Sendi kunci per latihan yang digunakan Biomechanical Validator
VALIDATOR_JOINTS = {
    "squat": [11, 12, 23, 24, 25, 26, 27, 28],      # Shoulders, Hips, Knees, Ankles
    "benchpress": [11, 12, 13, 14, 15, 16],          # Shoulders, Elbows, Wrists
    "deadlift": [11, 12, 23, 24],                    # Shoulders, Hips (Spine vector)
}

def detect_camera_view(raw_video_name: str) -> str:
    """Ekstraksi camera view dari nama file video mentah."""
    name_lower = raw_video_name.lower()
    if "lateral" in name_lower or "side" in name_lower:
        return "lateral"
    elif "frontal" in name_lower or "front" in name_lower:
        return "frontal"
    elif "sekunder" in name_lower or "kaggle" in name_lower:
        return "sekunder_kaggle"
    else:
        return "unspecified"

def is_collapsed_landmark_frame(frame_33x3: np.ndarray, landmark_idx: int) -> bool:
    """
    Mendeteksi apakah suatu landmark pada frame tertentu merupakan titik zero-filled/collapsed.
    Karakteristik zero-fill pada spatial_normalize():
      1. Koordinatnya sama persis dengan landmark lain di frame yang sama (terutama jika cluster 25..32 sama).
      2. Atau jika z-coordinate sangat mendekati 0.0 dan koordinat x,y sama persis dengan minimal 1 landmark lain.
    """
    target = frame_33x3[landmark_idx]
    # Cek kesamaan identik dengan landmark lain di frame yang sama
    matches = 0
    for other_idx in range(33):
        if other_idx != landmark_idx:
            # Jika koordinat sama persis hingga toleransi numerik floating-point
            if np.allclose(target, frame_33x3[other_idx], atol=1e-5):
                matches += 1
    # Jika landmark ini identik dengan minimal 1 landmark lain, ini adalah cluster zero-fill
    return matches >= 1

def run_full_audit():
    print("=" * 80)
    print("  MEMULAI AUDIT PREPROCESSING & ZERO-FILL PADA 487 TENSOR DATASET")
    print("=" * 80)

    # 1. Muat manifest dan pemetaan JSON jika ada
    with open(PEMETAAN_JSON, "r", encoding="utf-8") as f:
        pemetaan_data = json.load(f)
    
    mapping_dict = {
        item["nama_dataset"]: item for item in pemetaan_data["daftar_pemetaan_per_video"]
    }

    manifest_df = pd.read_csv(MANIFEST_PATH)
    total_videos = len(manifest_df)
    print(f"[INFO] Total video terdaftar: {total_videos}")

    video_results = []
    landmark_zero_counts = np.zeros(33, dtype=int)
    total_landmark_frames = total_videos * 64

    # Metrik per kategori
    stats_by_exercise = {
        "benchpress": {"total_videos": 0, "total_lm_frames": 0, "zero_lm_frames": 0, "corrupted_val_frames": 0},
        "squat":      {"total_videos": 0, "total_lm_frames": 0, "zero_lm_frames": 0, "corrupted_val_frames": 0},
        "deadlift":   {"total_videos": 0, "total_lm_frames": 0, "zero_lm_frames": 0, "corrupted_val_frames": 0},
    }

    stats_by_view = {
        "frontal":         {"total_videos": 0, "total_lm_frames": 0, "zero_lm_frames": 0, "corrupted_val_frames": 0},
        "lateral":         {"total_videos": 0, "total_lm_frames": 0, "zero_lm_frames": 0, "corrupted_val_frames": 0},
        "sekunder_kaggle": {"total_videos": 0, "total_lm_frames": 0, "zero_lm_frames": 0, "corrupted_val_frames": 0},
        "unspecified":     {"total_videos": 0, "total_lm_frames": 0, "zero_lm_frames": 0, "corrupted_val_frames": 0},
    }

    total_frames_dataset = total_videos * 64
    total_corrupted_validator_frames = 0

    for _, row in tqdm(manifest_df.iterrows(), total=total_videos, desc="Mengaudit tensor"):
        stem = Path(row["file_path"]).stem
        tensor_path = TENSORS_DIR / f"{stem}.npy"
        label_path  = LABELS_DIR / f"{stem}_labels.npy"

        if not tensor_path.exists() or not label_path.exists():
            continue

        tensor = np.load(str(tensor_path)).astype(np.float32)  # (64, 33, 3)
        labels = np.load(str(label_path)).astype(int)          # (64,)

        exercise = str(row["exercise"]).lower()
        mapping_info = mapping_dict.get(stem, {})
        raw_video_name = mapping_info.get("nama_video_mentah", f"{stem}.mp4")
        camera_view = detect_camera_view(raw_video_name)

        # Cek zero-fill per frame dan per landmark secara vektorisasi
        # tensor: (64, 33, 3)
        # diffs: (64, 33, 33, 3) -> max abs diff: (64, 33, 33)
        diffs = np.abs(tensor[:, :, None, :] - tensor[:, None, :, :]).max(axis=-1)
        # Set diagonal to 1.0 agar tidak match dengan dirinya sendiri
        for t in range(64):
            np.fill_diagonal(diffs[t], 1.0)
        
        # zero_mask: (64, 33) True jika koordinat landmark sama persis dengan minimal 1 landmark lain
        zero_mask = (diffs < 1e-5).any(axis=2)

        # Tambahkan ke akumulator landmark
        landmark_zero_counts += zero_mask.sum(axis=0)

        video_zero_count = int(zero_mask.sum())
        video_total_lm   = 64 * 33
        video_zero_pct   = (video_zero_count / video_total_lm) * 100.0

        # Cek apakah zero-fill mengenai sendi yang dipakai validator latihan ini
        val_joints = VALIDATOR_JOINTS.get(exercise, [])
        # Mask boolean untuk frame yang memiliki >= 1 sendi validator zero-filled
        val_zero_per_frame = zero_mask[:, val_joints].any(axis=1) # (64,)
        corrupted_frames_count = int(val_zero_per_frame.sum())
        total_corrupted_validator_frames += corrupted_frames_count

        # Akumulasi statistik
        stats_by_exercise[exercise]["total_videos"] += 1
        stats_by_exercise[exercise]["total_lm_frames"] += video_total_lm
        stats_by_exercise[exercise]["zero_lm_frames"] += video_zero_count
        stats_by_exercise[exercise]["corrupted_val_frames"] += corrupted_frames_count

        stats_by_view[camera_view]["total_videos"] += 1
        stats_by_view[camera_view]["total_lm_frames"] += video_total_lm
        stats_by_view[camera_view]["zero_lm_frames"] += video_zero_count
        stats_by_view[camera_view]["corrupted_val_frames"] += corrupted_frames_count

        # Rekap per sendi kunci
        shoulder_zero = int(zero_mask[:, [11, 12]].sum())
        elbow_zero    = int(zero_mask[:, [13, 14]].sum())
        wrist_zero    = int(zero_mask[:, [15, 16]].sum())
        hip_zero      = int(zero_mask[:, [23, 24]].sum())
        knee_zero     = int(zero_mask[:, [25, 26]].sum())
        ankle_zero    = int(zero_mask[:, [27, 28]].sum())

        video_results.append({
            "nama_dataset": stem,
            "nama_video_mentah": raw_video_name,
            "jenis_latihan": exercise,
            "camera_view": camera_view,
            "total_lm_frames": video_total_lm,
            "total_zero_filled_lm": video_zero_count,
            "persentase_zero_fill_pct": round(video_zero_pct, 2),
            "frame_validator_terdampak": corrupted_frames_count,
            "persentase_frame_val_terdampak_pct": round((corrupted_frames_count / 64.0) * 100.0, 2),
            "zero_shoulder": shoulder_zero,
            "zero_elbow": elbow_zero,
            "zero_wrist": wrist_zero,
            "zero_hip": hip_zero,
            "zero_knee": knee_zero,
            "zero_ankle": ankle_zero,
        })

    # Simpan hasil detail ke CSV
    df_results = pd.DataFrame(video_results)
    csv_path = OUTPUT_DIR / "Audit_Preprocessing_Zero_Fill_V2.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[SUKSES] File CSV audit tersimpan: {csv_path}")

    # 10 video dengan zero-fill terbanyak
    top10_zero = df_results.sort_values(by="persentase_zero_fill_pct", ascending=False).head(10)

    # 10 video dengan corrupted validator frames terbanyak
    top10_val_corrupted = df_results.sort_values(by="frame_validator_terdampak", ascending=False).head(10)

    # Ringkasan per landmark MediaPipe
    landmark_summary = []
    for j in range(33):
        cnt = int(landmark_zero_counts[j])
        pct = (cnt / total_landmark_frames) * 100.0
        landmark_summary.append({
            "landmark_id": j,
            "landmark_name": LM_NAMES[j],
            "total_zero_filled_frames": cnt,
            "total_frames_dataset": total_landmark_frames,
            "persentase_zero_fill_pct": round(pct, 2)
        })

    # Susun Laporan JSON Lengkap
    audit_report = {
        "metadata": {
            "total_video_diaudit": total_videos,
            "total_frame_dataset": total_frames_dataset,
            "total_landmark_frame_dataset": total_videos * 64 * 33,
            "total_zero_filled_landmark_frames": int(df_results["total_zero_filled_lm"].sum()),
            "overall_zero_fill_rate_pct": round(df_results["total_zero_filled_lm"].sum() / (total_videos * 64 * 33) * 100.0, 2),
            "total_frames_with_corrupted_validator_joints": total_corrupted_validator_frames,
            "overall_corrupted_validator_frames_pct": round((total_corrupted_validator_frames / total_frames_dataset) * 100.0, 2),
        },
        "distribusi_per_latihan": {
            ex: {
                "total_video": v["total_videos"],
                "persentase_zero_fill_pct": round((v["zero_lm_frames"] / v["total_lm_frames"]) * 100.0, 2) if v["total_lm_frames"] > 0 else 0,
                "frame_validator_terdampak": v["corrupted_val_frames"],
                "persentase_frame_val_terdampak_pct": round((v["corrupted_val_frames"] / (v["total_videos"] * 64)) * 100.0, 2) if v["total_videos"] > 0 else 0,
            } for ex, v in stats_by_exercise.items()
        },
        "distribusi_per_camera_view": {
            view: {
                "total_video": v["total_videos"],
                "persentase_zero_fill_pct": round((v["zero_lm_frames"] / v["total_lm_frames"]) * 100.0, 2) if v["total_lm_frames"] > 0 else 0,
                "frame_validator_terdampak": v["corrupted_val_frames"],
                "persentase_frame_val_terdampak_pct": round((v["corrupted_val_frames"] / (v["total_videos"] * 64)) * 100.0, 2) if v["total_videos"] > 0 else 0,
            } for view, v in stats_by_view.items() if v["total_videos"] > 0
        },
        "top_10_video_zero_fill_terbanyak": top10_zero[[
            "nama_dataset", "nama_video_mentah", "jenis_latihan", "camera_view", "persentase_zero_fill_pct", "frame_validator_terdampak"
        ]].to_dict(orient="records"),
        "top_10_video_validator_terdampak_terbanyak": top10_val_corrupted[[
            "nama_dataset", "nama_video_mentah", "jenis_latihan", "camera_view", "persentase_zero_fill_pct", "frame_validator_terdampak"
        ]].to_dict(orient="records"),
        "rekapitulasi_33_landmark": landmark_summary
    }

    json_path = OUTPUT_DIR / "Audit_Preprocessing_Zero_Fill_V2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"[SUKSES] File JSON audit tersimpan: {json_path}")

    # Print Ringkasan ke Terminal
    print("\n" + "=" * 80)
    print("  RINGKASAN HASIL AUDIT KUANTITATIF PREPROCESSING & ZERO-FILL")
    print("=" * 80)
    meta = audit_report["metadata"]
    print(f"Total Video Diaudit               : {meta['total_video_diaudit']} video")
    print(f"Total Frame Dataset               : {meta['total_frame_dataset']:,} frame")
    print(f"Total Landmark-Frame              : {meta['total_landmark_frame_dataset']:,} titik")
    print(f"Total Zero-Filled Landmark-Frame  : {meta['total_zero_filled_landmark_frames']:,} titik ({meta['overall_zero_fill_rate_pct']}%)")
    print(f"Total Frame Validator Terdampak   : {meta['total_frames_with_corrupted_validator_joints']:,} / {meta['total_frame_dataset']:,} frame ({meta['overall_corrupted_validator_frames_pct']}%)")

    print("\n--- DISTRIBUSI BERDASARKAN GERAKAN ---")
    for ex, d in audit_report["distribusi_per_latihan"].items():
        print(f"  {ex.capitalize():12s}: {d['total_video']:3d} video | Zero-Fill: {d['persentase_zero_fill_pct']:5.2f}% | Validator Terdampak: {d['frame_validator_terdampak']:5d} frame ({d['persentase_frame_val_terdampak_pct']:5.2f}%)")

    print("\n--- DISTRIBUSI BERDASARKAN CAMERA VIEW ---")
    for view, d in audit_report["distribusi_per_camera_view"].items():
        print(f"  {view.capitalize():15s}: {d['total_video']:3d} video | Zero-Fill: {d['persentase_zero_fill_pct']:5.2f}% | Validator Terdampak: {d['frame_validator_terdampak']:5d} frame ({d['persentase_frame_val_terdampak_pct']:5.2f}%)")

    print("\n--- TOP 10 LANDMARK PALING BANYAK ZERO-FILLED ---")
    sorted_lm = sorted(landmark_summary, key=lambda x: x["persentase_zero_fill_pct"], reverse=True)
    for item in sorted_lm[:10]:
        print(f"  [{item['landmark_id']:2d}] {item['landmark_name']:22s}: {item['total_zero_filled_frames']:6d} frame ({item['persentase_zero_fill_pct']:5.2f}%)")

    print("\n--- STATUS SENDI UTAMA BIOMECHANICAL VALIDATOR ---")
    for j in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        item = landmark_summary[j]
        print(f"  [{item['landmark_id']:2d}] {item['landmark_name']:22s}: {item['total_zero_filled_frames']:6d} frame ({item['persentase_zero_fill_pct']:5.2f}%)")

    print("=" * 80)

if __name__ == "__main__":
    run_full_audit()
