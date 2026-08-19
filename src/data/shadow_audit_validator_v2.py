# Release_V2_AttentiveSkel3D/src/data/shadow_audit_validator_v2.py
#
# ==============================================================================
# SHADOW AUDIT BIOMECHANICAL VALIDATOR V1 VS V2 (31.168 FRAMES)
# ==============================================================================
#
# Skrip audit perbandingan label Ground Truth antara:
#   - V1 (Eksisting): Evaluasi biner bereksekusi langsung pada tensor normalisasi
#     (frame dengan landmark zero-fill otomatis gagal/terdistorsi -> SALAH)
#   - V2 (Shadow Rule): Validasi bersyarat. Metrik biomekanik HANYA dihitung jika
#     seluruh sendi yang disyaratkan valid. Jika sendi invalid -> INVALID_POSE.
#
# Ketentuan:
#   1. Tidak melakukan overwrite terhadap dataset, manifest, atau model lama.
#   2. 3 kemungkinan status V2: BENAR, SALAH, INVALID_POSE.
#   3. Membandingkan seluruh 31.168 frame dari 487 video.
#   4. Breakdown: Bench Press, Squat, Deadlift, Primer Frontal, Primer Lateral, Sekunder.
#   5. Menampilkan 20 contoh transisi SALAH -> INVALID_POSE dengan referensi video mentah.
#   6. Output:
#        - Release_V2_AttentiveSkel3D/hasil_evaluasi/Label_Audit_V1_vs_V2.csv
#        - Release_V2_AttentiveSkel3D/hasil_evaluasi/Label_Audit_V1_vs_V2.json

import os
import sys
import json
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tambahkan PROJECT_ROOT ke sys.path untuk import validator
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.biomechanics_validator import BiomechanicalValidator

mp_pose = mp.solutions.pose
LM_NAMES = {lm.value: lm.name for lm in mp_pose.PoseLandmark}

# Definisi sendi prasyarat per latihan
REQUIRED_JOINTS = {
    "squat": [11, 12, 23, 24, 25, 26, 27, 28],      # Shoulders, Hips, Knees, Ankles
    "benchpress": [11, 12, 13, 14, 15, 16],          # Shoulders, Elbows, Wrists
    "deadlift": [11, 12, 23, 24],                    # Shoulders, Hips
}

def detect_camera_view(raw_video_name: str) -> str:
    name_lower = raw_video_name.lower()
    if "lateral" in name_lower or "side" in name_lower:
        return "Primer Lateral"
    elif "frontal" in name_lower or "front" in name_lower:
        return "Primer Frontal"
    elif "sekunder" in name_lower or "kaggle" in name_lower:
        return "Sekunder"
    else:
        return "Primer Frontal"  # default fallback jika primer

def main():
    print("=" * 80)
    print("  MEMULAI SHADOW AUDIT BIOMECHANICAL VALIDATOR V1 VS V2 (31.168 FRAMES)")
    print("=" * 80)

    validator = BiomechanicalValidator()

    # 1. Muat pemetaan JSON
    with open(PEMETAAN_JSON, "r", encoding="utf-8") as f:
        pemetaan_data = json.load(f)
    
    mapping_dict = {
        item["nama_dataset"]: item for item in pemetaan_data["daftar_pemetaan_per_video"]
    }

    manifest_df = pd.read_csv(MANIFEST_PATH)
    total_videos = len(manifest_df)
    print(f"[INFO] Total video terdaftar: {total_videos} video")

    frame_audit_records = []
    
    # Kategori breakdown
    breakdown_stats = {
        "overall": {"total": 0, "unchanged_BENAR": 0, "unchanged_SALAH": 0, "BENAR_to_SALAH": 0, "SALAH_to_BENAR": 0, "BENAR_to_INVALID": 0, "SALAH_to_INVALID": 0},
        "exercise": {
            "benchpress": {"total": 0, "unchanged_BENAR": 0, "unchanged_SALAH": 0, "BENAR_to_SALAH": 0, "SALAH_to_BENAR": 0, "BENAR_to_INVALID": 0, "SALAH_to_INVALID": 0},
            "squat":      {"total": 0, "unchanged_BENAR": 0, "unchanged_SALAH": 0, "BENAR_to_SALAH": 0, "SALAH_to_BENAR": 0, "BENAR_to_INVALID": 0, "SALAH_to_INVALID": 0},
            "deadlift":   {"total": 0, "unchanged_BENAR": 0, "unchanged_SALAH": 0, "BENAR_to_SALAH": 0, "SALAH_to_BENAR": 0, "BENAR_to_INVALID": 0, "SALAH_to_INVALID": 0},
        },
        "view": {
            "Primer Frontal": {"total": 0, "unchanged_BENAR": 0, "unchanged_SALAH": 0, "BENAR_to_SALAH": 0, "SALAH_to_BENAR": 0, "BENAR_to_INVALID": 0, "SALAH_to_INVALID": 0},
            "Primer Lateral": {"total": 0, "unchanged_BENAR": 0, "unchanged_SALAH": 0, "BENAR_to_SALAH": 0, "SALAH_to_BENAR": 0, "BENAR_to_INVALID": 0, "SALAH_to_INVALID": 0},
            "Sekunder":       {"total": 0, "unchanged_BENAR": 0, "unchanged_SALAH": 0, "BENAR_to_SALAH": 0, "SALAH_to_BENAR": 0, "BENAR_to_INVALID": 0, "SALAH_to_INVALID": 0},
        }
    }

    salah_to_invalid_examples = []

    for _, row in tqdm(manifest_df.iterrows(), total=total_videos, desc="Mengaudit 31.168 frame"):
        stem = Path(row["file_path"]).stem
        tensor_path = TENSORS_DIR / f"{stem}.npy"
        label_path  = LABELS_DIR / f"{stem}_labels.npy"

        if not tensor_path.exists() or not label_path.exists():
            continue

        tensor = np.load(str(tensor_path)).astype(np.float32)  # (64, 33, 3)
        labels_v1 = np.load(str(label_path)).astype(int)       # (64,)

        exercise = str(row["exercise"]).lower()
        mapping_info = mapping_dict.get(stem, {})
        raw_video_name = mapping_info.get("nama_video_mentah", f"{stem}.mp4")
        view_category  = detect_camera_view(raw_video_name)

        # Deteksi zero-fill per frame dan per landmark
        # diffs: (64, 33, 33)
        diffs = np.abs(tensor[:, :, None, :] - tensor[:, None, :, :]).max(axis=-1)
        for t in range(64):
            np.fill_diagonal(diffs[t], 1.0)
        zero_mask = (diffs < 1e-5).any(axis=2)  # (64, 33)

        req_joints = REQUIRED_JOINTS.get(exercise, [])

        for t in range(64):
            frame_tensor = tensor[t : t + 1, :, :]  # (1, 33, 3)
            val_v1 = int(labels_v1[t])
            status_v1 = "BENAR" if val_v1 == 0 else "SALAH"

            # Cek apakah ada sendi prasyarat yang zero-filled di frame ini
            invalid_joints_in_frame = [j for j in req_joints if zero_mask[t, j]]

            if len(invalid_joints_in_frame) > 0:
                # Sendi yang dibutuhkan tidak valid -> status V2 = INVALID_POSE
                status_v2 = "INVALID_POSE"
                reason_v2 = f"Invalid/Zero-filled joints: {[LM_NAMES[j] for j in invalid_joints_in_frame]}"
            else:
                # Semua sendi prasyarat valid -> hitung biomechanical metric aktual
                if exercise == "squat":
                    is_valid, reason = validator.validate_squat(frame_tensor)
                elif exercise == "benchpress":
                    is_valid, reason = validator.validate_benchpress(frame_tensor)
                elif exercise == "deadlift":
                    is_valid, reason = validator.validate_deadlift(frame_tensor)
                else:
                    is_valid, reason = False, "Unknown exercise"
                
                status_v2 = "BENAR" if is_valid else "SALAH"
                reason_v2 = reason

            # Tentukan kategori transisi
            if status_v1 == "BENAR" and status_v2 == "BENAR":
                transition = "unchanged BENAR"
            elif status_v1 == "SALAH" and status_v2 == "SALAH":
                transition = "unchanged SALAH"
            elif status_v1 == "BENAR" and status_v2 == "SALAH":
                transition = "BENAR->SALAH"
            elif status_v1 == "SALAH" and status_v2 == "BENAR":
                transition = "SALAH->BENAR"
            elif status_v1 == "BENAR" and status_v2 == "INVALID_POSE":
                transition = "BENAR->INVALID"
            elif status_v1 == "SALAH" and status_v2 == "INVALID_POSE":
                transition = "SALAH->INVALID"
            else:
                transition = "OTHER"

            # Akumulasi statistik
            breakdown_stats["overall"]["total"] += 1
            breakdown_stats["overall"][transition.replace("->", "_to_").replace(" ", "_")] += 1

            breakdown_stats["exercise"][exercise]["total"] += 1
            breakdown_stats["exercise"][exercise][transition.replace("->", "_to_").replace(" ", "_")] += 1

            breakdown_stats["view"][view_category]["total"] += 1
            breakdown_stats["view"][view_category][transition.replace("->", "_to_").replace(" ", "_")] += 1

            # Kumpulkan contoh SALAH -> INVALID_POSE
            if transition == "SALAH->INVALID" and len(salah_to_invalid_examples) < 20:
                salah_to_invalid_examples.append({
                    "sample_no": len(salah_to_invalid_examples) + 1,
                    "nama_dataset": stem,
                    "nama_video_mentah": raw_video_name,
                    "jenis_latihan": exercise,
                    "camera_view": view_category,
                    "frame_temporal_idx_0_63": t,
                    "frame_temporal_display_1_64": t + 1,
                    "status_v1": status_v1,
                    "status_v2": status_v2,
                    "transisi": transition,
                    "sendi_invalid": [LM_NAMES[j] for j in invalid_joints_in_frame],
                    "keterangan": reason_v2
                })

            # Record per baris
            frame_audit_records.append({
                "nama_dataset": stem,
                "nama_video_mentah": raw_video_name,
                "jenis_latihan": exercise,
                "camera_view": view_category,
                "frame_temporal_idx": t,
                "status_v1": status_v1,
                "status_v2": status_v2,
                "transisi": transition,
                "alasan_v2": reason_v2
            })

    # ── 2. Simpan CSV Detail ──────────────────────────────────────────────────
    df_all = pd.DataFrame(frame_audit_records)
    csv_path = OUTPUT_DIR / "Label_Audit_V1_vs_V2.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n[SUKSES] File CSV tersimpan: {csv_path}")

    # ── 3. Susun JSON Laporan Komprehensif ────────────────────────────────────
    json_report = {
        "metadata": {
            "total_video": total_videos,
            "total_frames_audited": len(df_all),
            "deskripsi": "Perbandingan label V1 (biner paksa) vs V2 Shadow Validator (3 status: BENAR, SALAH, INVALID_POSE)"
        },
        "ringkasan_transisi_global": {
            "total_frame": breakdown_stats["overall"]["total"],
            "unchanged_BENAR": breakdown_stats["overall"]["unchanged_BENAR"],
            "unchanged_SALAH": breakdown_stats["overall"]["unchanged_SALAH"],
            "BENAR_to_SALAH": breakdown_stats["overall"]["BENAR_to_SALAH"],
            "SALAH_to_BENAR": breakdown_stats["overall"]["SALAH_to_BENAR"],
            "BENAR_to_INVALID": breakdown_stats["overall"]["BENAR_to_INVALID"],
            "SALAH_to_INVALID": breakdown_stats["overall"]["SALAH_to_INVALID"],
            "persentase_SALAH_to_INVALID_pct": round((breakdown_stats["overall"]["SALAH_to_INVALID"] / breakdown_stats["overall"]["total"]) * 100.0, 2),
            "persentase_total_unchanged_pct": round(((breakdown_stats["overall"]["unchanged_BENAR"] + breakdown_stats["overall"]["unchanged_SALAH"]) / breakdown_stats["overall"]["total"]) * 100.0, 2),
        },
        "breakdown_per_exercise": breakdown_stats["exercise"],
        "breakdown_per_view": breakdown_stats["view"],
        "20_contoh_SALAH_to_INVALID": salah_to_invalid_examples
    }

    json_path = OUTPUT_DIR / "Label_Audit_V1_vs_V2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
    print(f"[SUKSES] File JSON tersimpan: {json_path}")

    # ── 4. Print Ringkasan Rapi ke Console ────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RINGKASAN HASIL AUDIT SHADOW VALIDATOR V1 VS V2")
    print("=" * 80)
    g = json_report["ringkasan_transisi_global"]
    print(f"Total Frame Diaudit          : {g['total_frame']:,} frame")
    print(f"  • Unchanged BENAR          : {g['unchanged_BENAR']:,} frame ({g['unchanged_BENAR']/g['total_frame']*100:.2f}%)")
    print(f"  • Unchanged SALAH          : {g['unchanged_SALAH']:,} frame ({g['unchanged_SALAH']/g['total_frame']*100:.2f}%)")
    print(f"  • BENAR -> SALAH           : {g['BENAR_to_SALAH']:,} frame ({g['BENAR_to_SALAH']/g['total_frame']*100:.2f}%)")
    print(f"  • SALAH -> BENAR           : {g['SALAH_to_BENAR']:,} frame ({g['SALAH_to_BENAR']/g['total_frame']*100:.2f}%)")
    print(f"  • BENAR -> INVALID_POSE    : {g['BENAR_to_INVALID']:,} frame ({g['BENAR_to_INVALID']/g['total_frame']*100:.2f}%)")
    print(f"  • SALAH -> INVALID_POSE    : {g['SALAH_to_INVALID']:,} frame ({g['SALAH_to_INVALID']/g['total_frame']*100:.2f}%)")
    print(f"Total Label Bersih/Tetap     : {g['unchanged_BENAR'] + g['unchanged_SALAH']:,} frame ({g['persentase_total_unchanged_pct']}%)")

    print("\n--- BREAKDOWN PER LATIHAN ---")
    for ex, d in json_report["breakdown_per_exercise"].items():
        tot = d["total"]
        print(f"[{ex.upper()}] (Total: {tot:,} frame)")
        print(f"   Unchanged BENAR: {d['unchanged_BENAR']:5d} | Unchanged SALAH: {d['unchanged_SALAH']:5d}")
        print(f"   SALAH -> INVALID: {d['SALAH_to_INVALID']:5d} ({d['SALAH_to_INVALID']/tot*100:5.2f}%) | BENAR -> INVALID: {d['BENAR_to_INVALID']:5d}")

    print("\n--- BREAKDOWN PER CAMERA VIEW ---")
    for vw, d in json_report["breakdown_per_view"].items():
        tot = d["total"]
        print(f"[{vw}] (Total: {tot:,} frame)")
        print(f"   Unchanged BENAR: {d['unchanged_BENAR']:5d} | Unchanged SALAH: {d['unchanged_SALAH']:5d}")
        print(f"   SALAH -> INVALID: {d['SALAH_to_INVALID']:5d} ({d['SALAH_to_INVALID']/tot*100:5.2f}%) | BENAR -> INVALID: {d['BENAR_to_INVALID']:5d}")

    print("\n" + "=" * 80)
    print("  20 CONTOH FRAME TRANSISI: SALAH (V1) -> INVALID_POSE (V2)")
    print("=" * 80)
    for ex_item in salah_to_invalid_examples:
        print(f"#{ex_item['sample_no']:2d} | {ex_item['nama_dataset']:15s} | Frame {ex_item['frame_temporal_display_1_64']:2d}/64 | {ex_item['jenis_latihan']:10s} ({ex_item['camera_view']})")
        print(f"     Video Mentah : {ex_item['nama_video_mentah']}")
        print(f"     Sendi Rusak  : {', '.join(ex_item['sendi_invalid'])}")
        print()

    print("=" * 80)

if __name__ == "__main__":
    main()
