# Release_V2_AttentiveSkel3D/src/data/generate_pemetaan_detail_per_frame_v2.py
#
# ==============================================================================
# PEMBANGKIT DOKUMENTASI PEMETAAN DATASET DETAIL ONE-ROW-PER-FRAME (31.168 ROWS)
# ==============================================================================
#
# Menghasilkan:
#   1. Release_V2_AttentiveSkel3D/hasil_evaluasi/Pemetaan_Detail_Per_Frame_V2.csv
#   2. Release_V2_AttentiveSkel3D/hasil_evaluasi/Pemetaan_Detail_Per_Frame_V2.json
#   3. Release_V2_AttentiveSkel3D/src/data/Pemetaan_Detail_Per_Frame_V2.csv
#   4. Release_V2_AttentiveSkel3D/src/data/Pemetaan_Detail_Per_Frame_V2.json
#
# Spesifikasi:
#   - Tepat 487 video x 64 frame = 31.168 baris data (di luar header).
#   - Sumber label: Pemetaan_Detail_Label_Per_Frame_V2.csv (Array_Label_64_Biner).
#   - Mapping temporal: Temporal_Index (1-64) -> Source_Frame_Index & Timestamp_Sec
#     berdasarkan formula temporal_resample aktual (interp1d linear linspace)
#     dan FPS video mentah aktual.
#   - Quality Assurance: Validasi 100% kecocokan dengan data video-level eksisting.

import os
import sys
import json
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── Konfigurasi Direktori ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
TENSORS_DIR  = PROJECT_ROOT / "data" / "tensors"
LABELS_DIR   = PROJECT_ROOT / "data" / "v2_labels"
SOURCE_CSV   = PROJECT_ROOT / "hasil_evaluasi" / "Pemetaan_Detail_Label_Per_Frame_V2.csv"
OUTPUT_DIR_EVAL = PROJECT_ROOT / "hasil_evaluasi"
OUTPUT_DIR_DATA = PROJECT_ROOT / "src" / "data"

OUTPUT_DIR_EVAL.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_DATA.mkdir(parents=True, exist_ok=True)


def parse_metadata_from_filename(filename: str, source_category: str) -> dict:
    """Ekstraksi metadata terstruktur dari nama file video mentah tanpa mengarang."""
    name_lower = filename.lower().replace(".mp4", "")
    
    # 1. Camera View
    if "lateral" in name_lower or "side" in name_lower:
        view = "Lateral"
    elif "frontal" in name_lower or "front" in name_lower:
        view = "Frontal"
    elif "sekunder" in name_lower or "kaggle" in name_lower:
        view = "Sekunder/Varied"
    else:
        view = "Frontal" if source_category == "Primer" else "Unspecified"

    # 2. Subject ID
    subjek_match = re.search(r"subjek\d+", name_lower)
    if subjek_match:
        subject_id = subjek_match.group(0)
    elif "kaggle" in name_lower:
        kaggle_id = re.search(r"kaggle\d+", name_lower)
        subject_id = f"Kaggle_Hasyim_{kaggle_id.group(0)}" if kaggle_id else "Kaggle_Hasyim"
    else:
        subject_id = "Unknown"

    # 3. Repetition ID
    rep_match = re.search(r"rep\d+", name_lower)
    repetition_id = rep_match.group(0) if rep_match else "rep_unspecified"

    return {
        "Camera_View": view,
        "Subject_ID": subject_id,
        "Repetition_ID": repetition_id
    }


def main():
    print("=" * 80)
    print("  MEMULAI PEMBUATAN PEMETAAN DETAIL ONE-ROW-PER-FRAME (31.168 BARIS)")
    print("=" * 80)

    # 1. Baca master video-level CSV sumber
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"File sumber tidak ditemukan: {SOURCE_CSV}")

    df_source = pd.read_csv(SOURCE_CSV)
    total_videos = len(df_source)
    print(f"[INFO] Membaca {total_videos} video dari: {SOURCE_CSV}")

    # 2. Bangun map path file video mentah di disk
    print("[INFO] Memindai file video mentah di data/raw/ ...")
    raw_files_map = {}
    for f in RAW_DIR.rglob("*.mp4"):
        raw_files_map[f.name] = f
    print(f"[INFO] Terindeks {len(raw_files_map)} file video mentah di disk.")

    # 3. Ekstraksi dan Bangun Dataset Per-Frame
    per_frame_rows = []
    json_hierarchical_data = []

    qa_errors = []
    global_benar_count = 0
    global_salah_count = 0

    for idx, row in tqdm(df_source.iterrows(), total=total_videos, desc="Memproses 487 video"):
        dataset_name = str(row["Nama_Dataset"]).strip()
        raw_filename = str(row["Nama_Video_Mentah"]).strip()
        exercise     = str(row["Jenis_Latihan"]).strip()
        kategori_src = str(row["Kategori_Sumber"]).strip()
        tensor_path  = str(row["Path_Tensor"]).strip()
        label_path   = str(row["Path_Label"]).strip()
        bin_str      = str(row["Array_Label_64_Biner"]).strip()

        # Validasi panjang string biner
        if len(bin_str) != 64 or not set(bin_str).issubset({"0", "1"}):
            qa_errors.append(f"Format Array_Label_64_Biner tidak valid untuk {dataset_name}: '{bin_str}'")
            continue

        # Parsing metadata tambahan
        meta_info = parse_metadata_from_filename(raw_filename, kategori_src)

        # Temukan file video mentah untuk hitung source frame mapping aktual
        v_path = raw_files_map.get(raw_filename)
        if v_path is not None and v_path.exists():
            cap = cv2.VideoCapture(str(v_path))
            if cap.isOpened():
                t_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if fps <= 0:
                    fps = 30.0
                cap.release()
                has_video_meta = True
            else:
                has_video_meta = False
                t_raw, fps = 64, 30.0
        else:
            has_video_meta = False
            t_raw, fps = 64, 30.0

        video_frame_list = []
        vid_benar_count = 0
        vid_salah_count = 0

        for t_idx in range(64):
            # Temporal_Index 1-indexed (1..64)
            temporal_index_display = t_idx + 1

            # Nilai label biner
            bin_val = int(bin_str[t_idx])
            status_str = "BENAR" if bin_val == 0 else "SALAH"

            if bin_val == 0:
                vid_benar_count += 1
                global_benar_count += 1
            else:
                vid_salah_count += 1
                global_salah_count += 1

            # Hitung mapping source frame index dan timestamp aktual
            if has_video_meta:
                # Formula identik temporal_resample (interp1d linspace 0..T_raw-1)
                src_frame_idx = int(np.round(t_idx * (t_raw - 1) / 63.0))
                timestamp_sec = round(src_frame_idx / fps, 4)
                src_frame_val = src_frame_idx
                timestamp_val = timestamp_sec
            else:
                src_frame_val = "NA"
                timestamp_val = "NA"

            row_data = {
                "Nama_Dataset": dataset_name,
                "Nama_Video_Mentah": raw_filename,
                "Latihan": exercise,
                "Kategori_Sumber": kategori_src,
                "Camera_View": meta_info["Camera_View"],
                "Subject_ID": meta_info["Subject_ID"],
                "Repetition_ID": meta_info["Repetition_ID"],
                "Temporal_Index": temporal_index_display,
                "Label_Biner": bin_val,
                "Status_Label": status_str,
                "Source_Frame_Index": src_frame_val,
                "Timestamp_Sec": timestamp_val,
                "Tensor_Path": tensor_path,
                "Label_Path": label_path
            }

            per_frame_rows.append(row_data)
            video_frame_list.append({
                "Temporal_Index": temporal_index_display,
                "Label_Biner": bin_val,
                "Status_Label": status_str,
                "Source_Frame_Index": src_frame_val,
                "Timestamp_Sec": timestamp_val
            })

        # QA Check per video terhadap sumber
        src_benar = int(row["Jumlah_Frame_Benar"])
        src_salah = int(row["Jumlah_Frame_Salah"])
        if vid_benar_count != src_benar or vid_salah_count != src_salah:
            qa_errors.append(
                f"Mismatch count pada {dataset_name}: Terhitung B={vid_benar_count}, S={vid_salah_count} "
                f"vs Sumber B={src_benar}, S={src_salah}"
            )

        json_hierarchical_data.append({
            "Nama_Dataset": dataset_name,
            "Nama_Video_Mentah": raw_filename,
            "Latihan": exercise,
            "Kategori_Sumber": kategori_src,
            "Camera_View": meta_info["Camera_View"],
            "Subject_ID": meta_info["Subject_ID"],
            "Repetition_ID": meta_info["Repetition_ID"],
            "Total_Frame": 64,
            "Jumlah_Frame_Benar": vid_benar_count,
            "Jumlah_Frame_Salah": vid_salah_count,
            "Tensor_Path": tensor_path,
            "Label_Path": label_path,
            "Frames": video_frame_list
        })

    # 4. Bangun DataFrame Final
    df_per_frame = pd.DataFrame(per_frame_rows)

    # 5. QA Check Komprehensif
    print("\n" + "=" * 80)
    print("  MENJALANKAN QUALITY ASSURANCE (QA) CHECK")
    print("=" * 80)

    total_rows = len(df_per_frame)
    unique_videos = df_per_frame["Nama_Dataset"].nunique()
    print(f"1. Total Baris Data         : {total_rows:,} baris (Target: 31,168) -> {'PASS' if total_rows == 31168 else 'FAIL'}")
    print(f"2. Total Video Unik         : {unique_videos} video (Target: 487)    -> {'PASS' if unique_videos == 487 else 'FAIL'}")

    # Check duplicates (Nama_Dataset, Temporal_Index)
    dups = df_per_frame.duplicated(subset=["Nama_Dataset", "Temporal_Index"]).sum()
    print(f"3. Duplikasi Baris          : {dups} duplikat (Target: 0)          -> {'PASS' if dups == 0 else 'FAIL'}")

    # Check temporal indices range per video
    temp_min = df_per_frame["Temporal_Index"].min()
    temp_max = df_per_frame["Temporal_Index"].max()
    print(f"4. Rentang Temporal Index   : {temp_min} s/d {temp_max} (Target: 1 s/d 64)   -> {'PASS' if temp_min == 1 and temp_max == 64 else 'FAIL'}")

    # Check label biner validity
    unique_labels = set(df_per_frame["Label_Biner"].unique())
    print(f"5. Himpunan Label Biner     : {unique_labels} (Target: {{0, 1}})        -> {'PASS' if unique_labels == {0, 1} else 'FAIL'}")

    # Check consistency Label_Biner <-> Status_Label
    mismatch_labels = df_per_frame[
        ((df_per_frame["Label_Biner"] == 0) & (df_per_frame["Status_Label"] != "BENAR")) |
        ((df_per_frame["Label_Biner"] == 1) & (df_per_frame["Status_Label"] != "SALAH"))
    ]
    print(f"6. Konsistensi Biner-Status : {len(mismatch_labels)} mismatch (Target: 0)       -> {'PASS' if len(mismatch_labels) == 0 else 'FAIL'}")

    # Check 100% video-level match with source
    print(f"7. Match dengan File Sumber : {len(qa_errors)} error (Target: 0/487)     -> {'PASS' if len(qa_errors) == 0 else 'FAIL'}")

    if qa_errors:
        print(f"[FATAL QA ERROR] Ditemukan {len(qa_errors)} ketidaksesuaian:")
        for err in qa_errors[:10]:
            print(f"  • {err}")
        return

    # 6. Simpan Output CSV
    csv_eval_path = OUTPUT_DIR_EVAL / "Pemetaan_Detail_Per_Frame_V2.csv"
    csv_data_path = OUTPUT_DIR_DATA / "Pemetaan_Detail_Per_Frame_V2.csv"
    df_per_frame.to_csv(csv_eval_path, index=False)
    df_per_frame.to_csv(csv_data_path, index=False)
    print(f"\n[SUKSES] CSV tersimpan:")
    print(f"  -> {csv_eval_path}")
    print(f"  -> {csv_data_path}")

    # 7. Simpan Output JSON
    json_eval_path = OUTPUT_DIR_EVAL / "Pemetaan_Detail_Per_Frame_V2.json"
    json_data_path = OUTPUT_DIR_DATA / "Pemetaan_Detail_Per_Frame_V2.json"
    
    json_output_payload = {
        "metadata": {
            "total_video": unique_videos,
            "total_frame_records": total_rows,
            "total_frame_benar": global_benar_count,
            "total_frame_salah": global_salah_count,
            "persentase_benar_pct": round((global_benar_count / total_rows) * 100.0, 2),
            "persentase_salah_pct": round((global_salah_count / total_rows) * 100.0, 2),
            "deskripsi": "Dokumentasi Pemetaan Dataset One-Row-Per-Frame (64 Frame per Video)"
        },
        "daftar_video": json_hierarchical_data
    }

    with open(json_eval_path, "w", encoding="utf-8") as f:
        json.dump(json_output_payload, f, indent=2)
    with open(json_data_path, "w", encoding="utf-8") as f:
        json.dump(json_output_payload, f, indent=2)
    print(f"[SUKSES] JSON tersimpan:")
    print(f"  -> {json_eval_path}")
    print(f"  -> {json_data_path}")

    # 8. Tampilkan Preview 10 Baris Pertama
    print("\n" + "=" * 80)
    print("  PREVIEW 10 BARIS DATA (SAMPEL BENCHPRESS_001)")
    print("=" * 80)
    preview_cols = ["Nama_Dataset", "Latihan", "Temporal_Index", "Label_Biner", "Status_Label", "Source_Frame_Index", "Timestamp_Sec", "Subject_ID"]
    print(df_per_frame[preview_cols].head(10).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
