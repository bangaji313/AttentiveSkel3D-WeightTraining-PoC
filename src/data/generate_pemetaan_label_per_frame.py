# Release_V2_AttentiveSkel3D/src/data/generate_pemetaan_label_per_frame.py
#
# Skrip Pembangkit Dokumentasi Pemetaan Detail Label Per-Frame
# Menghasilkan file JSON dan CSV yang secara eksplisit memetakan:
#   1. Nama dataset (contoh: Squat_001)
#   2. Nama file video mentah asli (contoh: primer_squat_frontal_subjek01_rep1.mp4)
#   3. Jenis latihan & kategori sumber (Primer / Sekunder)
#   4. Daftar indeks frame yang salah (1-indexed)
#   5. Rentang/segmen frame salah (contoh: Frame 1-10, Frame 59-63)
#   6. Seluruh urutan array 64 label biner
#
# Output:
#   - Release_V2_AttentiveSkel3D/hasil_evaluasi/Pemetaan_Detail_Label_Per_Frame_V2.json
#   - Release_V2_AttentiveSkel3D/hasil_evaluasi/Pemetaan_Detail_Label_Per_Frame_V2.csv
#   - Release_V2_AttentiveSkel3D/src/data/Pemetaan_Detail_Label_Per_Frame_V2.json

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Konfigurasi Direktori
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Release_V2_AttentiveSkel3D
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

MANIFEST_PATH = PROJECT_ROOT / "src" / "data" / "manifest_v2.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_DIR = PROJECT_ROOT / "data" / "v2_labels"
TENSORS_DIR = PROJECT_ROOT / "data" / "tensors"

OUTPUT_DIR_EVAL = PROJECT_ROOT / "hasil_evaluasi"
OUTPUT_DIR_DATA = PROJECT_ROOT / "src" / "data"

OUTPUT_DIR_EVAL.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_DATA.mkdir(parents=True, exist_ok=True)


def cari_nama_video_mentah(stem: str, raw_dir: Path) -> str:
    """
    Menemukan nama file video asli di folder raw/ berdasarkan stem dataset.
    Contoh: 'Squat_001' -> 'primer_squat_frontal_subjek01_rep1.mp4'
    """
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return "Unknown"
    
    ex_name = parts[0]
    try:
        idx = int(parts[1]) - 1
    except ValueError:
        return "Unknown"

    folder = None
    for sub in raw_dir.iterdir():
        if sub.is_dir() and sub.name.lower() == ex_name.lower():
            folder = sub
            break
            
    if folder:
        videos = sorted(list(folder.glob("*.mp4")))
        if 0 <= idx < len(videos):
            return videos[idx].name
            
    return "Unknown"


def kelompokkan_segmen_kontigu(frame_indices: list[int]) -> list[dict]:
    """
    Mengelompokkan daftar frame diskrit menjadi rentang segmen berurutan.
    Contoh: [1, 2, 3, 7, 8, 9] -> [{'mulai': 1, 'akhir': 3, 'durasi': 3}, {'mulai': 7, 'akhir': 9, 'durasi': 3}]
    """
    if not frame_indices:
        return []
    
    segmen = []
    start = frame_indices[0]
    prev = frame_indices[0]
    
    for f in frame_indices[1:]:
        if f == prev + 1:
            prev = f
        else:
            segmen.append({
                "mulai_frame": start,
                "akhir_frame": prev,
                "durasi_frame": prev - start + 1
            })
            start = f
            prev = f
            
    segmen.append({
        "mulai_frame": start,
        "akhir_frame": prev,
        "durasi_frame": prev - start + 1
    })
    return segmen


def format_segmen_string(segmen_list: list[dict]) -> str:
    """Mengubah daftar segmen menjadi format string yang mudah dibaca di Excel/CSV."""
    if not segmen_list:
        return "Tidak Ada Frame Salah (100% Sempurna)"
    parts = []
    for s in segmen_list:
        if s["mulai_frame"] == s["akhir_frame"]:
            parts.append(f"Frame {s['mulai_frame']}")
        else:
            parts.append(f"Frame {s['mulai_frame']}-{s['akhir_frame']} ({s['durasi_frame']} frame)")
    return "; ".join(parts)


def main():
    print("=" * 75)
    print("  PEMBANGKIT PEMETAAN DETAIL LABEL PER-FRAME — AttentiveSkel-3D V2")
    print("=" * 75)
    print(f"Manifest Path : {MANIFEST_PATH}")
    print(f"Raw Videos Dir: {RAW_DIR}")
    print(f"Labels Dir    : {LABELS_DIR}")
    
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest tidak ditemukan: {MANIFEST_PATH}")
        sys.exit(1)
        
    df_manifest = pd.read_csv(MANIFEST_PATH)
    print(f"[INFO] Memproses {len(df_manifest)} dataset klip video...\n")
    
    list_pemetaan_json = []
    list_pemetaan_csv = []
    
    rekap_per_gerakan = {
        "benchpress": {"total_video": 0, "video_primer": 0, "video_sekunder": 0, "frame_benar": 0, "frame_salah": 0},
        "squat": {"total_video": 0, "video_primer": 0, "video_sekunder": 0, "frame_benar": 0, "frame_salah": 0},
        "deadlift": {"total_video": 0, "video_primer": 0, "video_sekunder": 0, "frame_benar": 0, "frame_salah": 0},
    }
    
    for idx, row in df_manifest.iterrows():
        file_path_str = str(row["file_path"])
        labels_path_str = str(row["labels_path"])
        exercise = str(row["exercise"]).lower()
        stem = Path(file_path_str).stem  # misal: Squat_001
        
        # Cari file label
        p_label = Path(labels_path_str)
        if not p_label.exists():
            p_label = LABELS_DIR / f"{stem}_labels.npy"
        if not p_label.exists():
            p_label = WORKSPACE_ROOT / "data" / "processed" / "v2_labels" / f"{stem}_labels.npy"
            
        if not p_label.exists():
            print(f"[WARNING] File label tidak ditemukan untuk {stem}, dilewati.")
            continue
            
        labels_array = np.load(str(p_label)).astype(int).tolist()
        
        # Cari file tensor
        p_tensor = Path(file_path_str)
        if not p_tensor.exists():
            p_tensor = TENSORS_DIR / f"{stem}.npy"
            
        # Cari nama video asli
        nama_video_mentah = cari_nama_video_mentah(stem, RAW_DIR)
        
        # Tentukan sumber (Primer vs Sekunder)
        if nama_video_mentah.startswith("primer"):
            kategori_sumber = "Primer"
        elif nama_video_mentah.startswith("sekunder"):
            kategori_sumber = "Sekunder"
        else:
            kategori_sumber = "Primer" if int(stem.split("_")[1]) <= 100 else "Sekunder"
            
        # Hitung indeks frame salah dan benar (1-indexed untuk manusia)
        frame_salah_1indexed = [i + 1 for i, val in enumerate(labels_array) if val == 1]
        frame_benar_1indexed = [i + 1 for i, val in enumerate(labels_array) if val == 0]
        
        # Bentuk segmen rentang kesalahan
        segmen_salah = kelompokkan_segmen_kontigu(frame_salah_1indexed)
        segmen_salah_str = format_segmen_string(segmen_salah)
        
        n_salah = len(frame_salah_1indexed)
        n_benar = len(frame_benar_1indexed)
        pct_salah = round((n_salah / 64.0) * 100, 2)
        
        if n_salah == 0:
            status_kualitatif = "100% Sempurna (Tidak Ada Kesalahan)"
        elif n_salah <= 16:
            status_kualitatif = "Dominan Benar (Kesalahan Ringan/Fase Transisi)"
        elif n_salah <= 40:
            status_kualitatif = "Campuran (Sebagian Fase Gerakan Salah)"
        else:
            status_kualitatif = "Dominan Salah (Gerakan Gagal / Half Rep)"
            
        # Update rekapitulasi
        if exercise in rekap_per_gerakan:
            rekap_per_gerakan[exercise]["total_video"] += 1
            if kategori_sumber == "Primer":
                rekap_per_gerakan[exercise]["video_primer"] += 1
            else:
                rekap_per_gerakan[exercise]["video_sekunder"] += 1
            rekap_per_gerakan[exercise]["frame_benar"] += n_benar
            rekap_per_gerakan[exercise]["frame_salah"] += n_salah

        # Entri Objek JSON
        item_json = {
            "nama_dataset": stem,
            "nama_video_mentah": nama_video_mentah,
            "jenis_latihan": exercise.capitalize(),
            "kategori_sumber": kategori_sumber,
            "total_frame": 64,
            "jumlah_frame_benar": n_benar,
            "jumlah_frame_salah": n_salah,
            "persentase_frame_salah_pct": pct_salah,
            "status_kualitatif": status_kualitatif,
            "frame_salah_1indexed": frame_salah_1indexed,
            "frame_benar_1indexed": frame_benar_1indexed,
            "rentang_segmen_salah": segmen_salah,
            "ringkasan_rentang_salah": segmen_salah_str,
            "urutan_label_biner_64": labels_array,
            "path_file_tensor": str(p_tensor),
            "path_file_label": str(p_label)
        }
        list_pemetaan_json.append(item_json)
        
        # Entri Baris CSV
        item_csv = {
            "Nama_Dataset": stem,
            "Nama_Video_Mentah": nama_video_mentah,
            "Jenis_Latihan": exercise.capitalize(),
            "Kategori_Sumber": kategori_sumber,
            "Total_Frame": 64,
            "Jumlah_Frame_Benar": n_benar,
            "Jumlah_Frame_Salah": n_salah,
            "Persentase_Salah_Pct": pct_salah,
            "Status_Kualitatif": status_kualitatif,
            "Rentang_Frame_Salah": segmen_salah_str,
            "Daftar_Eksplisit_Frame_Salah": ", ".join(map(str, frame_salah_1indexed)) if frame_salah_1indexed else "-",
            "Daftar_Eksplisit_Frame_Benar": ", ".join(map(str, frame_benar_1indexed)) if frame_benar_1indexed else "-",
            "Array_Label_64_Biner": "".join(map(str, labels_array)),
            "Path_Tensor": str(p_tensor),
            "Path_Label": str(p_label)
        }
        list_pemetaan_csv.append(item_csv)

    # Susun Dokumen Master JSON
    total_vid = len(list_pemetaan_json)
    total_frm = total_vid * 64
    tot_bnr = sum(x["jumlah_frame_benar"] for x in list_pemetaan_json)
    tot_slh = sum(x["jumlah_frame_salah"] for x in list_pemetaan_json)
    
    master_json = {
        "metadata": {
            "judul": "Dokumentasi Pemetaan Detail Label Kualitas Gerak Per-Frame",
            "proyek": "AttentiveSkel-3D V2 — Sistem Klasifikasi Kualitas Angkat Beban Per-Frame",
            "author": "Seno Aji",
            "institusi": "Tugas Akhir Teknik Informatika",
            "dosen_pembimbing": "Pak Jasman, S.Kom., M.Kom.",
            "tanggal_dibuat": "19 Agustus 2026",
            "total_video": total_vid,
            "total_frame": total_frm,
            "total_frame_benar": tot_bnr,
            "total_frame_salah": tot_slh,
            "rasio_frame_benar_persen": round((tot_bnr / total_frm) * 100, 2),
            "rasio_frame_salah_persen": round((tot_slh / total_frm) * 100, 2),
            "rekapitulasi_per_gerakan": rekap_per_gerakan,
            "kriteria_evaluasi_biomekanik": {
                "squat": {
                    "kriteria_1": "Knee Valgus (Rasio lebar horizontal lutut / pergelangan kaki >= 0.85)",
                    "kriteria_2": "Hip Flexion Depth (Sudut Bahu-Pinggul-Lutut <= 137 derajat)",
                    "kriteria_3": "Knee Depth (Sudut Pinggul-Lutut-Pergelangan Kaki <= 100 derajat)"
                },
                "benchpress": {
                    "kriteria_1": "Elbow Range of Motion (Sudut Bahu-Siku-Pergelangan Tangan <= 85 derajat)"
                },
                "deadlift": {
                    "kriteria_1": "Hip Hinge Pattern (Sudut inklinasi punggung dari vertikal >= 20 derajat)",
                    "kriteria_2": "Lumbar Neutral / Anti-Rounded Back (Sudut inklinasi punggung <= 60 derajat)"
                }
            },
            "keterangan_nilai_label": {
                "0": "BENAR (Gerakan memenuhi seluruh kaidah biomekanik pada posisi tersebut)",
                "1": "SALAH (Gerakan melanggar salah satu/lebih kriteria biomekanik: valgus, half-rep, atau rounded back)"
            }
        },
        "daftar_pemetaan_per_video": list_pemetaan_json
    }
    
    # Simpan JSON
    json_path_eval = OUTPUT_DIR_EVAL / "Pemetaan_Detail_Label_Per_Frame_V2.json"
    json_path_data = OUTPUT_DIR_DATA / "Pemetaan_Detail_Label_Per_Frame_V2.json"
    
    with open(json_path_eval, "w", encoding="utf-8") as f:
        json.dump(master_json, f, indent=2, ensure_ascii=False)
        
    with open(json_path_data, "w", encoding="utf-8") as f:
        json.dump(master_json, f, indent=2, ensure_ascii=False)
        
    print(f"✓ JSON Tersimpan: {json_path_eval}")
    print(f"✓ JSON Tersimpan: {json_path_data}")
    
    # Simpan CSV
    df_out = pd.DataFrame(list_pemetaan_csv)
    csv_path_eval = OUTPUT_DIR_EVAL / "Pemetaan_Detail_Label_Per_Frame_V2.csv"
    csv_path_data = OUTPUT_DIR_DATA / "Pemetaan_Detail_Label_Per_Frame_V2.csv"
    
    df_out.to_csv(csv_path_eval, index=False, encoding="utf-8-sig")
    df_out.to_csv(csv_path_data, index=False, encoding="utf-8-sig")
    
    print(f"✓ CSV Tersimpan : {csv_path_eval}")
    print(f"✓ CSV Tersimpan : {csv_path_data}")
    
    print("\n" + "=" * 75)
    print("  RINGKASAN PEMETAAN:")
    print("=" * 75)
    print(f"Total Video Terproses : {total_vid}")
    print(f"Total Frame Berlabel  : {total_frm:,} frame")
    print(f"  Frame BENAR (0)     : {tot_bnr:,} ({tot_bnr/total_frm*100:.2f}%)")
    print(f"  Frame SALAH (1)     : {tot_slh:,} ({tot_slh/total_frm*100:.2f}%)")
    print("=" * 75)


if __name__ == "__main__":
    main()
