# src/data/v2/batch_baca_semua_npy.py
#
# Membongkar isi SELURUH 487 file tensor NPY beserta labelnya menjadi
# satu file CSV besar yang dapat dibuka di Microsoft Excel.
#
# Setiap baris di CSV merepresentasikan SATU FRAME dari satu dataset:
#   - Nama_Dataset     : nama file (contoh: Squat_001)
#   - Jenis_Latihan    : BenchPress / Deadlift / Squat
#   - Nomor_Sesi       : nomor urut dataset (contoh: 1, 2, ...)
#   - Frame_Ke         : frame ke-1 sampai ke-64
#   - Label_Biner      : 0 (BENAR) atau 1 (SALAH)
#   - Status           : "BENAR" atau "SALAH"
#   - 99 kolom koordinat: Nose_X ... Right_Foot_Index_Z
#
# Total baris: 487 dataset × 64 frame = 31.168 baris
# Total kolom: 6 identitas + 99 koordinat = 105 kolom
#
# Cara menjalankan:
#   conda activate attentiveskel
#   python src/data/v2/batch_baca_semua_npy.py

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

# ============================================================
# Konfigurasi path
# ============================================================
PROJECT_ROOT  = Path(__file__).resolve().parents[3]
DIR_FITUR     = PROJECT_ROOT / "data" / "processed" / "tensors"
DIR_LABEL     = PROJECT_ROOT / "data" / "processed" / "v2_labels"
OUTPUT_CSV    = PROJECT_ROOT / "data" / "processed" / "Wujud_Semua_NPY_v2.csv"

# ============================================================
# Daftar 33 Landmark BlazePose (urutan MediaPipe 0–32)
# ============================================================
NAMA_SENDI_33 = [
    "Nose", "Left_Eye_Inner", "Left_Eye", "Left_Eye_Outer",
    "Right_Eye_Inner", "Right_Eye", "Right_Eye_Outer",
    "Left_Ear", "Right_Ear", "Mouth_Left", "Mouth_Right",
    "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow",
    "Left_Wrist", "Right_Wrist",
    "Left_Pinky", "Right_Pinky",
    "Left_Index", "Right_Index",
    "Left_Thumb", "Right_Thumb",
    "Left_Hip", "Right_Hip",
    "Left_Knee", "Right_Knee",
    "Left_Ankle", "Right_Ankle",
    "Left_Heel", "Right_Heel",
    "Left_Foot_Index", "Right_Foot_Index",
]

# ============================================================
# Scan semua file fitur NPY
# ============================================================
semua_fitur = sorted(DIR_FITUR.glob("*.npy"))

print("=" * 70)
print("BATCH EKSPOR SEMUA FILE NPY → CSV (AttentiveSkel-3D v2)")
print("=" * 70)
print(f"Folder fitur : {DIR_FITUR}")
print(f"Folder label : {DIR_LABEL}")
print(f"Jumlah file  : {len(semua_fitur)} dataset")
print(f"Output CSV   : {OUTPUT_CSV}")
print()

# ============================================================
# Proses Setiap File NPY
# ============================================================
semua_baris = []
n_skip      = 0
n_ok        = 0

for path_fitur in tqdm(semua_fitur, desc="Memproses dataset"):
    stem = path_fitur.stem  # contoh: "Squat_001"

    # Tentukan jenis latihan dan nomor sesi dari nama file
    if stem.startswith("BenchPress"):
        jenis    = "BenchPress"
        no_sesi  = int(stem.replace("BenchPress_", ""))
    elif stem.startswith("Deadlift"):
        jenis    = "Deadlift"
        no_sesi  = int(stem.replace("Deadlift_", ""))
    elif stem.startswith("Squat"):
        jenis    = "Squat"
        no_sesi  = int(stem.replace("Squat_", ""))
    else:
        jenis    = "Unknown"
        no_sesi  = 0

    # Cari file label yang bersesuaian
    path_label = DIR_LABEL / f"{stem}_labels.npy"
    if not path_label.exists():
        tqdm.write(f"  [SKIP] Label tidak ditemukan: {path_label.name}")
        n_skip += 1
        continue

    # Load array
    fitur_npy = np.load(path_fitur).astype(np.float32)  # (64, 33, 3)
    label_npy = np.load(path_label).astype(np.int8)      # (64,)

    # Validasi shape
    if fitur_npy.shape != (64, 33, 3) or label_npy.shape != (64,):
        tqdm.write(f"  [SKIP] Shape tidak sesuai: {stem} | fitur={fitur_npy.shape} | label={label_npy.shape}")
        n_skip += 1
        continue

    # Loop 64 frame → 64 baris CSV
    for i in range(64):
        label_biner = int(label_npy[i])
        baris = {
            "Nama_Dataset"  : stem,
            "Jenis_Latihan" : jenis,
            "Nomor_Sesi"    : no_sesi,
            "Frame_Ke"      : i + 1,
            "Label_Biner"   : label_biner,
            "Status"        : "BENAR" if label_biner == 0 else "SALAH",
        }

        # 99 kolom koordinat (33 sendi × 3 sumbu X, Y, Z)
        for j, nama in enumerate(NAMA_SENDI_33):
            baris[f"{nama}_X"] = round(float(fitur_npy[i, j, 0]), 6)
            baris[f"{nama}_Y"] = round(float(fitur_npy[i, j, 1]), 6)
            baris[f"{nama}_Z"] = round(float(fitur_npy[i, j, 2]), 6)

        semua_baris.append(baris)

    n_ok += 1

# ============================================================
# Susun DataFrame & Simpan CSV
# ============================================================
print(f"\n[INFO] Berhasil diproses : {n_ok} dataset ({n_ok * 64:,} baris)")
if n_skip:
    print(f"[INFO] Dilewati (SKIP)  : {n_skip} dataset (label tidak ditemukan / shape salah)")

df = pd.DataFrame(semua_baris)

# Ringkasan distribusi label global
n_benar_total = (df["Label_Biner"] == 0).sum()
n_salah_total = (df["Label_Biner"] == 1).sum()
print(f"\nDistribusi Label Global ({len(df):,} total frame):")
print(f"  BENAR (Label 0) : {n_benar_total:>7,} frame  ({n_benar_total/len(df)*100:.1f}%)")
print(f"  SALAH (Label 1) : {n_salah_total:>7,} frame  ({n_salah_total/len(df)*100:.1f}%)")

# Ringkasan per jenis latihan
print("\nRingkasan per Jenis Latihan:")
ringkasan = df.groupby("Jenis_Latihan").agg(
    Total_Frame   = ("Frame_Ke", "count"),
    Frame_Benar   = ("Label_Biner", lambda x: (x == 0).sum()),
    Frame_Salah   = ("Label_Biner", lambda x: (x == 1).sum()),
    Jumlah_Dataset= ("Nama_Dataset", "nunique"),
).reset_index()
print(ringkasan.to_string(index=False))

# Simpan ke CSV (utf-8-sig agar terbaca Excel tanpa masalah karakter)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n[TERSIMPAN] File CSV siap dibuka di Excel:")
print(f"  {OUTPUT_CSV}")
print(f"  Total baris : {len(df):,}")
print(f"  Total kolom : {len(df.columns)}")
print("=" * 70)
