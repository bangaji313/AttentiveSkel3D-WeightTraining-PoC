# src/data/v2/baca_wujud_npy.py
#
# Membongkar isi file .npy (fitur dan label) agar wujud data per-frame
# dapat dibaca langsung oleh dosen penguji tanpa perlu memahami kode PyTorch.
#
# Output:
#   1. Tabel 64 baris tercetak penuh di terminal.
#   2. File CSV di data/processed/Wujud_Isi_NPY_Sample.csv (dapat dibuka Excel).
#
# Cara menjalankan dari root proyek:
#   conda activate attentiveskel
#   python src/data/v2/baca_wujud_npy.py

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Konfigurasi path
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Sampel yang digunakan sebagai demonstrasi (BenchPress_001)
PATH_FITUR = PROJECT_ROOT / "data" / "processed" / "tensors"       / "Squat_001.npy"
PATH_LABEL = PROJECT_ROOT / "data" / "processed" / "v2_labels"     / "Squat_001_labels.npy"
PATH_CSV   = PROJECT_ROOT / "data" / "processed" / "Wujud_Isi_NPY_Sample.csv"

# ============================================================
# Load kedua file NPY
# ============================================================
print("=" * 65)
print("PEMBONGKARAN ISI FILE NPY - AttentiveSkel-3D v2")
print("=" * 65)

# Fitur: shape (64, 33, 3) — 64 frame × 33 landmark × 3 sumbu (X,Y,Z)
fitur_npy = np.load(PATH_FITUR).astype(np.float32)
print(f"\n[FITUR NPY]  Path  : {PATH_FITUR.name}")
print(f"             Shape : {fitur_npy.shape}   -> (T=64 frame, L=33 landmark, C=3 sumbu)")

# Label: shape (64,) — satu label biner per frame
label_npy = np.load(PATH_LABEL).astype(np.int8)
print(f"\n[LABEL NPY]  Path  : {PATH_LABEL.name}")
print(f"             Shape : {label_npy.shape}   -> (T=64 frame,)")

# ============================================================
# Deteksi Sendi Utama Berdasarkan Jenis Latihan (Untuk Terminal)
# ============================================================
stem_lower = PATH_FITUR.stem.lower()
if "bench" in stem_lower:
    nama_utama = "Siku_Kanan"
    idx_utama  = 14
elif "deadlift" in stem_lower:
    nama_utama = "Pinggul_Kanan"
    idx_utama  = 24
else:
    nama_utama = "Lutut_Kanan"
    idx_utama  = 26

# ============================================================
# Bangun DataFrame — looping 64 frame temporal
# ============================================================
baris_data = []

# Daftar 33 landmark (sesuai urutan MediaPipe BlazePose 0-32)
NAMA_SENDI_33 = [
    "Nose", "Left_Eye_Inner", "Left_Eye", "Left_Eye_Outer", "Right_Eye_Inner", 
    "Right_Eye", "Right_Eye_Outer", "Left_Ear", "Right_Ear", "Mouth_Left", 
    "Mouth_Right", "Left_Shoulder", "Right_Shoulder", "Left_Elbow", "Right_Elbow", 
    "Left_Wrist", "Right_Wrist", "Left_Pinky", "Right_Pinky", "Left_Index", 
    "Right_Index", "Left_Thumb", "Right_Thumb", "Left_Hip", "Right_Hip", 
    "Left_Knee", "Right_Knee", "Left_Ankle", "Right_Ankle", "Left_Heel", 
    "Right_Heel", "Left_Foot_Index", "Right_Foot_Index"
]

for i in range(64):
    label_biner = int(label_npy[i])
    status      = "BENAR" if label_biner == 0 else "SALAH"

    # 1. Kolom wajib awal
    data_frame = {
        "Frame_Ke"   : i + 1,          # 1-indexed agar mudah dibaca manusia
        "Label_Biner": label_biner,     # Nilai 0 atau 1 dari NPY label
        "Status"     : status,          # "BENAR" atau "SALAH"
    }

    # 2. Kolom dinamis (hanya untuk ditampilkan ringkas di Terminal)
    data_frame[f"{nama_utama}_X (idx{idx_utama})"] = round(float(fitur_npy[i, idx_utama, 0]), 5)
    data_frame[f"{nama_utama}_Y (idx{idx_utama})"] = round(float(fitur_npy[i, idx_utama, 1]), 5)
    data_frame[f"{nama_utama}_Z (idx{idx_utama})"] = round(float(fitur_npy[i, idx_utama, 2]), 5)

    # 3. Kolom FULL 99 Koordinat (untuk bukti lengkap di Excel/CSV)
    for j in range(33):
        nama = NAMA_SENDI_33[j]
        data_frame[f"{nama}_X (idx{j})"] = round(float(fitur_npy[i, j, 0]), 5)
        data_frame[f"{nama}_Y (idx{j})"] = round(float(fitur_npy[i, j, 1]), 5)
        data_frame[f"{nama}_Z (idx{j})"] = round(float(fitur_npy[i, j, 2]), 5)

    baris_data.append(data_frame)

# ============================================================
# Cetak ke terminal (Hanya kolom ringkas agar terbaca)
# ============================================================
df_csv = pd.DataFrame(baris_data)

kolom_terminal = [
    "Frame_Ke", "Label_Biner", "Status", 
    f"{nama_utama}_X (idx{idx_utama})", 
    f"{nama_utama}_Y (idx{idx_utama})", 
    f"{nama_utama}_Z (idx{idx_utama})"
]
df_terminal = df_csv[kolom_terminal]

# Aktifkan opsi agar semua 64 baris dicetak penuh tanpa truncation
pd.set_option("display.max_rows",    None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width",       160)

print("\n" + "=" * 65)
print("TABEL WUJUD DATA PER-FRAME (DI TERMINAL HANYA DITAMPILKAN 1 SENDI UTAMA)")
print(f"Sumber fitur : {PATH_FITUR.relative_to(PROJECT_ROOT)}")
print(f"Sumber label : {PATH_LABEL.relative_to(PROJECT_ROOT)}")
print("=" * 65)
print(df_terminal.to_string(index=False))

# ============================================================
# Ringkasan distribusi label
# ============================================================
n_benar = (label_npy == 0).sum()
n_salah = (label_npy == 1).sum()
print("\n" + "-" * 65)
print(f"DISTRIBUSI LABEL - 1 Repetisi {PATH_FITUR.stem}:")
print(f"  Frame BENAR (0): {n_benar:2d} frame  ({n_benar/64*100:.1f}%)")
print(f"  Frame SALAH (1): {n_salah:2d} frame  ({n_salah/64*100:.1f}%)")
print("-" * 65)

# ============================================================
# Simpan ke CSV agar dapat dibuka via Microsoft Excel
# ============================================================
df_csv.to_csv(PATH_CSV, index=False, encoding="utf-8-sig")  # utf-8-sig agar Excel terbaca benar

print(f"\n[TERSIMPAN] File CSV (FULL 33 SENDI / 99 KOLOM KOORDINAT) siap dibuka di Excel:")
print(f"            {PATH_CSV}")
print("=" * 65)
