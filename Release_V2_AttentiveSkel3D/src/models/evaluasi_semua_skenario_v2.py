# ============================================================
# src/models/evaluasi_semua_skenario_v2.py
#
# Skrip Evaluasi Komparatif SEMUA Skenario — AttentiveSkel-3D V2
#
# Tujuan:
#   Mengevaluasi seluruh model yang telah dilatih (Baseline, Full Model,
#   dan semua varian Ablasi) secara otomatis dalam satu eksekusi,
#   kemudian menyimpan hasil perbandingan ke dalam satu file CSV yang rapi.
#
# Metrik yang dihitung (Per-Frame Level):
#   - Accuracy, F1-Score (Macro & Binary), Precision, Recall
#   - Confusion Matrix (TP, TN, FP, FN)
#   - Bobot Atensi Spatial (BSP) per sendi — jika model mendukung
#
# Cara menjalankan:
#   Pastikan Anda berada di dalam folder Release_V2_AttentiveSkel3D:
#
#   conda activate attentiveskel
#   python src/models/evaluasi_semua_skenario_v2.py
#
# Output yang dihasilkan (di dalam folder hasil_evaluasi/):
#   1. Perbandingan_Metrik_Semua_Skenario_V2.csv  — Tabel ringkasan utama
#   2. Confusion_Matrix_Semua_Skenario_V2.csv     — Detail TP/TN/FP/FN
#   3. Bobot_Atensi_Per_Skenario_V2.csv           — Perbandingan bobot sendi
# ============================================================

import sys
from pathlib import Path

# Direktori root adalah satu level di atas folder src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.arsitektur_v2 import AttentiveSkel3DPerFrame
from src.data.dataset_v2 import create_dataloaders_v2

# ============================================================
# Konfigurasi Path
# ============================================================

# Path ke file manifest (daftar dataset)
MANIFEST_PATH = PROJECT_ROOT / "src" / "data" / "manifest_v2.csv"

# Direktori tempat model-model .pth tersimpan
MODELS_DIR = PROJECT_ROOT / "bobot_model"

# Direktori output untuk menyimpan hasil evaluasi
OUTPUT_DIR = PROJECT_ROOT / "hasil_evaluasi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Definisi Skenario Evaluasi
# Kunci (key) adalah nama skenario, nilai (value) adalah nama file .pth
# ============================================================
SKENARIO = {
    "Skenario 1 — Baseline (Tanpa Atensi)": "best_model_baseline.pth",
    "Skenario 2 — Full Model (BSP + LearnedSpatial + Temporal)": "best_model_v2.pth",
    "Skenario 3a — Ablasi: Hanya BSP": "best_model_ablasi_a.pth",
    "Skenario 3b — Ablasi: BSP + LearnedSpatial": "best_model_ablasi_b.pth",
    "Skenario 3c — Ablasi: BSP + Temporal": "best_model_ablasi_c.pth",
}

# Nama 33 landmark MediaPipe Pose (urut dari indeks 0 s/d 32)
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


def muat_model(model_path: Path, device: torch.device):
    """
    Memuat model AttentiveSkel3DPerFrame dari file .pth.

    Fungsi ini mendeteksi secara otomatis fitur atensi apa saja yang
    dimiliki model (BSP, Learned Spatial, Temporal) berdasarkan nama
    kunci (key) dalam state_dict, sehingga tidak perlu konfigurasi manual.

    Mengembalikan:
        model              : Model yang sudah dimuat dan siap evaluasi.
        fitur_atensi (dict): Dictionary status aktif/tidaknya tiap fitur.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    # Muat checkpoint; toleransi terhadap format dengan/tanpa wrapper 'model_state_dict'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Deteksi otomatis fitur atensi dari kunci parameter
    use_bsp      = any(k.startswith("biomechanical_spatial_prior") for k in state_dict)
    use_learned  = any(k.startswith("learned_spatial_attention")   for k in state_dict)
    use_temporal = any(k.startswith("temporal_attention")           for k in state_dict)

    model = AttentiveSkel3DPerFrame(
        num_classes=2,
        use_spatial_prior=use_bsp,
        use_learned_spatial=use_learned,
        use_temporal_attention=use_temporal,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    fitur_atensi = {
        "BSP": use_bsp,
        "LearnedSpatial": use_learned,
        "Temporal": use_temporal,
    }

    return model, fitur_atensi


def inferensi_test_set(model, test_loader, device):
    """
    Menjalankan forward pass model pada seluruh test set.

    Setiap prediksi dan ground truth per-frame di-flatten agar bisa
    langsung dihitung metrik menggunakan sklearn (level frame, bukan video).

    Mengembalikan:
        semua_preds  (np.array): Prediksi kelas per-frame (0=BENAR, 1=SALAH)
        semua_truths (np.array): Ground truth per-frame
    """
    semua_preds  = []
    semua_truths = []

    with torch.no_grad():
        for batch_input, batch_labels in test_loader:
            batch_input  = batch_input.to(device)   # (B, 64, 33, 3)
            batch_labels = batch_labels.to(device)  # (B, 64)

            logits = model(batch_input)             # (B, 64, 2)
            preds  = logits.argmax(dim=2)           # (B, 64)

            # Flatten (B, 64) → (B*64,) untuk komputasi metrik frame-level
            semua_preds.extend(preds.cpu().numpy().flatten().tolist())
            semua_truths.extend(batch_labels.cpu().numpy().flatten().tolist())

    return np.array(semua_preds), np.array(semua_truths)


def hitung_metrik(preds, truths):
    """
    Menghitung seluruh metrik kinerja dari array prediksi dan ground truth.

    Mengembalikan dictionary berisi semua nilai metrik yang sudah dibulatkan.
    """
    # Hitung Confusion Matrix: urutan (TN, FP, FN, TP) untuk kasus biner
    cm = confusion_matrix(truths, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy (%)": round(accuracy_score(truths, preds) * 100, 2),
        "F1_Macro":     round(f1_score(truths, preds, average="macro", zero_division=0), 4),
        "F1_Binary":    round(f1_score(truths, preds, pos_label=1, average="binary", zero_division=0), 4),
        "Precision":    round(precision_score(truths, preds, pos_label=1, average="binary", zero_division=0), 4),
        "Recall":       round(recall_score(truths, preds, pos_label=1, average="binary", zero_division=0), 4),
        "TP":           int(tp),
        "TN":           int(tn),
        "FP":           int(fp),
        "FN":           int(fn),
        "Total_Frame":  len(truths),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("  EVALUASI KOMPARATIF SEMUA SKENARIO — AttentiveSkel-3D V2")
    print("=" * 75)
    print(f"  Device       : {device}")
    print(f"  Manifest     : {MANIFEST_PATH}")
    print(f"  Jumlah model : {len(SKENARIO)} skenario")
    print()

    # Verifikasi manifest tersedia
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest tidak ditemukan: {MANIFEST_PATH}")
        print("  Pastikan file manifest_v2.csv ada di dalam folder src/data/")
        sys.exit(1)

    # ============================================================
    # Siapkan Test Loader sekali — dipakai bersama semua skenario
    # (seed yang sama dengan saat training agar pembagian data konsisten)
    # ============================================================
    print("[INFO] Memuat dataset dan menyiapkan test loader...")
    _, _, test_loader = create_dataloaders_v2(
        csv_file=MANIFEST_PATH,
        batch_size=16,
        train_ratio=0.70,
        val_ratio=0.15,
        random_seed=42,       # <-- Harus sama dengan seed saat training
    )
    print(f"[INFO] Test loader siap. Jumlah batch: {len(test_loader)}\n")

    # ============================================================
    # Kontainer untuk akumulasi hasil semua skenario
    # ============================================================
    hasil_metrik   = []  # Daftar dict metrik per skenario
    hasil_cm       = []  # Daftar dict confusion matrix per skenario
    hasil_atensi   = []  # Daftar dict bobot atensi per skenario (jika BSP aktif)

    # ============================================================
    # Iterasi Evaluasi per Skenario
    # ============================================================
    for nama_skenario, nama_file in SKENARIO.items():
        model_path = MODELS_DIR / nama_file
        print(f"{'─'*75}")
        print(f"  Skenario : {nama_skenario}")
        print(f"  File     : {nama_file}")

        # Lewati jika file model tidak ada (misal: skenario belum dilatih)
        if not model_path.exists():
            print(f"  [LEWATI] File model tidak ditemukan, skenario ini dilewati.\n")
            continue

        try:
            # Muat model
            model, fitur = muat_model(model_path, device)
            status_fitur = " | ".join(
                f"{k}: {'✓' if v else '✗'}" for k, v in fitur.items()
            )
            print(f"  Fitur    : {status_fitur}")

            # Jalankan inferensi pada test set
            preds, truths = inferensi_test_set(model, test_loader, device)

            # Hitung metrik
            metrik = hitung_metrik(preds, truths)
            print(f"  Accuracy : {metrik['Accuracy (%)']:.2f}%  |  F1 Macro: {metrik['F1_Macro']:.4f}  |  F1 Binary: {metrik['F1_Binary']:.4f}")
            print(f"  TP={metrik['TP']:,}  TN={metrik['TN']:,}  FP={metrik['FP']:,}  FN={metrik['FN']:,}")

            # Simpan ke akumulator metrik
            hasil_metrik.append({
                "Skenario"         : nama_skenario,
                "File_Model"       : nama_file,
                "BSP"              : "✓" if fitur["BSP"] else "✗",
                "Learned_Spatial"  : "✓" if fitur["LearnedSpatial"] else "✗",
                "Temporal"         : "✓" if fitur["Temporal"] else "✗",
                **metrik,
            })

            # Simpan ke akumulator confusion matrix
            hasil_cm.append({
                "Skenario": nama_skenario,
                "TP"      : metrik["TP"],
                "TN"      : metrik["TN"],
                "FP"      : metrik["FP"],
                "FN"      : metrik["FN"],
            })

            # Ekstrak bobot atensi sendi (hanya jika BSP aktif)
            if fitur["BSP"]:
                bsp_weights = torch.sigmoid(model.biomechanical_spatial_prior)
                bsp_array   = bsp_weights.detach().cpu().squeeze().numpy()  # (33,)

                for idx_sendi, (nama_sendi, bobot) in enumerate(zip(NAMA_SENDI_33, bsp_array)):
                    hasil_atensi.append({
                        "Skenario"    : nama_skenario,
                        "Indeks_Sendi": idx_sendi,
                        "Nama_Sendi"  : nama_sendi,
                        "Bobot_Atensi": round(float(bobot), 6),
                    })

        except Exception as exc:
            print(f"  [ERROR] Gagal mengevaluasi: {exc}")

        print()

    # ============================================================
    # Ekspor Semua Hasil ke CSV
    # ============================================================
    print("=" * 75)
    print("  EKSPOR HASIL KE CSV")
    print("=" * 75)

    if hasil_metrik:
        # 1. Tabel Perbandingan Metrik Utama
        df_metrik = pd.DataFrame(hasil_metrik)
        path_metrik = OUTPUT_DIR / "Perbandingan_Metrik_Semua_Skenario_V2.csv"
        df_metrik.to_csv(path_metrik, index=False, encoding="utf-8-sig")
        print(f"\n  [TERSIMPAN] Perbandingan Metrik   → {path_metrik.name}")

        # Tampilkan tabel ringkas di terminal
        print("\n  RINGKASAN PERBANDINGAN SEMUA SKENARIO:")
        print(f"  {'Skenario':50s} {'Accuracy':>10s} {'F1 Macro':>10s} {'F1 Binary':>10s}")
        print(f"  {'─'*50} {'─'*10} {'─'*10} {'─'*10}")
        for baris in hasil_metrik:
            nama = baris['Skenario'][:50]
            print(f"  {nama:50s} {baris['Accuracy (%)']:>9.2f}% {baris['F1_Macro']:>10.4f} {baris['F1_Binary']:>10.4f}")

    if hasil_cm:
        # 2. Tabel Detail Confusion Matrix
        df_cm = pd.DataFrame(hasil_cm)
        path_cm = OUTPUT_DIR / "Confusion_Matrix_Semua_Skenario_V2.csv"
        df_cm.to_csv(path_cm, index=False, encoding="utf-8-sig")
        print(f"\n  [TERSIMPAN] Detail Confusion Matrix → {path_cm.name}")

    if hasil_atensi:
        # 3. Tabel Perbandingan Bobot Atensi per Sendi (pivot: skenario sebagai kolom)
        df_atensi = pd.DataFrame(hasil_atensi)
        df_pivot  = df_atensi.pivot_table(
            index  =["Indeks_Sendi", "Nama_Sendi"],
            columns="Skenario",
            values ="Bobot_Atensi"
        ).reset_index()
        df_pivot.columns.name = None  # Hilangkan nama index kolom pivot
        path_atensi = OUTPUT_DIR / "Bobot_Atensi_Per_Skenario_V2.csv"
        df_pivot.to_csv(path_atensi, index=False, encoding="utf-8-sig")
        print(f"\n  [TERSIMPAN] Bobot Atensi per Sendi → {path_atensi.name}")

    print(f"\n  Semua file tersimpan di folder: {OUTPUT_DIR}")
    print("=" * 75)
    print("  EVALUASI SELESAI.")
    print("=" * 75)


if __name__ == "__main__":
    main()
