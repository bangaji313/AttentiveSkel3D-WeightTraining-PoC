# src/models/v2/evaluate.py
#
# Script evaluasi komprehensif model AttentiveSkel-3D v2.
#
# Menghitung metrik kinerja PER FRAME (bukan per video):
#   - Accuracy, Precision, Recall, F1-Score
#   - Confusion Matrix (True Positive, True Negative, False Positive, False Negative)
#   - Distribusi Prediksi per Jenis Latihan
#
# PENTING: Evaluasi ini mempertahankan dimensi temporal (64 frame).
# Prediksi & Ground Truth di-flatten menjadi daftar panjang (N*64),
# sehingga setiap frame diperlakukan sebagai sampel independen.
#
# Cara menjalankan:
#   conda activate attentiveskel
#   python src/models/v2/evaluate.py
#   python src/models/v2/evaluate.py --model bobot_model/best_model_ablasi_a.pth

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

from src.data.dataset_v2 import create_dataloaders_v2
from src.models.arsitektur_v2 import AttentiveSkel3DPerFrame

# ============================================================
# Konfigurasi
# ============================================================
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest_v2.csv"
LABEL_NAMES   = ["BENAR (0)", "SALAH (1)"]

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


def load_model(model_path: Path, device: torch.device) -> AttentiveSkel3DPerFrame:
    """Muat model dari file .pth dengan deteksi fitur atensi otomatis."""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    use_spatial_prior      = any(k.startswith("biomechanical_spatial_prior") for k in state_dict)
    use_learned_spatial    = any(k.startswith("learned_spatial_attention")   for k in state_dict)
    use_temporal_attention = any(k.startswith("temporal_attention")           for k in state_dict)

    model = AttentiveSkel3DPerFrame(
        num_classes=2,
        use_spatial_prior=use_spatial_prior,
        use_learned_spatial=use_learned_spatial,
        use_temporal_attention=use_temporal_attention,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, use_spatial_prior


def run_evaluation(model_path: Path):
    """Menjalankan evaluasi penuh dan mencetak laporan kinerja."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("EVALUASI KINERJA PER-FRAME — AttentiveSkel-3D v2")
    print("=" * 70)
    print(f"Model   : {model_path.name}")
    print(f"Device  : {device}")

    model, has_bsp = load_model(model_path, device)
    print(f"  - Biomechanical Spatial Prior : {'Aktif' if has_bsp else 'Nonaktif'}")
    print()

    # Split dataset dengan seed yang sama seperti saat training
    _, _, test_loader = create_dataloaders_v2(
        csv_file=MANIFEST_PATH,
        batch_size=16,
        train_ratio=0.70,
        val_ratio=0.15,
        random_seed=42,
    )

    # ============================================================
    # Forward Pass pada Test Set — kumpulkan semua prediksi
    # ============================================================
    all_preds  = []  # Prediksi model per-frame
    all_truths = []  # Ground truth per-frame
    all_probs  = []  # Probabilitas kelas Salah (class 1)

    with torch.no_grad():
        for batch_input, batch_labels in test_loader:
            batch_input  = batch_input.to(device)    # (B, 64, 33, 3)
            batch_labels = batch_labels.to(device)   # (B, 64)

            logits = model(batch_input)              # (B, 64, 2)
            probs  = torch.softmax(logits, dim=2)   # (B, 64, 2)
            preds  = logits.argmax(dim=2)            # (B, 64)

            # Flatten (B, 64) → (B*64,)
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_truths.extend(batch_labels.cpu().numpy().flatten().tolist())
            all_probs.extend(probs[:, :, 1].cpu().numpy().flatten().tolist())

    all_preds  = np.array(all_preds)
    all_truths = np.array(all_truths)
    all_probs  = np.array(all_probs)

    total_frames = len(all_truths)

    # ============================================================
    # Hitung Metrik Kinerja
    # ============================================================
    acc = accuracy_score(all_truths, all_preds)
    cm  = confusion_matrix(all_truths, all_preds)

    tn, fp, fn, tp = cm.ravel()

    precision_salah = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_salah    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_salah        = f1_score(all_truths, all_preds, pos_label=1, average="binary")
    f1_macro        = f1_score(all_truths, all_preds, average="macro")

    # ============================================================
    # Cetak Laporan
    # ============================================================
    print(f"{'='*70}")
    print(f"  RINGKASAN EVALUASI TEST SET ({total_frames:,} prediksi frame)")
    print(f"{'='*70}")
    print(f"  Akurasi Keseluruhan       : {acc*100:.2f}%")
    print(f"  F1-Score (Macro)          : {f1_macro:.4f}")
    print()
    print(f"  ── Metrik untuk Kelas SALAH (1) ──")
    print(f"  True  Positive (TP)       : {tp:>6,}  (Frame Salah → Terdeteksi Salah ✓)")
    print(f"  True  Negative (TN)       : {tn:>6,}  (Frame Benar → Terdeteksi Benar ✓)")
    print(f"  False Positive (FP)       : {fp:>6,}  (Frame Benar → Dituduh Salah  ✗)")
    print(f"  False Negative (FN)       : {fn:>6,}  (Frame Salah → Tidak Terdeteksi ✗)")
    print()
    print(f"  Precision (Salah)         : {precision_salah:.4f}  (Ketepatan saat menuduh Salah)")
    print(f"  Recall    (Salah)         : {recall_salah:.4f}  (Kemampuan menemukan frame Salah)")
    print(f"  F1-Score  (Salah)         : {f1_salah:.4f}")
    print(f"{'='*70}")

    print("\n  CONFUSION MATRIX (Baris=Ground Truth, Kolom=Prediksi):")
    print(f"  {'':15s} {'Prediksi BENAR':>15s} {'Prediksi SALAH':>15s}")
    print(f"  {'Ground Truth BENAR':15s} {tn:>15,} {fp:>15,}")
    print(f"  {'Ground Truth SALAH':15s} {fn:>15,} {tp:>15,}")
    print(f"{'='*70}")

    print("\n  LAPORAN KLASIFIKASI LENGKAP:")
    print(classification_report(all_truths, all_preds, target_names=["BENAR (0)", "SALAH (1)"]))

    # ============================================================
    # Bobot Atensi Per-Sendi (jika BSP aktif)
    # ============================================================
    if has_bsp:
        bsp_weights = torch.sigmoid(model.biomechanical_spatial_prior)
        bsp_weights = bsp_weights.detach().cpu().squeeze().numpy()  # (33,)

        print(f"\n{'='*70}")
        print("  BOBOT ATENSI SPATIAL (Biomechanical Spatial Prior) per Sendi")
        print(f"  (Nilai mendekati 1.0 = Model sangat memerhatikan sendi ini)")
        print(f"{'='*70}")

        sendi_ranked = sorted(
            zip(NAMA_SENDI_33, bsp_weights),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"\n  {'Peringkat':>3} {'Nama Sendi':30s} {'Bobot Atensi':>12s} {'Bar':30s}")
        print(f"  {'-'*3} {'-'*30} {'-'*12} {'-'*30}")
        for rank, (nama, bobot) in enumerate(sendi_ranked, 1):
            bar = "█" * int(bobot * 30)
            print(f"  #{rank:>2}  {nama:30s} {bobot:>12.6f} {bar}")

        # Simpan ke CSV
        df_atensi = pd.DataFrame({
            "Peringkat"    : range(1, 34),
            "Nama_Sendi"   : [s[0] for s in sendi_ranked],
            "Bobot_Atensi" : [s[1] for s in sendi_ranked],
        })
        out_atensi = PROJECT_ROOT / "data" / "processed" / f"bobot_atensi_{model_path.stem}.csv"
        df_atensi.to_csv(out_atensi, index=False)
        print(f"\n  [TERSIMPAN] Tabel bobot atensi → {out_atensi}")

    # Simpan ringkasan metrik ke CSV
    df_metrik = pd.DataFrame([{
        "Model"         : model_path.name,
        "Accuracy"      : round(acc, 4),
        "F1_Macro"      : round(f1_macro, 4),
        "F1_Salah"      : round(f1_salah, 4),
        "Precision_Salah": round(precision_salah, 4),
        "Recall_Salah"  : round(recall_salah, 4),
        "TP"            : int(tp),
        "TN"            : int(tn),
        "FP"            : int(fp),
        "FN"            : int(fn),
        "Total_Frames"  : int(total_frames),
    }])

    out_metrik = PROJECT_ROOT / "data" / "processed" / f"metrik_{model_path.stem}.csv"
    df_metrik.to_csv(out_metrik, index=False)
    print(f"\n  [TERSIMPAN] Ringkasan metrik → {out_metrik}")
    print(f"{'='*70}\n")

    return all_preds, all_truths, all_probs


def main():
    parser = argparse.ArgumentParser(description="Evaluasi Model AttentiveSkel-3D v2")
    parser.add_argument(
        "--model",
        type=str,
        default="bobot_model/best_model_v2.pth",
        help="Path ke model .pth (default: best_model_v2.pth)"
    )
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        sys.exit(1)

    run_evaluation(model_path)


if __name__ == "__main__":
    main()
