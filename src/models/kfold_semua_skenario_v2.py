# ============================================================
# src/models/kfold_semua_skenario_v2.py
#
# Skrip K-Fold Cross Validation SEMUA Skenario — AttentiveSkel-3D V2
#
# Tujuan:
#   Menjalankan Stratified 5-Fold Cross Validation untuk SEMUA skenario
#   model (Baseline, Full, dan semua Ablasi) secara berurutan, kemudian
#   merangkum dan membandingkan hasil antar skenario dalam satu CSV rapi.
#
# Catatan Penting — Perbedaan dengan Evaluasi Biasa:
#   - Evaluasi biasa   : Model sudah dilatih, lalu diuji pada test set tetap.
#   - Cross Validation : Model dilatih ulang dari nol (scratch) sebanyak K kali
#                        dengan pembagian data yang berbeda-beda, lalu kinerjanya
#                        dirata-ratakan. Tujuannya adalah mengukur STABILITAS
#                        dan GENERALISASI model, bukan hanya performa terbaik.
#
# Protokol:
#   - K = 5 fold (Stratified, menjaga proporsi label tiap fold seimbang)
#   - Setiap fold: train dari scratch, evaluasi pada fold validasi
#   - Metrik yang dicatat: Accuracy & F1-Score per fold, lalu dihitung rata-rata ± std
#   - Epoch per fold: 30 (cukup untuk konvergensi, hemat waktu komputasi)
#
# Cara menjalankan:
#   Pastikan Anda berada di dalam folder Release_V2_AttentiveSkel3D:
#
#   conda activate attentiveskel
#   python src/models/kfold_semua_skenario_v2.py
#
#   Estimasi waktu: ~15-45 menit tergantung GPU (5 fold × 5 skenario × 30 epoch)
#
# Output yang dihasilkan (di dalam folder hasil_evaluasi/):
#   1. KFold_Hasil_Per_Fold_V2.csv        — Detail akurasi tiap fold tiap skenario
#   2. KFold_Ringkasan_Semua_Skenario_V2.csv — Mean ± Std semua skenario
# ============================================================

import sys
import copy
import time
from pathlib import Path

# Direktori root adalah satu level di atas folder src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.models.arsitektur_v2 import AttentiveSkel3DPerFrame
from src.data.dataset_v2 import PerFrameDataset

# ============================================================
# Konfigurasi Eksperimen
# ============================================================

# Path manifest dataset
MANIFEST_PATH = PROJECT_ROOT / "src" / "data" / "manifest_v2.csv"

# Direktori output
OUTPUT_DIR = PROJECT_ROOT / "hasil_evaluasi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameter K-Fold ──────────────────────────────────
K_FOLD          = 5     # Jumlah lipatan (fold)
EPOCH_PER_FOLD  = 100   # Jumlah epoch pelatihan per fold (SAMA dengan skenario training asli)
BATCH_SIZE      = 16    # Ukuran batch
LEARNING_RATE   = 1e-3  # Learning rate awal
WEIGHT_DECAY    = 1e-4  # Regularisasi L2
RANDOM_SEED     = 42    # Seed untuk reproduksibilitas

# ── Definisi Skenario ─────────────────────────────────────
# Setiap skenario mendefinisikan konfigurasi fitur atensi.
# Model DILATIH ULANG dari scratch pada setiap fold — tidak menggunakan
# bobot dari file .pth yang sudah ada.
SKENARIO = {
    "Skenario 1 — Baseline": {
        "use_spatial_prior"     : False,
        "use_learned_spatial"   : False,
        "use_temporal_attention": False,
    },
    "Skenario 2 — Full Model": {
        "use_spatial_prior"     : True,
        "use_learned_spatial"   : True,
        "use_temporal_attention": True,
    },
    "Skenario 3a — Ablasi: Hanya BSP": {
        "use_spatial_prior"     : True,
        "use_learned_spatial"   : False,
        "use_temporal_attention": False,
    },
    "Skenario 3b — Ablasi: BSP + LearnedSpatial": {
        "use_spatial_prior"     : True,
        "use_learned_spatial"   : True,
        "use_temporal_attention": False,
    },
    "Skenario 3c — Ablasi: BSP + Temporal": {
        "use_spatial_prior"     : True,
        "use_learned_spatial"   : False,
        "use_temporal_attention": True,
    },
}


def buat_model_baru(config: dict) -> AttentiveSkel3DPerFrame:
    """
    Membuat instance model baru dari nol (random initialization)
    berdasarkan konfigurasi fitur atensi yang diberikan.

    Parameter:
        config (dict): Dictionary berisi kunci use_spatial_prior,
                       use_learned_spatial, use_temporal_attention.

    Mengembalikan:
        Model AttentiveSkel3DPerFrame yang baru diinisialisasi.
    """
    return AttentiveSkel3DPerFrame(
        num_classes=2,
        use_spatial_prior=config["use_spatial_prior"],
        use_learned_spatial=config["use_learned_spatial"],
        use_temporal_attention=config["use_temporal_attention"],
    )


def latih_satu_epoch(model, loader, optimizer, criterion, device):
    """
    Melatih model selama satu epoch pada data yang diberikan.

    Setiap batch di-forward, loss dihitung, lalu gradient di-backward.
    Output model (B, 64, 2) dan label (B, 64) di-flatten sebelum komputasi loss
    agar setiap frame dihitung sebagai sampel independen.

    Mengembalikan:
        loss_rata_rata (float): Rata-rata training loss seluruh batch.
    """
    model.train()
    total_loss = 0.0

    for batch_input, batch_labels in loader:
        batch_input  = batch_input.float().to(device)   # (B, 64, 33, 3)
        batch_labels = batch_labels.long().to(device)   # (B, 64)

        optimizer.zero_grad()
        logits = model(batch_input)                     # (B, 64, 2)

        # Flatten untuk komputasi loss: (B*64, 2) vs (B*64,)
        loss = criterion(
            logits.view(-1, 2),
            batch_labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluasi_fold(model, loader, device):
    """
    Mengevaluasi model pada data validasi/test satu fold.

    Model berjalan dalam mode eval() (dropout & BN dimatikan).
    Prediksi per-frame diakumulasikan lalu dihitung metriknya.

    Mengembalikan:
        accuracy (float): Akurasi per-frame dalam persentase (0-100).
        f1_macro (float): F1-Score Macro.
        f1_binary(float): F1-Score Binary (kelas SALAH=1 sebagai positif).
    """
    model.eval()
    semua_preds  = []
    semua_truths = []

    with torch.no_grad():
        for batch_input, batch_labels in loader:
            batch_input  = batch_input.float().to(device)
            batch_labels = batch_labels.long().to(device)

            logits = model(batch_input)              # (B, 64, 2)
            preds  = logits.argmax(dim=2)            # (B, 64)

            semua_preds.extend(preds.cpu().numpy().flatten().tolist())
            semua_truths.extend(batch_labels.cpu().numpy().flatten().tolist())

    arr_preds  = np.array(semua_preds)
    arr_truths = np.array(semua_truths)

    accuracy  = accuracy_score(arr_truths, arr_preds) * 100
    f1_macro  = f1_score(arr_truths, arr_preds, average="macro",  zero_division=0)
    f1_binary = f1_score(arr_truths, arr_preds, average="binary", pos_label=1, zero_division=0)

    return round(accuracy, 2), round(f1_macro, 4), round(f1_binary, 4)


def jalankan_kfold_satu_skenario(nama_skenario, config, dataset, label_global, device):
    """
    Menjalankan K-Fold Cross Validation penuh untuk SATU skenario.

    Proses:
        1. StratifiedKFold membagi indeks dataset menjadi K pasangan (train, val).
        2. Untuk setiap fold, model baru dibuat, dilatih, lalu dievaluasi.
        3. Hasil tiap fold dikumpulkan untuk dihitung statistiknya.

    Parameter:
        nama_skenario (str)   : Nama skenario untuk label output.
        config        (dict)  : Konfigurasi fitur atensi model.
        dataset               : Dataset PyTorch penuh (PerFrameDataset).
        label_global  (array) : Label global tiap video (untuk stratifikasi).
        device                : Device PyTorch.

    Mengembalikan:
        hasil_fold (list): Daftar dict hasil per fold.
    """
    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_SEED)
    criterion = nn.CrossEntropyLoss()

    hasil_fold = []
    jumlah_param = None  # Akan dihitung dari model pertama

    print(f"\n  {'─'*70}")
    print(f"  {nama_skenario}")
    print(f"  {'─'*70}")

    for idx_fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(dataset)), label_global), start=1):

        t_mulai = time.time()

        # ── Buat DataLoader untuk fold ini ────────────────
        train_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=SubsetRandomSampler(train_idx),
        )
        val_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=SubsetRandomSampler(val_idx),
        )

        # ── Inisialisasi model baru dari scratch ──────────
        model = buat_model_baru(config).to(device)
        if jumlah_param is None:
            jumlah_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5, verbose=False
        )

        # ── Pelatihan ─────────────────────────────────────
        val_loss_terbaik = float("inf")
        state_terbaik    = None

        for epoch in range(1, EPOCH_PER_FOLD + 1):
            loss_train = latih_satu_epoch(model, train_loader, optimizer, criterion, device)

            # Hitung validation loss untuk scheduler
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for b_in, b_lbl in val_loader:
                    b_in  = b_in.float().to(device)
                    b_lbl = b_lbl.long().to(device)
                    out   = model(b_in)
                    total_val_loss += criterion(out.view(-1, 2), b_lbl.view(-1)).item()
            loss_val = total_val_loss / len(val_loader)
            scheduler.step(loss_val)

            # Simpan state terbaik berdasarkan val loss
            if loss_val < val_loss_terbaik:
                val_loss_terbaik = loss_val
                state_terbaik    = copy.deepcopy(model.state_dict())

            # Cetak progres setiap 10 epoch
            if epoch % 10 == 0 or epoch == EPOCH_PER_FOLD:
                print(f"    Fold {idx_fold}/{K_FOLD}  Epoch {epoch:>3}/{EPOCH_PER_FOLD}"
                      f"  TrainLoss={loss_train:.4f}  ValLoss={loss_val:.4f}")

        # ── Evaluasi menggunakan state terbaik ────────────
        model.load_state_dict(state_terbaik)
        akurasi, f1_mac, f1_bin = evaluasi_fold(model, val_loader, device)

        durasi = time.time() - t_mulai
        print(f"    [Fold {idx_fold}] Akurasi={akurasi:.2f}%  F1 Macro={f1_mac:.4f}"
              f"  F1 Binary={f1_bin:.4f}  Durasi={durasi:.1f}s")

        hasil_fold.append({
            "Skenario"      : nama_skenario,
            "Fold"          : idx_fold,
            "Accuracy (%)"  : akurasi,
            "F1_Macro"      : f1_mac,
            "F1_Binary"     : f1_bin,
            "Val_Loss_Terbaik": round(val_loss_terbaik, 6),
            "Jumlah_Parameter": jumlah_param,
        })

    # Ringkasan fold untuk skenario ini
    arr_acc = np.array([r["Accuracy (%)"] for r in hasil_fold])
    arr_f1  = np.array([r["F1_Macro"]     for r in hasil_fold])
    print(f"\n  Ringkasan {K_FOLD} Fold:")
    print(f"    Accuracy  : {arr_acc.mean():.2f}% ± {arr_acc.std():.2f}%")
    print(f"    F1 Macro  : {arr_f1.mean():.4f} ± {arr_f1.std():.4f}")

    return hasil_fold


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("  K-FOLD CROSS VALIDATION SEMUA SKENARIO — AttentiveSkel-3D V2")
    print("=" * 75)
    print(f"  K              : {K_FOLD} Fold")
    print(f"  Epoch per fold : {EPOCH_PER_FOLD}")
    print(f"  Device         : {device}")
    print(f"  Manifest       : {MANIFEST_PATH}")
    print()

    # Verifikasi manifest tersedia
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest tidak ditemukan: {MANIFEST_PATH}")
        print("  Pastikan file manifest_v2.csv ada di dalam folder src/data/")
        sys.exit(1)

    # ============================================================
    # Muat Dataset sekali — dibagi-bagi oleh KFold di dalam loop
    # ============================================================
    print("[INFO] Memuat dataset penuh...")
    dataset_penuh = PerFrameDataset(csv_file=MANIFEST_PATH)
    print(f"[INFO] Dataset dimuat. Jumlah sampel (video): {len(dataset_penuh)}\n")

    # Label global per-video: diambil dari label frame mayoritas (bisa disesuaikan)
    # Ini digunakan hanya untuk STRATIFIKASI pembagian fold, bukan untuk training.
    label_global = []
    for i in range(len(dataset_penuh)):
        _, labels = dataset_penuh[i]  # labels shape: (64,)
        # Label video = kelas mayoritas di antara 64 frame
        label_global.append(int(torch.mode(labels).values.item()))
    label_global = np.array(label_global)

    # ============================================================
    # Jalankan K-Fold untuk setiap skenario
    # ============================================================
    semua_hasil_fold     = []  # Detail per fold
    semua_ringkasan      = []  # Ringkasan mean ± std

    for nama_skenario, config in SKENARIO.items():
        hasil_fold = jalankan_kfold_satu_skenario(
            nama_skenario, config, dataset_penuh, label_global, device
        )
        semua_hasil_fold.extend(hasil_fold)

        # Hitung ringkasan statistik untuk skenario ini
        arr_acc    = np.array([r["Accuracy (%)"] for r in hasil_fold])
        arr_f1_mac = np.array([r["F1_Macro"]     for r in hasil_fold])
        arr_f1_bin = np.array([r["F1_Binary"]    for r in hasil_fold])

        semua_ringkasan.append({
            "Skenario"        : nama_skenario,
            "BSP"             : "✓" if config["use_spatial_prior"]      else "✗",
            "LearnedSpatial"  : "✓" if config["use_learned_spatial"]    else "✗",
            "Temporal"        : "✓" if config["use_temporal_attention"] else "✗",
            "Acc_Mean (%)"    : round(arr_acc.mean(), 2),
            "Acc_Std (%)"     : round(arr_acc.std(),  2),
            "F1_Macro_Mean"   : round(arr_f1_mac.mean(), 4),
            "F1_Macro_Std"    : round(arr_f1_mac.std(),  4),
            "F1_Binary_Mean"  : round(arr_f1_bin.mean(), 4),
            "F1_Binary_Std"   : round(arr_f1_bin.std(),  4),
            "Jumlah_Fold"     : K_FOLD,
            "Epoch_per_Fold"  : EPOCH_PER_FOLD,
        })

    # ============================================================
    # Ekspor Hasil ke CSV
    # ============================================================
    print("\n" + "=" * 75)
    print("  EKSPOR HASIL KE CSV")
    print("=" * 75)

    # 1. Detail setiap fold setiap skenario
    df_per_fold = pd.DataFrame(semua_hasil_fold)
    path_per_fold = OUTPUT_DIR / "KFold_Hasil_Per_Fold_V2.csv"
    df_per_fold.to_csv(path_per_fold, index=False, encoding="utf-8-sig")
    print(f"\n  [TERSIMPAN] Hasil per fold → {path_per_fold.name}")

    # 2. Ringkasan komparatif
    df_ringkasan = pd.DataFrame(semua_ringkasan)
    path_ringkasan = OUTPUT_DIR / "KFold_Ringkasan_Semua_Skenario_V2.csv"
    df_ringkasan.to_csv(path_ringkasan, index=False, encoding="utf-8-sig")
    print(f"  [TERSIMPAN] Ringkasan semua skenario → {path_ringkasan.name}")

    # Tampilkan tabel perbandingan akhir
    print("\n  PERBANDINGAN AKHIR (Mean ± Std dari 5 Fold):")
    print(f"  {'Skenario':50s} {'Accuracy':>15s} {'F1 Macro':>12s}")
    print(f"  {'─'*50} {'─'*15} {'─'*12}")
    for baris in semua_ringkasan:
        nama = baris['Skenario'][:50]
        acc_str = f"{baris['Acc_Mean (%)']:.2f}% ± {baris['Acc_Std (%)']:.2f}%"
        f1_str  = f"{baris['F1_Macro_Mean']:.4f} ± {baris['F1_Macro_Std']:.4f}"
        print(f"  {nama:50s} {acc_str:>15s} {f1_str:>12s}")

    print(f"\n  Semua file tersimpan di folder: {OUTPUT_DIR}")
    print("=" * 75)
    print("  K-FOLD CROSS VALIDATION SELESAI.")
    print("=" * 75)


if __name__ == "__main__":
    main()
