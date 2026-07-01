# src/models/kfold_training.py
#
# Modul 5-Fold Cross Validation untuk evaluasi komparatif seluruh skenario
# arsitektur AttentiveSkel-3D (Baseline, Ablasi A–C, Full Model).
#
# Fungsi utama:
#   run_kfold_experiment — eksekusi K-Fold CV lintas skenario,
#                          mengembalikan Pandas DataFrame ringkasan hasil.

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset

# ── Import internal proyek ────────────────────────────────────────────────────
# Tambahkan root proyek ke sys.path jika dijalankan langsung (bukan sebagai modul)
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import WeightTrainingDataset
from src.models.model_3dcnn import AttentiveSkel3D


# =============================================================================
# Definisi Skenario Arsitektur
# =============================================================================

#: Dictionary nama skenario → keyword-argument konstruktor AttentiveSkel3D.
#: Urutan ini menentukan urutan baris pada DataFrame output.
SCENARIOS: dict[str, dict[str, Any]] = {
    "Baseline 3D-CNN": {
        "use_attention":          False,
        "use_spatial_prior":      False,
        "use_learned_spatial":    False,
        "use_temporal_attention": False,
    },
    "Ablasi A — Tanpa Prior": {
        "use_attention":          True,
        "use_spatial_prior":      False,
        "use_learned_spatial":    True,
        "use_temporal_attention": True,
    },
    "Ablasi B — Tanpa Learned Spatial": {
        "use_attention":          True,
        "use_spatial_prior":      True,
        "use_learned_spatial":    False,
        "use_temporal_attention": True,
    },
    "Ablasi C — Tanpa Temporal": {
        "use_attention":          True,
        "use_spatial_prior":      True,
        "use_learned_spatial":    True,
        "use_temporal_attention": False,
    },
    "Full AttentiveSkel-3D": {
        "use_attention":          True,
        "use_spatial_prior":      True,
        "use_learned_spatial":    True,
        "use_temporal_attention": True,
    },
}


# =============================================================================
# Helper: satu epoch train
# =============================================================================

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Forward + backward satu epoch. Mengembalikan (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        n          += x.size(0)

    return total_loss / n, correct / n


# =============================================================================
# Helper: satu epoch evaluasi
# =============================================================================

def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluasi tanpa gradient. Mengembalikan (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    n          = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            n          += x.size(0)

    return total_loss / n, correct / n


# =============================================================================
# Fungsi Utama: run_kfold_experiment
# =============================================================================

def run_kfold_experiment(
    dataset_manifest_path: str | Path = "data/processed/dataset_manifest.csv",
    num_folds:  int   = 5,
    epochs:     int   = 50,
    batch_size: int   = 16,
    device:     str   = "cuda",
    lr:         float = 1e-3,
    num_workers: int  = 0,
    random_state: int = 42,
    verbose:    bool  = True,
) -> pd.DataFrame:
    """
    Jalankan K-Fold Cross Validation untuk seluruh skenario arsitektur.

    Untuk setiap skenario × fold, sebuah model baru diinisialisasi dari nol,
    dilatih selama ``epochs`` epoch, dan akurasi validasi terbaik (bukan
    akurasi epoch terakhir) dicatat.  Bobot model *tidak* disimpan ke disk —
    tujuan eksperimen ini murni evaluasi komparatif.

    Args:
        dataset_manifest_path : Path ke ``dataset_manifest.csv``.
        num_folds  : Jumlah fold K-Fold. Default = 5.
        epochs     : Jumlah epoch pelatihan per fold. Default = 50.
        batch_size : Ukuran batch DataLoader. Default = 16.
        device     : ``"cuda"`` atau ``"cpu"``. Otomatis fallback ke CPU jika
                     CUDA tidak tersedia.
        lr         : Learning-rate untuk optimizer Adam. Default = 1e-3.
        num_workers: Jumlah worker DataLoader. Gunakan 0 di Windows/Colab.
        random_state: Seed untuk KFold (reproducibility).
        verbose    : Cetak progres fold dan epoch ke stdout.

    Returns:
        pd.DataFrame dengan kolom:
            ['Skenario', 'Fold 1', …, 'Fold N', 'Mean Accuracy', 'Std Deviation']
        Nilai akurasi dalam rentang 0.0–1.0.

    Raises:
        FileNotFoundError : Jika manifest CSV tidak ditemukan.
        RuntimeError      : Jika dataset terlalu kecil untuk di-fold.
    """
    # ── Resolusi path & device ───────────────────────────────────────────────
    manifest_path = Path(dataset_manifest_path)
    if not manifest_path.is_absolute():
        # Coba relatif terhadap root proyek terlebih dahulu
        candidate = _PROJECT_ROOT / manifest_path
        if candidate.exists():
            manifest_path = candidate

    _device = torch.device(
        "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    if verbose:
        print(f"{'='*62}")
        print(f"  5-Fold Cross Validation — AttentiveSkel-3D")
        print(f"{'='*62}")
        print(f"  Manifest  : {manifest_path}")
        print(f"  Device    : {_device}  (diminta: {device})")
        print(f"  Folds     : {num_folds}")
        print(f"  Epochs    : {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  LR (Adam) : {lr}")
        print()

    # ── Muat dataset penuh (tanpa split manual) ──────────────────────────────
    full_dataset = WeightTrainingDataset(csv_file=manifest_path)
    n_samples    = len(full_dataset)

    if verbose:
        print(f"  Total sampel dataset : {n_samples}")

    if n_samples < num_folds * 2:
        raise RuntimeError(
            f"Dataset terlalu kecil ({n_samples} sampel) untuk {num_folds}-Fold CV. "
            "Butuh minimal 2 sampel per fold."
        )

    # Kumpulkan label untuk StratifiedKFold
    all_labels = np.array(
        [int(full_dataset.manifest.iloc[i]["label"]) for i in range(n_samples)]
    )
    unique, counts = np.unique(all_labels, return_counts=True)
    if verbose:
        dist_str = ", ".join(f"kelas {k}={v}" for k, v in zip(unique, counts))
        print(f"  Distribusi label     : {dist_str}")
        print()

    # Gunakan StratifiedKFold agar distribusi kelas seimbang di setiap fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    fold_splits = list(skf.split(np.zeros(n_samples), all_labels))

    # ── Inisialisasi wadah hasil ─────────────────────────────────────────────
    fold_col_names = [f"Fold {f+1}" for f in range(num_folds)]
    rows: list[dict] = []

    t_global_start = time.time()

    # ── Loop skenario ─────────────────────────────────────────────────────────
    for scenario_name, model_kwargs in SCENARIOS.items():
        if verbose:
            print(f"{'─'*62}")
            print(f"  Skenario : {scenario_name}")
            print(f"{'─'*62}")

        fold_accuracies: list[float] = []
        t_scenario_start = time.time()

        # ── Loop fold ────────────────────────────────────────────────────────
        for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
            fold_num = fold_idx + 1

            # ── Buat DataLoader fold ─────────────────────────────────────────
            train_subset = Subset(full_dataset, train_indices)
            val_subset   = Subset(full_dataset, val_indices)

            train_loader = DataLoader(
                train_subset,
                batch_size  = batch_size,
                shuffle     = True,
                num_workers = num_workers,
                pin_memory  = (_device.type == "cuda"),
                drop_last   = False,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size  = batch_size * 2,   # validasi bisa batch lebih besar
                shuffle     = False,
                num_workers = num_workers,
                pin_memory  = (_device.type == "cuda"),
            )

            # ── Inisialisasi model BARU di setiap fold (reset bobot penuh) ───
            model = AttentiveSkel3D(num_classes=2, **model_kwargs).to(_device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=1e-4
            )

            # Scheduler: kurangi LR jika val_loss stagnan selama 10 epoch
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, verbose=False
            )

            best_val_acc  = 0.0
            best_val_loss = float("inf")

            # ── Loop epoch ───────────────────────────────────────────────────
            for epoch in range(1, epochs + 1):
                train_loss, train_acc = _train_epoch(
                    model, train_loader, criterion, optimizer, _device
                )
                val_loss, val_acc = _eval_epoch(
                    model, val_loader, criterion, _device
                )
                scheduler.step(val_loss)

                # Simpan akurasi validasi terbaik berdasarkan val_loss terendah
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc  = val_acc

                if verbose and (epoch % 10 == 0 or epoch == 1):
                    print(
                        f"    Fold {fold_num}/{num_folds} | "
                        f"Epoch {epoch:3d}/{epochs} | "
                        f"TrainLoss: {train_loss:.4f}  TrainAcc: {train_acc:.3f} | "
                        f"ValLoss: {val_loss:.4f}  ValAcc: {val_acc:.3f}  "
                        f"[Best ValAcc: {best_val_acc:.3f}]"
                    )

            fold_accuracies.append(best_val_acc)

            if verbose:
                print(
                    f"  ✔ Fold {fold_num} selesai — "
                    f"Best Val Accuracy: {best_val_acc:.4f} "
                    f"({best_val_acc * 100:.1f}%)"
                )

            # Bebaskan memori GPU setelah setiap fold
            del model, optimizer, scheduler, criterion
            del train_loader, val_loader, train_subset, val_subset
            if _device.type == "cuda":
                torch.cuda.empty_cache()

        # ── Rekap skenario ───────────────────────────────────────────────────
        mean_acc = float(np.mean(fold_accuracies))
        std_acc  = float(np.std(fold_accuracies, ddof=1))   # ddof=1 → sample std
        t_elapsed = time.time() - t_scenario_start

        row: dict = {"Skenario": scenario_name}
        for i, acc in enumerate(fold_accuracies):
            row[f"Fold {i+1}"] = round(acc, 4)
        row["Mean Accuracy"] = round(mean_acc, 4)
        row["Std Deviation"]  = round(std_acc, 4)
        rows.append(row)

        if verbose:
            fold_str = "  ".join(f"{a*100:.1f}%" for a in fold_accuracies)
            print(
                f"\n  Ringkasan [{scenario_name}]\n"
                f"    Folds  : {fold_str}\n"
                f"    Mean   : {mean_acc*100:.2f}%  ±  {std_acc*100:.2f}%\n"
                f"    Waktu  : {t_elapsed:.1f} detik\n"
            )

    # ── Bangun DataFrame akhir ───────────────────────────────────────────────
    columns = ["Skenario"] + fold_col_names + ["Mean Accuracy", "Std Deviation"]
    df = pd.DataFrame(rows, columns=columns)

    t_total = time.time() - t_global_start
    if verbose:
        print(f"{'='*62}")
        print(f"  Eksperimen selesai.  Total waktu: {t_total:.1f} detik")
        print(f"{'='*62}\n")

    return df


# =============================================================================
# Entry point: jalankan langsung dengan python src/models/kfold_training.py
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="5-Fold Cross Validation — AttentiveSkel-3D"
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/dataset_manifest.csv",
        help="Path ke dataset_manifest.csv (relatif terhadap root proyek)",
    )
    parser.add_argument("--folds",      type=int,   default=5,      help="Jumlah fold")
    parser.add_argument("--epochs",     type=int,   default=100,     help="Epoch per fold")
    parser.add_argument("--batch-size", type=int,   default=16,     help="Batch size")
    parser.add_argument("--device",     default="cuda",             help="cuda / cpu")
    parser.add_argument("--lr",         type=float, default=1e-3,   help="Learning rate")
    parser.add_argument(
        "--output",
        default="data/processed/100_kfold_results.csv",
        help="Path output CSV hasil (opsional)",
    )
    args = parser.parse_args()

    result_df = run_kfold_experiment(
        dataset_manifest_path = args.manifest,
        num_folds             = args.folds,
        epochs                = args.epochs,
        batch_size            = args.batch_size,
        device                = args.device,
        lr                    = args.lr,
    )

    print(result_df.to_string(index=False))

    out_path = _PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\nHasil tersimpan ke: {out_path}")
