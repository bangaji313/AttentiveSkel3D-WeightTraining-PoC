# src/models/train.py
#
# Modul training loop standar PyTorch untuk model AttentiveSkel-3D.
#
# Fungsi utama:
#   train_model — menjalankan training & validasi per epoch, menyimpan bobot terbaik
#
# Konvensi:
#   - Model terbaik ditentukan berdasarkan val_loss terendah
#   - Bobot disimpan ke models/saved_models/best_model.pth
#   - Semua metric dikembalikan sebagai dict history untuk kemudahan plotting

import os
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Menjalankan satu epoch fase pelatihan (forward → loss → backward → update).

    Args:
        model     : Model PyTorch yang akan dilatih.
        loader    : DataLoader untuk data pelatihan.
        criterion : Fungsi loss (misalnya CrossEntropyLoss).
        optimizer : Optimizer (misalnya Adam).
        device    : Device target (cpu / cuda).

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
            - epoch_loss     : Rata-rata loss per sampel pada epoch ini.
            - epoch_accuracy : Akurasi klasifikasi pada data train (0.0–1.0).
    """
    model.train()  # Aktifkan mode train (dropout & batchnorm berjalan normal)

    running_loss    = 0.0
    correct_preds   = 0
    total_samples   = 0

    for batch_data, batch_labels in loader:
        # Pindahkan data ke device yang sesuai
        batch_data   = batch_data.to(device)    # (B, 64, 33, 3)
        batch_labels = batch_labels.to(device)  # (B,)

        # --- Forward pass ---
        optimizer.zero_grad()            # Reset gradient sebelum perhitungan baru
        logits = model(batch_data)       # (B, num_classes) → logit mentah

        # --- Hitung loss ---
        loss = criterion(logits, batch_labels)

        # --- Backward pass & update bobot ---
        loss.backward()
        optimizer.step()

        # --- Kumpulkan statistik ---
        running_loss  += loss.item() * batch_data.size(0)   # Akumulasi loss total
        preds          = logits.argmax(dim=1)               # Prediksi kelas
        correct_preds += (preds == batch_labels).sum().item()
        total_samples += batch_data.size(0)

    # Hitung rata-rata loss dan akurasi keseluruhan epoch
    epoch_loss     = running_loss / total_samples
    epoch_accuracy = correct_preds / total_samples

    return epoch_loss, epoch_accuracy


def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Menjalankan satu epoch fase evaluasi (tanpa gradient, tanpa update bobot).

    Args:
        model     : Model PyTorch yang dievaluasi.
        loader    : DataLoader untuk data validasi atau uji.
        criterion : Fungsi loss yang sama dengan saat pelatihan.
        device    : Device target (cpu / cuda).

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.eval()  # Nonaktifkan dropout; batchnorm gunakan statistik running

    running_loss  = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():  # Tidak perlu hitung gradient untuk evaluasi
        for batch_data, batch_labels in loader:
            batch_data   = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_data)
            loss   = criterion(logits, batch_labels)

            running_loss  += loss.item() * batch_data.size(0)
            preds          = logits.argmax(dim=1)
            correct_preds += (preds == batch_labels).sum().item()
            total_samples += batch_data.size(0)

    epoch_loss     = running_loss / total_samples
    epoch_accuracy = correct_preds / total_samples

    return epoch_loss, epoch_accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: str | Path = "models/saved_models",
    save_filename: str = "best_model.pth",
    verbose: bool = True,
) -> dict:
    """
    Loop pelatihan penuh: melatih model selama `num_epochs` epoch, melakukan
    validasi di setiap akhir epoch, dan menyimpan bobot model terbaik.

    Kriteria "terbaik": val_loss terendah selama pelatihan.

    Args:
        model         : Model PyTorch (nn.Module) yang akan dilatih.
        train_loader  : DataLoader untuk data pelatihan.
        val_loader    : DataLoader untuk data validasi.
        criterion     : Fungsi loss (misalnya nn.CrossEntropyLoss()).
        optimizer     : Optimizer (misalnya torch.optim.Adam).
        num_epochs    : Jumlah total epoch pelatihan.
        device        : Device target ('cpu' atau 'cuda').
        save_dir      : Direktori untuk menyimpan bobot model terbaik.
        save_filename : Nama file bobot model terbaik (.pth).
        verbose       : Jika True, cetak log per epoch ke konsol.

    Returns:
        dict: Riwayat metric pelatihan dengan kunci:
            - 'train_loss'    : list loss per epoch (data train)
            - 'train_acc'     : list akurasi per epoch (data train)
            - 'val_loss'      : list loss per epoch (data validasi)
            - 'val_acc'       : list akurasi per epoch (data validasi)
            - 'best_epoch'    : epoch dengan val_loss terbaik
            - 'best_val_loss' : nilai val_loss terbaik
    """
    save_dir  = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    # Inisialisasi history metric untuk setiap epoch
    history = {
        "train_loss": [],
        "train_acc" : [],
        "val_loss"  : [],
        "val_acc"   : [],
        "best_epoch"    : 0,
        "best_val_loss" : float("inf"),
    }

    # Simpan salinan bobot model terbaik di memori (untuk efisiensi)
    best_model_weights = copy.deepcopy(model.state_dict())

    if verbose:
        print(f"{'='*70}")
        print(f"  Memulai pelatihan AttentiveSkel-3D")
        print(f"  Device    : {device}")
        print(f"  Epochs    : {num_epochs}")
        print(f"  Save path : {save_path}")
        print(f"{'='*70}")

    total_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        # ── Fase 1: Training ──────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # ── Fase 2: Validasi ──────────────────────────────────────────────────
        val_loss, val_acc = evaluate_one_epoch(
            model, val_loader, criterion, device
        )

        epoch_duration = time.time() - epoch_start_time

        # ── Simpan metric ke history ───────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Simpan model terbaik berdasarkan val_loss terendah ─────────────────
        improved = val_loss < history["best_val_loss"]
        if improved:
            history["best_val_loss"] = val_loss
            history["best_epoch"]    = epoch
            best_model_weights       = copy.deepcopy(model.state_dict())

            # Simpan ke disk langsung saat ada peningkatan
            torch.save(
                {
                    "epoch"     : epoch,
                    "model_state_dict"    : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss"  : val_loss,
                    "val_acc"   : val_acc,
                },
                save_path,
            )

        # ── Log per epoch ─────────────────────────────────────────────────────
        if verbose:
            marker = " ✓" if improved else "  "
            print(
                f"Epoch [{epoch:>3}/{num_epochs}]{marker} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:>6.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:>6.2f}% | "
                f"Waktu: {epoch_duration:.1f}s"
            )

    # ── Pelatihan selesai ──────────────────────────────────────────────────────
    total_duration = time.time() - total_start_time

    if verbose:
        print(f"{'='*70}")
        print(f"  Pelatihan selesai dalam {total_duration:.1f} detik.")
        print(f"  Epoch terbaik : {history['best_epoch']}  (Val Loss = {history['best_val_loss']:.4f})")
        print(f"  Model terbaik disimpan ke: {save_path}")
        print(f"{'='*70}")

    # Muat kembali bobot model terbaik ke model saat ini
    model.load_state_dict(best_model_weights)

    return history
