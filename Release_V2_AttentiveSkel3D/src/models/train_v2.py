# src/models/v2/train.py
#
# Training loop untuk model per-frame v2.
#
# PERBEDAAN dari v1/train.py:
#   - Output model berbentuk (B, 64, 2), bukan (B, 2).
#   - Target label berbentuk (B, 64), bukan (B,).
#   - CrossEntropyLoss mengharapkan input (N, C), jadi tensor di-reshape:
#       logits  : (B, 64, 2) → reshape → (B*64, 2)
#       targets : (B, 64)    → reshape → (B*64,)
#     Loss dihitung sebagai rata-rata di seluruh B*64 prediksi.
#   - Akurasi dihitung per prediksi frame, bukan per video.
#
# ATURAN MUTLAK: CUDA wajib digunakan jika tersedia (RTX 3060 Ti).
#   - Fungsi train_model_v2 memeriksa torch.cuda.is_available() di awal.
#   - Model dan data wajib di-push ke 'cuda' sebelum training dimulai.

import os
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _check_cuda() -> torch.device:
    """
    Memeriksa ketersediaan CUDA dan mengembalikan device yang sesuai.
    Mencetak peringatan keras jika CUDA tidak tersedia.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[CUDA] GPU terdeteksi: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[PERINGATAN] CUDA tidak tersedia! Training berjalan di CPU — akan sangat lambat.")
    return device


def train_one_epoch_v2(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Satu epoch fase training untuk model per-frame.

    Args:
        model     : Model dengan output (B, 64, num_classes).
        loader    : DataLoader yang menghasilkan (tensor_input, tensor_labels).
                    tensor_input  : (B, 64, 33, 3)
                    tensor_labels : (B, 64)
        criterion : nn.CrossEntropyLoss() (reduction='mean').
        optimizer : Optimizer PyTorch.
        device    : Device CUDA atau CPU.

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
            - epoch_loss     : Rata-rata CrossEntropyLoss per prediksi frame.
            - epoch_accuracy : Akurasi rata-rata per prediksi frame (0.0–1.0).
    """
    model.train()

    running_loss  = 0.0
    correct_preds = 0
    total_preds   = 0

    for batch_input, batch_labels in loader:
        # Push data ke GPU
        batch_input  = batch_input.to(device)   # (B, 64, 33, 3)
        batch_labels = batch_labels.to(device)  # (B, 64) dtype=LongTensor

        # --- Forward pass ---
        optimizer.zero_grad()
        logits = model(batch_input)              # (B, 64, num_classes)

        # --- Reshape untuk CrossEntropyLoss ---
        # CrossEntropyLoss mengharapkan (N, C) dan (N,)
        B, T, C = logits.shape
        logits_flat  = logits.reshape(B * T, C)          # (B*64, num_classes)
        labels_flat  = batch_labels.reshape(B * T)        # (B*64,)

        loss = criterion(logits_flat, labels_flat)

        # --- Backward pass & update bobot ---
        loss.backward()
        optimizer.step()

        # --- Statistik per batch ---
        running_loss  += loss.item() * (B * T)            # Akumulasi total loss
        preds          = logits_flat.argmax(dim=1)         # Prediksi kelas per frame
        correct_preds += (preds == labels_flat).sum().item()
        total_preds   += B * T

    epoch_loss     = running_loss / total_preds
    epoch_accuracy = correct_preds / total_preds

    return epoch_loss, epoch_accuracy


def evaluate_one_epoch_v2(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Satu epoch fase evaluasi (validasi atau test) untuk model per-frame.
    Tidak ada gradient yang dihitung.

    Args:
        model     : Model yang dievaluasi.
        loader    : DataLoader validasi/test.
        criterion : Fungsi loss yang sama dengan saat training.
        device    : Device target.

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.eval()

    running_loss  = 0.0
    correct_preds = 0
    total_preds   = 0

    with torch.no_grad():
        for batch_input, batch_labels in loader:
            batch_input  = batch_input.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_input)              # (B, 64, num_classes)

            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            labels_flat = batch_labels.reshape(B * T)

            loss = criterion(logits_flat, labels_flat)

            running_loss  += loss.item() * (B * T)
            preds          = logits_flat.argmax(dim=1)
            correct_preds += (preds == labels_flat).sum().item()
            total_preds   += B * T

    epoch_loss     = running_loss / total_preds
    epoch_accuracy = correct_preds / total_preds

    return epoch_loss, epoch_accuracy


def train_model_v2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    save_dir: str | Path = "bobot_model",
    save_filename: str = "best_model_v2.pth",
    verbose: bool = True,
) -> dict:
    """
    Loop pelatihan penuh untuk model per-frame v2.

    Perbedaan dari v1:
    - Memeriksa CUDA secara otomatis di awal (via _check_cuda()).
    - Model dan data dipush ke device CUDA jika tersedia.
    - Loss dihitung per frame (B*64 prediksi per batch), bukan per video.
    - Akurasi diukur per prediksi frame.

    Args:
        model         : AttentiveSkel3DPerFrame (atau subkelas nn.Module lainnya).
        train_loader  : DataLoader training dari create_dataloaders_v2().
        val_loader    : DataLoader validasi.
        criterion     : nn.CrossEntropyLoss().
        optimizer     : Optimizer (misalnya torch.optim.Adam).
        num_epochs    : Jumlah epoch pelatihan.
        save_dir      : Direktori penyimpanan model terbaik.
        save_filename : Nama file .pth untuk model terbaik.
        verbose       : Cetak log per epoch ke konsol.

    Returns:
        dict: Riwayat metric dengan kunci:
            'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'best_epoch', 'best_val_loss'
    """
    # Periksa CUDA dan tentukan device
    device = _check_cuda()

    # Push model ke device (wajib dilakukan sebelum forward pass)
    model = model.to(device)

    save_dir  = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    history = {
        "train_loss"    : [],
        "train_acc"     : [],
        "val_loss"      : [],
        "val_acc"       : [],
        "best_epoch"    : 0,
        "best_val_loss" : float("inf"),
    }

    best_model_weights = copy.deepcopy(model.state_dict())

    if verbose:
        print(f"\n{'='*70}")
        print(f"  AttentiveSkel-3D v2 — Per-Frame Training")
        print(f"  Device    : {device}")
        print(f"  Epochs    : {num_epochs}")
        print(f"  Save path : {save_path}")
        print(f"{'='*70}")

    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Training ──────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch_v2(
            model, train_loader, criterion, optimizer, device
        )

        # ── Validasi ──────────────────────────────────────────────────────────
        val_loss, val_acc = evaluate_one_epoch_v2(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - t0

        # Simpan ke history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Simpan model terbaik berdasarkan val_loss terendah
        improved = val_loss < history["best_val_loss"]
        if improved:
            history["best_val_loss"] = val_loss
            history["best_epoch"]    = epoch
            best_model_weights       = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch"              : epoch,
                    "model_state_dict"   : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss"           : val_loss,
                    "val_acc"            : val_acc,
                },
                save_path,
            )

        if verbose:
            marker = " ✓" if improved else "  "
            print(
                f"Epoch [{epoch:>3}/{num_epochs}]{marker} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:>6.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:>6.2f}% | "
                f"Waktu: {elapsed:.1f}s"
            )

    total_duration = time.time() - total_start

    if verbose:
        print(f"{'='*70}")
        print(f"  Pelatihan selesai dalam {total_duration:.1f} detik.")
        print(f"  Epoch terbaik : {history['best_epoch']}  (Val Loss = {history['best_val_loss']:.4f})")
        print(f"  Model terbaik disimpan ke: {save_path}")
        print(f"{'='*70}\n")

    # Muat kembali bobot model terbaik ke model aktif
    model.load_state_dict(best_model_weights)

    return history
