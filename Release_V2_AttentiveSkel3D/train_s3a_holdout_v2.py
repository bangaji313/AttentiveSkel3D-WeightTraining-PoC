# ============================================================
# train_s3a_holdout_v2.py
#
# Training Script Khusus S3a — BSP Only, Holdout Per-Frame
#
# Ketentuan:
#   - Dataset  : src/data/manifest_v2.csv
#   - Split    : train=340, val=73, test=74 (seed=42, via create_dataloaders_v2)
#   - Arsitektur: BSP=True, LearnedSpatial=False, TemporalAttention=False
#   - Loss     : CrossEntropyLoss per-frame (B*64 prediksi per batch)
#   - Optimizer: Adam(lr=1e-3), batch=16, max 100 epoch
#   - Seed     : Python/NumPy/PyTorch/CUDA = 42
#   - Checkpoint: disimpan ke bobot_model/best_model_s3a_bsp_holdout_v2.pth
#   - TIDAK menimpa best_model_ablasi_a.pth sebelum verifikasi
#   - Test set hanya dievaluasi SEKALI setelah training selesai
# ============================================================

import sys
import copy
import time
import random
import hashlib
import json
from pathlib import Path

# ── Pastikan project root masuk sys.path ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent  # Release_V2_AttentiveSkel3D (script ada di root)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix,
)

from src.models.arsitektur_v2 import AttentiveSkel3DPerFrame
from src.data.dataset_v2 import create_dataloaders_v2

# ============================================================
# Konfigurasi Eksperimen (tidak boleh diubah)
# ============================================================
SEED             = 42
MANIFEST_PATH    = PROJECT_ROOT / "src" / "data" / "manifest_v2.csv"
SAVE_DIR         = PROJECT_ROOT / "bobot_model"
SAVE_FILENAME    = "best_model_s3a_bsp_holdout_v2.pth"   # ← BUKAN ablasi_a.pth
META_FILENAME    = "best_model_s3a_bsp_holdout_v2_metadata.json"

BATCH_SIZE       = 16
LEARNING_RATE    = 1e-3
NUM_EPOCHS       = 100
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
# test = 1 - 0.70 - 0.15 = 0.15 -> 74 video (dari total 487)

# Konfigurasi S3a
USE_BSP          = True
USE_LEARNED      = False
USE_TEMPORAL     = False

EXPECTED_PARAMS  = 101_891   # Jumlah parameter yang diharapkan


# ============================================================
# Fungsi bantu
# ============================================================

def set_all_seeds(seed: int):
    """Tetapkan seed ke Python, NumPy, PyTorch, dan CUDA secara seragam."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[SEED] Semua seed diterapkan ke {seed}")


def build_model() -> AttentiveSkel3DPerFrame:
    model = AttentiveSkel3DPerFrame(
        num_classes=2,
        use_spatial_prior=USE_BSP,
        use_learned_spatial=USE_LEARNED,
        use_temporal_attention=USE_TEMPORAL,
    )
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_architecture(model: nn.Module):
    """Verifikasi arsitektur S3a sesuai spesifikasi."""
    n_params = count_params(model)
    sd = model.state_dict()
    bsp      = any(k.startswith("biomechanical_spatial_prior") for k in sd)
    learned  = any(k.startswith("learned_spatial_attention")   for k in sd)
    temporal = any(k.startswith("temporal_attention")           for k in sd)

    print("\n[VERIFIKASI ARSITEKTUR]")
    print(f"  Parameter  : {n_params:,}  (diharapkan {EXPECTED_PARAMS:,})")
    print(f"  BSP        : {bsp}   (diharapkan True)")
    print(f"  LearnedSpatial: {learned}  (diharapkan False)")
    print(f"  Temporal   : {temporal}  (diharapkan False)")

    assert bsp      == True,  f"BSP seharusnya True, dapat {bsp}"
    assert learned  == False, f"LearnedSpatial seharusnya False, dapat {learned}"
    assert temporal == False, f"Temporal seharusnya False, dapat {temporal}"
    assert n_params == EXPECTED_PARAMS, (
        f"Jumlah parameter {n_params:,} ≠ {EXPECTED_PARAMS:,}"
    )
    print("  [OK] Semua verifikasi arsitektur LULUS.\n")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss  = 0.0
    correct_preds = 0
    total_preds   = 0

    for batch_input, batch_labels in loader:
        batch_input  = batch_input.float().to(device)   # (B, 64, 33, 3)
        batch_labels = batch_labels.long().to(device)   # (B, 64)

        optimizer.zero_grad()
        logits = model(batch_input)                     # (B, 64, 2)

        B, T, C  = logits.shape
        loss     = criterion(logits.reshape(B * T, C), batch_labels.reshape(B * T))
        loss.backward()
        optimizer.step()

        running_loss  += loss.item() * (B * T)
        preds          = logits.reshape(B * T, C).argmax(dim=1)
        correct_preds += (preds == batch_labels.reshape(B * T)).sum().item()
        total_preds   += B * T

    return running_loss / total_preds, correct_preds / total_preds


def evaluate_loader(model, loader, criterion, device):
    """Evaluasi loss & akurasi pada satu DataLoader (val atau test)."""
    model.eval()
    running_loss  = 0.0
    correct_preds = 0
    total_preds   = 0
    all_preds  = []
    all_truths = []

    with torch.no_grad():
        for batch_input, batch_labels in loader:
            batch_input  = batch_input.float().to(device)
            batch_labels = batch_labels.long().to(device)

            logits = model(batch_input)
            B, T, C = logits.shape

            loss = criterion(logits.reshape(B * T, C), batch_labels.reshape(B * T))
            running_loss  += loss.item() * (B * T)

            preds = logits.reshape(B * T, C).argmax(dim=1)
            correct_preds += (preds == batch_labels.reshape(B * T)).sum().item()
            total_preds   += B * T

            all_preds.extend(preds.cpu().numpy().tolist())
            all_truths.extend(batch_labels.reshape(B * T).cpu().numpy().tolist())

    avg_loss = running_loss / total_preds
    avg_acc  = correct_preds / total_preds
    return avg_loss, avg_acc, np.array(all_preds), np.array(all_truths)


def compute_metrics(preds, truths) -> dict:
    cm = confusion_matrix(truths, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy (%)": round(accuracy_score(truths, preds) * 100, 4),
        "F1_Macro":     round(f1_score(truths, preds, average="macro",  zero_division=0), 6),
        "F1_Binary":    round(f1_score(truths, preds, average="binary", pos_label=1, zero_division=0), 6),
        "Precision":    round(precision_score(truths, preds, pos_label=1, average="binary", zero_division=0), 6),
        "Recall":       round(recall_score(truths, preds, pos_label=1, average="binary", zero_division=0), 6),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Total_Frame": len(truths),
    }


# ============================================================
# Main Training Loop
# ============================================================

def main():
    print("=" * 75)
    print("  TRAINING S3a — BSP Only (Holdout Per-Frame)")
    print("=" * 75)

    # ── 1. Tetapkan seed (SEBELUM DataLoader & model dibuat) ─────────────────
    set_all_seeds(SEED)

    # ── 2. Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}" + (
        f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    ))

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    print(f"\n[DATA] Manifest  : {MANIFEST_PATH}")
    train_loader, val_loader, test_loader = create_dataloaders_v2(
        csv_file=MANIFEST_PATH,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        num_workers=0,
        random_seed=SEED,
    )
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(f"[DATA] Train={n_train} | Val={n_val} | Test={n_test}")

    # Verifikasi jumlah split sesuai spesifikasi
    assert n_train == 340, f"Train seharusnya 340, dapat {n_train}"
    assert n_val   == 73,  f"Val seharusnya 73, dapat {n_val}"
    assert n_test  == 74,  f"Test seharusnya 74, dapat {n_test}"
    print("[DATA] Split terverifikasi: train=340, val=73, test=74 OK")

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = build_model().to(device)
    verify_architecture(model)
    n_params = count_params(model)

    # ── 5. Optimizer & Loss ───────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / SAVE_FILENAME

    best_val_loss    = float("inf")
    best_epoch       = 0
    best_val_acc     = 0.0
    best_state_dict  = None
    history          = []

    print(f"\n[TRAINING] Mulai training {NUM_EPOCHS} epoch...")
    print(f"  Optimizer : Adam(lr={LEARNING_RATE})")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Save path : {save_path}")
    print("-" * 75)

    t_total = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_loader(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        improved = val_loss < best_val_loss

        if improved:
            best_val_loss   = val_loss
            best_val_acc    = val_acc
            best_epoch      = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

            torch.save({
                "epoch"              : epoch,
                "model_state_dict"   : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss"           : val_loss,
                "val_acc"            : val_acc,
                "config": {
                    "use_spatial_prior"     : USE_BSP,
                    "use_learned_spatial"   : USE_LEARNED,
                    "use_temporal_attention": USE_TEMPORAL,
                },
            }, save_path)

        marker = " OK" if improved else "  "
        print(
            f"Epoch [{epoch:>3}/{NUM_EPOCHS}]{marker} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:>6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:>6.2f}% | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
        })

    total_time = time.time() - t_total
    print("-" * 75)
    print(f"[TRAINING SELESAI] {total_time:.1f}s total")
    print(f"  Epoch terbaik : {best_epoch}  (Val Loss={best_val_loss:.6f}, Val Acc={best_val_acc*100:.4f}%)")
    print(f"  Checkpoint    : {save_path}")

    # ── 7. Evaluasi Test Set (SATU KALI) ──────────────────────────────────────
    print("\n" + "=" * 75)
    print("[TEST SET EVALUATION] Evaluasi satu kali pada 74 video test")
    print("  (Test set TIDAK pernah digunakan untuk pemilihan epoch/tuning)")
    print("=" * 75)

    # Muat bobot terbaik
    model.load_state_dict(best_state_dict)
    _, _, test_preds, test_truths = evaluate_loader(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_preds, test_truths)

    print(f"  Accuracy  : {test_metrics['Accuracy (%)']:.4f}%")
    print(f"  F1 Macro  : {test_metrics['F1_Macro']:.6f}")
    print(f"  F1 Binary : {test_metrics['F1_Binary']:.6f}")
    print(f"  Precision : {test_metrics['Precision']:.6f}")
    print(f"  Recall    : {test_metrics['Recall']:.6f}")
    print(f"  TP={test_metrics['TP']:,}  TN={test_metrics['TN']:,}  "
          f"FP={test_metrics['FP']:,}  FN={test_metrics['FN']:,}")

    # ── 8. Verifikasi Akhir Checkpoint ────────────────────────────────────────
    print("\n[VERIFIKASI CHECKPOINT]")
    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model_state_dict"]
    bsp      = any(k.startswith("biomechanical_spatial_prior") for k in sd)
    learned  = any(k.startswith("learned_spatial_attention")   for k in sd)
    temporal = any(k.startswith("temporal_attention")           for k in sd)
    sha256   = hashlib.sha256(save_path.read_bytes()).hexdigest()

    tmp_model = build_model()
    tmp_model.load_state_dict(sd)
    verified_params = count_params(tmp_model)

    print(f"  SHA256     : {sha256}")
    print(f"  PARAMETER  : {verified_params:,}  (diharapkan {EXPECTED_PARAMS:,})")
    print(f"  BSP        : {bsp}  (diharapkan True)")
    print(f"  LearnedSp  : {learned}  (diharapkan False)")
    print(f"  Temporal   : {temporal}  (diharapkan False)")

    assert verified_params == EXPECTED_PARAMS, \
        f"Parameter mismatch: {verified_params:,} ≠ {EXPECTED_PARAMS:,}"
    assert bsp      == True
    assert learned  == False
    assert temporal == False
    print("  [OK] Verifikasi checkpoint LULUS semua kriteria.")

    # ── 9. Simpan Metadata Eksperimen ─────────────────────────────────────────
    metadata = {
        "experiment"      : "S3a — BSP Only Holdout Per-Frame V2",
        "checkpoint_file" : SAVE_FILENAME,
        "sha256"          : sha256,
        "manifest"        : str(MANIFEST_PATH),
        "seed"            : SEED,
        "split": {
            "train": n_train,
            "val"  : n_val,
            "test" : n_test,
        },
        "hyperparameters": {
            "batch_size"    : BATCH_SIZE,
            "learning_rate" : LEARNING_RATE,
            "num_epochs"    : NUM_EPOCHS,
            "optimizer"     : "Adam",
            "loss"          : "CrossEntropyLoss",
        },
        "architecture": {
            "use_spatial_prior"     : USE_BSP,
            "use_learned_spatial"   : USE_LEARNED,
            "use_temporal_attention": USE_TEMPORAL,
            "parameter_count"       : verified_params,
        },
        "best_checkpoint": {
            "epoch"   : best_epoch,
            "val_loss": round(best_val_loss, 6),
            "val_acc" : round(best_val_acc,  6),
        },
        "test_set_evaluation": test_metrics,
        "training_history"   : history,
    }

    meta_path = SAVE_DIR / META_FILENAME
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n[METADATA] Tersimpan ke: {meta_path}")

    print("\n" + "=" * 75)
    print(f"  TRAINING S3a SELESAI")
    print(f"  Checkpoint baru : {save_path.name}")
    print(f"  best_model_ablasi_a.pth TIDAK diubah (menunggu verifikasi)")
    print("=" * 75)


if __name__ == "__main__":
    main()
