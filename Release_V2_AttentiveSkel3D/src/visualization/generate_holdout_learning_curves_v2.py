"""
src/visualization/generate_holdout_learning_curves_v2.py

Generator Kurva Pembelajaran Canonical S3 (S3a, S3b, S3c) dari Eksperimen Holdout.
Script ini mengekstraksi riwayat pelatihan 100 epoch untuk 3 skenario ablasi S3:
  - S3a (BSP Only)                  : S3a_BSP_Training_History_V2.csv
  - S3b (BSP + Learned Spatial)     : 03_train_scenario3_ablation.ipynb (Log Ablasi C)
  - S3c (BSP + Temporal Attention)  : 03_train_scenario3_ablation.ipynb (Log Ablasi B)

Mengekspor:
  1. hasil_evaluasi/Holdout_Training_History_S3_Canonical_V2.csv (300 baris data)
  2. bobot_model/curve_s3_ablation_comparison.png (Plot 2-panel Loss & Accuracy)
"""

import os
import sys
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Base directories setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # Release_V2_AttentiveSkel3D

CSV_S3A_PATH = PROJECT_ROOT / "hasil_evaluasi" / "S3a_BSP_Training_History_V2.csv"
NB_ABLA_PATH = PROJECT_ROOT / "notebooks" / "03_train_scenario3_ablation.ipynb"

OUT_CSV_PATH = PROJECT_ROOT / "hasil_evaluasi" / "Holdout_Training_History_S3_Canonical_V2.csv"
OUT_IMG_PATH = PROJECT_ROOT / "bobot_model" / "curve_s3_ablation_comparison.png"


def load_s3a_history() -> pd.DataFrame:
    """Memuat 100 epoch S3a dari CSV resmi S3a_BSP_Training_History_V2.csv."""
    if not CSV_S3A_PATH.exists():
        raise FileNotFoundError(f"Sumber data S3a tidak ditemukan: {CSV_S3A_PATH}")

    df = pd.read_csv(CSV_S3A_PATH)
    required_cols = {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Kolom CSV S3a tidak sesuai: {df.columns}")

    df["scenario"] = "S3a"
    df["modules"] = "BSP"
    df["source_artifact"] = "S3a_BSP_Training_History_V2.csv"
    return df


def parse_notebook_ablation_logs() -> dict:
    """Mengekstraksi log pelatihan Ablasi B (S3c) dan Ablasi C (S3b) dari notebook 03."""
    if not NB_ABLA_PATH.exists():
        raise FileNotFoundError(f"Notebook ablasi tidak ditemukan: {NB_ABLA_PATH}")

    with open(NB_ABLA_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Output training berada pada cell index 4
    cell_output_text = "".join(nb["cells"][4]["outputs"][0]["text"])

    sections = re.split(r"# Memulai: (Ablasi [A-C] \(.*?\))", cell_output_text)
    logs = {}

    for i in range(1, len(sections), 2):
        exp_name = sections[i]
        exp_body = sections[i + 1]

        rows = []
        for line in exp_body.splitlines():
            if "Train Loss:" in line:
                line_clean = line.encode("ascii", "ignore").decode()
                m = re.search(
                    r"Epoch\s*\[\s*(\d+)/100\]\s*\|\s*Train Loss:\s*([\d\.]+)\s*\|\s*Train Acc:\s*([\d\.]+)%\s*\|\s*Val Loss:\s*([\d\.]+)\s*\|\s*Val Acc:\s*([\d\.]+)%",
                    line_clean,
                )
                if m:
                    rows.append(
                        {
                            "epoch": int(m.group(1)),
                            "train_loss": float(m.group(2)),
                            "train_acc": float(m.group(3)) / 100.0,
                            "val_loss": float(m.group(4)),
                            "val_acc": float(m.group(5)) / 100.0,
                        }
                    )

        if len(rows) != 100:
            raise ValueError(f"Ekstraksi log {exp_name} gagal: ditemukan {len(rows)} epoch, diharapkan 100.")

        logs[exp_name] = pd.DataFrame(rows)

    # Mapping arsitektur aktual:
    # Ablasi C (Tanpa Temporal) = BSP + Learned Spatial -> S3b
    # Ablasi B (Tanpa Learned Spatial) = BSP + Temporal -> S3c
    df_s3b = logs["Ablasi C (Tanpa Temporal Attention)"].copy()
    df_s3b["scenario"] = "S3b"
    df_s3b["modules"] = "BSP + Learned Spatial"
    df_s3b["source_artifact"] = "03_train_scenario3_ablation.ipynb:Ablasi_C"

    df_s3c = logs["Ablasi B (Tanpa Learned Spatial)"].copy()
    df_s3c["scenario"] = "S3c"
    df_s3c["modules"] = "BSP + Temporal"
    df_s3c["source_artifact"] = "03_train_scenario3_ablation.ipynb:Ablasi_B"

    return {"S3b": df_s3b, "S3c": df_s3c}


def validate_canonical_history(df_all: pd.DataFrame):
    """Validasi integritas data 300 baris dan checkpoint terbaik per skenario."""
    if len(df_all) != 300:
        raise ValueError(f"Total baris data {len(df_all)} != 300.")

    scenarios = ["S3a", "S3b", "S3c"]
    expected_bests = {
        "S3a": {"epoch": 52, "val_loss": 0.204907, "val_acc_min": 0.9140},
        "S3b": {"epoch": 63, "val_loss": 0.208200, "val_acc_min": 0.9170},
        "S3c": {"epoch": 67, "val_loss": 0.217900, "val_acc_min": 0.9180},
    }

    for sc in scenarios:
        sub = df_all[df_all["scenario"] == sc]
        if len(sub) != 100:
            raise ValueError(f"Skenario {sc} memiliki {len(sub)} baris, diharapkan 100.")

        epochs = sub["epoch"].tolist()
        if epochs != list(range(1, 101)):
            raise ValueError(f"Sequence epoch skenario {sc} tidak berurutan 1..100.")

        # Best val loss validation
        best_row = sub.loc[sub["val_loss"].idxmin()]
        exp = expected_bests[sc]
        if int(best_row["epoch"]) != exp["epoch"]:
            raise ValueError(f"Epoch terbaik {sc}: {int(best_row['epoch'])} != {exp['epoch']}")

        if abs(best_row["val_loss"] - exp["val_loss"]) > 1e-4:
            raise ValueError(f"Val loss terbaik {sc}: {best_row['val_loss']} != {exp['val_loss']}")

    print("[VALIDASI] Seluruh 300 baris data provenance dan checkpoint terbaik TERVERIFIKASI SAMA!")


def generate_provenance_csv(df_all: pd.DataFrame):
    """Menyimpan CSV provenance canonical 300 baris."""
    cols_order = [
        "scenario",
        "modules",
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "source_artifact",
    ]
    df_out = df_all[cols_order].copy()
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV_PATH, index=False)
    print(f"[EXPOR CSV] Canonical history disimpan ke: {OUT_CSV_PATH} ({len(df_out)} baris)")


def plot_canonical_learning_curves(df_all: pd.DataFrame):
    """Membuat gambar kurva pembelajaran 2-panel (Loss & Accuracy) untuk S3a, S3b, S3c."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True, dpi=300)

    configs = {
        "S3a": {
            "label": "S3a — BSP Only (Epoch 52, Loss 0.2049, Acc 91.46%)",
            "color": "#1f77b4",  # Blue
            "marker_epoch": 52,
        },
        "S3b": {
            "label": "S3b — BSP + Learned Spatial (Epoch 63, Loss 0.2082, Acc 91.72%)",
            "color": "#2ca02c",  # Green
            "marker_epoch": 63,
        },
        "S3c": {
            "label": "S3c — BSP + Temporal Attention (Epoch 67, Loss 0.2179, Acc 91.89%)",
            "color": "#ff7f0e",  # Orange
            "marker_epoch": 67,
        },
    }

    # ── Panel 1: Validation Loss ────────────────────────────────────────────────
    for sc, cfg in configs.items():
        sub = df_all[df_all["scenario"] == sc]
        epochs = sub["epoch"].values
        val_loss = sub["val_loss"].values

        ax1.plot(epochs, val_loss, label=cfg["label"], color=cfg["color"], linewidth=2.0, alpha=0.9)

        # Highlight best epoch point
        best_row = sub[sub["epoch"] == cfg["marker_epoch"]].iloc[0]
        ax1.scatter(
            [best_row["epoch"]],
            [best_row["val_loss"]],
            color=cfg["color"],
            s=80,
            zorder=5,
            edgecolors="black",
            linewidth=1.2,
        )

    ax1.set_title(
        "Kurva Pembelajaran Evaluasi Holdout Model S3 (S3a, S3b, S3c)\n"
        "[Fixed-Split Holdout Dataset — 340 Train / 73 Val / 74 Test]",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel("Validation Loss (BCE)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    ax1.set_ylim(0.15, 0.65)

    # ── Panel 2: Validation Accuracy ───────────────────────────────────────────
    for sc, cfg in configs.items():
        sub = df_all[df_all["scenario"] == sc]
        epochs = sub["epoch"].values
        val_acc = sub["val_acc"].values * 100.0  # Percentage

        ax2.plot(epochs, val_acc, label=cfg["label"], color=cfg["color"], linewidth=2.0, alpha=0.9)

        # Highlight best epoch point
        best_row = sub[sub["epoch"] == cfg["marker_epoch"]].iloc[0]
        ax2.scatter(
            [best_row["epoch"]],
            [best_row["val_acc"] * 100.0],
            color=cfg["color"],
            s=80,
            zorder=5,
            edgecolors="black",
            linewidth=1.2,
        )

    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    ax2.set_ylim(50.0, 95.0)

    # Marker annotations on plot
    ax1.annotate(
        "S3a Best (Ep 52: 0.2049)",
        xy=(52, 0.2049),
        xytext=(35, 0.165),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
        fontsize=8.5,
        fontweight="bold",
        color="#1f77b4",
    )
    ax1.annotate(
        "S3b Best (Ep 63: 0.2082)",
        xy=(63, 0.2082),
        xytext=(65, 0.165),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        fontsize=8.5,
        fontweight="bold",
        color="#2ca02c",
    )
    ax1.annotate(
        "S3c Best (Ep 67: 0.2179)",
        xy=(67, 0.2179),
        xytext=(78, 0.185),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.2),
        fontsize=8.5,
        fontweight="bold",
        color="#ff7f0e",
    )

    plt.tight_layout()
    OUT_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_IMG_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[EXPORT GAMBAR] Kurva pembelajaran disintesis ke: {OUT_IMG_PATH}")


def main():
    print("=== PIPELINE GENERATOR KURVA PEMBELAJARAN CANONICAL S3 ===")
    df_s3a = load_s3a_history()
    logs_nb = parse_notebook_ablation_logs()

    df_s3b = logs_nb["S3b"]
    df_s3c = logs_nb["S3c"]

    df_all = pd.concat([df_s3a, df_s3b, df_s3c], ignore_index=True)

    validate_canonical_history(df_all)
    generate_provenance_csv(df_all)
    plot_canonical_learning_curves(df_all)
    print("=== PROSES GENERASI SELESAI DENGAN SUKSES ===")


if __name__ == "__main__":
    main()
