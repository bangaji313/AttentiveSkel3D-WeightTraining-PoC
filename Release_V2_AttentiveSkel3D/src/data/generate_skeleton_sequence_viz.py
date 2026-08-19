# Release_V2_AttentiveSkel3D/src/data/generate_skeleton_sequence_viz.py
#
# Menghasilkan visualisasi strip skeleton sequence (seperti Fig. 13 paper)
# untuk semua 487 video dalam dataset.
#
# Setiap video menghasilkan 1 PNG berisi 8 frame kunci yang dipilih secara
# cerdas (mencakup titik transisi BENAR↔SALAH), dengan:
#   - Skeleton HIJAU   = frame BENAR (label 0)
#   - Skeleton MERAH   = frame SALAH (label 1)
#   - Bar warna bawah  = timeline 64 frame (hijau/merah per frame)
#   - Judul + metadata video di atas strip
#
# Output: Release_V2_AttentiveSkel3D/hasil_evaluasi/skeleton_viz/{stem}.png
#
# Cara menjalankan dari root proyek:
#   conda run -n attentiveskel python Release_V2_AttentiveSkel3D/src/data/generate_skeleton_sequence_viz.py

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd
from tqdm import tqdm

# ─── Direktori ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "src" / "data" / "manifest_v2.csv"
TENSORS_DIR   = PROJECT_ROOT / "data" / "tensors"
LABELS_DIR    = PROJECT_ROOT / "data" / "v2_labels"
OUTPUT_DIR    = PROJECT_ROOT / "hasil_evaluasi" / "skeleton_viz"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Koneksi Skeleton MediaPipe BlazePose ──────────────────────────────────
# Pasangan indeks yang dihubungkan menjadi "tulang"
SKELETON_CONNECTIONS = [
    # Kepala – badan
    (0, 11), (0, 12),         # Nose → kedua bahu
    (11, 12),                  # Bahu kiri ↔ Bahu kanan
    # Torso
    (11, 23), (12, 24),        # Bahu → Pinggul
    (23, 24),                  # Pinggul kiri ↔ Pinggul kanan
    # Lengan kiri
    (11, 13), (13, 15),
    # Lengan kanan
    (12, 14), (14, 16),
    # Kaki kiri
    (23, 25), (25, 27),
    # Kaki kanan
    (24, 26), (26, 28),
    # Telapak kaki
    (27, 31), (28, 32),
]

# Landmark yang perlu diplot (subset penting)
IMPORTANT_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

# Warna per kategori label
COLOR_BENAR = "#2ecc71"   # Hijau
COLOR_SALAH = "#e74c3c"   # Merah
COLOR_JOINT_BENAR = "#27ae60"
COLOR_JOINT_SALAH = "#c0392b"


def pilih_frame_kunci(labels: np.ndarray, n_frames: int = 8) -> list[int]:
    """
    Memilih n_frames frame yang paling representatif untuk divisualisasi.
    Strategi:
    1. Sertakan frame pertama, terakhir, dan tengah
    2. Prioritaskan frame tepat sebelum/setelah transisi label
    3. Tambahkan frame terdistribusi rata untuk melengkapi
    """
    T = len(labels)
    kandidat = set()

    # Selalu sertakan frame pertama, tengah, dan terakhir
    kandidat.update([0, T // 4, T // 2, (3 * T) // 4, T - 1])

    # Tambahkan frame di titik transisi (BENAR→SALAH atau SALAH→BENAR)
    for i in range(1, T):
        if labels[i] != labels[i - 1]:
            kandidat.add(max(0, i - 1))
            kandidat.add(i)
            kandidat.add(min(T - 1, i + 1))

    # Konversi ke list terurut
    kandidat = sorted(kandidat)

    # Jika sudah cukup, pilih n_frames yang paling tersebar
    if len(kandidat) >= n_frames:
        # Ambil yang paling tersebar (evenly spaced dari kandidat)
        indices = np.round(np.linspace(0, len(kandidat) - 1, n_frames)).astype(int)
        return [kandidat[i] for i in indices]
    else:
        # Tambahkan frame terdistribusi rata sampai n_frames
        extra = sorted(set(np.round(np.linspace(0, T - 1, n_frames - len(kandidat))).astype(int).tolist()))
        semua = sorted(set(kandidat) | set(extra))
        if len(semua) > n_frames:
            indices = np.round(np.linspace(0, len(semua) - 1, n_frames)).astype(int)
            return [semua[i] for i in indices]
        return semua


def gambar_skeleton(ax, frame: np.ndarray, label: int, frame_idx: int,
                    exercise: str, show_xlabel: bool = True):
    """
    Menggambar satu frame skeleton pada axes matplotlib.
    frame: array (33, 3) koordinat [X, Y, Z] ternormalisasi
    label: 0=BENAR, 1=SALAH
    """
    warna = COLOR_BENAR if label == 0 else COLOR_SALAH
    warna_joint = COLOR_JOINT_BENAR if label == 0 else COLOR_JOINT_SALAH
    status = "BENAR" if label == 0 else "SALAH"

    # Background panel
    ax.set_facecolor("#1a1a2e" if label == 1 else "#0d2818")

    # Gambar tulang (koneksi)
    for (i, j) in SKELETON_CONNECTIONS:
        if i < 33 and j < 33:
            x_vals = [frame[i, 0], frame[j, 0]]
            y_vals = [-frame[i, 1], -frame[j, 1]]  # Flip Y: MediaPipe Y+ = bawah
            ax.plot(x_vals, y_vals,
                    color=warna, linewidth=1.8, alpha=0.9, solid_capstyle='round')

    # Gambar titik sendi
    for ji in IMPORTANT_JOINTS:
        if ji < 33:
            ax.scatter(frame[ji, 0], -frame[ji, 1],
                       color=warna_joint, s=25, zorder=5, linewidths=0.5,
                       edgecolors='white')

    # Label frame dan status
    ax.set_title(f"F{frame_idx + 1:02d}\n{status}",
                 fontsize=7, color=warna, fontweight='bold', pad=2)

    # Bingkai berwarna sesuai label
    for spine in ax.spines.values():
        spine.set_edgecolor(warna)
        spine.set_linewidth(2.5)

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def buat_timeline_bar(ax, labels: np.ndarray, frame_kunci: list[int]):
    """
    Menampilkan timeline 64 frame di bawah strip sebagai bar warna.
    Hijau = BENAR, Merah = SALAH. Tanda segitiga menandai frame kunci.
    """
    T = len(labels)
    colors = [COLOR_BENAR if lbl == 0 else COLOR_SALAH for lbl in labels]

    for i, c in enumerate(colors):
        ax.barh(0, 1, left=i, height=1, color=c, linewidth=0)

    # Tandai frame kunci dengan tanda panah
    for fi in frame_kunci:
        ax.annotate("▼", xy=(fi + 0.5, 1.0), fontsize=8, ha='center',
                    va='bottom', color='white', fontweight='bold')

    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, 2.0)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 63])
    ax.set_xticklabels(["F1", "F9", "F17", "F25", "F33", "F41", "F49", "F57", "F64"],
                       fontsize=6, color='white')
    ax.set_yticks([])
    ax.set_facecolor("#0f0f1a")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legenda mini
    n_benar = int(np.sum(labels == 0))
    n_salah = int(np.sum(labels == 1))
    ax.text(T + 1, 0.4, f"✓ {n_benar}", color=COLOR_BENAR, fontsize=7, va='center', fontweight='bold')
    ax.text(T + 1, -0.1, f"✗ {n_salah}", color=COLOR_SALAH, fontsize=7, va='center', fontweight='bold')


def buat_strip_satu_video(stem: str, exercise: str, tensor: np.ndarray,
                           labels: np.ndarray, video_name: str,
                           n_key_frames: int = 8) -> Path:
    """
    Membuat satu gambar PNG skeleton strip untuk satu video.
    Mengembalikan path file yang disimpan.
    """
    frame_kunci = pilih_frame_kunci(labels, n_key_frames)
    n_kf = len(frame_kunci)

    # Layout: n_kf kolom skeleton + 1 baris timeline
    fig = plt.figure(figsize=(n_kf * 2.0, 4.5), facecolor="#0f0f1a")
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.01, right=0.93,
                        hspace=0.1, wspace=0.08)

    gs = GridSpec(2, n_kf, figure=fig,
                  height_ratios=[4, 0.5],
                  hspace=0.3, wspace=0.08,
                  left=0.01, right=0.93,
                  top=0.88, bottom=0.18)

    # — Baris skeleton —
    for col, fi in enumerate(frame_kunci):
        ax = fig.add_subplot(gs[0, col])
        gambar_skeleton(ax, tensor[fi], int(labels[fi]), fi, exercise)

    # — Baris timeline —
    ax_tl = fig.add_subplot(gs[1, :])
    buat_timeline_bar(ax_tl, labels, frame_kunci)

    # — Judul —
    n_benar = int(np.sum(labels == 0))
    n_salah = int(np.sum(labels == 1))
    pct_benar = n_benar / len(labels) * 100

    judul = (f"{stem}  |  {exercise.upper()}  |  {video_name}\n"
             f"Frame BENAR: {n_benar}/64 ({pct_benar:.0f}%)   "
             f"Frame SALAH: {n_salah}/64 ({100-pct_benar:.0f}%)")
    fig.text(0.5, 0.97, judul, ha='center', va='top',
             color='white', fontsize=8, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#2c2c4e', alpha=0.8))

    # Legenda
    patch_benar = mpatches.Patch(color=COLOR_BENAR, label="BENAR (Label 0)")
    patch_salah = mpatches.Patch(color=COLOR_SALAH, label="SALAH (Label 1)")
    fig.legend(handles=[patch_benar, patch_salah],
               loc='lower right', fontsize=7,
               facecolor='#1a1a2e', edgecolor='gray',
               labelcolor='white', framealpha=0.9,
               bbox_to_anchor=(0.99, 0.01))

    # Simpan
    out_path = OUTPUT_DIR / f"{stem}_skel_viz.png"
    fig.savefig(str(out_path), dpi=100, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    print("=" * 70)
    print("  PEMBANGKIT VISUALISASI SKELETON SEQUENCE — AttentiveSkel-3D V2")
    print("=" * 70)
    print(f"Output Dir : {OUTPUT_DIR}")

    df = pd.read_csv(MANIFEST_PATH)
    print(f"[INFO] Total video: {len(df)}")
    print()

    ok = 0
    skip = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rendering skeleton strips"):
        stem     = Path(row["file_path"]).stem
        exercise = str(row["exercise"]).lower()

        # Load tensor
        p_tensor = TENSORS_DIR / f"{stem}.npy"
        p_label  = LABELS_DIR / f"{stem}_labels.npy"

        if not p_tensor.exists() or not p_label.exists():
            tqdm.write(f"[SKIP] {stem} — file tidak ditemukan")
            skip += 1
            continue

        tensor = np.load(str(p_tensor)).astype(np.float32)
        labels = np.load(str(p_label)).astype(int)

        if tensor.ndim != 3 or tensor.shape != (64, 33, 3):
            tqdm.write(f"[SKIP] {stem} — shape tidak valid: {tensor.shape}")
            skip += 1
            continue

        # Cari nama video mentah dari manifest jika ada
        video_name = f"{stem}.mp4"

        buat_strip_satu_video(
            stem=stem,
            exercise=exercise,
            tensor=tensor,
            labels=labels,
            video_name=video_name,
            n_key_frames=8
        )
        ok += 1

    print()
    print("=" * 70)
    print(f"  SELESAI: {ok} PNG dibuat | {skip} dilewati")
    print(f"  Lokasi output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
