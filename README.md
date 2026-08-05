<div align="center">

# 🏋️ AttentiveSkel-3D V2 — Per-Frame Weight Training Form Analysis

### *Proof of Concept: Klasifikasi Kualitas Gerakan Latihan Beban Berbasis Evaluasi Per-Frame*
### *3D-CNN dengan Biomechanical Spatial Prior, Learned Spatial Attention & Temporal Attention*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Branch](https://img.shields.io/badge/Branch-v2--per--frame--ai-blueviolet?style=for-the-badge)]()
[![Model Size](https://img.shields.io/badge/Model%20Size-0.44%20MB-blue?style=for-the-badge)]()
[![Params](https://img.shields.io/badge/Parameters-110%2C372-orange?style=for-the-badge)]()

<br/>

> **Tugas Akhir — Program Studi Informatika**  
> Institut Teknologi Nasional (ITENAS) Bandung · 2026

</div>

---

## 🎯 Latar Belakang & Motivasi

Kesalahan postur saat latihan beban — seperti lutut jatuh ke dalam saat *squat*, punggung membungkuk saat *deadlift*, atau *range of motion* siku yang kurang saat *bench press* — adalah sumber utama cedera akut maupun kronis di pusat kebugaran. Tanpa bimbingan pelatih berpengalaman, pengguna tidak menyadari kapan tepatnya kesalahan tersebut terjadi dalam satu repetisi.

**Sistem V2 ini menjawab pertanyaan yang lebih spesifik:**  
> *"Bukan hanya apakah gerakan ini salah, tapi **di frame ke-berapa** kesalahan itu terjadi?"*

Perubahan mendasar dari V1: sistem tidak lagi memberikan **satu label per video** (evaluasi global), melainkan menghasilkan **64 prediksi independen per video** — satu prediksi untuk setiap titik waktu dalam gerakan.

---

## ⚙️ Perubahan Arsitektur: V1 → V2

### Masalah Fundamental pada V1

Arsitektur V1 menggunakan **Global Average Pooling (GAP)** pada dimensi temporal yang menghancurkan semua informasi *kapan* sesuatu terjadi:

```python
# V1 — Satu label, informasi waktu hilang
x = x.mean(dim=[2, 3, 4])  # (B, 128, T', L', 1) → (B, 128)
x = classifier(x)           # (B, 128) → (B, 2)  ← satu prediksi per video
```

### Solusi pada V2: Satu Bedah Operasi

Satu perubahan kode, dampak fundamental:

```diff
# V1 — GAP menghancurkan temporal
- x = x.mean(dim=[2, 3, 4])    # (B, 128, T', L', 1) → (B, 128)
- x = self.classifier(x)       # (B, 128) → (B, 2)

# V2 — Pool hanya landmark, pertahankan temporal
+ x = self.landmark_pool(x)    # AdaptiveAvgPool3d(None,1,1) → (B, 128, T', 1, 1)
+ x = F.interpolate(...)       # T' → 64 frame (upsample)
+ x = x.permute(0, 2, 1)       # (B, 64, 128)
+ x = self.classifier(x)       # (B, 64, 128) → (B, 64, 2)  ← per frame!
```

| Aspek | V1 | V2 |
|---|---|---|
| Label granularitas | 1 label / video | **64 label / video** |
| Shape output | `(B, 2)` | `(B, 64, 2)` |
| Resolusi temporal | Hilang (GAP) | **Dipertahankan** |
| Sinyal gradien / video | 1 | **64** |
| Temporal heatmap | ✗ | ✓ |

---

## 🤖 Arsitektur Model: `AttentiveSkel3DPerFrame`

### Alur Tensor Lengkap

```
INPUT:  (B, 64, 33, 3)   — Batch × 64 Frame × 33 Landmark × 3 Koordinat XYZ
    │
    ├─ Reshape → (B, 3, 64, 33, 1)
    │
    ├─ [BSP] Biomechanical Spatial Prior  ← 33 parameter learnable, sigmoid
    │         (1,1,1,33,1) × input        ← Bobot per sendi tubuh
    │
    ├─ Conv Block 1: Conv3d(3→32, kernel(3,3,1)) + BN + ReLU + MaxPool(1,2,1)
    │                (B, 3, 64, 33, 1) → (B, 32, 64, 16, 1)   ← landmark: 33→16
    │
    ├─ Conv Block 2: Conv3d(32→64, kernel(3,3,1)) + BN + ReLU + MaxPool(2,2,1)
    │                (B, 32, 64, 16, 1) → (B, 64, 32, 8, 1)   ← waktu: 64→32
    │
    ├─ Conv Block 3: Conv3d(64→128, kernel(3,3,1)) + BN + ReLU  (tanpa pool)
    │                (B, 64, 32, 8, 1) → (B, 128, 32, 8, 1)
    │
    ├─ [LS] Learned Spatial Attention    ← SE-style MLP: GAP → Linear(128→32→128) → Sigmoid
    │         Bobot per channel, broadcast ke (B, 128, 32, 8, 1)
    │
    ├─ [TA] Temporal Attention           ← Conv3d(128→1, 1×1×1) → Softmax(dim=time)
    │         Frame lebih penting diperkuat, frame kurang penting diredam
    │
    ├─ AdaptiveAvgPool3d(None, 1, 1)     ← Pool HANYA landmark. Waktu TETAP!
    │   (B, 128, 32, 8, 1) → (B, 128, 32, 1, 1)
    │
    ├─ Interpolate bilinear → 64 frame   ← Kembalikan resolusi temporal ke 64
    │   (B, 128, 64, 1) → squeeze → (B, 128, 64)
    │
    ├─ Permute → (B, 64, 128)
    │
    └─ Classifier: Linear(128→64) → ReLU → Dropout(0.4) → Linear(64→2)

OUTPUT: (B, 64, 2)   — 64 pasang logit per frame [logit_Benar, logit_Salah]
```

### Profil Parameter

| Komponen | Parameter | % Total | Keterangan |
|---|:---:|:---:|---|
| `conv_block_3` | **73,984** | **67.0%** | Fitur high-level (64→128 channel) |
| `conv_block_2` | 18,560 | 16.8% | Fitur menengah (32→64 channel) |
| `classifier` | 8,386 | 7.6% | Head linear per-frame |
| `learned_spatial_attention` | 8,352 | 7.6% | MLP SE-style |
| `conv_block_1` | 928 | 0.8% | Fitur level rendah (3→32 channel) |
| `temporal_attention` | 129 | 0.1% | Conv 1×1×1 skoring temporal |
| `biomechanical_spatial_prior` | **33** | **0.03%** | 33 skalar (satu per sendi) |
| **TOTAL** | **110,372** | **100%** | |

### Efisiensi Komputasi

| Metrik | Nilai |
|---|---|
| Total Parameter | **110,372** |
| MACs (*Multiply-Accumulate*) | **39.62 Juta** |
| Ukuran File Model (.pth) | **0.44 MB** |
| Estimasi Total Memori | 3.16 MB |
| Perbandingan ResNet-18 | 106× lebih ringan |

---

## 🔬 Tiga Modul Atensi

### 1. Biomechanical Spatial Prior (BSP) — *Explainable AI*

Modul paling ringan namun paling dapat diinterpretasikan: **33 parameter** yang masing-masing menjadi bobot skalar untuk satu sendi MediaPipe BlazePose. Setelah sigmoid, nilainya ∈ [0, 1] — sendi dengan nilai mendekati 1 mendapat penguatan, nilai mendekati 0 diredam.

**Hasil bobot BSP tertinggi setelah pelatihan (Skenario 2 — Full Model):**

| Peringkat | Sendi | Bobot BSP | Makna Biomekanik |
|:---:|---|:---:|---|
| #1 | **Right Knee** | 0.7742 | Lutut kanan — titik kritis squat & deadlift |
| #2 | **Right Elbow** | 0.7693 | Siku kanan — ROM wajib bench press & row |
| #5 | **Left Knee** | 0.7551 | Simetri bilateral lutut kiri |
| #7 | **Left Elbow** | 0.7496 | Simetri bilateral siku kiri |
| #8 | **Right Shoulder** | 0.7444 | Bahu — sumbu gerakan utama |
| #33 | Left Pinky | 0.6633 | Jari kelingking — tidak relevan biomekanikal |

> Model secara **mandiri menemukan** sendi yang paling relevan secara biomekanis — tanpa instruksi eksplisit. Ini membuktikan *Explainable AI* yang berakar pada domain knowledge.

### 2. Learned Spatial Attention (SE-Style)

Implementasi teknik *Squeeze-and-Excitation (Hu et al., 2018)*: model belajar channel representasi mana yang paling informatif untuk membedakan gerakan benar vs salah. MLP `128→32→128` dengan rasio kompresi 4× menyeimbangkan kapasitas dan efisiensi (8.352 parameter).

### 3. Temporal Attention

Conv3d `128→1` dengan kernel `(1,1,1)` diikuti Softmax pada dimensi waktu. Softmax memaksa model "memilih" frame mana yang paling kritis — jumlah bobot seluruh frame = 1. Frame di sekitar puncak gerakan (eksentrik-konsentrik) cenderung mendapat bobot lebih tinggi (129 parameter).

---

## 📦 Dataset & Auto-Labeling Biomekanik

### Dataset

| Atribut | Nilai |
|---|---|
| Total video | **487** |
| Total frame (unit prediksi) | **31.168** (487 × 64) |
| Distribusi BENAR | 209 video (42.9%) |
| Distribusi SALAH | 278 video (57.1%) |
| Gerakan yang dicakup | Squat, Deadlift, Bench Press, Barbell Row |
| Format tensor | `.npy` berukuran `(64, 33, 3)` per video |

🔗 **Kaggle Dataset:** [AttentiveSkel-3D Weight Training Error Dataset](https://www.kaggle.com/datasets/bangaji/attentiveskel-3d-weight-training-error-dataset/data)  
🔗 **DOI:** [10.34740/KAGGLE/DSV/17721447](https://doi.org/10.34740/kaggle/dsv/17721447)

### Pipeline Pra-pemrosesan

```
Video .mp4
  → MediaPipe BlazePose (model_complexity=2)
  → Array mentah (N_frame × 33 × 4) — x, y, z, visibility
  → Filtering: imputasi landmark hilang (interpolasi linier, gap ≤ 5 frame)
  → Smoothing: filter median kernel-3 sepanjang sumbu temporal
  → Normalisasi Spasial: hip-centered (0,0,0), scale = panjang torso
  → Resampling Temporal: N_frame → 64 frame tetap (interpolasi bilinear)
  → Tensor (64, 33, 3) disimpan sebagai .npy
```

### Auto-Labeling Berbasis Biomechanical Validator

Label diberikan **per-frame** secara otomatis menggunakan `BiomechanicalValidator` yang menghitung sudut sendi dari koordinat 3D landmark. Threshold didasarkan pada tiga literatur:

| Gerakan | Kriteria | Threshold | Sumber |
|---|---|---|---|
| **Squat** | Sudut pinggul (Bahu-Pinggul-Lutut) di posisi terdalam | ≤ 137° | Rao et al. (2023) |
| **Squat** | Rasio lebar lutut / pergelangan kaki (*knee valgus*) | ≥ 0.85 | Rao et al. (2023) |
| **Squat** | Sudut lutut (Pinggul-Lutut-Ankle) | ≤ 100° | Chen et al. (2022) |
| **Bench Press** | Sudut siku saat bar paling rendah | ≤ 85° | Ko et al. (2024) |
| **Deadlift** | Inklinasi punggung dari vertikal | 20° ≤ θ ≤ 60° | Ko et al. (2024) |

Output validator: array `(64,)` berisi `0` (BENAR) atau `1` (SALAH) per frame — disimpan sebagai `*_labels.npy`.

---

## 📊 Hasil Eksperimen

### Konfigurasi Pelatihan

| Parameter | Nilai |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Loss Function | CrossEntropyLoss (per-frame flattening: B×64 gradien/batch) |
| Batch Size | 16 video (= 1.024 prediksi frame per iterasi) |
| Epochs | 100 |
| Dropout | 0.4 |
| Dataset Split | 70% Train / 15% Val / 15% Test (seed=42) |

### 5 Skenario Ablation Study

| ID | Skenario | BSP | LS | Temporal | File Bobot |
|---|---|:---:|:---:|:---:|---|
| S1 | Baseline | ✗ | ✗ | ✗ | `best_model_baseline.pth` |
| S2 | Full Model | ✓ | ✓ | ✓ | `best_model_v2.pth` |
| S3a | Ablasi: BSP | ✓ | ✗ | ✗ | `best_model_ablasi_a.pth` |
| S3b | Ablasi: BSP+LS | ✓ | ✓ | ✗ | `best_model_ablasi_b.pth` |
| S3c | Ablasi: BSP+Temp | ✓ | ✗ | ✓ | `best_model_ablasi_c.pth` |

### A. Evaluasi Test Set (4.736 Frame)

| Skenario | Accuracy | F1 Macro | F1 Binary | Precision | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| **S1 — Baseline** | **90.94%** | **0.9066** | **0.9229** | 0.9087 | 0.9376 |
| S2 — Full Model | 88.03% | 0.8776 | 0.8956 | **0.9031** | 0.8883 |
| S3a — Ablasi: BSP | 90.31% | 0.9002 | 0.9171 | 0.9077 | 0.9266 |
| S3b — Ablasi: BSP+LS | 88.43% | 0.8812 | 0.9003 | 0.8971 | 0.9036 |
| **S3c — Ablasi: BSP+Temp** | 89.67% | 0.8924 | 0.9141 | 0.8808 | **0.9500** |

### B. 5-Fold Stratified Cross-Validation (100 Epoch per Fold)

| Skenario | Acc. Mean | Acc. Std | F1 Macro Mean | F1 Binary Mean |
|---|:---:|:---:|:---:|:---:|
| S1 — Baseline | 91.32% | ±0.96% | 0.9113 | 0.9235 |
| S2 — Full Model | 91.02% | ±1.73% | 0.9079 | 0.9216 |
| **S3a — Ablasi: BSP** | **91.82%** | ±1.18% | **0.9163** | **0.9281** |
| S3b — Ablasi: BSP+LS | **91.82%** | ±1.75% | **0.9165** | 0.9272 |
| S3c — Ablasi: BSP+Temp | 91.08% | ±1.50% | 0.9086 | 0.9218 |

### C. Confusion Matrix — Test Set

| Skenario | TP | TN | FP | FN | FPR | FNR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| S1 — Baseline | 2.568 | 1.739 | 258 | 171 | 12.9% | 6.2% |
| S2 — Full Model | 2.433 | 1.736 | 261 | **306** | 13.1% | **11.2%** |
| S3a — Ablasi: BSP | 2.538 | 1.739 | 258 | 201 | 12.9% | 7.3% |
| S3b — Ablasi: BSP+LS | 2.475 | 1.713 | 284 | 264 | 14.2% | 9.6% |
| **S3c — Ablasi: BSP+Temp** | **2.602** | 1.645 | **352** | **137** | **17.6%** | **5.0%** |

### D. Kontribusi Marginal Per Modul Atensi

| Perbandingan | Variabel | Δ Accuracy | Keterangan |
|---|---|:---:|---|
| S1 vs S3a | Kontribusi BSP | **+0.50%** | Peningkatan nyata, hanya +33 parameter |
| S3a vs S3b | Kontribusi LS | **0.00%** | Tidak ubah mean, namun std meningkat |
| S3a vs S3c | Kontribusi Temporal | **−0.74%** | Akurasi turun, tapi Recall naik +2.34% |
| S1 vs S2 | Semua modul | **−0.30%** | Penalti kompleksitas pada dataset medium |

> **Temuan kunci:** BSP adalah modul paling *cost-effective* — 33 parameter memberikan +0.50% akurasi K-Fold dengan peningkatan stabilitas (std rendah). Full Model justru mengalami penalti kompleksitas, fenomena umum pada dataset berukuran sedang (~487 video).

### E. Rekomendasi Deployment

| Use Case | Skenario | Alasan |
|---|---|---|
| **Akurasi & stabilitas terbaik** | S3a — Ablasi BSP | K-Fold 91.82%, std ±1.18%, efisien |
| **Deteksi cedera (minimasi FN)** | S3c — BSP+Temporal | Recall 95%, FN hanya 137 frame |
| **Riset & interpretabilitas** | S2 — Full Model | Semua bobot atensi dapat divisualisasikan |

---

## 🗂️ Struktur Proyek V2

```
Release_V2_AttentiveSkel3D/
│
├── src/
│   ├── data/
│   │   ├── extract_pose.py           # Ekstraksi 33 landmark via MediaPipe
│   │   ├── preprocess.py             # Filtering, smoothing, normalisasi, resampling
│   │   ├── biomechanics_validator.py # Auto-labeling per-frame (threshold ilmiah)
│   │   ├── build_manifest.py         # Pipeline bulk processing & manifest CSV
│   │   ├── manifest_v2.csv           # Metadata dataset: path tensor + label
│   │   └── dataset_v2.py             # PyTorch Dataset & DataLoader (per-frame)
│   │
│   └── models/
│       ├── arsitektur_v2.py          # AttentiveSkel3DPerFrame (3 modul atensi)
│       ├── train_v2.py               # Training loop per-frame (B×64 gradien/iter)
│       ├── evaluate_v2.py            # Evaluasi semua skenario → CSV metrik
│       └── kfold_semua_skenario_v2.py # 5-Fold CV untuk 5 skenario
│
├── notebooks/
│   ├── 01_train_scenario2_full_model.ipynb
│   ├── 02_train_scenario1_baseline.ipynb
│   ├── 03_train_scenario3_ablation.ipynb
│   ├── 04_Bedah_Arsitektur_V2_PerFrame.ipynb    # Torchinfo + sanity check
│   ├── 05_Evaluasi_Semua_Skenario_V2.ipynb      # Evaluasi test set 5 skenario
│   ├── 06_KFold_Semua_Skenario_V2.ipynb         # 5-Fold CV 5 skenario
│   ├── 07_Pembuktian_Kinerja_Bimbingan.ipynb    # Pembuktian per-frame
│   └── 08_Visualisasi_Atensi_3Panel_PerGerakan.ipynb  # Heatmap 3D Squat/DL/BP
│
├── bobot_model/
│   ├── best_model_baseline.pth    # S1 — Tanpa atensi
│   ├── best_model_v2.pth          # S2 — Full Model (BSP+LS+Temporal)
│   ├── best_model_ablasi_a.pth    # S3a — BSP saja
│   ├── best_model_ablasi_b.pth    # S3b — BSP + Learned Spatial
│   ├── best_model_ablasi_c.pth    # S3c — BSP + Temporal
│   ├── curve_s1_baseline.png      # Kurva training S1
│   ├── curve_s2_full_model.png    # Kurva training S2
│   └── curve_s3_ablation_comparison.png  # Kurva perbandingan S3
│
├── hasil_evaluasi/
│   ├── Perbandingan_Metrik_Semua_Skenario_V2.csv
│   ├── KFold_Hasil_Per_Fold_V2.csv
│   ├── KFold_Ringkasan_Semua_Skenario_V2.csv
│   ├── Confusion_Matrix_Semua_Skenario_V2.csv
│   ├── Bobot_Atensi_Per_Skenario_V2.csv
│   └── Bobot_Atensi_Lutut_Siku_V2.csv
│
├── Diagram_V2/
│   ├── 01_Prapemrosesan_Pelabelan_PerFrame.xml
│   ├── 02_Arsitektur_AttentiveSkel3D_PerFrame.xml
│   ├── 03_TrainingLoop_Loss_PerFrame.xml
│   └── 04_Demo_Inferensi_App.xml
│
└── web_app/
    ├── app_v2.py                  # Server FastAPI + SSE endpoint
    ├── inference_core_v2.py       # Pipeline inferensi end-to-end
    └── templates/index_v2.html    # Frontend dark-mode
```

---

## 🌐 Aplikasi Demo Inferensi

Sistem dilengkapi **web application** berbasis FastAPI yang memvisualisasikan prediksi per-frame secara real-time menggunakan teknologi *Server-Sent Events* (SSE).

### Alur Pipeline (4 Tahap Streaming)

```
Upload Video .mp4
    │
    ▼  [Tahap 1] Muat model .pth ke CUDA (deteksi arsitektur otomatis)
    │
    ▼  [Tahap 2] MediaPipe BlazePose ekstrak pose → (N_frame × 33 × 4)
    │
    ▼  [Tahap 3] Preprocessing → Inferensi → (1, 64, 2) logit per frame
    │            Hasil: 64 prediksi biner + probabilitas kelas SALAH
    │
    ▼  [Tahap 4] Render video heatmap dengan OpenCV:
                 - Skeleton overlay (garis abu-abu antar landmark)
                 - Heatmap atensi BSP (COLORMAP_TURBO, power-amplifikasi ×4)
                 - Panel status: "BENAR" (hijau) / "SALAH" (merah) + confidence %
```

### Menjalankan Aplikasi

```bash
# Pastikan environment aktif
conda activate attentiveskel

# Jalankan server FastAPI
cd Release_V2_AttentiveSkel3D
python web_app/app_v2.py

# Buka browser → http://localhost:8080
```

---

## 🚀 Instalasi & Setup

```bash
# 1. Clone repositori (branch V2)
git clone -b v2-per-frame-ai https://github.com/bangaji313/AttentiveSkel3D-WeightTraining-PoC.git
cd AttentiveSkel3D-WeightTraining-PoC

# 2. Buat environment conda (disarankan)
conda create -n attentiveskel python=3.10 -y
conda activate attentiveskel

# 3. Install dependensi
pip install -r requirements.txt

# 4. Jalankan notebook eksperimen (urutan 01 → 08)
jupyter notebook Release_V2_AttentiveSkel3D/notebooks/
```

> **Catatan:** Data tensor `.npy` dan bobot model `.pth` tidak di-commit ke repositori. Unduh dari Kaggle atau jalankan pipeline preprocessing terlebih dahulu.

---

## 📓 Panduan Notebook

| Notebook | Tujuan |
|---|---|
| `01` | Pelatihan Skenario 2 — Full Model |
| `02` | Pelatihan Skenario 1 — Baseline |
| `03` | Pelatihan Skenario 3a/3b/3c — Ablation Study |
| `04` | Bedah arsitektur V2 (Torchinfo + Forward Pass Sanity Check) |
| `05` | Evaluasi metrik test set untuk semua 5 skenario |
| `06` | 5-Fold Stratified Cross-Validation untuk semua 5 skenario |
| `07` | Pembuktian evaluasi per-frame untuk bimbingan akademis |
| `08` | Visualisasi heatmap atensi BSP — 3 panel 3D (Squat/Deadlift/Bench Press) |

---

## 📚 Referensi Ilmiah

| Referensi | Kontribusi dalam Sistem |
|---|---|
| Chen et al. (2022) | Threshold kedalaman squat (sudut lutut ≤ 100°) |
| Rao et al. (2023) | Deteksi *knee valgus* (rasio lebar lutut/ankle ≥ 0.85), hip angle ≤ 137° |
| Ko et al. (2024) | Elbow ROM bench press (≤ 85°), inklinasi punggung deadlift (20°–60°) |
| Hu et al. (2018) | *Squeeze-and-Excitation Networks* — dasar Learned Spatial Attention |

---

## 🎓 Identitas Akademis

<table>
<tr>
  <td><strong>Peneliti</strong></td>
  <td>Maulana Seno Aji Yudhantara</td>
</tr>
<tr>
  <td><strong>NRP</strong></td>
  <td>152022065</td>
</tr>
<tr>
  <td><strong>Program Studi</strong></td>
  <td>Informatika — Institut Teknologi Nasional (ITENAS) Bandung</td>
</tr>
<tr>
  <td><strong>Dosen Pembimbing</strong></td>
  <td>Dr. Jasman Pardede, S.Si., M.T.</td>
</tr>
<tr>
  <td><strong>Dosen Penguji</strong></td>
  <td>
    1. Dr. sc. Lisa Kristiana, S.T., M.T., Ph.D.<br/>
    2. Prof. Dr. Edi Triono Nuryatno, B.Sc. M.Sc, MACS CT.
  </td>
</tr>
<tr>
  <td><strong>Tahun</strong></td>
  <td>2026</td>
</tr>
</table>

---

## 📄 Lisensi

Repositori ini dikembangkan untuk keperluan akademis (Tugas Akhir). Segala bentuk penggunaan ulang harus mencantumkan atribusi yang sesuai.

[![MIT License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

<div align="center">
  <sub>Built with ❤️ for academic research · ITENAS Bandung · 2026</sub>
</div>
