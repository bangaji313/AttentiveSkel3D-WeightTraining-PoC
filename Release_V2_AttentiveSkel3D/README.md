# 📦 Release_V2_AttentiveSkel3D — Technical Documentation & Post-Seminar Revision Audit

Dokumen ini berisi spesifikasi teknis, arsitektur, pemetaan checkpoint model, panduan eksekusi, serta hasil evaluasi dan benchmark latensi canonical pasca-Seminar Hasil untuk paket **AttentiveSkel-3D V2**.

---

## 📌 Status Revisi Pasca-Seminar Hasil (`revision/post-semhas`)

Pada cabang `revision/post-semhas`, dilakukan serangkaian audit, perbaikan metodologis, serta retraining model untuk memenuhi masukan penguji Seminar Hasil:

1. **Retraining & Koreksi Mapping Checkpoint S3a (BSP-Only)**:
   - Model S3a dilatih ulang khusus untuk klasifikasi per-frame menggunakan holdout split berbasis level video (340 train, 73 val, 74 test, `seed=42`).
   - Checkpoint resmi S3a tersimpan pada `bobot_model/best_model_s3a_bsp_holdout_v2.pth` (`val_loss = 0.2049`, `val_acc = 91.46%`, `test_acc = 90.20%`, SHA256: `2a5cc2c17ed33172a2f7e3ce1450edb580af2f69e6bba9742fe5d1551a28fad4`).
   - Checkpoint lama berbasis level video yang belum terverifikasi diarsipkan ke `bobot_model/archive/legacy_video_level_best_model_ablasi_a_sha34c84b37.pth`.
2. **Koreksi Resolusi Path Cross-Platform (`dataset_v2.py`)**:
   - `_remap_path()` diperbaiki agar sepenuhnya independen terhadap *Current Working Directory* (CWD) dan mendukung separator `/` maupun `\`.
3. **PemberSIHAN Single Raw Tensor Write per Run (`extract_pose.py`)**:
   - Ditambahkan parameter `save_output: bool = True` pada `PoseExtractor.extract_video()`.
   - Pada benchmark end-to-end, dipanggil `save_output=False` sehingga penulisan file `.npy` mentah hanya dilakukan tepat 1x pada Stage 3 (`raw_tensor_saving_ms`).
4. **Pembaruan Publikasi Kaggle Dataset & DOI**:
   - Judul Dataset: **AttentiveSkel3D-WeightTraining Dataset Per-Frame**
   - URL Kaggle: [https://www.kaggle.com/datasets/bangaji/attentiveskel3d-weighttraining-dataset-per-frame](https://www.kaggle.com/datasets/bangaji/attentiveskel3d-weighttraining-dataset-per-frame)
   - DOI Terbaru: [`10.34740/kaggle/dsv/19043769`](https://doi.org/10.34740/kaggle/dsv/19043769)
   - *(Catatan: DOI terdahulu `10.34740/kaggle/dsv/19038457` dan `10.34740/kaggle/dsv/19041891` tetap disimpan dalam arsip historis sebagai bukti evolusi iterasi dataset).*
5. **Koreksi & Sintesis Kurva Pembelajaran S3 (Canonical Holdout Curves)**:
   - Evaluasi K-Fold Cross-Validation (5-fold) dan Evaluasi Holdout (fixed-split) merupakan dua jalur evaluasi independen. Metrik kuantitatif canonical keduanya tidak berubah.
   - Sumbu grafik disesuaikan ke `Validation Loss (Cross-Entropy)` dan batas rentang plot disesuaikan (0.15 hingga >0.67) agar seluruh titik data 100 epoch per skenario terlihat tanpa terpotong.
   - Provenance data 300 baris diekspor ke `hasil_evaluasi/Holdout_Training_History_S3_Canonical_V2.csv` (S3a Ep 52 val_loss 0.2049, S3b Ep 63 val_loss 0.2082, S3c Ep 67 val_loss 0.2179).
   - Notebook `notebooks/06_KFold_Semua_Skenario_V2.ipynb` adalah notebook K-Fold canonical resmi, sedangkan `notebooks/06_kfold_cross_validation_v2_PerFrame.ipynb` (SHA256: `6882066291d0b1571201ceaa12f7daee47fa511748eb59cda5f39c6991c02894`) merupakan duplikat historis yang dipertahankan.

---

## 📂 Struktur Direktori `Release_V2_AttentiveSkel3D/`

```text
Release_V2_AttentiveSkel3D/
├── README.md                                          # Dokumentasi teknis V2 (File ini)
├── .gitignore                                         # Aturan ignore spesifik paket V2
├── bobot_model/                                       # Checkpoint resmi PyTorch model V2 (.pth)
│   ├── best_model_baseline.pth                        # S1 Baseline 3D-CNN (101,858 params)
│   ├── best_model_v2.pth                              # S2 Full Model (BSP + Learned + Temporal) (110,372 params)
│   ├── best_model_s3a_bsp_holdout_v2.pth              # S3a BSP-Only (Holdout Per-Frame Resmi) (101,891 params)
│   ├── best_model_ablasi_b.pth                        # S3c BSP + Temporal Attention (102,020 params)
│   ├── best_model_ablasi_c.pth                        # S3b BSP + Learned Spatial (Model Praktis) (110,243 params)
│   ├── best_model_s3a_bsp_holdout_v2_metadata.json    # Metadata training S3a
│   └── archive/                                       # Local legacy weights archive (TIDAK di-track)
│       └── legacy_video_level_best_model_ablasi_a_sha34c84b37.pth
├── data/                                              # Dataset & manifest
│   ├── raw/                                           # Video mentah (BenchPress, Deadlift, Squat)
│   ├── tensors/                                       # Tensor 3D per-frame (64, 33, 3)
│   ├── v2_labels/                                     # Label biner per-frame (64,)
│   └── manifest_v2.csv                                # Manifest 487 sampel video V2
├── hasil_evaluasi/                                    # Artefak hasil evaluasi & benchmark canonical
│   ├── Latency_EndToEnd_AttentiveSkel3D_V2.json       # Metadata & detail per-run 10-run benchmark
│   ├── Latency_EndToEnd_PerRun_AttentiveSkel3D_V2.csv  # Raw data 33 baris per-run (3 warm-up + 30 measured)
│   ├── Latency_EndToEnd_Summary_AttentiveSkel3D_V2.csv # Ringkasan statistik 10-run end-to-end
│   ├── Latency_ModelOnly_AttentiveSkel3D_V2.csv       # Ringkasan latency model-only (CPU vs CUDA)
│   ├── Latency_ModelOnly_AttentiveSkel3D_V2.json      # Metadata latency model-only
│   ├── Confusion_Matrix_Semua_Skenario_V2.csv         # Matrix evaluasi 5 skenario V2
│   ├── Perbandingan_Metrik_Semua_Skenario_V2.csv      # Ringkasan metrik akurasi, F1, val loss 5 skenario
│   ├── Pemetaan_Detail_Per_Frame_V2.csv               # Pemetaan 31.168 frame
│   └── archive/                                       # Arsip hasil evaluasi terdahulu
│       ├── legacy_pre_v2/                             # Arsip berkas legacy sebelum penataan V2
│       └── mapping_lama/                              # Hasil evaluasi sebelum koreksi mapping S3a
├── src/                                               # Source code utama
│   ├── data/                                          # Data loading & preprocessing (dataset_v2.py, extract_pose.py, preprocess.py)
│   ├── models/                                        # Arsitektur & script evaluasi (arsitektur_v2.py, evaluasi_semua_skenario_v2.py)
│   └── benchmark/                                     # Script benchmark (benchmark_latency_v2.py, benchmark_end_to_end_v2.py)
└── web_app/                                           # Antarmuka Web Explainer & CLI
    ├── app_v2.py                                      # Server FastAPI & Dashboard Web UI
    ├── explainer_v2.py                                # Modul XAI Joint Influence Attribution
    ├── inference_cli_v2.py                            # CLI Inference Tool
    └── templates/                                     # Template HTML UI
```

---

## 🎯 Pemetaan Checkpoint & Metrik Evaluasi Canonical

Evaluasi dilakukan pada 4.736 frame tes independen dari 74 video holdout:

| Skenario | Konfigurasi Atensi | Checkpoint Resmi | Parameter | Accuracy (%) | Precision | Recall (Deviasi) | F1-Binary |
|---|---|---|---|---|---|---|---|
| **S1 (Baseline)** | 3D-CNN Tanpa Atensi | `best_model_baseline.pth` | 101,858 | **90.94%** | 0.9087 | 0.9376 | 0.9229 |
| **S2 (Full Model)** | BSP + Learned Spatial + Temporal | `best_model_v2.pth` | 110,372 | 88.03% | 0.9031 | 0.8883 | 0.8956 |
| **S3a (Ablasi A)** | BSP Only | `best_model_s3a_bsp_holdout_v2.pth` | 101,891 | 90.20% | **0.9373** | 0.8901 | 0.9131 |
| **S3b (Ablasi C)** | BSP + Learned Spatial | `best_model_ablasi_c.pth` | 110,243 | 89.67% | 0.8808 | **95.00%** | **0.9141** |
| **S3c (Ablasi B)** | BSP + Temporal Attention | `best_model_ablasi_b.pth` | 102,020 | 88.43% | 0.8971 | 0.9036 | 0.9003 |

> **Rasional Pemilihan Model Praktis (S3b)**:
> Model **S3b (`best_model_ablasi_c.pth`)** dipilih sebagai model praktis utama dalam aplikasi demo analisis pascaperekaman dan benchmark pipeline karena:
> 1. Mengintegrasikan *Biomechanical Spatial Prior* (BSP) dan *Learned Spatial Attention* untuk penyajian atribusi tingkat sendi;
> 2. Mencapai **Recall deviasi postur tertinggi (95.00%)** yang sangat krusial agar kesalahan gerakan tidak terlewat (meminimalisasi *False Negative*);
> 3. Menghindari *overhead* komputasi modul atensi temporal sehingga lebih efisien.

---

## ⚡ Hasil Benchmark Latency Model-Only (CPU vs CUDA)

Diuji pada tensor input `(1, 64, 33, 3)` selama 50 warm-up iterations dan 500 measured iterations:

| Skenario | Checkpoint | Parameter | CPU Mean (ms) | CPU P95 (ms) | CUDA Mean (ms) | CUDA P95 (ms) | CUDA Throughput (Sekuens/s) |
|---|---|---|---|---|---|---|---|
| **S1 Baseline** | `best_model_baseline.pth` | 101,858 | 1.194 ms | 1.495 ms | 0.734 ms | 0.804 ms | 1361.71 sekuens/s |
| **S2 Full Model** | `best_model_v2.pth` | 110,372 | 1.630 ms | 1.989 ms | 1.087 ms | 1.148 ms | 919.80 sekuens/s |
| **S3a BSP-Only** | `best_model_s3a_bsp_holdout_v2.pth` | 101,891 | 1.275 ms | 1.575 ms | 0.771 ms | 0.837 ms | 1297.24 sekuens/s |
| **S3b BSP+Learned** | `best_model_ablasi_c.pth` | 110,243 | **1.264 ms** | **1.517 ms** | **0.954 ms** | **1.017 ms** | **1047.73 sekuens/s** |
| **S3c BSP+Temporal** | `best_model_ablasi_b.pth` | 102,020 | 1.468 ms | 1.737 ms | 0.897 ms | 0.985 ms | 1114.43 sekuens/s |

*Catatan: Nilai CUDA Throughput (misal 1047.73 sekuens/s pada S3b) menyatakan kecepatan komputasi tensor model PyTorch per detik dan BUKAN kecepatan FPS pemrosesan video.*

---

## ⏱️ Hasil Benchmark Latency End-to-End Canonical (10 Measured Runs)

Pengukuran dilakukan pada 3 video mentah representatif menggunakan model praktis **S3b (`best_model_ablasi_c.pth`)** di perangkat GPU CUDA. Cold-start load time model: **101.21 ms**.

### 1. Ringkasan Kinerja Utama Per Latihan

| Latihan | Resolusi Video | Durasi Video | Analysis-Ready Mean (ms) | Analysis-Ready P95 (ms) | Eff. Pipeline Speed | Real-Time Factor (RTF) | Sequential Cap+Proc Est. |
|---|---|---|---|---|---|---|---|
| **Bench Press** | 1080x1920 | 3.13 s | **8,722.14 ms** | 8,916.84 ms | 10.8 FPS | 2.784x | 11.86 s |
| **Deadlift** | 1080x1920 | 7.37 s | **20,208.91 ms** | 20,747.57 ms | 10.9 FPS | 2.743x | 27.58 s |
| **Squat** | 1080x1920 | 3.30 s | **9,198.78 ms** | 9,265.72 ms | 10.8 FPS | 2.788x | 12.50 s |

---

### 2. Rincian Stage Breakdown & Kontribusi Persentase (Mean ± Std ms)

#### A. Bench Press (`primer_benchpress_frontal_subjek01_rep1.mp4` — 94 Frame)
1. **Video Stream Open & Metadata Inspection**: `22.54 ms ± 0.66 ms` (0.26% dari Analysis-Ready)
2. **Video Decoding & BlazePose Extraction**: `8,602.07 ms ± 112.26 ms` (**98.62%** dari Analysis-Ready)
3. **Raw Tensor Saving (Single Write)**: `1.84 ms ± 0.46 ms` (0.02% dari Analysis-Ready)
4. **Preprocessing (Resample 64 frame, Smooth, Norm)**: `26.51 ms ± 1.61 ms` (0.30% dari Analysis-Ready)
5. **Model Inference & Joint Attribution**: `69.18 ms ± 11.18 ms` (0.79% dari Analysis-Ready)
- ⏱️ **TIME-TO-ANALYSIS-READY**: **`8,722.14 ms ± 121.26 ms`** (100.00%)
6. **Video Heatmap Rendering (64 frame)**: `7,307.75 ms ± 108.45 ms` (45.6% dari Total)
- 🎬 **TOTAL WITH RENDERING**: **`16,029.89 ms ± 222.29 ms`**

#### B. Deadlift (`primer_deadlift_lateral_subjek01_rep1.mp4` — 221 Frame)
1. **Video Stream Open & Metadata Inspection**: `15.78 ms ± 1.01 ms` (0.08% dari Analysis-Ready)
2. **Video Decoding & BlazePose Extraction**: `20,066.65 ms ± 321.69 ms` (**99.30%** dari Analysis-Ready)
3. **Raw Tensor Saving (Single Write)**: `1.57 ms ± 0.39 ms` (0.01% dari Analysis-Ready)
4. **Preprocessing (Resample 64 frame, Smooth, Norm)**: `30.58 ms ± 1.04 ms` (0.15% dari Analysis-Ready)
5. **Model Inference & Joint Attribution**: `94.33 ms ± 62.00 ms` (0.47% dari Analysis-Ready)
- ⏱️ **TIME-TO-ANALYSIS-READY**: **`20,208.91 ms ± 341.88 ms`** (100.00%)
6. **Video Heatmap Rendering (64 frame)**: `7,593.78 ms ± 184.53 ms` (27.3% dari Total)
- 🎬 **TOTAL WITH RENDERING**: **`27,802.69 ms ± 471.74 ms`**

#### C. Squat (`primer_squat_frontal_subjek01_rep1.mp4` — 99 Frame)
1. **Video Stream Open & Metadata Inspection**: `19.25 ms ± 0.98 ms` (0.21% dari Analysis-Ready)
2. **Video Decoding & BlazePose Extraction**: `9,075.45 ms ± 59.37 ms` (**98.66%** dari Analysis-Ready)
3. **Raw Tensor Saving (Single Write)**: `1.88 ms ± 0.49 ms` (0.02% dari Analysis-Ready)
4. **Preprocessing (Resample 64 frame, Smooth, Norm)**: `25.97 ms ± 2.85 ms` (0.28% dari Analysis-Ready)
5. **Model Inference & Joint Attribution**: `76.23 ms ± 19.66 ms` (0.83% dari Analysis-Ready)
- ⏱️ **TIME-TO-ANALYSIS-READY**: **`9,198.78 ms ± 67.85 ms`** (100.00%)
6. **Video Heatmap Rendering (64 frame)**: `7,370.43 ms ± 51.72 ms` (44.5% dari Total)
- 🎬 **TOTAL WITH RENDERING**: **`16,569.21 ms ± 95.98 ms`**

---

## 🔍 Catatan Metodologis & Penjelasan Operasional

1. **Perbedaan Pure Model Inference vs Model Inference + Joint Attribution**:
   - Forward pass model-only PyTorch untuk tensor $(1, 64, 33, 3)$ pada GPU CUDA sangat ringan (**0.954 ms**).
   - Namun, Stage 5 (`model_inference_and_joint_attribution_ms`) mencakup pembuatan 33 tensor perturbasi sendi (ablasia temporal-mean per-joint) dan mengeksekusi **3 kali model forward pass** (1x model forward asli, 1x original logits di `joint_influence`, dan 1x batched perturbation forward dengan $B=33$), serta kalkulasi skor atribusi biomekanik. Hal ini menyebabkan durasi Stage 5 menjadi **~69 – 94 ms**.
2. **Cakupan Stage 2 (`video_decoding_and_blazepose_extraction_ms`)**:
   - Stage 2 mencakup pembacaan frame video menggunakan OpenCV `cap.read()`, konversi warna BGR-ke-RGB, dan inferensi MediaPipe Pose (`model_complexity=2`). Tahap ini mengonsumsi **>98.6%** dari total waktu *Time-to-Analysis-Ready*, menjadikannya *bottleneck* utama sistem.
3. **Penulisan Single Raw Tensor per Run**:
   - Dengan menyetel `save_output=False` pada `PoseExtractor`, penulisan file `.npy` mentah ke disk hanya dilakukan **tepat 1x** pada Stage 3 (`raw_tensor_saving_ms`), menghindari I/O ganda.
4. **Inisialisasi Per-Run MediaPipe Pose**:
   - Setiap iterasi *run* benchmark membuat ulang objek `PoseExtractor`, sehingga mengukur latensi cold-start inisialisasi model MediaPipe Pose pada setiap *run*, bukan *persistent steady-state daemon*.
5. **Status Performa & Keterbatasan Metodologis**:
   - Nilai *Real-Time Factor* (RTF) sistem saat ini berkisar antara **2.74x hingga 2.78x** ($RTF > 1.0$), sehingga pipeline saat ini **belum memenuhi kriteria real-time** pada pemrosesan sekuensial tunggal.
   - **Metodologi Pengujian**: Pengukuran ini menggunakan video *pre-recorded* lokal, tidak mencakup latensi jaringan (network upload/download), tidak mencakup latensi UI browser, bukan *live-camera latency*, memerlukan konteks sekuens 64 frame untuk klasifikasi, dan **belum membuktikan kemampuan pencegahan cedera secara klinis**.

---

## 🛠️ Panduan Eksekusi Script

Seluruh perintah di bawah ini dijalankan dari root repository atau folder `Release_V2_AttentiveSkel3D/`:

### 1. Evaluasi Akurasi 5 Skenario Model V2
```bash
python Release_V2_AttentiveSkel3D/src/models/evaluasi_semua_skenario_v2.py
```

### 2. Benchmark Latency Model-Only (CPU & CUDA)
```bash
python Release_V2_AttentiveSkel3D/src/benchmark/benchmark_latency_v2.py
```

### 3. Benchmark Latency End-to-End Canonical (10-Run)
```bash
python Release_V2_AttentiveSkel3D/src/benchmark/benchmark_end_to_end_v2.py
```

### 4. Menjalankan Dashboard Web App (Aplikasi Demo Analisis Pascaperekaman)
```bash
python Release_V2_AttentiveSkel3D/web_app/app_v2.py
```
Akses di browser pada `http://localhost:8000`.
