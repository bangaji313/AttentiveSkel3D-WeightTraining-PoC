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
4. **Benchmark Latency End-to-End Canonical (10 Measured Runs)**:
   - Dijalankan benchmark steady-state 10-run (+ 1 warm-up) per video pada 3 video mentah (*Bench Press*, *Deadlift*, *Squat*).
   - Nomenklatur tahap disempurnakan menjadi `time_to_analysis_ready_ms`, `video_stream_open_and_metadata_inspection_ms`, `video_decoding_and_blazepose_extraction_ms`, dan `model_inference_and_joint_attribution_ms`.

---

## 📂 Struktur Direktori `Release_V2_AttentiveSkel3D/`

```text
Release_V2_AttentiveSkel3D/
├── README.md                                          # Dokumentasi teknis V2 (File ini)
├── .gitignore                                         # Aturan ignore spesifik paket V2
├── bobot_model/                                       # Checkpoint resmi PyTorch model V2 (.pth)
│   ├── best_model_baseline.pth                        # S1 Baseline 3D-CNN
│   ├── best_model_v2.pth                              # S2 Full Model (BSP + Learned + Temporal)
│   ├── best_model_s3a_bsp_holdout_v2.pth              # S3a BSP-Only (Holdout Per-Frame Resmi)
│   ├── best_model_ablasi_b.pth                        # S3c BSP + Temporal Attention
│   ├── best_model_ablasi_c.pth                        # S3b BSP + Learned Spatial (Model Praktis Web App)
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
│       └── mapping_lama/                              # Hasil evaluasi sebelum koreksi mapping S3a
├── src/                                               # Source code utama
│   ├── data/                                          # Data loading & preprocessing
│   │   ├── dataset_v2.py                              # PerFrameDataset & Dataloader
│   │   ├── extract_pose.py                            # PoseExtractor (MediaPipe BlazePose)
│   │   └── preprocess.py                             # DataPreprocessor (Resampling, Smooth, Norm)
│   ├── models/                                        # Arsitektur & script evaluasi
│   │   ├── arsitektur_v2.py                           # Model AttentiveSkel3DPerFrame
│   │   └── evaluasi_semua_skenario_v2.py              # Script evaluasi 5 skenario model V2
│   ├── benchmark/                                     # Script pengujian latensi
│   │   ├── benchmark_latency_v2.py                    # Benchmark Model-Only (CPU & CUDA)
│   │   └── benchmark_end_to_end_v2.py                 # Benchmark End-to-End Pipeline (10-Run)
│   └── visualization/                                 # Script generator visualisasi
└── web_app/                                           # Antarmuka Web Explainer & CLI
    ├── app_v2.py                                      # Server FastAPI & Dashboard Web UI
    ├── explainer_v2.py                                # Modul XAI Joint Influence Attribution
    ├── inference_cli_v2.py                            # CLI Inference Tool
    └── templates/                                     # Template HTML UI
```

---

## 🎯 Pemetaan Checkpoint Resmi 5 Skenario

| Skenario | Konfigurasi Atensi | File Checkpoint Resmi | Parameter | Accuracy (Test) | Val Loss |
|---|---|---|---|---|---|
| **S1 (Baseline)** | 3D-CNN Tanpa Atensi | `best_model_baseline.pth` | 100,546 | 89.19% | 0.2514 |
| **S2 (Full Model)** | BSP + Learned Spatial + Temporal | `best_model_v2.pth` | 110,372 | 90.54% | 0.2241 |
| **S3a (Ablasi A)** | BSP Only | `best_model_s3a_bsp_holdout_v2.pth` | 101,891 | 90.20% | **0.2049** |
| **S3b (Ablasi C)** | BSP + Learned Spatial | `best_model_ablasi_c.pth` | 101,956 | **91.89%** | 0.2185 |
| **S3c (Ablasi B)** | BSP + Temporal Attention | `best_model_ablasi_b.pth` | 108,962 | 90.54% | 0.2215 |

> **Model Praktis Utama**: **S3b (BSP + Learned Spatial)** dipadukan dalam Web App (`explainer_v2.py`) dan Benchmark End-to-End karena mencapai akurasi test tertinggi (**91.89%**) dengan estimasi atensi spasial yang cepat tanpa overhead atensi temporal.

---

## ⚡ Hasil Benchmark Latency Model-Only (CPU vs CUDA)

Diuji pada tensor input `(1, 64, 33, 3)` selama 50 warm-up iterations dan 500 measured iterations:

| Skenario | CPU Latency Mean (ms) | CPU Latency P95 (ms) | CUDA Latency Mean (ms) | CUDA Latency P95 (ms) | CUDA Speedup |
|---|---|---|---|---|---|
| **S1 Baseline** | 4.85 ms | 5.42 ms | 1.12 ms | 1.35 ms | 4.33x |
| **S2 Full Model** | 5.68 ms | 6.21 ms | 1.38 ms | 1.62 ms | 4.11x |
| **S3a BSP-Only** | 5.12 ms | 5.65 ms | 1.18 ms | 1.40 ms | 4.34x |
| **S3b BSP + Learned** | 5.24 ms | 5.80 ms | 1.21 ms | 1.42 ms | 4.33x |
| **S3c BSP + Temporal** | 5.51 ms | 6.08 ms | 1.32 ms | 1.55 ms | 4.17x |

---

## ⏱️ Hasil Benchmark Latency End-to-End Canonical (10 Measured Runs)

Pengukuran dilakukan pada 3 video mentah representatif menggunakan model praktis **S3b (`best_model_ablasi_c.pth`)** di perangkat NVIDIA GPU (CUDA). Cold-start load time model: **101.21 ms**.

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
   - Pure forward pass model PyTorch untuk tensor $(1, 64, 33, 3)$ pada GPU CUDA hanya membutuhkan waktu **~1.21 ms**.
   - Namun, Stage 5 (`model_inference_and_joint_attribution_ms`) mencakup pembuatan 33 tensor perturbasi sendi (temporal-mean ablation) dan mengeksekusi **3 kali model forward pass** (1x model forward asli, 1x original logits di `joint_influence`, dan 1x batched perturbation forward dengan $B=33$), serta kalkulasi skor attribution biomekanis. Hal ini menyebabkan durasi Stage 5 menjadi **~69 – 94 ms**. Jika terjadi fallback memori GPU, eksekusi dilakukan sebanyak 35 forward passes.
2. **Cakupan Stage 2 (`video_decoding_and_blazepose_extraction_ms`)**:
   - Stage 2 mencakup pembacaan frame video menggunakan OpenCV `cap.read()`, konversi warna BGR-ke-RGB, dan inferensi MediaPipe Pose (`model_complexity=2`). Tahap ini mengonsumsi **>98.6%** dari total waktu *Time-to-Analysis-Ready*, menjadikannya *bottleneck* utama sistem.
3. **Penulisan Single Raw Tensor per Run**:
   - Dengan menyetel `save_output=False` pada `PoseExtractor`, penulisan file `.npy` mentah ke disk hanya dilakukan **tepat 1x** pada Stage 3 (`raw_tensor_saving_ms`), menghindari I/O ganda.
4. **Inisialisasi Per-Run MediaPipe Pose**:
   - Setiap iterasi *run* benchmark membuat ulang objek `PoseExtractor`, sehingga mengukur latensi cold-start inisialisasi model MediaPipe Pose pada setiap *run*, bukan *persistent steady-state daemon*.
5. **Status Real-Time & Keterbatasan Metodologis**:
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
*(Gunakan `--smoke-test` untuk pengujian cepat 1 warm-up + 1 run).*

### 4. Menjalankan Dashboard Web App (FastAPI UI)
```bash
python Release_V2_AttentiveSkel3D/web_app/app_v2.py
```
Akses di browser pada `http://localhost:8000`.
