<div align="center">

# 🏋️ AttentiveSkel-3D V2 — Per-Frame Weight Training Form Analysis

### *Proof of Concept: Evaluasi Gerakan Latihan Beban Spasio-Temporal Per-Frame*
### *3D-CNN dengan Biomechanical Spatial Prior, Learned Spatial Attention & Temporal Attention*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Kaggle DOI](https://img.shields.io/badge/Kaggle%20DOI-10.34740%2Fkaggle%2Fdsv%2F19043769-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://doi.org/10.34740/kaggle/dsv/19043769)
[![Branch](https://img.shields.io/badge/Branch-revision%2Fpost--semhas-blueviolet?style=for-the-badge)]()

<br/>

> **📌 Post-Seminar Revision Branch (`revision/post-semhas`)**
> Cabang ini merepresentasikan struktur final dan dokumentasi terevaluasi pasca-Seminar Hasil.
> **Seluruh materi teknis, source code, data, dan script benchmark terpusat di folder [`Release_V2_AttentiveSkel3D/`](Release_V2_AttentiveSkel3D/).**
> 📖 Dokumentasi teknis lengkap dan panduan reproduksi: **[Release_V2_AttentiveSkel3D/README.md](Release_V2_AttentiveSkel3D/README.md)**

> **Tugas Akhir — Program Studi Informatika**
> Institut Teknologi Nasional (ITENAS) Bandung · 2026

</div>

---

## 📌 Ringkasan Proyek

Repositori ini berisi implementasi **AttentiveSkel-3D V2**, sebuah pendekatan klasifikasi Spatio-Temporal berbasis 3D-CNN per-frame untuk analisis indikasi kesalahan gerakan pada latihan beban (*Bench Press*, *Deadlift*, dan *Squat*).

### 🌐 Dataset & Publikasi Kaggle
- **Judul Dataset**: *AttentiveSkel3D-WeightTraining Dataset Per-Frame*
- **Kaggle URL**: [https://www.kaggle.com/datasets/bangaji/attentiveskel3d-weighttraining-dataset-per-frame](https://www.kaggle.com/datasets/bangaji/attentiveskel3d-weighttraining-dataset-per-frame)
- **DOI Terbaru**: [`10.34740/kaggle/dsv/19043769`](https://doi.org/10.34740/kaggle/dsv/19043769)
- **Ukuran Data**: 487 video latihan beban (total 31.168 frame berlabel biner per-frame).

---

## 📊 Ringkasan Hasil Model & Benchmark

### 1. Model Praktis Utama (S3b — BSP + Learned Spatial)
Model **S3b (`best_model_ablasi_c.pth`)** dipilih sebagai model praktis utama untuk evaluasi pascagerakan karena menyediakan atribusi spasial antar-sendi dan mencapai **Recall deviasi postur tertinggi (95.00%)** dengan Akurasi **89.67%** (4.736 frame tes).

### 2. Pengukuran Latensi Model-Only vs End-to-End
- **Model-Only Forward Pass (CUDA GPU)**: Membutuhkan **0.954 ms** per sekuens $(1, 64, 33, 3)$ (throughput **1047.73 sekuens/detik**). *Catatan: Nilai throughput sekuens/detik merupakan kecepatan komputasi tensor model dan bukan FPS pemrosesan video.*
- **Pipeline End-to-End**: Pengukuran pascaperekaman dari video mentah hingga analisis siap membutuhkan **8.72 s – 20.21 s** (*Real-Time Factor* **2.74x – 2.78x**).
- **Status Performa**: Pipeline end-to-end **belum memenuhi kriteria real-time** ($RTF > 1.0$) karena pembacaan frame OpenCV dan ekstraksi MediaPipe Pose mengonsumsi >98% durasi.

---

## 📁 Struktur Direktori Aktif

Seluruh pengembangan dan eksperimen V2 berpusat pada direktori utama:

```text
AttentiveSkel3D-WeightTraining-PoC/
├── README.md                                    # Landing page ini
├── .gitignore                                   # Konfigurasi ignore repository
└── Release_V2_AttentiveSkel3D/                  # PUSAT SELURUH KODE & ARTEFAK V2
    ├── README.md                                # Dokumentasi teknis & panduan eksekusi
    ├── bobot_model/                             # Checkpoint resmi PyTorch (.pth)
    ├── data/                                    # Raw videos, tensors 3D, & labels
    ├── hasil_evaluasi/                          # Hasil evaluasi CSV/JSON canonical
    ├── src/                                     # Source code dataset, model, & benchmark
    ├── web_app/                                 # Web App FastAPI & CLI
    └── archive/                                 # Arsip kode & hasil eksperimen terdahulu
```

📖 **Untuk rincian metodologi, hasil evaluasi per skenario, dan langkah menjalankan script, silakan buka: [Release_V2_AttentiveSkel3D/README.md](Release_V2_AttentiveSkel3D/README.md)**
