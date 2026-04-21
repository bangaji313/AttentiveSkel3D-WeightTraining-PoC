<div align="center">

# 🏋️ AttentiveSkel3D — Weight Training Form Error Detection

### *A Proof of Concept for Enhancing Weight Training Form Error Detection*
### *Using 3D-CNN and Biomechanical Attention Mechanism*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=for-the-badge)]()
[![Model Size](https://img.shields.io/badge/Model%20Size-~0.39%20MB-blue?style=for-the-badge)]()

<br/>

> **Tugas Akhir — Program Studi Informatika**
> Institut Teknologi Nasional (ITENAS) Bandung · 2026

</div>

---

## 🧠 Apa Ini & Mengapa Penting?

Pernahkah Anda pergi ke gym dan berlatih sendirian tanpa pelatih? Tanpa bimbingan yang tepat, sangat mudah melakukan gerakan yang **salah** — dan kesalahan yang terlihat sepele seperti lutut masuk ke dalam saat squat, punggung membungkuk saat deadlift, atau siku terlalu terbuka saat bench press, dapat berujung pada **cedera serius** yang mengganggu aktivitas sehari-hari bahkan dalam jangka panjang.

Sayangnya, **jasa pelatih pribadi (personal trainer)** tidak terjangkau oleh semua orang. Di sinilah proyek ini hadir sebagai solusi.

### 🎯 Solusi: Pelatih Virtual Berbasis AI

**AttentiveSkel-3D** adalah sistem kecerdasan buatan yang bertindak layaknya seorang pelatih virtual. Cukup rekam latihan Anda menggunakan **kamera biasa** (tidak perlu sensor khusus), dan sistem ini akan:

| Gerakan | Yang Dideteksi |
|---|---|
| 🦵 **Squat** | Kedalaman squat kurang memadai, *knee valgus* (lutut jatuh ke dalam) |
| 🏗️ **Deadlift** | Punggung membungkuk berlebihan (*spine flexion*), posisi tidak netral |
| 🏋️ **Bench Press** | *Range of motion* siku kurang penuh, sudut lengan tidak optimal |

Sistem secara otomatis mengklasifikasikan setiap repetisi sebagai **Benar ✅** atau **Salah ❌**, memberikan umpan balik berbasis biomekanika yang selama ini hanya bisa diberikan oleh pelatih berpengalaman.

---

## ⚙️ Bagaimana Cara Kerjanya? *(Penjelasan Teknis)*

### 1. 🎥 Pipeline Pemrosesan Data

Sistem memproses video latihan melalui serangkaian tahap yang terstruktur:

```
Video .mp4  (data/raw/<Exercise>/)
      │
      ▼  [MediaPipe BlazePose — model_complexity=2]
      │  Ekstraksi 33 pose keypoints per frame → (T, 33, 4) [x, y, z, visibility]
      │
      ▼  [Preprocessing & Smoothing — src/data/preprocess.py]
      │  • Imputasi landmark hilang (interpolasi linier)
      │  • Smoothing temporal (Savitzky-Golay filter)
      │  • Normalisasi spasial (hip-centered, unit-scale)
      │  • Resampling temporal ke 64 frame tetap
      │
      ▼  Tensor Siap Model: (64, 33, 3)  — 64 frame × 33 landmark × [x, y, z]
      │
      ▼  [BiomechanicalValidator — Auto Ground Truth Labeling]
         Evaluasi sudut sendi berdasarkan referensi jurnal biomekanika:
         • Chen (2022) — Squat depth & Deadlift spine alignment
         • Rao (2023)  — Knee valgus detection
         • Ko (2024)   — Bench Press elbow ROM & Deadlift criteria
         → Label: 0 (Benar) atau 1 (Salah)
```

### 2. 🤖 Arsitektur Model: AttentiveSkel-3D

Model dirancang **ringan** namun **cerdas** dengan menggabungkan dua komponen utama:

#### 🔷 Biomechanical Attention Module

Sebelum data masuk ke jaringan konvolusi, modul atensi biomekanik memberikan **bobot berbeda** kepada setiap sendi tubuh. Sendi yang kritis secara biomekanika (misalnya lutut dan pinggul untuk squat) mendapat perhatian lebih tinggi, sehingga model berfokus pada informasi yang paling relevan.

Mekanisme atensi terdiri dari tiga komponen:
- **Spatial Prior Mask** — *learnable parameter* `(1,1,1,33,1)` yang secara implisit mempelajari kepentingan relatif tiap dari 33 sendi tubuh
- **Learned Spatial Attention** — dioptimasi bersama seluruh parameter model via *backpropagation*
- **Temporal Attention** — representasi spatio-temporal memungkinkan model menangkap pola gerakan lintas waktu

#### 🔷 3D-CNN Backbone

Setelah dibobot oleh modul atensi, data diproses oleh tiga blok konvolusi 3D yang menangkap pola spasial (konfigurasi sendi) dan temporal (pergerakan antar frame) secara bersamaan:

```
Input (B, 64, 33, 3)
  → Permute + Unsqueeze → (B, 3, 64, 33, 1)
  → ×sigmoid(Spatial Prior)              ← Biomechanical Attention
  → Conv3D Block 1: 3→32 ch, kernel(3,3,1), BN, ReLU, MaxPool
  → Conv3D Block 2: 32→64 ch, kernel(3,3,1), BN, ReLU, MaxPool
  → Conv3D Block 3: 64→128 ch, kernel(3,3,1), BN, ReLU, AdaptiveAvgPool
  → Flatten → Linear(128→64) → ReLU → Dropout(0.4) → Linear(64→2)
  → Output: [logit_Benar, logit_Salah]
```

#### 📊 Efisiensi Model

| Metrik | Nilai |
|---|---|
| Total Parameter | **101.891** |
| Ukuran Model | **~0.39 MB** |
| Input Tensor | `(B, 64, 33, 3)` |
| Output | 2 kelas (Benar / Salah) |
| Framework | PyTorch 2.x |

Model ini dirancang untuk dapat berjalan cepat bahkan tanpa GPU khusus, menjadikannya kandidat kuat untuk *deployment* pada perangkat *edge* atau aplikasi mobile di masa depan.

---

## 🗺️ Diagram Arsitektur Pipeline

```mermaid
flowchart LR
    A([🎥 Input Video\n.mp4]) --> B[MediaPipe\nBlazePose\nPose Extraction]
    B --> C[/"Sekuens Skeleton 3D\n(T × 33 × 4)"/]
    C --> D[Preprocessing\n& Smoothing\nNormalisasi + Resample]
    D --> E[/"Spatio-Temporal\nTensor\n64 × 33 × 3"/]
    E --> F[[Biomechanical\nAttention Module\nSpatial Prior Mask]]
    F --> G[[3D-CNN\nBackbone\n3 Conv3D Blocks]]
    G --> H[Global\nAverage\nPooling]
    H --> I[Fully Connected\nClassifier\n128→64→2]
    I --> J{Klasifikasi}
    J --> K([✅ Benar\nLabel: 0])
    J --> L([❌ Salah\nLabel: 1])

    style A fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style E fill:#7B68EE,stroke:#4B3BAA,color:#fff
    style F fill:#E8534A,stroke:#B03028,color:#fff
    style G fill:#E8534A,stroke:#B03028,color:#fff
    style J fill:#F5A623,stroke:#C07800,color:#fff
    style K fill:#27AE60,stroke:#1A7A42,color:#fff
    style L fill:#E74C3C,stroke:#A93226,color:#fff
```

**Secara sederhana:** Video latihan Anda "dibaca" oleh sistem kamera, posisi 33 titik tubuh dilacak setiap saat, lalu pola gerakan tersebut dianalisis oleh AI yang sudah dilatih untuk membedakan gerakan benar dan salah — persis seperti seorang pelatih yang mengamati dan menilai teknik Anda.

**Secara teknis:** Video diproses frame-by-frame oleh MediaPipe BlazePose menghasilkan tensor `(T, 33, 4)`. Setelah preprocessing (interpolasi, smoothing Savitzky-Golay, normalisasi hip-centered, resampling), tensor berukuran tetap `(64, 33, 3)` dibentuk. Tensor ini dimodulasi oleh *Spatial Prior Mask* berukuran `(1,1,1,33,1)` via sigmoid sebelum memasuki tiga blok Conv3D dengan kernel `(3,3,1)` yang mengekstraksi fitur spatio-temporal. Representasi akhir di-*pool* dan diklasifikasikan oleh MLP dua lapis dengan Dropout(0.4) sebagai regularisasi.

---

## 📓 Struktur Notebook Eksperimen

| # | Notebook | Deskripsi |
|---|---|---|
| 01 | `01_pose_extraction_test.ipynb` | Uji ekstraksi pose MediaPipe BlazePose dari video |
| 02 | `02_data_preprocessing_test.ipynb` | Uji pipeline preprocessing & normalisasi skeleton |
| 02b | `02b_auto_labeling_test.ipynb` | Simulasi & verifikasi sistem pelabelan otomatis berbasis biomekanika |
| 03 | `03_model_architecture_test.ipynb` | Uji arsitektur AttentiveSkel-3D & parameter count |
| 04 | `04_dataloader_test.ipynb` | Uji bulk processing, manifest CSV, & DataLoader PyTorch |
| 05 | `05_training_test.ipynb` | Uji loop pelatihan, validasi, & penyimpanan checkpoint |
| 06 | `06_attention_visualization.ipynb` | Visualisasi Biomechanical Attention — overlay heatmap per sendi |

---

## 🗂️ Struktur Folder

```
AttentiveSkel3D-WeightTraining-PoC/
│
├── data/                        # ⚠️  Diabaikan oleh Git (.gitignore)
│   ├── raw/
│   │   ├── Squat/               # Video .mp4 gerakan Squat
│   │   ├── Deadlift/            # Video .mp4 gerakan Deadlift
│   │   └── BenchPress/          # Video .mp4 gerakan Bench Press
│   └── processed/
│       ├── tensors/             # File .npy hasil ekstraksi & preprocessing
│       └── manifest.csv         # Label otomatis + audit trail per sampel
│
├── notebooks/                   # Jupyter Notebook eksperimen (01 s.d. 06)
│
├── src/
│   ├── data/
│   │   ├── extract_pose.py      # Ekstraksi 33 keypoints via MediaPipe
│   │   ├── preprocess.py        # Smoothing, normalisasi, resampling
│   │   ├── build_dataset.py     # Pipeline bulk processing + auto-labeling
│   │   ├── dataset.py           # PyTorch Dataset & DataLoader
│   │   └── biomechanics_validator.py  # Validator otomatis berbasis jurnal
│   └── models/
│       ├── model_3dcnn.py       # Arsitektur AttentiveSkel-3D
│       └── train.py             # Training loop & checkpoint saving
│
├── models/
│   └── saved_models/            # ⚠️  Bobot .pth, diabaikan oleh Git
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Instalasi & Menjalankan Proyek

```bash
# 1. Clone repositori
git clone https://github.com/bangaji313/AttentiveSkel3D-WeightTraining-PoC.git
cd AttentiveSkel3D-WeightTraining-PoC

# 2. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install semua dependensi
pip install -r requirements.txt

# 4. Jalankan notebook eksperimen secara berurutan
jupyter notebook notebooks/
```

> **Catatan:** Letakkan video latihan di `data/raw/<NamaLatihan>/` (contoh: `data/raw/Squat/video1.mp4`). Sistem akan secara otomatis mengekstraksi pose dan memberikan label via `BiomechanicalValidator`.

---

## 🔬 Referensi Biomekanika

Kriteria validasi gerakan dalam `BiomechanicalValidator` didasarkan **eksklusif** pada tiga publikasi ilmiah berikut:

| Referensi | Kontribusi dalam Sistem |
|---|---|
| Chen et al. (2022) | Threshold kedalaman squat (knee angle ≥ 100°) & alignment tulang belakang deadlift |
| Rao et al. (2023) | Deteksi *knee valgus* (rasio lebar lutut-per-pinggul < 0.85) |
| Ko et al. (2024) | *Elbow ROM* bench press (angle ≤ 85°) & kriteria deadlift |

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

Repositori ini dikembangkan untuk keperluan akademis (Tugas Akhir). Segala bentuk penggunaan ulang harus mencantumkan atribusi yang sesuai kepada peneliti dan institusi.

[![MIT License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

<div align="center">
  <sub>Built with ❤️ for academic research · ITENAS Bandung · 2026</sub>
</div>
