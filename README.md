# AttentiveSkel3D-WeightTraining-PoC

> **Tugas Akhir / Skripsi**
> *A Proof of Concept for Enhancing Weight Training Form Error Detection Using 3D-CNN and Biomechanical Attention Mechanism*

---

## Deskripsi

Repositori ini merupakan implementasi *Proof of Concept* (PoC) dari penelitian Tugas Akhir yang bertujuan untuk mendeteksi kesalahan gerakan latihan beban — mencakup **Squat**, **Deadlift**, dan **Bench Press** — secara otomatis menggunakan pendekatan *deep learning*.

Model yang diusulkan, **AttentiveSkel-3D**, menggabungkan arsitektur **3D Convolutional Neural Network (3D-CNN)** ringan dengan mekanisme **Biomechanical Attention** berbasis sendi. Model menerima urutan *pose skeleton* 3D yang diekstraksi menggunakan **MediaPipe BlazePose** sebagai input, kemudian mempelajari representasi spatio-temporal gerakan untuk mengklasifikasikan apakah eksekusi gerakan dilakukan dengan **benar** atau **salah**.

Penelitian ini diharapkan dapat menjadi fondasi pengembangan sistem *real-time feedback* yang dapat membantu atlet maupun individu dalam melakukan latihan beban secara aman dan efisien tanpa bergantung pada perangkat sensor khusus.

---

## Teknologi Utama

| Komponen | Library / Framework |
|---|---|
| Deep Learning | PyTorch, TorchVision |
| Pose Estimation | MediaPipe BlazePose |
| Computer Vision | OpenCV |
| Analisis Data | NumPy, Pandas, SciPy |
| Visualisasi | Matplotlib, Seaborn |
| Eksperimen | Jupyter Notebook |

---

## Struktur Folder

```
AttentiveSkel3D-WeightTraining-PoC/
│
├── data/
│   ├── raw/              # Video mentah dataset (Primer & Sekunder), diabaikan oleh Git
│   └── processed/        # File .npy hasil ekstraksi pose skeleton, diabaikan oleh Git
│
├── notebooks/            # Jupyter Notebook untuk eksperimen, eksplorasi, dan visualisasi
│
├── src/
│   ├── data/             # Modul pemrosesan data (ekstraksi pose, dataset loader)
│   └── models/           # Modul arsitektur model (3D-CNN, Biomechanical Attention)
│
├── models/
│   └── saved_models/     # Bobot model terlatih (.pth), diabaikan oleh Git
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Instalasi

```bash
# 1. Clone repositori ini
git clone <url-repositori>
cd AttentiveSkel3D-WeightTraining-PoC

# 2. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Install semua dependensi
pip install -r requirements.txt
```

---

## Alur Kerja Penelitian

```
Video Mentah (data/raw/)
        │
        ▼
[ MediaPipe BlazePose ]  ←  src/data/extract_pose.py
        │  33 Keypoints × (x, y, z) per frame
        ▼
Sekuens Skeleton 3D (data/processed/*.npy)
        │
        ▼
[ AttentiveSkel-3D Model ]  ←  src/models/
  ├── 3D-CNN Encoder (fitur spatio-temporal)
  └── Biomechanical Attention Layer
        │
        ▼
Klasifikasi: Gerakan Benar / Salah
```

---

## Lisensi

Repositori ini dikembangkan untuk keperluan akademis (Tugas Akhir). Segala bentuk penggunaan ulang harus mencantumkan atribusi yang sesuai.
