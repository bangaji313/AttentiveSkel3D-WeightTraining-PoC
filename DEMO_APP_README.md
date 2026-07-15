# 🏋️ AttentiveSkel-3D Web Demo — Panduan Menjalankan

Dokumentasi lengkap untuk menjalankan aplikasi Streamlit web demo **AttentiveSkel-3D: Proof of Concept**.

## 📋 Daftar Isi

1. [Prasyarat](#-prasyarat)
2. [Instalasi & Setup](#-instalasi--setup)
3. [Menjalankan Aplikasi](#-menjalankan-aplikasi)
4. [Fitur Aplikasi](#-fitur-aplikasi)
5. [Struktur Aplikasi](#-struktur-aplikasi)
6. [Troubleshooting](#-troubleshooting)

---

## 🔧 Prasyarat

Sebelum menjalankan aplikasi, pastikan Anda telah:

1. **Python 3.9+** terinstal
2. **Dependencies** sudah diinstal (lihat `requirements.txt` di root project)
3. **Model Files** sudah tersimpan di folder `models/saved_models/`:
   - `baseline_3dcnn_model.pth`
   - `ablasi_a_no_prior.pth`
   - `ablasi_b_no_learned.pth`
   - `ablasi_c_no_temporal.pth`
   - `AttentiveSkel3D_Final.pth`

### Verifikasi Dependencies

Pastikan semua dependencies dari `requirements.txt` sudah terinstal:

```bash
pip install -r requirements.txt
```

Dependencies utama:
- `streamlit==1.58.0` — framework web app
- `torch==2.5.1` dan `torchvision==0.20.1` — PyTorch untuk model
- `mediapipe==0.10.14` — untuk pose estimation
- `opencv-python==4.13.0.92` — video processing
- `matplotlib==3.10.8` — visualisasi chart
- `numpy`, `pandas` — data processing

---

## 📦 Instalasi & Setup

### 1. Aktivasi Virtual Environment (Opsional tapi Recommended)

**Pada Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Pada Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instal Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verifikasi Struktur Folder

Pastikan struktur folder project seperti berikut:

```
AttentiveSkel3D-WeightTraining-PoC/
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   └── demo_app.py          ← File aplikasi Streamlit (BARU)
│   ├── data/
│   │   ├── extract_pose.py
│   │   ├── dataset.py
│   │   ├── build_dataset.py
│   │   └── biomechanics_validator.py
│   └── models/
│       └── model_3dcnn.py
├── models/
│   └── saved_models/
│       ├── baseline_3dcnn_model.pth
│       ├── ablasi_a_no_prior.pth
│       ├── ablasi_b_no_learned.pth
│       ├── ablasi_c_no_temporal.pth
│       └── AttentiveSkel3D_Final.pth
├── requirements.txt
└── README.md
```

---

## 🚀 Menjalankan Aplikasi

### Cara 1: Command Line (Recommended)

**Pada Windows (Command Prompt atau PowerShell):**
```bash
cd g:\data-aji\KULIAH\Semester 8\IFB500-TUGAS_AKHIR-AA\AttentiveSkel3D-WeightTraining-PoC
streamlit run src/app/demo_app.py
```

**Pada Linux/macOS:**
```bash
cd /path/to/AttentiveSkel3D-WeightTraining-PoC
streamlit run src/app/demo_app.py
```

### Cara 2: Dari VS Code

1. Buka terminal di VS Code (`Ctrl+```)
2. Jalankan command:
   ```bash
   streamlit run src/app/demo_app.py
   ```

### Output yang Diharapkan

Setelah menjalankan command, Anda akan melihat output seperti:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  Edit source code and save to rerun instantly.
```

Browser akan otomatis membuka di `http://localhost:8501`. Jika tidak, buka URL tersebut secara manual.

---

## 🎯 Fitur Aplikasi

### 1. **Konfigurasi Halaman & UI**
- ✅ Wide layout (`st.set_page_config(layout="wide")`)
- ✅ Custom CSS untuk UI profesional dan rapi
- ✅ Video player dengan ukuran maksimal 450px (centered)

### 2. **Sidebar — Kontrol Input**
- ✅ **File Uploader**: Upload video latihan beban format `.mp4`
- ✅ **Model Selection Dropdown**: 6 pilihan skenario
  - Semua Skenario (Komparasi 5 Model)
  - Baseline (3D-CNN Murni)
  - Ablasi A (Tanpa Prior)
  - Ablasi B (Tanpa Learned Spatial)
  - Ablasi C (Tanpa Temporal)
  - Full AttentiveSkel-3D

### 3. **Logika Pemrosesan**
- ✅ Ekstraksi pose menggunakan `PoseExtractor` (MediaPipe BlazePose)
- ✅ Output tensor: (64, 33, 3) — 64 frame, 33 sendi, 3 koordinat (x, y, z)
- ✅ Prediksi menggunakan bobot model yang sesuai
- ✅ Model caching dengan `@st.cache_resource` untuk efisiensi

### 4. **Output Visual & Hasil**

#### Single Scenario:
- Tampilkan hasil prediksi dalam kartu hasil (result card) dengan styling gradient
- Label gerakan: "Benar" atau "Salah"
- Confidence Score dengan format persentase

#### All Scenarios (Komparasi 5 Model):
- Layout 5 kolom untuk menampilkan hasil semua model secara bersamaan
- Setiap kolom menampilkan:
  - Nama skenario model
  - Label prediksi
  - Confidence score

### 5. **Grafik Pendukung (Bukti Interpretabilitas)**
- ✅ **Bar Chart**: Top 5 Bobot Atensi Sendi
- ✅ Sumbu X: Nama sendi (Bahu, Siku, Pinggul, Lutut, dll)
- ✅ Sumbu Y: Probabilitas atensi (0.0 - 1.0)
- ✅ Membuktikan model fokus pada sendi yang relevan secara biomekanis

---

## 📁 Struktur Aplikasi

### Organisasi Kode

```python
# 1. Imports dan Konfigurasi
#    - Import modul eksternal (streamlit, torch, cv2, mediapipe, dll)
#    - Setup path untuk import modul lokal
#    - Konfigurasi Streamlit page config
#    - Custom CSS untuk UI

# 2. Konstanta Global
#    - MODEL_PATHS: mapping nama model ke file .pth
#    - SCENARIO_NAMES: mapping skenario ke display name
#    - SCENARIO_CONFIGS: konfigurasi untuk setiap skenario model
#    - LANDMARK_NAMES: nama 33 sendi MediaPipe
#    - CRITICAL_JOINTS: sendi penting untuk visualisasi

# 3. Fungsi Utilitas (dengan @st.cache_resource untuk caching)
#    - load_pose_extractor(): Load PoseExtractor
#    - load_model(model_key): Load model PyTorch
#    - extract_pose_from_video(video_file): Ekstraksi pose dari video
#    - normalize_pose(pose_array): Normalisasi pose
#    - pad_or_truncate_pose(pose_tensor): Pad/truncate ke 64 frame
#    - predict_single_model(pose_tensor, model): Prediksi single model
#    - extract_attention_weights(model, pose_tensor): Extract attention weights
#    - visualize_skeleton_with_heatmap(...): Visualisasi skeleton dengan heatmap
#    - create_attention_bar_chart(attention_weights): Buat bar chart

# 4. Main Application Logic
#    - main(): Fungsi utama Streamlit
#    - Header dan judul
#    - Sidebar kontrol input
#    - Processing logic (single vs all scenarios)
#    - Display hasil dengan styling CSS
```

### File-File Terkait

```
src/
├── app/
│   ├── __init__.py                  # Package initialization
│   └── demo_app.py                  # Aplikasi Streamlit utama (file baru)
├── data/
│   ├── extract_pose.py              # PoseExtractor class
│   ├── dataset.py                   # Dataset dan DataLoader
│   └── biomechanics_validator.py    # Validasi biomekanik
└── models/
    └── model_3dcnn.py               # AttentiveSkel3D model architecture
```

---

## 🐛 Troubleshooting

### ❌ Error: "ModuleNotFoundError: No module named 'streamlit'"

**Solusi:**
```bash
pip install streamlit==1.58.0
```

### ❌ Error: "ModuleNotFoundError: No module named 'data'"

**Solusi:**
Pastikan Anda menjalankan aplikasi dari directory project root:
```bash
cd g:\data-aji\KULIAH\Semester 8\IFB500-TUGAS_AKHIR-AA\AttentiveSkel3D-WeightTraining-PoC
streamlit run src/app/demo_app.py
```

### ❌ Error: "File model '[model_key]' tidak ditemukan"

**Solusi:**
Verifikasi bahwa file model sudah tersimpan di `models/saved_models/`:
```bash
dir models/saved_models/
```

Pastikan file `.pth` ada:
- `baseline_3dcnn_model.pth`
- `ablasi_a_no_prior.pth`
- `ablasi_b_no_learned.pth`
- `ablasi_c_no_temporal.pth`
- `AttentiveSkel3D_Final.pth`

### ❌ Error: "CUDA out of memory"

**Solusi:**
Model dimuat ke CPU secara default. Jika masih ada error, restart browser atau clear cache:
```bash
# Restart Streamlit dengan cache dinonaktifkan
streamlit run src/app/demo_app.py --logger.level=debug
```

### ❌ Video tidak ter-upload atau error saat ekstraksi pose

**Solusi:**
1. Pastikan video format MP4
2. Pastikan video tidak terlalu besar (< 200MB)
3. Pastikan OpenCV dan MediaPipe terinstal:
   ```bash
   pip install opencv-python==4.13.0.92 mediapipe==0.10.14
   ```

### ❌ Chart tidak tampil atau error saat visualisasi

**Solusi:**
Pastikan matplotlib terinstal:
```bash
pip install matplotlib==3.10.8
```

---

## 📊 Contoh Workflow Penggunaan

### Scenario 1: Single Model Prediction

1. Jalankan aplikasi: `streamlit run src/app/demo_app.py`
2. Di sidebar, unggah file video MP4
3. Pilih skenario "Full AttentiveSkel-3D"
4. Aplikasi akan:
   - Mengekstraksi pose skeleton dari video
   - Melakukan prediksi dengan model Full
   - Menampilkan hasil prediksi (Label + Confidence)
   - Menampilkan bar chart Top 5 Attention Weights

### Scenario 2: Komparasi 5 Model

1. Jalankan aplikasi: `streamlit run src/app/demo_app.py`
2. Di sidebar, unggah file video MP4
3. Pilih skenario "Semua Skenario (Komparasi 5 Model)"
4. Aplikasi akan:
   - Mengekstraksi pose skeleton dari video
   - Melakukan prediksi dengan ke-5 model secara bersamaan
   - Menampilkan hasil dalam layout 5 kolom
   - Menampilkan bar chart Top 5 Attention Weights dari Full model

---

## 💡 Tips & Best Practices

### 1. Performance Optimization
- Model dan PoseExtractor di-cache dengan `@st.cache_resource`
- Inference dilakukan di CPU (lebih stabil untuk browser)
- Video temporary file dihapus setelah pemrosesan

### 2. Video Input
- Format optimal: MP4 dengan codec H.264
- Resolusi: 640x480 atau lebih tinggi
- Durasi: 10-60 detik untuk hasil optimal

### 3. Model Selection
- **Full Model** (AttentiveSkel-3D): Rekomendasi untuk akurasi terbaik
- **Baseline**: Untuk baseline comparison (tanpa attention)
- **Ablation Models**: Untuk bukti kontribusi setiap modul attention

### 4. Interpretabilitas
- Bar chart Top 5 Attention Weights membuktikan model fokus pada sendi penting
- Model learned spatial/temporal attention untuk konteks gerak dinamis

---

## 📞 Support & Documentation

Untuk informasi lebih lanjut tentang:

- **AttentiveSkel-3D Model**: Lihat `src/models/model_3dcnn.py`
- **Pose Extraction**: Lihat `src/data/extract_pose.py`
- **Biomechanics Validation**: Lihat `src/data/biomechanics_validator.py`
- **Dataset Management**: Lihat `src/data/dataset.py`

---

## 📚 Referensi Akademis

1. Chen, K.-Y., et al. (2022). "Fitness Movement Types and Completeness Detection Using Transfer-Learning-Based Deep Neural Network."
2. Rao, P., et al. (2023). "Real-time Posture Correction of Squat Exercise: A Deep Learning Approach for Performance Analysis."
3. Ko, Y.-M., et al. (2024). "Real-Time AI Posture Correction for Powerlifting Exercises Using YOLOv5 and MediaPipe."

---

**Last Updated:** 2026-01-14  
**Version:** 1.0  
**Status:** ✅ Production Ready
