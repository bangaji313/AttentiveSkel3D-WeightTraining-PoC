# 🏋️ AttentiveSkel-3D Proof of Concept — Demo App Tutorial

**Tujuan Utama**: Demonstrasi Sanity Check Fokus Atensi Spasial Model AI untuk dosen penguji.

---

## 📋 Daftar Isi

1. [Persiapan](#persiapan)
2. [Cara Menjalankan](#cara-menjalankan)
3. [Penjelasan 3 Tab](#penjelasan-3-tab)
4. [Interpretasi Hasil](#interpretasi-hasil)
5. [Troubleshooting](#troubleshooting)

---

## ⚙️ Persiapan

### Persyaratan
- Python 3.9+
- Dependencies sudah terdaftar di `requirements.txt`:
  - `streamlit>=1.28.0`
  - `opencv-python-headless>=4.8.0`
  - `torch>=2.0.0`
  - `mediapipe>=0.10.0`
  - `numpy>=1.24.0`
  - `pandas>=2.0.0`
  - `matplotlib>=3.7.0`

### Instalasi Dependencies
```bash
pip install -r requirements.txt
```

### File Model yang Dibutuhkan
Pastikan file model berikut ada di `models/saved_models/`:
- `baseline_3dcnn_model.pth` — Model baseline (3D-CNN tanpa atensi)
- `ablasi_a_no_prior.pth` — Ablasi A (tanpa spatial prior)
- `ablasi_b_no_learned.pth` — Ablasi B (tanpa learned spatial attention)
- `ablasi_c_no_temporal.pth` — Ablasi C (tanpa temporal attention)
- `AttentiveSkel3D_Final.pth` — Full model dengan semua modul atensi

---

## 🚀 Cara Menjalankan

### Opsi 1: Jalankan dari Command Line
```bash
streamlit run src/app/demo_app.py
```

### Opsi 2: Jalankan Langsung di Terminal VS Code
```bash
cd AttentiveSkel3D-WeightTraining-PoC
streamlit run src/app/demo_app.py
```

Aplikasi akan membuka di browser pada `http://localhost:8501`

---

## 📊 Penjelasan 3 Tab

### TAB 1: 📊 Fase 1 — Wujud Data Tensor

**Fokus**: Menampilkan **tensor raw (64, 33, 3)** dalam format tabel.

**Apa yang ditampilkan**:
- **Dataframe Tensor**: 64 baris (frame) × 99 kolom (33 sendi × 3 koordinat)
- **Nilai**: Float32 ternormalisasi dalam range [-1, 1]
- **Statistik**: Min, Max, Mean per dimensi (X, Y, Z)

**Mengapa Penting**:
- Buktikan kepada dosen penguji bahwa input model adalah **data numerik nyata**
- Transparansi: Tidak ada "black box" dalam preprocessing
- Menunjukkan bahwa pose sudah dinormalisasi terhadap mid-hip

**Interpretasi**:
```
Frame 0:  Hidung (X)  Hidung (Y)  Hidung (Z)  Mata Kiri Dalam (X)  ...
          -0.1234     0.5678      -0.0234     -0.2567              ...
Frame 1:  -0.1111     0.5789      -0.0123     -0.2456              ...
...
```

---

### TAB 2: 🎬 Fase 2 — Lokalisasi Waktu & Validasi

**Fokus**: **Eksplorasi temporal** frame-by-frame dengan validasi biomekanik.

**Apa yang ditampilkan**:
- **Video Frame Asli**: Frame dari indeks 0-63 sesuai slider
- **Informasi Frame**:
  - Indeks frame dan waktu (dalam detik)
  - Status validasi biomekanik (✅ Valid / ⚠️ Tidak Valid)
  - Min/Max/Mean koordinat frame tersebut
- **Expander "Koordinat Sendi"**: Tabel detail 33 landmark (x, y, z)

**Mengapa Penting**:
- Memungkinkan dosen penguji **melihat gerakan asli** dan memverifikasi pose extraction
- Status validasi membuktikan bahwa data telah divalidasi terhadap kriteria biomekanik
- Eksplorasi interaktif: slide untuk melihat berbagai frame tanpa harus proses ulang

**Cara Menggunakan**:
1. Geser slider "Pilih Frame (0-63)" untuk memilih frame tertentu
2. Lihat video frame asli di sebelah kiri
3. Cek status validasi di sebelah kanan
4. Klik expander untuk melihat koordinat detail 33 sendi

---

### TAB 3: 👁️ Fase 3 — Sanity Check Mata AI & Komparasi (PALING PENTING)

**Fokus**: **Membuktikan bahwa model fokus pada sendi yang benar**.

**Apa yang ditampilkan** (tergantung pilihan di sidebar):

#### A. Jika Pilih "Semua Skenario (Komparasi 5 Model)"

**Layout**: 5 kolom berdampingan
- **Kolom 1**: Baseline (kontrol: fokus acak)
- **Kolom 2-4**: Ablasi A, B, C
- **Kolom 5**: Full Model (fokus akurat)

**Setiap Kolom Menampilkan**:
1. **Skeleton Overlay**: Frame tengah (frame 32) dengan skeleton + heatmap atensi
2. **Prediksi**: Label (Benar/Salah) + confidence score
3. **Warna Sendi**:
   - 🔴 **Merah Menyala** = Atensi Tinggi (> 0.6) — AI fokus di sini
   - 🟠 **Orange/Kuning** = Atensi Sedang (0.3-0.6)
   - ⚪ **Abu-abu** = Atensi Rendah (< 0.3)

**Bar Chart**: Top 5 Bobot Atensi Sendi Full Model

**Interpretasi Hasil**:
```
BASELINE: Lingkaran merah tersebar ACAK di seluruh tubuh
          → Membuktikan bahwa model tanpa atensi fokus tidak konsisten

ABLASI A/B/C: Lingkaran merah mulai terkonsentrasi di beberapa sendi
             → Pembuktian incremental bahwa setiap modul atensi berkontribusi

FULL MODEL: Lingkaran merah TERKONSENTRASI di lutut, pinggul, ankle
           → Pembuktian final bahwa model fokus pada sendi penting untuk gerakan beban
```

#### B. Jika Pilih "Single Model" (Baseline/Ablasi A/B/C/Full)

**Layout**: 2 kolom

**Kolom Kiri**:
- Skeleton dengan heatmap atensi (frame 32)
- Visualisasi langsung fokus model

**Kolom Kanan**:
- Hasil prediksi (label + confidence)
- Konfigurasi model (status 3 modul atensi: ✅/❌)

**Bawah**:
- Bar chart Top 5 Bobot Atensi Sendi

---

## 💡 Interpretasi Hasil

### Skenario 1: Squat (Latihan Kaki)
**Expected Attention Pattern**:
- 🔴 Merah tinggi di: **Lutut Kiri/Kanan, Pinggul Kiri/Kanan, Ankle Kiri/Kanan**
- ⚪ Abu-abu di: Kepala, lengan (tidak penting untuk squat)

**Indikator Kualitas**:
- ✅ **BAIK**: Baseline fokus acak, Full Model fokus di kaki → model belajar dengan benar
- ❌ **BURUK**: Full Model fokus di kepala saat squat → model belum konvergen

### Skenario 2: Bench Press (Latihan Dada)
**Expected Attention Pattern**:
- 🔴 Merah tinggi di: **Bahu, Siku, Pergelangan Tangan**
- ⚪ Abu-abu di: Kepala, kaki (tidak penting untuk bench press)

### Skenario 3: Deadlift (Latihan Punggung)
**Expected Attention Pattern**:
- 🔴 Merah tinggi di: **Pinggul, Lutut, Pergelangan Kaki**
- ⚪ Abu-abu di: Kepala, lengan atas

---

## 📝 Cara Menceritakan ke Dosen Penguji

### Presentasi Ideal:
1. **Buka TAB 1**: "Ini adalah tensor raw (64×33×3) yang menjadi input model..."
2. **Buka TAB 2**: "Kita bisa eksplorasi frame per frame untuk verifikasi pose extraction..."
3. **Buka TAB 3 - PALING PENTING**:
   - "Pilih 'Semua Skenario' untuk melihat komparasi 5 model"
   - "Lihat Baseline: lingkaran merah tersebar acak → model tanpa atensi fokus tidak konsisten"
   - "Lihat Full Model: lingkaran merah terkonsentrasi di lutut/pinggul → model belajar fokus di sendi penting!"
   - "Ini membuktikan bahwa modul atensi spatial bekerja dengan benar"

---

## 🔧 Troubleshooting

### Error: "❌ Gagal mengimport modul: ..."
**Solusi**:
- Pastikan `SRC_DIR` di-set dengan benar di `demo_app.py`
- Pastikan file `extract_pose.py`, `model_3dcnn.py`, `biomechanics_validator.py` ada di `src/`
- Run: `python -c "import sys; print(sys.path)"`

### Error: "❌ Tidak dapat memuat model untuk skenario..."
**Solusi**:
- Pastikan file `.pth` ada di `models/saved_models/`
- Pastikan nama file cocok dengan konstanta `MODEL_PATHS` di `demo_app.py`
- Cek bahwa file bukan corrupt: `python -c "import torch; torch.load('models/saved_models/AttentiveSkel3D_Final.pth')"`

### Error: "❌ Tidak ada pose yang terdeteksi dalam video"
**Solusi**:
- Video harus berisi tubuh lengkap yang terlihat jelas
- Pencahayaan harus cukup terang
- Jarak kamera tidak terlalu jauh (pose harus terdeteksi oleh MediaPipe)
- Format video: MP4 dengan codec H.264

### Aplikasi Lambat Saat Load Model
**Solusi**:
- Ini normal pertama kali (caching berjalan)
- Streamlit cache akan simpan model di memori, reload lebih cepat
- Jika masih lambat, run: `streamlit cache clear`

### Error: "Frame Index Out of Range"
**Solusi**:
- Video terlalu pendek (< 64 frame)
- Aplikasi akan auto-pad frame dengan replikasi frame terakhir
- Ini normal, silakan lanjutkan

---

## 📚 Referensi Teknis

### Struktur Pose Tensor
```
Shape: (64, 33, 3)
- 64 frames dari video
- 33 landmarks: BlazePose dari MediaPipe
  (kepala, mata, telinga, bahu, siku, pergelangan tangan, pinggul, lutut, ankle)
- 3 koordinat: (x, y, z) ternormalisasi dalam range [-1, 1]
```

### Model Architecture
- **Baseline**: 3D-CNN tanpa modul atensi (kontrol)
- **Ablasi A**: 3D-CNN + Learned Spatial Attention + Temporal Attention
- **Ablasi B**: 3D-CNN + Biomechanical Spatial Prior + Temporal Attention
- **Ablasi C**: 3D-CNN + Biomechanical Spatial Prior + Learned Spatial Attention
- **Full**: 3D-CNN + Biomechanical Spatial Prior + Learned Spatial Attention + Temporal Attention

### Attention Weights Extraction
- Attention weights diambil dari `biomechanical_spatial_prior` parameter
- Normalized ke range [0, 1] untuk visualisasi
- Digunakan untuk color-coding skeleton sendi

---

## ✅ Checklist Sebelum Demo

- [ ] Semua 5 file model ada di `models/saved_models/`
- [ ] Python dependencies ter-install: `pip install -r requirements.txt`
- [ ] Video test (squat, bench press, atau deadlift) sudah disiapkan
- [ ] Run test: `streamlit run src/app/demo_app.py` → app membuka di browser
- [ ] Coba upload video test → pose extraction berhasil
- [ ] Coba ketiga tab → semuanya berjalan lancar
- [ ] Siapkan script presentasi untuk dosen penguji

---

## 📞 Kontak & Support

Jika ada pertanyaan teknis mengenai aplikasi ini, silakan cek:
- Kode di `src/app/demo_app.py`
- Dokumentasi model di `src/models/model_3dcnn.py`
- Pose extraction di `src/data/extract_pose.py`

---

**Last Updated**: 2025
**App Version**: 1.0 (3-Tab Sanity Check)
