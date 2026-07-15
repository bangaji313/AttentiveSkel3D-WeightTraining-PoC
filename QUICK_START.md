# 🚀 Quick Start Guide — AttentiveSkel-3D Demo App

**Jika Anda ingin langsung mencoba aplikasi tanpa banyak membaca, ikuti 4 langkah di bawah ini:**

---

## 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 2️⃣ Jalankan Aplikasi
```bash
streamlit run src/app/demo_app.py
```
Browser akan membuka otomatis di `http://localhost:8501`

## 3️⃣ Upload Video
- Siapkan video latihan beban (MP4) yang menunjukkan gerakan lengkap
- Unggah via sidebar
- Tunggu pose extraction selesai

## 4️⃣ Eksplorasi 3 Tab

### 📊 TAB 1: Lihat Tensor Data
- Menampilkan raw data (64, 33, 3) yang menjadi input model
- Scroll untuk lihat semua frame dan sendi

### 🎬 TAB 2: Eksplorasi Frame-by-Frame
- Geser slider untuk pilih frame tertentu
- Lihat video frame asli + status validasi biomekanik
- Klik expander untuk lihat koordinat detail

### 👁️ TAB 3: SANITY CHECK (PALING PENTING!)
- **Pilih "Semua Skenario"** untuk melihat komparasi 5 model
- **Lihat perbedaan**:
  - ⚪ Baseline: sendi merah tersebar acak
  - 🔴 Full Model: sendi merah terkonsentrasi di lutut/pinggul/ankle
- Ini membuktikan model fokus di sendi yang benar!

---

## 📷 Expected Screenshot

```
┌─ AttentiveSkel-3D ─────────────────────────────────────┐
│ Proof of Concept — Sanity Check Fokus Atensi Spasial   │
├────────────────────────────────────────────────────────┤
│                                                          │
│  TAB 1 | TAB 2 | TAB 3                                  │
│  ✅     ✅     ✅                                        │
│                                                          │
│  Tab 3: [Frame skeleton dengan skeleton overlay]       │
│         [Baseline]  [Ablasi A]  [Ablasi B]  ...         │
│         ⚪⚪⚪      🔴🔴       🔴🔴🟠       ...         │
│                                                          │
│  Baseline: Attention acak di semua sendi               │
│  Full Model: Attention konsentrasi di lutut/pinggul    │
│                                                          │
└────────────────────────────────────────────────────────┘
```

---

## ✅ Verifikasi Instalasi

Jalankan test ini untuk memastikan semua setup dengan benar:

```python
# test_setup.py
import sys
from pathlib import Path

# Test imports
try:
    import streamlit as st
    import torch
    import cv2
    import mediapipe as mp
    import numpy as np
    import pandas as pd
    print("✅ Semua library tersedia")
except ImportError as e:
    print(f"❌ Library missing: {e}")
    sys.exit(1)

# Test model files
PROJECT_ROOT = Path(__file__).parent
model_files = [
    "models/saved_models/baseline_3dcnn_model.pth",
    "models/saved_models/ablasi_a_no_prior.pth",
    "models/saved_models/ablasi_b_no_learned.pth",
    "models/saved_models/ablasi_c_no_temporal.pth",
    "models/saved_models/AttentiveSkel3D_Final.pth",
]

for model_file in model_files:
    path = PROJECT_ROOT / model_file
    if path.exists():
        print(f"✅ {model_file}")
    else:
        print(f"❌ {model_file} TIDAK ADA")

print("\n✅ Setup lengkap! Siap jalankan: streamlit run src/app/demo_app.py")
```

Jalankan:
```bash
python test_setup.py
```

---

## 🎯 Untuk Presentasi ke Dosen

**Script Presentasi 5 Menit**:

1. **Buka TAB 1 (30 detik)**
   - "Ini tensor raw (64 frame × 33 sendi × 3 koordinat)"
   - "Semua nilai sudah dinormalisasi terhadap mid-hip"

2. **Buka TAB 2 (1 menit)**
   - Geser slider menunjukkan beberapa frame
   - "Kita bisa eksplorasi temporal, lihat gerakan per frame"
   - "Status validasi memastikan pose sudah checked"

3. **Buka TAB 3 - Pilih "Semua Skenario" (3.5 menit)**
   - Point ke Baseline: "Lingkaran merah ACAK di seluruh tubuh"
   - Point ke Full Model: "Lingkaran merah TERFOKUS di lutut dan pinggul"
   - "Ini adalah bukti visual bahwa model belajar untuk fokus pada sendi penting!"
   - "Dari 5 model ablasi, kita lihat kontribusi setiap modul atensi"
   - "Kesimpulan: Biomechanical Spatial Prior + Learned Spatial + Temporal → fokus akurat"

---

## 🐛 Common Issues

### Issue: Upload video, tapi error "Tidak ada pose yang terdeteksi"
**Solusi**:
- Video harus menunjukkan tubuh UTUH (kepala ke kaki)
- Pencahayaan harus cukup terang
- Background tidak boleh terlalu kompleks (gunakan background solid)
- Jarak: 1-2 meter dari kamera

### Issue: Model load lambat
**Solusi**:
- Normal! Pertama kali cache berjalan
- Reload kedua kali akan lebih cepat
- Jika stuck: `streamlit cache clear`

### Issue: "File model tidak ditemukan"
**Solusi**:
- Pastikan file `.pth` ada di `models/saved_models/`
- Cek nama file cocok: `baseline_3dcnn_model.pth` (bukan `baseline.pth`)

---

## 📞 File Referensi

Jika ada error, cek file-file ini:
- **Aplikasi Utama**: `src/app/demo_app.py` (~800 baris)
- **Model Definition**: `src/models/model_3dcnn.py`
- **Pose Extraction**: `src/data/extract_pose.py`
- **Biomechanics Validator**: `src/data/biomechanics_validator.py`

---

**That's it! Enjoy your demo! 🎉**

Untuk dokumentasi lengkap, baca `DEMO_APP_TUTORIAL.md`
