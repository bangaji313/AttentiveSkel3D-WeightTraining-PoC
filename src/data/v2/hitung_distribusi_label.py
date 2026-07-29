import pandas as pd
import numpy as np
from pathlib import Path

def hitung_distribusi():
    project_root = Path(__file__).resolve().parents[3]
    manifest_path = project_root / "data" / "processed" / "manifest_v2.csv"
    
    if not manifest_path.exists():
        print(f"Error: Manifest tidak ditemukan di {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    
    # Inisialisasi dictionary untuk menyimpan hasil per kelas
    # Format: { 'Kelas': {'Benar': 0, 'Salah': 0, 'Total': 0} }
    distribusi = {
        'BenchPress': {'Benar': 0, 'Salah': 0, 'Total': 0},
        'Squat': {'Benar': 0, 'Salah': 0, 'Total': 0},
        'Deadlift': {'Benar': 0, 'Salah': 0, 'Total': 0},
        'GLOBAL (KESELURUHAN)': {'Benar': 0, 'Salah': 0, 'Total': 0}
    }

    print("Sedang menghitung total dataset (harap tunggu sebentar)...\n")

    for _, row in df.iterrows():
        label_path = project_root / row['labels_path']
        file_path_str = str(row['file_path']).lower()
        
        # Deteksi kelas
        if 'bench' in file_path_str:
            kelas = 'BenchPress'
        elif 'squat' in file_path_str:
            kelas = 'Squat'
        elif 'deadlift' in file_path_str:
            kelas = 'Deadlift'
        else:
            continue

        try:
            # Baca file NPY label
            labels = np.load(label_path)
            
            # Hitung jumlah 0 (Benar) dan 1 (Salah)
            n_benar = (labels == 0).sum()
            n_salah = (labels == 1).sum()
            n_total = len(labels) # Harusnya selalu 64
            
            # Tambahkan ke kelas spesifik
            distribusi[kelas]['Benar'] += n_benar
            distribusi[kelas]['Salah'] += n_salah
            distribusi[kelas]['Total'] += n_total
            
            # Tambahkan ke GLOBAL
            distribusi['GLOBAL (KESELURUHAN)']['Benar'] += n_benar
            distribusi['GLOBAL (KESELURUHAN)']['Salah'] += n_salah
            distribusi['GLOBAL (KESELURUHAN)']['Total'] += n_total
            
        except Exception as e:
            print(f"Gagal membaca {label_path}: {e}")

    # ============================================================
    # Tampilkan hasil perhitungan ke terminal
    # ============================================================
    print("=" * 65)
    print("DISTRIBUSI KELAS DATASET (PER-FRAME) - AttentiveSkel-3D V2")
    print("=" * 65)

    for kelas, data in distribusi.items():
        if data['Total'] == 0:
            continue
            
        benar = data['Benar']
        salah = data['Salah']
        total = data['Total']
        
        pct_benar = (benar / total) * 100
        pct_salah = (salah / total) * 100
        
        if kelas == 'GLOBAL (KESELURUHAN)':
            print("-" * 65)
            
        print(f"[{kelas}]")
        print(f"  Total Frame : {total:,} frames")
        print(f"  Benar (0)   : {benar:,} frames ({pct_benar:.2f}%)")
        print(f"  Salah (1)   : {salah:,} frames ({pct_salah:.2f}%)")
        print()

if __name__ == "__main__":
    hitung_distribusi()
