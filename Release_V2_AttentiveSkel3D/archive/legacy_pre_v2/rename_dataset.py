import os
from pathlib import Path

def rename_files_in_folder(folder_path, prefix, source_name):
    """
    Mengubah nama semua file .mp4 di dalam folder menjadi format:
    <prefix>_<jenis_latihan>_<source_name><ID>_rep<ID>.mp4

    Contoh: sekunder_benchpress_kaggle01_rep1.mp4
    """
    # Konversi string path ke objek Path
    p = Path(folder_path)

    # Cek apakah folder ada
    if not p.exists() or not p.is_dir():
        print(f"Error: Folder '{folder_path}' tidak ditemukan!")
        return

    # Ambil jenis latihan dan kelas (Benar/Salah) dari struktur folder
    # Asumsi path: data/raw/BenchPress/Benar
    jenis_latihan = p.parent.name.lower() # Contoh: benchpress
    kelas = p.name.lower() # Contoh: benar atau salah

    # Ambil semua file .mp4 di folder tersebut
    mp4_files = list(p.glob("*.mp4"))

    if not mp4_files:
        print(f"Tidak ada file .mp4 di '{folder_path}'")
        return

    print(f"Ditemukan {len(mp4_files)} file di '{folder_path}'. Memulai proses ganti nama...")

    # Lakukan looping dan ganti nama
    for index, file_path in enumerate(mp4_files, start=1):
        # Format ID (misal: 01, 02, ..., 100)
        formatted_id = f"{index:02d}" 

        # Buat nama baru
        # Format: sekunder_benchpress_benar_kaggle01_rep1.mp4
        # Note: Saya tambahkan nama kelas (benar/salah) agar kamu tidak bingung nanti
        new_filename = f"{prefix}_{jenis_latihan}_{kelas}_{source_name}{formatted_id}_rep{formatted_id}.mp4"
        new_filepath = p / new_filename

        # Peringatan jika nama sudah benar agar tidak ter-rename dua kali
        if file_path.name == new_filename:
            print(f"  [SKIP] File sudah memiliki nama yang benar: {new_filename}")
            continue

        try:
            os.rename(file_path, new_filepath)
            print(f"  Renamed: {file_path.name} -> {new_filename}")
        except FileExistsError:
            print(f"  [ERROR] File {new_filename} sudah ada!")
        except Exception as e:
            print(f"  [ERROR] Gagal rename {file_path.name}: {e}")

    print("Proses selesai!\n")

# === CARA PENGGUNAAN ===
if __name__ == "__main__":
    # GANTI PATH INI SESUAI DENGAN FOLDER DATA KAMU
    # Contoh untuk Bench Press Benar
    folder_target = "data\sekunder\Squat" 

    rename_files_in_folder(
        folder_path=folder_target,
        prefix="sekunder",
        source_name="kaggle"
    )