"""
sortir_dataset.py
Membaca anotasi kerusakan dari dataset Broken Interfaces,
menghitung persentase area kerusakan (ground truth dari overlay peneliti),
lalu menyortir citra layar ke folder dataset/ringan, sedang, dan berat.

Jalankan dari root proyek:
    python program/sortir_dataset.py
"""
import os
import glob
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Path folder
BASE_DIR = Path("dataset-asli/broken-interfaces-master")
OVERLAY_DIR = BASE_DIR / "phone_damage_overlay"
CROPPED_DIR = BASE_DIR / "phone_cropped_display"
ORIGINAL_DIR = BASE_DIR / "phone_originals"
DATASET_DIR = Path("dataset")

# Ambang batas persentase area retakan/kerusakan:
# Crack adalah garis tipis, sehingga rasio piksel kerusakan:
# - Ringan : < 2.0%
# - Sedang : 2.0% - 7.0%
# - Berat  : > 7.0%
TH_RINGAN = 2.0
TH_SEDANG = 7.0

def tentukan_kelas(rasio: float) -> str:
    if rasio < TH_RINGAN:
        return "ringan"
    elif rasio <= TH_SEDANG:
        return "sedang"
    else:
        return "berat"

def main():
    if not OVERLAY_DIR.exists() or not CROPPED_DIR.exists():
        print("[!] Folder dataset belum lengkap di dataset-asli/broken-interfaces-master")
        return

    overlays = sorted(glob.glob(str(OVERLAY_DIR / "*.png")))
    if not overlays:
        print("[!] Tidak ada file di folder phone_damage_overlay")
        return

    print(f"Menganalisis {len(overlays)} gambar dan menghitung rasio kerusakan...")
    print("-" * 65)
    print(f"{'Nama File':20s} {'Piksel Rusak':>12s} {'Rasio (%)':>10s} {'Kelas':>10s}")
    print("-" * 65)

    data_log = []
    hasil_kelas = {"ringan": 0, "sedang": 0, "berat": 0}

    for p in overlays:
        nama = Path(p).name
        disp_path = CROPPED_DIR / nama

        if not disp_path.exists():
            continue

        img_overlay = cv2.imread(p)
        img_disp = cv2.imread(str(disp_path))

        if img_overlay is None or img_disp is None:
            continue

        # Hitung selisih piksel antara overlay dan display asli
        diff = cv2.absdiff(img_overlay, img_disp)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Anotasi warna ungu (#756bb1) memiliki perbedaan nilai yang jelas
        damage_pixels = int(np.count_nonzero(gray_diff > 20))
        total_pixels = int(img_disp.shape[0] * img_disp.shape[1])
        rasio = round((damage_pixels / total_pixels) * 100, 3)

        kelas = tentukan_kelas(rasio)
        hasil_kelas[kelas] += 1

        # Copy gambar cropped display ke folder dataset kelas masing-masing
        target_folder = DATASET_DIR / kelas
        target_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(disp_path, target_folder / nama)

        print(f"{nama:20s} {damage_pixels:12,d} {rasio:9.2f}% {kelas:>10s}")

        data_log.append({
            "filename": nama,
            "damage_pixels": damage_pixels,
            "total_pixels": total_pixels,
            "damage_ratio_pct": rasio,
            "kelas": kelas
        })

    # Simpan catatan log ke CSV di folder hasil/
    hasil_dir = Path("hasil")
    hasil_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(data_log)
    df.to_csv(hasil_dir / "label_dataset_summary.csv", index=False)

    print("=" * 65)
    print("HASIL SORTIR DATASET OTOMATIS:")
    print(f"  - Ringan : {hasil_kelas['ringan']} gambar")
    print(f"  - Sedang : {hasil_kelas['sedang']} gambar")
    print(f"  - Berat  : {hasil_kelas['berat']} gambar")
    print(f"  Total    : {sum(hasil_kelas.values())} gambar")
    print("=" * 65)
    print(f"[OK] Label dan rasio disimpan ke {hasil_dir / 'label_dataset_summary.csv'}")

if __name__ == "__main__":
    main()
