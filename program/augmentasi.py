"""
augmentasi.py
Melakukan augmentasi citra pada dataset agar mencapai target 100 foto per kelas (total 300 foto).
Teknik augmentasi yang digunakan:
- Flip horizontal & vertikal
- Rotasi (90, 180, 270 derajat)
- Penyesuaian kontras dan kecerahan minor

Jalankan dari root proyek:
    python program/augmentasi.py
"""
import cv2
import random
from pathlib import Path

DATASET_DIR = Path("dataset")
KELAS_LIST = ["ringan", "sedang", "berat"]
TARGET_PER_KELAS = 100
EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def buat_augmentasi(img):
    """Menghasilkan variasi citra dengan transformasi geometrik dan fotometrik."""
    variasi = []
    
    # 1. Flip horizontal & vertikal
    variasi.append(("fliph", cv2.flip(img, 1)))
    variasi.append(("flipv", cv2.flip(img, 0)))
    variasi.append(("fliphv", cv2.flip(img, -1)))
    
    # 2. Rotasi
    variasi.append(("rot90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)))
    variasi.append(("rot180", cv2.rotate(img, cv2.ROTATE_180)))
    variasi.append(("rot270", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)))
    
    # 3. Kecerahan & Kontras
    variasi.append(("bright_plus", cv2.convertScaleAbs(img, alpha=1.1, beta=15)))
    variasi.append(("bright_minus", cv2.convertScaleAbs(img, alpha=0.9, beta=-15)))
    variasi.append(("contrast_plus", cv2.convertScaleAbs(img, alpha=1.2, beta=0)))
    
    return variasi

def main():
    print("=" * 60)
    print(f"MEMULAI DATA AUGMENTATION (Target: {TARGET_PER_KELAS} foto/kelas)")
    print("=" * 60)

    total_baru = 0

    for kelas in KELAS_LIST:
        folder = DATASET_DIR / kelas
        if not folder.exists():
            continue

        # Ambil gambar asli (bukan hasil augmentasi sebelumnya)
        semua_file = [f for f in folder.iterdir() if f.suffix.lower() in EKSTENSI]
        file_asli = [f for f in semua_file if not f.name.startswith("aug_")]
        
        jumlah_sekarang = len(semua_file)
        kekurangan = TARGET_PER_KELAS - jumlah_sekarang

        print(f"\nKelas '{kelas}':")
        print(f"  - Foto saat ini : {jumlah_sekarang}")
        print(f"  - Target        : {TARGET_PER_KELAS}")

        if kekurangan <= 0:
            print("  [OK] Sudah memenuhi atau melebihi target.")
            continue

        print(f"  -> Perlu membuat {kekurangan} foto augmentasi...")
        dibuat = 0

        # Looping sampai jumlah kekurangan terpenuhi
        while dibuat < kekurangan:
            # Acak urutan foto asli agar variasi merata
            shuffled_asli = list(file_asli)
            random.shuffle(shuffled_asli)

            for f_asli in shuffled_asli:
                if dibuat >= kekurangan:
                    break

                img = cv2.imread(str(f_asli))
                if img is None:
                    continue

                daftar_variasi = buat_augmentasi(img)
                # Acak variasi yang dipilih
                random.shuffle(daftar_variasi)

                for tag, img_aug in daftar_variasi:
                    if dibuat >= kekurangan:
                        break

                    nama_baru = f"aug_{tag}_{dibuat+1}_{f_asli.name}"
                    tujuan = folder / nama_baru

                    if not tujuan.exists():
                        cv2.imwrite(str(tujuan), img_aug)
                        dibuat += 1
                        total_baru += 1

        print(f"  [OK] Berhasil menambahkan {dibuat} foto baru ke '{kelas}'.")

    print("\n" + "=" * 60)
    print(f"AUGMENTASI SELESAI! Total {total_baru} foto baru ditambahkan.")
    print("=" * 60)

if __name__ == "__main__":
    main()
