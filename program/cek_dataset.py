"""
cek_dataset.py
Menghitung jumlah gambar per kelas dan menampilkan ringkasan dataset.
Jalankan dari root proyek: python program/cek_dataset.py
"""
from pathlib import Path

DATASET = Path("dataset")
EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KELAS = ["ringan", "sedang", "berat"]

total = 0
print("=" * 40)
print("  Pemeriksaan Dataset")
print("=" * 40)

for kelas in KELAS:
    folder = DATASET / kelas
    if not folder.exists():
        print(f"  {kelas:8s}: folder belum tersedia")
        continue
    gambar = [
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in EKSTENSI
    ]
    print(f"  {kelas:8s}: {len(gambar):4d} gambar")
    total += len(gambar)

print("=" * 40)
print(f"  Total   : {total:4d} gambar")
print(f"  Target  :  300 gambar (100 per kelas)")

if total == 0:
    print("\n  [!] Dataset masih kosong — masukkan foto dulu.")
elif total < 300:
    print(f"\n  [!] Kurang {300 - total} gambar dari target minimum.")
else:
    print("\n  [OK] Jumlah gambar sudah memenuhi target minimum.")
