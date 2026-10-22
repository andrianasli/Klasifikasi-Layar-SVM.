"""
prediksi.py
Menguji citra layar ponsel baru dengan model SVM yang telah dilatih.

Penggunaan:
    python program/prediksi.py <path_ke_file_gambar>

Contoh:
    python program/prediksi.py dataset/berat/27413-13.png
"""
import sys
import cv2
import joblib
import numpy as np
from pathlib import Path
from train_svm import ekstraksi_fitur, KELAS_LIST

MODEL_PATH = Path("hasil/model_svm.joblib")
SCALER_PATH = Path("hasil/scaler.joblib")

def prediksi(gambar_path: str):
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        print("[!] Model atau Scaler belum ditemukan di folder hasil/")
        print("    Jalankan 'python program/train_svm.py' terlebih dahulu.")
        return

    path = Path(gambar_path)
    if not path.exists():
        print(f"[!] File tidak ditemukan: {gambar_path}")
        return

    citra = cv2.imread(str(path))
    if citra is None:
        print(f"[!] Gagal membaca citra: {gambar_path}")
        return

    # Muat model dan scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Ekstraksi fitur citra & scaling
    fitur = ekstraksi_fitur(citra)
    fitur_scaled = scaler.transform([fitur])

    # Prediksi kelas dan probabilitas
    pred_idx = model.predict(fitur_scaled)[0]
    kelas_prediksi = KELAS_LIST[pred_idx]

    print("\n" + "=" * 50)
    print("      HASIL PREDIKSI KERUSAKAN LAYAR (SVM)")
    print("=" * 50)
    print(f"  File            : {path.name}")
    print(f"  Prediksi Kelas  : {kelas_prediksi.upper()}")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(fitur_scaled)[0]
        print("\n  Probabilitas:")
        for k, p in zip(KELAS_LIST, probs):
            bar = "#" * int(p * 20)
            print(f"    - {k:8s} : {p * 100:5.1f}%  [{bar:<20s}]")

    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python program/prediksi.py <path_ke_file_gambar>")
        print("Contoh: python program/prediksi.py dataset/berat/27413-13.png")
    else:
        prediksi(sys.argv[1])
