"""
train_svm.py
Pipeline Lengkap: Ekstraksi Fitur Citra, Pelatihan SVM, dan Evaluasi Model.

Jalankan dari root proyek:
    python program/train_svm.py
"""
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# ── Konfigurasi ─────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")
HASIL_DIR = Path("hasil")
KELAS_LIST = ["ringan", "sedang", "berat"]
UKURAN_CITRA = (512, 512)  # Standarisasi ukuran citra sebelum ekstraksi


def hitung_entropi(citra_gray: np.ndarray) -> float:
    """Menghitung entropi tekstur Shannon dari histogram keabuan."""
    hist = cv2.calcHist([citra_gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-7)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def ekstraksi_fitur(citra_bgr: np.ndarray) -> list[float]:
    """
    Mengekstraksi representasi fitur tekstur dan tepi keretakan layar:
    1. Kerapatan tepi (Canny Edge Density)
    2. Variansi Laplacian (Ketajaman retakan)
    3. Rata-rata & deviasi standar gradien Sobel
    4. Persentil 75% & 90% magnitudo Sobel
    5. Jumlah & total panjang kontur retakan
    6. Kontras & entropi intensitas keabuan
    """
    resized = cv2.resize(citra_bgr, UKURAN_CITRA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Deteksi Tepi Canny
    edges = cv2.Canny(blurred, 50, 150)
    total_piksel = UKURAN_CITRA[0] * UKURAN_CITRA[1]
    edge_density = np.count_nonzero(edges) / total_piksel

    # 2. Variansi & Mean Laplacian (derajat diskontinuitas tekstur)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(laplacian.var())
    lap_mean = float(np.abs(laplacian).mean())

    # 3. Gradien Sobel X dan Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mean = float(sobel_mag.mean())
    sobel_std = float(sobel_mag.std())
    sobel_p75 = float(np.percentile(sobel_mag, 75))
    sobel_p90 = float(np.percentile(sobel_mag, 90))

    # 4. Statistik Kontur Garis Retakan
    kontur, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    jum_kontur = len(kontur)
    total_panjang_kontur = sum(cv2.arcLength(c, False) for c in kontur)
    norm_panjang_kontur = total_panjang_kontur / np.sqrt(total_piksel)

    # 5. Statistik Keabuan
    gray_mean = float(gray.mean())
    gray_std = float(gray.std())
    entropi = hitung_entropi(gray)

    return [
        edge_density,
        lap_var,
        lap_mean,
        sobel_mean,
        sobel_std,
        sobel_p75,
        sobel_p90,
        float(jum_kontur),
        norm_panjang_kontur,
        gray_mean,
        gray_std,
        entropi
    ]


def muat_dataset():
    """Membaca semua gambar dari folder dataset/ dan mengekstrak fiturnya."""
    X, y, nama_file = [], [], []
    ekstensi = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    print("=" * 60)
    print("1. MEMUAT DATASET & EKSTRAKSI FITUR CITRA")
    print("=" * 60)

    for label_idx, kelas in enumerate(KELAS_LIST):
        folder = DATASET_DIR / kelas
        if not folder.exists():
            continue
        file_list = [f for f in folder.iterdir() if f.suffix.lower() in ekstensi]
        print(f"  Memproses kelas '{kelas}' : {len(file_list)} gambar...")

        for f in file_list:
            citra = cv2.imread(str(f))
            if citra is None:
                continue
            fitur = ekstraksi_fitur(citra)
            X.append(fitur)
            y.append(label_idx)
            nama_file.append(f.name)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"\n[OK] Total data diekstraksi : {len(X)} sampel, {X.shape[1]} fitur.")
    return X, y, nama_file


def main():
    HASIL_DIR.mkdir(exist_ok=True)

    # 1. Ekstraksi Fitur
    X, y, _ = muat_dataset()
    if len(X) == 0:
        print("[!] Tidak ada data untuk dilatih. Jalankan sortir_dataset.py terlebih dahulu.")
        return

    # 2. Split Data: 80% Pelatihan, 20% Pengujian (Stratified)
    print("\n" + "=" * 60)
    print("2. PEMBAGIAN DATASET (80% Train, 20% Test)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  - Jumlah Data Latih : {len(X_train)} sampel")
    print(f"  - Jumlah Data Uji   : {len(X_test)} sampel")

    # 3. Standarisasi Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Hyperparameter Tuning SVM (GridSearchCV)
    print("\n" + "=" * 60)
    print("3. PELATIHAN MODEL SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 60)
    param_grid = {
        "C": [0.1, 1, 10, 50, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto", 0.01, 0.1]
    }
    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    print("  Sedang melakukan 5-Fold Cross Validation untuk parameter terbaik...")
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    print(f"  [OK] Parameter Terbaik SVM : {grid.best_params_}")
    print(f"  [OK] Akurasi CV Terbaik    : {grid.best_score_ * 100:.2f}%")

    # 5. Evaluasi pada Data Uji (Test Set)
    print("\n" + "=" * 60)
    print("4. EVALUASI MODEL PADA DATA UJI (20%)")
    print("=" * 60)
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    laporan = classification_report(y_test, y_pred, target_names=KELAS_LIST)

    print(f"  Akurasi Data Uji: {acc * 100:.2f}%\n")
    print("  Classification Report:")
    print(laporan)

    # 6. Simpan Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=KELAS_LIST)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix SVM (Akurasi: {acc * 100:.1f}%)")
    plt.tight_layout()

    cm_path = HASIL_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  [OK] Confusion matrix disimpan ke : {cm_path}")

    # 7. Simpan Model & Scaler
    model_path = HASIL_DIR / "model_svm.joblib"
    scaler_path = HASIL_DIR / "scaler.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  [OK] Model SVM tersimpan di        : {model_path}")
    print(f"  [OK] Scaler tersimpan di           : {scaler_path}")

    # 8. Simpan Laporan Tertulis
    laporan_path = HASIL_DIR / "laporan_evaluasi.txt"
    with open(laporan_path, "w", encoding="utf-8") as f:
        f.write("=== LAPORAN EVALUASI MODEL SVM KLASIFIKASI KERUSAKAN LAYAR ===\n\n")
        f.write(f"Parameter Terbaik : {grid.best_params_}\n")
        f.write(f"Akurasi CV Latih  : {grid.best_score_ * 100:.2f}%\n")
        f.write(f"Akurasi Data Uji  : {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(laporan)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"  [OK] Laporan teks tersimpan di     : {laporan_path}")
    print("\n" + "=" * 60)
    print("PIPELINE PELATIHAN & EVALUASI SELESAI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
