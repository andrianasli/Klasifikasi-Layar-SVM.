"""
ekspor_excel.py
Mengekspor seluruh data penelitian ke file Excel (.xlsx) rapi dengan 3 Sheet:
1. Matriks_Fitur_Dataset (300 foto + 12 nilai fitur + label)
2. Hasil_Pengujian_Data_Uji (60 data uji + prediksi SVM + status BENAR/SALAH)
3. Ringkasan_Metrik_Evaluasi (Akurasi, Precision, Recall, F1-score, Confusion Matrix)

Jalankan dari root proyek:
    python program/ekspor_excel.py
"""
import sys
import joblib
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(str(Path("program").resolve()))
from train_svm import ekstraksi_fitur, muat_dataset, KELAS_LIST

MODEL_PATH = Path("hasil/model_svm.joblib")
SCALER_PATH = Path("hasil/scaler.joblib")
OUTPUT_EXCEL = Path("hasil/Laporan_Data_SVM_Layar.xlsx")

NAMA_FITUR = [
    "F1_Edge_Density",
    "F2_Laplacian_Variance",
    "F3_Laplacian_Mean",
    "F4_Sobel_Mean",
    "F5_Sobel_Std",
    "F6_Sobel_P75",
    "F7_Sobel_P90",
    "F8_Contour_Count",
    "F9_Norm_Contour_Length",
    "F10_Grayscale_Mean",
    "F11_Grayscale_Std",
    "F12_Shannon_Entropy"
]

def main():
    print("=" * 60)
    print("MEMULAI EKSPOR DATA PENELITIAN KE EXCEL (.xlsx)")
    print("=" * 60)

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        print("[!] Model belum dilatih. Jalankan train_svm.py dulu.")
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 1. Muat dataset 300 foto dan ekstrak fitur
    X, y, nama_file = muat_dataset()

    # 2. Buat DataFrame Sheet 1: Matriks Fitur Dataset
    records_sheet1 = []
    for i in range(len(X)):
        row = {
            "No": i + 1,
            "Nama_File": nama_file[i],
            "Kelas": KELAS_LIST[y[i]],
            "Tipe_Data": "Augmentasi" if nama_file[i].startswith("aug_") else "Asli"
        }
        for j, fn in enumerate(NAMA_FITUR):
            row[fn] = round(float(X[i][j]), 4)
        records_sheet1.append(row)

    df_sheet1 = pd.DataFrame(records_sheet1)

    # 3. Split 80/20 identik dengan train_svm.py (random_state=42, stratify=y)
    indices = np.arange(len(X))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.20, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X[idx_test])
    y_pred = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

    # 4. Buat DataFrame Sheet 2: Hasil Pengujian Data Uji
    records_sheet2 = []
    for rank, i_test in enumerate(idx_test):
        pred_label = KELAS_LIST[y_pred[rank]]
        actual_label = KELAS_LIST[y[i_test]]
        is_correct = "BENAR" if pred_label == actual_label else "SALAH"

        row = {
            "No": rank + 1,
            "Nama_File": nama_file[i_test],
            "Kelas_Aktual": actual_label,
            "Kelas_Prediksi_SVM": pred_label,
            "Status_Prediksi": is_correct,
            "Prob_Ringan (%)": round(float(probs[rank][0] * 100), 2) if probs is not None else "-",
            "Prob_Sedang (%)": round(float(probs[rank][1] * 100), 2) if probs is not None else "-",
            "Prob_Berat (%)": round(float(probs[rank][2] * 100), 2) if probs is not None else "-"
        }
        records_sheet2.append(row)

    df_sheet2 = pd.DataFrame(records_sheet2)

    # 5. Buat DataFrame Sheet 3: Ringkasan Metrik & Confusion Matrix
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=KELAS_LIST, output_dict=True)

    summary_info = [
        {"Parameter": "Algoritma", "Nilai": "Support Vector Machine (SVM)"},
        {"Parameter": "Kernel", "Nilai": "Radial Basis Function (RBF)"},
        {"Parameter": "Hyperparameter Terbaik", "Nilai": "C=10, gamma=0.1"},
        {"Parameter": "Total Sampel Dataset", "Nilai": f"{len(X)} citra"},
        {"Parameter": "Data Latih (80%)", "Nilai": f"{len(idx_train)} citra"},
        {"Parameter": "Data Uji (20%)", "Nilai": f"{len(idx_test)} citra"},
        {"Parameter": "Akurasi Cross-Validation Latih (5-Fold)", "Nilai": "87.50%"},
        {"Parameter": "Akurasi Data Uji", "Nilai": f"{acc * 100:.2f}%"},
        {"Parameter": "Jumlah Prediksi Tepat", "Nilai": f"{(y_test == y_pred).sum()} dari {len(y_test)} citra"},
        {"Parameter": "Jumlah Prediksi Meleset", "Nilai": f"{(y_test != y_pred).sum()} dari {len(y_test)} citra"}
    ]
    df_summary = pd.DataFrame(summary_info)

    # Tabel Metrik per Kelas
    metrics_class = []
    for k in KELAS_LIST:
        metrics_class.append({
            "Kelas": k.capitalize(),
            "Precision": f"{cr[k]['precision'] * 100:.2f}%",
            "Recall": f"{cr[k]['recall'] * 100:.2f}%",
            "F1-Score": f"{cr[k]['f1-score']:.3f}",
            "Support (Jumlah Data Uji)": int(cr[k]['support'])
        })
    df_metrics = pd.DataFrame(metrics_class)

    # Tabel Confusion Matrix
    df_cm = pd.DataFrame(
        cm,
        index=[f"Aktual_{k.capitalize()}" for k in KELAS_LIST],
        columns=[f"Prediksi_{k.capitalize()}" for k in KELAS_LIST]
    ).reset_index().rename(columns={"index": "Kelas_Aktual"})

    # 6. Tulis ke file Excel menggunakan Pandas ExcelWriter
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df_sheet1.to_excel(writer, sheet_name="Matriks_Fitur_Dataset", index=False)
        df_sheet2.to_excel(writer, sheet_name="Hasil_Pengujian_Data_Uji", index=False)
        
        # Di Sheet 3 kita tulis summary, tabel metrik, dan confusion matrix
        df_summary.to_excel(writer, sheet_name="Ringkasan_Metrik", startrow=0, index=False)
        start_row_metrics = len(df_summary) + 3
        df_metrics.to_excel(writer, sheet_name="Ringkasan_Metrik", startrow=start_row_metrics, index=False)
        start_row_cm = start_row_metrics + len(df_metrics) + 3
        df_cm.to_excel(writer, sheet_name="Ringkasan_Metrik", startrow=start_row_cm, index=False)

    print(f"\n[OK] File Excel berhasil dibuat di: {OUTPUT_EXCEL}")
    print("=" * 60)

if __name__ == "__main__":
    main()
