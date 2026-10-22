# 📱 Klasifikasi Tingkat Keparahan Kerusakan Layar Smartphone (SVM)

Proyek klasifikasi citra cerdas berbasis **Support Vector Machine (SVM)** dan **Ekstraksi Fitur Tepi/Tekstur Digital** untuk mengkategorikan tingkat keparahan retak/pecah layar smartphone ke dalam tiga kelas: **Ringan**, **Sedang**, dan **Berat**.

---

## 🎯 Ringkasan Kinerja Model

- **Dataset:** 300 Citra Layar Smartphone (100 per kelas berimbang)
- **Metode Klasifikasi:** Support Vector Machine (SVM)
- **Kernel Terbaik:** Radial Basis Function (RBF) dengan $C=10, \gamma=0.1$
- **Akurasi Cross-Validation (5-Fold Latih):** **87.50%**
- **Akurasi Data Uji (Test Set 20%):** **86.67%** (52 dari 60 sampel diprediksi tepat)
- **Macro F1-Score:** **0.868** (86.85%)

### 📊 Laporan Klasifikasi (*Classification Report*)
| Kelas Kerusakan | Presisi (*Precision*) | Kepekaan (*Recall*) | *F1-Score* | Jumlah Data Uji (*Support*) |
|:---|:---:|:---:|:---:|:---:|
| **Ringan** | **76.00%** | **95.00%** | **0.844** | 20 |
| **Sedang** | **88.89%** | **80.00%** | **0.842** | 20 |
| **Berat** | **100.00%** | **85.00%** | **0.919** | 20 |
| **Rata-rata Makro** | **88.30%** | **86.67%** | **0.868** | **60** |

---

## 🏗️ Struktur Direktori Proyek

```
Klasifikasi-Layar-SVM/
├── dataset/                     # 300 citra berlabel (100 ringan, 100 sedang, 100 berat)
│   ├── ringan/
│   ├── sedang/
│   └── berat/
├── program/                     # Skrip pemrosesan data & model
│   ├── cek_dataset.py           # Verifikasi kuantitas data tiap kelas
│   ├── sortir_dataset.py        # Ekstraksi anotasi & pemetaan rasio kerusakan
│   ├── augmentasi.py            # Augmentasi geometris & fotometrik (flip, rotasi, kontras)
│   ├── train_svm.py             # Ekstraksi 12 fitur citra, GridSearch 5-fold & evaluasi
│   ├── prediksi.py              # Inferensi citra tunggal via command-line
│   └── ekspor_excel.py          # Ekspor dataset matriks fitur & data uji ke Excel
├── hasil/                       # Output eksperimen
│   ├── model_svm.joblib         # Bobot model SVM terlatih
│   ├── scaler.joblib            # StandardScaler fitur citra
│   ├── confusion_matrix.png     # Heatmap visualisasi matriks konfusi
│   ├── Laporan_Data_SVM_Layar.xlsx # File Excel 3-Sheet lengkap
│   ├── DRAF_LAPORAN_PENELITIAN.md  # Laporan ilmiah komprehensif
│   └── laporan_evaluasi.txt     # Catatan metrik evaluasi
├── app.py                       # Aplikasi Web Interaktif (Streamlit)
├── requirements.txt             # Dependensi Python
├── .gitignore
└── README.md
```

---

## 🔬 Rincian 12 Fitur Citra yang Diekstrak

Setiap citra distandarisasi ke resolusi $512 \times 512$ piksel dan diekstraksi 12 fitur tekstur & tepi:
1. `F1_Edge_Density`: Kerapatan piksel tepi detektor Canny ($T_1=50, T_2=150$).
2. `F2_Laplacian_Variance`: Variansi operator Laplacian (ketajaman retak).
3. `F3_Laplacian_Mean`: Rata-rata magnitudo turunan kedua citra.
4. `F4_Sobel_Mean`: Rata-rata gradien Sobel gabungan horizontal dan vertikal.
5. `F5_Sobel_Std`: Deviasi standar kekuatan gradien.
6. `F6_Sobel_P75`: Persentil ke-75 magnitudo gradien retak.
7. `F7_Sobel_P90`: Persentil ke-90 magnitudo gradien retak terkuat.
8. `F8_Contour_Count`: Jumlah kontur segmen pecahan kaca yang terdeteksi.
9. `F9_Norm_Contour_Length`: Total panjang kontur retakan ternormalisasi terhadap dimensi citra.
10. `F10_Grayscale_Mean`: Rata-rata intensitas pencahayaan citra.
11. `F11_Grayscale_Std`: Kontras global intensitas abu-abu.
12. `F12_Shannon_Entropy`: Entropi tekstur Shannon dari histogram distribusi intensitas.

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

### 2. Siapkan Virtual Environment
```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip & install dependensi
pip install -r requirements.txt
```

### 3. Melatih Ulang Model SVM
```powershell
python program/train_svm.py
```

### 4. Menjalankan Aplikasi Web (Streamlit)
```powershell
streamlit run app.py
```
Buka browser pada alamat `http://localhost:8501`, lalu unggah foto layar untuk melihat prediksi secara *real-time*.

### 5. Menguji Citra Tunggal via Terminal
```powershell
python program/prediksi.py dataset/berat/27413-13.png
```

---

## 📜 Lisensi & Atribusi Dataset
Dataset dasar bersumber dari *Broken Interfaces Dataset* (Florian Schaub et al., ACM CHI 2014) di bawah lisensi *Creative Commons Attribution 4.0 International License*.
