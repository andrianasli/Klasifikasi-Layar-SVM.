"""
app.py
Aplikasi Web Antarmuka Pengguna (UI) untuk Klasifikasi Kerusakan Layar Smartphone
menggunakan Streamlit dan Model SVM.

Jalankan dengan perintah:
    .\.venv\Scripts\streamlit run app.py
"""
import streamlit as st
import cv2
import joblib
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Tambahkan path folder program ke sys.path
sys.path.append(str(Path("program").resolve()))
from train_svm import ekstraksi_fitur, KELAS_LIST

# ── Konfigurasi Halaman Streamlit ──────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Kerusakan Layar — SVM",
    page_icon="📱",
    layout="wide"
)

MODEL_PATH = Path("hasil/model_svm.joblib")
SCALER_PATH = Path("hasil/scaler.joblib")
CM_PATH = Path("hasil/confusion_matrix.png")
LAPORAN_PATH = Path("hasil/laporan_evaluasi.txt")

# ── Cache Model & Scaler ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📱 Klasifikasi Tingkat Keparahan Kerusakan Layar Smartphone")
st.markdown(
    "Sistem cerdas berbasis **Support Vector Machine (SVM)** dan **Ekstraksi Fitur Tepi/Tekstur Citra** "
    "untuk mendeteksi tingkat kerusakan layar ponsel (*Ringan*, *Sedang*, *Berat*)."
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Info Sistem & Model")
    st.success("✅ **Status Model:** Siap Digunakan")
    st.metric(label="Akurasi Data Uji", value="86.67%")
    st.metric(label="Akurasi CV (5-Fold)", value="87.50%")
    st.markdown("---")
    st.markdown("**Spesifikasi Penelitian:**")
    st.markdown("- **Metode:** Support Vector Machine (SVM)")
    st.markdown("- **Kernel:** Radial Basis Function (RBF)")
    st.markdown("- **Dataset:** 300 Citra (100 per kelas)")
    st.markdown("- **Pembagian:** 80% Latih / 20% Uji")
    st.markdown("---")
    st.markdown("© Proyek Riset Klasifikasi Layar SVM")

# ── Tab Antarmuka ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Kerusakan", "📊 Evaluasi Model", "📄 Draf Laporan"])

# ── Tab 1: Prediksi ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Citra Layar Smartphone")
    uploaded_file = st.file_uploader(
        "Pilih file foto layar ponsel (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        citra_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        citra_rgb = cv2.cvtColor(citra_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(citra_rgb, caption="Citra Layar yang Diunggah", use_container_width=True)

        # Visualisasi tepi Canny
        resized = cv2.resize(citra_bgr, (512, 512))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        with col2:
            st.image(edges, caption="Visualisasi Deteksi Tepi (Canny Edges)", use_container_width=True)

        if model is None or scaler is None:
            st.error("Model belum dilatih. Silakan jalankan `python program/train_svm.py` terlebih dahulu.")
        else:
            with st.spinner("Mengekstrak 12 fitur citra & melakukan klasifikasi SVM..."):
                fitur = ekstraksi_fitur(citra_bgr)
                fitur_scaled = scaler.transform([fitur])
                pred_idx = model.predict(fitur_scaled)[0]
                kelas_prediksi = KELAS_LIST[pred_idx]

                probs = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(fitur_scaled)[0]

            st.markdown("---")
            st.subheader("Hasil Klasifikasi:")

            # Badge warna berdasarkan kelas
            if kelas_prediksi == "ringan":
                st.success(f"### 🎉 TINGKAT KEPARAHAN: **{kelas_prediksi.upper()}**")
                st.info("Karakteristik: Retakan kecil terisolasi / goresan minor. Layar masih sangat fungsional.")
            elif kelas_prediksi == "sedang":
                st.warning(f"### ⚠️ TINGKAT KEPARAHAN: **{kelas_prediksi.upper()}**")
                st.info("Karakteristik: Jalur retakan menyebar di sebagian area layar. Memerlukan perbaikan segera.")
            else:
                st.error(f"### 🚨 TINGKAT KEPARAHAN: **{kelas_prediksi.upper()}**")
                st.info("Karakteristik: Pecahan kaca parah / jaring laba-laba padat. Risiko serpihan kaca dan disfungsi total.")

            if probs is not None:
                st.markdown("**Tingkat Keyakinan (Confidence Level):**")
                p_col1, p_col2, p_col3 = st.columns(3)
                p_col1.metric("Probabilitas Ringan", f"{probs[0]*100:.1f}%")
                p_col1.progress(float(probs[0]))
                p_col2.metric("Probabilitas Sedang", f"{probs[1]*100:.1f}%")
                p_col2.progress(float(probs[1]))
                p_col3.metric("Probabilitas Berat", f"{probs[2]*100:.1f}%")
                p_col3.progress(float(probs[2]))

            with st.expander("Lihat Nilai 12 Fitur Citra yang Diekstraksi"):
                st.json({
                    "1_Edge_Density": f"{fitur[0]:.5f}",
                    "2_Laplacian_Variance": f"{fitur[1]:.2f}",
                    "3_Laplacian_Mean": f"{fitur[2]:.2f}",
                    "4_Sobel_Mean": f"{fitur[3]:.2f}",
                    "5_Sobel_Std": f"{fitur[4]:.2f}",
                    "6_Sobel_Percentile_75": f"{fitur[5]:.2f}",
                    "7_Sobel_Percentile_90": f"{fitur[6]:.2f}",
                    "8_Contour_Count": int(fitur[7]),
                    "9_Normalized_Contour_Length": f"{fitur[8]:.2f}",
                    "10_Grayscale_Mean": f"{fitur[9]:.2f}",
                    "11_Grayscale_Std": f"{fitur[10]:.2f}",
                    "12_Shannon_Entropy": f"{fitur[11]:.2f}"
                })

# ── Tab 2: Evaluasi Model ──────────────────────────────────────────────────
with tab2:
    st.subheader("Evaluasi Performa Model SVM pada Data Uji (20%)")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        if CM_PATH.exists():
            st.image(str(CM_PATH), caption="Confusion Matrix Model SVM (Akurasi: 86.7%)", use_container_width=True)
        else:
            st.info("Confusion matrix belum dibuat.")

    with col_b:
        if LAPORAN_PATH.exists():
            st.text_area("Laporan Metrik Evaluasi:", LAPORAN_PATH.read_text(encoding="utf-8"), height=350)
        else:
            st.info("Laporan evaluasi belum tersedia.")

# ── Tab 3: Draf Laporan ────────────────────────────────────────────────────
with tab3:
    st.subheader("Draf Laporan Penelitian (Siap Salin)")
    laporan_file = Path("hasil/DRAF_LAPORAN_PENELITIAN.md")
    if laporan_file.exists():
        st.markdown(laporan_file.read_text(encoding="utf-8"))
    else:
        st.info("File draf laporan belum tersedia di folder hasil.")
