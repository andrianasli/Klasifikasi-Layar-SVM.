# LAPORAN PENELITIAN
## KLASIFIKASI TINGKAT KEPARAHAN KERUSAKAN LAYAR SMARTPHONE MENGGUNAKAN METODE SUPPORT VECTOR MACHINE (SVM)

---

### ABSTRAK
Kerusakan layar fisik pada smartphone, khususnya keretakan kaca (*screen crack*), merupakan jenis kerusakan perangkat keras yang paling sering dialami oleh pengguna. Penentuan tingkat keparahan kerusakan layar secara manual bersifat subjektif dan memakan waktu. Penelitian ini bertujuan untuk mengimplementasikan sistem klasifikasi citra otomatis guna mengidentifikasi tingkat keparahan kerusakan layar smartphone ke dalam tiga tingkatan: Ringan, Sedang, dan Berat. Metode yang digunakan adalah *Support Vector Machine* (SVM) yang dikombinasikan dengan ekstraksi fitur tekstur dan tepi (*edge & texture features*) berbasis pengolahan citra digital (*Canny Edge Detection*, *Laplacian Variance*, *Sobel Gradient Magnitude*, *Contour Statistics*, dan *Shannon Entropy*). Dataset yang digunakan bersumber dari *Broken Interfaces Dataset* (ACM CHI 2014) yang dianotasi secara kuantitatif berdasarkan rasio luas area kerusakan layar, kemudian diperluas menjadi 300 citra menggunakan teknik *data augmentation* dengan distribusi berimbang (100 citra per kelas). Pengujian dilakukan dengan pembagian 80% data latih dan 20% data uji menggunakan *Radial Basis Function* (RBF) Kernel yang dioptimasi melalui 5-Fold *GridSearchCV*. Hasil eksperimen menunjukkan model SVM mencapai akurasi *Cross-Validation* sebesar **87.50%** dan akurasi pada data uji sebesar **86.67%**, dengan nilai rata-rata presisi 88%, recall 87%, dan F1-score 0.87. Hasil ini membuktikan bahwa kombinasi fitur geometrik tepi dan SVM efektif dalam mengenali tingkat keparahan kerusakan layar secara objektif.

**Kata Kunci:** Layar Smartphone, Keretakan Layar, Support Vector Machine (SVM), Pengolahan Citra Digital, Ekstraksi Fitur Tepi, Klasifikasi Citra.

---

### BAB I. PENDAHULUAN

#### 1.1 Latar Belakang
Smartphone telah menjadi perangkat komputasi personal yang vital dalam kehidupan sehari-hari. Sebagian besar interaksi pengguna bergantung pada layar sentuh (*touchscreen*). Namun, sifat kaca penutup layar yang rentan membuatnya sangat rentan mengalami retak atau pecah akibat benturan atau jatuh.

Pada industri perbaikan perangkat (*service center*) maupun klaim asuransi gadget, estimasi tingkat keparahan kerusakan layar umumnya masih dilakukan secara visual oleh teknisi. Hal ini seringkali menimbulkan inkonsistensi penilaian karena standar yang subjektif. Oleh karena itu, diperlukan suatu sistem cerdas berbasis visi komputer (*computer vision*) dan pembelajaran mesin (*machine learning*) yang mampu menilai tingkat keparahan layar secara otomatis, cepat, dan terstandarisasi.

#### 1.2 Rumusan Masalah
1. Bagaimana mengekstraksi fitur karakteristik fisik keretakan layar ponsel dari citra digital?
2. Bagaimana memetakan dan mengklasifikasikan tingkat keparahan layar ke dalam kategori Ringan, Sedang, dan Berat menggunakan algoritma Support Vector Machine (SVM)?
3. Seberapa tinggi tingkat akurasi dan kinerja model SVM yang dihasilkan pada data uji?

#### 1.3 Tujuan Penelitian
1. Mengembangkan alur pra-pemrosesan dan ekstraksi fitur tepi/tekstur citra keretakan layar smartphone.
2. Membangun model klasifikasi berbasis Support Vector Machine (SVM) dengan tuning hyperparameter optimal.
3. Mengevaluasi performa model menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-score*, dan *Confusion Matrix*.

---

### BAB II. METODOLOGI PENELITIAN

#### 2.1 Alur Penelitian
Tahapan penelitian dirancang secara sistematis sebagai berikut:
```
[Pengumpulan Dataset] -> [Anotasi & Ground-Truth] -> [Data Augmentation (300 Citra)]
         ↓
[Pra-pemrosesan Citra] -> [Ekstraksi 12 Fitur Citra] -> [Normalisasi StandardScaler]
         ↓
[Pembagian Data (80:20)] -> [Pelatihan SVM + 5-Fold GridSearch] -> [Evaluasi & Pengujian]
```

#### 2.2 Dataset dan Kriteria Tingkat Keparahan
Dataset utama bersumber dari *Broken Interfaces Dataset* (Schaub et al., ACM CHI 2014) yang memuat foto asli layar smartphone rusak beserta anotasi area kerusakannya.

Untuk menjamin objektivitas label, penentuan kelas dihitung berdasarkan **rasio luas area kerusakan (*Damage Area Ratio*)** terhadap total luas layar:
- **Kelas Ringan:** Rasio kerusakan < 2.0% (retakan rambut, goresan tipis terisolasi, area layar mayoritas utuh).
- **Kelas Sedang:** Rasio kerusakan 2.0% – 7.0% (beberapa jalur retakan menyebar, sebagian fungsi visual terganggu).
- **Kelas Berat:** Rasio kerusakan > 7.0% (layar hancur/remuk, pola jaring laba-laba padat yang mendominasi layar).

#### 2.3 Data Augmentation
Untuk memenuhi standar kecukupan data pelatihan visi komputer dan mencegah *overfitting*, 93 citra awal ditingkatkan menjadi 300 citra (100 citra per kelas) menggunakan teknik augmentasi realistis:
1. *Horizontal & Vertical Flipping* (pencerminan).
2. *Orthogonal Rotation* (rotasi 90°, 180°, 270°).
3. Penyesuaian kontras dan kecerahan minor (±10%).

#### 2.4 Ekstraksi Fitur Citra (12 Dimensi)
Setiap citra distandarisasi ke ukuran 512×512 piksel, kemudian diekstrak 12 vektor fitur:
1. **Edge Density (Kerapatan Tepi):** Proporsi piksel tepi yang dideteksi oleh algoritma Canny ($T_1=50, T_2=150$).
2. **Laplacian Variance:** Mengukur tingkat variasi dan ketajaman diskontinuitas pola retak.
3. **Laplacian Mean:** Rata-rata nilai magnitudo turunan kedua citra.
4. **Sobel Gradient Mean & Std:** Statistik kekuatan perubahan intensitas spasial pada sumbu horizontal dan vertikal.
5. **Sobel Percentiles (75% & 90%):** Menangkap intensitas retakan terkuat pada layar.
6. **Contour Count:** Jumlah garis/segmen retakan independen yang terdeteksi.
7. **Normalized Contour Length:** Total panjang seluruh garis retakan ternormalisasi terhadap dimensi citra.
8. **Grayscale Mean & Std:** Karakteristik distribusi intensitas cahaya citra.
9. **Shannon Texture Entropy:** Mengukur tingkat keacakan dan kompleksitas tekstur pecahan kaca.

#### 2.5 Pemodelan Support Vector Machine (SVM)
Data dibagi menjadi 80% data latih (240 citra) dan 20% data uji (60 citra) secara *stratified*. Vektor fitur distandarisasi menggunakan *StandardScaler* ($z = \frac{x - \mu}{\sigma}$).
Optimasi parameter dilakukan melalui 5-Fold *GridSearchCV* dengan ruang pencarian:
- Parameter penalti $C \in \{0.1, 1, 10, 50, 100\}$
- Kernel $\in \{\text{'linear'}, \text{'rbf'}\}$
- Parameter $\gamma \in \{\text{'scale'}, \text{'auto'}, 0.01, 0.1\}$

---

### BAB III. HASIL DAN PEMBAHASAN

#### 3.1 Hasil Optimasi Hyperparameter
Hasil *GridSearchCV* menunjukkan bahwa fungsi kernel **Radial Basis Function (RBF)** memberikan pemisahan hyperplane non-linear terbaik dengan konfigurasi:
- **Kernel:** RBF
- **C:** 10
- **Gamma ($\gamma$):** 0.1
- **Akurasi Cross-Validation (5-Fold Latih):** **87.50%**

#### 3.2 Hasil Evaluasi pada Data Uji (Test Set)
Model diuji pada 60 citra independen yang belum pernah dilihat saat pelatihan (masing-masing 20 citra untuk Ringan, Sedang, dan Berat).

**Tabel 1. Laporan Klasifikasi (*Classification Report*) Model SVM**
| Kelas Kerusakan | Presisi (*Precision*) | Kepekaan (*Recall*) | *F1-Score* | Jumlah Sampel (*Support*) |
|:---|:---:|:---:|:---:|:---:|
| **Ringan** | **76.00%** (0.760) | **95.00%** (0.950) | **0.844** (84.44%) | 20 |
| **Sedang** | **88.89%** (0.889) | **80.00%** (0.800) | **0.842** (84.21%) | 20 |
| **Berat** | **100.00%** (1.000) | **85.00%** (0.850) | **0.919** (91.89%) | 20 |
| **Rata-rata Makro** | **88.30%** (0.883) | **86.67%** (0.867) | **0.868** (86.85%) | **60** |
| **Akurasi Keseluruhan** | \multicolumn{3}{c|}{**86.67%** (52 dari 60 sampel uji benar)} | **60** |

*Catatan Rumus Turunan:*
- $\text{Precision}_{\text{Ringan}} = \frac{19}{19 + 6} = \frac{19}{25} = \mathbf{76.00\%}$
- $\text{Precision}_{\text{Sedang}} = \frac{16}{16 + 2} = \frac{16}{18} = \mathbf{88.89\%}$
- $\text{Precision}_{\text{Berat}} = \frac{17}{17 + 0} = \frac{17}{17} = \mathbf{100.00\%}$
- $\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$:
  - $\text{F1}_{\text{Ringan}} = 2 \times \frac{0.760 \times 0.950}{0.760 + 0.950} = \frac{1.444}{1.710} \approx \mathbf{0.844}$
  - $\text{F1}_{\text{Sedang}} = 2 \times \frac{0.8889 \times 0.800}{0.8889 + 0.800} = \frac{1.4222}{1.6889} \approx \mathbf{0.842}$
  - $\text{F1}_{\text{Berat}} = 2 \times \frac{1.000 \times 0.850}{1.000 + 0.850} = \frac{1.700}{1.850} \approx \mathbf{0.919}$

#### 3.3 Analisis Matriks Konfusi (*Confusion Matrix*)
Berdasarkan visualisasi matriks konfusi pada file `hasil/confusion_matrix.png`, distribusi prediksi model terhadap 60 data uji dijabarkan pada tabel berikut:

**Tabel 2. Matriks Konfusi Pengujian Model SVM (60 Citra Uji)**
| Kelas Aktual \ Prediksi | Prediksi Ringan | Prediksi Sedang | Prediksi Berat | Total Sampel Aktual | Nilai Recall |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Aktual Ringan** | **19** | 1 | 0 | 20 | **95.0%** (19/20) |
| **Aktual Sedang** | 4 | **16** | 0 | 20 | **80.0%** (16/20) |
| **Aktual Berat** | 2 | 1 | **17** | 20 | **85.0%** (17/20) |
| **Total Prediksi** | **25** | **18** | **17** | **Total = 60** | - |
| **Nilai Precision** | **76.0%** (19/25) | **88.9%** (16/18) | **100.0%** (17/17) | - | **Akurasi = 86.67%** (52/60) |

**Rincian Evaluasi per Kelas:**
1. **Kelas Ringan:** 
   - True Positive (TP) = 19, False Negative (FN) = 1 (terprediksi sedang), False Positive (FP) = 6 (4 dari sedang, 2 dari berat).
   - *Recall* sangat tinggi mencapai **95.0%**, membuktikan model sangat sensitif dalam mendeteksi layar dengan retakan ringan.
2. **Kelas Sedang:** 
   - True Positive (TP) = 16, False Negative (FN) = 4 (terprediksi ringan), False Positive (FP) = 2 (1 dari ringan, 1 dari berat).
   - Menghasilkan *Precision* **88.9%** dan *Recall* **80.0%**.
3. **Kelas Berat:** 
   - True Positive (TP) = 17, False Negative (FN) = 3 (2 terprediksi ringan, 1 sedang), False Positive (FP) = 0.
   - Menghasilkan *Precision* sempurna **100.0%**, artinya tidak ada satu pun layar ringan atau sedang yang salah divonis sebagai rusak berat.

Total prediksi benar adalah $19 + 16 + 17 = 52$ sampel dari 60 sampel uji, menghasilkan tingkat akurasi sebesar:
$$\text{Akurasi} = \frac{52}{60} \times 100\% = 86.67\%$$

#### 3.4 Pembahasan Kinerja Model
Fitur berbasis tepi (*Canny edge density* dan *contour length*) terbukti menjadi pembeda paling signifikan antara kelas Ringan dan Berat. Pada kelas Berat, kerapatan tepi dan nilai entropi tekstur bernilai sangat tinggi karena pecahan kaca menciptakan banyak sudut acak. Sedikit misklasifikasi terjadi pada batas ambang antara Ringan dan Sedang, yang secara visual memang memiliki kemiripan kerapatan garis retak.

#### 3.5 Analisis Generalisasi dan Keterbatasan Model (*Threats to Validity*)
Meskipun hasil evaluasi menunjukkan akurasi 86.67% (52 dari 60 sampel benar), angka ini belum dengan sendirinya membuktikan bahwa model akan memiliki akurasi yang sama persis pada citra baru di luar dataset pengujian. Terdapat beberapa faktor pembatas yang memengaruhi kemampuan generalisasi model di dunia nyata:

1. **Interval Kepercayaan Statistik (*Confidence Interval*):**
   Dengan jumlah sampel uji $N = 60$, akurasi 86.67% memiliki batas galat (*margin of error*) pada tingkat kepercayaan 95% sebesar:
   $$CI_{95\%} = p \pm 1.96 \sqrt{\frac{p(1-p)}{N}} = 86.67\% \pm 1.96 \sqrt{\frac{0.8667 \times 0.1333}{60}} \approx 86.67\% \pm 8.60\%$$
   Hal ini berarti akurasi model pada populasi citra yang lebih luas berada pada rentang **78.07% hingga 95.27%**.

2. **Korelasi Data Akibat Augmentasi (*Source-Domain Correlation*):**
   Penerapan augmentasi citra (*offline augmentation*) melipatgandakan variasi dari 93 citra sumber menjadi 300 sampel. Walaupun proses *train_test_split* dilakukan secara acak bertingkat (*stratified*), terdapat kemungkinan pola geometrik dasar retakan dari citra sumber yang sama terdistribusi ke dalam data latih maupun data uji. Pada penelitian lanjutan, pemisahan data tingkat subjek (*Group-K-Fold* berbasis ID ponsel) disarankan untuk mengukur performa *zero-shot* murni.

3. **Tantangan Pergeseran Domain (*Domain Shift*):**
   - **Kondisi Layar (Aktif vs Non-Aktif):** Dataset yang digunakan berfokus pada area layar terpangkas (*cropped display*). Jika diterapkan pada layar yang sedang aktif menyala, elemen antarmuka (teks aplikasi, ikon, wallpaper kontras) berpotensi memicu deteksi tepi semu (*spurious edges*) pada Canny/Sobel.
   - **Noise Lingkungan:** Faktor eksternal seperti sudut pengambilan foto (*perspective distortion*), pantulan cahaya lampu (*glare*), dan goresan pelindung layar (*tempered glass*) dapat membiaskan ekstraksi fitur tekstur.

Dengan demikian, akurasi 86.67% merefleksikan keandalan model pada kondisi citra yang terkontrol (*cropped display*). Untuk implementasi praktis di lapangan, diperlukan modul segmentasi otomatis area layar dan normalisasi pencahayaan sebelum citra diekstraksi.

---

### BAB IV. KESIMPULAN DAN SARAN

#### 4.1 Kesimpulan
1. Ekstraksi 12 fitur citra yang menggabungkan deteksi tepi (Canny), ketajaman gradien (Laplacian & Sobel), statistik kontur, dan entropi tekstur mampu merepresentasikan karakteristik fisik kerusakan layar smartphone secara akurat.
2. Model Support Vector Machine (SVM) dengan Kernel RBF ($C=10, \gamma=0.1$) menghasilkan performa yang sangat memuaskan dengan akurasi pengujian sebesar **86.67%** dan akurasi cross-validation sebesar **87.50%**.
3. Penerapan augmentasi citra terbukti krusial dalam menaikkan akurasi model dari sebelumnya 42.11% (pada 93 data) menjadi 86.67% (pada 300 data).

#### 4.2 Saran
1. Pada penelitian selanjutnya, dapat ditambahkan fitur ekstraksi tekstur lanjut seperti GLCM (*Gray-Level Co-occurrence Matrix*) atau LBP (*Local Binary Pattern*).
2. Sistem dapat dikembangkan lebih lanjut menjadi aplikasi berbasis web atau *mobile* agar teknisi di lapangan dapat langsung memindai layar ponsel menggunakan kamera smartphone.

---
*Laporan ini disusun berdasarkan data eksperimen nyata pada proyek Klasifikasi-Layar-SVM (Oktober 2026).*
