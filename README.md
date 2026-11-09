# CURE & IndoBERT + RCNN Research Pipeline for Child Stunting Analysis

Platform penelitian terintegrasi untuk klasterisasi dan pelabelan otomatis data stunting balita di Indonesia menggunakan metode **CURE (Clustering Using Representatives)** dan arsitektur neural **IndoBERT + RCNN**, diselaraskan dengan standar **Permenkes RI No. 2 Tahun 2020** dan **WHO Child Growth Standards 2006**.

---

## 1. Peran IndoBERT: Pelabelan Otomatis Klaster (*Automatic Cluster Labeling*)

Sesuai metodologi inti penelitian tesis, **IndoBERT + RCNN bukan sekadar model klasifikasi tingkat record individual yang terpisah, melainkan bertindak sebagai pemberi label semantik otomatis bagi klaster yang dibentuk oleh CURE**:

1. **Pembentukan Klaster Numerik (CURE)**:
   - CURE mengelompokkan data balita berdasarkan fitur antropometri numerik terstandarisasi ($[HAZ, WAZ, WHZ, \text{normalized\_age}]$) ke dalam klaster internal dengan ID numerik: `0`, `1`, `2`, dst.
2. **Pelabelan Semantik Klaster (IndoBERT + RCNN)**:
   - Untuk setiap klaster numerik $C_j$, model IndoBERT + RCNN menganalisis catatan klinis/keluhan bidan dari anggota-anggota klaster tersebut.
   - Model menghitung distribusi probabilitas posterior kelas untuk masing-masing catatan teks anggota klaster: $P(\text{class} = c \mid \text{text}_i)$.
   - **Konsensus Klaster**: Label semantik klaster $\text{Cluster\_Label}(C_j)$ ditentukan berdasarkan rata-rata probabilitas posterior tertinggi antar-anggota klaster:
     $$\bar{P}(c \mid C_j) = \frac{1}{|C_j|} \sum_{i \in C_j} P(c \mid \text{text}_i)$$
     $$\text{Best\_Class}(C_j) = \arg\max_c \bar{P}(c \mid C_j)$$
   - Klaster memperoleh nama semantik yang bermakna dan dapat diinterpretasikan secara klinis, misal: `Klaster 0: Stunting Kronis / Sangat Pendek (HAZ < -3 SD)`.
3. **Pembedaan Tegas Antara `Cluster_ID` vs `Cluster_Label`**:
   - `CURE_Cluster_ID`: Pengenal numerik internal hasil partisi CURE (`0`, `1`, `2`, atau `-1` jika data antropometri tidak lengkap).
   - `CURE_Cluster_Label`: Nama semantik klaster yang dihasilkan melalui konsensus representasi kontekstual IndoBERT.
   - `IndoBERT_Record_Prediksi`: Prediksi individual per-anak (dipisahkan secara tegas dari label klaster agregat).

---

## 2. Diskrepansi Metodologis dalam Naskah Tesis

Naskah tesis mendeskripsikan hubungan CURE dan IndoBERT dengan dua interpretasi urutan yang berbeda pada bagian bab yang berbeda:

- **Interpretasi A (Klasterisasi CURE Dahulu $\to$ Pelabelan IndoBERT Kemudian)**:
  - CURE dijalankan terlebih dahulu pada Z-score antropometri untuk menemukan partisi geometri alami balita tanpa supervisi.
  - IndoBERT kemudian membaca catatan teks pada masing-masing klaster untuk menetapkan label semantik klaster secara otomatis.
- **Interpretasi B (Representasi Teks IndoBERT Dahulu $\to$ Klasterisasi CURE Kemudian)**:
  - IndoBERT mengekstraksi dense embeddings teks terlebih dahulu, kemudian digabungkan dengan fitur antropometri, dan CURE melakukan klasterisasi pada ruang fitur gabungan.

**Keputusan Implementasi**:
Codebase ini mengimplementasikan **Interpretasi A** sebagai alur kerja utama karena secara langsung menjawab kebutuhan pelabelan otomatis klaster (`Cluster_Label`). Pipeline didesain modular melalui parameter `processing_sequence=["cure", "indobert"]` sehingga urutan dapat diubah dengan konfigurasi minimal jika komite tesis menghendaki Interpretasi B.

---

## 3. Struktur Modul Proyek

```text
├── cure.py               # Algoritma CURE otentik, validasi dimensi & NaN/Inf, penanganan dataset kecil
├── indobert_rcnn.py      # PyTorch IndoBERT+RCNN, masking pooling, split data, training, label_cluster & label_all_clusters
├── pipeline.py           # Pipeline modular: pembacaan Excel, validasi non-destruktif, pelabelan klaster CURE_Cluster_Label
├── evaluation.py         # Metrik evaluasi: Silhouette CURE, evaluate_cluster_labeling, evaluasi klasifikasi independen
├── app.py                # Server Flask: REST API, endpoint training, sanitasi PII, demo synthetic disclaimer
├── lib/
│   └── stunting-pipeline.ts  # Implementasi CURE & antropometri client-side untuk dashboard interaktif
├── app/
│   ├── page.tsx          # Antarmuka web Next.js App Router (upload Excel, preview, filter, download)
│   └── layout.tsx        # Layout root aplikasi
├── tests/
│   └── test_pipeline.py  # 23 unit tests otomatis mencakup seluruh kriteria metodologi penelitian
└── requirements.txt      # Dependensi Python untuk lingkungan server/GPU
```

---

## 4. Alur Pelatihan, Validasi, dan Pengujian Model

### A. Pencegahan Data Leakage & Pembagian Dataset
Fungsi `prepare_dataset_splits()`:
- Memvalidasi keselarasan jumlah teks dan label.
- Mendeteksi teks duplikat dan menempatkannya ke dalam kelompok partisi yang sama untuk mencegah kebocoran data (*data leakage*) antara train, val, dan test.
- Melakukan pembagian terstratifikasi (*stratified split*) sesuai proporsi (misal 70% train, 15% val, 15% test).
- Menolak dataset yang terlalu kecil (minimal 10 sampel).

### B. Pelatihan dengan Pemilihan Checkpoint Terbaik
Metode `IndoBERTRCNNPipeline.train()`:
- Menjalankan iterasi epoch dengan penghitungan loss latih (*train loss*).
- Mengevaluasi set validasi pada setiap epoch (*validation loss* dan *accuracy*).
- Menyimpan bobot terbaik (*best checkpoint*) berdasarkan metrik validasi terendah/terbaik.
- Menyimpan metadata eksperimen lengkap ke `indobert_rcnn_stunting.pt`.

---

## 5. Perintah Eksekusi

### A. Menjalankan Tes Otomatis (Automated Tests)
```bash
python3 -m unittest discover tests/
```
Semua 23 pengujian mencakup:
1. Pelabelan semantik klaster via konsensus IndoBERT dan evaluasi kemurnian klaster.
2. Pembedaan tegas `Cluster_ID` vs `Cluster_Label`.
3. Validasi input pelatihan dan keselarasan teks-label.
4. Penolakan label tidak dikenal dan dataset tidak memadai.
5. Pembagian terstratifikasi bebas *data leakage*.
6. Attention masking pada RCNN pooling.
7. Peniadaan rumus polinomial aproksimasi HAZ/WAZ.
8. Penanganan missing data non-destruktif pada CURE.
9. Validasi CURE (dimensi, penolakan NaN/Inf, dataset kecil, fitur konstan).
10. Silhouette score pada matriks fitur terskala.
11. Penolakan evaluasi diri (*self-evaluation*) dan placeholder error pada klasifikasi.

### B. Menjalankan Server Aplikasi
```bash
# Antarmuka Web Interaktif Next.js (Port 3000):
npm run dev

# Backend REST API Flask (Port 5000):
python3 app.py
```

### C. Melatih Model via API
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Balita tampak sangat kurus KMS pita merah",
      "Pertumbuhan anak stabil di kurva hijau KMS",
      "Anak sangat pendek asupan protein minim",
      "Balita aktif gizi baik nafsu makan optimal",
      "Berat badan dua bulan tidak naik garis kuning",
      "Tinggi badan sesuai usia motorik baik",
      "LiLA 11.2 cm asupan gizi makro kurang",
      "Pertumbuhan optimal imunisasi lengkap",
      "Gejala stunting sanitasi air tercemar",
      "Balita sehat pola makan seimbang"
    ],
    "labels": [
      "STUNTING", "NORMAL", "STUNTING", "NORMAL",
      "STUNTING", "NORMAL", "STUNTING", "NORMAL",
      "STUNTING", "NORMAL"
    ],
    "epochs": 3,
    "batch_size": 4,
    "lr": 0.00002
  }'
```

---

## 6. Transparansi & Batasan Metodologi Penelitian
1. **Model Checkpoint**: Pelabelan otomatis klaster dan inferensi neural membutuhkan ketersediaan bobot latih `.pt`. Jika belum tersedia, sistem transparan mengembalikan status belum dilatih tanpa memalsukan tebakan kata kunci.
2. **Evaluasi Terpisah**: Evaluasi klasterisasi (Silhouette Score geometri) dipisahkan secara tegas dari evaluasi pelabelan semantik klaster (kemurnian klaster / purity).
3. **Data Demonstrasi**: Berkas contoh pada `/sample-dataset` ditegaskan sebagai **Data Sintetis untuk Pengujian Perangkat Lunak**, bukan data klinis riil.
4. **Rekomendasi Intervensi**: Berupa panduan gizi masyarakat umum berbasis Permenkes RI No. 2/2020, bukan vonis diagnosis atau resep medis perseorangan.

---

## 7. Reproduksi Eksperimen Skripsi (folder `research/`)

Data e-PPGBM bersifat pribadi: **jangan di-commit** (`*.xlsx`, `research/out/`, dan `checkpoints/` sudah ada di `.gitignore`).

```bash
python research/prepare_data.py "BY NAME BY ADDRESS 2023-2025.xlsx"   # 27.934 rekam bersih + catatan naratif + label acuan 7 kelas
python research/exp_cure.py 14          # grid CURE (k 2-7, c 5/10, alpha 0,1-0,7) + K-Means, per lembar
python research/exp_cure_detail.py      # profil konfigurasi kandidat
python research/exp_time.py             # waktu tahap hierarkis vs ukuran sampel
python research/exp_indobert.py 3       # IndoBERT-RCNN, TF-IDF+LR, ablasi [CLS] (3 epoch)
python research/exp_indobert.py 8 rcnn  # IndoBERT-RCNN 8 epoch
python research/finalize_model.py       # pilih checkpoint akhir
python research/exp_final.py 6 5 0.5    # pipeline lengkap + evaluasi label klaster
python research/blackbox.py             # uji black box (Flask :5000 + Next.js :3000)
```
