# CURE & IndoBERT + RCNN Research Pipeline for Child Stunting Analysis

Platform penelitian terintegrasi untuk klasterisasi dan klasifikasi data stunting balita di Indonesia menggunakan metode **CURE (Clustering Using Representatives)** dan arsitektur neural **IndoBERT + RCNN**, diselaraskan dengan standar **Permenkes RI No. 2 Tahun 2020** dan **WHO Child Growth Standards 2006**.

---

## 1. Ringkasan Metodologi & Arsitektur Penelitian

### A. Klasterisasi CURE (Clustering Using Representatives)
- **Representasi Klaster**: Setiap klaster direpresentasikan oleh sejumlah $c$ titik perwakilan (*representative points*) yang dipilih secara maksimal menyebar (*well-scattered*) menggunakan algoritma *max-min distance*.
- **Penyusutan (*Shrinkage*)**: Titik-titik perwakilan disusutkan ke arah titik pusat (*centroid*) klaster sebesar faktor $\alpha \in [0, 1]$:
  $$p' = p + \alpha \cdot (\mu - p)$$
- **Jarak Antar-Klaster**: Dihitung secara ketat sebagai jarak Euclidean minimum antara pasangan titik perwakilan tersusut antar-dua klaster:
  $$d(C_i, C_j) = \min_{p \in Rep(C_i), q \in Rep(C_j)} \|p - q\|_2$$
- **Penanganan Nilai Hilang Non-Destruktif**:
  - Matriks fitur CURE mensyaratkan 4 dimensi antropometri lengkap: `[HAZ, WAZ, WHZ, normalized_age]`.
  - Data yang kehilangan salah satu nilai indikator **tidak disubstitusi secara sembarangan** dengan angka $0.0$ atau $0.5$.
  - Observasi dengan data tidak lengkap dipisahkan dan diberi label `Cluster_ID = -1`, sehingga geometri klaster tidak terkontaminasi imputasi artifisial.
- **Konsistensi Ruang Evaluasi**:
  - Evaluasi Silhouette Score dihitung secara eksak pada matriks fitur terstandarisasi (`X_scaled`) yang dihasilkan oleh CURE, bukan pada matriks mentah.

### B. Arsitektur Neural IndoBERT + RCNN
- **Representasi Kontekstual**: IndoBERT (`indobenchmark/indobert-base-p1`) mengekstraksi representasi vektor 768 dimensi dari catatan klinis/keluhan posyandu.
- **Komponen Rekuren**: *Bidirectional LSTM* (256 unit/arah = 512 total) menangkap dependensi sekuensial narasi klinis.
- **Komponen Konvolusional & Masking**: *1D Convolution* (kernel size 3) dengan *attention-masked max-over-time pooling*. Posisi *padding tokens* dimasking ke $-10^9$ sebelum pooling, menjamin token padding tidak mendominasi atau mempengaruhi aktivasi fitur.
- **Kepala Klasifikasi**: Dense layers dengan ReLU, *Dropout* (0.3), dan Softmax.
- **Taksonomi yang Dapat Dikonfigurasi**:
  - Kelas target dapat dikonfigurasi (`label2id`) sesuai definisi operasional penelitian (misal: biner stunting vs normal, 3 kelas, atau multi-kelas).
  - Sistem menolak label tidak dikenal dan tidak memalsukan prediksi menggunakan heuristik kata kunci.

### C. Validasi Antropometri Permenkes RI No. 2/2020
- **Prioritas Data**:
  1. Membaca Z-score resmi yang tercatat pada dataset (rentang biologis $-6.0 \le Z \le +6.0$).
  2. Apabila Z-score tidak tersedia, indikator ditandai `None` (tidak tersedia).
  3. **Persamaan polinomial aproksimasi HAZ/WAZ telah dihapus sepenuhnya** untuk mencegah klaim keliru terhadap standar WHO.

---

## 2. Struktur Modul Proyek

```text
├── cure.py               # Algoritma CURE otentik, validasi dimensi & NaN/Inf, penanganan dataset kecil
├── indobert_rcnn.py      # Arsitektur PyTorch IndoBERT+RCNN, masking pooling, split data, training & validasi
├── pipeline.py           # Pipeline modular: pembacaan Excel, validasi non-destruktif, eksekusi terpisah
├── evaluation.py         # Metrik evaluasi eksak: Akurasi, Precision, Recall, F1, Silhouette pada ruang terskala
├── app.py                # Server Flask: REST API, endpoint training, sanitasi PII, demo synthetic disclaimer
├── lib/
│   └── stunting-pipeline.ts  # Implementasi CURE & antropometri client-side untuk dashboard interaktif
├── app/
│   ├── page.tsx          # Antarmuka web Next.js App Router (upload Excel, preview, filter, download)
│   └── layout.tsx        # Layout root aplikasi
├── tests/
│   └── test_pipeline.py  # 22 unit tests otomatis mencakup seluruh kriteria metodologi penelitian
└── requirements.txt      # Dependensi Python untuk lingkungan server/GPU
```

---

## 3. Alur Pelatihan, Validasi, dan Pengujian Model

### A. Pencegahan Data Leakage & Pembagian Dataset
Fungsi `prepare_dataset_splits()`:
- Memvalidasi keselarasan jumlah teks dan label.
- Mendeteksi teks duplikat dan menempatkannya ke dalam kelompok partisi yang sama untuk mencegah kebocoran data (*data leakage*) antara train, val, dan test.
- Melakukan pembagian terstratifikasi (*stratified split*) sesuai proporsi (misal 70% train, 15% val, 15% test).
- Menolak dataset yang terlalu kecil (minimal 10 sampel) agar evaluasi memiliki dasar statistik yang memadai.

### B. Pelatihan dengan Pemilihan Checkpoint Terbaik
Metode `IndoBERTRCNNPipeline.train()`:
- Menjalankan iterasi epoch dengan penghitungan loss latih (*train loss*).
- Mengevaluasi set validasi pada setiap epoch (*validation loss* dan *accuracy*).
- Menyimpan bobot terbaik (*best checkpoint*) berdasarkan metrik validasi terendah/terbaik.
- Menyimpan metadata eksperimen lengkap: bobot model, pemetaan label, rasio split, dan riwayat pelatihan ke `indobert_rcnn_stunting.pt`.

---

## 4. Perintah Eksekusi

### A. Menjalankan Tes Otomatis (Automated Tests)
```bash
python3 -m unittest tests/test_pipeline.py
```
Semua 22 pengujian mencakup:
1. Validasi input pelatihan dan keselarasan teks-label.
2. Penolakan label tidak dikenal dan dataset tidak memadai.
3. Pembagian terstratifikasi bebas *data leakage*.
4. Attention masking pada RCNN pooling.
5. Peniadaan rumus polinomial aproksimasi HAZ/WAZ.
6. Penanganan missing data non-destruktif pada CURE.
7. Validasi CURE (dimensi, penolakan NaN/Inf, dataset kecil, fitur konstan).
8. Silhouette score pada matriks fitur terskala.
9. Penolakan evaluasi diri (*self-evaluation*) dan placeholder error pada klasifikasi.
10. Konfigurasi modular tahapan pipeline penelitian.

### B. Menjalankan Server Aplikasi
```bash
# Frontend Next.js (Port 3000):
npm run dev

# Backend Flask (Port 5000):
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

## 5. Transparansi & Batasan Metodologi Penelitian
1. **Model Checkpoint**: Inferensi model neural mewajibkan adanya berkas bobot latih `.pt`. Jika belum tersedia, sistem transparan mengembalikan `ModelNotTrainedError` tanpa memalsukan tebakan heuristik.
2. **Evaluasi Klasifikasi**: Metrik klasifikasi (Akurasi, F1, Confusion Matrix) hanya dihitung apabila berkas memuat kolom referensi independen (*ground truth*).
3. **Data Demonstrasi**: Berkas contoh pada `/sample-dataset` ditegaskan sebagai **Data Sintetis untuk Pengujian Perangkat Lunak**, bukan data klinis riil lapangan.
4. **Rekomendasi Intervensi**: Berupa panduan gizi masyarakat umum berbasis Permenkes RI No. 2/2020, bukan vonis diagnosis atau resep medis perseorangan.
