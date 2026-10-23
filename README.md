# CURE & IndoBERT + RCNN Pipeline for Child Stunting Analysis

Platform penelitian terintegrasi untuk klasterisasi dan klasifikasi data stunting balita di Indonesia menggunakan metode **CURE (Clustering Using Representatives)** dan arsitektur neural **IndoBERT + RCNN**, berbasis standar **Permenkes RI No. 2 Tahun 2020** dan **WHO Child Growth Standards 2006**.

---

## 1. Arsitektur & Metodologi Penelitian

### A. CURE (Clustering Using Representatives)
- **Representasi Klaster**: Setiap klaster direpresentasikan oleh sejumlah $c$ titik perwakilan (*representative points*) yang tersebar secara maksimal (*well-scattered*).
- **Penyusutan (*Shrinkage*)**: Titik perwakilan disusutkan ke arah titik pusat (*centroid*) klaster sebesar faktor $\alpha \in [0, 1]$:
  $$p' = p + \alpha \cdot (\mu - p)$$
- **Jarak Antar-Klaster**: Didefinisikan secara ketat sebagai jarak minimum antara pasangan titik perwakilan yang telah disusutkan:
  $$d(C_i, C_j) = \min_{p \in Rep(C_i), q \in Rep(C_j)} \|p - q\|_2$$
- **Fitur Masukan**: Matriks antropometri balita terstandarisasi:
  - Height-for-Age Z-Score ($\text{HAZ} / \text{TB/U}$)
  - Weight-for-Age Z-Score ($\text{WAZ} / \text{BB/U}$)
  - Weight-for-Height Z-Score ($\text{WHZ} / \text{BB/TB}$)
  - Usia Balita Ternormalisasi ($\text{Usia (Bulan)} / 60$)
- **Keunggulan CURE**: Tahan terhadap *outlier* dan mampu mendeteksi bentuk klaster non-sferis yang tidak dapat dipisahkan secara optimal oleh K-Means biasa.

### B. Arsitektur Neural IndoBERT + RCNN
- **Representasi Kontekstual**: IndoBERT (`indobenchmark/indobert-base-p1`) mengekstraksi representasi vektor 768 dimensi dari catatan klinis/keluhan bidan/posyandu.
- **Komponen Rekuren**: Bidirectional LSTM (256 unit per arah = 512 total) menangkap ketergantungan sekuensial narasi klinis ke depan dan ke belakang.
- **Komponen Konvolusional**: 1D Convolution dengan adaptive max-pooling untuk mengekstraksi fitur n-gram klinis paling relevan.
- **Kepala Klasifikasi**: Dense layer dengan aktivasi ReLU, Dropout (0.3), dan Softmax.
- **Taksonomi Label Konsisten**:
  - `0`: `STUNTING_BERAT_KOMORBID` (Stunting Berat dengan Komorbiditas Klinis)
  - `1`: `STUNTING_KRONIS` (Stunting Kronis / Sangat Pendek, HAZ < -3 SD)
  - `2`: `STUNTING_SEDANG` (Stunting Sedang / Pendek, -3 SD <= HAZ < -2 SD)
  - `3`: `GIZI_BURUK_AKUT` (Gizi Buruk Akut / Wasting Berat, WHZ < -3 SD)
  - `4`: `GIZI_KURANG_RISIKO` (Gizi Kurang / Risiko Gagal Tumbuh)
  - `5`: `NORMAL_OPTIMAL` (Pertumbuhan Normal / Gizi Optimal)
  - `6`: `RISIKO_GIZI_LEBIH` (Risiko Gizi Lebih / Overweight)
- **Prinsip Transparansi**: Sistem tidak menggunakan tebakan heuristik kata kunci yang disamarkan sebagai prediksi model. Jika *checkpoint* belum dilatih/tersedia, sistem secara transparan memberikan pesan kesalahan `ModelNotTrainedError`.

### C. Standar Antropometri Permenkes RI No. 2/2020 & WHO 2006
- Klasifikasi status gizi berbasis bukti medis yang dipisahkan secara tegas dari output model neural:
  - $\text{HAZ} < -3\,\text{SD}$: Sangat Pendek (*Severely Stunted*)
  - $-3\,\text{SD} \le \text{HAZ} < -2\,\text{SD}$: Pendek (*Stunted*)
  - $-2\,\text{SD} \le \text{HAZ} \le +3\,\text{SD}$: Normal
  - $\text{HAZ} > +3\,\text{SD}$: Tinggi

---

## 2. Struktur Modul Proyek

```text
├── cure.py               # Algoritma CURE otentik (titik perwakilan, shrinkage, hierarchical merge)
├── indobert_rcnn.py      # Arsitektur PyTorch IndoBERT + RCNN, dataset, training, inference
├── pipeline.py           # Pipeline modular: pembacaan Excel, normalisasi, CURE & inferensi
├── evaluation.py         # Metrik evaluasi: Accuracy, Precision, Recall, F1, Confusion Matrix, Silhouette Score
├── app.py                # Backend REST API Flask untuk pemrosesan file & endpoint training
├── lib/
│   └── stunting-pipeline.ts  # Implementasi CURE & antropometri client-side untuk dashboard interaktif
├── app/
│   ├── page.tsx          # Antarmuka web Next.js App Router (upload Excel, preview, filter, download)
│   └── layout.tsx        # Layout root aplikasi
├── tests/
│   └── test_pipeline.py  # 14 unit tests otomatis (CURE, IndoBERT contract, parsing, metrik)
├── templates/            # Template antarmuka web Flask (index.html, result.html)
└── requirements.txt      # Dependensi Python untuk deployment lingkungan server
```

---

## 3. Instalasi dan Persiapan Lingkungan

### A. Menjalankan Dashboard Web Next.js (Port 3000)
Aplikasi antarmuka web telah terintegrasi secara interaktif:
```bash
npm run dev
# Atau untuk build produksi:
npm run build
npm run start
```
Buka browser pada port `3000` untuk menggunakan antarmuka interaktif pemrosesan data Excel.

### B. Menjalankan Lingkungan Python & PyTorch
```bash
# Instal dependensi Python
pip install -r requirements.txt

# Menjalankan server Flask API
python3 app.py
```

---

## 4. Pelatihan Model & Inferensi IndoBERT + RCNN

### A. Melatih Model dengan Data Terlabel
Endpoint API disediakan pada `/api/train`:
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Balita tampak sangat kurus, KMS garis merah, riwayat diare berulang",
      "Pertumbuhan anak stabil di kurva hijau, motorik optimal, ASI eksklusif",
      "Anak sangat pendek untuk usianya, TB/U -3.4 SD, asupan protein kurang"
    ],
    "labels": [0, 5, 1],
    "epochs": 3,
    "batch_size": 8,
    "lr": 0.00002
  }'
```
Model yang telah dilatih otomatis disimpan ke berkas `checkpoints/indobert_rcnn_stunting.pt`.

### B. Pengujian Otomatis (Automated Tests)
Jalankan rangkaian tes verifikasi:
```bash
python3 -m unittest tests/test_pipeline.py
```
Semua 14 tes mencakup:
- Validasi parsing angka format desimal koma Indonesia (`-2,45`, `12,8 kg`, `78.5 cm`)
- Normalisasi skema kolom multi-sheet Posyandu/e-PPGBM
- Geometri penyusutan (*shrinkage*) dan pemilihan titik perwakilan CURE
- Penanganan dataset kecil ($N \le k$) dan fitur konstan
- Kontrak pengecualian `ModelNotTrainedError` saat checkpoint tidak ada
- Perhitungan metrik evaluasi klasifikasi dan Silhouette Score klasterisasi.

---

## 5. Batasan dan Status Penelitian (Research Transparency)
1. **Model Checkpoint**: Model IndoBERT + RCNN membutuhkan dataset terlabel riil untuk mencapai bobot terverifikasi. Sistem tidak merekayasa hasil evaluasi dan menolak tebakan heuristik kata kunci.
2. **Evaluasi Klasifikasi**: Metrik klasifikasi (Akurasi, F1, Confusion Matrix) hanya dihitung apabila berkas Excel memuat kolom *ground truth* independen (misalnya `Label_Manual` atau `Diagnosis_Dokter`).
3. **Rekomendasi Intervensi**: Saran gizi dan medis disajikan sebagai panduan berbasis standar Permenkes, bukan sebagai kesimpulan diagnosis klinis mandiri.
