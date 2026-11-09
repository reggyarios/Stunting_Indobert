# Menjalankan & Deploy

Aplikasi punya dua bagian:

- **Backend Flask** (`app.py`, port 5000) menjalankan CURE (`cure.py`) dan IndoBERT-RCNN dari checkpoint `checkpoints/indobert_rcnn_stunting.pt`. Endpoint untuk front end: `POST /api/analyze`.
- **Front end Next.js** (`app/`, `lib/`). Bila `NEXT_PUBLIC_API_URL` diset, berkas dikirim ke backend. Bila tidak diset, analisis berjalan di browser dalam **mode tanpa model**: CURE (port TypeScript dari `cure.py`) ditambah **Kategori Permenkes (aturan ambang)**, tanpa label IndoBERT. Mode yang aktif selalu ditampilkan sebagai badge di halaman.

## 1. Lokal

```bash
# Backend (Python 3.11+; torch CPU sudah cukup)
pip install -r requirements.txt
python app.py                                   # http://localhost:5000  (cek: /api/health)

# Front end, di terminal lain
npm install
NEXT_PUBLIC_API_URL=http://localhost:5000 npm run dev      # http://localhost:3000
# PowerShell: $env:NEXT_PUBLIC_API_URL="http://localhost:5000"; npm run dev
```

Tanpa `NEXT_PUBLIC_API_URL`, `npm run dev` berjalan dalam mode tanpa model.
Tes: `python -m unittest discover tests/` (tes port CURE TypeScript butuh Node ≥ 22.6).

Parameter `POST /api/analyze` (multipart): `file` (.xlsx/.xls), `n_clusters`, `n_representatives`, `shrink_factor`, `sample_size`, `reference_date` (YYYY-MM-DD). `reference_date` adalah tanggal ukur yang dipakai untuk menghitung usia bila berkas hanya punya `Tgl Lahir`.

## 2. Backend di Hugging Face Spaces (Docker, CPU)

1. Buat Space baru: **SDK = Docker**, hardware **CPU basic**.
2. Isi repo Space dengan: `Dockerfile`, `requirements.txt`, semua `*.py` (`app.py`, `cure.py`, `indobert_rcnn.py`, `pipeline.py`, `evaluation.py`), folder `templates/`, dan `checkpoints/indobert_rcnn_stunting.pt`.
   Checkpoint (~500 MB) wajib diunggah lewat Git LFS (`git lfs track "*.pt"`) atau `huggingface-cli upload <user>/<space> checkpoints/indobert_rcnn_stunting.pt checkpoints/indobert_rcnn_stunting.pt`.
   **Jangan** ikut mengunggah berkas data (`*.xlsx`).
3. Tambahkan front matter ini di bagian atas `README.md` Space:
   ```yaml
   ---
   title: CURE IndoBERT Stunting API
   sdk: docker
   app_port: 7860
   ---
   ```
4. Di *Settings → Variables*, set `ALLOWED_ORIGINS=https://cure-indobert.vercel.app` (bisa lebih dari satu, dipisah koma).
5. Setelah build selesai, cek `https://<user>-<space>.hf.space/api/health`. Hasilnya harus `"has_trained_checkpoint": true`.

Catatan: CURE hierarkis di Python lambat untuk `sample_size` besar. Gunicorn diberi timeout 30 menit, tetapi gunakan sampel kecil (mis. 300–800) di CPU basic.

## 3. Front end di Vercel

*Project → Settings → Environment Variables*: tambahkan `NEXT_PUBLIC_API_URL=https://<user>-<space>.hf.space` (tanpa `/` di akhir), lalu **Redeploy**. Variabel ini ditanam saat build, jadi perubahan baru berlaku setelah redeploy.

## 4. Privasi data (UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi)

- Data anak termasuk **data pribadi yang bersifat spesifik**. Unggah hanya data yang sudah **dianonimkan**: hapus Nama, NIK, alamat, dan nama orang tua. Kolom-kolom itu juga dibuang otomatis oleh API dan UI, tetapi jangan mengandalkannya.
- Backend tidak menyimpan berkas unggahan. Salinan sementara dihapus setelah diproses, dan log tidak memuat nama. Space publik tetap dapat dipanggil siapa pun, jadi jangan unggah data yang belum dianonimkan.
- Jangan menyimpan data balita di repositori GitHub atau Space. Bila pernah ter-*commit*, hapus dari riwayat git dan jadikan repositori privat.
- Rekomendasi gizi di aplikasi adalah panduan umum berbasis Permenkes No. 2/2020, bukan diagnosis medis.
