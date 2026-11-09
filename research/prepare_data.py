"""
Praproses data BNBA e-PPGBM untuk pipeline CURE + IndoBERT-RCNN (repo ini).

Data e-PPGBM tidak memuat kolom usia maupun catatan pemeriksaan, padahal
pipeline.py membutuhkan keduanya. Skrip ini:
  1. membaca lembar 2024 dan 2025 (baris judul = baris ke-2),
  2. membersihkan data bertahap (Tabel 3.8 skripsi) dan mencatat funnel-nya,
  3. menganonimkan nama menjadi ID anak (SHA-256 + salt),
  4. menghitung Usia_Bulan terhadap tanggal acuan pengukuran (Tabel 3.9),
  5. menyusun Catatan_Pemeriksaan (narasi antropometri berbahasa Indonesia),
  6. menurunkan label acuan 7 kelas (taksonomi DEFAULT_LABEL2ID) dari z-score
     menurut ambang Permenkes No. 2/2020.
Keluaran: research/out/data_bersih.xlsx (tanpa nama) dan funnel.csv.

Jalankan: python research/prepare_data.py "BY NAME BY ADDRESS 2023-2025.xlsx"
"""
from __future__ import annotations

import hashlib
import os
import re
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import NOTE_TEMPLATES, build_note, permenkes_reference_label  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SALT = os.environ.get("ID_SALT", "skripsi-reyvo-2026")

# Tanggal acuan pengukuran (median tanggal ukur tersirat per tahun-kecamatan, Tabel 3.9)
TGL_ACUAN = {
    ("2024", "MEDAN KOTA BELAWAN"): "2024-12-13",
    ("2024", "MEDAN LABUHAN"): "2024-02-24",
    ("2024", "MEDAN MARELAN"): "2024-02-10",
    ("2025", "MEDAN KOTA BELAWAN"): "2025-12-17",
    ("2025", "MEDAN LABUHAN"): "2025-12-16",
    ("2025", "MEDAN MARELAN"): "2025-12-15",
}
LABELS_TBU = {"Sangat Pendek", "Pendek", "Normal", "Tinggi"}
LBL_BBU = {"Berat Badan Sangat Kurang", "Berat Badan Kurang", "Sangat Kurang", "Kurang",
           "Normal", "Risiko Lebih", "Risiko BB Lebih", "Risiko Berat Badan Lebih"}
LBL_BBTB = {"Gizi Buruk", "Gizi Kurang", "Gizi Baik", "Normal", "Berisiko Gizi Lebih",
            "Risiko Gizi Lebih", "Gizi Lebih", "Obesitas"}


def catatan(r: pd.Series, rng: random.Random) -> str:
    """Narasi antropometri ala catatan kader (tanpa kata status gizi), lihat pipeline.build_note."""
    ind = {"gender": r["JK"], "age_months": r["Usia_Bulan"], "weight_kg": r["Berat"],
           "height_cm": r["Tinggi"], "position": r["Posisi"],
           "haz": r["ZS TB/U"], "waz": r["ZS BB/U"], "whz": r["ZS BB/TB"]}
    return build_note(ind, rng.randrange(len(NOTE_TEMPLATES)))


def child_id(nama, dob, jk) -> str:
    nama = " ".join(re.findall(r"[a-z0-9]+", str(nama).lower()))  # huruf kecil, tanpa tanda baca
    key = f"{nama}|{dob:%Y-%m-%d}|{jk}|{SALT}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def clean(src: str) -> tuple[pd.DataFrame, list]:
    frames = []
    for s in ("2024", "2025"):
        d = pd.read_excel(src, sheet_name=s, header=1)
        d["Tahun"] = s
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Kec"] = df["Kec"].astype(str).str.strip().str.upper()
    funnel = [("Data mentah", df)]

    def step(name, d):
        funnel.append((name, d))
        return d

    cols = [c for c in df.columns if c not in ("No",)]
    d = step("Duplikat persis", df[~df.duplicated(subset=cols, keep="first")])
    d = d.assign(dob=pd.to_datetime(d["Tgl Lahir"], errors="coerce"))
    d = step("Tanggal lahir tidak valid", d[d["dob"].notna()])
    d = step("Label TB/U tidak valid", d[d["TB/U"].isin(LABELS_TBU) & d["ZS TB/U"].notna()])
    z = d[["ZS BB/U", "ZS TB/U", "ZS BB/TB"]].apply(pd.to_numeric, errors="coerce")
    bad = (z.abs() >= 999).any(axis=1) | z.isna().any(axis=1) | d["BB/U"].isin(["Outlier", "-"]) | d["BB/TB"].isin(["Outlier", "-"])
    d = step("Kode 999,99 / label BB tidak valid", d[~bad])
    z = d[["ZS BB/U", "ZS TB/U", "ZS BB/TB"]].apply(pd.to_numeric, errors="coerce")
    flag = (z["ZS TB/U"].abs() > 6) | (z["ZS BB/U"] < -6) | (z["ZS BB/U"] > 5) | (z["ZS BB/TB"].abs() > 5)
    d = step("Flag WHO", d[~flag])
    d = step("Berat/tinggi di luar rentang", d[d["Berat"].between(1, 40) & d["Tinggi"].between(40, 130)])
    acuan = pd.Series(pd.to_datetime([TGL_ACUAN[(t, k)] for t, k in zip(d["Tahun"], d["Kec"])]), index=d.index)
    d = d.assign(tgl_acuan=acuan, usia_hari=(acuan - d["dob"]).dt.days)
    d = step("Usia di luar 0-60 bulan", d[d["usia_hari"].between(0, 1826)])
    d = d.assign(ID_Anak=[child_id(n, b, j) for n, b, j in zip(d["Nama"], d["dob"], d["JK"])])
    d = step("Anak tercatat ganda (tahun sama)",
             d.sort_values(["Tahun", "No"]).drop_duplicates(subset=["Tahun", "ID_Anak"], keep="first"))
    return d.copy(), funnel


def main(src: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, funnel = clean(src)
    rows = []
    prev = None
    for name, f in funnel:
        n24, n25 = int((f["Tahun"] == "2024").sum()), int((f["Tahun"] == "2025").sum())
        rows.append({"Tahap": name, "Sisa": len(f), "2024": n24, "2025": n25,
                     "Dikeluarkan": 0 if prev is None else prev - len(f)})
        prev = len(f)
    pd.DataFrame(rows).to_csv(OUT / "funnel.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    d["Usia_Bulan"] = (d["usia_hari"] / 30.4375).round(2)
    cara = d["Cara Ukur"] if "Cara Ukur" in d else pd.Series(np.nan, index=d.index)
    d["Posisi"] = np.where(cara.isin(["Berdiri", "Terlentang"]), cara,
                           np.where(d["Usia_Bulan"] < 24, "Terlentang", "Berdiri"))
    for c in ("ZS BB/U", "ZS TB/U", "ZS BB/TB"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    rng = random.Random(42)
    d["Catatan_Pemeriksaan"] = [catatan(r, rng) for _, r in d.iterrows()]
    d["Label_Acuan"] = [permenkes_reference_label(h, w, x) for h, w, x in zip(d["ZS TB/U"], d["ZS BB/U"], d["ZS BB/TB"])]

    keep = ["ID_Anak", "JK", "Usia_Bulan", "Kec", "Pukesmas", "Berat", "Tinggi", "Posisi",
            "BB/U", "ZS BB/U", "TB/U", "ZS TB/U", "BB/TB", "ZS BB/TB", "Catatan_Pemeriksaan", "Label_Acuan"]
    with pd.ExcelWriter(OUT / "data_bersih.xlsx") as w:
        for s in ("2024", "2025"):
            part = d[d["Tahun"] == s][keep].reset_index(drop=True)
            part.insert(0, "No", range(1, len(part) + 1))
            part.to_excel(w, sheet_name=s, index=False)
    print("\nLabel acuan:\n", pd.crosstab(d["Label_Acuan"], d["Tahun"], margins=True))
    print("\nContoh catatan:\n", "\n".join(d["Catatan_Pemeriksaan"].head(3)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "BY NAME BY ADDRESS 2023-2025.xlsx")
