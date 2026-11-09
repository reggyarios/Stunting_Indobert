"""Berkas uji black box (nama dianonimkan) di research/out/uji/."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import TGL_ACUAN  # noqa: E402

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "BY NAME BY ADDRESS 2023-2025.xlsx")
OUT = Path(__file__).resolve().parent / "out" / "uji"
OUT.mkdir(parents=True, exist_ok=True)


def load(sheet):
    d = pd.read_excel(SRC, sheet_name=sheet, header=1)
    d["Nama"] = [f"Balita {sheet}-{i + 1:05d}" for i in range(len(d))]
    d["Desa/Kel"] = None
    kec = d["Kec"].astype(str).str.strip().str.upper()
    d.insert(4, "Tgl Ukur", pd.to_datetime([TGL_ACUAN.get((sheet, k)) for k in kec]))
    return d


def write(path, sheets, title=True):
    wb = Workbook(); wb.remove(wb.active)
    for name, df in sheets.items():
        ws = wb.create_sheet(name)
        if title:
            ws.append([f"TAHUN {name}"])
        ws.append(list(df.columns))
        for row in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and np.isnan(v)) else (v.to_pydatetime() if hasattr(v, "to_pydatetime") else v) for v in row])
    wb.save(path)


full = {s: load(s) for s in ("2024", "2025")}
rng = np.random.default_rng(7)
samp = {s: d.iloc[np.sort(rng.choice(len(d), 150, replace=False))].reset_index(drop=True) for s, d in full.items()}
write(OUT / "uji_lengkap_anonim.xlsx", full)
write(OUT / "uji_sampel_300_anonim.xlsx", samp)
write(OUT / "uji_satu_sheet_2025_anonim.xlsx", {"2025": samp["2025"]})
write(OUT / "uji_kolom_tidak_sesuai.xlsx", {"Data": pd.DataFrame({"Nama": ["A", "B"], "Alamat": ["x", "y"], "Hobi": ["bola", "lari"]})}, title=False)
write(OUT / "uji_kosong_header_saja.xlsx", {"2025": samp["2025"].iloc[:0]})
write(OUT / "uji_tanpa_zscore_dan_usia.xlsx", {"2025": samp["2025"].drop(columns=["ZS BB/U", "ZS TB/U", "ZS BB/TB", "Tgl Lahir", "Tgl Ukur"])})
bad = samp["2025"].copy().astype(object)
bad.loc[0, "Berat"] = "abc"; bad.loc[1, "ZS TB/U"] = 999.99; bad.loc[2, "Tinggi"] = 5; bad.loc[3, "Tinggi"] = 250
bad.loc[4, "Tgl Lahir"] = "31/02/2023"; bad.loc[5, "ZS TB/U"] = -24.60; bad.loc[6, "ZS BB/TB"] = 7.5
write(OUT / "uji_nilai_tidak_valid.xlsx", {"2025": bad})
samp["2025"].to_csv(OUT / "uji_format_csv.csv", index=False)
(OUT / "uji_dokumen.pdf").write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")
(OUT / "uji_rusak.xlsx").write_bytes(os.urandom(4096))
print(sorted(p.name for p in OUT.iterdir()))
