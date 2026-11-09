"""Waktu tahap hierarkis cure.py terhadap ukuran sampel (lembar 2024, seed 42)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cure import CUREClusterer
OUT = Path(__file__).resolve().parent / "out"
d = pd.read_excel(OUT / "data_bersih.xlsx", sheet_name="2024")
X = np.c_[d["ZS TB/U"], d["ZS BB/U"], d["ZS BB/TB"], d["Usia_Bulan"] / 60].astype(float)
res = []
for n in (250, 500, 1000, 1500, 2000):
    idx = np.sort(np.random.default_rng(42).choice(len(X), n, replace=False))
    t = time.perf_counter(); m = CUREClusterer(6, 5, 0.5).fit(X[idx].tolist()); fit = time.perf_counter() - t
    t = time.perf_counter(); m.predict(X.tolist()); pred = time.perf_counter() - t
    res.append({"n": n, "fit": fit, "predict_all": pred, "n_all": len(X)}); print(res[-1], flush=True)
json.dump(res, open(OUT / "cure_time.json", "w"), indent=1)
