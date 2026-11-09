"""Profil klaster rinci untuk konfigurasi CURE kandidat (label disimpan untuk gambar Bab 4)."""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cure import CUREClusterer  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
STUNT = {"STUNTING_BERAT_KOMORBID", "STUNTING_KRONIS", "STUNTING_SEDANG"}
CANDS = [(4, 5, 0.3), (5, 5, 0.5), (6, 5, 0.5), (7, 5, 0.5), (6, 10, 0.3), (7, 10, 0.3)]


def run(args):
    sheet, k, c, a = args
    d = pd.read_excel(OUT / "data_bersih.xlsx", sheet_name=sheet)
    X = np.c_[d["ZS TB/U"], d["ZS BB/U"], d["ZS BB/TB"], d["Usia_Bulan"] / 60].astype(float)
    idx = np.sort(np.random.default_rng(42).choice(len(X), 1500, replace=False))
    t = time.time()
    m = CUREClusterer(k, c, a).fit(X[idx].tolist())
    lab = np.asarray(m.predict(X.tolist()))
    secs = time.time() - t
    y = d["Label_Acuan"].to_numpy()
    st = np.isin(y, list(STUNT)); ab = y != "NORMAL_OPTIMAL"
    big = np.bincount(lab).argmax()
    prof = []
    for j in range(k):
        mk = lab == j
        prof.append({"j": j, "n": int(mk.sum()), "stunting": int(st[mk].sum()), "abnormal": int(ab[mk].sum()),
                     "haz": float(X[mk, 0].mean()) if mk.any() else None, "waz": float(X[mk, 1].mean()) if mk.any() else None,
                     "whz": float(X[mk, 2].mean()) if mk.any() else None, "usia": float(X[mk, 3].mean() * 60) if mk.any() else None,
                     "kelas": pd.Series(y[mk]).value_counts().to_dict()})
    minor = lab != big
    np.save(OUT / f"lab_{sheet}_k{k}_c{c}_a{a}.npy", lab)
    return {"lembar": sheet, "k": k, "c": c, "alpha": a, "detik": secs, "profil": prof,
            "minor_n": int(minor.sum()), "minor_stunting": int(st[minor].sum()), "stunting_total": int(st.sum()),
            "minor_abnormal": int(ab[minor].sum()), "abnormal_total": int(ab.sum()),
            "reps": m.representatives_, "mean": m.feature_means_, "std": m.feature_stds_}


if __name__ == "__main__":
    jobs = [(s, *c) for s in ("2024", "2025") for c in CANDS]
    with Pool(12) as p:
        res = p.map(run, jobs)
    json.dump(res, open(OUT / "cure_detail.json", "w"), indent=1)
    for r in res:
        print(r["lembar"], r["k"], r["c"], r["alpha"], f"{r['detik']:.0f}s",
              "minor n", r["minor_n"], "stunting", r["minor_stunting"], "/", r["stunting_total"],
              "abn", r["minor_abnormal"], "/", r["abnormal_total"],
              [(p_["n"], p_["stunting"], p_["abnormal"]) for p_ in r["profil"]])
