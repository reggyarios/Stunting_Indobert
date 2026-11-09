"""
Eksperimen grid parameter CURE (cure.py repo ini) per lembar data, dengan sampling
yang sama persis seperti StuntingResearchPipeline.process_excel:
  fitur [HAZ, WAZ, WHZ, usia/60] -> sampel acak sample_size -> fit -> predict seluruh data.
Pembanding: K-Means (scikit-learn) dengan k yang sama pada fitur terstandardisasi.
Keluaran: research/out/cure_grid.csv

Jalankan: python research/exp_cure.py [jumlah_proses]
"""
from __future__ import annotations

import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cure import CUREClusterer  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
STUNT = {"STUNTING_BERAT_KOMORBID", "STUNTING_KRONIS", "STUNTING_SEDANG"}


def load(sheet: str):
    d = pd.read_excel(OUT / "data_bersih.xlsx", sheet_name=sheet)
    X = np.c_[d["ZS TB/U"], d["ZS BB/U"], d["ZS BB/TB"], d["Usia_Bulan"] / 60.0].astype(float)
    return d, X


def summarize(name, sheet, k, c, a, seed, X, Xs_sample, lab_sample, lab_all, y, secs):
    sizes = np.bincount(lab_all, minlength=k)
    st = np.isin(y, list(STUNT))
    purity = sum(pd.Series(y[lab_all == j]).value_counts().iloc[0] for j in range(k) if sizes[j]) / len(y)
    prev = [st[lab_all == j].mean() if sizes[j] else 0 for j in range(k)]
    ok = len(set(lab_sample)) > 1
    return {
        "metode": name, "lembar": sheet, "k": k, "c": c, "alpha": a, "seed": seed,
        "silhouette": silhouette_score(Xs_sample, lab_sample) if ok else np.nan,
        "dbi": davies_bouldin_score(Xs_sample, lab_sample) if ok else np.nan,
        "chi": calinski_harabasz_score(Xs_sample, lab_sample) if ok else np.nan,
        "min_pct": 100 * sizes.min() / len(y), "max_pct": 100 * sizes.max() / len(y),
        "purity_7kelas": purity,
        "stunting_terkonsentrasi_pct": 100 * max(prev),
        "detik": secs,
    }


def run_cure(args):
    sheet, k, c, a, seed, n_sample = args
    d, X = load(sheet)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), n_sample, replace=False))
    t = time.time()
    m = CUREClusterer(n_clusters=k, n_representatives=c, shrink_factor=a).fit(X[idx].tolist())
    lab_all = np.asarray(m.predict(X.tolist()))
    secs = time.time() - t
    return summarize("CURE", sheet, k, c, a, seed, X, np.asarray(m.X_scaled_), np.asarray(m.labels_),
                     lab_all, d["Label_Acuan"].to_numpy(), secs)


def run_kmeans(sheet, k, seed, n_sample):
    d, X = load(sheet)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), n_sample, replace=False))
    t = time.time()
    km = KMeans(k, n_init=10, random_state=seed).fit(Z[idx])
    lab_all = km.predict(Z)
    return summarize("K-Means", sheet, k, 0, np.nan, seed, X, Z[idx], km.labels_, lab_all,
                     d["Label_Acuan"].to_numpy(), time.time() - t)


if __name__ == "__main__":
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_sample = 1500
    jobs = [(s, k, c, a, 42, n_sample) for s, k, c, a in itertools.product(
        ("2024", "2025"), range(2, 8), (5, 10), (0.1, 0.3, 0.5, 0.7))]
    t0 = time.time()
    with Pool(procs) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(run_cure, jobs), 1):
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['lembar']} k={r['k']} c={r['c']} a={r['alpha']} "
                  f"sil={r['silhouette']:.3f} min%={r['min_pct']:.2f} {r['detik']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "cure_grid.csv", index=False)
    rows += [run_kmeans(s, k, 42, n_sample) for s in ("2024", "2025") for k in range(2, 8)]
    pd.DataFrame(rows).sort_values(["metode", "lembar", "k", "c", "alpha"]).to_csv(OUT / "cure_grid.csv", index=False)
    print("selesai", round(time.time() - t0), "detik")
