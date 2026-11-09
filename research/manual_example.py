"""Contoh perhitungan manual CURE (subbab 3.9) dengan logika cure.py, k=3, c=2, alpha=0,3."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cure import CUREClusterer  # noqa: E402

OUT = ROOT / "research" / "out"
IDS = ["ab7cdb06b8fe", "37835d4764ce", "71d33e00e585", "9f36992c8022",
       "22eae35678c0", "7c08b11595fd", "64b875a04758", "68a6d09a2d73"]
NEW = ["1420e3b61731", "d2faaeabc523"]


def main():
    d = pd.concat([pd.read_excel(OUT / "data_bersih.xlsx", sheet_name=s) for s in ("2024", "2025")])
    d = d.drop_duplicates("ID_Anak").set_index("ID_Anak")
    ex = d.loc[IDS]
    X = np.c_[ex["ZS TB/U"], ex["ZS BB/U"], ex["ZS BB/TB"], ex["Usia_Bulan"] / 60].astype(float)
    k, c, a = 3, 2, 0.3
    m = CUREClusterer(k, c, a).fit(X.tolist())
    mu, sd = np.array(m.feature_means_), np.array(m.feature_stds_)
    Z = np.array(m.X_scaled_)
    D0 = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    # replika loop cure.py untuk mencatat iterasi
    cl = {i: {"idx": [i], "cent": Z[i].tolist(), "reps": [Z[i].tolist()]} for i in range(len(Z))}
    active = list(range(len(Z)))
    cache = {(i, j): float(D0[i, j]) for i in range(len(Z)) for j in range(i + 1, len(Z))}
    nid, iters = len(Z), []
    while len(active) > k:
        pair, dmin = min(cache.items(), key=lambda kv: kv[1])
        u, v = pair
        idx = cl[u]["idx"] + cl[v]["idx"]
        su, sv = len(cl[u]["idx"]), len(cl[v]["idx"])
        cent = [(su * cl[u]["cent"][j] + sv * cl[v]["cent"][j]) / (su + sv) for j in range(4)]
        reps = m._select_and_shrink_reps([Z[i].tolist() for i in idx], cent, c, a)
        iters.append({"gabung": [sorted(cl[u]["idx"]), sorted(cl[v]["idx"])], "jarak": dmin,
                      "anggota": sorted(idx), "mean": cent, "reps": reps})
        for x in (u, v):
            active.remove(x); del cl[x]
        cache = {kk: vv for kk, vv in cache.items() if u not in kk and v not in kk}
        cl[nid] = {"idx": idx, "cent": cent, "reps": reps}
        for o in active:
            cache[(min(nid, o), max(nid, o))] = m._cluster_distance(reps, cl[o]["reps"])
        active.append(nid); nid += 1
    labels = [0] * len(Z)
    for f, cid in enumerate(active):
        for i in cl[cid]["idx"]:
            labels[i] = f
    assert labels == list(m.labels_), (labels, m.labels_)
    nb = d.loc[NEW]
    XB = np.c_[nb["ZS TB/U"], nb["ZS BB/U"], nb["ZS BB/TB"], nb["Usia_Bulan"] / 60].astype(float)
    ZB = (XB - mu) / sd
    distB = [[[float(np.linalg.norm(zb - np.array(r))) for r in reps] for reps in m.representatives_] for zb in ZB]
    from sklearn.metrics import silhouette_samples
    sil = silhouette_samples(Z, m.labels_) if len(set(m.labels_)) > 1 else []
    res = {
        "rekam": ex.reset_index()[["ID_Anak", "JK", "Usia_Bulan", "Tinggi", "Berat", "ZS TB/U", "ZS BB/U", "ZS BB/TB", "TB/U", "Label_Acuan", "Catatan_Pemeriksaan"]].to_dict("records"),
        "mean": mu.tolist(), "std": sd.tolist(), "Z": Z.tolist(), "D0": D0.tolist(),
        "iterasi": iters, "labels": list(map(int, m.labels_)), "reps": m.representatives_,
        "baru": nb.reset_index()[["JK", "Usia_Bulan", "Tinggi", "Berat", "ZS TB/U", "ZS BB/U", "ZS BB/TB", "TB/U", "Label_Acuan"]].to_dict("records"),
        "ZB": ZB.tolist(), "distB": distB, "pred_baru": [int(x) for x in m.predict(XB.tolist())],
        "silhouette": [float(s) for s in sil],
    }
    json.dump(res, open(OUT / "manual_example.json", "w"), indent=1, default=float)
    np.set_printoptions(precision=4, suppress=True)
    print("mean", mu, "std", sd); print(Z); print(np.round(D0, 3))
    for it in iters:
        print(it["gabung"], round(it["jarak"], 4), np.round(it["reps"], 4).tolist())
    print("labels", labels, "pred baru", res["pred_baru"], [[[round(x, 4) for x in r] for r in b] for b in distB])
    print("sil", np.round(sil, 4), np.mean(sil))


if __name__ == "__main__":
    main()
