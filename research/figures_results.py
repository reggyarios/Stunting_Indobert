"""Grafik hasil Bab 4 dari berkas research/out/*.json|csv."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
GRAY = ["#222222", "#666666", "#999999", "#bbbbbb"]
SHORT = {"STUNTING_BERAT_KOMORBID": "Stunting berat\nkomorbid", "STUNTING_KRONIS": "Stunting\nkronis",
         "STUNTING_SEDANG": "Stunting\nsedang", "GIZI_BURUK_AKUT": "Gizi buruk\nakut",
         "GIZI_KURANG_RISIKO": "Gizi kurang", "NORMAL_OPTIMAL": "Normal", "RISIKO_GIZI_LEBIH": "Risiko gizi\nlebih"}


def dec(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def grid():
    g = pd.read_csv(OUT / "cure_grid.csv")
    g["lembar"] = g["lembar"].astype(str)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for col, s in enumerate(("2024", "2025")):
        x = g[g.lembar == s]
        for row, met, lab in ((0, "silhouette", "Silhouette (↑)"), (1, "dbi", "DBI (↓)")):
            ax = axs[row, col]
            for (a, c), st in zip([(0.3, 5), (0.5, 5), (0.7, 5)], ["-o", "-s", "-^"]):
                y = x[(x.metode == "CURE") & (x.alpha == a) & (x.c == c)].sort_values("k")
                ax.plot(y.k, y[met], st, color=GRAY[[0.3, 0.5, 0.7].index(a)], label=f"CURE c=5, α={str(a).replace('.', ',')}")
            km = x[x.metode == "K-Means"].sort_values("k")
            ax.plot(km.k, km[met], "--x", color="#000000", alpha=0.6, label="K-Means")
            ax.set_xlabel("Jumlah klaster k"); ax.set_ylabel(lab); dec(ax)
            ax.set_title(f"Lembar {s}", fontsize=11)
    axs[0, 0].legend(fontsize=9, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "g4_grid.png", dpi=200); plt.close(fig)


def time_fig():
    t = json.load(open(OUT / "cure_time.json"))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot([r["n"] for r in t], [r["fit"] for r in t], "-o", color="#222")
    for r in t:
        ax.annotate(f"{r['fit']:.1f}".replace(".", ","), (r["n"], r["fit"]), textcoords="offset points", xytext=(-8, 8), fontsize=9)
    ax.set_xlabel("Ukuran sampel s"); ax.set_ylabel("Waktu tahap hierarkis (detik)"); dec(ax)
    fig.tight_layout(); fig.savefig(FIG / "g4_time.png", dpi=200); plt.close(fig)


def loss():
    res = {r["model"]: r for r in json.load(open(OUT / "indobert_results.json"))}
    h = res["IndoBERT + RCNN"]["history"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(h["epochs"], h["train_loss"], "-o", color="#222", label="Loss latih")
    ax.plot(h["epochs"], h["val_loss"], "-s", color="#888", label="Loss validasi")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.set_xticks(h["epochs"]); dec(ax)
    ax2 = ax.twinx(); ax2.plot(h["epochs"], h["val_accuracy"], ":^", color="#555", label="Akurasi validasi")
    ax2.set_ylabel("Akurasi validasi"); ax2.spines["top"].set_visible(False)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ",")))
    l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "g4_loss.png", dpi=200); plt.close(fig)


def cm():
    res = {r["model"]: r for r in json.load(open(OUT / "indobert_results.json"))}
    r = res["IndoBERT + RCNN"]
    order = ["STUNTING_BERAT_KOMORBID", "STUNTING_KRONIS", "STUNTING_SEDANG", "GIZI_BURUK_AKUT",
             "GIZI_KURANG_RISIKO", "NORMAL_OPTIMAL", "RISIKO_GIZI_LEBIH"]
    cls = r["classes"]; M = np.array(r["confusion_matrix"])
    keep = [c for c in order if c in cls]
    ix = [cls.index(c) for c in keep]
    M = M[np.ix_(ix, ix)]
    fig, ax = plt.subplots(figsize=(7.2, 6))
    norm = M / np.maximum(M.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(keep)):
        for j in range(len(keep)):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black", fontsize=11)
    lab = [SHORT[c] for c in keep]
    ax.set_xticks(range(len(keep)), lab, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(keep)), lab, fontsize=9)
    ax.set_xlabel("Prediksi IndoBERT-RCNN"); ax.set_ylabel("Label acuan (Permenkes)")
    ax.spines[:].set_visible(True)
    fig.tight_layout(); fig.savefig(FIG / "g4_cm.png", dpi=200); plt.close(fig)


def clusters():
    fs = json.load(open(OUT / "final_summary.json"))
    d = {s: pd.read_excel(OUT / "hasil_pipeline.xlsx", sheet_name=s) for s in ("2024", "2025")}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    marks = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for ax, s in zip(axs, ("2024", "2025")):
        x = d[s]
        prof = {p["cluster_id"]: p for p in next(l for l in fs["lembar"] if l["lembar"] == s)["profil"]}
        for j, cid in enumerate(sorted(x["CURE_Cluster_ID"].unique())):
            m = x["CURE_Cluster_ID"] == cid
            big = m.mean() > 0.5
            ax.scatter(x.loc[m, "ZS TB/U"], x.loc[m, "ZS BB/TB"], s=4 if big else 14, marker=marks[j % 8],
                       color="#bbbbbb" if big else GRAY[j % 3], alpha=0.35 if big else 0.9,
                       label=f"K{cid} (n={num(int(m.sum()))})")
        ax.axvline(-2, color="#000", lw=0.8, ls="--"); ax.axvline(-3, color="#000", lw=0.8, ls=":")
        ax.axhline(-2, color="#000", lw=0.6, ls="--", alpha=0.6)
        ax.set_xlabel("Z-score TB/U (HAZ)"); ax.set_ylabel("Z-score BB/TB (WHZ)"); ax.set_title(f"Lembar {s}", fontsize=11)
        ax.legend(fontsize=8, frameon=False, markerscale=1.5, loc="lower right")
        dec(ax)
    fig.tight_layout(); fig.savefig(FIG / "g4_cluster.png", dpi=200); plt.close(fig)


def num(x):
    return f"{x:,}".replace(",", ".")


if __name__ == "__main__":
    for fn in sys.argv[1:] or ["grid", "time_fig", "loss", "cm", "clusters"]:
        try:
            globals()[fn]()
            print("ok", fn)
        except Exception as e:  # noqa: BLE001
            print("lewati", fn, type(e).__name__, e)
