"""Diagram Bab 3 (arsitektur, flowchart CURE, pelabelan klaster, arsitektur IndoBERT-RCNN)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, Rectangle

OUT = Path(__file__).resolve().parent / "out" / "fig"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 15})


class D:
    def __init__(self, w, h):
        self.fig, self.ax = plt.subplots(figsize=(w, h))
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.ax.invert_yaxis()

    def box(self, x, y, w, h, t, kind="rect", fill="white", fs=15, bold=False):
        a = self.ax
        if kind == "rect":
            a.add_patch(Rectangle((x, y), w, h, fc=fill, ec="black", lw=1.2))
        elif kind == "round":
            a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h/2}", fc=fill, ec="black", lw=1.2))
        elif kind == "soft":
            a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fill, ec="black", lw=1.2))
        elif kind == "para":
            s = 0.25
            a.add_patch(Polygon([(x + s, y), (x + w, y), (x + w - s, y + h), (x, y + h)], fc=fill, ec="black", lw=1.2))
        elif kind == "diamond":
            a.add_patch(Polygon([(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)], fc=fill, ec="black", lw=1.2))
        elif kind == "doc":
            import numpy as np
            xs = np.linspace(x, x + w, 50)
            ys = y + h - 0.08 * np.sin((xs - x) / w * 2 * np.pi)
            a.add_patch(Polygon([(x, y), (x + w, y)] + list(zip(xs[::-1], ys[::-1])), fc=fill, ec="black", lw=1.2))
        elif kind == "cyl":
            a.add_patch(Ellipse((x + w / 2, y + h - 0.1), w, 0.2, fc=fill, ec="black", lw=1.2, zorder=2))
            a.add_patch(Rectangle((x, y + 0.1), w, h - 0.2, fc=fill, ec="none", zorder=3))
            a.plot([x, x], [y + 0.1, y + h - 0.1], c="black", lw=1.2, zorder=4); a.plot([x + w, x + w], [y + 0.1, y + h - 0.1], c="black", lw=1.2, zorder=4)
            a.add_patch(Ellipse((x + w / 2, y + 0.1), w, 0.2, fc=fill, ec="black", lw=1.2, zorder=4))
        a.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, zorder=5, weight="bold" if bold else "normal",
               linespacing=1.25)
        return (x, y, w, h)

    def arrow(self, pts, label=None, lx=0.08, ly=0, dashed=False, fs=14):
        a = self.ax
        for p, q in zip(pts[:-2], pts[1:-1]):
            a.plot([p[0], q[0]], [p[1], q[1]], c="black", lw=1.2, ls="--" if dashed else "-")
        a.annotate("", xy=pts[-1], xytext=pts[-2],
                   arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, ls="--" if dashed else "-", mutation_scale=16))
        if label:
            mx, my = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
            a.text(mx + lx, my + ly, label, fontsize=fs, va="center", ha="left")

    def text(self, x, y, t, **kw):
        self.ax.text(x, y, t, **kw)

    def save(self, name):
        self.fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(self.fig)


def vflow(d, x, w, items, y0, gap=0.55, h=0.75):
    """Kotak bertumpuk vertikal; items: (teks, jenis, label_panah_berikutnya)."""
    y = y0
    boxes = []
    for i, (t, k, lab) in enumerate(items):
        hh = h * (1 + t.count("\n") * 0.45)
        boxes.append(d.box(x, y, w, hh, t, k))
        if i < len(items) - 1:
            d.arrow([(x + w / 2, y + hh), (x + w / 2, y + hh + gap)], lab)
        y += hh + gap
    return boxes, y


def arsitektur():
    d = D(15.5, 15.9)
    ax = d.ax
    for x0, t in ((0.2, "Fase A: Pengembangan Model\n(offline, Python)"), (8.1, "Fase B: Penggunaan Sistem\n(aplikasi web Flask + Next.js)")):
        ax.add_patch(FancyBboxPatch((x0, 0.2), 7.3, 15.5, boxstyle="round,pad=0,rounding_size=0.15", fc="white", ec="black", lw=1.2))
        ax.add_patch(Rectangle((x0, 0.2), 7.3, 1.1, fc="#e6e6e6", ec="black", lw=1.2))
        ax.text(x0 + 3.65, 0.75, t, ha="center", va="center", fontsize=16, weight="bold")
    a_items = [
        ("Dataset BNBA\ne-PPGBM (Excel)", "doc", "data mentah"),
        ("Praproses & anonimisasi", "rect", "data bersih"),
        ("Usia, posisi ukur &\nlabel acuan 7 kelas", "rect", "rekam terlabel"),
        ("Penyusunan catatan\nnaratif antropometri", "rect", "teks + label"),
        ("Pembagian data\n70% / 15% / 15%", "rect", "latih, validasi, uji"),
        ("Fine-tuning\nIndoBERT-RCNN", "rect", "bobot terbaik (val. loss)"),
        ("Evaluasi data uji &\nklasterisasi CURE", "rect", "model terbaik"),
        ("Checkpoint .pt\nIndoBERT-RCNN", "cyl", None),
    ]
    bA, _ = vflow(d, 0.7, 4.6, a_items, 1.7, gap=0.62, h=0.8)
    b_items = [
        ("Berkas Excel\ne-PPGBM", "para", "workbook"),
        ("Deteksi header,\nvalidasi kolom & data", "rect", "rekam valid"),
        ("Fitur [HAZ, WAZ,\nWHZ, usia] terstandar", "rect", "fitur"),
        ("Klasterisasi CURE\n(sampel s; k, c, α)", "rect", "ID klaster"),
        ("Catatan naratif &\ninferensi IndoBERT-RCNN", "rect", "probabilitas kelas"),
        ("Label klaster\n(konsensus probabilitas)", "rect", "label + confidence"),
        ("Profil klaster, kategori\nPermenkes & rekomendasi", "rect", "hasil"),
        ("Dasbor & unduh\nExcel hasil", "para", None),
    ]
    bB, _ = vflow(d, 9.2, 5.0, b_items, 1.7, gap=0.62, h=0.8)
    ax.text(10.0, 1.45, "", fontsize=1)
    cy = bA[-1]; tgt = bB[4]
    d.arrow([(cy[0] + cy[2], cy[1] + cy[3] / 2), (7.85, cy[1] + cy[3] / 2), (7.85, tgt[1] + tgt[3] / 2), (tgt[0], tgt[1] + tgt[3] / 2)], dashed=True)
    ax.text(5.6, cy[1] + cy[3] / 2 - 0.35, "model\ndigunakan", fontsize=14, ha="left", va="bottom")
    d.save("g3_2_arsitektur.png")


def flow_cure():
    d = D(15, 15.5)
    L, W = 0.6, 5.6
    d.box(L + 1.3, 0.3, 3.0, 0.8, "Mulai", "round")
    d.arrow([(L + 2.8, 1.1), (L + 2.8, 1.5)])
    d.box(L, 1.5, W, 1.3, "Fitur [HAZ, WAZ, WHZ, usia/60];\nparameter k, c, α, s", "para")
    d.arrow([(L + 2.8, 2.8), (L + 2.8, 3.2)])
    d.box(L, 3.2, W, 1.1, "Standardisasi fitur\n(z-score per fitur)", "rect")
    d.arrow([(L + 2.8, 4.3), (L + 2.8, 4.7)])
    d.box(L + 0.4, 4.7, W - 0.8, 1.6, "n > s ?", "diamond")
    d.arrow([(L + 2.8, 6.3), (L + 2.8, 6.8)], "Ya")
    d.box(L, 6.8, W, 1.1, "Ambil sampel acak\nberukuran s", "rect")
    d.arrow([(L + W - 0.4, 5.5), (L + W + 0.3, 5.5), (L + W + 0.3, 8.45), (L + 2.8, 8.45)], "Tidak", lx=-0.3, ly=-0.3)
    d.arrow([(L + 2.8, 7.9), (L + 2.8, 8.6)])
    d.box(L, 8.6, W, 1.4, "Setiap titik = 1 klaster;\nhitung jarak semua\npasangan (Euclidean)", "rect")
    d.arrow([(L + 2.8, 10.0), (L + 2.8, 10.5)])
    d.box(L + 0.2, 10.5, W - 0.4, 1.8, "Jumlah\nklaster > k ?", "diamond")
    R, RW = 8.4, 5.8
    d.arrow([(L + W - 0.2, 11.4), (7.6, 11.4), (7.6, 0.85), (R, 0.85)], "Ya", lx=-0.1, ly=-0.3)
    items = [("Pilih pasangan klaster\nu, v dengan jarak terkecil", None),
             ("Gabung menjadi w;\nmean tertimbang (3.6)", None),
             ("Pilih c titik tersebar\n(farthest-first, 3.7)", None),
             ("Susutkan titik sebesar α\nke arah mean (3.8)", None),
             ("Hitung jarak w ke semua\nklaster: min antar-titik (3.5)", None)]
    y = 0.3
    for i, (t, _) in enumerate(items):
        d.box(R, y, RW, 1.1, t, "rect")
        if i < len(items) - 1:
            d.arrow([(R + RW / 2, y + 1.1), (R + RW / 2, y + 1.6)])
        y += 1.6
    yb = y - 0.5
    d.arrow([(R + RW / 2, yb), (R + RW / 2, 13.0), (L + 2.8, 13.0), (L + 2.8, 12.3)])
    d.arrow([(L + 0.2, 11.4), (0.25, 11.4), (0.25, 14.3), (L, 14.3)], "Tidak", lx=0.0, ly=0.3)
    d.box(L, 13.7, W, 1.2, "Labeli seluruh data ke titik\nrepresentatif terdekat (3.9)", "rect")
    d.arrow([(L + W, 14.3), (R - 0.2, 14.3)])
    d.box(R - 0.2, 13.7, 3.9, 1.2, "ID klaster &\ntitik representatif", "para")
    d.arrow([(R + 3.7, 14.3), (R + 4.2, 14.3)])
    d.box(R + 4.2, 13.9, 1.9, 0.8, "Selesai", "round")
    d.save("g3_8_flow_cure.png")


def flow_label():
    d = D(15, 13.2)
    L, W = 0.5, 6.2
    d.box(L + 1.6, 0.3, 3.0, 0.8, "Mulai", "round")
    d.arrow([(L + 3.1, 1.1), (L + 3.1, 1.5)])
    d.box(L, 1.5, W, 1.2, "Data valid + ID klaster CURE\n+ checkpoint IndoBERT-RCNN", "para")
    d.arrow([(L + 3.1, 2.7), (L + 3.1, 3.1)])
    d.box(L, 3.1, W, 1.2, "Susun catatan naratif\nsetiap balita (tanpa status)", "rect")
    d.arrow([(L + 3.1, 4.3), (L + 3.1, 4.7)])
    d.box(L, 4.7, W, 1.2, "Tokenisasi WordPiece\n(maks. 128 token)", "rect")
    d.arrow([(L + 3.1, 5.9), (L + 3.1, 6.3)])
    d.box(L, 6.3, W, 1.4, "Inferensi IndoBERT-RCNN:\nP(c | teks) untuk 7 kelas", "rect")
    d.arrow([(L + 3.1, 7.7), (L + 3.1, 8.1)])
    d.box(L, 8.1, W, 1.2, "Label balita = argmax P;\nconfidence = P maksimum", "rect")
    R, RW = 8.3, 6.2
    d.arrow([(L + W, 8.7), (7.5, 8.7), (7.5, 1.1), (R, 1.1)])
    d.box(R, 0.5, RW, 1.2, "Kelompokkan prediksi\nmenurut ID klaster C~j~".replace("~", ""), "rect")
    d.arrow([(R + 3.1, 1.7), (R + 3.1, 2.1)])
    d.box(R, 2.1, RW, 1.5, "Rata-rata probabilitas\nposterior anggota:\nP̄(c | Cj)", "rect")
    d.arrow([(R + 3.1, 3.6), (R + 3.1, 4.0)])
    d.box(R, 4.0, RW, 1.3, "Label klaster = argmax P̄;\nconfidence = P̄ maksimum", "rect")
    d.arrow([(R + 3.1, 5.3), (R + 3.1, 5.7)])
    d.box(R, 5.7, RW, 1.3, "Kemurnian konsensus =\nproporsi suara kelas terpilih", "rect")
    d.arrow([(R + 3.1, 7.0), (R + 3.1, 7.4)])
    d.box(R, 7.4, RW, 1.4, "Bandingkan dengan label acuan\nPermenkes (purity, akurasi)", "rect")
    d.arrow([(R + 3.1, 8.8), (R + 3.1, 9.3)])
    d.box(R, 9.3, RW, 1.2, "Label balita & label klaster\n+ confidence", "para")
    d.arrow([(R + 3.1, 10.5), (R + 3.1, 10.9)])
    d.box(R + 1.6, 10.9, 3.0, 0.8, "Selesai", "round")
    d.save("g3_10_flow_label.png")


def arsitektur_rcnn():
    d = D(16, 17)
    ax = d.ax
    d.box(0.3, 0.2, 15.4, 1.3, "Catatan masukan (rekam A6): “Pengukuran berdiri pada anak laki-laki 43 bulan menunjukkan tinggi 84,3 cm\ndan berat 10,5 kg dengan z-score TB/U -4,07, BB/U -3,20, dan BB/TB -1,13.”", "rect", fill="#efefef", fs=14)
    d.arrow([(8, 1.5), (8, 2.0)])
    d.box(1.5, 2.0, 13, 1.1, "Tokenizer WordPiece IndoBERT (indobenchmark/indobert-base-p1)\n[CLS] + token + [SEP] + [PAD] hingga 128 token; attention mask", "rect", fs=14)
    d.arrow([(8, 3.1), (8, 3.6)])
    ax.add_patch(FancyBboxPatch((0.3, 3.6), 15.4, 3.0, boxstyle="round,pad=0,rounding_size=0.15", fc="#f0f0f0", ec="black", lw=1.2))
    ax.text(0.6, 3.95, "IndoBERT: embedding + 12 lapisan encoder Transformer (hidden 768)", fontsize=15, weight="bold", va="center")
    d.box(0.8, 4.5, 6.6, 1.6, "Lapisan 1–8: dibekukan\n(bobot pra-latih tidak diubah)", "rect", fill="#dcdcdc", fs=14)
    d.box(8.6, 4.5, 6.6, 1.6, "Lapisan 9–12: dilatih ulang\n(multi-head self-attention + FFN)", "rect", fs=14)
    d.arrow([(7.4, 5.3), (8.6, 5.3)])
    d.arrow([(8, 6.6), (8, 7.1)])
    d.box(3.5, 7.1, 9, 1.0, "Representasi token H (L × 768); token padding dinolkan (mask)", "rect", fs=14)
    d.arrow([(5.0, 8.1), (5.0, 8.7)])
    d.box(1.0, 8.7, 6.2, 1.3, "BiLSTM (256 + 256)\nkonteks kiri & kanan, (L × 512)", "rect", fs=14)
    d.arrow([(11.0, 8.1), (11.0, 8.7)])
    d.box(8.6, 8.7, 6.2, 1.3, "Skip: embedding token\nIndoBERT, (L × 768)", "rect", fs=14)
    d.arrow([(4.1, 10.0), (4.1, 10.5), (7.0, 10.5), (7.0, 10.8)])
    d.arrow([(11.7, 10.0), (11.7, 10.5), (9.0, 10.5), (9.0, 10.8)])
    d.box(3.5, 10.8, 9, 1.0, "Konkatenasi [H ; BiLSTM] → (L × 1280)", "rect", fs=14)
    d.arrow([(8, 11.8), (8, 12.2)])
    d.box(3.5, 12.2, 9, 1.0, "Conv1D (256 filter, kernel 3) + ReLU", "rect", fs=14)
    d.arrow([(8, 13.2), (8, 13.6)])
    d.box(3.5, 13.6, 9, 1.0, "Max-pooling sepanjang token (padding di-mask)", "rect", fs=14)
    d.arrow([(8, 14.6), (8, 15.0)])
    d.box(0.3, 15.0, 15.4, 1.0, "Dropout 0,3 → Linear 256→128 + ReLU → Dropout 0,3 → Linear 128→7 → Softmax", "rect", fs=14)
    d.arrow([(8, 16.0), (8, 16.35)])
    ax.text(8, 16.7, "label = argmax p;  confidence = p maksimum  (7 kelas taksonomi status gizi)", ha="center", va="center", fontsize=15)
    d.save("g3_11_rcnn.png")


if __name__ == "__main__":
    arsitektur(); flow_cure(); flow_label(); arsitektur_rcnn()
    print("ok", sorted(p.name for p in OUT.glob("g3_*.png")))
