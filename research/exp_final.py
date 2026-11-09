"""
Pipeline lengkap (pipeline.process_excel) dengan checkpoint IndoBERT-RCNN dan konfigurasi CURE
terpilih pada data bersih, lalu evaluasi pelabelan semantik klaster (evaluation.py).
Keluaran: research/out/hasil_pipeline.xlsx, research/out/final_summary.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation import evaluate_cluster_labeling  # noqa: E402
from indobert_rcnn import DEFAULT_LABEL_DESCRIPTIONS  # noqa: E402
from pipeline import StuntingResearchPipeline  # noqa: E402

OUT = ROOT / "research" / "out"
K, C, A, S = (int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), 1500) if len(sys.argv) > 3 else (6, 5, 0.5, 1500)
STUNT = {"STUNTING_BERAT_KOMORBID", "STUNTING_KRONIS", "STUNTING_SEDANG"}
DESC2KEY = {v: k for k, v in DEFAULT_LABEL_DESCRIPTIONS.items()}


def main():
    pipe = StuntingResearchPipeline(n_clusters=K, n_representatives=C, shrink_factor=A, sample_size=S,
                                    checkpoint_path=str(ROOT / "checkpoints" / "indobert_rcnn_stunting.pt"))
    t = time.time()
    summ = pipe.process_excel(OUT / "data_bersih.xlsx", OUT / "hasil_pipeline.xlsx")
    total = time.time() - t
    out = {"params": summ["cure_parameters"], "detik_total": total, "lembar": []}
    for sh in summ["sheets"]:
        d = pd.read_excel(OUT / "hasil_pipeline.xlsx", sheet_name=sh["sheet_name"])
        cid = d["CURE_Cluster_ID"].astype(int).tolist()
        cl_label = d["CURE_Cluster_Label"].tolist()
        gt = d["Label_Acuan"].tolist()
        ev = evaluate_cluster_labeling(cid, cl_label, gt)
        rec_key = d["IndoBERT_Record_Prediksi"].map(DESC2KEY)
        prof = []
        st_all = d["Label_Acuan"].isin(STUNT).mean()
        for c in sorted(set(cid)):
            m = d["CURE_Cluster_ID"] == c
            lab_desc = cl_label[m.idxmax()].split(": ", 1)[-1]
            key = DESC2KEY.get(lab_desc, lab_desc)
            maj = d.loc[m, "Label_Acuan"].value_counts()
            st = d.loc[m, "Label_Acuan"].isin(STUNT)
            prof.append({
                "cluster_id": c, "n": int(m.sum()), "pct": 100 * m.mean(),
                "usia": d.loc[m, "Usia_Bulan"].mean(), "haz": d.loc[m, "ZS TB/U"].mean(),
                "waz": d.loc[m, "ZS BB/U"].mean(), "whz": d.loc[m, "ZS BB/TB"].mean(),
                "label_klaster": key, "confidence": float(d.loc[m, "Cluster_Label_Confidence"].iloc[0]),
                "kemurnian_konsensus": float((rec_key[m] == key).mean()),
                "mayoritas_acuan": maj.index[0], "purity": float(maj.iloc[0] / m.sum()),
                "label_sesuai_mayoritas": key == maj.index[0],
                "stunting": int(st.sum()), "prev_stunting": float(st.mean()),
                "lift": float(st.mean() / st_all) if st_all else None,
                "abnormal": int((d.loc[m, "Label_Acuan"] != "NORMAL_OPTIMAL").sum()),
                "distribusi_acuan": maj.to_dict(),
                "distribusi_prediksi": rec_key[m].value_counts().to_dict(),
            })
        agree = float((rec_key == d["Label_Acuan"]).mean())
        out["lembar"].append({"lembar": sh["sheet_name"], "n": sh["rows"], "lengkap": sh["complete_anthropometric_rows"],
                              "silhouette_github": sh["silhouette_score"], "evaluasi": ev, "profil": prof,
                              "kesesuaian_rekam_seluruh_data": agree})
        print(sh["sheet_name"], "purity", ev["overall_cluster_purity"], "agree", round(agree, 4))
        for p in prof:
            print("  ", p["cluster_id"], p["n"], p["label_klaster"], round(p["confidence"], 3), "maj", p["mayoritas_acuan"],
                  round(p["purity"], 3), "stunt", p["stunting"], "abn", p["abnormal"], "kpur", round(p["kemurnian_konsensus"], 3))
    out["detik_total"] = total
    json.dump(out, open(OUT / "final_summary.json", "w"), indent=1, default=float)
    print("total detik", round(total, 1))


if __name__ == "__main__":
    main()
