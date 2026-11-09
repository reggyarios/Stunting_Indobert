"""
Pelatihan dan evaluasi IndoBERT + RCNN (indobert_rcnn.py repo ini) pada catatan
antropometri hasil prepare_data.py, memakai prepare_dataset_splits (70/15/15,
stratified, teks identik dikelompokkan) dan IndoBERTRCNNPipeline.train().

Pembanding pada data uji yang sama:
  - TF-IDF (n-gram karakter) + regresi logistik,
  - ablasi IndoBERT tanpa RCNN (token [CLS] -> linear, 8 lapisan awal dibekukan sama).
Keluaran: checkpoints/indobert_rcnn_stunting.pt, research/out/indobert_*.json|csv

Jalankan: python research/exp_indobert.py [epochs]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import indobert_rcnn as R  # noqa: E402
from evaluation import evaluate_classification  # noqa: E402

OUT = ROOT / "research" / "out"
CKPT = ROOT / "checkpoints" / "indobert_rcnn_stunting.pt"
STUNT = {"STUNTING_BERAT_KOMORBID", "STUNTING_KRONIS", "STUNTING_SEDANG"}


def seed_all(s: int = 42) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def binary(y_true, y_pred) -> dict:
    t = np.isin(y_true, list(STUNT)); p = np.isin(y_pred, list(STUNT))
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int((~t & ~p).sum()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "mcc": float(matthews_corrcoef(t, p))}


def report(name, y_true, y_pred, secs, extra=None) -> dict:
    ev = evaluate_classification(list(y_true), list(y_pred), labels=list(R.DEFAULT_LABEL2ID))
    out = {"model": name, "detik_latih": round(secs, 1), "accuracy": ev["accuracy"],
           "macro": ev["macro_avg"], "weighted": ev["weighted_avg"], "per_class": ev["per_class"],
           "classes": ev["classes"], "confusion_matrix": ev["confusion_matrix"],
           "mcc_multiclass": float(matthews_corrcoef(y_true, y_pred)), "stunting_biner": binary(y_true, y_pred)}
    out.update(extra or {})
    print(f"\n== {name}: acc={out['accuracy']} macroF1={out['macro']['f1_score']} "
          f"stuntingF1={out['stunting_biner']['f1']:.3f} ({secs:.0f}s)", flush=True)
    return out


class CLSLinear(nn.Module):
    """Ablasi: IndoBERT yang sama, kepala linear pada [CLS] (tanpa BiLSTM/Conv1D)."""

    def __init__(self, name, num_classes, freeze=8):
        super().__init__()
        self.bert = R.AutoModel.from_pretrained(name)
        for layer in self.bert.encoder.layer[:freeze]:
            for p in layer.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.fc(self.dropout(h))


def main(epochs: int, rcnn_only: bool = False) -> None:
    global CKPT
    tag = f"_e{epochs}" if rcnn_only else ""
    if rcnn_only:
        CKPT = ROOT / "checkpoints" / f"indobert_rcnn_stunting{tag}.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_excel(OUT / "data_bersih.xlsx", sheet_name=s) for s in ("2024", "2025")])
    texts, labels = df["Catatan_Pemeriksaan"].tolist(), df["Label_Acuan"].tolist()
    sp = R.prepare_dataset_splits(texts, labels, R.DEFAULT_LABEL2ID, 0.70, 0.15, 0.15, seed=42)
    id2 = R.DEFAULT_ID2LABEL
    dist = {part: pd.Series([id2[i] for i in sp[f"{part}_labels"]]).value_counts().to_dict()
            for part in ("train", "val", "test")}
    json.dump({"counts": sp["counts"], "duplicates_grouped": sp["stats"]["duplicates_grouped"],
               "distribution": dist}, open(OUT / "indobert_split.json", "w"), indent=2)
    print(sp["counts"], dist, flush=True)
    y_test = [id2[i] for i in sp["test_labels"]]
    results = []

    # 1) IndoBERT + RCNN (kode repo, tanpa perubahan hiperparameter baku kecuali epoch)
    seed_all(42)
    pipe = R.IndoBERTRCNNPipeline()
    pipe.initialize_new_model()
    n_par = sum(p.numel() for p in pipe.model.parameters())
    n_trn = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
    t = time.time()
    hist = pipe.train(sp["train_texts"], sp["train_labels"], sp["val_texts"], sp["val_labels"],
                      epochs=epochs, batch_size=8, lr=2e-5, checkpoint_save_path=str(CKPT))
    secs = time.time() - t
    t = time.time()
    preds = pipe.predict(sp["test_texts"])
    infer = time.time() - t
    y_pred = [p["label_key"] for p in preds]
    results.append(report("IndoBERT + RCNN", y_test, y_pred, secs,
                          {"history": hist, "params_total": n_par, "params_trainable": n_trn,
                           "detik_inferensi_uji": round(infer, 1)}))
    pd.DataFrame({"teks": sp["test_texts"], "acuan": y_test, "prediksi": y_pred,
                  "confidence": [p["confidence"] for p in preds]}).to_csv(OUT / f"indobert_test_pred{tag}.csv", index=False)
    if rcnn_only:
        json.dump(results, open(OUT / f"indobert_results{tag}.json", "w"), indent=2, default=float)
        return

    # 2) TF-IDF + regresi logistik
    t = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(sp["train_texts"])
    lr = LogisticRegression(max_iter=3000, C=10).fit(Xtr, [id2[i] for i in sp["train_labels"]])
    y_lr = lr.predict(vec.transform(sp["test_texts"]))
    results.append(report("TF-IDF + Regresi Logistik", y_test, list(y_lr), time.time() - t))

    # 3) Ablasi IndoBERT [CLS] linear, pelatihan identik (train loop repo)
    seed_all(42)
    abl = R.IndoBERTRCNNPipeline()
    abl.tokenizer = R.AutoTokenizer.from_pretrained(abl.model_name)
    abl.model = CLSLinear(abl.model_name, abl.num_classes)
    abl.is_loaded = True
    t = time.time()
    abl.train(sp["train_texts"], sp["train_labels"], sp["val_texts"], sp["val_labels"],
              epochs=epochs, batch_size=8, lr=2e-5)
    secs = time.time() - t
    y_abl = [p["label_key"] for p in abl.predict(sp["test_texts"])]
    results.append(report("IndoBERT [CLS] + Linear (tanpa RCNN)", y_test, y_abl, secs))

    json.dump(results, open(OUT / "indobert_results.json", "w"), indent=2, default=float)
    print("selesai")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3, len(sys.argv) > 2 and sys.argv[2] == "rcnn")
