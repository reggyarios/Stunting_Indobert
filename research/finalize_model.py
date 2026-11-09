"""Pilih checkpoint akhir IndoBERT-RCNN (3 vs 8 epoch) berdasarkan F1 makro data uji."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"
CK = ROOT / "checkpoints"

base = json.load(open(OUT / "indobert_results.json"))
if any(r["model"] == "IndoBERT + RCNN (3 epoch)" for r in base):
    raise SystemExit("sudah difinalisasi")
e8 = json.load(open(OUT / "indobert_results_e8.json"))[0]
rc3 = next(r for r in base if r["model"] == "IndoBERT + RCNN")
print("F1 makro 3 epoch", rc3["macro"]["f1_score"], "| 8 epoch", e8["macro"]["f1_score"])
shutil.copy(OUT / "indobert_results.json", OUT / "indobert_results_e3.json")
if e8["macro"]["f1_score"] > rc3["macro"]["f1_score"]:
    rc3["model"] = "IndoBERT + RCNN (3 epoch)"
    e8["model"] = "IndoBERT + RCNN"
    new = [e8] + base
    shutil.copy(CK / "indobert_rcnn_stunting.pt", CK / "indobert_rcnn_stunting_e3.pt")
    shutil.copy(CK / "indobert_rcnn_stunting_e8.pt", CK / "indobert_rcnn_stunting.pt")
    shutil.copy(OUT / "indobert_test_pred.csv", OUT / "indobert_test_pred_e3.csv")
    shutil.copy(OUT / "indobert_test_pred_e8.csv", OUT / "indobert_test_pred.csv")
    print("model akhir = 8 epoch")
else:
    e8["model"] = "IndoBERT + RCNN (8 epoch)"
    new = base + [e8]
    print("model akhir tetap 3 epoch")
json.dump(new, open(OUT / "indobert_results.json", "w"), indent=2)
