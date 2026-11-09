"""Contoh tokenisasi, softmax, dan konsensus klaster (subbab 3.9) dari checkpoint akhir."""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import indobert_rcnn as R  # noqa: E402

OUT = ROOT / "research" / "out"


def main():
    mx = json.load(open(OUT / "manual_example.json", encoding="utf-8"))
    pipe = R.IndoBERTRCNNPipeline(checkpoint_path=str(ROOT / "checkpoints" / "indobert_rcnn_stunting.pt"))
    tok = pipe.tokenizer
    notes = [r["Catatan_Pemeriksaan"] for r in mx["rekam"]]
    a6 = notes[5]
    ids = tok(a6)["input_ids"]
    tokens = [[t, i] for t, i in zip(tok.convert_ids_to_tokens(ids), ids)]
    enc = tok(a6, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    pipe.model.cpu().eval()
    with torch.no_grad():
        logits = pipe.model(enc["input_ids"], enc["attention_mask"])[0]
    probs = F.softmax(logits, -1).tolist()
    classes = [pipe.id2label[i] for i in range(pipe.num_classes)]
    # konsensus klaster stunting contoh {A3, A6} = label 1 pada manual_example
    lab = mx["labels"]
    members = [i for i, l in enumerate(lab) if l == lab[5]]
    preds = pipe.predict([notes[i] for i in members])
    info = pipe.label_cluster([notes[i] for i in members], cluster_id=2, predictions=preds)
    mean = {c: sum(p["probabilities"][c] for p in preds) / len(preds) for c in classes}
    top2 = sorted(mean, key=mean.get, reverse=True)[:2]
    rinc = "; ".join(f"{c.replace('_', '_' + chr(8203))} = ({' + '.join(str(round(p['probabilities'][c], 4)).replace('.', ',') for p in preds)}) / {len(preds)} = {str(round(mean[c], 4)).replace('.', ',')}" for c in top2)
    res = {
        "spec": {"vocab": tok.vocab_size},
        "tokens": tokens,
        "a6": {"logits": logits.tolist(), "probs": dict(zip(classes, probs)), "classes": classes},
        "konsensus": {"klaster": "K2", "anggota": ", ".join(f"A{i+1}" for i in members), "rincian": rinc,
                      "label": info["best_label_key"], "conf": info["confidence"], "kpur": info["consensus_purity"],
                      "nama": info["cluster_label"].replace("Klaster 2", "Klaster K2"),
                      "pred_anggota": [(f"A{i+1}", p["label_key"], p["confidence"]) for i, p in zip(members, preds)]},
    }
    json.dump(res, open(OUT / "manual_model.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "tokens"}, indent=1, ensure_ascii=False)[:1500])
    print(len(tokens), "token")


if __name__ == "__main__":
    main()
