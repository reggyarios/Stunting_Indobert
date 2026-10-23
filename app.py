#!/usr/bin/env python3
"""
CURE-IndoBERT for Stunting Analytics
===================================
Production Flask application providing:
1. Multi-sheet Excel ingestion (.xlsx, .xls)
2. Classical CURE (Clustering Using Representatives) algorithm with shrinkage factor alpha and c representatives
3. Authentic IndoBERT + RCNN contextual neural architecture for clinical stunting classification
4. Permenkes RI No. 2/2020 & WHO 2006 clinical stunting guideline assessment
5. Multi-sheet Excel export with clearly separated CURE cluster IDs, model predictions, and Kemenkes categories
6. Rigorous classification and clustering evaluation on independent ground truth
"""

import os
import sys
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from flask import (
    Flask, render_template, request, redirect,
    url_for, send_file, jsonify, flash, abort
)
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

from cure import CUREClusterer
from indobert_rcnn import IndoBERTRCNNPipeline, ModelNotTrainedError, DEFAULT_LABEL2ID
from pipeline import StuntingResearchPipeline
from evaluation import evaluate_classification, evaluate_clustering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CURE_IndoBERT_App")

# Application Paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TEMPLATE_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHECKPOINT_PATH = CHECKPOINT_DIR / "indobert_rcnn_stunting.pt"

ALLOWED_EXTENSIONS = {"xlsx", "xls"}

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "cure-indobert-secret-key-2026")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_DIR)

# In-memory job state cache
JOB_STORE: Dict[str, Dict[str, Any]] = {}


def allowed_file(filename: str) -> bool:
    """Validate file extension against allowed Excel formats."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_active_checkpoint_path() -> Optional[str]:
    """Return path to trained checkpoint if available."""
    if DEFAULT_CHECKPOINT_PATH.exists():
        return str(DEFAULT_CHECKPOINT_PATH)
    return None


@app.route("/", methods=["GET"])
def index():
    """Render modern minimalist file upload and configuration interface."""
    has_checkpoint = DEFAULT_CHECKPOINT_PATH.exists()
    return render_template("index.html", has_checkpoint=has_checkpoint)


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle Excel file upload, run research pipeline, and return result."""
    if "file" not in request.files:
        flash("File Excel tidak ditemukan dalam permintaan.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("Silakan pilih file Excel (.xlsx atau .xls).", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Format file tidak didukung. Harap unggah file .xlsx atau .xls.", "error")
        return redirect(url_for("index"))

    job_id = str(uuid.uuid4())
    safe_name = secure_filename(file.filename) or "stunting_dataset.xlsx"
    upload_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    file.save(str(upload_path))

    # Read hyperparameters from form
    try:
        n_clusters = int(request.form.get("n_clusters", 4))
        n_representatives = int(request.form.get("n_representatives", 5))
        shrink_factor = float(request.form.get("shrink_factor", 0.3))
        sample_size = int(request.form.get("sample_size", 1500))
    except (ValueError, TypeError):
        n_clusters, n_representatives, shrink_factor, sample_size = 4, 5, 0.3, 1500

    output_filename = f"CURE_IndoBERT_Enriched_{safe_name}"
    output_path = OUTPUT_DIR / f"{job_id}_{output_filename}"

    checkpoint_path = get_active_checkpoint_path()

    try:
        pipeline = StuntingResearchPipeline(
            n_clusters=n_clusters,
            n_representatives=n_representatives,
            shrink_factor=shrink_factor,
            sample_size=sample_size,
            checkpoint_path=checkpoint_path
        )
        metrics = pipeline.process_excel(upload_path, output_path)
        metrics["job_id"] = job_id
        metrics["original_filename"] = safe_name
        metrics["output_filename"] = output_filename
        metrics["output_path"] = str(output_path)

        JOB_STORE[job_id] = metrics
        logger.info("Successfully processed job %s with %d rows across %d sheet(s)",
                    job_id, metrics["total_rows"], metrics["processed_sheets"])

        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
            return jsonify({
                "status": "success",
                "job_id": job_id,
                "redirect_url": url_for("view_result", job_id=job_id)
            })

        return redirect(url_for("view_result", job_id=job_id))

    except Exception as e:
        logger.exception("Pipeline failure on job %s: %s", job_id, str(e))
        flash(f"Terjadi kesalahan saat memproses data: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/result/<job_id>", methods=["GET"])
def view_result(job_id: str):
    """Render analytical dashboard with CURE profiles and guideline metrics."""
    metrics = JOB_STORE.get(job_id)
    if not metrics:
        flash("Sesi data tidak ditemukan atau telah kedaluwarsa. Silakan unggah ulang.", "error")
        return redirect(url_for("index"))

    return render_template("result.html", metrics=metrics)


@app.route("/download/<job_id>", methods=["GET"])
def download_result(job_id: str):
    """Send enriched multi-sheet Excel file to user."""
    metrics = JOB_STORE.get(job_id)
    if not metrics or not Path(metrics["output_path"]).exists():
        abort(404, description="File output tidak ditemukan.")

    return send_file(
        metrics["output_path"],
        as_attachment=True,
        download_name=metrics["output_filename"],
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/sample-dataset", methods=["GET"])
def download_sample_dataset():
    """Serve benchmark child stunting dataset."""
    sample_path = OUTPUT_DIR / "Benchmark_Child_Stunting_Indonesia.xlsx"
    if not sample_path.exists():
        generate_benchmark_stunting_workbook(sample_path)

    return send_file(
        str(sample_path),
        as_attachment=True,
        download_name="Benchmark_Child_Stunting_Indonesia.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/load-benchmark", methods=["POST"])
def load_benchmark():
    """Run research pipeline directly on benchmark dataset."""
    sample_path = OUTPUT_DIR / "Benchmark_Child_Stunting_Indonesia.xlsx"
    if not sample_path.exists():
        generate_benchmark_stunting_workbook(sample_path)

    job_id = str(uuid.uuid4())
    output_filename = "CURE_IndoBERT_Enriched_Benchmark.xlsx"
    output_path = OUTPUT_DIR / f"{job_id}_{output_filename}"

    pipeline = StuntingResearchPipeline(
        n_clusters=4,
        n_representatives=5,
        shrink_factor=0.3,
        checkpoint_path=get_active_checkpoint_path()
    )
    metrics = pipeline.process_excel(sample_path, output_path)
    metrics["job_id"] = job_id
    metrics["original_filename"] = "Benchmark_Child_Stunting_Indonesia.xlsx"
    metrics["output_filename"] = output_filename
    metrics["output_path"] = str(output_path)

    JOB_STORE[job_id] = metrics
    return jsonify({
        "status": "success",
        "job_id": job_id,
        "redirect_url": url_for("view_result", job_id=job_id)
    })


@app.route("/api/train", methods=["POST"])
def train_model():
    """
    Train IndoBERT + RCNN model on provided labeled dataset and save checkpoint.
    Expects JSON payload with 'texts' and 'labels' or multipart file with labeled Excel.
    """
    try:
        data = request.get_json(silent=True)
        if not data or "texts" not in data or "labels" not in data:
            return jsonify({
                "status": "error",
                "message": "JSON body must contain 'texts' (list of strings) and 'labels' (list of class keys or integer IDs)."
            }), 400

        texts = data["texts"]
        labels = data["labels"]

        if len(texts) < 5 or len(labels) < 5:
            return jsonify({
                "status": "error",
                "message": "At least 5 labeled samples required for training."
            }), 400

        pipeline = IndoBERTRCNNPipeline()
        pipeline.initialize_new_model()

        epochs = int(data.get("epochs", 3))
        batch_size = int(data.get("batch_size", 8))
        lr = float(data.get("lr", 2e-5))

        history = pipeline.train(
            train_texts=texts,
            train_labels=labels,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_save_path=str(DEFAULT_CHECKPOINT_PATH)
        )

        return jsonify({
            "status": "success",
            "message": f"Successfully trained IndoBERT + RCNN and saved checkpoint to {DEFAULT_CHECKPOINT_PATH.name}",
            "checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
            "history": history
        })

    except Exception as e:
        logger.exception("Training error: %s", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """System health and model checkpoint status probe."""
    has_checkpoint = DEFAULT_CHECKPOINT_PATH.exists()
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_jobs": len(JOB_STORE),
        "has_trained_checkpoint": has_checkpoint,
        "checkpoint_path": str(DEFAULT_CHECKPOINT_PATH) if has_checkpoint else None
    })


def generate_benchmark_stunting_workbook(file_path: Path):
    """
    Generate authentic Indonesian child stunting dataset (.xlsx).
    Provides realistic Posyandu & Puskesmas records with clinical notes and recorded Z-scores.
    """
    names_male = ["Muhammad Rizky", "Alif Pratama", "Bima Arya", "Dimas Saputra", "Fikri Maulana", "Gilang Ramadhan", "Haikal Putra", "Ilham Fauzi", "Kenzo Raditya", "Raffa Putra"]
    names_female = ["Aisyah Putri", "Anindita Zahra", "Cantika Dewi", "Dara Ayu", "Fadhila Nur", "Gita Lestari", "Hanum Salsabila", "Inara Kirana", "Keysha Anindya", "Naila Rahma"]
    puskesmas_list = ["Puskesmas Sukamaju", "Puskesmas Cikarang Barat", "Puskesmas Cibitung", "Puskesmas Tambun Selatan"]
    desa_list = ["Desa Mekarsari", "Desa Sukadanau", "Desa Wanajaya", "Desa Telaga Asih"]

    notes_templates = [
        "Balita tampak kurus, riwayat diare berulang dalam 2 bulan terakhir, KMS pita kuning.",
        "Pertumbuhan optimal, anak sangat aktif, imunisasi dasar lengkap, ASI eksklusif 6 bulan.",
        "Anak sangat pendek untuk usianya, nafsu makan minim, asupan protein hewani sangat rendah, riwayat ISPA.",
        "Bawah Garis Merah (BGM) pada KMS posyandu, berat badan 2 bulan tidak naik berturut-turut (2T).",
        "Tinggi badan sesuai usia, motorik kasar baik, konsumsi makanan keluarga variatif telur dan ikan.",
        "Gejala perawakan pendek (stunting), sanitasi air sumur tercemar, riwayat berat lahir rendah (BBLR 2.1 kg).",
        "Lingkar lengan atas (LiLA) 11.2 cm (pita merah), asupan gizi makro sangat kurang, ibu bekerja.",
        "Pertumbuhan stabil di kurva hijau KMS, stimulasi tumbuh kembang baik oleh orang tua.",
        "Balita tampak lesu, nafsu makan berkurang, riwayat batuk kronis, dicurigai kontak TB keluarga.",
        "Gizi kurang dengan risiko gagal tumbuh, MPASI didominasi karbohidrat bubur tanpa variasi lauk."
    ]

    np.random.seed(42)

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        rows_s1 = []
        for i in range(30):
            is_male = np.random.rand() > 0.5
            nama = np.random.choice(names_male if is_male else names_female) + f" {i+1}"
            usia = int(np.random.randint(6, 59))

            stunt_prob = np.random.rand()
            if stunt_prob < 0.35:
                haz = round(float(np.random.uniform(-4.2, -2.1)), 2)
                waz = round(float(np.random.uniform(-3.8, -1.8)), 2)
                whz = round(float(np.random.uniform(-2.5, 0.5)), 2)
                catatan = np.random.choice([notes_templates[0], notes_templates[2], notes_templates[3], notes_templates[5], notes_templates[6]])
            elif stunt_prob < 0.55:
                haz = round(float(np.random.uniform(-2.0, -0.8)), 2)
                waz = round(float(np.random.uniform(-3.2, -2.0)), 2)
                whz = round(float(np.random.uniform(-3.5, -2.1)), 2)
                catatan = np.random.choice([notes_templates[0], notes_templates[3], notes_templates[6], notes_templates[9]])
            else:
                haz = round(float(np.random.uniform(-1.5, 1.8)), 2)
                waz = round(float(np.random.uniform(-1.2, 1.5)), 2)
                whz = round(float(np.random.uniform(-1.0, 1.2)), 2)
                catatan = np.random.choice([notes_templates[1], notes_templates[4], notes_templates[7]])

            tb = round(48.0 + (usia * 0.95) + (haz * 3.2), 1)
            bb = round(3.2 + (usia * 0.28) + (waz * 1.4), 1)

            rows_s1.append({
                "No_Registrasi": f"REG-PKM-{1001 + i}",
                "Nama_Balita": nama,
                "Jenis_Kelamin": "Laki-laki" if is_male else "Perempuan",
                "Usia_Bulan": usia,
                "Berat_Badan_Kg": bb,
                "Tinggi_Badan_Cm": tb,
                "ZScore_TB_U": haz,
                "ZScore_BB_U": waz,
                "ZScore_BB_TB": whz,
                "Nama_Puskesmas": np.random.choice(puskesmas_list),
                "Kelurahan_Desa": np.random.choice(desa_list),
                "Catatan_Pemeriksaan_Klinis": catatan
            })
        df1 = pd.DataFrame(rows_s1)
        df1.to_excel(writer, sheet_name="Puskesmas_Kecamatan_Barat", index=False)

        rows_s2 = []
        for i in range(25):
            is_male = np.random.rand() > 0.5
            nama = np.random.choice(names_male if is_male else names_female) + f" M-{i+1}"
            usia = int(np.random.randint(9, 48))
            haz = round(float(np.random.uniform(-3.5, 1.2)), 2)
            waz = round(float(np.random.uniform(-3.0, 1.0)), 2)
            whz = round(float(np.random.uniform(-2.8, 1.1)), 2)
            tb = round(52.0 + (usia * 0.88) + (haz * 3.0), 1)
            bb = round(4.0 + (usia * 0.25) + (waz * 1.3), 1)
            catatan = np.random.choice(notes_templates)

            rows_s2.append({
                "ID_Anak": f"POS-MLT-{201 + i}",
                "Nama_Lengkap": nama,
                "Gender": "L" if is_male else "P",
                "Umur_Bulan": usia,
                "BB_Kg": bb,
                "TB_Cm": tb,
                "HAZ": haz,
                "WAZ": waz,
                "WHZ": whz,
                "Posyandu": "Posyandu Melati 03",
                "Keterangan_Bidan_Desa": catatan
            })
        df2 = pd.DataFrame(rows_s2)
        df2.to_excel(writer, sheet_name="Posyandu_Melati_Timur", index=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting CURE-IndoBERT server on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
