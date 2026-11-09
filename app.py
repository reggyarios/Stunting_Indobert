#!/usr/bin/env python3
"""
CURE-IndoBERT for Stunting Analytics
===================================
Production Flask application providing:
1. Multi-sheet Excel ingestion (.xlsx, .xls)
2. Classical CURE (Clustering Using Representatives) algorithm with shrinkage factor alpha and c representatives
3. IndoBERT + RCNN neural architecture for research stunting classification with validation tracking
4. Permenkes RI No. 2/2020 guideline assessment without unsupported polynomial approximations
5. Multi-sheet Excel export preserving source data
6. Rigorous classification and clustering evaluation on independent ground truth
"""

import gc
import os
import sys
import json
import uuid
import secrets
import logging
import tempfile
from datetime import datetime, date
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from flask import (
        Flask, render_template, request, redirect,
        url_for, send_file, jsonify, flash, abort
    )
    from werkzeug.utils import secure_filename
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    Flask = None
    render_template = request = redirect = url_for = send_file = jsonify = flash = abort = None
    secure_filename = lambda f: f

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from cure import CUREClusterer
from indobert_rcnn import (
    IndoBERTRCNNPipeline, ModelNotTrainedError, DEFAULT_LABEL2ID,
    prepare_dataset_splits
)
from pipeline import StuntingResearchPipeline, normalize_column_name, parse_indonesian_numeric
from evaluation import evaluate_classification, evaluate_clustering

# Configure logging without sensitive PII
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

if HAS_FLASK and Flask is not None:
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
    app.config["OUTPUT_FOLDER"] = str(OUTPUT_DIR)
else:
    class DummyFlask:
        config = {}
        def route(self, *args, **kwargs):
            return lambda fn: fn
        errorhandler = route

        def after_request(self, fn):
            return fn
    app = DummyFlask()
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

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


@lru_cache(maxsize=1)
def _load_classifier(path: str, mtime: float) -> Optional[IndoBERTRCNNPipeline]:
    try:
        return IndoBERTRCNNPipeline(checkpoint_path=path)
    except Exception as e:  # checkpoint rusak/tidak cocok: jalan tanpa model, tanpa label palsu
        logger.warning("Checkpoint IndoBERT tidak dapat dimuat: %s", e)
        return None


def get_classifier() -> Optional[IndoBERTRCNNPipeline]:
    """Checkpoint dimuat sekali dan dipakai ulang antar-permintaan (dimuat ulang bila berkasnya berubah)."""
    path = get_active_checkpoint_path()
    return _load_classifier(path, os.path.getmtime(path)) if path else None


# ---------------------------------------------------------------------------
# Validasi unggahan (dipakai /upload dan /api/analyze)
# ---------------------------------------------------------------------------
XLSX_MAGIC = b"PK\x03\x04"
XLS_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

REQUIRED_COLUMNS = {
    "gender": "JK", "weight": "Berat", "height": "Tinggi",
    "haz": "ZS TB/U", "waz": "ZS BB/U", "whz": "ZS BB/TB",
}
IDENTIFIER_TOKENS = {"nama", "name", "nik", "kk", "alamat", "address", "ibu", "ayah",
                     "ortu", "wali", "telp", "telepon", "hp"}
# Rentang wajar balita 0-60 bulan (masih di dalam rentang yang diterima pipeline.py)
WEIGHT_RANGE_KG = (1.5, 35.0)
HEIGHT_RANGE_CM = (38.0, 130.0)
AGE_RANGE_MONTHS = (0.0, 60.0)
DAYS_PER_MONTH = 30.4375
MAX_CLUSTERS, MAX_REPRESENTATIVES, MAX_SAMPLE_SIZE = 50, 50, 5000


class UploadError(ValueError):
    """Unggahan/parameter ditolak; pesan berbahasa Indonesia untuk pengguna (HTTP 400)."""

    def __init__(self, message: str, **extra: Any):
        super().__init__(message)
        self.extra = extra


def column_role(col: Any) -> Optional[str]:
    """Peran kolom: haz/waz/whz/age/dob/measured/height/weight/gender/identifier atau None."""
    k = normalize_column_name(col)
    toks = set(k.split("_"))
    if toks & IDENTIFIER_TOKENS:
        return "identifier"
    if toks & {"status", "kategori"}:
        return None
    is_z = bool(toks & {"zs", "zscore", "z"})
    if "haz" in toks or (is_z and ("tb_u" in k or "pb_u" in k)):
        return "haz"
    if "waz" in toks or (is_z and "bb_u" in k):
        return "waz"
    if "whz" in toks or (is_z and ("bb_tb" in k or "bb_pb" in k)):
        return "whz"
    if any(p in k for p in ("tb_u", "pb_u", "bb_u", "bb_tb", "bb_pb")):
        return None  # kolom status teks (TB/U, BB/U, BB/TB)
    if toks & {"tgl", "tanggal"}:
        if toks & {"lahir", "lhr"}:
            return "dob"
        if toks & {"ukur", "timbang", "pengukuran", "kunjungan"}:
            return "measured"
        return None
    if toks & {"usia", "umur"}:
        return "age"
    if toks & {"tinggi", "panjang", "tb", "pb"}:
        return "height"
    if toks & {"berat", "bb"}:
        return "weight"
    if toks & {"jk", "gender", "sex"} or "jenis_kelamin" in k:
        return "gender"
    return None


def _parse_gender(v: Any) -> Optional[str]:
    # Sama dengan aturan pipeline._extract_indicators
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).strip().upper()
    if "L" in s or "PRIA" in s or "LAKI" in s:
        return "L"
    if "P" in s or "WANITA" in s or "PEREMPUAN" in s:
        return "P"
    return None


def _to_timestamp(v: Any):
    ts = pd.to_datetime(v, errors="coerce")
    return None if pd.isna(ts) else ts


def row_issues(haz, waz, whz, weight, height, age, gender) -> List[str]:
    """Alasan baris dikeluarkan: data tidak lengkap, WHO flag, atau ukuran tidak wajar."""
    missing = [label for label, v in (("JK", gender), ("Berat", weight), ("Tinggi", height),
                                      ("ZS TB/U", haz), ("ZS BB/U", waz), ("ZS BB/TB", whz),
                                      ("Usia/Tgl Lahir", age)) if v is None]
    issues = [f"Data tidak lengkap/tidak terbaca: {', '.join(missing)}"] if missing else []
    if haz is not None and abs(haz) > 6:
        issues.append(f"ZS TB/U {haz:g} di luar batas WHO (|z| > 6)")
    if waz is not None and not (-6 <= waz <= 5):
        issues.append(f"ZS BB/U {waz:g} di luar batas WHO (-6 s.d. +5)")
    if whz is not None and abs(whz) > 5:
        issues.append(f"ZS BB/TB {whz:g} di luar batas WHO (|z| > 5)")
    if weight is not None and not (WEIGHT_RANGE_KG[0] <= weight <= WEIGHT_RANGE_KG[1]):
        issues.append(f"Berat {weight:g} kg di luar rentang wajar ({WEIGHT_RANGE_KG[0]:g}-{WEIGHT_RANGE_KG[1]:g} kg)")
    if height is not None and not (HEIGHT_RANGE_CM[0] <= height <= HEIGHT_RANGE_CM[1]):
        issues.append(f"Tinggi {height:g} cm di luar rentang wajar ({HEIGHT_RANGE_CM[0]:g}-{HEIGHT_RANGE_CM[1]:g} cm)")
    if age is not None and not (AGE_RANGE_MONTHS[0] <= age <= AGE_RANGE_MONTHS[1]):
        issues.append(f"Usia {age:g} bulan di luar rentang balita 0-60 bulan (periksa tanggal acuan)")
    return issues


def parse_analysis_params(form: Any) -> Dict[str, Any]:
    try:
        params = {
            "n_clusters": int(form.get("n_clusters", 4)),
            "n_representatives": int(form.get("n_representatives", 5)),
            "shrink_factor": float(form.get("shrink_factor", 0.3)),
            "sample_size": int(form.get("sample_size", 1500)),
        }
    except (TypeError, ValueError):
        raise UploadError("Parameter CURE harus berupa angka.")
    if not (1 <= params["n_clusters"] <= MAX_CLUSTERS and 1 <= params["n_representatives"] <= MAX_REPRESENTATIVES
            and 0.0 <= params["shrink_factor"] <= 1.0 and 2 <= params["sample_size"] <= MAX_SAMPLE_SIZE):
        raise UploadError(f"Parameter CURE di luar rentang: 1 <= k <= {MAX_CLUSTERS}, 1 <= c <= {MAX_REPRESENTATIVES}, "
                          f"0 <= alpha <= 1, 2 <= ukuran sampel <= {MAX_SAMPLE_SIZE}.")
    ref = (form.get("reference_date") or "").strip()
    try:
        params["reference_date"] = date.fromisoformat(ref) if ref else date.today()
    except ValueError:
        raise UploadError("Tanggal acuan harus berformat YYYY-MM-DD.")
    return params


def _read_raw_sheets(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        head = fh.read(8)
    if not (head.startswith(XLSX_MAGIC) or head.startswith(XLS_MAGIC)):
        raise UploadError("Isi berkas bukan workbook Excel (.xlsx/.xls). Berkas mungkin rusak, "
                          "atau berupa CSV/PDF/berkas lain yang diganti ekstensinya.")
    try:
        with pd.ExcelFile(str(path)) as xf:
            return {s: pd.read_excel(xf, sheet_name=s, header=None) for s in xf.sheet_names}
    except ImportError:
        raise UploadError("Server belum mendukung format .xls lama (paket xlrd tidak terpasang). "
                          "Simpan ulang berkas sebagai .xlsx lalu unggah kembali.")
    except Exception:
        raise UploadError("Workbook Excel tidak dapat dibaca: berkas rusak, terenkripsi, atau bukan berkas Excel.")


def prepare_workbook(src_path: Path, clean_path: Path, reference_date: date) -> Dict[str, Any]:
    """
    Validasi workbook lalu tulis salinan bersih untuk pipeline: kolom identitas (nama/NIK/alamat)
    dibuang, baris tidak lengkap / WHO flag / ukuran tidak wajar dikeluarkan dan dilaporkan.
    """
    raw_sheets = _read_raw_sheets(src_path)
    detector = StuntingResearchPipeline()
    ref_ts = pd.Timestamp(reference_date)
    sheets, skipped, cleaned = [], [], {}
    any_data, valid_sheets = False, 0

    for s_name, df_raw in raw_sheets.items():
        if df_raw.empty or len(df_raw) < 2:
            skipped.append({"sheet_name": s_name, "reason": "Lembar kosong"})
            continue
        header_idx = detector._detect_header_row(df_raw)
        headers, seen = [], {}
        for c in df_raw.iloc[header_idx]:
            h = str(c).strip()
            seen[h] = seen.get(h, 0) + 1
            headers.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
        data = df_raw.iloc[header_idx + 1:].copy()
        data.columns = headers
        data = data.dropna(how="all")
        if data.empty:
            skipped.append({"sheet_name": s_name, "reason": "Tidak ada baris data balita"})
            continue
        any_data = True

        role_col: Dict[str, str] = {}
        for h in headers:
            role = column_role(h)
            if role and role != "identifier":
                role_col.setdefault(role, h)
        missing = [label for role, label in REQUIRED_COLUMNS.items() if role not in role_col]
        if "age" not in role_col and "dob" not in role_col:
            missing.append("Usia (bulan) atau Tgl Lahir")
        if missing:
            skipped.append({"sheet_name": s_name, "reason": "Kolom wajib tidak ditemukan: " + ", ".join(missing)})
            continue
        valid_sheets += 1

        use_measured = "age" not in role_col and "measured" in role_col
        age_source = ("kolom usia" if "age" in role_col else
                      "Tgl Lahir terhadap kolom tanggal ukur" if use_measured else
                      f"Tgl Lahir terhadap tanggal acuan {reference_date.isoformat()}")
        num = lambda row, role: parse_indonesian_numeric(row[role_col[role]])
        keep, excluded, ages = [], [], []
        for idx, row in data.iterrows():
            if "age" in role_col:
                age = num(row, "age")
            else:
                dob = _to_timestamp(row[role_col["dob"]])
                ref = (_to_timestamp(row[role_col["measured"]]) if use_measured else None) or ref_ts
                age = round((ref - dob).days / DAYS_PER_MONTH, 2) if dob is not None else None
            issues = row_issues(num(row, "haz"), num(row, "waz"), num(row, "whz"), num(row, "weight"),
                                num(row, "height"), age, _parse_gender(row[role_col["gender"]]))
            keep.append(not issues)
            ages.append(age)
            if issues:
                excluded.append({"row": int(idx) + 1, "reasons": issues})

        clean = data.loc[keep, [h for h in headers if column_role(h) != "identifier"]].copy()
        if use_measured:
            clean["Usia_Bulan_Saat_Ukur"] = [a for a, k in zip(ages, keep) if k]
        clean.insert(0, "Baris_Excel", [int(i) + 1 for i, k in zip(data.index, keep) if k])
        if not len(clean):
            skipped.append({"sheet_name": s_name, "reason": f"Semua {len(excluded)} baris dikeluarkan saat validasi "
                            f"(contoh baris {excluded[0]['row']}: {'; '.join(excluded[0]['reasons'])})"})
            continue
        sheets.append({"sheet_name": s_name, "input_rows": len(data), "valid_rows": len(clean),
                       "excluded_rows": excluded, "age_source": age_source})
        cleaned[s_name[:31]] = clean

    if not any_data:
        raise UploadError("Tidak ada data balita pada berkas ini.", skipped_sheets=skipped)
    if not valid_sheets:
        raise UploadError("Tidak ada lembar dengan kolom wajib (JK, Berat, Tinggi, ZS TB/U, ZS BB/U, ZS BB/TB, "
                          "serta Usia atau Tgl Lahir). Data tidak diberi label.", skipped_sheets=skipped)
    if not cleaned:
        raise UploadError("Tidak ada data balita yang lolos validasi (semua baris tidak lengkap atau tidak wajar).",
                          skipped_sheets=skipped)

    with pd.ExcelWriter(str(clean_path), engine="openpyxl") as writer:
        for name, df in cleaned.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return {"sheets": sheets, "skipped_sheets": skipped}


def run_analysis(src_path: Path, clean_path: Path, output_path: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi + CURE + IndoBERT-RCNN (bila checkpoint ada). Melempar UploadError untuk input buruk."""
    prep = prepare_workbook(src_path, clean_path, params["reference_date"])
    pipeline = StuntingResearchPipeline(
        n_clusters=params["n_clusters"],
        n_representatives=params["n_representatives"],
        shrink_factor=params["shrink_factor"],
        sample_size=params["sample_size"],
        checkpoint_path=None,
        reference_date=params["reference_date"].isoformat(),
    )
    pipeline.classifier_pipeline = get_classifier()
    summary = pipeline.process_excel(clean_path, output_path)
    # process_excel tidak menutup pd.ExcelFile; lepaskan handle agar salinan data sementara
    # benar-benar terhapus (di Windows berkas yang masih terbuka tidak bisa dihapus).
    gc.collect()
    clf = pipeline.classifier_pipeline
    return {"summary": summary, "prep": prep, "indobert_active": bool(clf and clf.is_trained)}


def _json_response(obj: Any, status: int = 200):
    body = json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return app.response_class(body, status=status, mimetype="application/json")


def _api_error(message: str, status: int = 400, **extra: Any):
    return _json_response({"status": "error", "message": message, **extra}, status)


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
    clean_path = UPLOAD_DIR / f"{job_id}_validated.xlsx"
    file.save(str(upload_path))

    output_filename = f"CURE_IndoBERT_Enriched_{Path(safe_name).stem}.xlsx"
    output_path = OUTPUT_DIR / f"{job_id}_{output_filename}"

    try:
        params = parse_analysis_params(request.form)
        result = run_analysis(upload_path, clean_path, output_path, params)
        metrics = result["summary"]
        metrics["excluded_rows_total"] = sum(len(s["excluded_rows"]) for s in result["prep"]["sheets"])
        metrics["skipped_sheets"] = result["prep"]["skipped_sheets"]
        metrics["reference_date"] = params["reference_date"].isoformat()
        metrics["indobert_active"] = result["indobert_active"]
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

    except UploadError as e:
        flash(str(e), "error")
        return redirect(url_for("index"))
    except Exception as e:
        logger.exception("Pipeline failure on job %s: %s", job_id, str(e))
        # User-friendly message without exposing internal system paths
        flash("Terjadi kesalahan internal saat memproses data.", "error")
        return redirect(url_for("index"))
    finally:
        # Data balita mentah tidak disimpan setelah diproses (UU 27/2022 PDP)
        for p in (upload_path, clean_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    JSON API untuk front end Next.js. multipart: file, n_clusters, n_representatives,
    shrink_factor, sample_size, reference_date (YYYY-MM-DD). Baris hasil TANPA kolom identitas.
    """
    file = request.files.get("file")
    if file is None or not file.filename:
        return _api_error("Berkas tidak ditemukan. Kirim berkas Excel pada field 'file'.")
    if not allowed_file(file.filename):
        return _api_error("Format berkas tidak didukung. Unggah berkas Excel .xlsx atau .xls.")

    try:
        params = parse_analysis_params(request.form)
        with tempfile.TemporaryDirectory(prefix="stunting_", ignore_cleanup_errors=True) as tmp:
            tmp_dir = Path(tmp)
            src = tmp_dir / ("unggahan." + file.filename.rsplit(".", 1)[1].lower())
            file.save(str(src))
            out = tmp_dir / "hasil.xlsx"
            result = run_analysis(src, tmp_dir / "tervalidasi.xlsx", out, params)
            out_sheets = pd.read_excel(str(out), sheet_name=None)
    except UploadError as e:
        return _api_error(str(e), **e.extra)
    except Exception:
        logger.exception("api_analyze gagal")
        return _api_error("Terjadi kesalahan internal saat memproses data.", status=500)

    sheets = []
    for info in result["prep"]["sheets"]:
        df = out_sheets.get(info["sheet_name"][:31])
        if df is None:
            continue
        df = df[[c for c in df.columns if column_role(c) != "identifier"]]
        split = json.loads(df.to_json(orient="split", index=False, date_format="iso"))
        sheets.append({**info, "columns": split["columns"], "rows": split["data"]})

    return _json_response({
        "status": "success",
        "indobert_active": result["indobert_active"],
        "reference_date": params["reference_date"].isoformat(),
        "summary": result["summary"],
        "sheets": sheets,
        "skipped_sheets": result["prep"]["skipped_sheets"],
    })


@app.after_request
def add_cors_headers(response):
    """CORS sederhana untuk /api/*. ALLOWED_ORIGINS: daftar origin dipisah koma (default '*')."""
    if request.path.startswith("/api/"):
        allowed = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
        origin = request.headers.get("Origin")
        if "*" in allowed:
            response.headers["Access-Control-Allow-Origin"] = "*"
        elif origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.errorhandler(413)
def too_large(_e):
    msg = "Ukuran berkas melebihi batas 64 MB."
    if request.path.startswith("/api/"):
        return _api_error(msg, status=413)
    flash(msg, "error")
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
    """Serve synthetic demonstration child stunting dataset."""
    sample_path = OUTPUT_DIR / "Demonstration_Synthetic_Stunting_Data.xlsx"
    if not sample_path.exists():
        generate_benchmark_stunting_workbook(sample_path)

    return send_file(
        str(sample_path),
        as_attachment=True,
        download_name="Demonstration_Synthetic_Stunting_Data.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/load-benchmark", methods=["POST"])
def load_benchmark():
    """Run research pipeline on synthetic demonstration dataset."""
    sample_path = OUTPUT_DIR / "Demonstration_Synthetic_Stunting_Data.xlsx"
    if not sample_path.exists():
        generate_benchmark_stunting_workbook(sample_path)

    job_id = str(uuid.uuid4())
    output_filename = "CURE_IndoBERT_Enriched_Demo.xlsx"
    output_path = OUTPUT_DIR / f"{job_id}_{output_filename}"

    pipeline = StuntingResearchPipeline(
        n_clusters=4,
        n_representatives=5,
        shrink_factor=0.3,
        checkpoint_path=None
    )
    pipeline.classifier_pipeline = get_classifier()
    metrics = pipeline.process_excel(sample_path, output_path)
    metrics["indobert_active"] = bool(pipeline.classifier_pipeline and pipeline.classifier_pipeline.is_trained)
    metrics["job_id"] = job_id
    metrics["original_filename"] = "Demonstration_Synthetic_Stunting_Data.xlsx"
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
    Train IndoBERT + RCNN model on provided labeled dataset with stratified splitting
    and validation tracking.
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

        if len(texts) < 10 or len(labels) < 10:
            return jsonify({
                "status": "error",
                "message": (
                    f"Provided {len(texts)} samples. In accordance with research methodology, "
                    "at least 10 samples are required to perform a defensible train/val/test split."
                )
            }), 400

        pipeline = IndoBERTRCNNPipeline()
        pipeline.initialize_new_model()

        # Configurable split ratios
        train_ratio = float(data.get("train_ratio", 0.70))
        val_ratio = float(data.get("val_ratio", 0.15))
        test_ratio = float(data.get("test_ratio", 0.15))

        # Dataset splitting with leakage prevention
        splits = prepare_dataset_splits(
            texts=texts,
            labels=labels,
            label2id=pipeline.label2id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        epochs = int(data.get("epochs", 3))
        batch_size = int(data.get("batch_size", 8))
        lr = float(data.get("lr", 2e-5))

        history = pipeline.train(
            train_texts=splits["train_texts"],
            train_labels=splits["train_labels"],
            val_texts=splits["val_texts"],
            val_labels=splits["val_labels"],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_save_path=str(DEFAULT_CHECKPOINT_PATH)
        )

        return jsonify({
            "status": "success",
            "message": f"Successfully trained IndoBERT + RCNN and saved checkpoint to {DEFAULT_CHECKPOINT_PATH.name}",
            "checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
            "split_counts": splits["counts"],
            "split_stats": splits["stats"],
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
        "checkpoint_path": DEFAULT_CHECKPOINT_PATH.name if has_checkpoint else None  # tanpa path absolut server
    })


def generate_benchmark_stunting_workbook(file_path: Path):
    """
    Generate SYNTHETIC DEMONSTRATION child stunting dataset (.xlsx).

    IMPORTANT RESEARCH NOTICE:
    This dataset is entirely synthetic, generated for software testing and user interface
    demonstration purposes only. It must NOT be described as authentic, clinically collected,
    or representative of actual research participants, nor used to claim real-world model accuracy.
    """
    names_male = ["Anak A", "Anak B", "Anak C", "Anak D", "Anak E", "Anak F", "Anak G", "Anak H", "Anak I", "Anak J"]
    names_female = ["Anak K", "Anak L", "Anak M", "Anak N", "Anak O", "Anak P", "Anak Q", "Anak R", "Anak S", "Anak T"]
    puskesmas_list = ["Puskesmas Wilayah Barat", "Puskesmas Wilayah Timur"]
    desa_list = ["Desa A", "Desa B"]

    notes_templates = [
        "Balita tampak kurus, riwayat batuk berulang dalam 2 bulan terakhir, KMS pita kuning.",
        "Pertumbuhan optimal, anak sangat aktif, imunisasi dasar lengkap, ASI eksklusif 6 bulan.",
        "Anak sangat pendek untuk usianya, nafsu makan minim, asupan protein hewani rendah.",
        "Bawah Garis Merah (BGM) pada KMS posyandu, berat badan 2 bulan tidak naik berturut-turut.",
        "Tinggi badan sesuai usia, motorik kasar baik, konsumsi makanan keluarga variatif.",
        "Gejala perawakan pendek, riwayat berat lahir rendah (BBLR 2.2 kg).",
        "Lingkar lengan atas (LiLA) 11.2 cm, asupan gizi makro kurang.",
        "Pertumbuhan stabil di kurva hijau KMS, stimulasi tumbuh kembang baik.",
        "Balita tampak lesu, nafsu makan berkurang, riwayat kontak batuk kronis keluarga.",
        "Gizi kurang dengan risiko gagal tumbuh, MPASI didominasi bubur karbohidrat."
    ]

    np.random.seed(42)

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        rows_s1 = []
        for i in range(30):
            is_male = np.random.rand() > 0.5
            nama = np.random.choice(names_male if is_male else names_female) + f"-{i+1}"
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
                "No_Registrasi": f"DEMO-{1001 + i}",
                "Nama_Balita": nama,
                "Jenis_Kelamin": "Laki-laki" if is_male else "Perempuan",
                "Usia_Bulan": usia,
                "Berat_Badan_Kg": bb,
                "Tinggi_Badan_Cm": tb,
                "ZScore_TB_U": haz,
                "ZScore_BB_U": waz,
                "ZScore_BB_TB": whz,
                "Puskesmas": np.random.choice(puskesmas_list),
                "Kelurahan_Desa": np.random.choice(desa_list),
                "Catatan_Pemeriksaan": catatan
            })
        df1 = pd.DataFrame(rows_s1)
        df1.to_excel(writer, sheet_name="Demo_Wilayah_Barat", index=False)

        rows_s2 = []
        for i in range(25):
            is_male = np.random.rand() > 0.5
            nama = np.random.choice(names_male if is_male else names_female) + f"-{i+101}"
            usia = int(np.random.randint(9, 48))
            haz = round(float(np.random.uniform(-3.5, 1.2)), 2)
            waz = round(float(np.random.uniform(-3.0, 1.0)), 2)
            whz = round(float(np.random.uniform(-2.8, 1.1)), 2)
            tb = round(52.0 + (usia * 0.88) + (haz * 3.0), 1)
            bb = round(4.0 + (usia * 0.25) + (waz * 1.3), 1)
            catatan = np.random.choice(notes_templates)

            rows_s2.append({
                "ID_Anak": f"DEMO-{201 + i}",
                "Nama_Lengkap": nama,
                "Gender": "L" if is_male else "P",
                "Umur_Bulan": usia,
                "BB_Kg": bb,
                "TB_Cm": tb,
                "HAZ": haz,
                "WAZ": waz,
                "WHZ": whz,
                "Posyandu": "Posyandu Demo",
                "Keterangan_Pemeriksaan": catatan
            })
        df2 = pd.DataFrame(rows_s2)
        df2.to_excel(writer, sheet_name="Demo_Wilayah_Timur", index=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting CURE-IndoBERT server on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
