from __future__ import annotations

import os
import re
import math
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from cure import CUREClusterer
from indobert_rcnn import IndoBERTRCNNPipeline, ModelNotTrainedError, DEFAULT_LABEL_DESCRIPTIONS
from evaluation import evaluate_classification, evaluate_clustering

logger = logging.getLogger("StuntingResearchPipeline")


def normalize_column_name(col: str) -> str:
    """Standardize column names to snake_case lowercase."""
    s = str(col or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _is_nan(val: Any) -> bool:
    if val is None:
        return True
    if pd is not None:
        try:
            return bool(pd.isna(val))
        except Exception:
            pass
    if isinstance(val, float) and (val != val or math.isnan(val)):
        return True
    return False


def parse_indonesian_numeric(val: Any) -> Optional[float]:
    """
    Robust numeric parser for Indonesian Excel data:
    - Handles comma decimal: '-2,45' -> -2.45
    - Strips units: '75.2 cm', '11,4 kg', '-2.1 SD'
    - Replaces unicode dashes and symbols
    """
    if _is_nan(val):
        return None
    if isinstance(val, (int, float)):
        return float(val) if not _is_nan(val) else None

    s = str(val).strip()
    if s in {"-", "--", "null", "nan", "None", ""}:
        return None

    # Replace unicode dashes with standard minus
    s = re.sub(r"[\u2212\u2013\u2014\u2015\uFE63\uFF0D]", "-", s)
    # Replace decimal comma with dot
    s = s.replace(",", ".")

    match = re.search(r"[-+]?\s*\d*\.?\d+", s)
    if match:
        try:
            num = float(match.group(0).replace(" ", ""))
            return num if not _is_nan(num) else None
        except ValueError:
            return None
    return None


class StuntingResearchPipeline:
    """
    End-to-End Modular Research Pipeline for Child Stunting Analysis.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        n_representatives: int = 5,
        shrink_factor: float = 0.3,
        sample_size: int = 1500,
        checkpoint_path: Optional[str] = None
    ):
        self.n_clusters = int(n_clusters)
        self.n_representatives = int(n_representatives)
        self.shrink_factor = float(shrink_factor)
        self.sample_size = int(sample_size)
        self.checkpoint_path = checkpoint_path

        # IndoBERT + RCNN Pipeline
        self.classifier_pipeline: Optional[IndoBERTRCNNPipeline] = None
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                self.classifier_pipeline = IndoBERTRCNNPipeline(checkpoint_path=checkpoint_path)
                logger.info("IndoBERT + RCNN classifier initialized from checkpoint: %s", checkpoint_path)
            except Exception as e:
                logger.warning("Could not initialize IndoBERT checkpoint (%s): %s", checkpoint_path, e)
                self.classifier_pipeline = None

    def _detect_header_row(self, df_raw: pd.DataFrame) -> int:
        """
        Scan top rows to find genuine header index, skipping Puskesmas title banners.
        """
        keywords = {
            "nama", "balita", "anak", "bayi", "nik", "no", "nomor",
            "usia", "umur", "bulan", "bln", "jk", "gender", "bb", "tb",
            "pb", "haz", "waz", "whz", "puskesmas", "posyandu", "desa",
            "berat", "tinggi", "panjang", "catatan", "keterangan"
        }

        max_scan = min(15, len(df_raw))
        best_idx = 0
        max_matches = -1

        for idx in range(max_scan):
            row_vals = [normalize_column_name(str(x)) for x in df_raw.iloc[idx].dropna()]
            match_count = 0
            for val in row_vals:
                tokens = val.split("_")
                if any(t in keywords for t in tokens):
                    match_count += 1
            if match_count > max_matches:
                max_matches = match_count
                best_idx = idx

        return best_idx

    def _extract_indicators(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extract anthropometrics and clinical notes from a row with fuzzy key matching.
        """
        norm_map = {normalize_column_name(k): v for k, v in row.items()}

        haz, waz, whz = None, None, None
        age_months = None
        weight_kg = None
        height_cm = None
        gender = None
        notes = []
        ground_truth = None

        for k, v in norm_map.items():
            if pd.isna(v):
                continue
            # Z-scores
            if any(term in k for term in ["zscore_tb_u", "haz", "tb_u_zscore", "zs_tb_u", "tb_u"]):
                if "status" not in k and "kategori" not in k:
                    val = parse_indonesian_numeric(v)
                    if val is not None and -6.0 <= val <= 6.0:
                        haz = val
            elif any(term in k for term in ["zscore_bb_u", "waz", "bb_u_zscore", "zs_bb_u", "bb_u"]):
                if "status" not in k and "kategori" not in k:
                    val = parse_indonesian_numeric(v)
                    if val is not None and -6.0 <= val <= 6.0:
                        waz = val
            elif any(term in k for term in ["zscore_bb_tb", "whz", "bb_tb_zscore", "zs_bb_tb", "bb_tb"]):
                if "status" not in k and "kategori" not in k:
                    val = parse_indonesian_numeric(v)
                    if val is not None and -6.0 <= val <= 6.0:
                        whz = val

            # Age
            elif any(term in k for term in ["usia_bulan", "umur_bulan", "usia_bln", "umur_bln", "usia", "umur"]):
                val = parse_indonesian_numeric(v)
                if val is not None and 0 <= val <= 120:
                    age_months = val

            # Height & Weight
            elif any(term in k for term in ["tinggi_badan", "panjang_badan", "tb_cm", "pb_cm", "tinggi", "tb", "pb"]):
                val = parse_indonesian_numeric(v)
                if val is not None and 30 <= val <= 150:
                    height_cm = val
            elif any(term in k for term in ["berat_badan", "bb_kg", "berat", "bb"]):
                val = parse_indonesian_numeric(v)
                if val is not None and 1.5 <= val <= 50:
                    weight_kg = val

            # Gender
            elif any(term in k for term in ["jenis_kelamin", "gender", "jk"]):
                s_gen = str(v).strip().upper()
                if "L" in s_gen or "PRIA" in s_gen or "LAKI" in s_gen:
                    gender = "L"
                elif "P" in s_gen or "WANITA" in s_gen or "PEREMPUAN" in s_gen:
                    gender = "P"

            # Clinical notes
            elif any(term in k for term in ["catatan", "keterangan", "keluhan", "diagnosa", "riwayat", "notes"]):
                notes.append(str(v).strip())

            # Independent Ground Truth column
            elif any(term in k for term in ["label_manual", "ground_truth", "target_stunting", "diagnosis_dokter"]):
                ground_truth = str(v).strip()

        # WHO 2006 reference approximation if Z-scores were missing but measurements exist
        if haz is None and height_cm is not None and age_months is not None:
            is_boy = gender != "P"
            h_med = (50.0 + 1.85 * age_months - 0.035 * age_months**2) if is_boy else (49.0 + 1.80 * age_months - 0.035 * age_months**2)
            h_sd = 2.1 + 0.04 * age_months
            haz = round(float((height_cm - h_med) / h_sd), 2)

        if waz is None and weight_kg is not None and age_months is not None:
            is_boy = gender != "P"
            w_med = (3.3 + 0.52 * age_months - 0.005 * age_months**2) if is_boy else (3.2 + 0.49 * age_months - 0.005 * age_months**2)
            w_sd = 0.5 + 0.03 * age_months
            waz = round(float((weight_kg - w_med) / w_sd), 2)

        return {
            "haz": haz,
            "waz": waz,
            "whz": whz,
            "age_months": age_months,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "gender": gender,
            "notes": " ".join(notes).strip(),
            "ground_truth": ground_truth
        }

    def _classify_kemenkes_permenkes_2_2020(self, haz: Optional[float], waz: Optional[float], whz: Optional[float]) -> str:
        """
        Permenkes RI No. 2/2020 standard anthropometric category (rule-based clinical reference).
        """
        if haz is None:
            return "Data Antropometri Belum Lengkap"
        if haz < -3.0:
            return "Sangat Pendek (Severely Stunted)"
        elif -3.0 <= haz < -2.0:
            return "Pendek (Stunted)"
        elif -2.0 <= haz <= 3.0:
            return "Normal"
        else:
            return "Tinggi"

    def _generate_guideline_recommendations(
        self,
        kemenkes_cat: str,
        haz: Optional[float],
        waz: Optional[float],
        whz: Optional[float],
        age_months: Optional[float]
    ) -> Tuple[str, str, str, str]:
        """
        Generate non-diagnostic recommendations grounded in Permenkes No. 2/2020 & Kemenkes guidelines.
        """
        if "Sangat Pendek" in kemenkes_cat:
            prioritas = "Tinggi (Intervensi Gizi Spesifik)"
            nutrisi = "Pemberian PMT Pemulihan tinggi protein hewani (telur, ikan, daging), suplementasi Taburia/Zink."
            medis = "Pemeriksaan komorbiditas infeksi (TB balita, ISPA, diare berulang) ke Puskesmas/RSUD."
            monitoring = "Evaluasi penimbangan & pengukuran panjang badan setiap 1 minggu di Posyandu/Puskesmas."
        elif "Pendek" in kemenkes_cat:
            prioritas = "Sedang (Pemberian Makanan Tambahan PMT)"
            nutrisi = "Edukasi MPASI kaya protein hewani dan mikronutrien, konsumsi minimal 1 butir telur per hari."
            medis = "Pemberian vitamin A dosis tinggi sesuai jadwal, obat cacing, dan skrining sanitasi jamban sehat."
            monitoring = "Pemantauan kurva KMS berkala setiap 2 minggu."
        elif "Normal" in kemenkes_cat:
            prioritas = "Pemeliharaan (Pemantauan Rutin)"
            nutrisi = "Pertahankan gizi seimbang Isi Piringku, pemenuhan protein nabati dan hewani seimbang."
            medis = "Kelengkapan imunisasi dasar dan imunisasi lanjutan, akses air bersih layak minum."
            monitoring = "Penimbangan bulanan rutin di Posyandu setiap bulan."
        else:
            prioritas = "Rendah (Pola Makan Seimbang)"
            nutrisi = "Batasi asupan gula sederhana, perbanyak aktivitas fisik aktif."
            medis = "Konsultasi ahli gizi untuk pengaturan pola makan keluarga seimbang."
            monitoring = "Pemantauan rutin KMS tiap bulan."

        return prioritas, nutrisi, medis, monitoring

    def process_excel(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Execute full research pipeline across all workbook sheets.
        """
        input_p = Path(input_path)
        output_p = Path(output_path)

        if not input_p.exists():
            raise FileNotFoundError(f"Uploaded Excel file not found: {input_p}")

        excel_file = pd.ExcelFile(str(input_p))
        sheet_names = excel_file.sheet_names

        processed_sheets_summary = []
        total_rows_all = 0

        with pd.ExcelWriter(str(output_p), engine="openpyxl") as writer:
            for s_name in sheet_names:
                df_raw = pd.read_excel(excel_file, sheet_name=s_name, header=None)
                if df_raw.empty or len(df_raw) < 2:
                    continue

                header_idx = self._detect_header_row(df_raw)
                headers = [str(c).strip() for c in df_raw.iloc[header_idx]]
                data_rows = df_raw.iloc[header_idx + 1 :].copy()
                data_rows.columns = headers
                data_rows = data_rows.dropna(how="all").reset_index(drop=True)

                N = len(data_rows)
                if N == 0:
                    continue

                total_rows_all += N

                # Extract indicators
                indicators_list = [self._extract_indicators(row) for _, row in data_rows.iterrows()]

                # Build numerical feature matrix for CURE: [HAZ, WAZ, WHZ, normalized_age]
                feature_matrix = []
                for ind in indicators_list:
                    h = ind["haz"] if ind["haz"] is not None else 0.0
                    w = ind["waz"] if ind["waz"] is not None else 0.0
                    wh = ind["whz"] if ind["whz"] is not None else 0.0
                    age_norm = (ind["age_months"] / 60.0) if ind["age_months"] is not None else 0.5
                    feature_matrix.append([h, w, wh, age_norm])

                X = np.array(feature_matrix, dtype=np.float64)

                # Execute CURE Clustering
                k_cure = min(self.n_clusters, N)
                clusterer = CUREClusterer(
                    n_clusters=k_cure,
                    n_representatives=self.n_representatives,
                    shrink_factor=self.shrink_factor
                )
                cure_labels = clusterer.fit_predict(X)

                # IndoBERT + RCNN Inference (if checkpoint loaded)
                notes_texts = [ind["notes"] for ind in indicators_list]
                indobert_preds = []
                indobert_confs = []

                if self.classifier_pipeline and self.classifier_pipeline.is_loaded:
                    try:
                        preds = self.classifier_pipeline.predict(notes_texts)
                        for p in preds:
                            indobert_preds.append(p["label_description"])
                            indobert_confs.append(p["confidence"])
                    except Exception as e:
                        logger.error("IndoBERT inference failure on sheet %s: %s", s_name, e)
                        indobert_preds = ["Error Inference IndoBERT"] * N
                        indobert_confs = [None] * N
                else:
                    # Explicit honest status: no fake heuristic predictions
                    indobert_preds = ["Model Belum Dilatih / Checkpoint Tidak Tersedia"] * N
                    indobert_confs = [None] * N

                # Permenkes & Guideline Recommendations
                kemenkes_cats = []
                prioritas_list = []
                nutrisi_list = []
                medis_list = []
                monitoring_list = []

                for ind in indicators_list:
                    cat = self._classify_kemenkes_permenkes_2_2020(ind["haz"], ind["waz"], ind["whz"])
                    prio, nutr, med, mon = self._generate_guideline_recommendations(
                        cat, ind["haz"], ind["waz"], ind["whz"], ind["age_months"]
                    )
                    kemenkes_cats.append(cat)
                    prioritas_list.append(prio)
                    nutrisi_list.append(nutr)
                    medis_list.append(med)
                    monitoring_list.append(mon)

                # Assemble enriched DataFrame
                df_out = data_rows.copy()
                df_out["CURE_Cluster_ID"] = cure_labels
                df_out["Kemenkes_Kategori_Gizi"] = kemenkes_cats
                df_out["IndoBERT_RCNN_Prediksi"] = indobert_preds
                df_out["IndoBERT_Confidence"] = indobert_confs
                df_out["Prioritas_Intervensi"] = prioritas_list
                df_out["Saran_Asupan_Nutrisi"] = nutrisi_list
                df_out["Tindakan_Medis_Sanitasi"] = medis_list
                df_out["Jadwal_Monitoring"] = monitoring_list

                # Evaluate clustering
                clustering_eval = evaluate_clustering(X, cure_labels)

                # Evaluate classification ONLY IF valid independent ground truth exists
                ground_truths = [ind["ground_truth"] for ind in indicators_list if ind["ground_truth"] is not None]
                has_independent_gt = (len(ground_truths) == N and any(g != "" for g in ground_truths))

                classification_eval = None
                if has_independent_gt and self.classifier_pipeline and self.classifier_pipeline.is_loaded:
                    valid_preds = [p for p in indobert_preds]
                    classification_eval = evaluate_classification(ground_truths, valid_preds)

                # Write sheet to output
                sheet_clean_name = s_name[:31]
                df_out.to_excel(writer, sheet_name=sheet_clean_name, index=False)

                summary = {
                    "sheet_name": s_name,
                    "rows": N,
                    "cure_clusters": k_cure,
                    "silhouette_score": clustering_eval["silhouette_score"],
                    "cluster_distribution": clustering_eval["cluster_distribution"],
                    "has_independent_gt": has_independent_gt,
                    "classification_eval": classification_eval
                }
                processed_sheets_summary.append(summary)

        return {
            "status": "success",
            "total_rows": total_rows_all,
            "processed_sheets": len(processed_sheets_summary),
            "sheets": processed_sheets_summary,
            "cure_parameters": {
                "n_clusters": self.n_clusters,
                "n_representatives": self.n_representatives,
                "shrink_factor": self.shrink_factor
            }
        }
