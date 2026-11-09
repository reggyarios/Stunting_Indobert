"""
Stunting Research Pipeline: CURE Clustering & IndoBERT + RCNN Integration
==========================================================================
Coordinates:
1. Multi-sheet Excel workbook ingestion & schema normalization
2. Permenkes RI No. 2/2020 anthropometric validation (NO unsupported polynomial approximations)
3. Non-destructive missing-value handling: incomplete records excluded from CURE rather than arbitrarily imputed
4. Classical CURE clustering on validated feature representations with exposed standardized space
5. Real IndoBERT + RCNN inference with loaded checkpoint and error transparency
6. Evaluation metrics calculation on independent reference data
7. Clean multi-sheet Excel export preserving original data integrity
"""

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


NOTE_TEMPLATES = [
    "Anak {jk} usia {u} bulan, berat badan {bb} kg, {ukur} badan {tb} cm diukur {pos}. "
    "Z-score TB/U {haz}, BB/U {waz}, BB/TB {whz}.",
    "Hasil penimbangan balita {jk} berumur {u} bulan: BB {bb} kg dan {ukur} {tb} cm ({pos}); "
    "z-score tinggi menurut umur {haz} SD, berat menurut umur {waz} SD, berat menurut tinggi {whz} SD.",
    "Pengukuran {pos} pada anak {jk} {u} bulan menunjukkan {ukur} {tb} cm dan berat {bb} kg "
    "dengan z-score TB/U {haz}, BB/U {waz}, dan BB/TB {whz}.",
]


def build_note(ind: Dict[str, Any], template_idx: int = 0) -> str:
    """
    Narasi antropometri (catatan pemeriksaan) untuk data e-PPGBM yang tidak memiliki
    kolom catatan. Hanya memuat ukuran dan z-score, TIDAK memuat kata status gizi.
    Mengembalikan "" bila ada indikator yang tidak tersedia.
    """
    need = ("gender", "age_months", "weight_kg", "height_cm", "haz", "waz", "whz")
    if any(ind.get(k) is None for k in need):
        return ""
    pos = ind.get("position") or ("Terlentang" if ind["age_months"] < 24 else "Berdiri")
    tel = pos == "Terlentang"
    f = lambda x, nd: f"{x:.{nd}f}".replace(".", ",")
    return NOTE_TEMPLATES[template_idx % len(NOTE_TEMPLATES)].format(
        jk="laki-laki" if ind["gender"] == "L" else "perempuan",
        u=int(ind["age_months"]), bb=f(ind["weight_kg"], 1), tb=f(ind["height_cm"], 1),
        ukur="panjang" if tel else "tinggi", pos="telentang" if tel else "berdiri",
        haz=f(ind["haz"], 2), waz=f(ind["waz"], 2), whz=f(ind["whz"], 2),
    )


def permenkes_reference_label(haz: Optional[float], waz: Optional[float], whz: Optional[float]) -> Optional[str]:
    """
    Label acuan operasional 7 kelas (taksonomi DEFAULT_LABEL2ID) dari ambang
    Permenkes RI No. 2/2020, prioritas indeks TB/U. Dipakai sebagai acuan evaluasi.
    """
    if haz is None or waz is None or whz is None:
        return None
    if haz < -3:
        return "STUNTING_BERAT_KOMORBID" if (waz < -2 or whz < -2) else "STUNTING_KRONIS"
    if haz < -2:
        return "STUNTING_SEDANG"
    if whz < -3:
        return "GIZI_BURUK_AKUT"
    if whz < -2 or waz < -2:
        return "GIZI_KURANG_RISIKO"
    if whz > 1:
        return "RISIKO_GIZI_LEBIH"
    return "NORMAL_OPTIMAL"


class StuntingResearchPipeline:
    """
    End-to-End Modular Research Pipeline for Child Stunting Analysis.

    Design Principles:
    - Decoupled stages: CURE clustering (anthropometric) and IndoBERT+RCNN (textual)
      can be run independently or in configured order.
    - Zero fabrication: no unsupported polynomial Z-score equations.
    - Defensible missing-value handling: incomplete observations are excluded from CURE
      rather than corrupted by arbitrary 0.0 substitutions.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        n_representatives: int = 5,
        shrink_factor: float = 0.3,
        sample_size: int = 1500,
        checkpoint_path: Optional[str] = None,
        processing_sequence: Optional[List[str]] = None,
        random_state: int = 42,
        reference_date: Optional[Any] = None
    ):
        self.n_clusters = int(n_clusters)
        self.n_representatives = int(n_representatives)
        self.shrink_factor = float(shrink_factor)
        self.sample_size = int(sample_size)
        self.checkpoint_path = checkpoint_path
        self.processing_sequence = processing_sequence or ["cure", "indobert"]
        self.random_state = int(random_state)
        # Tanggal ukur acuan untuk menghitung usia dari tanggal lahir bila kolom usia tidak ada
        self.reference_date = pd.Timestamp(reference_date) if (reference_date and pd is not None) else None

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
        """Scan top rows to find genuine header index, skipping Puskesmas title banners."""
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

        Methodological Rule:
        - Only read valid existing Z-scores recorded in the dataset.
        - Validate biological plausibility range: -6.0 <= Z <= +6.0.
        - If Z-score is absent, mark as None (unavailable).
        - DO NOT invent or substitute approximate polynomial formulas as official WHO Z-scores.
        """
        norm_map = {normalize_column_name(k): v for k, v in row.items()}

        haz: Optional[float] = None
        waz: Optional[float] = None
        whz: Optional[float] = None
        age_months: Optional[float] = None
        weight_kg: Optional[float] = None
        height_cm: Optional[float] = None
        gender: Optional[str] = None
        dob: Optional[Any] = None
        position: Optional[str] = None
        notes: List[str] = []
        ground_truth: Optional[str] = None

        for k, v in norm_map.items():
            if _is_nan(v):
                continue

            # Z-scores: Height-for-Age (HAZ / TB/U)
            if any(term in k for term in ["zscore_tb_u", "haz", "tb_u_zscore", "zs_tb_u", "tb_u"]):
                if "status" not in k and "kategori" not in k and "ket" not in k:
                    val = parse_indonesian_numeric(v)
                    if val is not None and -6.0 <= val <= 6.0:
                        haz = val

            # Z-scores: Weight-for-Age (WAZ / BB/U)
            elif any(term in k for term in ["zscore_bb_u", "waz", "bb_u_zscore", "zs_bb_u", "bb_u"]):
                if "status" not in k and "kategori" not in k and "ket" not in k:
                    val = parse_indonesian_numeric(v)
                    if val is not None and -6.0 <= val <= 6.0:
                        waz = val

            # Z-scores: Weight-for-Height (WHZ / BB/TB)
            elif any(term in k for term in ["zscore_bb_tb", "whz", "bb_tb_zscore", "zs_bb_tb", "bb_tb"]):
                if "status" not in k and "kategori" not in k and "ket" not in k:
                    val = parse_indonesian_numeric(v)
                    if val is not None and -6.0 <= val <= 6.0:
                        whz = val

            # Age in months
            elif any(term in k for term in ["usia_bulan", "umur_bulan", "usia_bln", "umur_bln", "usia", "umur"]):
                val = parse_indonesian_numeric(v)
                if val is not None and 0.0 <= val <= 120.0:
                    age_months = val

            # Height & Weight raw measurements
            elif any(term in k for term in ["tinggi_badan", "panjang_badan", "tb_cm", "pb_cm", "tinggi", "tb", "pb"]):
                val = parse_indonesian_numeric(v)
                if val is not None and 30.0 <= val <= 150.0:
                    height_cm = val
            elif any(term in k for term in ["berat_badan", "bb_kg", "berat", "bb"]):
                val = parse_indonesian_numeric(v)
                if val is not None and 1.5 <= val <= 50.0:
                    weight_kg = val

            # Gender
            elif any(term in k for term in ["jenis_kelamin", "gender", "jk"]):
                s_gen = str(v).strip().upper()
                if "L" in s_gen or "PRIA" in s_gen or "LAKI" in s_gen:
                    gender = "L"
                elif "P" in s_gen or "WANITA" in s_gen or "PEREMPUAN" in s_gen:
                    gender = "P"

            # Date of birth (e-PPGBM: "Tgl Lahir") and measurement position ("Cara Ukur")
            elif any(term in k for term in ["tgl_lahir", "tanggal_lahir"]):
                ts = pd.to_datetime(v, errors="coerce") if pd is not None else None
                dob = None if ts is None or pd.isna(ts) else ts
            elif any(term in k for term in ["cara_ukur", "posisi"]):
                s_pos = str(v).strip().lower()
                position = "Terlentang" if ("terlentang" in s_pos or "telentang" in s_pos) else (
                    "Berdiri" if "berdiri" in s_pos else None)

            # Clinical notes / observations
            elif any(term in k for term in ["catatan", "keterangan", "keluhan", "diagnosa", "riwayat", "notes"]):
                notes.append(str(v).strip())

            # Independent Ground Truth column
            elif any(term in k for term in ["label_manual", "ground_truth", "target_stunting", "diagnosis_dokter"]):
                ground_truth = str(v).strip()

        # Usia dihitung dari tanggal lahir bila tidak ada kolom usia (data e-PPGBM)
        if age_months is None and dob is not None and self.reference_date is not None:
            days = (self.reference_date - dob).days
            if 0 <= days <= 1826:
                age_months = round(days / 30.4375, 2)

        # NOTE: In accordance with research methodology Task 3, unsupported polynomial
        # approximations for HAZ and WAZ have been completely removed.
        # If HAZ or WAZ are missing from the dataset, they remain None.

        return {
            "haz": haz,
            "waz": waz,
            "whz": whz,
            "age_months": age_months,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "gender": gender,
            "position": position,
            "notes": " ".join(notes).strip(),
            "ground_truth": ground_truth
        }

    def _classify_kemenkes_permenkes_2_2020(self, haz: Optional[float]) -> str:
        """
        Permenkes RI No. 2/2020 standard anthropometric category based on Height-for-Age (TB/U).
        Strictly separated from neural model predictions.
        """
        if haz is None:
            return "Indikator TB/U Tidak Tersedia"
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
        kemenkes_cat: str
    ) -> Tuple[str, str, str, str]:
        """
        General public health nutrition guidance aligned with Permenkes RI No. 2/2020.
        Explicitly non-diagnostic and without unsupported medical prescriptions.
        """
        if "Sangat Pendek" in kemenkes_cat:
            prioritas = "Tinggi (Intervensi Gizi Spesifik)"
            nutrisi = "Konseling PMT berbasis pangan lokal kaya protein hewani sesuai pedoman gizi seimbang."
            medis = "Skrining kesehatan dan penelusuran faktor risiko tumbuh kembang di fasilitas pelayanan kesehatan."
            monitoring = "Pemantauan berkala berat badan dan panjang/tinggi badan di Posyandu/Puskesmas."
        elif "Pendek" in kemenkes_cat:
            prioritas = "Sedang (Edukasi Gizi & Pemantauan)"
            nutrisi = "Edukasi pemenuhan gizi seimbang harian dan variasi lauk hewani bagi balita."
            medis = "Pemeriksaan kesehatan rutin dan skrining sanitasi lingkungan keluarga."
            monitoring = "Penimbangan dan pengukuran berkala di Posyandu."
        elif "Normal" in kemenkes_cat:
            prioritas = "Pemeliharaan (Pemantauan Rutin)"
            nutrisi = "Pertahankan pola makan seimbang Isi Piringku dan kecukupan air bersih."
            medis = "Lengkapi imunisasi dasar dan pantau stimulasi tumbuh kembang anak."
            monitoring = "Pemantauan pertumbuhan rutin setiap bulan di Posyandu."
        else:
            prioritas = "Rendah (Pola Makan Seimbang)"
            nutrisi = "Edukasi pola makan beragam dan pembatasan makanan berkadar gula/lemak berlebih."
            medis = "Konsultasi tumbuh kembang balita sesuai jadwal."
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

        if pd is None:
            raise ImportError("Pandas is required for Excel workbook processing.")

        excel_file = pd.ExcelFile(str(input_p))
        sheet_names = excel_file.sheet_names

        processed_sheets_summary = []
        total_rows_all = 0

        with excel_file, pd.ExcelWriter(str(output_p), engine="openpyxl") as writer:
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
                # Data e-PPGBM tidak memiliki catatan teks: susun narasi antropometri
                for i, ind in enumerate(indicators_list):
                    if not ind["notes"]:
                        ind["notes"] = build_note(ind, template_idx=i)

                # Build numerical feature matrix for CURE: [HAZ, WAZ, WHZ, normalized_age]
                # Methodological Rule (Task 4):
                # Separate complete observations from incomplete observations.
                # Do NOT replace missing measurements with arbitrary 0.0 or 0.5.
                valid_cure_indices: List[int] = []
                cure_features: List[List[float]] = []

                for idx, ind in enumerate(indicators_list):
                    if (
                        ind["haz"] is not None
                        and ind["waz"] is not None
                        and ind["whz"] is not None
                        and ind["age_months"] is not None
                    ):
                        valid_cure_indices.append(idx)
                        cure_features.append([
                            float(ind["haz"]),
                            float(ind["waz"]),
                            float(ind["whz"]),
                            float(ind["age_months"] / 60.0)
                        ])

                n_complete = len(valid_cure_indices)
                n_incomplete = N - n_complete

                cure_labels = [-1] * N  # -1 represents unclustered due to incomplete features
                clustering_eval: Dict[str, Any] = {
                    "status": "unavailable",
                    "reason": "Insufficient complete anthropometric records for clustering."
                }

                if n_complete >= 2:
                    k_cure = min(self.n_clusters, n_complete)
                    clusterer = CUREClusterer(
                        n_clusters=k_cure,
                        n_representatives=self.n_representatives,
                        shrink_factor=self.shrink_factor
                    )
                    # CURE (Guha et al., 1998): tahap hierarkis pada sampel acak berukuran
                    # sample_size, lalu seluruh data diberi label ke titik representatif terdekat.
                    if n_complete > self.sample_size:
                        rng = np.random.default_rng(self.random_state)
                        sample_idx = np.sort(rng.choice(n_complete, self.sample_size, replace=False))
                        clusterer.fit([cure_features[i] for i in sample_idx])
                        fitted_labels = [int(x) for x in clusterer.predict(cure_features)]
                    else:
                        fitted_labels = [int(x) for x in clusterer.fit_predict(cure_features)]

                    for orig_idx, clab in zip(valid_cure_indices, fitted_labels):
                        cure_labels[orig_idx] = int(clab)

                    # Evaluate clustering strictly on the standardized feature matrix produced by CURE
                    scaled_matrix = clusterer.X_scaled_
                    clustering_eval = evaluate_clustering(scaled_matrix, clusterer.labels_)
                else:
                    logger.warning(
                        "Sheet '%s': only %d complete observations; minimum 2 required for CURE clustering.",
                        s_name, n_complete
                    )

                # Cluster-level semantic labeling via IndoBERT + RCNN (Research Methodology)
                # Group clinical notes by CURE cluster ID
                cluster_notes_map: Dict[int, List[str]] = {}
                for orig_idx, clab in zip(valid_cure_indices, fitted_labels if n_complete >= 2 else []):
                    cid = int(clab)
                    if cid not in cluster_notes_map:
                        cluster_notes_map[cid] = []
                    cluster_notes_map[cid].append(indicators_list[orig_idx]["notes"])

                cluster_semantic_labels: Dict[int, Dict[str, Any]] = {}
                cluster_label_map: Dict[int, str] = {}
                cluster_conf_map: Dict[int, Optional[float]] = {}

                notes_texts = [ind["notes"] for ind in indicators_list]
                record_preds_raw: Optional[List[Dict[str, Any]]] = None
                if self.classifier_pipeline and self.classifier_pipeline.is_loaded and self.classifier_pipeline.is_trained:
                    try:
                        record_preds_raw = self.classifier_pipeline.predict(notes_texts)
                    except Exception as e:
                        logger.error("IndoBERT record-level inference error on sheet %s: %s", s_name, e)

                if record_preds_raw is not None:
                    try:
                        cluster_preds_map: Dict[int, List[Dict[str, Any]]] = {}
                        for orig_idx, clab in zip(valid_cure_indices, fitted_labels if n_complete >= 2 else []):
                            if notes_texts[orig_idx].strip():
                                cluster_preds_map.setdefault(int(clab), []).append(record_preds_raw[orig_idx])
                        cluster_semantic_labels = {
                            cid: self.classifier_pipeline.label_cluster(
                                cluster_notes_map[cid], cluster_id=cid, predictions=cluster_preds_map.get(cid, []))
                            for cid in cluster_notes_map
                        }
                        for cid, info in cluster_semantic_labels.items():
                            cluster_label_map[cid] = info["cluster_label"]
                            cluster_conf_map[cid] = info["confidence"]
                    except Exception as e:
                        logger.error("IndoBERT cluster labeling error on sheet %s: %s", s_name, e)
                        for cid in cluster_notes_map:
                            cluster_label_map[cid] = f"Klaster {cid} (Error Labeling)"
                            cluster_conf_map[cid] = None
                else:
                    for cid in cluster_notes_map:
                        cluster_label_map[cid] = f"Klaster {cid} (Label Semantik Belum Dilatih / Checkpoint Tidak Tersedia)"
                        cluster_conf_map[cid] = None

                # Unclustered / incomplete records
                cluster_label_map[-1] = "Tidak Terklaster (Data Antropometri Tidak Lengkap)"
                cluster_conf_map[-1] = None

                # Individual record-level IndoBERT + RCNN inference (distinguished from cluster-level label)
                indobert_record_preds: List[str] = []
                indobert_record_confs: List[Optional[float]] = []

                if record_preds_raw is not None:
                    for p in record_preds_raw:
                        indobert_record_preds.append(p["label_description"])
                        indobert_record_confs.append(p["confidence"])
                elif self.classifier_pipeline and self.classifier_pipeline.is_trained:
                    indobert_record_preds = ["Error Inference IndoBERT"] * N
                    indobert_record_confs = [None] * N
                else:
                    indobert_record_preds = ["Model Belum Dilatih / Checkpoint Tidak Tersedia"] * N
                    indobert_record_confs = [None] * N

                # Permenkes RI No. 2/2020 Standard Anthropometric Categories & Guidelines
                kemenkes_cats = []
                prioritas_list = []
                nutrisi_list = []
                medis_list = []
                monitoring_list = []

                for ind in indicators_list:
                    cat = self._classify_kemenkes_permenkes_2_2020(ind["haz"])
                    prio, nutr, med, mon = self._generate_guideline_recommendations(cat)
                    kemenkes_cats.append(cat)
                    prioritas_list.append(prio)
                    nutrisi_list.append(nutr)
                    medis_list.append(med)
                    monitoring_list.append(mon)

                # Assemble enriched DataFrame with explicit separation between Cluster_ID and Cluster_Label
                df_out = data_rows.copy()
                df_out["Usia_Bulan_Dihitung"] = [ind["age_months"] for ind in indicators_list]
                df_out["Catatan_Input_IndoBERT"] = notes_texts
                df_out["Kategori_Permenkes_7Kelas"] = [
                    DEFAULT_LABEL_DESCRIPTIONS.get(permenkes_reference_label(ind["haz"], ind["waz"], ind["whz"]), "Indikator Tidak Lengkap")
                    for ind in indicators_list
                ]
                df_out["CURE_Cluster_ID"] = cure_labels
                df_out["CURE_Cluster_Label"] = [cluster_label_map.get(cid, f"Klaster {cid}") for cid in cure_labels]
                df_out["Cluster_Label_Confidence"] = [cluster_conf_map.get(cid, None) for cid in cure_labels]
                df_out["IndoBERT_Record_Prediksi"] = indobert_record_preds
                df_out["IndoBERT_Record_Confidence"] = indobert_record_confs
                df_out["Kemenkes_Kategori_Gizi"] = kemenkes_cats
                df_out["Prioritas_Intervensi"] = prioritas_list
                df_out["Saran_Asupan_Nutrisi"] = nutrisi_list
                df_out["Tindakan_Medis_Sanitasi"] = medis_list
                df_out["Jadwal_Monitoring"] = monitoring_list

                # Evaluate classification ONLY IF valid independent ground truth exists
                ground_truths = [ind["ground_truth"] for ind in indicators_list if ind["ground_truth"] is not None]
                has_independent_gt = (len(ground_truths) == N and any(g != "" for g in ground_truths))

                classification_eval = None
                if has_independent_gt and self.classifier_pipeline and self.classifier_pipeline.is_trained:
                    classification_eval = evaluate_classification(ground_truths, indobert_record_preds)

                # Write sheet to output
                sheet_clean_name = s_name[:31]
                df_out.to_excel(writer, sheet_name=sheet_clean_name, index=False)

                unique_cids = sorted(list(set(l for l in cure_labels if l >= 0)))
                semantic_summary = [
                    {
                        "cluster_id": cid,
                        "cluster_label": cluster_label_map.get(cid),
                        "confidence": cluster_conf_map.get(cid),
                        "size": sum(1 for l in cure_labels if l == cid)
                    }
                    for cid in unique_cids
                ]

                summary = {
                    "sheet_name": s_name,
                    "rows": N,
                    "complete_anthropometric_rows": n_complete,
                    "incomplete_anthropometric_rows": n_incomplete,
                    "cure_clusters_formed": len(unique_cids),
                    "cluster_semantic_labels": semantic_summary,
                    "silhouette_score": clustering_eval.get("silhouette_score"),
                    "clustering_eval": clustering_eval,
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
                "shrink_factor": self.shrink_factor,
                "sample_size": self.sample_size
            },
            "processing_sequence": self.processing_sequence
        }
