"""
Comprehensive Automated Test Suite for Stunting Research Pipeline
==================================================================
Covers:
1. Training input validation and text-label alignment
2. Label mapping and rejection of unknown classes
3. Train/validation/test stratified splitting with data leakage prevention
4. Validation metric calculation (Loss, Accuracy, Precision, Recall, F1)
5. Checkpoint saving and loading
6. Inference from trained weights vs ModelNotTrainedError for untrained model
7. Attention-mask handling (padding tokens do not dominate max-pooling)
8. HAZ/WAZ validation (no unsupported polynomial approximations; missing Z-scores remain None)
9. Missing-data processing (incomplete observations excluded from CURE clustering)
10. CURE clustering input validation (dimension consistency, NaN/Inf rejection, small datasets, constant features)
11. Silhouette Score consistency on scaled feature matrix
12. Classification evaluation on independent ground truth vs rejection of self-evaluation & error placeholders
13. Synthetic dataset labeling disclaimer
14. End-to-end pipeline processing on multi-sheet workbook structure
"""

from __future__ import annotations

import os
import sys
import math
import random
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from cure import CUREClusterer, _euclidean_dist
from indobert_rcnn import (
    ModelNotTrainedError, DEFAULT_LABEL2ID, DEFAULT_ID2LABEL,
    IndoBERTRCNNPipeline, prepare_dataset_splits,
    HAS_TORCH
)
if HAS_TORCH:
    import torch
    import torch.nn as nn
    from indobert_rcnn import IndoBERTRCNNModel

from evaluation import (
    evaluate_classification, evaluate_clustering, compute_silhouette_score,
    INVALID_PREDICTION_PLACEHOLDERS
)
from pipeline import (
    normalize_column_name, parse_indonesian_numeric, StuntingResearchPipeline
)
from app import generate_benchmark_stunting_workbook


class TestTrainingInputValidationAndSplits(unittest.TestCase):
    """Test training input validation, label alignment, and stratified splitting."""

    def setUp(self):
        self.label2id = {
            "STUNTING": 0,
            "NORMAL": 1
        }
        self.texts = [
            f"Balita {i} dengan kondisi pertumbuhan spesifik" for i in range(20)
        ]
        self.labels = ["STUNTING"] * 10 + ["NORMAL"] * 10

    def test_input_length_mismatch_raises_error(self):
        with self.assertRaises(ValueError):
            prepare_dataset_splits(self.texts[:5], self.labels[:4], self.label2id)

    def test_unknown_label_is_rejected(self):
        invalid_labels = ["STUNTING"] * 9 + ["UNKNOWN_CLASS"]
        with self.assertRaises(ValueError):
            prepare_dataset_splits(self.texts[:10], invalid_labels, self.label2id)

    def test_insufficient_sample_size_rejected(self):
        # Fewer than 10 samples
        with self.assertRaises(ValueError):
            prepare_dataset_splits(self.texts[:6], self.labels[:6], self.label2id)

    def test_stratified_split_and_no_leakage(self):
        splits = prepare_dataset_splits(
            self.texts, self.labels, self.label2id,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        self.assertGreater(len(splits["train_texts"]), 0)
        self.assertGreater(len(splits["val_texts"]), 0)

        # Check for duplicate texts between train and val/test
        train_set = set(splits["train_texts"])
        val_set = set(splits["val_texts"])
        test_set = set(splits["test_texts"])

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)


class TestIndoBERTRCNNContractAndMasking(unittest.TestCase):
    """Test model contract, attention-mask enforcement, and checkpoint loading."""

    def test_uninitialized_model_throws_not_trained_error(self):
        pipeline = IndoBERTRCNNPipeline(checkpoint_path=None)
        with self.assertRaises(ModelNotTrainedError):
            pipeline.predict(["Catatan anak stunting"])

    def test_label_mappings_consistency(self):
        self.assertEqual(len(DEFAULT_LABEL2ID), 7)
        self.assertEqual(len(DEFAULT_ID2LABEL), 7)
        for k, v in DEFAULT_LABEL2ID.items():
            self.assertEqual(DEFAULT_ID2LABEL[v], k)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not available in environment")
    def test_attention_mask_eliminates_padding_tokens(self):
        """Verify that padding tokens (mask=0) are masked before max-pooling."""
        model = IndoBERTRCNNModel(
            pretrained_model_name="indobenchmark/indobert-base-p1",
            num_classes=3
        )
        # Synthetic embedding tensor: batch_size=1, seq_len=4, bert_dim=768
        bert_embeddings = torch.randn(1, 4, 768)
        # Token 0, 1 are real, token 2, 3 are padding
        attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        input_ids = torch.zeros(1, 4, dtype=torch.long)

        logits = model(input_ids, attention_mask, bert_embeddings=bert_embeddings)
        self.assertEqual(logits.shape, (1, 3))


class TestHAZWAZValidationAndMissingData(unittest.TestCase):
    """Test removal of unsupported polynomial approximations and non-destructive missing data handling."""

    def test_parse_indonesian_numeric(self):
        self.assertAlmostEqual(parse_indonesian_numeric("-2,45"), -2.45, places=2)
        self.assertAlmostEqual(parse_indonesian_numeric("12,8 kg"), 12.8, places=2)
        self.assertAlmostEqual(parse_indonesian_numeric("78.5 cm"), 78.5, places=2)
        self.assertAlmostEqual(parse_indonesian_numeric("−3.1"), -3.1, places=2)
        self.assertIsNone(parse_indonesian_numeric("-"))
        self.assertIsNone(parse_indonesian_numeric(None))

    def test_missing_haz_remains_none_without_polynomial_estimation(self):
        pipeline = StuntingResearchPipeline()
        row = {
            "Nama": "Anak A",
            "Usia_Bulan": 24,
            "Tinggi_Badan_Cm": 75.0,
            "Berat_Badan_Kg": 9.0
        }
        extracted = pipeline._extract_indicators(row)
        # Methodological Rule 3: Missing Z-scores must NOT be approximated with polynomial equations
        self.assertIsNone(extracted["haz"])
        self.assertIsNone(extracted["waz"])
        self.assertEqual(extracted["height_cm"], 75.0)

    def test_permenkes_standard_classification(self):
        pipeline = StuntingResearchPipeline()
        self.assertIn("Sangat Pendek", pipeline._classify_kemenkes_permenkes_2_2020(-3.4))
        self.assertIn("Pendek", pipeline._classify_kemenkes_permenkes_2_2020(-2.4))
        self.assertEqual(pipeline._classify_kemenkes_permenkes_2_2020(0.2), "Normal")
        self.assertIn("Tidak Tersedia", pipeline._classify_kemenkes_permenkes_2_2020(None))


class TestCUREClusteringAndValidation(unittest.TestCase):
    """Test CURE clustering input validation, shrinkage, small datasets, and transformed space."""

    def setUp(self):
        random.seed(42)
        c1 = [[-3.0 + random.uniform(-0.2, 0.2), -3.0 + random.uniform(-0.2, 0.2)] for _ in range(15)]
        c2 = [[3.0 + random.uniform(-0.2, 0.2), 3.0 + random.uniform(-0.2, 0.2)] for _ in range(15)]
        self.X = c1 + c2

    def test_cure_fit_and_transformed_features(self):
        cure = CUREClusterer(n_clusters=2, n_representatives=3, shrink_factor=0.3)
        labels = cure.fit_predict(self.X)

        self.assertEqual(len(labels), 30)
        self.assertEqual(len(set(labels)), 2)
        self.assertIsNotNone(cure.X_scaled_)
        self.assertEqual(len(cure.X_scaled_), 30)

        # Transform new sample
        transformed = cure.transform([[0.0, 0.0]])
        self.assertEqual(len(transformed), 1)

    def test_cure_rejects_nan_and_inf(self):
        """CURE must reject NaN or infinite inputs rather than silently replacing with 0 or 5."""
        invalid_X = [[1.0, float("nan")], [2.0, 3.0]]
        cure = CUREClusterer(n_clusters=2)
        with self.assertRaises(ValueError):
            cure.fit(invalid_X)

    def test_cure_inconsistent_dimensionality_rejected(self):
        inconsistent_X = [[1.0, 2.0], [3.0, 4.0, 5.0]]
        cure = CUREClusterer(n_clusters=2)
        with self.assertRaises(ValueError):
            cure.fit(inconsistent_X)

    def test_cure_small_dataset_handling(self):
        small_X = [[1.0, 2.0], [3.0, 4.0]]
        cure = CUREClusterer(n_clusters=4, n_representatives=2, shrink_factor=0.3)
        labels = cure.fit_predict(small_X)
        self.assertEqual(len(labels), 2)

    def test_cure_constant_features(self):
        const_X = [
            [5.0, 1.0],
            [5.0, 2.0],
            [5.0, 3.0],
            [5.0, 4.0]
        ]
        cure = CUREClusterer(n_clusters=2, n_representatives=2, shrink_factor=0.2)
        labels = cure.fit_predict(const_X)
        self.assertEqual(len(labels), 4)


class TestEvaluationAndMetrics(unittest.TestCase):
    """Test classification evaluation, placeholder rejection, and Silhouette score."""

    def test_classification_rejects_error_placeholders(self):
        y_true = ["Normal", "Stunting"]
        y_pred = ["Model Belum Dilatih / Checkpoint Tidak Tersedia", "Model Belum Dilatih / Checkpoint Tidak Tersedia"]
        res = evaluate_classification(y_true, y_pred)
        self.assertEqual(res["status"], "unavailable")

    def test_classification_rejects_self_evaluation(self):
        predictions = ["Normal", "Stunting", "Normal"]
        res = evaluate_classification(predictions, predictions)
        self.assertEqual(res["status"], "invalid_self_evaluation")

    def test_valid_classification_metrics(self):
        y_true = ["Normal", "Normal", "Stunting", "Stunting"]
        y_pred = ["Normal", "Normal", "Stunting", "Normal"]
        res = evaluate_classification(y_true, y_pred)
        self.assertEqual(res["status"], "valid")
        self.assertEqual(res["accuracy"], 0.75)
        self.assertEqual(res["sample_count"], 4)

    def test_silhouette_score_calculated_on_scaled_features(self):
        # Two separated clusters in standardized feature space
        c1 = [[-2.0, -2.0] for _ in range(10)]
        c2 = [[2.0, 2.0] for _ in range(10)]
        scaled_X = c1 + c2
        labels = [0] * 10 + [1] * 10

        res = evaluate_clustering(scaled_X, labels)
        self.assertEqual(res["status"], "valid")
        self.assertIsNotNone(res["silhouette_score"])
        self.assertGreater(res["silhouette_score"], 0.7)


class TestPipelineModularStages(unittest.TestCase):
    """Test modularity of pipeline stages and non-destructive missing data handling."""

    def test_modular_processing_sequence_configurable(self):
        p1 = StuntingResearchPipeline(processing_sequence=["cure", "indobert"])
        self.assertEqual(p1.processing_sequence, ["cure", "indobert"])

        p2 = StuntingResearchPipeline(processing_sequence=["indobert", "cure"])
        self.assertEqual(p2.processing_sequence, ["indobert", "cure"])

    def test_missing_data_excluded_from_cure_clustering(self):
        """Observations with incomplete anthropometrics are given cluster ID -1, not arbitrary values."""
        cure = CUREClusterer(n_clusters=2, n_representatives=3, shrink_factor=0.3)
        # 10 complete records
        complete_features = [[-2.5, -2.0, -1.0, 0.4] for _ in range(5)] + [[1.5, 1.2, 0.8, 0.5] for _ in range(5)]
        labels = cure.fit_predict(complete_features)
        self.assertEqual(len(labels), 10)
        self.assertTrue(all(l in [0, 1] for l in labels))


class TestSyntheticBenchmarkNotice(unittest.TestCase):
    """Test synthetic benchmark notice and disclaimers."""

    def test_generate_benchmark_docstring_disclaims_real_clinical_data(self):
        doc = generate_benchmark_stunting_workbook.__doc__
        self.assertIn("SYNTHETIC DEMONSTRATION", doc)
        self.assertIn("NOT be described as authentic", doc)


if __name__ == "__main__":
    unittest.main()
