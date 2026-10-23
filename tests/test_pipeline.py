"""
Automated Test Suite for CURE Clustering, IndoBERT+RCNN, Preprocessing, and Evaluation
======================================================================================
Runs with Python standard unittest runner:
    python3 -m unittest tests/test_pipeline.py
"""

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
    IndoBERTRCNNPipeline
)
from evaluation import (
    evaluate_classification, evaluate_clustering, compute_silhouette_score
)
from pipeline import (
    normalize_column_name, parse_indonesian_numeric, StuntingResearchPipeline
)


class TestPreprocessingAndParsers(unittest.TestCase):
    """Test column normalization, Indonesian decimal parsing, and value extraction."""

    def test_normalize_column_name(self):
        self.assertEqual(normalize_column_name("Z-Score TB/U"), "z_score_tb_u")
        self.assertEqual(normalize_column_name(" Berat Badan (kg) "), "berat_badan_kg")
        self.assertEqual(normalize_column_name("Jenis Kelamin (L/P)"), "jenis_kelamin_l_p")
        self.assertEqual(normalize_column_name(""), "")

    def test_parse_indonesian_numeric(self):
        # Indonesian comma decimal
        self.assertAlmostEqual(parse_indonesian_numeric("-2,45"), -2.45, places=2)
        self.assertAlmostEqual(parse_indonesian_numeric("12,8 kg"), 12.8, places=2)
        self.assertAlmostEqual(parse_indonesian_numeric("78.5 cm"), 78.5, places=2)
        # Unicode minus signs
        self.assertAlmostEqual(parse_indonesian_numeric("−3.1"), -3.1, places=2)
        self.assertAlmostEqual(parse_indonesian_numeric("–2,4"), -2.4, places=2)
        # Null values
        self.assertIsNone(parse_indonesian_numeric("-"))
        self.assertIsNone(parse_indonesian_numeric("--"))
        self.assertIsNone(parse_indonesian_numeric(None))
        self.assertIsNone(parse_indonesian_numeric("null"))


class TestCUREClustering(unittest.TestCase):
    """Test CURE clustering representative point selection, shrinkage, and merging."""

    def setUp(self):
        random.seed(42)
        # Generate 3 distinct synthetic 2D clusters
        c1 = [[-3.0 + random.uniform(-0.3, 0.3), -3.0 + random.uniform(-0.3, 0.3)] for _ in range(25)]
        c2 = [[0.0 + random.uniform(-0.3, 0.3), 0.0 + random.uniform(-0.3, 0.3)] for _ in range(25)]
        c3 = [[3.0 + random.uniform(-0.3, 0.3), 3.0 + random.uniform(-0.3, 0.3)] for _ in range(25)]
        self.X = c1 + c2 + c3

    def test_cure_fit_and_clusters_count(self):
        cure = CUREClusterer(n_clusters=3, n_representatives=4, shrink_factor=0.3)
        labels = cure.fit_predict(self.X)

        self.assertEqual(len(labels), len(self.X))
        unique_labels = set(labels)
        self.assertEqual(len(unique_labels), 3)
        self.assertEqual(len(cure.centroids_), 3)
        self.assertEqual(len(cure.representatives_), 3)

    def test_cure_shrinkage_geometry(self):
        """Verify that representative points are strictly moved towards centroid by alpha."""
        cure = CUREClusterer(n_clusters=2, n_representatives=3, shrink_factor=0.5)
        # Line cluster
        pts = [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ]
        cent = [3.0, 3.0]
        reps = cure._select_and_shrink_reps(pts, cent, c=2, alpha=0.5)

        self.assertEqual(len(reps), 2)
        # Each rep must be closer to centroid than extreme points
        orig_max_dist = _euclidean_dist(pts[0], cent)
        for r in reps:
            dist_to_cent = _euclidean_dist(r, cent)
            self.assertLessEqual(dist_to_cent, orig_max_dist)

    def test_cure_small_dataset_handling(self):
        """When N <= k, CURE must safely partition without errors."""
        small_X = [[1.0, 2.0], [3.0, 4.0]]
        cure = CUREClusterer(n_clusters=4, n_representatives=3, shrink_factor=0.3)
        labels = cure.fit_predict(small_X)
        self.assertEqual(len(labels), 2)

    def test_cure_constant_features(self):
        """Handle zero-variance / constant features without division-by-zero."""
        const_X = [
            [5.0, 1.0],
            [5.0, 2.0],
            [5.0, 3.0],
            [5.0, 4.0],
            [5.0, 5.0]
        ]
        cure = CUREClusterer(n_clusters=2, n_representatives=2, shrink_factor=0.2)
        labels = cure.fit_predict(const_X)
        self.assertEqual(len(labels), 5)
        self.assertTrue(all(isinstance(l, (int, float)) for l in labels))

    def test_cure_predict_out_of_sample(self):
        """Predict method assigns new point to closest shrunk representative."""
        cure = CUREClusterer(n_clusters=2, n_representatives=3, shrink_factor=0.3)
        cure.fit(self.X[:50])
        preds = cure.predict(self.X[50:60])
        self.assertEqual(len(preds), 10)
        self.assertTrue(all(p in [0, 1] for p in preds))


    def test_consistent_label_mappings(self):
        self.assertEqual(len(DEFAULT_LABEL2ID), 7)
        self.assertEqual(len(DEFAULT_ID2LABEL), 7)
        for k, v in DEFAULT_LABEL2ID.items():
            self.assertEqual(DEFAULT_ID2LABEL[v], k)

    def test_uninitialized_model_throws_not_trained_error(self):
        """Ensure uncheckpointed pipeline raises ModelNotTrainedError and does NOT silently fake output."""
        pipeline = IndoBERTRCNNPipeline(checkpoint_path=None)
        with self.assertRaises(ModelNotTrainedError):
            pipeline.predict(["Anak tampak sangat kurus dan KMS garis merah"])


class TestEvaluationMetrics(unittest.TestCase):
    """Test classification metrics and clustering silhouette evaluation."""

    def test_classification_metrics_exactness(self):
        y_true = ["Normal", "Normal", "Stunting", "Stunting", "Stunting"]
        y_pred = ["Normal", "Normal", "Stunting", "Stunting", "Normal"]

        res = evaluate_classification(y_true, y_pred)
        self.assertEqual(res["status"], "valid")
        self.assertEqual(res["sample_count"], 5)
        self.assertAlmostEqual(res["accuracy"], 0.80, places=2)
        cm = res["confusion_matrix"]
        self.assertEqual(len(cm), 2)

    def test_silhouette_score_calculation(self):
        # Two well-separated clusters
        c1 = [[0.0, 0.0] for _ in range(10)]
        c2 = [[10.0, 10.0] for _ in range(10)]
        X = c1 + c2
        labels = [0] * 10 + [1] * 10

        score = compute_silhouette_score(X, labels)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.8)

    def test_clustering_evaluation_profiles(self):
        random.seed(123)
        X = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(20)]
        labels = [0] * 10 + [1] * 10
        res = evaluate_clustering(X, labels)

        self.assertEqual(res["num_clusters"], 2)
        self.assertEqual(res["sample_count"], 20)
        self.assertEqual(res["cluster_distribution"][0], 10)
        self.assertEqual(res["cluster_percentages"][0], 50.0)


class TestStuntingPipelineRules(unittest.TestCase):
    """Test Permenkes No. 2/2020 guideline classification and indicator logic."""

    def test_kemenkes_standard_classification(self):
        pipeline = StuntingResearchPipeline()
        self.assertIn("Sangat Pendek", pipeline._classify_kemenkes_permenkes_2_2020(-3.4, -2.0, -1.0))
        self.assertIn("Pendek", pipeline._classify_kemenkes_permenkes_2_2020(-2.4, -1.0, 0.0))
        self.assertEqual(pipeline._classify_kemenkes_permenkes_2_2020(0.2, 0.1, 0.0), "Normal")

    def test_guideline_recommendations(self):
        pipeline = StuntingResearchPipeline()
        prio, nutr, med, mon = pipeline._generate_guideline_recommendations("Sangat Pendek", -3.4, -2.5, -1.5, 24)
        self.assertIn("Tinggi", prio)
        self.assertIn("PMT Pemulihan", nutr)
        self.assertIn("komorbiditas", med)
        self.assertIn("1 minggu", mon)


if __name__ == "__main__":
    unittest.main()
