"""
Evaluation Module for Stunting Classification and Clustering
============================================================
Provides rigorous mathematical evaluation:
1. Classification Metrics: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted),
   F1-Score (Macro/Weighted), and Confusion Matrix.
   Strict requirement: Only computed when independent, verified ground-truth labels exist.
2. Clustering Metrics: Silhouette Score, Cluster Purity, and Distribution Summaries.
3. Dual-mode support: Seamlessly executes with NumPy or pure Python standard library.
"""

import math
from typing import Dict, List, Any, Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def evaluate_classification(
    y_true: List[Union[str, int]],
    y_pred: List[Union[str, int]],
    labels: Optional[List[Union[str, int]]] = None
) -> Dict[str, Any]:
    """
    Evaluate multi-class classification against independent ground truth.

    Parameters
    ----------
    y_true : list of true reference labels (must be independently observed)
    y_pred : list of model predictions
    labels : optional explicit class order

    Returns
    -------
    dict containing accuracy, precision, recall, f1, and confusion matrix.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "status": "unavailable",
            "message": "Empty label arrays provided."
        }

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    y_t = [str(x).strip() for x in y_true]
    y_p = [str(x).strip() for x in y_pred]
    num_samples = len(y_t)

    # Unique classes
    unique_set = set(y_t).union(set(y_p))
    if labels is not None:
        unique_classes = [str(l).strip() for l in labels if str(l).strip() in unique_set]
    else:
        unique_classes = sorted(list(unique_set))

    correct_count = sum(1 for yt, yp in zip(y_t, y_p) if yt == yp)
    accuracy = float(correct_count / num_samples)

    # Confusion matrix
    n_c = len(unique_classes)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    cm = [[0 for _ in range(n_c)] for _ in range(n_c)]

    for yt, yp in zip(y_t, y_p):
        if yt in class_to_idx and yp in class_to_idx:
            cm[class_to_idx[yt]][class_to_idx[yp]] += 1

    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c in unique_classes:
        idx = class_to_idx[c]
        tp = cm[idx][idx]
        fp = sum(cm[r][idx] for r in range(n_c)) - tp
        fn = sum(cm[idx][c_idx] for c_idx in range(n_c)) - tp
        support = sum(cm[idx][c_idx] for c_idx in range(n_c))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        per_class_metrics[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": int(support)
        }

    total_support = max(1, sum(supports))
    macro_precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
    macro_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0

    weighted_precision = float(sum(p * s for p, s in zip(precisions, supports)) / total_support)
    weighted_recall = float(sum(r * s for r, s in zip(recalls, supports)) / total_support)
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_support)

    return {
        "status": "valid",
        "sample_count": num_samples,
        "classes": unique_classes,
        "accuracy": round(accuracy, 4),
        "macro_avg": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1_score": round(macro_f1, 4)
        },
        "weighted_avg": {
            "precision": round(weighted_precision, 4),
            "recall": round(weighted_recall, 4),
            "f1_score": round(weighted_f1, 4)
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm
    }


def _vec_dist(p1: List[float], p2: List[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(p1, p2)))


def compute_silhouette_score(X: Any, labels: Any) -> Optional[float]:
    """
    Calculate the Silhouette Coefficient for cluster validation.
    Returns float in [-1, 1], or None if cannot be computed.
    """
    # Convert input to list of lists
    if HAS_NUMPY and isinstance(X, np.ndarray):
        rows = X.tolist()
    else:
        rows = [[float(v) for v in r] for r in X]

    if HAS_NUMPY and isinstance(labels, np.ndarray):
        labs = labels.tolist()
    else:
        labs = [int(l) for l in labels]

    N = len(rows)
    unique_labels = sorted(list(set(labs)))
    k = len(unique_labels)

    if k <= 1 or k >= N:
        return None

    # Subsample if N > 300 for speed
    if N > 300:
        step = N / 300
        indices = [int(i * step) for i in range(300)]
        sub_rows = [rows[i] for i in indices]
        sub_labs = [labs[i] for i in indices]
    else:
        sub_rows = rows
        sub_labs = labs

    n_sub = len(sub_rows)
    s_vals = []

    for i in range(n_sub):
        ci = sub_labs[i]
        same_indices = [idx for idx, l in enumerate(sub_labs) if l == ci and idx != i]
        if len(same_indices) == 0:
            s_vals.append(0.0)
            continue

        # a(i): mean distance to points in same cluster
        a_i = sum(_vec_dist(sub_rows[i], sub_rows[idx]) for idx in same_indices) / len(same_indices)

        # b(i): min mean distance to other clusters
        b_i = float("inf")
        for other_c in unique_labels:
            if other_c == ci:
                continue
            other_indices = [idx for idx, l in enumerate(sub_labs) if l == other_c]
            if len(other_indices) == 0:
                continue
            mean_dist = sum(_vec_dist(sub_rows[i], sub_rows[idx]) for idx in other_indices) / len(other_indices)
            if mean_dist < b_i:
                b_i = mean_dist

        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 0 else 0.0
        s_vals.append(s_i)

    return round(float(sum(s_vals) / len(s_vals)), 4) if s_vals else None


def evaluate_clustering(
    X: Any,
    labels: Any,
    ground_truth_labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate cluster structure, separation, and purity.
    """
    if HAS_NUMPY and isinstance(X, np.ndarray):
        rows = X.tolist()
    else:
        rows = [[float(v) for v in r] for r in X]

    if HAS_NUMPY and isinstance(labels, np.ndarray):
        labs = labels.tolist()
    else:
        labs = [int(l) for l in labels]

    N = len(rows)
    D = len(rows[0]) if N > 0 else 0
    unique_labels = sorted(list(set(labs)))
    k = len(unique_labels)

    sil_score = compute_silhouette_score(rows, labs)

    # Cluster sizes & proportions
    cluster_counts = {int(c): sum(1 for l in labs if l == c) for c in unique_labels}
    cluster_percentages = {
        int(c): round(float(count / max(1, N) * 100), 2)
        for c, count in cluster_counts.items()
    }

    # Cluster centroids and spread
    cluster_profiles = []
    for c in unique_labels:
        c_pts = [rows[i] for i in range(N) if labs[i] == c]
        if len(c_pts) > 0:
            cent = [sum(pt[j] for pt in c_pts) / len(c_pts) for j in range(D)]
            spread = sum(_vec_dist(pt, cent) for pt in c_pts) / len(c_pts)
        else:
            spread = 0.0

        profile = {
            "cluster_id": int(c),
            "size": len(c_pts),
            "percentage": cluster_percentages[int(c)],
            "spread": round(float(spread), 4)
        }
        cluster_profiles.append(profile)

    result: Dict[str, Any] = {
        "num_clusters": k,
        "sample_count": N,
        "silhouette_score": sil_score,
        "cluster_distribution": cluster_counts,
        "cluster_percentages": cluster_percentages,
        "cluster_profiles": cluster_profiles
    }

    # Purity score if ground truth is available
    if ground_truth_labels is not None and len(ground_truth_labels) == N:
        gt = [str(x).strip() for x in ground_truth_labels]
        total_purity_matches = 0
        for c in unique_labels:
            c_gt = [gt[i] for i in range(N) if labs[i] == c]
            if len(c_gt) > 0:
                counts: Dict[str, int] = {}
                for item in c_gt:
                    counts[item] = counts.get(item, 0) + 1
                total_purity_matches += max(counts.values())
        purity = float(total_purity_matches / max(1, N))
        result["purity_score"] = round(purity, 4)

    return result
