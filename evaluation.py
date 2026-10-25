"""
Evaluation Module for Stunting Classification and Clustering
============================================================
Provides rigorous mathematical evaluation:
1. Classification Metrics: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted),
   F1-Score (Macro/Weighted), and Confusion Matrix.
   Strict requirement: Only computed when independent, verified ground-truth labels exist.
   Rejects error placeholders or self-evaluations.
2. Clustering Metrics: Silhouette Score, Cluster Purity, and Distribution Summaries.
   Uses the exact standardized feature matrix produced by CURE.
"""

from __future__ import annotations

import math
from typing import Dict, List, Any, Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


# Placeholders or error strings that must never be evaluated as genuine model classes
INVALID_PREDICTION_PLACEHOLDERS = {
    "model belum dilatih / checkpoint tidak tersedia",
    "checkpoint tidak tersedia (belum dilatih)",
    "error inference indobert",
    "unspecified",
    "none",
    "nan",
    "null",
    ""
}


def evaluate_classification(
    y_true: List[Union[str, int]],
    y_pred: List[Union[str, int]],
    labels: Optional[List[Union[str, int]]] = None
) -> Dict[str, Any]:
    """
    Evaluate multi-class classification against independent ground truth.

    Requirements:
    - Never evaluates self-generated predictions as ground truth.
    - Excludes invalid placeholders (e.g. 'Model Belum Dilatih').
    - Reports valid evaluated count vs excluded count.
    - Multi-class Macro and Weighted Precision, Recall, F1, and Confusion Matrix.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "status": "unavailable",
            "message": "Empty label arrays provided."
        }

    # Detect fraudulent self-evaluation (identical array passed as true and pred)
    if y_true is y_pred:
        return {
            "status": "invalid_self_evaluation",
            "message": "Model-generated predictions cannot be evaluated against themselves as ground truth."
        }

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )

    # Filter out invalid placeholders or ungrounded predictions
    clean_trues: List[str] = []
    clean_preds: List[str] = []
    excluded_count = 0

    for yt, yp in zip(y_true, y_pred):
        st_yt = str(yt).strip()
        st_yp = str(yp).strip()
        lower_yp = st_yp.lower()

        if lower_yp in INVALID_PREDICTION_PLACEHOLDERS or not st_yt or st_yt.lower() in INVALID_PREDICTION_PLACEHOLDERS:
            excluded_count += 1
            continue

        clean_trues.append(st_yt)
        clean_preds.append(st_yp)

    if len(clean_trues) == 0:
        return {
            "status": "unavailable",
            "message": "No valid evaluated observations with independent ground truth found.",
            "total_samples": len(y_true),
            "excluded_samples": excluded_count
        }

    num_samples = len(clean_trues)

    # Detect fraudulent self-evaluation (identical pointer/list passed)
    if clean_trues is clean_preds:
        return {
            "status": "invalid_self_evaluation",
            "message": "Model-generated predictions cannot be evaluated against themselves as ground truth."
        }

    unique_set = set(clean_trues).union(set(clean_preds))
    if labels is not None:
        unique_classes = [str(l).strip() for l in labels if str(l).strip() in unique_set]
        # Append any remaining classes
        for c in sorted(list(unique_set)):
            if c not in unique_classes:
                unique_classes.append(c)
    else:
        unique_classes = sorted(list(unique_set))

    correct_count = sum(1 for yt, yp in zip(clean_trues, clean_preds) if yt == yp)
    accuracy = float(correct_count / num_samples)

    # Confusion matrix
    n_c = len(unique_classes)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    cm = [[0 for _ in range(n_c)] for _ in range(n_c)]

    for yt, yp in zip(clean_trues, clean_preds):
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
        "excluded_count": excluded_count,
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
    Must be computed on the exact standardized feature matrix used by CURE.
    Returns float in [-1, 1], or None if cannot be computed.
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
    unique_labels = sorted(list(set(labs)))
    # Exclude noise / unassigned cluster ID (-1)
    unique_labels = [l for l in unique_labels if l >= 0]
    k = len(unique_labels)

    if k <= 1 or k >= N:
        return None

    # Subsample if N > 400 for responsiveness and reproducibility
    if N > 400:
        step = N / 400
        indices = [int(i * step) for i in range(400)]
        sub_rows = [rows[i] for i in indices]
        sub_labs = [labs[i] for i in indices]
    else:
        sub_rows = rows
        sub_labs = labs

    n_sub = len(sub_rows)
    s_vals = []

    for i in range(n_sub):
        ci = sub_labs[i]
        if ci < 0:
            continue
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
    X_scaled: Any,
    labels: Any,
    ground_truth_labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate cluster separation and profile distributions on standardized CURE feature space.
    """
    if HAS_NUMPY and isinstance(X_scaled, np.ndarray):
        rows = X_scaled.tolist()
    else:
        rows = [[float(v) for v in r] for r in X_scaled]

    if HAS_NUMPY and isinstance(labels, np.ndarray):
        labs = labels.tolist()
    else:
        labs = [int(l) for l in labels]

    N = len(rows)
    D = len(rows[0]) if N > 0 else 0
    unique_labels = sorted(list(set(labs)))
    valid_clusters = [l for l in unique_labels if l >= 0]
    k = len(valid_clusters)

    sil_score = compute_silhouette_score(rows, labs)

    cluster_counts = {int(c): sum(1 for l in labs if l == c) for c in unique_labels}
    cluster_percentages = {
        int(c): round(float(count / max(1, N) * 100), 2)
        for c, count in cluster_counts.items()
    }

    # Cluster profiles in standardized space
    cluster_profiles = []
    for c in valid_clusters:
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
        "status": "valid" if k >= 2 else "insufficient_clusters",
        "num_clusters": k,
        "sample_count": N,
        "silhouette_score": sil_score,
        "silhouette_interpretation": (
            "Silhouette score reflects geometric partition separation in standardized "
            "anthropometric feature space and does not constitute clinical proof."
        ),
        "cluster_distribution": cluster_counts,
        "cluster_percentages": cluster_percentages,
        "cluster_profiles": cluster_profiles
    }

    if ground_truth_labels is not None and len(ground_truth_labels) == N:
        gt = [str(x).strip() for x in ground_truth_labels]
        total_purity_matches = 0
        valid_n = 0
        for c in valid_clusters:
            c_gt = [gt[i] for i in range(N) if labs[i] == c and gt[i].lower() not in INVALID_PREDICTION_PLACEHOLDERS]
            if len(c_gt) > 0:
                valid_n += len(c_gt)
                counts: Dict[str, int] = {}
                for item in c_gt:
                    counts[item] = counts.get(item, 0) + 1
                total_purity_matches += max(counts.values())
        if valid_n > 0:
            result["purity_score"] = round(float(total_purity_matches / valid_n), 4)

    return result


def evaluate_cluster_labeling(
    cluster_ids: List[int],
    assigned_cluster_labels: List[str],
    ground_truth_labels: List[Union[str, int]]
) -> Dict[str, Any]:
    """
    Evaluate cluster-level semantic labeling against independent ground-truth labels.
    Explicitly distinguishes clustering evaluation (geometric) from semantic label assignment.

    Parameters
    ----------
    cluster_ids : List[int]
        Cluster IDs assigned by CURE.
    assigned_cluster_labels : List[str]
        Semantic cluster labels assigned by IndoBERT to each observation.
    ground_truth_labels : List[Union[str, int]]
        Independently observed true reference labels.

    Returns
    -------
    Dict containing:
        - cluster_profiles: majority class and purity for each cluster
        - overall_cluster_purity: float
        - sample_count: int
        - evaluated_clusters: int
    """
    if len(cluster_ids) != len(assigned_cluster_labels) or len(cluster_ids) != len(ground_truth_labels):
        raise ValueError("Length mismatch between cluster_ids, cluster_labels, and ground_truth_labels.")

    N = len(cluster_ids)
    clean_gt = [str(x).strip() for x in ground_truth_labels]
    valid_indices = [
        i for i in range(N)
        if cluster_ids[i] >= 0 and clean_gt[i].lower() not in INVALID_PREDICTION_PLACEHOLDERS
    ]

    if not valid_indices:
        return {
            "status": "unavailable",
            "message": "No valid observations with independent ground truth found for cluster labeling evaluation."
        }

    unique_cids = sorted(list(set(cluster_ids[i] for i in valid_indices)))
    cluster_evaluation_profiles = []
    total_majority_matches = 0

    for cid in unique_cids:
        c_gt = [clean_gt[i] for i in valid_indices if cluster_ids[i] == cid]
        c_labels = [assigned_cluster_labels[i] for i in valid_indices if cluster_ids[i] == cid]
        assigned_name = c_labels[0] if c_labels else f"Klaster {cid}"

        gt_counts: Dict[str, int] = {}
        for g in c_gt:
            gt_counts[g] = gt_counts.get(g, 0) + 1

        majority_class = max(gt_counts.keys(), key=lambda k: gt_counts[k])
        majority_count = gt_counts[majority_class]
        total_majority_matches += majority_count
        purity = round(float(majority_count / len(c_gt)), 4) if c_gt else 0.0

        cluster_evaluation_profiles.append({
            "cluster_id": cid,
            "assigned_cluster_label": assigned_name,
            "majority_ground_truth": majority_class,
            "cluster_size": len(c_gt),
            "purity": purity,
            "ground_truth_distribution": gt_counts
        })

    overall_purity = round(float(total_majority_matches / len(valid_indices)), 4)

    return {
        "status": "valid",
        "sample_count": len(valid_indices),
        "evaluated_clusters": len(unique_cids),
        "overall_cluster_purity": overall_purity,
        "cluster_profiles": cluster_evaluation_profiles
    }
