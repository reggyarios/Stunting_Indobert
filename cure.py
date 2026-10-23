"""
CURE (Clustering Using Representatives) Algorithm
=================================================
Reference: Guha, S., Rastogi, R., & Shim, K. (1998). CURE: An Efficient Clustering
Algorithm for Large Databases. ACM SIGMOD Record, 27(2), 73-84.

Key Principles:
1. Represents each cluster using a fixed number (c) of well-scattered points.
2. Shrinks representative points towards the cluster centroid by alpha (0 <= alpha <= 1).
3. Cluster distance = minimum distance between any pair of shrunk representatives.
4. Robust against outliers and non-spherical clusters.
5. Fully compatible with both NumPy environments and pure-Python standard environments.
"""

import math
from typing import List, Tuple, Dict, Optional, Any, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def _euclidean_dist(p1: Union[List[float], Any], p2: Union[List[float], Any]) -> float:
    """Euclidean distance between two vectors."""
    if HAS_NUMPY and isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        return float(np.linalg.norm(p1 - p2))
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(p1, p2)))


class CUREClusterer:
    """
    Standard CURE (Clustering Using Representatives) Clustering Algorithm.

    Parameters
    ----------
    n_clusters : int, default=4
        The target number of clusters to form (k).
    n_representatives : int, default=5
        The number of representative points (c) maintained for each cluster.
    shrink_factor : float, default=0.3
        Shrinkage parameter (alpha) between 0 and 1. Representative points
        are shrunk towards the cluster centroid: p' = p + alpha * (centroid - p).
    """

    def __init__(
        self,
        n_clusters: int = 4,
        n_representatives: int = 5,
        shrink_factor: float = 0.3
    ) -> None:
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if n_representatives < 1:
            raise ValueError(f"n_representatives must be >= 1, got {n_representatives}")
        if not (0.0 <= shrink_factor <= 1.0):
            raise ValueError(f"shrink_factor must be between 0.0 and 1.0, got {shrink_factor}")

        self.n_clusters = int(n_clusters)
        self.n_representatives = int(n_representatives)
        self.shrink_factor = float(shrink_factor)

        # Fitted attributes
        self.labels_: Optional[Union[List[int], Any]] = None
        self.centroids_: List[List[float]] = []
        self.representatives_: List[List[List[float]]] = []
        self.cluster_sizes_: List[int] = []
        self.feature_means_: Optional[List[float]] = None
        self.feature_stds_: Optional[List[float]] = None
        self.is_fitted: bool = False

    def _select_and_shrink_reps(
        self,
        points: List[List[float]],
        centroid: List[float],
        c: int,
        alpha: float
    ) -> List[List[float]]:
        """
        Select c well-scattered representative points from cluster points
        and shrink them towards centroid by alpha.
        """
        num_pts = len(points)
        if num_pts == 0:
            return [[float(x) for x in centroid]]

        if num_pts <= c:
            selected = [[float(x) for x in p] for p in points]
        else:
            # 1st representative: point farthest from centroid
            dists_to_cent = [_euclidean_dist(p, centroid) for p in points]
            first_idx = int(max(range(num_pts), key=lambda i: dists_to_cent[i]))
            selected = [[float(x) for x in points[first_idx]]]
            selected_indices = {first_idx}

            # Subsequent representatives: max-min distance from current representatives
            for _ in range(1, c):
                max_min_dist = -1.0
                best_idx = -1
                for i in range(num_pts):
                    if i in selected_indices:
                        continue
                    min_d = min(_euclidean_dist(points[i], sel) for sel in selected)
                    if min_d > max_min_dist:
                        max_min_dist = min_d
                        best_idx = i

                if best_idx != -1:
                    selected.append([float(x) for x in points[best_idx]])
                    selected_indices.add(best_idx)
                else:
                    break

        # Shrink representative points towards centroid
        # p' = p + alpha * (centroid - p)
        shrunk_reps = [
            [p[j] + alpha * (centroid[j] - p[j]) for j in range(len(p))]
            for p in selected
        ]
        return shrunk_reps

    def _cluster_distance(
        self,
        reps_u: List[List[float]],
        reps_v: List[List[float]]
    ) -> float:
        """Compute inter-cluster distance as min distance between shrunk representatives."""
        min_dist = float("inf")
        for p in reps_u:
            for q in reps_v:
                d = _euclidean_dist(p, q)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def _to_clean_list(self, X: Any) -> Tuple[List[List[float]], int, int]:
        """Convert input array or list of lists into clean Python float matrix."""
        if HAS_NUMPY and isinstance(X, np.ndarray):
            X_clean = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
            rows = X_clean.tolist()
        else:
            rows = []
            for r in X:
                clean_r = []
                for val in r:
                    try:
                        v = float(val)
                        if math.isnan(v):
                            v = 0.0
                    except (ValueError, TypeError):
                        v = 0.0
                    clean_r.append(v)
                rows.append(clean_r)

        N = len(rows)
        D = len(rows[0]) if N > 0 else 0
        return rows, N, D

    def fit(self, X: Any) -> "CUREClusterer":
        """Fit CURE clustering on feature matrix X."""
        rows, N, D = self._to_clean_list(X)
        if N == 0:
            raise ValueError("Input feature matrix X has 0 samples.")

        # Compute means and stds for normalization
        means = [0.0] * D
        for r in rows:
            for j in range(D):
                means[j] += r[j]
        means = [m / N for m in means]

        stds = [0.0] * D
        for r in rows:
            for j in range(D):
                diff = r[j] - means[j]
                stds[j] += diff * diff
        stds = [math.sqrt(s / N) for s in stds]
        # Avoid division by zero for constant features
        stds = [s if s > 1e-7 else 1.0 for s in stds]

        self.feature_means_ = means
        self.feature_stds_ = stds

        # Scale features
        scaled = [
            [(r[j] - means[j]) / stds[j] for j in range(D)]
            for r in rows
        ]

        target_k = min(self.n_clusters, N)
        if N <= target_k:
            labels_list = list(range(N))
            self.labels_ = np.array(labels_list) if HAS_NUMPY else labels_list
            self.centroids_ = [list(scaled[i]) for i in range(N)]
            self.representatives_ = [[[float(x) for x in scaled[i]]] for i in range(N)]
            self.cluster_sizes_ = [1] * N
            self.is_fitted = True
            return self

        # Initialize clusters: each point is initially its own cluster
        clusters: Dict[int, Dict[str, Any]] = {}
        for i in range(N):
            cent = list(scaled[i])
            clusters[i] = {
                "id": i,
                "points_idx": [i],
                "centroid": cent,
                "reps": [list(cent)]
            }

        active_ids = list(range(N))

        # Initial pairwise distances
        dist_cache: Dict[Tuple[int, int], float] = {}
        for i in range(len(active_ids)):
            id_i = active_ids[i]
            for j in range(i + 1, len(active_ids)):
                id_j = active_ids[j]
                dist_cache[(id_i, id_j)] = _euclidean_dist(scaled[id_i], scaled[id_j])

        next_cluster_id = N

        # Hierarchical merge
        while len(active_ids) > target_k:
            best_pair: Optional[Tuple[int, int]] = None
            min_dist = float("inf")

            for pair, d in dist_cache.items():
                if d < min_dist:
                    min_dist = d
                    best_pair = pair

            if best_pair is None:
                break

            u_id, v_id = best_pair
            u = clusters[u_id]
            v = clusters[v_id]

            merged_pts_idx = u["points_idx"] + v["points_idx"]
            merged_pts = [scaled[idx] for idx in merged_pts_idx]

            sz_u = len(u["points_idx"])
            sz_v = len(v["points_idx"])
            new_cent = [
                (sz_u * u["centroid"][j] + sz_v * v["centroid"][j]) / (sz_u + sz_v)
                for j in range(D)
            ]

            new_reps = self._select_and_shrink_reps(
                merged_pts,
                new_cent,
                self.n_representatives,
                self.shrink_factor
            )

            new_cluster = {
                "id": next_cluster_id,
                "points_idx": merged_pts_idx,
                "centroid": new_cent,
                "reps": new_reps
            }

            active_ids.remove(u_id)
            active_ids.remove(v_id)
            del clusters[u_id]
            del clusters[v_id]

            keys_to_delete = [
                k for k in dist_cache
                if k[0] == u_id or k[1] == u_id or k[0] == v_id or k[1] == v_id
            ]
            for k in keys_to_delete:
                del dist_cache[k]

            clusters[next_cluster_id] = new_cluster
            for other_id in active_ids:
                other_reps = clusters[other_id]["reps"]
                d = self._cluster_distance(new_reps, other_reps)
                pair_key = (min(next_cluster_id, other_id), max(next_cluster_id, other_id))
                dist_cache[pair_key] = d

            active_ids.append(next_cluster_id)
            next_cluster_id += 1

        self.centroids_ = []
        self.representatives_ = []
        self.cluster_sizes_ = []
        labels_list = [0] * N

        for final_idx, c_id in enumerate(active_ids):
            c_info = clusters[c_id]
            self.centroids_.append(c_info["centroid"])
            self.representatives_.append(c_info["reps"])
            self.cluster_sizes_.append(len(c_info["points_idx"]))
            for p_idx in c_info["points_idx"]:
                labels_list[p_idx] = final_idx

        self.labels_ = np.array(labels_list) if HAS_NUMPY else labels_list
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> Union[List[int], Any]:
        """Assign each sample in X to the nearest cluster representative."""
        if not self.is_fitted:
            raise RuntimeError("CUREClusterer must be fitted before predict.")

        rows, M, D = self._to_clean_list(X)
        means = self.feature_means_
        stds = self.feature_stds_

        scaled = [
            [(r[j] - means[j]) / stds[j] for j in range(D)]
            for r in rows
        ]

        preds = [0] * M
        for i in range(M):
            pt = scaled[i]
            min_dist = float("inf")
            best_c = 0
            for c_idx, reps in enumerate(self.representatives_):
                for rep in reps:
                    d = _euclidean_dist(pt, rep)
                    if d < min_dist:
                        min_dist = d
                        best_c = c_idx
            preds[i] = best_c

        return np.array(preds) if HAS_NUMPY else preds

    def fit_predict(self, X: Any) -> Union[List[int], Any]:
        """Fit CURE and return cluster labels."""
        self.fit(X)
        return self.labels_
