"""
core/prototype.py — Module 1: Federated Private Prototype Extraction.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from config import Config
from core.privacy import add_dp_noise, privacy_report


def labeled_prototypes(features, labels, n_classes):
    d = features.shape[1]
    P = np.zeros((n_classes, d), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int64)
    for k in range(n_classes):
        m = labels == k
        counts[k] = int(m.sum())
        if m.sum() > 0:
            P[k] = features[m].mean(axis=0)
    return P, counts


def unlabeled_prototypes(features, n_clusters, n_init=20, max_iter=500, seed=42):
    km = KMeans(n_clusters=n_clusters, n_init=n_init,
                max_iter=max_iter, random_state=seed)
    assignments = km.fit_predict(normalize(features, norm="l2"))
    counts = np.bincount(assignments, minlength=n_clusters)
    return km.cluster_centers_.astype(np.float32), assignments.astype(np.int64), counts.astype(np.int64)


class PrototypeBank:
    def __init__(self, cfg: Config, side: str):
        assert side in ("t", "d")
        self.cfg  = cfg
        self.side = side
        self.rng  = np.random.default_rng(cfg.seed)

        self.raw_prototypes   = None
        self.noisy_prototypes = None
        self.assignments      = None
        self.sigma            = None

    def build_d_side(self, features, labels) -> "PrototypeBank":
        self.raw_prototypes, counts = labeled_prototypes(
            features, labels, self.cfg.data.n_classes)
        self.noisy_prototypes, self.sigma = add_dp_noise(
            self.raw_prototypes, self.cfg.privacy.epsilon,
            self.cfg.privacy.delta, self.cfg.privacy.max_norm, self.rng,
            counts=counts)
        return self

    def build_t_side(self, features) -> "PrototypeBank":
        cfg_p = self.cfg.prototype
        centers, assignments, counts = unlabeled_prototypes(
            features, cfg_p.n_clusters, cfg_p.kmeans_n_init,
            cfg_p.kmeans_max_iter, self.cfg.seed)
        self.raw_prototypes  = centers
        self.assignments     = assignments
        # Target-side clusters are consumed locally for semantic assignment and
        # are not transmitted as a federated message. Keep them clean; DP is
        # applied to the transmitted d-side class prototypes.
        self.noisy_prototypes = self.raw_prototypes.copy()
        self.sigma = np.zeros((len(self.raw_prototypes), 1), dtype=np.float32)
        return self

    def transmit(self) -> np.ndarray:
        return self.noisy_prototypes.copy()

    def privacy_report(self) -> dict:
        return privacy_report(
            self.cfg.privacy.epsilon, self.cfg.privacy.delta,
            self.cfg.privacy.max_norm, self.sigma)
