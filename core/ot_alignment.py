"""
core/ot_alignment.py — Module 2: Partial OT Semantic Alignment.
"""

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from config import Config


def _cost_matrix(proto_t, proto_d, metric):
    if metric == "cosine":
        sim = normalize(proto_t, "l2") @ normalize(proto_d, "l2").T
        M   = (1.0 - sim).clip(0)
    else:
        diff = proto_t[:, None, :] - proto_d[None, :, :]
        M    = (diff ** 2).sum(-1)
    return (M / (M.max() + 1e-8)).astype(np.float64)


def _intra_cost(proto, metric):
    if metric == "cosine":
        sim = normalize(proto, "l2") @ normalize(proto, "l2").T
        M = (1.0 - sim).clip(0)
    else:
        diff = proto[:, None, :] - proto[None, :, :]
        M = (diff ** 2).sum(-1)
    return (M / (M.max() + 1e-8)).astype(np.float64)


def _relational_cost(proto_t, proto_d, metric):
    Ct = _intra_cost(proto_t, metric)
    Cd = _intra_cost(proto_d, metric)
    sig_t = np.sort(Ct, axis=1)
    sig_d = np.sort(Cd, axis=1)
    diff = sig_t[:, None, :] - sig_d[None, :, :]
    M = (diff ** 2).sum(-1)
    return (M / (M.max() + 1e-8)).astype(np.float64)


def _best_mass(proto_t, proto_d, M, grid):
    C, K = M.shape
    a = np.ones(C) / C
    b = np.ones(K) / K
    best_s, best_score = 1.0, -np.inf
    for s in grid:
        try:
            T     = ot.partial.partial_wasserstein(a, b, M, m=s)
            T_n   = T / (T.sum(axis=1, keepdims=True) + 1e-10)
            score = float(T_n.max(axis=1).mean())
            if score > best_score:
                best_score, best_s = score, s
        except Exception:
            continue
    return best_s


class PartialOTAligner:
    def __init__(self, cfg: Config):
        self.cfg              = cfg
        self.T_star           = None
        self.cluster_to_class = None
        self.mass             = None

    def fit(self, proto_t, proto_d) -> "PartialOTAligner":
        cfg_ot = self.cfg.ot
        C, K   = len(proto_t), len(proto_d)
        # Store t-side prototypes for ProtoFTL-style conditioning (use_proto_condition).
        self.proto_t = proto_t.astype(np.float32)
        M      = _cost_matrix(proto_t, proto_d, cfg_ot.cost_metric)
        a      = np.ones(C) / C
        b      = np.ones(K) / K

        # Always use the relational (structural) cost for OT so that T_soft
        # captures within-domain class geometry rather than cross-domain
        # prototype distances.  This is critical for two reasons:
        #   1. OC features are disjoint CNN descriptor halves — direct
        #      cross-half cosine distance is semantically meaningless.
        #   2. DP noise perturbs individual prototype coordinates, making
        #      direct cross-domain distances unreliable even for CWRU.
        # Relational cost only depends on WITHIN-domain pairwise distances,
        # which are far more robust to DP perturbations.
        M_ot = _relational_cost(proto_t, proto_d, cfg_ot.cost_metric)

        self.mass = cfg_ot.partial_mass or _best_mass(
            proto_t, proto_d, M_ot, cfg_ot.mass_search_grid)

        T = ot.partial.partial_wasserstein(a, b, M_ot, m=self.mass)
        # T_soft: raw Sinkhorn output used for soft OT-mixture conditioning.
        # T_star: may be overwritten by bijection in _cluster_mapping (for heatmap).
        self.T_soft = T.astype(np.float32)
        self.T_star = T.astype(np.float32)
        self.cluster_to_class = self._cluster_mapping(self.T_star, M, proto_t, proto_d)
        return self

    def _cluster_mapping(self, T, M, proto_t, proto_d):
        T_n = T / (T.sum(axis=1, keepdims=True) + 1e-10)
        C, K = T_n.shape
        # FIX: relational_cost_weight read from config (default 0.25, was hardcoded 0.50).
        # Reducing from 0.50 to 0.25 avoids over-penalising structural mismatch when
        # prototypes are noisy from DP perturbation, leading to better assignments.
        rel_w = getattr(self.cfg.ot, "relational_cost_weight", 0.25)
        if C == K:
            # C==K: one-to-one bipartite assignment prevents pseudo-label collapse.
            # T_n is computed from the relational cost matrix for ALL datasets
            # (see fit()).  Use T_n directly — subtracting R again would
            # double-penalise structural mismatch and degrade the bijection.
            score = T_n
            row_ind, col_ind = linear_sum_assignment(-score)
            mapping = np.zeros(C, dtype=np.int64)
            mapping[row_ind] = col_ind
            T_assign = np.zeros_like(T_n, dtype=np.float32)
            T_assign[row_ind, col_ind] = 1.0
            self.T_star = T_assign / max(C, 1)
            return mapping
        # C != K (e.g. CWRU: 10 clusters → 4 classes).
        # Use confidence-weighted argmax: rows with low total transport mass
        # are unreliable; fall back to nearest-prototype for those clusters.
        row_mass    = T.sum(axis=1)                     # total mass per cluster
        mass_thresh = np.percentile(row_mass, 20)       # bottom 20% = low confidence
        argmax_map  = T_n.argmax(axis=1)
        # For low-confidence clusters, use simple cosine nearest-prototype
        align_dim = min(proto_t.shape[1], proto_d.shape[1])
        t_n = normalize(proto_t[:, :align_dim], norm="l2")
        d_n = normalize(proto_d[:, :align_dim], norm="l2")
        cos_map = (t_n @ d_n.T).argmax(axis=1)
        mapping = np.where(row_mass >= mass_thresh, argmax_map, cos_map)
        return mapping.astype(np.int64)

    def compute_transport_conditions(self, proto_d_noisy, cluster_assignments,
                                     hard: bool = False, uniform: bool = False,
                                     sample_features=None):
        """
        Compute per-sample transport conditions.

        uniform=True (ablation "w/o Partial OT"):
            No OT alignment — every sample receives the plain average of all
            d-prototypes.  Tests whether OT-guided alignment contributes.

        hard=True (ablation "w/o Soft Cond."):
            Hard assignment — each sample is conditioned on the single
            d-prototype selected by hard argmax.
            When use_proto_condition=True: argmax of t-prototype similarity.
            When use_proto_condition=False: OT bijection (T_star / LAP).

        hard=False, uniform=False (default, Full FedPOT):
            When use_proto_condition=True (OC):
                ProtoFTL-style soft conditions — temperature-scaled softmax
                of t-prototype similarity weights d-prototypes reordered by
                the OT bijection.  Reliable for OC where cross-domain OT
                conditions are unreliable (disjoint CNN feature halves).
                Yields smooth, well-calibrated features → better AUC.
            When use_proto_condition=False (CWRU, default):
                Soft OT transport plan (T_soft) — each cluster's condition is
                a convex combination of d-prototypes weighted by the continuous
                Sinkhorn partial-transport plan.
        """
        if uniform:
            # No alignment: average all d-prototypes uniformly
            avg = proto_d_noisy.mean(axis=0, keepdims=True)
            return np.tile(avg, (len(cluster_assignments), 1)).astype(np.float32)

        use_proto = getattr(self.cfg.ot, "use_proto_condition", False)
        if use_proto and sample_features is not None and hasattr(self, "proto_t"):
            # ProtoFTL-style: t-side prototype similarity → weighted d-prototype mix.
            # Uses t-space similarities (reliable, same feature space) instead of
            # cross-domain OT distances (unreliable when features are disjoint).
            # Reorders d-prototypes by OT bijection so cluster k → d_class k.
            t_n   = normalize(sample_features, "l2")            # [N, t_dim]
            mu_t  = normalize(self.proto_t,    "l2")            # [C, t_dim]
            sim   = t_n @ mu_t.T                                # [N, C] raw cosine
            d_ord = proto_d_noisy[self.cluster_to_class]        # [C, d_dim]
            if hard:
                # Hard: nearest t-prototype → single d-prototype (one-hot)
                nn_idx = sim.argmax(axis=1)                     # [N]
                return d_ord[nn_idx].astype(np.float32)
            # Soft: temperature-scaled softmax — smooth convex combination
            temp  = float(getattr(self.cfg.ot, "proto_cond_temp", 5.0))
            s     = sim * temp
            s    -= s.max(axis=1, keepdims=True)                # numerical stability
            w     = np.exp(s)
            w    /= w.sum(axis=1, keepdims=True) + 1e-10        # [N, C]
            return (w @ d_ord).astype(np.float32)

        if hard:
            # Hard bijection (T_star / LAP): one d-prototype per cluster.
            # Equivalent to one-hot W @ proto_d, implemented as direct index.
            return proto_d_noisy[self.cluster_to_class[cluster_assignments]].astype(
                np.float32)

        # Default Full FedPOT: use T_soft (continuous Sinkhorn plan).
        # T_soft is set in fit() from the raw partial_wasserstein output and
        # is NEVER overwritten by the bijection step (unlike T_star).
        # Row-normalise to obtain a valid convex combination per cluster.
        T_for_cond = self.T_soft
        W = T_for_cond[cluster_assignments]
        W = W / (W.sum(axis=1, keepdims=True) + 1e-10)
        ot_cond = (W @ proto_d_noisy).astype(np.float32)
        nn_w = float(getattr(self.cfg.ot, "nn_condition_weight", 0.0))
        if nn_w > 0.0 and sample_features is not None:
            nn_cond = self._sample_nn_conditions(proto_d_noisy, sample_features)
            nn_w = min(max(nn_w, 0.0), 1.0)
            return ((1.0 - nn_w) * ot_cond + nn_w * nn_cond).astype(np.float32)
        return ot_cond

    def _sample_nn_conditions(self, proto_d_noisy, sample_features):
        align_dim = min(sample_features.shape[1], proto_d_noisy.shape[1])
        X_n = normalize(sample_features[:, :align_dim], norm="l2")
        D_n = normalize(proto_d_noisy[:, :align_dim], norm="l2")
        nn_idx = (X_n @ D_n.T).argmax(axis=1)
        return proto_d_noisy[nn_idx].astype(np.float32)

    def get_pseudo_labels(self, cluster_assignments):
        return self.cluster_to_class[cluster_assignments]

    def alignment_summary(self) -> dict:
        T_n     = self.T_star / (self.T_star.sum(1, keepdims=True) + 1e-10)
        entropy = -(T_n * np.log(T_n + 1e-10)).sum(1)
        return {
            "transport_mass":     self.mass,
            "cluster_to_class":   self.cluster_to_class.tolist(),
            "T_star_shape":       list(self.T_star.shape),
            "T_star_row_entropy": float(entropy.mean()),
        }
