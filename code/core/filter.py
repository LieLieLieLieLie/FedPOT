"""
core/filter.py — Module 4: Uncertainty-Aware Augmentation Filtering.
"""

import numpy as np
from sklearn.preprocessing import normalize
from config import Config


def compute_semantic_uncertainty(generated, conditions):
    gn = normalize(generated,  norm="l2")
    cn = normalize(conditions, norm="l2")
    return (1.0 - (gn * cn).sum(axis=1)).astype(np.float32)


class UncertaintyFilter:
    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self.tau_recon = None
        self.tau_sem   = None

    def fit(self, recon_var, sem_unc) -> "UncertaintyFilter":
        self.tau_recon = float(np.percentile(recon_var, self.cfg.filter.recon_uncertainty_pct))
        self.tau_sem   = float(np.percentile(sem_unc,   self.cfg.filter.sem_uncertainty_pct))
        return self

    def apply(self, generated, conditions, recon_var, pseudo_labels):
        sem_unc  = compute_semantic_uncertainty(generated, conditions)
        mask     = (recon_var <= self.tau_recon) & (sem_unc <= self.tau_sem)
        n_total  = len(generated)
        min_keep = max(1, int(self.cfg.filter.min_keep_ratio * n_total))

        if mask.sum() < min_keep:
            combined  = _rank_asc(recon_var) + _rank_asc(sem_unc)
            top       = np.argsort(combined)[:min_keep]
            mask      = np.zeros(n_total, dtype=bool)
            mask[top] = True

        stats = {
            "n_total":    n_total,
            "n_kept":     int(mask.sum()),
            "keep_ratio": float(mask.sum() / n_total),
            "tau_recon":  self.tau_recon,
            "tau_sem":    self.tau_sem,
            "mean_recon_uncertainty": float(recon_var[mask].mean()),
            "mean_sem_uncertainty":   float(sem_unc[mask].mean()),
        }
        kept_idx = np.flatnonzero(mask).astype(np.int64)
        return generated[mask], conditions[mask], pseudo_labels[mask], kept_idx, stats


def _rank_asc(arr):
    return arr.argsort().argsort().astype(np.float32)
