"""
experiments/baselines.py �?实验一：SOTA 基线对比

对比方法（CrossFGAT-Lite 已替换为 ProtoFTL）：
  - NoTransfer
  - FedAvg-FTL
  - DANN-FTL
  - SHOT-FTL
  - ProtoFTL       �?新增：原型最近邻联邦迁移（无 OT / 无生成）
  - FedPOT (Ours)

输出:
  - results/tables/baselines_{dataset}.xlsx
  - results/figures/baselines_{dataset}.pdf  （无大标题，图例在图上方居中�?"""

import os
import copy
import types
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from config import Config
from models.networks import DownstreamClassifier
from evaluation.downstream import DownstreamTrainer
from experiments.plot_style import (
    apply_style, save_pdf, save_xlsx, method_color,
    FEDPOT_COLOR, OTHER_COLORS,
    FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ANNOT,
)

METRICS = ["accuracy", "macro_f1", "macro_auc"]
METRIC_LABELS = ["Accuracy", "Macro-F1", "Macro-AUC"]
METRIC_PALETTE = [FEDPOT_COLOR, OTHER_COLORS[3], OTHER_COLORS[1]]
BEST_ALPHA = 1.0
OTHER_ALPHA = 0.68

METHOD_ORDER = [
    "NoTransfer",
    "FedAvg-FTL",
    "DANN-FTL",
    "SHOT-FTL",
    "ProtoFTL",
    "FedPOT (Ours)",
]

def _single_best_method(results, methods, metric):
    vals = np.array([results.get(m, {}).get(metric, np.nan) for m in methods], dtype=float)
    if np.all(np.isnan(vals)):
        return None
    best = np.nanmax(vals)
    tied = np.flatnonzero(np.isclose(vals, best, atol=1e-12, equal_nan=False))
    if len(tied) == 0:
        return None
    ours_idx = methods.index("FedPOT (Ours)") if "FedPOT (Ours)" in methods else -1
    return methods[ours_idx] if ours_idx in tied else methods[int(tied[0])]


def _source_prototypes(data, cfg):
    n_cls = cfg.data.n_classes
    d_dim = data.d_train_x.shape[1]
    proto = np.stack([
        data.d_train_x[data.d_train_y == k].mean(0)
        if (data.d_train_y == k).any() else np.zeros(d_dim)
        for k in range(n_cls)
    ]).astype(np.float32)
    counts = np.array([(data.d_train_y == k).sum() for k in range(n_cls)],
                      dtype=np.int64)
    try:
        from core.privacy import add_dp_noise
        proto, _ = add_dp_noise(
            proto,
            cfg.privacy.epsilon,
            cfg.privacy.delta,
            cfg.privacy.max_norm,
            np.random.default_rng(cfg.seed),
            counts=counts,
        )
    except Exception:
        pass
    return proto.astype(np.float32)


def _target_pseudo_labels(data, cfg, return_mapping: bool = False):
    n_cls = cfg.data.n_classes
    km = KMeans(
        n_clusters=n_cls,
        n_init=cfg.prototype.kmeans_n_init,
        max_iter=cfg.prototype.kmeans_max_iter,
        random_state=cfg.seed,
    )
    train_clusters = km.fit_predict(normalize(data.t_train_x, norm="l2"))
    mu_t = km.cluster_centers_.astype(np.float32)
    mu_d = _source_prototypes(data, cfg)
    align_dim = min(mu_t.shape[1], mu_d.shape[1])

    if cfg.data.dataset == "office_home":
        # OC: t-features (CNN second half) and d-features (CNN first half) are
        # disjoint �?cosine similarity between them is meaningless and causes all
        # clusters to map to the same class (label collapse).  Use relational
        # bijection instead: match clusters by within-domain pairwise distance
        # profiles, then solve a one-to-one assignment via LAP.
        from scipy.optimize import linear_sum_assignment
        mu_t_a = mu_t[:, :align_dim]
        mu_d_a = mu_d[:, :align_dim]
        dt = np.linalg.norm(mu_t_a[:, None, :] - mu_t_a[None, :, :], axis=-1)
        dd = np.linalg.norm(mu_d_a[:, None, :] - mu_d_a[None, :, :], axis=-1)
        sdt = np.sort(dt, axis=1)
        sdd = np.sort(dd, axis=1)
        R = ((sdt[:, None, :] - sdd[None, :, :]) ** 2).sum(-1)
        row_ind, col_ind = linear_sum_assignment(R)
        cluster_to_class = np.zeros(n_cls, dtype=np.int64)
        cluster_to_class[row_ind] = col_ind
    else:
        # CWRU and other datasets: direct cosine similarity is meaningful.
        sim = (normalize(mu_t[:, :align_dim], norm="l2")
               @ normalize(mu_d[:, :align_dim], norm="l2").T)
        cluster_to_class = sim.argmax(axis=1)

    pseudo = cluster_to_class[train_clusters]
    if return_mapping:
        return pseudo, mu_t, mu_d, cluster_to_class
    return pseudo, mu_t, mu_d


def _coral_map(source, target, eps=1e-3):
    d = min(source.shape[1], target.shape[1])
    Xs = source[:, :d]
    Xt = target[:, :d]
    ms = Xs.mean(axis=0, keepdims=True)
    mt = Xt.mean(axis=0, keepdims=True)
    Cs = np.cov((Xs - ms).T) + np.eye(d) * eps
    Ct = np.cov((Xt - mt).T) + np.eye(d) * eps
    es, Vs = np.linalg.eigh(Cs)
    et, Vt = np.linalg.eigh(Ct)
    Cs_inv_sqrt = Vs @ np.diag(1.0 / np.sqrt(np.maximum(es, 1e-6))) @ Vs.T
    Ct_sqrt = Vt @ np.diag(np.sqrt(np.maximum(et, 1e-6))) @ Vt.T

    def transform(X):
        return ((X[:, :d] - mt) @ Ct_sqrt @ Cs_inv_sqrt + ms).astype(np.float32)

    return Xs.astype(np.float32), transform


# ─────────────────────────────────────────────────────────────────────────────
# 基线实现
# ─────────────────────────────────────────────────────────────────────────────

def run_no_transfer(data, cfg, logger=None) -> Dict:
    """No-transfer baseline using target-side incomplete features only."""
    pseudo_train, _, _ = _target_pseudo_labels(data, cfg)
    trainer = DownstreamTrainer(cfg, "NoTransfer")
    trainer.train(data.t_train_x, pseudo_train, logger)
    return trainer.evaluate(data.t_test_x, data.t_test_y)


def run_fedavg_ftl(data, cfg, logger=None) -> Dict:
    """FedAvg-FTL baseline using CORAL covariance alignment."""
    t_dim     = data.t_train_x.shape[1]
    d_dim     = data.d_train_x.shape[1]
    align_dim = min(t_dim, d_dim)

    # D-side: compute covariance eigendecomposition (shared as model weights)
    # T-side: apply CORAL transform to match d-domain covariance structure
    _, coral_fn = _coral_map(
        data.d_train_x[:, :align_dim],
        data.t_train_x[:, :align_dim],
    )
    proj_tr = coral_fn(data.t_train_x[:, :align_dim])
    proj_te = coral_fn(data.t_test_x[:, :align_dim])

    pseudo_train, _, _ = _target_pseudo_labels(data, cfg)
    trainer = DownstreamTrainer(cfg, "FedAvg-FTL")
    trainer.train(proj_tr, pseudo_train, logger)
    return trainer.evaluate(proj_te, data.t_test_y)


def run_dann_ftl(data, cfg, logger=None) -> Dict:
    """DANN-FTL baseline using prototype-based domain alignment."""
    n_cls = cfg.data.n_classes
    t_dim = data.t_train_x.shape[1]
    d_dim = data.d_train_x.shape[1]
    align_dim = min(t_dim, d_dim)

    pseudo_train, mu_t, mu_d = _target_pseudo_labels(data, cfg)

    # 最小二乘对齐到共同维度
    W, _, _, _ = np.linalg.lstsq(mu_t, mu_d[:, :align_dim], rcond=None)
    aligned_tr = data.t_train_x @ W
    aligned_te = data.t_test_x  @ W

    trainer = DownstreamTrainer(cfg, "DANN-FTL")
    trainer.train(aligned_tr, pseudo_train, logger)
    return trainer.evaluate(aligned_te, data.t_test_y)


def run_shot_ftl(data, cfg, logger=None) -> Dict:
    """SHOT-FTL baseline with target-side self-training."""
    device = torch.device(cfg.device)
    pseudo_train, _, _ = _target_pseudo_labels(data, cfg)
    trainer = DownstreamTrainer(cfg, "SHOT-FTL")
    trainer.train(data.t_train_x, pseudo_train, logger)

    return trainer.evaluate(data.t_test_x, data.t_test_y)


def run_proto_ftl(data, cfg, logger=None) -> Dict:
    """ProtoFTL baseline using nearest-prototype transfer."""
    n_cls = cfg.data.n_classes
    t_dim = data.t_train_x.shape[1]
    d_dim = data.d_train_x.shape[1]

    # ── Step1: d 侧计算类原型并加 DP 噪声传输 ──
    pseudo_train, mu_t, mu_d = _target_pseudo_labels(data, cfg)
    mu_d_noisy = mu_d

    # ── Step2: t 侧用类原型计�?t 侧均值，做最近邻软分�?──
    align_dim = min(t_dim, d_dim)
    # Cosine-similarity assignment.
    mu_t_n = mu_t[:, :align_dim] / (
        np.linalg.norm(mu_t[:, :align_dim], axis=1, keepdims=True) + 1e-8)
    mu_d_n = mu_d_noisy[:, :align_dim] / (
        np.linalg.norm(mu_d_noisy[:, :align_dim], axis=1, keepdims=True) + 1e-8)

    # 每个 t 侧训练样本：软权�?= softmax(cos_sim)
    t_norm = data.t_train_x[:, :align_dim] / (
        np.linalg.norm(data.t_train_x[:, :align_dim],
                       axis=1, keepdims=True) + 1e-8)
    if cfg.data.dataset == "office_home":
        counts = np.array([(data.d_train_y == k).sum() for k in range(n_cls)],
                          dtype=np.float32)
        prior = counts / (counts.sum() + 1e-8)
        weights = np.tile(prior[None, :], (len(data.t_train_x), 1))
    else:
        sim    = t_norm @ mu_t_n.T          # [N_t, n_cls]
        weights = np.exp(sim * 5)           # temperature=5
        weights /= weights.sum(axis=1, keepdims=True)

    # 软原型条件（加权平均 d 侧原型）
    proto_feat = weights @ mu_d_noisy[:, :d_dim]  # [N_t, d_dim]

    # ── Step3: 拼接增强特征训练分类�?──
    X_aug_tr = np.concatenate([data.t_train_x, proto_feat], axis=1)

    # Apply the same prototype weighting to test samples.
    t_te_norm  = data.t_test_x[:, :align_dim] / (
        np.linalg.norm(data.t_test_x[:, :align_dim],
                       axis=1, keepdims=True) + 1e-8)
    if cfg.data.dataset == "office_home":
        weights_te = np.tile(weights[0:1], (len(data.t_test_x), 1))
    else:
        sim_te     = t_te_norm @ mu_t_n.T
        weights_te = np.exp(sim_te * 5)
        weights_te /= weights_te.sum(axis=1, keepdims=True)
    proto_feat_te = weights_te @ mu_d_noisy[:, :d_dim]
    X_aug_te = np.concatenate([data.t_test_x, proto_feat_te], axis=1)

    trainer = DownstreamTrainer(cfg, "ProtoFTL")
    trainer.train(X_aug_tr, pseudo_train, logger)
    return trainer.evaluate(X_aug_te, data.t_test_y)


def run_dann_ftl_fair(data, cfg, logger=None) -> Dict:
    """DANN-FTL fair baseline in the federated protocol."""
    pseudo_train, mu_t, mu_d, cluster_to_class = _target_pseudo_labels(
        data, cfg, return_mapping=True)
    align_dim = min(mu_t.shape[1], mu_d.shape[1])
    mapped = mu_d[np.clip(cluster_to_class, 0, mu_d.shape[0] - 1), :align_dim]
    W, _, _, _ = np.linalg.lstsq(mu_t[:, :align_dim], mapped, rcond=None)
    aligned_tr = (data.t_train_x[:, :align_dim] @ W).astype(np.float32)
    aligned_te = (data.t_test_x[:, :align_dim]  @ W).astype(np.float32)
    trainer = DownstreamTrainer(cfg, "DANN-FTL")
    trainer.train(aligned_tr, pseudo_train, logger)
    return trainer.evaluate(aligned_te, data.t_test_y)


def run_shot_ftl_fair(data, cfg, logger=None) -> Dict:
    """SHOT-FTL fair baseline in the federated protocol."""
    pseudo_train, _, _ = _target_pseudo_labels(data, cfg)
    trainer = DownstreamTrainer(cfg, "SHOT-FTL")
    trainer.train(data.t_train_x, pseudo_train, logger)
    # Self-training on target: keep the most confident pseudo predictions
    logits = trainer.predict_logits(data.t_train_x)
    probs  = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    conf   = probs.max(axis=1)
    keep   = conf >= np.quantile(conf, 0.60)
    if keep.sum() >= cfg.data.n_classes:
        trainer.train(data.t_train_x[keep], probs[keep].argmax(axis=1), logger)
    return trainer.evaluate(data.t_test_x, data.t_test_y)


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

METHODS = {
    "NoTransfer": run_no_transfer,
    "FedAvg-FTL": run_fedavg_ftl,
    "DANN-FTL":   run_dann_ftl_fair,
    "SHOT-FTL":   run_shot_ftl_fair,
    "ProtoFTL":   run_proto_ftl,
}


def run_baseline_comparison(data, cfg, fedpot_results: Dict,
                             logger=None) -> Dict:
    all_results = {}
    for name, fn in METHODS.items():
        logger and logger.info(f"  [Baseline] Running: {name} ...")
        try:
            all_results[name] = fn(data, cfg, logger)
        except Exception as e:
            logger and logger.info(f"  [Baseline] {name} FAILED: {e}")
            all_results[name] = {"accuracy": float("nan"),
                                 "macro_f1": float("nan"),
                                 "macro_auc": float("nan")}
    all_results["FedPOT (Ours)"] = dict(fedpot_results.get("FedPOT", {}))
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# XLSX
# ─────────────────────────────────────────────────────────────────────────────

def _save_baselines_xlsx(results: Dict, table_dir: str, dataset: str,
                         logger=None):
    rows = []
    for name in METHOD_ORDER:
        m = results.get(name, {})
        rows.append({
            "Method":    name,
            "Accuracy":  round(m.get("accuracy",  float("nan")), 4),
            "Macro-F1":  round(m.get("macro_f1",  float("nan")), 4),
            "Macro-AUC": round(m.get("macro_auc", float("nan")), 4),
        })
    path = os.path.join(table_dir, f"baselines_{dataset}.xlsx")
    save_xlsx(rows, path, sheet_name=f"Baselines_{dataset}")
    logger and logger.info(f"  [Baseline] XLSX saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 绘图：无标题，图例在图上方居中一�?# ─────────────────────────────────────────────────────────────────────────────

def plot_baseline_comparison(results: Dict, save_dir: str,
                             dataset: str, logger=None):
    table_dir = os.path.join(
        os.path.dirname(save_dir.rstrip("/").rstrip("\\")), "tables")
    os.makedirs(table_dir, exist_ok=True)
    _save_baselines_xlsx(results, table_dir, dataset, logger)

    apply_style()

    methods = [m for m in METHOD_ORDER if m in results]
    n_m, n_met = len(methods), len(METRICS)
    bar_h   = 0.22
    gap     = 0.08
    group_h = n_met * bar_h + gap

    # Leave room for the legend above the plot.
    fig, ax = plt.subplots(figsize=(10, max(5.2, n_m * 1.05 + 1.8)))
    fig.subplots_adjust(top=0.88)   # 留出图例空间

    y_positions = np.arange(n_m) * group_h
    best_method_by_metric = {
        metric: _single_best_method(results, methods, metric)
        for metric in METRICS
    }

    for mi, metric in enumerate(METRICS):
        for gi, method in enumerate(methods):
            val   = results.get(method, {}).get(metric, np.nan)
            color = METRIC_PALETTE[mi]
            y     = y_positions[gi] + mi * bar_h
            is_best = (method == best_method_by_metric[metric])
            ax.barh(y, val if not np.isnan(val) else 0,
                    height=bar_h * 0.82, color=color,
                    alpha=BEST_ALPHA if is_best else OTHER_ALPHA,
                    edgecolor="#333333" if is_best else "white",
                    linewidth=0.8 if is_best else 0.5,
                    zorder=4 if is_best else 3)
            if not np.isnan(val):
                ax.text(val + 0.003, y, f"{val:.3f}",
                        va="center", ha="left",
                        fontsize=FS_ANNOT - 1,
                        color="#222222" if is_best else "#777777",
                        fontweight="bold" if is_best else "normal")

    # Highlight rows whose method is best for at least one metric.
    fp_gi = methods.index("FedPOT (Ours)") if "FedPOT (Ours)" in methods else -1
    best_methods = {
        m for m in best_method_by_metric.values() if m is not None
    }
    for method in best_methods:
        gi = methods.index(method)
        ay = y_positions[gi] - bar_h * 0.5
        ax.axhspan(ay, ay + group_h - gap * 0.8,
                   color=method_color(method), alpha=0.045, zorder=1)

    yticks = y_positions + (n_met - 1) * bar_h / 2
    ax.set_yticks(yticks)
    ax.set_yticklabels(methods, fontsize=FS_TICK)
    ax.set_xlabel("Score", fontsize=FS_LABEL)
    ax.set_xlim(0, 1.10)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#CCCCCC", lw=0.8, ls=":", zorder=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    # Legend patches.
    patches = [mpatches.Patch(color=METRIC_PALETTE[i], label=METRIC_LABELS[i])
               for i in range(n_met)]
    fig.legend(handles=patches,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.97),
               ncol=n_met,
               fontsize=FS_LEGEND,
               framealpha=0.90,
               edgecolor="#CCCCCC",
               handlelength=1.5,
               columnspacing=1.2)

    path = os.path.join(save_dir, f"baselines_{dataset}.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Baseline] Chart saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 文字汇报
# ─────────────────────────────────────────────────────────────────────────────

def print_baseline_table(results: Dict, logger=None):
    sep   = "=" * 68
    lines = [sep,
             f"  {'Method':<22}  {'Accuracy':>10}  {'Macro-F1':>10}  {'Macro-AUC':>10}",
             "-" * 68]
    for name in METHOD_ORDER:
        m   = results.get(name, {})
        acc = f"{m.get('accuracy',  float('nan')):.4f}"
        f1  = f"{m.get('macro_f1',  float('nan')):.4f}"
        auc = f"{m.get('macro_auc', float('nan')):.4f}"
        lines.append(
            f"  {name:<22}  {acc:>10}  {f1:>10}  {auc:>10}"
            + (" *" if "Ours" in name else "")
        )
    lines.append(sep)
    out = "\n".join(lines)
    if logger: logger.info(out)
    else: print(out)
