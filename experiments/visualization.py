"""
experiments/visualization.py — 实验五：可视化分析

修复：
  - t-SNE 三子图间距收紧（wspace 减小）
  - OT 热力图删除 transport mass 标注（移至论文 caption）

输出:
  - results/figures/tsne_{dataset}.pdf
  - results/figures/ot_heatmap_{dataset}.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde

from experiments.plot_style import (
    apply_style, save_pdf, make_seq_cmap,
    CLASS_COLORS, FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ANNOT,
)


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE 辅助
# ─────────────────────────────────────────────────────────────────────────────

def _subsample(X, y, n=300, seed=42):
    if len(X) > n:
        idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
        return X[idx], y[idx]
    return X, y


def _kde_contour(ax, emb_2d, labels, n_cls,
                 alpha_fill=0.12, alpha_line=0.55):
    for k in range(n_cls):
        mask = labels == k
        pts  = emb_2d[mask]
        if pts.shape[0] < 10:
            continue
        color = CLASS_COLORS[k % len(CLASS_COLORS)]
        try:
            kde  = gaussian_kde(pts.T, bw_method=0.35)
            xmin, xmax = emb_2d[:, 0].min() - 2, emb_2d[:, 0].max() + 2
            ymin, ymax = emb_2d[:, 1].min() - 2, emb_2d[:, 1].max() + 2
            xx, yy = np.mgrid[xmin:xmax:60j, ymin:ymax:60j]
            zz     = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            thresh = zz.max() * 0.15
            ax.contourf(xx, yy, zz, levels=[thresh, zz.max()],
                        colors=[color], alpha=alpha_fill)
            ax.contour(xx, yy, zz, levels=[thresh],
                       colors=[color], alpha=alpha_line, linewidths=1.2)
        except Exception:
            pass


def _scatter_panel(ax, emb_2d, labels, class_names, title,
                   marker="o", s=28, alpha=0.80):
    n_cls = len(class_names)
    _kde_contour(ax, emb_2d, labels, n_cls)
    for k in range(n_cls):
        mask  = labels == k
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS[k % len(CLASS_COLORS)]
        lbl   = class_names[k] if len(class_names) <= 6 else f"C{k}"
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=color, marker=marker, alpha=alpha, s=s,
                   edgecolors="white", linewidths=0.4,
                   label=lbl, zorder=3)

    ax.set_title(title, fontsize=FS_TITLE, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE Dim 1", fontsize=FS_LABEL)
    ax.set_ylabel("t-SNE Dim 2", fontsize=FS_LABEL)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#AAAAAA")


# ─────────────────────────────────────────────────────────────────────────────
# 公开 API
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne(t_features, d_features, gen_features,
              t_labels, d_labels, class_names,
              save_dir, dataset, logger=None):
    """三面板 t-SNE，含 KDE 等高线，子图间距收紧。"""
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger and logger.info("  [Viz] sklearn not available; skipping t-SNE")
        return

    apply_style()
    logger and logger.info("  [Viz] Running t-SNE ...")

    min_dim = min(t_features.shape[1], d_features.shape[1],
                  gen_features.shape[1])
    t_f, t_lb = _subsample(t_features[:, :min_dim],  t_labels, n=300, seed=42)
    d_f, d_lb = _subsample(d_features[:, :min_dim],  d_labels, n=300, seed=43)
    g_f, g_lb = _subsample(gen_features[:, :min_dim],
                            t_labels[:len(gen_features)], n=300, seed=44)

    all_feat = np.concatenate([t_f, d_f, g_f], axis=0)
    all_feat = StandardScaler().fit_transform(all_feat)
    n_t, n_d = len(t_f), len(d_f)

    emb   = TSNE(n_components=2, random_state=42, perplexity=30,
                 n_iter=1000, init="pca").fit_transform(all_feat)
    t_emb = emb[:n_t]
    d_emb = emb[n_t:n_t + n_d]
    g_emb = emb[n_t + n_d:]

    # wspace=0.12 使三张子图更紧凑
    viz_font = 21
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.45),
        gridspec_kw={"wspace": 0.12}
    )
    fig.subplots_adjust(bottom=0.28)

    _scatter_panel(axes[0], t_emb, t_lb, class_names,
                   "(a) t-side Original Features", marker="o")
    _scatter_panel(axes[1], d_emb, d_lb, class_names,
                   "(b) d-side Original Features", marker="o")
    _scatter_panel(axes[2], g_emb, g_lb, class_names,
                   "(c) FedPOT Generated Features",
                   marker="^", alpha=0.75)

    # 生成特征来源注释
    for idx_ax, ax in enumerate(axes):
        ax.set_xlabel("t-SNE-1", fontsize=viz_font)
        ax.set_ylabel("t-SNE-2" if idx_ax == 0 else "", fontsize=viz_font)
        ax.set_title(ax.get_title(), fontsize=viz_font)
        ax.tick_params(axis="both", labelsize=viz_font)

    legend_handles = [
        mlines.Line2D([], [], linestyle="None", marker="o", markersize=10,
                      markerfacecolor=CLASS_COLORS[i % len(CLASS_COLORS)],
                      markeredgecolor="white", markeredgewidth=0.5,
                      label=name)
        for i, name in enumerate(class_names)
    ]
    legend_cols = 5 if dataset == "office_caltech" else len(class_names)
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=legend_cols,
               fontsize=viz_font,
               framealpha=0.88, edgecolor="#CCCCCC",
               handletextpad=0.35, columnspacing=0.8)

    path = os.path.join(save_dir, f"tsne_{dataset}.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Viz] t-SNE saved → {path}")


def plot_ot_heatmap(T_star, cluster_names, class_names,
                    save_dir, dataset, transport_mass, logger=None):
    """
    OT 传输计划热力图（两面板）。
    transport_mass 参数保留接口但不再显示在图上（移至论文 caption）。
    """
    apply_style()

    C, K   = T_star.shape
    T_norm = T_star / (T_star.sum(axis=1, keepdims=True) + 1e-10)
    cmap   = make_seq_cmap()

    heat_font = 21
    value_font = 15 if max(C, K) <= 6 else 13
    h = 6.5
    fig, axes = plt.subplots(1, 2, figsize=(14, h),
                             gridspec_kw={"wspace": 0.38})

    value_fmt = "{:.1f}" if dataset == "office_caltech" else "{:.2f}"
    panels = [
        (T_star, "(a) Raw Transport Plan $T^*$"),
        (T_norm, "(b) Alignment Probability (Row-Normalized $\\bar{T}^*$)"),
    ]

    for panel_idx, (ax, (data, title)) in enumerate(zip(axes, panels)):
        vmax = data.max() if data.max() > 0 else 1.0
        im   = ax.imshow(data, cmap=cmap, aspect="auto",
                         vmin=0.0, vmax=vmax, interpolation="nearest")

        if panel_idx == 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.03)
            cbar.ax.tick_params(labelsize=heat_font)
            cbar.outline.set_linewidth(0.6)

        for i in range(C):
            for j in range(K):
                val       = data[i, j]
                rel       = val / (vmax + 1e-10)
                txt_color = "white" if rel > 0.55 else "#333333"
                ax.text(j, i, value_fmt.format(val),
                        ha="center", va="center",
                        fontsize=value_font,
                        color=txt_color)

        ax.set_xticks(range(K))
        ax.set_xticklabels(class_names, rotation=35, ha="right",
                           fontsize=heat_font)
        ax.set_yticks(range(C))
        ax.set_yticklabels(cluster_names, fontsize=heat_font)
        ax.set_xlabel("d-side Classes", fontsize=heat_font)
        ax.set_ylabel("t-side Clusters" if panel_idx == 0 else "",
                      fontsize=heat_font)
        ax.set_title(title, fontsize=heat_font, pad=14)
        ax.tick_params(length=0)

        # 红框标记每行最大值
        for i in range(C):
            best_j = np.argmax(data[i])
            ax.add_patch(
                plt.Rectangle((best_j - 0.48, i - 0.48), 0.96, 0.96,
                               fill=False, edgecolor="#FF6666",
                               linewidth=1.8, zorder=5)
            )

    # ── transport mass 标注已移除（请写入论文 caption）──

    path = os.path.join(save_dir, f"ot_heatmap_{dataset}.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Viz] OT heatmap saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

def run_visualization(trainer, cfg, save_dir, dataset, logger=None):
    os.makedirs(save_dir, exist_ok=True)

    class_names = (
        ["BackPack", "Bike", "Calc", "Headphone", "Keyboard",
         "Laptop", "Monitor", "Mouse", "Mug", "Projector"]
        if dataset == "office_caltech"
        else ["Normal", "InnerRace", "Ball", "OuterRace"]
    )
    cluster_names = [f"Cluster-{c}" for c in range(cfg.prototype.n_clusters)]

    try:
        proto_d    = trainer.d_bank.transmit()
        t_assign   = trainer.t_bank.assignments
        hard    = getattr(trainer.cfg.ot, "hard_condition",    False)
        uniform = getattr(trainer.cfg.ot, "uniform_condition", False)
        cond_train = trainer.aligner.compute_transport_conditions(
            proto_d, t_assign, hard=hard, uniform=uniform,
            sample_features=trainer.data.t_train_x)
        gen_train, _ = trainer.generator.generate(
            trainer.data.t_train_x, cond_train)
        plot_tsne(
            t_features=trainer.data.t_train_x,
            d_features=trainer.data.d_train_x,
            gen_features=gen_train,
            t_labels=trainer.data.t_train_y,
            d_labels=trainer.data.d_train_y,
            class_names=class_names,
            save_dir=save_dir,
            dataset=dataset,
            logger=logger,
        )
    except Exception as e:
        logger and logger.info(f"  [Viz] t-SNE failed: {e}")

    try:
        plot_ot_heatmap(
            T_star=trainer.aligner.T_star,
            cluster_names=cluster_names,
            class_names=class_names[:trainer.aligner.T_star.shape[1]],
            save_dir=save_dir,
            dataset=dataset,
            transport_mass=trainer.aligner.mass,
            logger=logger,
        )
    except Exception as e:
        logger and logger.info(f"  [Viz] OT heatmap failed: {e}")
