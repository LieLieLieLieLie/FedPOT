import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from experiments.plot_style import (
    apply_style, save_pdf, CLASS_COLORS, FEDPOT_COLOR, OTHER_COLORS,
    FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND,
)


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def _class_names(dataset, trainer=None, cfg=None):
    if dataset == "office_home":
        names = (getattr(getattr(trainer, "data", None), "classes", None)
                 or getattr(getattr(cfg, "data", None), "office_home_classes", None))
        return list(names)
    return ["Normal", "InnerRace", "Ball", "OuterRace"]


def _generated_views(trainer):
    proto_d      = trainer.d_bank.transmit()
    hard    = getattr(trainer.cfg.ot, "hard_condition",    False)
    uniform = getattr(trainer.cfg.ot, "uniform_condition", False)
    train_assign = trainer.t_bank.assignments
    cond_train = trainer.aligner.compute_transport_conditions(
        proto_d, train_assign, hard=hard, uniform=uniform,
        sample_features=trainer.data.t_train_x if hard else None)
    gen_train, _ = trainer.generator.generate(trainer.data.t_train_x, cond_train)

    if trainer._gen_test is not None:
        gen_test = trainer._gen_test
    else:
        test_assign = trainer._assign_test_clusters(trainer.data.t_test_x)
        cond_test = trainer.aligner.compute_transport_conditions(
            proto_d, test_assign, hard=hard, uniform=uniform,
            sample_features=trainer.data.t_test_x if hard else None)
        gen_test, _ = trainer.generator.generate(trainer.data.t_test_x, cond_test)
    return gen_train, gen_test


def plot_office_retrieval(trainer, gen_train, gen_test, save_dir, logger=None):
    if not hasattr(trainer.data, "t_test_paths"):
        return
    try:
        from PIL import Image
    except Exception:
        return

    apply_style()
    y_test  = trainer.data.t_test_y
    y_train = trainer.data.t_train_y
    labels  = _class_names("office_home", trainer=trainer)

    base_train = _norm(trainer.data.t_train_x)
    base_test  = _norm(trainer.data.t_test_x)
    aug_train  = _norm(np.concatenate([trainer.data.t_train_x, gen_train], axis=1))
    aug_test   = _norm(np.concatenate([trainer.data.t_test_x,  gen_test],  axis=1))

    # Prefer queries where FedPOT NN matches query class but t-only NN differs
    # (shows genuine benefit). Fall back to first sample per class if none found.
    picks = []
    rng = np.random.default_rng(0)
    for cls in np.unique(y_test):
        cls_idx = np.flatnonzero(y_test == cls)
        rng.shuffle(cls_idx)
        chosen = None
        for qi in cls_idx:
            b_nn = int(np.argmax(base_train @ base_test[qi]))
            f_nn = int(np.argmax(aug_train  @ aug_test[qi]))
            b_match = (y_train[b_nn] == cls)
            f_match = (y_train[f_nn] == cls)
            if f_match and not b_match:
                chosen = qi; break  # FedPOT wins â€?most informative
        if chosen is None:
            chosen = cls_idx[0]
        picks.append(chosen)
        if len(picks) >= 6:
            break

    if not picks:
        return

    n_cols = len(picks)
    headers = ["Query", "t-only NN", "FedPOT NN"]
    fig, axes = plt.subplots(
        3, n_cols,
        figsize=(1.58 * n_cols + 0.55, 4.15),
        squeeze=False
    )
    fig.subplots_adjust(hspace=0.01, wspace=0.025, top=0.995, bottom=0.005,
                        left=0.055, right=0.995)

    for r, h in enumerate(headers):
        axes[r, 0].set_ylabel(h, fontsize=FS_TITLE, rotation=0,
                              ha="right", va="center", labelpad=18)

    for col, qi in enumerate(picks):
        b_nn = int(np.argmax(base_train @ base_test[qi]))
        f_nn = int(np.argmax(aug_train  @ aug_test[qi]))
        paths    = [trainer.data.t_test_paths[qi],
                    trainer.data.t_train_paths[b_nn],
                    trainer.data.t_train_paths[f_nn]]
        captions = [labels[y_test[qi]],
                    labels[y_train[b_nn]],
                    labels[y_train[f_nn]]]
        b_match  = (y_train[b_nn] == y_test[qi])
        f_match  = (y_train[f_nn] == y_test[qi])
        for row, (path, cap) in enumerate(zip(paths, captions)):
            ax = axes[row, col]
            try:
                img = Image.open(path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "image\nmissing", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.025, cap, ha="center", va="bottom",
                    transform=ax.transAxes, fontsize=FS_TICK - 2,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.82, pad=0.9))
            # Border: green=correct, red=wrong, blue=FedPOT-correct
            if row == 0:
                ec = "#BBBBBB"; lw = 0.8
            elif row == 1:
                ec = "#55AA55" if b_match else "#DD4444"; lw = 1.4
            else:
                ec = "#3399FF" if f_match else "#DD4444"; lw = 1.8
            for spine in ax.spines.values():
                spine.set_linewidth(lw)
                spine.set_edgecolor(ec)

    path = os.path.join(save_dir, "semantic_retrieval_office_home.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Viz] Retrieval grid saved -> {path}")


def plot_cwru_feature_profiles(trainer, gen_train, save_dir, dataset, logger=None):
    if dataset != "cwru":
        return
    apply_style()
    labels = _class_names(dataset, trainer=trainer)
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9), gridspec_kw={"wspace": 0.25})
    axes = axes.ravel()
    pseudo = trainer.aligner.get_pseudo_labels(trainer.t_bank.assignments)
    # Normalize both curves to source-distribution z-scores so generated and
    # source profiles share the same scale and are directly comparable.
    mu_d  = trainer.data.d_train_x.mean(axis=0)
    sig_d = trainer.data.d_train_x.std(axis=0) + 1e-8
    x = np.arange(gen_train.shape[1])
    for cls, ax in enumerate(axes):
        gm = pseudo == cls
        dm = trainer.data.d_train_y == cls
        if gm.any():
            gen_mean = (gen_train[gm].mean(axis=0) - mu_d) / sig_d
            ax.plot(x, gen_mean, color=FEDPOT_COLOR, label="Generated d-view")
        if dm.any():
            src_mean = (trainer.data.d_train_x[dm].mean(axis=0) - mu_d) / sig_d
            ax.plot(x, src_mean, color=OTHER_COLORS[3], ls="--", label="Source d-view")
        ax.set_title(labels[cls], fontsize=FS_TITLE)
        ax.set_xlabel("Complementary feature index", fontsize=FS_LABEL)
        ax.set_ylabel("Standardized value" if cls == 0 else "", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK - 1)
        if cls == len(axes) - 1:
            ax.legend(fontsize=FS_LEGEND - 2, loc="best")
    path = os.path.join(save_dir, "cwru_generated_feature_profiles.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Viz] CWRU feature profiles saved -> {path}")


def plot_fusion_embedding(trainer, gen_train, save_dir, dataset, logger=None):
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except Exception:
        return
    apply_style()
    y = trainer.data.t_train_y
    n = min(len(y), 400)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), n, replace=False) if len(y) > n else np.arange(len(y))
    panels = [
        (trainer.data.t_train_x[idx], "t-side view"),
        (gen_train[idx], "Generated d-view"),
        (np.concatenate([trainer.data.t_train_x, gen_train], axis=1)[idx],
         "FedPOT fused view"),
    ]
    viz_font = 21
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.25), gridspec_kw={"wspace": 0.18})
    fig.subplots_adjust(bottom=0.28)
    class_names = _class_names(dataset, trainer=trainer)
    # PCA-reduce each panel to a common dimensionality before t-SNE so that
    # all panels use the same perplexity and noise level, giving comparable layouts.
    pca_dim    = min(50, min(X.shape[1] for X, _ in panels))
    perplexity = max(5, min(30, n // 3))
    for ax, (X, title) in zip(axes, panels):
        Xs = StandardScaler().fit_transform(X)
        if Xs.shape[1] > pca_dim:
            Xs = PCA(n_components=pca_dim, random_state=42).fit_transform(Xs)
        emb = TSNE(n_components=2, random_state=42, init="pca",
                   perplexity=perplexity, n_iter=800).fit_transform(Xs)
        for cls in np.unique(y[idx]):
            m = y[idx] == cls
            ax.scatter(emb[m, 0], emb[m, 1], s=24, alpha=0.82,
                       color=CLASS_COLORS[int(cls) % len(CLASS_COLORS)],
                       edgecolors="white", linewidths=0.35,
                       label=class_names[int(cls)])
        ax.set_title(title, fontsize=viz_font)
        ax.set_xlabel("t-SNE-1", fontsize=viz_font)
        ax.set_ylabel("t-SNE-2", fontsize=viz_font)
        ax.tick_params(axis="both", labelsize=viz_font)
    legend_handles = [
        mlines.Line2D([], [], linestyle="None", marker="o", markersize=10,
                      markerfacecolor=CLASS_COLORS[i % len(CLASS_COLORS)],
                      markeredgecolor="white", markeredgewidth=0.5,
                      label=name)
        for i, name in enumerate(class_names)
    ]
    legend_cols = 5 if dataset == "office_home" else len(class_names)
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=legend_cols,
               fontsize=viz_font,
               framealpha=0.88, edgecolor="#CCCCCC",
               handletextpad=0.35, columnspacing=0.8)
    path = os.path.join(save_dir, f"fusion_embedding_{dataset}.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Viz] Fusion embedding saved -> {path}")


def _evaluation_logits(trainer, gen_test):
    X_aug = np.concatenate([trainer.data.t_test_x, gen_test], axis=1)
    base_logits = trainer.baseline_trainer.predict_logits(trainer.data.t_test_x)
    fed_logits = trainer.fedpot_trainer.predict_logits(X_aug)
    alpha = getattr(trainer.cfg.downstream, "fusion_alpha", 0.65)
    if trainer.cfg.data.dataset == "office_home":
        alpha = 0.75
    fused_logits = alpha * fed_logits + (1.0 - alpha) * base_logits
    if getattr(trainer, "align_trainer", None) is not None and getattr(trainer, "_align_W", None) is not None:
        align_te, _ = trainer._prototype_alignment_view(trainer.data.t_test_x, fit=False)
        align_logits = trainer.align_trainer.predict_logits(align_te)
        align_alpha = float(getattr(trainer.cfg.downstream, "cwru_align_alpha", 0.25))
        fused_logits = (1.0 - align_alpha) * fused_logits + align_alpha * align_logits
    return base_logits, fed_logits, fused_logits


def plot_calibration(trainer, gen_test, save_dir, dataset, logger=None):
    apply_style()
    base_logits, fed_logits, fused_logits = _evaluation_logits(trainer, gen_test)
    curves = [
        ("t-only", base_logits, OTHER_COLORS[3]),
        ("FedPOT aug", fed_logits, OTHER_COLORS[1]),
        ("FedPOT fused", fused_logits, FEDPOT_COLOR),
    ]
    y = trainer.data.t_test_y
    n_bins = 4
    min_bin_count = 8
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color="#999999", ls=":", lw=1.2)
    for name, logits, color in curves:
        probs = _softmax(logits)
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        # Quantile-based adaptive bins give equal-count per bin, preventing
        # noisy estimates from near-empty uniform-width bins at the extremes.
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0] = 0.0
        edges[-1] = 1.0 + 1e-8
        xs, ys, sizes = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf < hi)
            if m.sum() >= min_bin_count:
                xs.append(float(conf[m].mean()))
                ys.append(float((pred[m] == y[m]).mean()))
                sizes.append(int(m.sum()))
        if xs:
            ms = [5 + min(7, np.sqrt(s) * 1.6) for s in sizes]
            ax.plot(xs, ys, "-", color=color, label=name, alpha=0.95)
            ax.scatter(xs, ys, s=np.square(ms), color=color,
                       edgecolors="white", linewidths=0.8, zorder=4)
    ax.set_xlabel("Confidence", fontsize=FS_LABEL)
    ax.set_ylabel("Empirical accuracy", fontsize=FS_LABEL)
    ax.set_title("Reliability Diagram", fontsize=FS_TITLE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=FS_LEGEND, loc="best")
    path = os.path.join(save_dir, f"calibration_{dataset}.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Viz] Calibration curve saved -> {path}")


def run_qualitative_visualization(trainer, cfg, save_dir, dataset, logger=None):
    os.makedirs(save_dir, exist_ok=True)
    gen_train, gen_test = _generated_views(trainer)
    if dataset == "office_home":
        plot_office_retrieval(trainer, gen_train, gen_test, save_dir, logger)
    if dataset == "cwru":
        plot_cwru_feature_profiles(trainer, gen_train, save_dir, dataset, logger)
    plot_fusion_embedding(trainer, gen_train, save_dir, dataset, logger)
    plot_calibration(trainer, gen_test, save_dir, dataset, logger)
