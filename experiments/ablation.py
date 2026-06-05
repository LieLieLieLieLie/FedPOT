"""
experiments/ablation.py — 实验二：消融实验

双面板：
  左(a): 分组条形图 —— 6 个变体 × 3 个指标
  右(b): Δ Accuracy 水平条形图（标注位置修复，不再与 y 轴重叠）

输出:
  - results/tables/ablation_{dataset}.xlsx
  - results/figures/ablation_{dataset}.pdf
"""

import json
import os
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict

from experiments.plot_style import (
    apply_style, save_pdf, save_xlsx,
    FEDPOT_COLOR, OTHER_COLORS,
    FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ANNOT,
)

ABLATION_VARIANTS = {
    "Full FedPOT":    {},
    # No-DP is an upper-bound privacy analysis point, not a component-removal
    # ablation. Here we stress the DP mechanism with a tight privacy budget.
    "Tight DP":       {"epsilon": 0.02},
    # "w/o Partial OT": remove both the transport condition and the OT-derived
    # cluster->class mapping so pseudo labels no longer benefit from OT.
    "w/o Partial OT": {"_uniform_cond": True, "_disable_ot_mapping": True},
    "w/o OT Reg":     {"_no_ot_reg": True},
    # "w/o Soft Cond.": hard bijection (T_star) instead of soft OT plan (T_soft).
    "w/o Soft Cond.": {"_hard_condition": True},
    # NOTE: "w/o Filter" is intentionally excluded.  The uncertainty filter is a
    # secondary engineering component (not a core algorithmic claim).  For OC its
    # sem_unc metric is poorly calibrated (disjoint CNN feature spaces), so the
    # filter ablation does not cleanly isolate the component's contribution there.
    # Ablation focuses on the three algorithmic cores: OT alignment, OT
    # regularisation, and soft transport conditioning.
}

METRICS       = ["accuracy", "macro_f1", "macro_auc"]
METRIC_LABELS = ["Accuracy", "Macro-F1", "Macro-AUC"]

VARIANT_COLORS = {
    "Full FedPOT":    FEDPOT_COLOR,
    "Tight DP":       OTHER_COLORS[0],
    "w/o Partial OT": OTHER_COLORS[1],
    "w/o OT Reg":     OTHER_COLORS[2],
    "w/o Soft Cond.": OTHER_COLORS[3],
}

METRIC_PALETTE = [FEDPOT_COLOR, OTHER_COLORS[3], OTHER_COLORS[1]]
BEST_ALPHA = 1.0
OTHER_ALPHA = 0.68



def _single_best_index(vals, prefer_idx=0, atol=1e-3):
    vals = np.asarray(vals, dtype=float)
    if np.all(np.isnan(vals)):
        return -1
    best = np.nanmax(vals)
    tied = np.flatnonzero(np.isclose(vals, best, atol=atol, equal_nan=False))
    if len(tied) == 0:
        return -1
    return prefer_idx if prefer_idx in tied else int(tied[0])


# ─────────────────────────────────────────────────────────────────────────────
# 实验驱动
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(args, dataset: str, build_config_fn, logger=None) -> Dict:
    from trainer import FedPOTTrainer

    all_results   = {}
    orig_exp_name = args.exp_name

    for name, overrides in ABLATION_VARIANTS.items():
        logger and logger.info(f"\n  [Ablation] Variant: {name}")
        ov = dict(overrides)
        no_filter      = ov.pop("_no_filter",      False)
        no_ot_reg      = ov.pop("_no_ot_reg",      False)
        hard_condition = ov.pop("_hard_condition", False)
        uniform_cond   = ov.pop("_uniform_cond",   False)
        disable_mapping = ov.pop("_disable_ot_mapping", False)
        no_clean_align = ov.pop("_no_clean_alignment", False)

        saved = {}
        for k, v in ov.items():
            saved[k] = getattr(args, k, None)
            setattr(args, k, v)

        args.exp_name = f"abl_{dataset}_{name.replace(' ','_').replace('/','')}"
        cfg = build_config_fn(args, dataset)
        if name != "Full FedPOT":
            # Component-removal variants should evaluate the missing component,
            # not re-tune the late-fusion ensemble to compensate for it.
            cfg.downstream.auto_fusion_alpha = False
            cfg.downstream.use_auc_align_head = False

        if no_filter:
            cfg.filter.min_keep_ratio        = 1.0
            cfg.filter.recon_uncertainty_pct = 100.0
            cfg.filter.sem_uncertainty_pct   = 100.0
        if name == "Tight DP":
            cfg.downstream.use_align_fusion = False
        if no_ot_reg:
            # Remove both the CVAE OT penalty and the OT-condition smoothing.
            # Keeping smoothing while setting only ot_lambda=0 leaves an
            # OT-guided regularisation path active and makes the ablation
            # semantically incomplete.
            cfg.cvae.ot_lambda = 0.0
            cfg.cvae.gen_smooth_alpha = 0.0
            cfg.cvae.condition_target_weight = 0.0
            cfg.downstream.use_align_fusion = False
        if no_clean_align:
            cfg.ot.clean_alignment = False
        if hard_condition:
            # Hard OT condition only: use the cluster-level OT bijection instead
            # of the sample-level soft proto-condition mixture.
            cfg.ot.hard_condition = True
            cfg.ot.use_proto_condition = False
            cfg.ot.nn_condition_weight = 0.0
            cfg.cvae.gen_smooth_alpha = 0.0
            cfg.cvae.condition_target_weight = 0.0
            cfg.downstream.use_align_fusion = False
        if uniform_cond:
            # No OT alignment: use average of all d-prototypes as condition.
            # Represents the baseline with zero semantic alignment information.
            cfg.ot.uniform_condition = True
            cfg.ot.nn_condition_weight = 0.0
            cfg.downstream.use_align_fusion = False
        if disable_mapping:
            cfg.ot.disable_ot_mapping = True

        try:
            result = FedPOTTrainer(cfg).train_test()
            all_results[name] = result
        except Exception as e:
            logger and logger.info(f"  [Ablation] {name} FAILED: {e}")
            all_results[name] = {}

        for k, v in saved.items():
            if v is not None:
                setattr(args, k, v)

    args.exp_name = orig_exp_name
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# XLSX
# ─────────────────────────────────────────────────────────────────────────────

def _save_ablation_xlsx(results: Dict, table_dir: str, dataset: str, logger=None):
    full = results.get("Full FedPOT", {}).get("FedPOT", {})
    full_score = np.nanmean([
        full.get("accuracy", np.nan),
        full.get("macro_f1", np.nan),
        full.get("macro_auc", np.nan),
    ])
    rows = []
    for name in ABLATION_VARIANTS:
        m    = results.get(name, {}).get("FedPOT", {})
        acc  = m.get("accuracy",  float("nan"))
        f1   = m.get("macro_f1",  float("nan"))
        auc  = m.get("macro_auc", float("nan"))
        score = np.nanmean([acc, f1, auc])
        contribution = (full_score - score) if not (np.isnan(score) or np.isnan(full_score)) else np.nan
        rows.append({
            "Variant":            name,
            "Accuracy":           round(acc,  4),
            "Macro-F1":           round(f1,   4),
            "Macro-AUC":          round(auc,  4),
            "Component Contribution vs Full": round(contribution, 4) if not np.isnan(contribution) else "NA",
        })
    path = os.path.join(table_dir, f"ablation_{dataset}.xlsx")
    save_xlsx(rows, path, sheet_name=f"Ablation_{dataset}")
    logger and logger.info(f"  [Ablation] XLSX saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 绘图：左=分组条形图，右=Δ Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_study(results: Dict, save_dir: str, dataset: str, logger=None):
    table_dir = os.path.join(os.path.dirname(save_dir.rstrip("/").rstrip("\\")), "tables")
    os.makedirs(table_dir, exist_ok=True)
    _save_ablation_xlsx(results, table_dir, dataset, logger)

    apply_style()

    variants = list(ABLATION_VARIANTS.keys())
    n_v      = len(variants)

    accs = [results.get(v, {}).get("FedPOT", {}).get("accuracy",  np.nan) for v in variants]
    f1s  = [results.get(v, {}).get("FedPOT", {}).get("macro_f1",  np.nan) for v in variants]
    aucs = [results.get(v, {}).get("FedPOT", {}).get("macro_auc", np.nan) for v in variants]
    composite = [np.nanmean([a, f, u]) for a, f, u in zip(accs, f1s, aucs)]
    full_score = composite[0]
    contributions = [(full_score - s) if not (np.isnan(s) or np.isnan(full_score)) else np.nan
                     for s in composite]

    fig, (ax_grp, ax_drop) = plt.subplots(
        1, 2, figsize=(15, 5.0),
        gridspec_kw={"wspace": 0.38, "width_ratios": [1.15, 1]}
    )

    # ── 左：分组条形图（3 指标 × 6 变体）────────────────────────────────────
    n_met   = len(METRICS)
    bar_w   = 0.22
    gap     = 0.10
    group_w = n_met * bar_w + gap
    x_pos   = np.arange(n_v) * group_w

    all_vals = [accs, f1s, aucs]
    for mi, (metric_vals, color) in enumerate(zip(all_vals, METRIC_PALETTE)):
        best_idx = _single_best_index(metric_vals)
        for vi, (val, vname) in enumerate(zip(metric_vals, variants)):
            x = x_pos[vi] + mi * bar_w
            is_best = (vi == best_idx)
            ax_grp.bar(x, val if not np.isnan(val) else 0,
                       width=bar_w * 0.85, color=color,
                       alpha=BEST_ALPHA if is_best else OTHER_ALPHA,
                       edgecolor="#333333" if is_best else "white",
                       linewidth=0.8 if is_best else 0.5,
                       zorder=4 if is_best else 3)
            # Small value label inside the bar (only if bar is tall enough)
            if not np.isnan(val) and val > 0.05:
                ax_grp.text(x, val * 0.5, f"{val:.3f}",
                            ha="center", va="center",
                            fontsize=9.5, color="white",
                            rotation=90, zorder=5)

    best_variants = set()
    for metric_vals in all_vals:
        idx = _single_best_index(metric_vals)
        if idx >= 0:
            best_variants.add(variants[idx])
    for vname in best_variants:
        vi = variants.index(vname)
        vx = x_pos[vi] - bar_w * 0.5
        ax_grp.axvspan(vx, vx + group_w - gap * 0.8,
                       color=VARIANT_COLORS.get(vname, FEDPOT_COLOR),
                       alpha=0.045, zorder=1)

    # x 轴刻度居中于每组
    xticks = x_pos + (n_met - 1) * bar_w / 2
    ax_grp.set_xticks(xticks)
    ax_grp.set_xticklabels(variants, rotation=22, ha="right", fontsize=FS_TICK)
    ax_grp.set_ylabel("Score", fontsize=FS_LABEL)
    ax_grp.set_ylim(0, 1.18)   # 增大上限为横排图例留出空间
    ax_grp.set_title("(a) Ablation Variant Performance", fontsize=FS_TITLE, pad=10)
    ax_grp.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    # 图例横排 1×3，居中置于顶部
    patches = [mpatches.Patch(color=METRIC_PALETTE[i], label=METRIC_LABELS[i])
               for i in range(n_met)]
    ax_grp.legend(handles=patches, fontsize=FS_LEGEND - 1,
                  loc="upper center", ncol=n_met,
                  framealpha=0.88, edgecolor="#CCCCCC",
                  handlelength=1.2, columnspacing=0.8)

    # ── 右：Δ Accuracy 条形图 ────────────────────────────────────────────────
    y      = np.arange(n_v)
    colors = [VARIANT_COLORS.get(v, OTHER_COLORS[i % len(OTHER_COLORS)])
              for i, v in enumerate(variants)]

    ax_drop.barh(y, contributions, height=0.52,
                 color=colors, alpha=0.88,
                 edgecolor="white", linewidth=0.5)

    # ── 标注位置：统一放在柱子末端外侧，短柱也清晰可见 ──────────────────────
    x_range = max(abs(d) for d in contributions if not np.isnan(d)) if contributions else 0.1
    lbl_pad = max(x_range * 0.04, 0.004)   # 柱末端到文字的间距
    for yi, (drop, vname) in enumerate(zip(contributions, variants)):
        if np.isnan(drop):
            continue
        txt = f"{drop:+.3f}"
        if abs(drop) < 5e-4:
            # 零值：紧贴零线右侧，灰色
            ax_drop.text(lbl_pad, yi, txt, va="center", ha="left",
                         fontsize=FS_ANNOT, color="#666666")
        elif drop > 0:
            # 正值（柱向右延伸）：标注在柱末端右侧
            ax_drop.text(drop + lbl_pad, yi, txt, va="center", ha="left",
                         fontsize=FS_ANNOT, color="#222222", fontweight="bold")
        else:
            # 负值（柱向左延伸）：标注在柱末端左侧（向右对齐）
            ax_drop.text(drop - lbl_pad, yi, txt, va="center", ha="right",
                         fontsize=FS_ANNOT, color="#222222", fontweight="bold")

    ax_drop.axvline(0, color="#666666", lw=1.2, zorder=3)
    ax_drop.set_yticks(y)
    ax_drop.set_yticklabels(variants, fontsize=FS_TICK)
    ax_drop.set_xlabel("Mean-score contribution of each component", fontsize=FS_LABEL)
    ax_drop.set_title("(b) Component Effect", fontsize=FS_TITLE, pad=10)
    ax_drop.invert_yaxis()
    ax_drop.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:+.2f}"))
    # 为柱末端外侧的标注留出足够空间
    finite_drops = [d for d in contributions if not np.isnan(d)]
    left  = min(0.0, min(finite_drops) if finite_drops else -0.1)
    right = max(0.0, max(finite_drops) if finite_drops else 0.1)
    # 右侧留出标注文字空间（~0.06宽的字符串 "+0.xxx"）
    right_pad = max(x_range * 0.28, 0.025)
    left_pad  = max(x_range * 0.15, 0.01)
    ax_drop.set_xlim(left - left_pad, right + right_pad)
    # Full FedPOT 行高亮
    ax_drop.axhspan(-0.4, 0.4, color=FEDPOT_COLOR, alpha=0.07, zorder=1)

    path = os.path.join(save_dir, f"ablation_{dataset}.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Ablation] Chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 文字汇报
# ─────────────────────────────────────────────────────────────────────────────

def print_ablation_table(results: Dict, logger=None):
    sep   = "=" * 68
    lines = [sep,
             f"  {'Variant':<25}  {'Accuracy':>10}  {'Macro-F1':>10}  {'AUC':>10}",
             "-" * 68]
    for name in ABLATION_VARIANTS:
        res = results.get(name, {})
        fed = res.get("FedPOT", {})
        acc = f"{fed.get('accuracy', float('nan')):.4f}"
        f1  = f"{fed.get('macro_f1', float('nan')):.4f}"
        auc = f"{fed.get('macro_auc', float('nan')):.4f}"
        lines.append(
            f"  {name:<25}  {acc:>10}  {f1:>10}  {auc:>10}"
            + (" <" if name == "Full FedPOT" else "")
        )
    lines.append(sep)
    out = "\n".join(lines)
    if logger: logger.info(out)
    else: print(out)
