"""
experiments/sweep_viz.py — Sweep 结果可视化（新增模块）

将 --sweep 跑出的全域名对结果可视化为精致热力图，
适合直接放入 KBS/ESWA 论文。

输出:
  - results/figures/sweep_{dataset}.pdf    准确率热力图（域名对矩阵）
  - results/tables/sweep_{dataset}.xlsx    完整数值表
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional

from experiments.plot_style import (
    apply_style, save_pdf, save_xlsx, make_seq_cmap, make_div_cmap,
    FEDPOT_COLOR, OTHER_COLORS,
    FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ANNOT,
)

OC_DOMAINS  = ["Amazon", "Caltech", "DSLR", "Webcam"]
CWRU_LOADS  = ["Load0", "Load1", "Load2", "Load3"]


# ─────────────────────────────────────────────────────────────────────────────
# XLSX
# ─────────────────────────────────────────────────────────────────────────────

def _save_sweep_xlsx(sweep_results: Dict, table_dir: str, dataset: str,
                     logger=None):
    rows = []
    for pair_tag, res in sweep_results.items():
        fp = res.get("FedPOT", {})
        bl = res.get("Baseline", {})
        rows.append({
            "Pair":          pair_tag,
            "FedPOT Acc":   round(fp.get("accuracy",  float("nan")), 4),
            "FedPOT F1":    round(fp.get("macro_f1",  float("nan")), 4),
            "FedPOT AUC":   round(fp.get("macro_auc", float("nan")), 4),
            "Baseline Acc": round(bl.get("accuracy",  float("nan")), 4),
            "Δ Accuracy":   round(
                fp.get("accuracy", float("nan")) - bl.get("accuracy", float("nan")), 4
            ),
        })
    path = os.path.join(table_dir, f"sweep_{dataset}.xlsx")
    save_xlsx(rows, path, sheet_name=f"Sweep_{dataset}")
    logger and logger.info(f"  [Sweep] XLSX saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweep_heatmap(sweep_results: Dict, save_dir: str,
                       dataset: str, logger=None):
    """
    将全域名 sweep 结果画成两张并排热力图：
      左: FedPOT Accuracy（纯正值热力图, 白→#007FFF）
      右: Δ Accuracy = FedPOT - Baseline（含正负，#FF4F4F→白→#007FFF）
    对角线（自传）用灰色填充。
    """
    apply_style()

    domains = OC_DOMAINS if dataset == "office_caltech" else CWRU_LOADS
    n       = len(domains)

    acc_mat   = np.full((n, n), np.nan)
    delta_mat = np.full((n, n), np.nan)

    for pair_tag, res in sweep_results.items():
        # pair_tag 格式: "amazon->dslr" 或 "0->2"
        parts = pair_tag.replace("→", "->").split("->")
        if len(parts) != 2:
            continue
        src_raw, tgt_raw = parts[0].strip(), parts[1].strip()

        if dataset == "office_caltech":
            src_map = {d.lower(): i for i, d in enumerate(OC_DOMAINS)}
            src_i   = src_map.get(src_raw.lower(), -1)
            tgt_i   = src_map.get(tgt_raw.lower(), -1)
        else:
            load_map = {str(i): i for i in range(n)}
            load_map.update({f"load{i}": i for i in range(n)})
            src_i   = load_map.get(src_raw.lower(), -1)
            tgt_i   = load_map.get(tgt_raw.lower(), -1)

        if src_i < 0 or tgt_i < 0 or src_i == tgt_i:
            continue

        fp  = res.get("FedPOT",   {}).get("accuracy", np.nan)
        bl  = res.get("Baseline", {}).get("accuracy", np.nan)
        acc_mat[src_i, tgt_i]   = fp
        delta_mat[src_i, tgt_i] = (fp - bl) if (not np.isnan(fp) and not np.isnan(bl)) else np.nan

    # 若矩阵全 NaN（数据未到位），生成占位图
    _has_data = not np.all(np.isnan(acc_mat))

    cmap_seq = make_seq_cmap()
    cmap_div = make_div_cmap()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={"wspace": 0.38})

    panels = [
        (acc_mat,   cmap_seq, False, "FedPOT Accuracy", ".3f"),
        (delta_mat, cmap_div, True,  "Δ Accuracy  (FedPOT − Baseline)", "+.3f"),
    ]

    for ax_i, (ax, (mat, cmap, is_div, title, fmt)) in \
            enumerate(zip(axes, panels)):

        vabs = np.nanmax(np.abs(mat)) if _has_data else 1.0
        if is_div:
            vmin, vmax = -vabs, vabs
        else:
            vmin = max(0.0, np.nanmin(mat) - 0.02) if _has_data else 0.0
            vmax = min(1.0, np.nanmax(mat) + 0.01) if _has_data else 1.0

        # 背景（对角线 / NaN 用浅灰）
        display_mat = mat.copy()
        for i in range(n):
            display_mat[i, i] = np.nan   # 对角线设 NaN

        masked = np.ma.masked_invalid(display_mat)
        im     = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="equal", interpolation="nearest")

        # 对角线填充灰色
        for i in range(n):
            ax.add_patch(plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                color="#DDDDDD", zorder=2
            ))
            ax.text(i, i, "—", ha="center", va="center",
                    fontsize=FS_ANNOT - 1, color="#999999", zorder=3)

        # 数值标注
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = mat[i, j]
                if np.isnan(val):
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=FS_ANNOT - 2, color="#AAAAAA", zorder=4)
                    continue
                rel = (val - vmin) / max(vmax - vmin, 1e-8)
                txt_color = "white" if (is_div and rel > 0.72) or \
                                        (not is_div and rel > 0.60) else "#222222"
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center",
                        fontsize=max(7, FS_ANNOT - 1 - max(0, n - 5)),
                        color=txt_color, zorder=4)

        cbar = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
        cbar.ax.tick_params(labelsize=FS_TICK - 1)
        cbar.outline.set_linewidth(0.6)

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=FS_TICK)
        ax.set_yticklabels(domains, fontsize=FS_TICK)
        ax.set_xlabel("Target Domain", fontsize=FS_LABEL)
        ax.set_ylabel("Source Domain", fontsize=FS_LABEL)
        letter = chr(ord("a") + ax_i)
        ax.set_title(f"({letter}) {title}", fontsize=FS_TITLE, pad=10)
        ax.tick_params(length=0)

    # 数据集注释
    ds_label = ("Office-Caltech10" if dataset == "office_caltech"
                else "CWRU Bearing Fault")
    fig.text(0.5, 0.01, f"Dataset: {ds_label}  |  All {n*(n-1)} cross-domain pairs",
             ha="center", fontsize=FS_ANNOT, color="#666666", style="italic")

    path = os.path.join(save_dir, f"sweep_{dataset}.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Sweep] Heatmap saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_visualization(sweep_results: Dict, save_dir: str,
                             dataset: str, logger=None):
    """从 sweep 结果字典生成热力图 + XLSX。"""
    os.makedirs(save_dir, exist_ok=True)
    table_dir = os.path.join(os.path.dirname(save_dir.rstrip("/")), "tables")
    os.makedirs(table_dir, exist_ok=True)

    _save_sweep_xlsx(sweep_results, table_dir, dataset, logger)
    plot_sweep_heatmap(sweep_results, save_dir, dataset, logger)
