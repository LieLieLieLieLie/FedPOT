"""
experiments/comm_analysis.py — 实验六：通信效率分析

修复：CrossFGAT-Lite 替换为 ProtoFTL。

输出:
  - results/tables/comm_analysis.xlsx
  - results/figures/comm_total_cost.pdf
  - results/figures/comm_rounds_per_data.pdf
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict

from experiments.plot_style import (
    apply_style, save_pdf, save_xlsx, method_color,
    FEDPOT_COLOR, OTHER_COLORS,
    FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ANNOT,
)


# ─────────────────────────────────────────────────────────────────────────────
# 通信量计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_comm_costs(cfg) -> Dict:
    if cfg.data.dataset == "cwru":
        t_dim = cfg.data.cwru_feature_dim // 2
        d_dim = cfg.data.cwru_feature_dim - t_dim
    else:
        t_dim  = cfg.data.split_dim
        d_dim  = cfg.data.feature_dim - cfg.data.split_dim
    K      = cfg.data.n_classes
    C      = cfg.prototype.n_clusters
    n_t    = 2000
    rounds = 50
    h      = cfg.downstream.hidden_dims[0]
    model_params = d_dim * h + h * h + h * K

    results = {
        "NoTransfer": {
            "desc":    "No communication",
            "data_sent": 0, "rounds": 0, "total": 0,
            "formula": "—",
        },
        "FedAvg-FTL": {
            "desc":    "Transfer model params (once)",
            "data_sent": model_params, "rounds": 1, "total": model_params,
            "formula": "O(P)",
        },
        "DANN-FTL": {
            "desc":    "Transfer feature extractor params",
            "data_sent": model_params, "rounds": 1, "total": model_params,
            "formula": "O(P)",
        },
        "SHOT-FTL": {
            "desc":    "Transfer source model params",
            "data_sent": model_params, "rounds": 1, "total": model_params,
            "formula": "O(P)",
        },
        # ProtoFTL：每类传一个原型向量，单轮
        "ProtoFTL": {
            "desc":    f"Transmit {K} class prototypes (once)",
            "data_sent": K * d_dim,
            "rounds":    1,
            "total":     K * d_dim,
            "formula":   "O(K·d)",
        },
        "FedPOT (Ours)": {
            "desc":    "Prototype vectors only (K+C), one round",
            "data_sent": (K + C) * d_dim,
            "rounds":    1,
            "total":     (K + C) * d_dim,
            "formula":   "O((K+C)·d)",
        },
    }

    fedpot_total = max(results["FedPOT (Ours)"]["total"], 1)
    for v in results.values():
        t = v["total"]
        v["ratio_vs_fedpot"] = t / fedpot_total if t > 0 else 0.0

    return results


# ─────────────────────────────────────────────────────────────────────────────
# XLSX
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(n: float) -> str:
    if n == 0:   return "0"
    if n >= 1e9: return f"{n/1e9:.2f}G"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(int(n))


def _save_comm_xlsx(results: Dict, table_dir: str, logger=None):
    rows = []
    for name, v in results.items():
        rows.append({
            "Method":             name,
            "Formula":            v.get("formula", ""),
            "Data Sent (floats)": _fmt(v["data_sent"]),
            "Rounds":             v["rounds"],
            "Total (floats)":     _fmt(v["total"]),
            "xFedPOT":            f"{v['ratio_vs_fedpot']:.1f}x",
        })
    path = os.path.join(table_dir, "comm_analysis.xlsx")
    save_xlsx(rows, path, sheet_name="CommAnalysis")
    logger and logger.info(f"  [Comm] XLSX saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_comm_comparison(results: Dict, save_dir: str, cfg, logger=None):
    apply_style()

    names   = list(results.keys())
    totals  = [max(results[n]["total"], 1) for n in names]
    rounds_ = [results[n]["rounds"] for n in names]
    colors  = [method_color(n) for n in names]
    tick_fs  = FS_TICK + 3
    label_fs = FS_LABEL + 4
    annot_fs = FS_ANNOT + 4
    fig_size = (9.6, 7.0)
    axis_box = [0.16, 0.24, 0.78, 0.66]

    # Total communication cost.
    fig1, ax1 = plt.subplots(figsize=fig_size)
    ax1.set_position(axis_box)
    x = np.arange(len(names))
    for xi, (tot, color, name) in enumerate(zip(totals, colors, names)):
        is_fp = "FedPOT" in name
        ax1.vlines(xi, 1, tot, colors=color,
                   lw=4.2 if is_fp else 3.0, alpha=0.96, zorder=3)
        ax1.scatter(xi, tot, color=color,
                    s=180 if is_fp else 120,
                    zorder=5, edgecolors="white", linewidths=1.2)

        ratio = results[name]["ratio_vs_fedpot"]
        label_text = _fmt(tot)
        ratio_text = (f"\n({ratio:.0f}x)" if ratio > 1.5
                      else "\n(1x)" if is_fp else "")
        ax1.annotate(
            label_text + ratio_text,
            xy=(xi, tot),
            xytext=(0, 18 if is_fp else 10),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=annot_fs - 1 if not is_fp else annot_fs,
            color=color,
            fontweight="bold" if is_fp else "normal",
        )

    if "FedPOT (Ours)" in names:
        fp_xi = names.index("FedPOT (Ours)")
        ax1.axvspan(fp_xi - 0.4, fp_xi + 0.4,
                    color=FEDPOT_COLOR, alpha=0.08, zorder=1)

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=28, ha="right", fontsize=tick_fs)
    ax1.set_ylabel("Total Communication  (# floats, log scale)",
                   fontsize=label_fs)
    ax1.tick_params(axis="y", labelsize=tick_fs)
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: _fmt(v)))
    ax1.set_xlim(-0.6, len(names) - 0.4)
    path1 = os.path.join(save_dir, "comm_total_cost.pdf")
    save_pdf(fig1, path1, tight=False)

    # Rounds vs. data per round.
    fig2, ax2 = plt.subplots(figsize=fig_size)
    ax2.set_position(axis_box)
    data_sents = [max(results[n]["data_sent"], 1) for n in names]
    round_vals = [max(r, 0.5) for r in rounds_]

    for name, ds, rv, color in zip(names, data_sents, round_vals, colors):
        is_fp = "FedPOT" in name
        tot   = results[name]["total"]
        ax2.scatter(ds, rv,
                    s=np.log10(max(tot, 10)) * 135,
                    color=color, alpha=0.92,
                    edgecolors="white", linewidths=1.5, zorder=4)
        label_offsets = {
            "NoTransfer": (54, -54),
            "FedAvg-FTL": (-88, 76),
            "DANN-FTL": (0, -82),
            "SHOT-FTL": (-92, -66),
            "ProtoFTL": (-92, -76),
            "FedPOT (Ours)": (-132, 70),
        }
        dx, dy = label_offsets.get(name, (8, 5))
        ax2.annotate(name, xy=(ds, rv),
                     xytext=(dx, dy),
                     textcoords="offset points",
                     fontsize=annot_fs - 1,
                     color=color,
                     fontweight="bold" if is_fp else "normal",
                     bbox=dict(facecolor="white", edgecolor="none",
                               alpha=0.72, pad=1.1),
                     arrowprops=dict(arrowstyle="-", color=color, lw=1.0,
                                     alpha=0.72, shrinkA=2, shrinkB=5))

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Data Sent per Round  (floats, log scale)",
                   fontsize=label_fs)
    ax2.set_ylabel("Communication Rounds  (log scale)", fontsize=label_fs)
    ax2.tick_params(axis="both", labelsize=tick_fs)
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: _fmt(v)))
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: str(int(v)) if v >= 1 else "0"))
    ax2.margins(x=0.28, y=0.34)
    ax2.text(0.98, 0.02,
             "Bubble size proportional to log(total cost)",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=annot_fs - 1, color="#888888", style="italic")

    path2 = os.path.join(save_dir, "comm_rounds_per_data.pdf")
    save_pdf(fig2, path2, tight=False)
    logger and logger.info(f"  [Comm] Charts saved → {path1}; {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# 文字汇报
# ─────────────────────────────────────────────────────────────────────────────

def print_comm_table(results: Dict, logger=None):
    sep   = "=" * 78
    lines = [sep,
             f"  {'Method':<22}  {'Formula':>14}  {'Data Sent':>12}  "
             f"{'Rounds':>7}  {'Total':>14}  {'xFedPOT':>8}",
             "-" * 78]
    for name, v in results.items():
        ratio  = v.get("ratio_vs_fedpot", 0)
        rstr   = f"{ratio:.1f}x" if ratio > 0 else "1.0x"
        lines.append(
            f"  {name:<22}  {v.get('formula',''):>14}  "
            f"{_fmt(v['data_sent']):>12}  {v['rounds']:>7d}  "
            f"{_fmt(v['total']):>14}  {rstr:>8}"
            + (" *" if "Ours" in name else "")
        )
    lines.append(sep)
    out = "\n".join(lines)
    if logger: logger.info(out)
    else: print(out)


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

def run_comm_analysis(cfg, save_dir: str, logger=None) -> Dict:
    results = compute_comm_costs(cfg)
    print_comm_table(results, logger)

    table_dir = os.path.join(
        os.path.dirname(save_dir.rstrip("/").rstrip("\\")), "tables")
    os.makedirs(table_dir, exist_ok=True)
    _save_comm_xlsx(results, table_dir, logger)
    plot_comm_comparison(results, save_dir, cfg, logger)

    out = os.path.join(save_dir, "comm_analysis.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger and logger.info(f"  [Comm] JSON saved → {out}")
    return results

