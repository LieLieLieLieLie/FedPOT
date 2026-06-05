"""
experiments/privacy_analysis.py �� ʵ��������˽Ԥ�� �� vs ����

�޸���
  - ɾ����ͼ����⣨set_title ��Ϊ��������ͼ��ĸ��ע��
  - ͼ���������Ͻ�

���:
  - results/tables/privacy_{dataset}.xlsx
  - results/figures/privacy_curve_{dataset}.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict

from experiments.plot_style import (
    apply_style, save_pdf, save_xlsx,
    FEDPOT_COLOR, OTHER_COLORS,
    FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ANNOT,
)

EPSILON_GRID   = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 1e6]
EPSILON_LABELS = ["0.1", "0.5", "1.0", "2.0", "5.0", "10.0", "inf"]
EPSILON_XTICK  = ["0.1", "0.5", "1.0", "2.0", "5.0", "10.0", "No DP\n(inf)"]


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# ʵ������
# ����������������������������������������������������������������������������������������������������������������������������������������������������������

def run_privacy_analysis(args, dataset: str, build_config_fn,
                         logger=None) -> Dict:
    """Fixed-backbone privacy sweep.

    Train once at No-DP (��=��) to fix the CVAE backbone, then for each �� only
    re-run Phase 1 (DP noise on prototypes), Phase 2 (OT), Phase 3 (generation
    with frozen CVAE), Phase 4 (filtering), and Phase 5 (downstream re-train).
    This isolates the DP-noise effect from CVAE training stochasticity and
    produces a clean, monotonic privacy-accuracy curve.
    """
    from trainer import FedPOTTrainer

    all_results = {}
    orig_eps  = args.epsilon
    orig_name = args.exp_name

    # Train the fixed backbone once at No-DP
    logger and logger.info("  [Privacy] Training fixed backbone at No-DP (��=��) ...")
    args.epsilon  = 1e6
    args.exp_name = f"privacy_{dataset}_backbone"
    cfg_backbone  = build_config_fn(args, dataset)
    backbone      = FedPOTTrainer(cfg_backbone)
    no_dp_result  = backbone.train_test()
    all_results["inf"] = no_dp_result.get("FedPOT", {})

    # Sweep �� with frozen CVAE backbone
    for eps, label in zip(EPSILON_GRID[:-1], EPSILON_LABELS[:-1]):
        logger and logger.info(f"  [Privacy] �� = {label} (fixed backbone) ...")
        try:
            result = backbone.retrain_with_epsilon(eps)
            all_results[label] = result.get("FedPOT", {})
        except Exception as e:
            logger and logger.info(f"  [Privacy] ��={label} FAILED: {e}")
            all_results[label] = {}

    args.epsilon  = orig_eps
    args.exp_name = orig_name
    return all_results


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# XLSX
# ����������������������������������������������������������������������������������������������������������������������������������������������������������

def _save_privacy_xlsx(results: Dict, table_dir: str, dataset: str,
                       logger=None):
    rows = []
    for label in EPSILON_LABELS:
        m = results.get(label, {})
        rows.append({
            "Privacy Budget (��)": label,
            "Accuracy":           round(m.get("accuracy",  float("nan")), 4),
            "Macro-F1":           round(m.get("macro_f1",  float("nan")), 4),
            "Macro-AUC":          round(m.get("macro_auc", float("nan")), 4),
        })
    path = os.path.join(table_dir, f"privacy_{dataset}.xlsx")
    save_xlsx(rows, path, sheet_name=f"Privacy_{dataset}")
    logger and logger.info(f"  [Privacy] XLSX saved -> {path}")


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# ��ͼ���ģ�ͼ�����Ͻǣ��޴���⣩
# ����������������������������������������������������������������������������������������������������������������������������������������������������������

def _extract(results):
    accs  = [results.get(l, {}).get("accuracy",  np.nan) for l in EPSILON_LABELS]
    f1s   = [results.get(l, {}).get("macro_f1",  np.nan) for l in EPSILON_LABELS]
    aucs  = [results.get(l, {}).get("macro_auc", np.nan) for l in EPSILON_LABELS]
    return np.array(accs), np.array(f1s), np.array(aucs)


def _draw_panel(ax, results: Dict):
    accs, f1s, aucs = _extract(results)
    x = np.arange(len(EPSILON_LABELS))

    ax.plot(x, accs, "o-",  color=FEDPOT_COLOR,   lw=2.5, ms=8,
            label="Accuracy",  zorder=4)
    ax.plot(x, f1s,  "s--", color=OTHER_COLORS[2], lw=2.2, ms=7,
            label="Macro-F1",  zorder=4)
    ax.plot(x, aucs, "^:",  color=OTHER_COLORS[3], lw=2.0, ms=7,
            label="Macro-AUC", zorder=4)

    # �� DP �Ͻ�
    no_dp_acc = accs[-1]
    if not np.isnan(no_dp_acc):
        ax.axhline(no_dp_acc, color="#AAAAAA", lw=1.2, ls=":",
                   label=f"No-DP ceiling ({no_dp_acc:.3f})")

    # Ĭ�� ��=1.0
    default_label = "2.0"
    default_x = EPSILON_LABELS.index(default_label)
    ax.axvline(default_x, color="#FFAA53", lw=1.5, ls="--", alpha=0.85,
               label=f"Default epsilon = {default_label}", zorder=3)
    ax.axvspan(default_x - 0.3, default_x + 0.3,
               color="#FFAA53", alpha=0.08, zorder=2)

    # ��עĬ�� �� ��������
    if not np.isnan(accs[default_x]) and not np.isnan(no_dp_acc):
        y_anchor = accs[default_x]
        y_text   = min(0.98, y_anchor + 0.08)
        # Only show "% retained" when the value is �� No-DP ceiling.
        # When default �� gives the same or better Acc than No-DP (can happen
        # due to training noise on small datasets), show the absolute value.
        if no_dp_acc > 0 and accs[default_x] <= no_dp_acc + 1e-4:
            retain   = accs[default_x] / no_dp_acc * 100
            annot_txt = f"{retain:.1f}% retained"
        else:
            annot_txt = f"��={default_label}: {accs[default_x]:.3f}"
        ax.annotate(
            annot_txt,
            xy=(default_x, y_anchor),
            xytext=(default_x + 0.75, y_text),
            fontsize=FS_ANNOT, color=FEDPOT_COLOR,
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=FEDPOT_COLOR, alpha=0.92, lw=0.8),
            arrowprops=dict(arrowstyle="-", color=FEDPOT_COLOR,
                            lw=1.2, alpha=0.75,
                            connectionstyle="arc3,rad=0.18")
        )

    ax.set_xticks(x)
    ax.set_xticklabels(EPSILON_XTICK, fontsize=FS_TICK)
    ax.set_xlabel("Privacy Budget ��", fontsize=FS_LABEL)
    ax.set_ylabel("Score", fontsize=FS_LABEL)

    all_finite = [v for v in np.concatenate([accs, f1s, aucs])
                  if not np.isnan(v)]
    ymin = max(0, min(all_finite) - 0.06) if all_finite else 0
    ax.set_ylim(ymin, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    # ͼ�������Ͻ�
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper right",
              framealpha=0.88, edgecolor="#CCCCCC", ncol=1)


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# �������
# ����������������������������������������������������������������������������������������������������������������������������������������������������������

def plot_privacy_curve(results: Dict, save_dir: str, dataset: str,
                       logger=None):
    """�����ݼ��汾��--privacy �������ã���"""
    table_dir = os.path.join(
        os.path.dirname(save_dir.rstrip("/").rstrip("\\")), "tables")
    os.makedirs(table_dir, exist_ok=True)
    _save_privacy_xlsx(results, table_dir, dataset, logger)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _draw_panel(ax, results)

    path = os.path.join(save_dir, f"privacy_curve_{dataset}.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Privacy] Curve saved -> {path}")


def plot_privacy_curve_combined(results_oc: Dict, results_cwru: Dict,
                                save_dir: str, logger=None):
    """˫���ݼ����Ű汾��--full ģʽ����"""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0),
                             gridspec_kw={"wspace": 0.38})
    _draw_panel(axes[0], results_oc)
    _draw_panel(axes[1], results_cwru)

    # ������ͼ�ֱ�����ݼ���ǩ�����Ͻ�С�֣�
    for ax, label in zip(axes, ["Office-Home", "CWRU Bearing Fault"]):
        ax.text(0.98, 0.04, label, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FS_ANNOT,
                color="#555555", style="italic")

    path = os.path.join(save_dir, "privacy_curve.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Privacy] Combined curve saved -> {path}")


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# ���ֻ㱨
# ����������������������������������������������������������������������������������������������������������������������������������������������������������

def print_privacy_table(results: Dict, logger=None):
    sep   = "=" * 60
    lines = [sep,
             f"  {'��':<14}  {'Accuracy':>10}  {'Macro-F1':>10}  {'AUC':>10}",
             "-" * 60]
    for label in EPSILON_LABELS:
        m      = results.get(label, {})
        acc    = f"{m.get('accuracy',  float('nan')):.4f}"
        f1     = f"{m.get('macro_f1',  float('nan')):.4f}"
        auc    = f"{m.get('macro_auc', float('nan')):.4f}"
        marker = " * default" if label == "2.0" else ""
        lines.append(f"  {label:<14}  {acc:>10}  {f1:>10}  {auc:>10}{marker}")
    lines.append(sep)
    out = "\n".join(lines)
    if logger: logger.info(out)
    else: print(out)

