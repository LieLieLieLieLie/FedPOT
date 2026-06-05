"""
experiments/hyperparam.py �� ʵ���ģ������������Է���

�޸���ͼ��ͳһ����������ͼ�·�����һ�У������������ص���

���:
  - results/tables/hyperparam_{dataset}.xlsx
  - results/figures/hyperparam_{dataset}.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from typing import Dict

from experiments.plot_style import (
    apply_style, save_pdf, save_xlsx,
    FEDPOT_COLOR, OTHER_COLORS,
    FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ANNOT,
)

PARAM_GRIDS = {
    "beta":       [0.1, 0.5, 1.0, 2.0, 5.0],
    "ot_lambda":  [0.0, 0.05, 0.1, 0.5, 1.0],
    "latent_dim": [32, 64, 128, 256],
}

PARAM_LABELS = {
    "beta":       r"$\beta$  (KL Weight)",
    "ot_lambda":  r"$\lambda$  (OT Reg Weight)",
    "latent_dim": "Latent Dimension  $z$",
}

PARAM_XLABELS = {
    "beta":       ["0.1", "0.5", "1.0", "2.0", "5.0"],
    "ot_lambda":  ["0.0", "0.05", "0.1", "0.5", "1.0"],
    "latent_dim": ["32", "64", "128", "256"],
}

PARAM_DEFAULTS = {
    "beta":       "1.0",
    "ot_lambda":  "0.1",
    "latent_dim": "128",
}

DATASET_DEFAULTS = {
    "office_home": {
        "beta": "2.0",
        "ot_lambda": "0.0",
        "latent_dim": "128",
    },
    "cwru": {
        "beta":       "0.1",   # ��=0.1 gives best CWRU Acc/F1 in sweep
        "ot_lambda":  "0.1",
        "latent_dim": "128",
    },
}


def _defaults_for(dataset):
    return DATASET_DEFAULTS.get(dataset, PARAM_DEFAULTS)

PARAM_COLORS = {
    "beta":       FEDPOT_COLOR,
    "ot_lambda":  OTHER_COLORS[3],
    "latent_dim": OTHER_COLORS[1],
}


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# ʵ������
# ����������������������������������������������������������������������������������������������������������������������������������������������������������

def run_hyperparam_analysis(args, dataset: str, build_config_fn,
                            logger=None) -> Dict:
    from trainer import FedPOTTrainer
    all_results = {}
    orig_name   = args.exp_name

    for param, grid in PARAM_GRIDS.items():
        all_results[param] = {}
        orig_val = getattr(args, param)
        for val in grid:
            label = str(val)
            logger and logger.info(f"  [Hyperparam] {param} = {val} ...")
            setattr(args, param, val)
            args.exp_name = f"hp_{dataset}_{param}_{label}"
            cfg = build_config_fn(args, dataset)
            try:
                result = FedPOTTrainer(cfg).train_test()
                all_results[param][label] = result.get("FedPOT", {})
            except Exception as e:
                logger and logger.info(f"  [Hyperparam] {param}={val} FAILED: {e}")
                all_results[param][label] = {}
        setattr(args, param, orig_val)

    args.exp_name = orig_name
    return all_results


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# XLSX
# ����������������������������������������������������������������������������������������������������������������������������������������������������������

def _save_hp_xlsx(results: Dict, table_dir: str, dataset: str, logger=None):
    rows = []
    defaults = _defaults_for(dataset)
    for param, data in results.items():
        for val_str, m in data.items():
            rows.append({
                "Parameter": PARAM_LABELS[param].replace("$", "").replace("\\", ""),
                "Value":     val_str,
                "Default":   "*" if val_str == defaults[param] else "",
                "Accuracy":  round(m.get("accuracy",  float("nan")), 4),
                "Macro-F1":  round(m.get("macro_f1",  float("nan")), 4),
                "Macro-AUC": round(m.get("macro_auc", float("nan")), 4),
            })
    path = os.path.join(table_dir, f"hyperparam_{dataset}.xlsx")
    save_xlsx(rows, path, sheet_name=f"Hyperparam_{dataset}")
    logger and logger.info(f"  [Hyperparam] XLSX saved -> {path}")


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# ��ͼ
# ����������������������������������������������������������������������������������������������������������������������������������������������������������

def plot_hyperparam_sensitivity(results: Dict, save_dir: str,
                                dataset: str, logger=None):
    table_dir = os.path.join(
        os.path.dirname(save_dir.rstrip("/").rstrip("\\")), "tables")
    os.makedirs(table_dir, exist_ok=True)
    _save_hp_xlsx(results, table_dir, dataset, logger)

    apply_style()

    # Ϊ�ײ�ͼ�������ռ�
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5.0),
        gridspec_kw={"wspace": 0.38}
    )
    fig.subplots_adjust(bottom=0.22)   # �ײ����ո�����ͼ��

    for ax, param in zip(axes, PARAM_GRIDS.keys()):
        data    = results.get(param, {})
        xlabels = PARAM_XLABELS[param]
        accs    = [data.get(l, {}).get("accuracy",  np.nan) for l in xlabels]
        f1s     = [data.get(l, {}).get("macro_f1",  np.nan) for l in xlabels]
        x       = np.arange(len(xlabels))
        color   = PARAM_COLORS[param]

        ax.plot(x, accs, "o-",  color=color,          lw=2.5, ms=8, zorder=4)
        ax.plot(x, f1s,  "s--", color=OTHER_COLORS[4], lw=2.0, ms=7, zorder=4)

        # �ȶ���
        acc_arr = np.array([v for v in accs if not np.isnan(v)])
        if len(acc_arr):
            best = acc_arr.max()
            ax.axhspan(best - 0.02, best + 0.001,
                       color=color, alpha=0.08)

        # Ĭ��ֵ���� + ����
        default_str = _defaults_for(dataset)[param]
        if default_str in xlabels:
            di = xlabels.index(default_str)
            ax.axvspan(di - 0.35, di + 0.35, color="#FFF0A0", alpha=0.55, zorder=2)
            ax.axvline(di, color="#FFAA53", lw=1.6, ls="--", alpha=0.90, zorder=3)

        # ÿ���ע��ֵ������ Accuracy��
        for xi, acc in zip(x, accs):
            if not np.isnan(acc):
                ax.annotate(f"{acc:.3f}", xy=(xi, acc),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=FS_ANNOT - 1,
                            color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=FS_TICK)
        ax.set_xlabel(PARAM_LABELS[param], fontsize=FS_LABEL)
        ax.set_ylabel("Score", fontsize=FS_LABEL)

        all_vals = [v for v in accs + f1s if not np.isnan(v)]
        ymin = max(0, min(all_vals) - 0.06) if all_vals else 0
        ax.set_ylim(ymin, 1.03)

        subplot_letter = chr(ord("a") + list(PARAM_GRIDS.keys()).index(param))
        ax.set_title(
            f"({subplot_letter}) Sensitivity to {PARAM_LABELS[param]}",
            fontsize=FS_TITLE, pad=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    # ���� ����ͼ������ͼ�·�����һ�� ������������������������������������������������������������������������������������
    legend_handles = [
        mlines.Line2D([], [], color=FEDPOT_COLOR, marker="o", lw=2.5,
                      ms=8, label=r"Accuracy ($\beta$ panel)"),
        mlines.Line2D([], [], color=OTHER_COLORS[3], marker="o", lw=2.5,
                      ms=8, label=r"Accuracy ($\lambda$ panel)"),
        mlines.Line2D([], [], color=OTHER_COLORS[1], marker="o", lw=2.5,
                      ms=8, label="Accuracy (latent panel)"),
        mlines.Line2D([], [], color=OTHER_COLORS[4], marker="s", lw=2.0,
                      ms=7,  ls="--", label="Macro-F1"),
        mlines.Line2D([], [], color="#FFF0A0", lw=8, alpha=0.9,
                      label="Default value region"),
    ]
    fig.legend(handles=legend_handles,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.01),
               ncol=5,
               fontsize=FS_LEGEND,
               framealpha=0.90,
               edgecolor="#CCCCCC",
               handlelength=1.8,
               columnspacing=1.2)

    path = os.path.join(save_dir, f"hyperparam_{dataset}.pdf")
    save_pdf(fig, path)
    logger and logger.info(f"  [Hyperparam] Plot saved -> {path}")


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# ���ֻ㱨
# ����������������������������������������������������������������������������������������������������������������������������������������������������������

def print_hyperparam_table(results: Dict, logger=None):
    for param, data in results.items():
        sep   = "-" * 48
        lines = [f"\n  Param: {PARAM_LABELS[param]}", sep,
                 f"  {'Value':<12}  {'Accuracy':>10}  {'Macro-F1':>8}"]
        lines.append(sep)
        for val_str, m in data.items():
            acc    = f"{m.get('accuracy', float('nan')):.4f}"
            f1     = f"{m.get('macro_f1', float('nan')):.4f}"
            marker = " * default" if val_str == _defaults_for("").get(param) else ""
            lines.append(f"  {val_str:<12}  {acc:>10}  {f1:>8}{marker}")
        lines.append(sep)
        out = "\n".join(lines)
        if logger: logger.info(out)
        else: print(out)

