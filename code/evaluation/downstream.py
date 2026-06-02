"""
evaluation/downstream.py — Downstream task training & evaluation.
Metrics: Accuracy, Macro-F1, Macro-AUC
"""

import numpy as np
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from config import Config
from models.networks import DownstreamClassifier
from evaluation.utils import AverageMeter


def _stable_tag_seed(seed: int, tag: str) -> int:
    h = 0
    for ch in str(tag):
        h = (h * 131 + ord(ch)) % 1000003
    return int(seed + h)


def _smooth_cross_entropy(logits, targets, n_classes: int, eps: float = 0.0,
                          weight: Optional[torch.Tensor] = None):
    """
    Cross-entropy with optional label smoothing.

    Label smoothing (eps > 0) converts hard one-hot pseudo labels to soft
    targets: correct class = 1-eps+(eps/K), others = eps/K.  This prevents
    the classifier from becoming over-confident on noisy pseudo labels and
    substantially improves generalisation when ~20-40% of pseudo labels are
    incorrect (typical in unsupervised cluster-to-class assignment).
    """
    if eps <= 0.0:
        loss = F.cross_entropy(logits, targets, reduction="none")
    else:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, eps / max(n_classes - 1, 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - eps)
        loss = -(smooth * log_probs).sum(dim=-1)
    if weight is not None:
        loss = loss * weight
        return loss.sum() / weight.sum().clamp_min(1e-8)
    return loss.mean()


class DownstreamTrainer:
    def __init__(self, cfg: Config, tag: str = ""):
        self.cfg    = cfg
        self.tag    = tag
        self.device = cfg.device
        self.model: Optional[DownstreamClassifier] = None

    def train(self, X_train, y_train, logger=None, sample_weight=None):
        cfg_ds = self.cfg.downstream
        n_cls  = self.cfg.data.n_classes
        eps    = getattr(cfg_ds, "label_smoothing", 0.0)
        torch.manual_seed(_stable_tag_seed(self.cfg.seed, self.tag))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_stable_tag_seed(self.cfg.seed, self.tag))

        self.model = DownstreamClassifier(
            X_train.shape[1], n_cls,
            cfg_ds.hidden_dims, cfg_ds.dropout).to(self.device)

        opt   = torch.optim.Adam(self.model.parameters(),
                                 lr=cfg_ds.lr, weight_decay=cfg_ds.weight_decay)
        sched = StepLR(opt, cfg_ds.lr_scheduler_step, cfg_ds.lr_scheduler_gamma)
        tensors = [
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
        ]
        if sample_weight is not None:
            tensors.append(torch.from_numpy(sample_weight).float())
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=cfg_ds.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.cfg.seed))
        meter = AverageMeter("ce")

        self.model.train()
        for epoch in range(1, cfg_ds.epochs + 1):
            meter.reset()
            for batch in loader:
                xb, yb = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(xb)
                wb = batch[2].to(self.device) if len(batch) > 2 else None
                loss = _smooth_cross_entropy(logits, yb, n_cls, eps=eps, weight=wb)
                opt.zero_grad()
                loss.backward()
                # Gradient clipping for training stability with noisy pseudo labels
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                meter.update(loss.item(), xb.size(0))
            sched.step()
            if logger and epoch % 40 == 0:
                logger.info(f"  [{self.tag}] epoch {epoch:>3d}  CE={meter.avg:.4f}")

    @torch.no_grad()
    def predict_logits(self, X_test):
        self.model.eval()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test).float()),
            batch_size=256, shuffle=False)
        return torch.cat([self.model(xb[0].to(self.device)).cpu()
                          for xb in loader]).numpy()

    @torch.no_grad()
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        logits = self.predict_logits(X_test)
        return metrics_from_logits(logits, y_test, self.cfg.data.n_classes)


def metrics_from_logits(logits, y_test, n_classes: int) -> Dict[str, float]:
        preds  = logits.argmax(1)
        probs  = torch.from_numpy(logits).softmax(-1).numpy()

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="macro", zero_division=0)
        try:
            yb  = label_binarize(y_test, classes=list(range(n_classes)))
            auc = roc_auc_score(yb, probs, average="macro", multi_class="ovr")
        except Exception:
            auc = float("nan")
        return {"accuracy": float(acc), "macro_f1": float(f1), "macro_auc": float(auc)}


def print_results(results: Dict, logger=None):
    sep   = "=" * 55
    lines = [sep, "  Results", sep]
    for name, m in results.items():
        lines.append(f"  [{name}]  "
                     f"Acc={m['accuracy']:.4f}  "
                     f"F1={m['macro_f1']:.4f}  "
                     f"AUC={m['macro_auc']:.4f}")
    lines.append(sep)
    out = "\n".join(lines)
    if logger: logger.info(out)
    else: print(out)
