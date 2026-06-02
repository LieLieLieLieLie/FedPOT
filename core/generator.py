"""
core/generator.py — Module 3: Transport-Conditioned CVAE Generation.
Loss = Reconstruction (Gaussian NLL) + β·KL(warm-up) + λ·Sinkhorn OT regularization
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from config import Config
from models.networks import CVAE
from evaluation.utils import AverageMeter


def _sinkhorn_loss(x_gen, cond, reg):
    if x_gen.size(0) <= 1:
        return x_gen.new_tensor(0.0)
    M = torch.cdist(x_gen, cond, p=2).pow(2)
    M = M / (M.detach().max().clamp_min(1e-8))
    log_K = -M / max(reg, 1e-6)
    log_u = torch.zeros(x_gen.size(0), device=x_gen.device, dtype=x_gen.dtype)
    log_v = torch.zeros(cond.size(0),  device=x_gen.device, dtype=x_gen.dtype)
    log_a = torch.full_like(log_u, -np.log(x_gen.size(0)))
    log_b = torch.full_like(log_v, -np.log(cond.size(0)))
    for _ in range(30):
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_K.T + log_u[None, :], dim=1)
    T = torch.exp(log_u[:, None] + log_K + log_v[None, :])
    return (T * M).sum()


def _cvae_loss(x_mu, x_lv, x_tgt, z_mu, z_lv, cond,
               beta, ot_lam, ot_reg):
    var   = x_lv.exp().clamp(1e-4)
    recon = (0.5 * (x_lv + (x_tgt - x_mu).pow(2) / var)).mean()
    kl    = (-0.5 * (1 + z_lv - z_mu.pow(2) - z_lv.exp())).mean()
    # OT regularisation aligns the distribution of generated features with
    # the d-domain condition prototypes, encouraging domain-adapted outputs.
    ot_l  = _sinkhorn_loss(x_mu, cond, ot_reg)
    total = recon + beta * kl + ot_lam * ot_l
    return {"total": total, "recon": recon, "kl": kl, "ot": ot_l}


class CVAEGenerator:
    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.device = cfg.device
        self.cvae   = None

    def train(self, t_features, conditions, logger=None):
        cfg_c    = self.cfg.cvae
        x_dim    = t_features.shape[1]
        cond_dim = conditions.shape[1]
        out_dim  = cond_dim

        # BUG FIX 1: out_dim was cond_dim — should be x_dim.
        # The decoder reconstructs t-domain features (xb), not d-domain conditions.
        self.cvae = CVAE(x_dim, cond_dim, out_dim,
                         cfg_c.hidden_dims, cfg_c.latent_dim).to(self.device)
        opt   = torch.optim.Adam(self.cvae.parameters(),
                                 lr=cfg_c.lr, weight_decay=cfg_c.weight_decay)
        sched = StepLR(opt, cfg_c.lr_scheduler_step, cfg_c.lr_scheduler_gamma)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(t_features).float(),
                          torch.from_numpy(conditions).float()),
            batch_size=cfg_c.batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator().manual_seed(self.cfg.seed))
        meters = {k: AverageMeter(k) for k in ("total", "recon", "kl", "ot")}

        warmup = getattr(cfg_c, "kl_warmup_epochs", 0)

        self.cvae.train()
        for epoch in range(1, cfg_c.epochs + 1):
            # KL warm-up: ramp beta from 0→cfg_c.beta over the first
            # kl_warmup_epochs to prevent posterior collapse early in training.
            if warmup > 0 and epoch <= warmup:
                beta_eff = cfg_c.beta * (epoch / warmup)
            else:
                beta_eff = cfg_c.beta

            for m in meters.values(): m.reset()
            for xb, cb in loader:
                xb, cb = xb.to(self.device), cb.to(self.device)
                x_mu, x_lv, z_mu, z_lv = self.cvae(xb, cb)
                # BUG FIX 2: reconstruction target was cb (condition).
                # This allowed the encoder to trivially pass cb through z and
                # the decoder to output cb unchanged — making generated features
                # constant per cluster and freezing all Acc/F1 metrics.
                # Correct target is xb: the encoder (which no longer sees cb)
                # must encode t-domain semantics into z, and the decoder uses
                # (z, cb) to reconstruct xb, blending t-content with d-style.
                tgt_w = float(getattr(cfg_c, "condition_target_weight", 0.35))
                tgt_w = min(max(tgt_w, 0.0), 1.0)
                if xb.shape[1] == cb.shape[1]:
                    x_tgt = (1.0 - tgt_w) * xb + tgt_w * cb
                else:
                    x_tgt = cb
                losses = _cvae_loss(x_mu, x_lv, x_tgt, z_mu, z_lv, cb,
                                    beta_eff, cfg_c.ot_lambda, cfg_c.ot_reg)
                opt.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(self.cvae.parameters(), cfg_c.grad_clip)
                opt.step()
                n = xb.size(0)
                for k, v in losses.items(): meters[k].update(v.item(), n)
            sched.step()
            if logger and epoch % 20 == 0:
                logger.info(f"  [CVAE] {epoch:>3d}/{cfg_c.epochs}  "
                            f"total={meters['total'].avg:.4f}  "
                            f"recon={meters['recon'].avg:.4f}  "
                            f"kl={meters['kl'].avg:.4f}  "
                            f"ot={meters['ot'].avg:.4f}  "
                            f"β={beta_eff:.3f}")

    @torch.no_grad()
    def generate(self, t_features, conditions):
        self.cvae.eval()
        all_gen, all_var = [], []
        loader = DataLoader(
            TensorDataset(torch.from_numpy(t_features).float(),
                          torch.from_numpy(conditions).float()),
            batch_size=256, shuffle=False)
        for xb, cb in loader:
            xb, cb = xb.to(self.device), cb.to(self.device)
            # BUG FIX 3: encoder no longer takes cond (see networks.py fix).
            # Use posterior mean z_mu (not prior sample) for deterministic
            # generation at inference time.
            z_mu, _z_lv = self.cvae.encoder(xb)
            x_gen, x_lv = self.cvae.decoder(z_mu, cb)
            recon_var = x_lv.exp().mean(dim=-1)
            all_gen.append(x_gen.cpu().numpy())
            all_var.append(recon_var.cpu().numpy())
        return (np.concatenate(all_gen).astype(np.float32),
                np.concatenate(all_var).astype(np.float32))
