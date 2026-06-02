"""
models/networks.py — CVAE + DownstreamClassifier architectures.
"""

from typing import List
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.1, use_bn=True):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h)]
            if use_bn: layers += [nn.BatchNorm1d(h)]
            layers += [nn.LeakyReLU(0.2)]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class CVAEEncoder(nn.Module):
    """
    BUG FIX: encoder now takes ONLY x (t-domain features), NOT [x, cond].

    Original bug: encoding [xb, cb] with reconstruction target cb meant the
    encoder could trivially pass cb through z and let the decoder output cb
    without using any xb information — causing generated features to be
    constant per cluster and freezing all Acc/F1 metrics.

    With cond removed, z must compress xb; the decoder then combines
    t-domain latent structure with the d-domain condition to reconstruct xb.
    """
    def __init__(self, x_dim, cond_dim, hidden_dims, latent_dim, dropout=0.1):
        super().__init__()
        self.shared    = MLP(x_dim, hidden_dims, hidden_dims[-1], dropout)
        self.fc_mu     = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x, cond=None):
        # cond kept for API compatibility but intentionally not used
        h = self.shared(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 2)


class CVAEDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dims, out_dim, dropout=0.1):
        super().__init__()
        self.shared    = MLP(latent_dim + cond_dim, hidden_dims, hidden_dims[-1], dropout)
        self.fc_mu     = nn.Linear(hidden_dims[-1], out_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, z, cond):
        h = self.shared(torch.cat([z, cond], -1))
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 2)


class CVAE(nn.Module):
    def __init__(self, x_dim, cond_dim, out_dim, hidden_dims, latent_dim, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CVAEEncoder(x_dim, cond_dim, hidden_dims, latent_dim, dropout)
        self.decoder = CVAEDecoder(latent_dim, cond_dim,
                                   list(reversed(hidden_dims)), out_dim, dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return mu

    def forward(self, x, cond):
        z_mu, z_lv = self.encoder(x)   # encoder no longer receives cond
        z          = self.reparameterize(z_mu, z_lv)
        x_mu, x_lv = self.decoder(z, cond)
        return x_mu, x_lv, z_mu, z_lv

    @torch.no_grad()
    def generate_with_uncertainty(self, cond):
        self.eval()
        z = torch.randn(cond.size(0), self.latent_dim, device=cond.device)
        x_mu, x_lv = self.decoder(z, cond)
        return x_mu, x_lv.exp().mean(dim=-1)


class DownstreamClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dims, dropout=0.3):
        super().__init__()
        self.net = MLP(in_dim, hidden_dims, n_classes, dropout)

    def forward(self, x): return self.net(x)
