"""
Non-stationarity / anomaly (Phase 4 stub).

Extend with a full VAE on traffic windows + clustering for adaptive fine-tuning.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrafficVAEStub(nn.Module):
    """Minimal VAE skeleton for flattened (seq_len * N) windows — not trained by default."""

    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 2 * latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, input_dim))
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        recon = self.dec(z)
        return recon, mu, logvar
