from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .utils import kl_diag_gaussian, log_density_gaussian

def recon_loss(x_recon_logits: Tensor, x: Tensor, kind: str) -> Tensor:
    # returns (B,)
    if kind == "bce":
        # BCEWithLogits over pixels
        loss = F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction="none")
        return loss.flatten(1).sum(dim=1)
    if kind == "mse":
        # assume decoder outputs raw values; use MSE
        loss = F.mse_loss(torch.sigmoid(x_recon_logits), x, reduction="none")
        return loss.flatten(1).sum(dim=1)
    raise ValueError("recon_loss kind must be 'bce' or 'mse'.")

def tc_vae_decompose(mu: Tensor, logvar: Tensor, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Beta-TC-VAE decomposition estimates (B,) for MI, TC, DW-KL.

    Following Chen et al. (2018) style minibatch estimator.
    """
    B, D = z.shape
    # log q(z|x): sum over dims
    log_q_zCx = log_density_gaussian(z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0)).sum(dim=2)  # (B,B)

    # log q(z): log mean_x q(z|x)
    # log q(z) for each sample in batch:
    log_q_z = torch.logsumexp(log_q_zCx, dim=1) - torch.log(torch.tensor(float(B), device=z.device))

    # log prod_j q(z_j)
    # log q(z_j): log mean_x q(z_j|x)
    log_q_zj = torch.logsumexp(log_density_gaussian(
        z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0)  # (B,B,D)
    ), dim=1) - torch.log(torch.tensor(float(B), device=z.device))  # (B,D)
    log_prod_q_zj = log_q_zj.sum(dim=1)

    # log p(z) = standard normal
    log_p_z = (-0.5 * (z.pow(2) + torch.log(torch.tensor(2*torch.pi, device=z.device)))).sum(dim=1)

    mi = (log_q_zCx.diag() - log_q_z)                      # (B,)
    tc = (log_q_z - log_prod_q_zj)                         # (B,)
    dw_kl = (log_prod_q_zj - log_p_z)                      # (B,)
    return mi, tc, dw_kl

def dip_regularizer(mu: Tensor, logvar: Tensor, dip_type: str, lam_diag: float, lam_offdiag: float) -> Tensor:
    """DIP-VAE regularizer scalar (batch estimate)."""
    B, D = mu.shape
    mu_centered = mu - mu.mean(dim=0, keepdim=True)
    cov_mu = (mu_centered.t() @ mu_centered) / float(B)  # (D,D)

    if dip_type == "ii":
        # add expected covariance from encoder variance
        exp_var = logvar.exp().mean(dim=0)  # (D,)
        cov = cov_mu + torch.diag(exp_var)
    else:
        cov = cov_mu

    diag = torch.diagonal(cov)
    off = cov - torch.diag(diag)

    reg = lam_diag * ((diag - 1.0) ** 2).sum() + lam_offdiag * (off ** 2).sum()
    return reg
