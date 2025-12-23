from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional

def standardize_per_instance(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Standardize along the last dimension (rows) per batch item."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)

def masked_mean(x: Tensor, mask: Optional[Tensor], dim: int) -> Tensor:
    if mask is None:
        return x.mean(dim=dim)
    m = mask.to(dtype=x.dtype)
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(1e-6)
    return num / den
