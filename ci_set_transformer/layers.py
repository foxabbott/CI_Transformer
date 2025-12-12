from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional

class MAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        ff_dim = ff_dim or (4 * dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, X: Tensor, Y: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        attn_out, _ = self.attn(query=X, key=Y, value=Y, key_padding_mask=key_padding_mask)
        H = self.ln1(X + attn_out)
        return self.ln2(H + self.ff(H))

class SAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(dim, num_heads, ff_dim=ff_dim, dropout=dropout)

    def forward(self, X: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return self.mab(X, X, key_padding_mask=key_padding_mask)

class ISAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_inducing: int = 32, ff_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)
        self.mab1 = MAB(dim, num_heads, ff_dim=ff_dim, dropout=dropout)
        self.mab2 = MAB(dim, num_heads, ff_dim=ff_dim, dropout=dropout)

    def forward(self, X: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        B = X.size(0)
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X, key_padding_mask=key_padding_mask)
        return self.mab2(X, H)

class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, ff_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim, num_heads, ff_dim=ff_dim, dropout=dropout)

    def forward(self, X: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        B = X.size(0)
        S = self.S.expand(B, -1, -1)
        return self.mab(S, X, key_padding_mask=key_padding_mask)
