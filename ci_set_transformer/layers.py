from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional

class MAB(nn.Module):
    """
    Multihead Attention Block (MAB) as used in Set Transformer architectures.

    Applies multi-head self-attention between a 'query' input X and 
    context input Y, followed by layer normalization and a feed-forward block 
    with residual connection. Can be used for both self- and cross-attention.

    Args:
        dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        ff_dim (Optional[int]): Dimension for the hidden feedforward (FF) layer. Defaults to 4*dim if None.
        dropout (float): Dropout probability.

    Inputs:
        X (Tensor): Query tensor of shape (B, N, dim).
        Y (Tensor): Context tensor (as both key and value) of shape (B, M, dim).

    Returns:
        Tensor: Output tensor of shape (B, N, dim).
    """
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

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        attn_out, _ = self.attn(query=X, key=Y, value=Y)
        H = self.ln1(X + attn_out)
        return self.ln2(H + self.ff(H))

class ISAB(nn.Module):
    """
    Induced Set Attention Block (ISAB).

    Applies attention using a set of learned inducing points for scalable 
    set attention. Consists of two MABs: 
    (1) Inducing points attend to the input set.
    (2) Input attends to the induced set.

    Args:
        dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_inducing (int): Number of inducing points.
        ff_dim (Optional[int]): Feedforward hidden dimension.
        dropout (float): Dropout probability.

    Inputs:
        X (Tensor): Input tensor of shape (B, N, dim).

    Returns:
        Tensor: Output tensor of shape (B, N, dim).
    """
    def __init__(self, dim: int, num_heads: int, num_inducing: int = 32, ff_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)  # Initialise inducing points with small magnitude to prevent large DPs early in training
        self.mab1 = MAB(dim, num_heads, ff_dim=ff_dim, dropout=dropout)
        self.mab2 = MAB(dim, num_heads, ff_dim=ff_dim, dropout=dropout)

    def forward(self, X: Tensor) -> Tensor:
        B = X.size(0)
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X)
        return self.mab2(X, H)

class PMA(nn.Module):
    """
    Pooling by Multihead Attention (PMA).

    Uses a learned set of seed vectors to perform attention-based 
    pooling over a set. Outputs one (or more) set-level representations 
    for the input set.

    Args:
        dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_seeds (int): Number of seed vectors (i.e., output pooled representations). Default: 1.
        ff_dim (Optional[int]): Feedforward hidden dimension.
        dropout (float): Dropout probability.

    Inputs:
        X (Tensor): Input tensor of shape (B, N, dim).

    Returns:
        Tensor: Output tensor of shape (B, num_seeds, dim).
    """
    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, ff_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim, num_heads, ff_dim=ff_dim, dropout=dropout)

    def forward(self, X: Tensor) -> Tensor:
        B = X.size(0)
        S = self.S.expand(B, -1, -1)
        return self.mab(S, X)
