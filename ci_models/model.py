from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn, Tensor

from .layers import ISAB, PMA
from .utils import standardize_per_instance

@dataclass
class CISetTransformerConfig:
    """Hyperparameters controlling CI set transformer architecture and behavior."""
    dim: int = 128
    num_heads: int = 4
    num_inducing: int = 32
    num_isab_layers: int = 2
    dropout: float = 0.0
    z_aggr: str = "attn"          # "mean" or "attn"
    use_pair_encoder: bool = True 
    standardize_inputs: bool = True 
    use_cross_encoder: bool = True 

class ColumnEncoder(nn.Module):
    """Encodes a single column (set of rows) into a fixed-length representation."""
    def __init__(self, cfg: CISetTransformerConfig):
        super().__init__()
        d = cfg.dim
        self.row_mlp = nn.Sequential(
            nn.Linear(1, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.blocks = nn.ModuleList([
            ISAB(d, cfg.num_heads, num_inducing=cfg.num_inducing, dropout=cfg.dropout)
            for _ in range(cfg.num_isab_layers)
        ])
        self.pma = PMA(d, cfg.num_heads, num_seeds=1, dropout=cfg.dropout)
        self.out_ln = nn.LayerNorm(d)

    def forward(self, v: Tensor) -> Tensor:
        h = self.row_mlp(v.unsqueeze(-1))  # (B,N,d)
        for blk in self.blocks:
            h = blk(h)
        pooled = self.pma(h).squeeze(1)    # (B,d)
        return self.out_ln(pooled)

class ZAggregator(nn.Module):
    """Aggregates encoded conditioning columns using attention or mean pooling."""
    def __init__(self, cfg: CISetTransformerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        if cfg.z_aggr == "attn":
            self.score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        elif cfg.z_aggr == "mean":
            self.score = None
        else:
            raise ValueError("z_aggr must be 'mean' or 'attn'.")

    def forward(self, cZ: Tensor, z_mask: Optional[Tensor]) -> Tensor:
        # cZ: (B,M,d)
        B, M, d = cZ.shape
        if M == 0:
            return torch.zeros((B, d), device=cZ.device, dtype=cZ.dtype)

        if self.cfg.z_aggr == "mean":
            if z_mask is None:
                return cZ.mean(dim=1)
            m = z_mask.to(dtype=cZ.dtype).unsqueeze(-1)
            return (cZ * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)

        logits = self.score(cZ).squeeze(-1)  # (B,M)
        if z_mask is not None:
            logits = logits.masked_fill(~z_mask.bool(), -1e9)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (cZ * w).sum(dim=1)

class PairEncoder(nn.Module):
    """Encodes pairwise relationship between X and Y columns with permutation invariance."""
    def __init__(self, cfg: CISetTransformerConfig):
        super().__init__()
        d = cfg.dim
        self.row_mlp = nn.Sequential(
            nn.Linear(3, d),  # [x+y, |x-y|, x*y]
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.blocks = nn.ModuleList([
            ISAB(d, cfg.num_heads, num_inducing=cfg.num_inducing, dropout=cfg.dropout)
            for _ in range(cfg.num_isab_layers)
        ])
        self.pma = PMA(d, cfg.num_heads, num_seeds=1, dropout=cfg.dropout)
        self.out_ln = nn.LayerNorm(d)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        feats = torch.stack([x + y, (x - y).abs(), x * y], dim=-1)  # (B,N,3)
        h = self.row_mlp(feats)
        for blk in self.blocks:
            h = blk(h)
        pooled = self.pma(h).squeeze(1)
        return self.out_ln(pooled)


class CrossPairEncoder(nn.Module):
    """
    Encodes relationship between target column t (x or y) and one conditioning column z.
    Row-permutation invariant (ISAB+PMA). Not symmetric in (t,z) (doesn't need to be).
    """
    def __init__(self, cfg: CISetTransformerConfig):
        super().__init__()
        d = cfg.dim
        self.row_mlp = nn.Sequential(
            nn.Linear(4, d),  # [t, z, t*z, |t-z|]
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.blocks = nn.ModuleList([
            ISAB(d, cfg.num_heads, num_inducing=cfg.num_inducing, dropout=cfg.dropout)
            for _ in range(cfg.num_isab_layers)
        ])
        self.pma = PMA(d, cfg.num_heads, num_seeds=1, dropout=cfg.dropout)
        self.out_ln = nn.LayerNorm(d)

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        # t,z: (B,N)
        feats = torch.stack([t, z, t * z, (t - z).abs()], dim=-1)  # (B,N,4)
        h = self.row_mlp(feats)  # (B,N,d)
        for blk in self.blocks:
            h = blk(h)
        pooled = self.pma(h).squeeze(1)  # (B,d)
        return self.out_ln(pooled)


class CISetTransformer(nn.Module):
    """Full model for conditional independence classification on set-valued columns."""
    def __init__(self, cfg: CISetTransformerConfig = CISetTransformerConfig()):
        super().__init__()
        self.cfg = cfg
        self.col_enc = ColumnEncoder(cfg)
        self.z_aggr = ZAggregator(cfg)
        self.use_cross = cfg.use_cross_encoder
        if self.use_cross:
            # share weights between XZ and YZ to preserve X<->Y symmetry
            self.tz_enc = CrossPairEncoder(cfg)
            self.tz_aggr = ZAggregator(cfg)

        self.use_pair = cfg.use_pair_encoder
        if self.use_pair:
            self.pair_enc = PairEncoder(cfg)

        d = cfg.dim
        head_in = 3*d + d + 1  # add/abs/mul[sym(X), sym(Y)] + Z + m
        if self.use_cross:
            head_in += 3*d      # sym(XZ,YZ): add/abs/mul
        if self.use_pair:
            head_in += d        # sym(XY)

        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor, y: Tensor, z: Optional[Tensor] = None, z_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Standardize inputs per instance if configured
        if self.cfg.standardize_inputs:
            x = standardize_per_instance(x)
            y = standardize_per_instance(y)
            if z is not None and z.numel() > 0:
                z = standardize_per_instance(z)

        B, N = x.shape
        # Encode x and y columns as set embeddings
        cx = self.col_enc(x)
        cy = self.col_enc(y)

        if z is None:
            # No conditioning set: zero out all Z-related features
            cZagg = x.new_zeros((B, self.cfg.dim))
            m = x.new_zeros((B, 1))
            cXZagg = x.new_zeros((B, self.cfg.dim))
            cYZagg = x.new_zeros((B, self.cfg.dim))

        else:
            Bz, M, Nz = z.shape
            assert Bz == B and Nz == N
            if M == 0:
                # Empty conditioning set: zero out all Z-related features
                cZagg = x.new_zeros((B, self.cfg.dim))
                m = x.new_zeros((B, 1))
                cXZagg = x.new_zeros((B, self.cfg.dim))
                cYZagg = x.new_zeros((B, self.cfg.dim))
            else:
                # Encode each Z column, then aggregate across Z set
                cz = self.col_enc(z.reshape(B*M, N)).reshape(B, M, -1)
                cZagg = self.z_aggr(cz, z_mask)
                
                if self.use_cross:
                    # Encode cross-pair relationships: (X, Z_i) and (Y, Z_i) for each Z_i
                    z_flat = z.reshape(B * M, N)  # (B*M, N)
                    x_rep = x.unsqueeze(1).expand(B, M, N).reshape(B * M, N)
                    y_rep = y.unsqueeze(1).expand(B, M, N).reshape(B * M, N)

                    exz = self.tz_enc(x_rep, z_flat).reshape(B, M, -1)  # (B,M,d)
                    eyz = self.tz_enc(y_rep, z_flat).reshape(B, M, -1)  # (B,M,d)

                    # Aggregate cross-pair encodings across Z set
                    cXZagg = self.tz_aggr(exz, z_mask)  # (B,d)
                    cYZagg = self.tz_aggr(eyz, z_mask)  # (B,d)
                else:
                    cXZagg = x.new_zeros((B, self.cfg.dim))
                    cYZagg = x.new_zeros((B, self.cfg.dim))

                if z_mask is None:
                    m = x.new_full((B, 1), float(M))  # how many conditioning columns in each batch?
                else:
                    mask = z_mask.to(dtype=x.dtype).unsqueeze(-1)       # (B,M,1)
                    m = mask.sum(dim=1)                                  # (B,1)

        # Symmetric features from X and Y column encodings
        s_add = cx + cy
        s_abs = (cx - cy).abs()
        s_mul = cx * cy

        # Build feature vector: XY symmetric features, Z aggregate, cross-pairs (optional), pair encoding (optional), count
        feats = [s_add, s_abs, s_mul, cZagg]
        if self.use_cross:
            # Symmetric features from cross-pair encodings
            xz_add = cXZagg + cYZagg
            xz_abs = (cXZagg - cYZagg).abs()
            xz_mul = cXZagg * cYZagg
            feats.extend([xz_add, xz_abs, xz_mul])
        if self.use_pair:
            feats.append(self.pair_enc(x, y))
        feats.append(m.to(dtype=x.dtype))

        # Final prediction head
        u = torch.cat(feats, dim=1)
        logits = self.head(u).squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs


