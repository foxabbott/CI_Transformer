from __future__ import annotations
from typing import Dict, Any
from .config import VAETrainConfig
from .models import ConvVAE, FactorVAEWrapper

def load_model(cfg_dict: Dict[str, Any]):
    cfg = VAETrainConfig(**cfg_dict)
    base = ConvVAE(image_size=cfg.image_size, in_ch=cfg.channels, z_dim=cfg.z_dim, base_ch=cfg.base_channels)
    if cfg.model == "factor_vae":
        return FactorVAEWrapper(base, z_dim=cfg.z_dim)
    return base
