from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal, Optional

ModelName = Literal["vae", "beta_vae", "tc_vae", "dip_vae", "factor_vae"]
ReconLoss = Literal["bce", "mse"]
DIPType = Literal["i", "ii"]

@dataclass
class VAETrainConfig:
    # data
    data_dir: str
    out_dir: str = "runs/vae"
    image_size: int = 64
    channels: int = 3
    batch_size: int = 64
    num_workers: int = 4

    # optimization
    steps: int = 200_000
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 200
    eval_every: int = 0
    save_every: int = 10_000
    seed: int = 0
    device: Optional[str] = None

    # model
    model: ModelName = "vae"
    z_dim: int = 32
    base_channels: int = 32
    recon_loss: ReconLoss = "bce"  # expects images in [0,1] for BCE

    # objective weights
    beta: float = 4.0               # beta-VAE KL multiplier
    tc_beta: float = 6.0            # TC-VAE TC multiplier (gamma in some papers)
    kl_weight: float = 1.0          # multiplier on the whole KL term

    # DIP-VAE
    dip_type: DIPType = "i"
    dip_lambda_diag: float = 10.0
    dip_lambda_offdiag: float = 5.0

    # FactorVAE
    factor_gamma: float = 10.0
    disc_lr: float = 1e-4
    disc_steps: int = 1             # discriminator updates per generator update
