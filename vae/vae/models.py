from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn, Tensor

@dataclass
class VAEOutput:
    x_recon: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor

class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, z_dim: int, image_size: int):
        super().__init__()
        # 4 downsamples: /2 each time
        ch = base_ch
        layers = []
        cur = in_ch
        for _ in range(4):
            layers += [nn.Conv2d(cur, ch, 4, 2, 1), nn.GELU()]
            cur = ch
            ch *= 2
        self.net = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, image_size, image_size)
            h = self.net(dummy)
            self.out_shape = h.shape[1:]          # (C,H,W)
            self.flat_dim = h.numel()

        self.fc_mu = nn.Linear(self.flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, z_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.net(x).reshape(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

class ConvDecoder(nn.Module):
    def __init__(self, out_ch: int, base_ch: int, z_dim: int, out_shape: Tuple[int,int,int], flat_dim: int):
        super().__init__()
        self.out_shape = out_shape
        self.flat_dim = flat_dim
        self.fc = nn.Sequential(nn.Linear(z_dim, flat_dim), nn.GELU())

        c, _, _ = out_shape
        layers = []
        cur = c
        # 3 upsample blocks, then final
        for _ in range(3):
            nxt = max(out_ch, cur // 2)
            layers += [nn.ConvTranspose2d(cur, nxt, 4, 2, 1), nn.GELU()]
            cur = nxt
        layers += [nn.ConvTranspose2d(cur, out_ch, 4, 2, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z)
        B = z.size(0)
        c, hh, ww = self.out_shape
        h = h.view(B, c, hh, ww)
        return self.net(h)

class ConvVAE(nn.Module):
    def __init__(self, image_size: int = 64, in_ch: int = 3, z_dim: int = 32, base_ch: int = 32):
        super().__init__()
        self.enc = ConvEncoder(in_ch, base_ch, z_dim, image_size)
        self.dec = ConvDecoder(in_ch, base_ch, z_dim, self.enc.out_shape, self.enc.flat_dim)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.enc(x)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        return self.dec(z)

    def forward(self, x: Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return VAEOutput(x_recon=x_recon, mu=mu, logvar=logvar, z=z)

class FactorDiscriminator(nn.Module):
    def __init__(self, z_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 2),
        )
    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)

class FactorVAEWrapper(nn.Module):
    """Wraps a base VAE and adds a discriminator used only for training FactorVAE."""
    def __init__(self, vae: ConvVAE, z_dim: int):
        super().__init__()
        self.vae = vae
        self.disc = FactorDiscriminator(z_dim=z_dim)

    def forward(self, x: Tensor) -> VAEOutput:
        return self.vae(x)
