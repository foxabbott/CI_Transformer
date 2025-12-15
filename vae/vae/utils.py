from __future__ import annotations
import os, json, math
from typing import Any, Dict
import torch
from torch import Tensor

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def kl_diag_gaussian(mu: Tensor, logvar: Tensor) -> Tensor:
    # (B,)
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)

def log_density_gaussian(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    # returns same shape as x (elementwise)
    return -0.5 * (math.log(2*math.pi) + logvar + (x - mu).pow(2) / logvar.exp())

def permute_dims(z: Tensor) -> Tensor:
    # independently permute each latent dimension across the batch
    B, D = z.shape
    out = []
    for d in range(D):
        idx = torch.randperm(B, device=z.device)
        out.append(z[idx, d])
    return torch.stack(out, dim=1)

def ema(prev: float | None, x: float, alpha: float = 0.98) -> float:
    return x if prev is None else alpha * prev + (1 - alpha) * x
