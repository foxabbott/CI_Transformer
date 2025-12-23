
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import math
import torch
from torch import nn, Tensor
from torchvision.utils import make_grid, save_image

from .data import make_loader
from .model_adapters import resolve_encode_decode, load_model_with_factory


@dataclass
class TraversalConfig:
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    num_samples: int = 8
    steps: int = 9
    vmin: float = -3.0
    vmax: float = 3.0
    dims: Optional[Sequence[int]] = None
    use_posterior_mean: bool = True
    clamp: bool = True
    device: Optional[str] = None


@dataclass
class PairInterventionConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    num_pairs: int = 8
    steps: int = 9
    dims: Optional[Sequence[int]] = None
    use_posterior_mean: bool = True
    clamp: bool = True
    device: Optional[str] = None


def _default_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_vae(ckpt_path: str, factory: Optional[str] = None, map_location: str = "cpu") -> nn.Module:
    obj = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj

    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], nn.Module):
        return obj["model"]

    if isinstance(obj, dict):
        state = obj.get("model_state") or obj.get("state_dict") or obj.get("model_state_dict")
        cfg = obj.get("model_cfg") or obj.get("cfg") or obj.get("config") or {}
        if state is None:
            raise ValueError(
                f"Checkpoint at {ckpt_path} is a dict but doesn't contain a recognized state dict key "
                "(expected model_state/state_dict/model_state_dict)."
            )
        if factory is None:
            raise ValueError(
                "Checkpoint contains only a state_dict. Provide --factory like 'my.module:build_model' "
                "so we can instantiate the architecture and load weights."
            )
        return load_model_with_factory(factory, cfg, state)

    raise ValueError(f"Unrecognized checkpoint format: {type(obj)}")


@torch.no_grad()
def run_traversals(model: nn.Module, data_dir: str, out_dir: str, cfg: TraversalConfig = TraversalConfig()) -> None:

    os.makedirs(out_dir, exist_ok=True)
    dev = _default_device(cfg.device)
    model = model.to(dev).eval()

    loader = make_loader(data_dir, cfg.image_size, cfg.batch_size, cfg.num_workers)
    x = next(iter(loader)).to(dev)[: cfg.num_samples]
    B = x.size(0)

    encode, decode, z_dim = resolve_encode_decode(model)
    z0 = encode(x, use_mean=cfg.use_posterior_mean)  # (B,z)

    dims = list(cfg.dims) if cfg.dims is not None else list(range(z_dim))
    values = torch.linspace(cfg.vmin, cfg.vmax, steps=cfg.steps, device=dev)

    # base + recon
    xrec = torch.sigmoid(decode(z0))

    if cfg.clamp:
        xrec = xrec.clamp(0, 1)
    save_image(make_grid(torch.cat([x, xrec], dim=0), nrow=B), os.path.join(out_dir, "base_and_recon.png"))

    avg_dists_by_dim = []
    for i in dims:
        imgs = []
        for v in values:
            z = z0.clone()
            z[:, i] = v
            y = torch.sigmoid(decode(z))
            if cfg.clamp:
                y = y.clamp(0, 1)
            imgs.append(y)

        stack = torch.stack(imgs, dim=1)   # (B, steps, C, H, W)
        stack = stack.flatten(0, 1)        # (B*steps, C, H, W)
        grid = make_grid(stack, nrow=cfg.steps)  # each row is a sample
        save_image(grid, os.path.join(out_dir, f"traversal_dim_{i:03d}.png"))

        # Compute distance between first and last image in the traversal per sample and print average
        # imgs: list of [B, C, H, W] (len = steps), stack: [B, steps, C, H, W]
        imgs_tensor = torch.stack(imgs, dim=1)  # (B, steps, C, H, W)
        first_imgs = imgs_tensor[:, 0]  # (B, C, H, W)
        last_imgs = imgs_tensor[:, -1]  # (B, C, H, W)
        # Compute per-image L2 distance
        dists = ((first_imgs - last_imgs).pow(2).view(B, -1).sum(dim=1).sqrt())  # (B,)
        avg_dist = dists.mean().item()
        avg_dists_by_dim.append((i, avg_dist))
        print(f"Average L2 distance between first and last image for dim {i}: {avg_dist:.6f}")

    # After traversing all dims, compute top 10% of dimensions by average L2 distance and print them in order
    if len(avg_dists_by_dim) > 0:
        n_top = max(1, math.ceil(len(avg_dists_by_dim) * 0.10))
        sorted_dims = sorted(avg_dists_by_dim, key=lambda x: x[1], reverse=True)
        top_dims = sorted_dims[:n_top]
        print("\nTop {:d}% of dimensions by average L2 distance:".format(int(100 * n_top / len(avg_dists_by_dim))))
        for idx, dist in top_dims:
            print(f"  dim {idx}: avg L2 distance = {dist:.6f}")



@torch.no_grad()
def run_pair_interventions(model: nn.Module, data_dir: str, out_dir: str, cfg: PairInterventionConfig = PairInterventionConfig()) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dev = _default_device(cfg.device)
    model = model.to(dev).eval()

    loader = make_loader(data_dir, cfg.image_size, cfg.batch_size, cfg.num_workers)
    x = next(iter(loader)).to(dev)
    if x.size(0) < 2 * cfg.num_pairs:
        raise ValueError(f"Need at least {2*cfg.num_pairs} images in one batch. Increase --batch or reduce --num-pairs.")

    xa = x[: cfg.num_pairs]
    xb = x[cfg.num_pairs: 2*cfg.num_pairs]

    encode, decode, z_dim = resolve_encode_decode(model)
    za = encode(xa, use_mean=cfg.use_posterior_mean)
    zb = encode(xb, use_mean=cfg.use_posterior_mean)

    dims = list(cfg.dims) if cfg.dims is not None else list(range(min(cfg.steps, z_dim)))

    reca = torch.sigmoid(decode(za))
    recb = torch.sigmoid(decode(zb))
    if cfg.clamp:
        reca = reca.clamp(0, 1)
        recb = recb.clamp(0, 1)

    for p in range(cfg.num_pairs):
        row = [xa[p:p+1], xb[p:p+1], reca[p:p+1], recb[p:p+1]]
        for d in dims:
            z = za[p:p+1].clone()
            z[:, d] = zb[p:p+1, d]
            y = torch.sigmoid(decode(z))
            if cfg.clamp:
                y = y.clamp(0, 1)
            row.append(y)

        grid = make_grid(torch.cat(row, dim=0), nrow=len(row))
        save_image(grid, os.path.join(out_dir, f"pair_{p:03d}.png"))
