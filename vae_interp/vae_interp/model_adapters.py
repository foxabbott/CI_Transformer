
from __future__ import annotations

import importlib
from typing import Callable, Tuple, Optional, Any, Dict

import torch
from torch import nn, Tensor


def _import_from_string(spec: str) -> Callable:
    if ":" not in spec:
        raise ValueError("Factory must be like 'package.module:callable_name'")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Could not find callable '{fn_name}' in module '{mod_name}'")
    return fn


def load_model_with_factory(factory: str, model_cfg: Dict[str, Any], state_dict: Dict[str, Any]) -> nn.Module:
    fn = _import_from_string(factory)
    model = fn(model_cfg)
    if not isinstance(model, nn.Module):
        raise TypeError("Factory did not return a torch.nn.Module")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[vae_interp] warning: unexpected keys (showing up to 10): {unexpected[:10]}")
    if missing:
        print(f"[vae_interp] warning: missing keys (showing up to 10): {missing[:10]}")
    return model



def resolve_encode_decode(model: nn.Module) -> Tuple[Callable, Callable, int]:
    """
    Returns:
      encode(x, sample: bool) -> z or (mu, logvar, z) depending on sample flag
      decode(z) -> x_recon (in [0,1] if your decoder does sigmoid; otherwise logits)
      z_dim: int
    """

    # --- ConvVAE from your vae package ---
    if hasattr(model, "encode") and hasattr(model, "decode"):
        # Infer z_dim without probing a dummy input
        z_dim = None
        if hasattr(model, "z_dim"):
            z_dim = int(getattr(model, "z_dim"))
        elif hasattr(model, "enc") and hasattr(model.enc, "fc_mu"):
            z_dim = int(model.enc.fc_mu.out_features)
        else:
            raise ValueError("Could not infer z_dim from model. Expected model.z_dim or model.enc.fc_mu.")

        def encode(x: Tensor, use_mean: bool = True):
            mu, logvar = model.encode(x)  # (B,z), (B,z)
            if use_mean:
                return mu
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(z: Tensor):
            out = model.decode(z)
            # your decode might return logits or already-squashed images; we just return it
            return out

        return encode, decode, z_dim

    raise ValueError(f"Don't know how to adapt model type: {type(model)}")
