# vae

Minimal PyTorch implementations of:

- **VAE** (standard ELBO)
- **beta-VAE** (beta-weighted KL)
- **TC-VAE** (Beta-TC-VAE style: decomposes KL into MI + TC + DW-KL)
- **DIP-VAE** (regularizes covariance of aggregated posterior)
- **FactorVAE** (adversarial total-correlation estimator with a discriminator)

This package is intended as an add-on to your existing repo. It assumes you already have PyTorch installed.

## Install (editable)

From your repo root (after copying this folder in):

```bash
pip install -e vae
```

## Run training

```bash
python -m vae.train --data /path/to/images --out runs/vae_demo --model beta_vae --image-size 64 --steps 50000
```

The `--data` directory should be ImageFolder-compatible (subfolders allowed). Labels are ignored.

Artifacts:
- `final.pt` (model state + config)
- periodic `ckpt_*.pt`
- `config.json`

## Using the trained encoder

```python
import torch
from vae import load_model

ckpt = torch.load("runs/vae_demo/final.pt", map_location="cpu")
model = load_model(ckpt["config"])
model.load_state_dict(ckpt["model_state"])
model.eval()

with torch.no_grad():
    out = model(images)   # images: (B,C,H,W)
    z = out.z             # (B, z_dim)
```
