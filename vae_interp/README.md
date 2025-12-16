# vae_interp

Utilities to visualize what a trained VAE's latent dimensions represent:
- Latent traversals
- Random-pair interventions (swap one latent dimension between two images)

## Install
```bash
pip install -e vae_interp
```

## Latent traversals
```bash
python -m vae_interp.cli.traverse --ckpt runs/vae_pets/final.pt --data data/oxford_pets/train --out runs/vae_pets_interp/traversals --image-size 224
```

## Random-pair interventions
```bash
python -m vae_interp.cli.pair_intervene --ckpt runs/vae_pets/final.pt --data data/oxford_pets/train --out runs/vae_pets_interp/pairs --image-size 224
```

### If your checkpoint only contains a state_dict
Provide a factory that builds the model from the checkpoint's config:
```bash
python -m vae_interp.cli.traverse --ckpt runs/vae_pets/final.pt --factory "my.module:build_model" --data data/oxford_pets/train --out runs/vae_pets_interp/traversals
```

Where `build_model(model_cfg: dict) -> torch.nn.Module`.
