from __future__ import annotations
import argparse
from .config import VAETrainConfig
from .trainer import train_vae

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=str, help="ImageFolder-style directory of images")
    p.add_argument("--out", default="runs/vae", type=str)
    p.add_argument("--model", default="vae", choices=["vae","beta_vae","tc_vae","dip_vae","factor_vae"])
    p.add_argument("--image-size", default=64, type=int)
    p.add_argument("--channels", default=3, type=int)
    p.add_argument("--z-dim", default=32, type=int)
    p.add_argument("--base-channels", default=32, type=int)
    p.add_argument("--batch", default=64, type=int)
    p.add_argument("--steps", default=200000, type=int)
    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--recon-loss", default="bce", choices=["bce","mse","l1"])

    # objective knobs
    p.add_argument("--beta", default=4.0, type=float)
    p.add_argument("--tc-beta", default=6.0, type=float)
    p.add_argument("--kl-weight", default=1.0, type=float)
    p.add_argument("--dip-type", default="i", choices=["i","ii"])
    p.add_argument("--dip-lam-diag", default=10.0, type=float)
    p.add_argument("--dip-lam-offdiag", default=5.0, type=float)
    p.add_argument("--factor-gamma", default=10.0, type=float)
    p.add_argument("--disc-lr", default=1e-4, type=float)

    p.add_argument("--log-every", default=1, type=int)
    p.add_argument("--save-every", default=10000, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--device", default=None, type=str)
    args = p.parse_args()

    cfg = VAETrainConfig(
        data_dir=args.data,
        out_dir=args.out,
        model=args.model,
        image_size=args.image_size,
        channels=args.channels,
        z_dim=args.z_dim,
        base_channels=args.base_channels,
        batch_size=args.batch,
        steps=args.steps,
        lr=args.lr,
        num_workers=args.workers,
        recon_loss=args.recon_loss,
        beta=args.beta,
        tc_beta=args.tc_beta,
        kl_weight=args.kl_weight,
        dip_type=args.dip_type,
        dip_lambda_diag=args.dip_lam_diag,
        dip_lambda_offdiag=args.dip_lam_offdiag,
        factor_gamma=args.factor_gamma,
        disc_lr=args.disc_lr,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
    )
    train_vae(cfg)

if __name__ == "__main__":
    main()
