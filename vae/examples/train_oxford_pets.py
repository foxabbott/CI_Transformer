import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="ImageFolder root, e.g. data/pets_imgfolder")
    p.add_argument("--out", type=str, default="runs/vae_pets")
    p.add_argument("--model", type=str, default="vae",
                   choices=["vae", "beta_vae", "tc_vae", "dip_vae", "factor_vae"])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--steps", type=int, default=3_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--recon-loss", type=str, default="mse", choices=["mse", "bce"])
    args = p.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(f"--data path does not exist: {data}")

    cmd = [
        sys.executable, "-m", "vae.train",
        "--data", str(data),
        "--out", args.out,
        "--model", args.model,
        "--image-size", str(args.image_size),
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--z-dim", str(args.z_dim),
        "--workers", str(args.workers),
        "--recon-loss", args.recon_loss,
        "--channels", "3",
    ]

    print("Running:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
