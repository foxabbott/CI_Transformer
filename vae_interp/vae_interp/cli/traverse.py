
from __future__ import annotations
import argparse
from vae_interp.api import load_vae, run_traversals, TraversalConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--factory", type=str, default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--vmin", type=float, default=-3.0)
    p.add_argument("--vmax", type=float, default=3.0)
    p.add_argument("--dims", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    dims = None
    if args.dims:
        dims = [int(x) for x in args.dims.split(",") if x.strip()]

    model = load_vae(args.ckpt, factory=args.factory)
    cfg = TraversalConfig(
        image_size=args.image_size,
        batch_size=args.batch,
        num_workers=args.workers,
        num_samples=args.num_samples,
        steps=args.steps,
        vmin=args.vmin,
        vmax=args.vmax,
        dims=dims,
        device=args.device,
    )
    run_traversals(model, args.data, args.out, cfg)

if __name__ == "__main__":
    main()
