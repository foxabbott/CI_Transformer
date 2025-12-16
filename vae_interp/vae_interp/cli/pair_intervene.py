
from __future__ import annotations
import argparse
from vae_interp.api import load_vae, run_pair_interventions, PairInterventionConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--factory", type=str, default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--dims", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    dims = None
    if args.dims:
        dims = [int(x) for x in args.dims.split(",") if x.strip()]

    model = load_vae(args.ckpt, factory=args.factory)
    cfg = PairInterventionConfig(
        image_size=args.image_size,
        batch_size=args.batch,
        num_workers=args.workers,
        num_pairs=args.num_pairs,
        steps=args.steps,
        dims=dims,
        device=args.device,
    )
    run_pair_interventions(model, args.data, args.out, cfg)

if __name__ == "__main__":
    main()
