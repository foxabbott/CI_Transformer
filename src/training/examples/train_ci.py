"""Train the CI classifier on streaming synthetic data.

Run (from repo root, with editable installs for causal_synth + ci_models + training):
    python examples/train_ci.py --out runs/ci_demo

"""
import argparse
from training import train_ci_model, Curriculum
from ci_models import CISetTransformerConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="runs/ci_demo")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--m_max", type=int, default=5)
    p.add_argument("--n_rows", type=int, default=500)
    args = p.parse_args()

    # curriculum = Curriculum(stages=((0, 5000), (1, 10000), (2, 15000), (5, 10**9)))
    # curriculum = Curriculum(stages=((0, 100), (1, 200), (2, 200), (3, 200), (4, 200), (5, 10**9)))
    curriculum = Curriculum(stages=((5, 100000000),))
    model_cfg = CISetTransformerConfig(
        dim=128, num_heads=4, num_inducing=32, num_isab_layers=2,
        z_aggr="attn", use_pair_encoder=True, standardize_inputs=True
    )

    train_ci_model(
        out_dir=args.out,
        steps=args.steps,
        n_rows=args.n_rows,
        batch_size=args.batch,
        lr=args.lr,
        num_workers=args.workers,
        m_max=args.m_max,
        curriculum=curriculum,
        model_cfg=model_cfg,
        log_every=1,
        eval_every=100000,
    )

if __name__ == "__main__":
    main()
