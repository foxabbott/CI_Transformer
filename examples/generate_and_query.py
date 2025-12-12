from causal_synth import CausalSCM, RandomDAGConfig, generate_dataset

def main():
    cfg = RandomDAGConfig(d=8, edge_prob=0.25, max_parents=3, ensure_connected=True)
    ds, scm = generate_dataset(n=5000, cfg=cfg, seed=0)

    print("X shape:", ds.X.shape)
    print("Adjacency (i->j):\n", ds.meta.adj)

    i, j, S = 2, 6, [0, 3]
    print(f"CI truth: z_{i} ⟂ z_{j} | z_{S} ? ", scm.is_ci_true(i, j, S))

    ds.save_npz("example_dataset.npz")
    print("Saved example_dataset.npz")

if __name__ == "__main__":
    main()
