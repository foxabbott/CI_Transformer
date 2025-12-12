from causal_synth import CausalSCM, RandomDAGConfig

cfg = RandomDAGConfig(d=8, edge_prob=0.5, max_parents=7)
scm = CausalSCM.random(cfg, seed=0)

X = scm.sample(n=1000)
print(X.shape)                 # (1000, 8)
print(scm.adjacency())
print(scm.is_ci_true(0, 1, [2]))
