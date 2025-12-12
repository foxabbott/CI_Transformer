# causal_synth

A lightweight Python package for generating **continuous** synthetic datasets from a randomly generated
**latent causal DAG** (an SCM). It is designed for tasks like training CI classifiers, causal discovery
benchmarking, and representation learning.

Key properties:

- **DAG is random** (acyclic by construction).
- **Up to 15 variables** (hard-enforced).
- **All variables continuous**.
- **Diverse mechanisms**: linear, polynomial, trigonometric, GP-like random Fourier features, random MLP,
  heteroskedastic noise, multiplicative noise, rational functions, etc.
- **Noise distributions are diverse** and not restricted to Gaussian: Laplace, Student-t, logistic,
  mixture of Gaussians, shifted gamma, uniform, etc.
- No single assumption (normality, additivity, etc.) is required across all variables: mechanisms and
  noises vary per-node and per-dataset.

## Install / import

This is a simple folder package. In a project, you can either:

- Copy `causal_synth/` into your repo and `import causal_synth`, or
- Add the parent directory to `PYTHONPATH`.

## Quickstart

```python
from causal_synth import CausalSCM, RandomDAGConfig

cfg = RandomDAGConfig(d=12, edge_prob=0.22, max_parents=4, ensure_connected=True)
scm = CausalSCM.random(cfg, seed=0)

X = scm.sample(n=5000, seed=1)    # shape: (5000, 12)
A = scm.adjacency()               # ground-truth adjacency, A[i,j]=1 means i -> j
meta = scm.metadata()             # includes parents list, mechanism/noise descriptions

# ground-truth CI query (d-separation on the DAG)
print(scm.is_ci_true(i=3, j=7, S=[0, 2]))
```

## Generating datasets + saving

```python
from causal_synth import generate_dataset, RandomDAGConfig

ds, scm = generate_dataset(
    n=20000,
    cfg=RandomDAGConfig(d=10, edge_prob=0.25, max_parents=3),
    seed=123,
    standardize_vars=True,
)

ds.save_npz("toy_scm_dataset.npz")
```

The `.npz` contains:
- `X`: data matrix (n, d)
- `adj`: adjacency matrix (d, d)
- `weights`: edge weights (not used by all mechanisms; provided for reference)
- `order`: topological order used for graph construction
- `parents`: parent list per node
- `mechanisms`: a list of JSON-like dicts describing each node's mechanism family + parameters
- `noises`: a list of noise specs (distribution name + parameters)

## Extending mechanisms / noises

Mechanisms live in `causal_synth/mechanisms.py`. Each mechanism is a factory:

```python
def mech_name(rng, k) -> (f, meta):
    def f(P, eps): ...
    return f, meta
```

- `P` is a matrix of parent values with shape `(n, k)` (or `(n, 1)` for roots).
- `eps` is a vector of exogenous noise with shape `(n,)`.
- `f(P, eps)` returns a vector `(n,)` of generated values.

Add your mechanism factory to `MECHANISM_FACTORIES`.

## Notes / caveats

- The package generates **observational** samples (i.i.d.). It does not currently generate
  interventions or multiple environments.
- Variables are lightly clipped and optionally standardized to keep numerical ranges reasonable.
- The provided `is_ci_true()` answers CI queries **in the latent DAG** (d-separation), which matches
  CI relations under standard SCM semantics (Markov + faithfulness assumptions).

## Suggested workflow for CI-classifier training

1. Sample many SCMs with different seeds/configs.
2. For each SCM, sample a dataset `X`.
3. Create CI labels by querying `scm.is_ci_true(i,j,S)` for your chosen set-family.
4. Train a model `T_psi` on datasets + labels.

Example label generation snippet:

```python
import itertools
pairs = [(i,j) for i in range(d) for j in range(i+1, d)]
S_family = lambda i,j: [[], *[[k] for k in range(d) if k not in (i,j)]]

labels = []
for (i,j) in pairs:
    y = any(scm.is_ci_true(i,j,S) for S in S_family(i,j))
    labels.append(int(y))
```

