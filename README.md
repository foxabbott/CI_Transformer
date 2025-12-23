# Causal Data Gen: Synthetic Causal Data, CI Benchmarks, and Invariant Classifiers

## Project Overview

**Causal Data Gen** is a modular, research-oriented Python toolkit for:
- **Generating diverse, continuous synthetic datasets** using randomly generated latent causal DAGs (SCMs)
- **Training and benchmarking Conditional Independence (CI) classifiers**
- Supporting scalable workflow for causal structure discovery, CI-test benchmarking, and representation learning

This repo unifies three core components:
- `causal_data_gen`: Synthetic SCM + dataset generator with diverse, nonparametric mechanisms and noises
- `ci_models`: Neural classifier (PyTorch) for CI testing, with symmetry- and permutation-invariance
- `training`: Streaming dataset/utilities to generate on-the-fly CI query data, and curriculum-based CI classifier training

---

## Features
- **Truly random DAGs (acyclic)**, up to 15 variables (configurable)
- **All variables continuous**
- **Mechanism diversity**: linear, polynomial, trigonometric, rational, Fourier, random MLP, heteroskedastic & multiplicative noise
- **Noise diversity**: Laplace, Student-t, mixture of Gaussians, uniform, and many more (not just Gaussian)
- **Extensible**: Add your own mechanism/noise family simply
- **No global normality/additivity assumption**: Each node/dataset randomized independently
- **Ground-truth CI queries**: d-separation in latent DAG, not from data!

---

## Installation

Typical install (from repo root, after cloning):

```bash
pip install -e .
```

You will need Python>=3.9, numpy, pytorch, and other common deps.

---

## Quickstart

### 1. Generate an SCM & Dataset

```python
from causal_data_gen import CausalSCM, RandomDAGConfig

cfg = RandomDAGConfig(d=12, edge_prob=0.22, max_parents=4, ensure_connected=True)
scm = CausalSCM.random(cfg, seed=0)

X = scm.sample(n=5000, seed=1)    # shape: (5000, 12)
A = scm.adjacency()               # adjacency matrix (ground-truth), A[i, j]=1 iff i → j
meta = scm.metadata()             # parent lists, mechanisms, noises, etc

# Query ground-truth CI via d-separation
print(scm.is_ci_true(i=3, j=7, S=[0,2]))
```

### 2. Generate and Save Ground-Truth Datasets

```python
from causal_data_gen import generate_dataset, RandomDAGConfig

ds, scm = generate_dataset(n=20000, cfg=RandomDAGConfig(d=10, edge_prob=0.25, max_parents=3), seed=123)
ds.save_npz("toy_scm_dataset.npz")
```
The `.npz` contains:
- `X`: data matrix (n, d)
- `adj`: adjacency matrix (d, d)
- `weights`: edge weights
- `order`: topological order of graph construction
- `parents`: parents per node
- `mechanisms`: JSON-like dicts for each node's mechanism
- `noises`: noise family + parameters for each node

### 3. Train a CI Classifier (Full Pipeline)

From root (after editable install of all 3 components):

```bash
python training/examples/train_ci.py --out runs/ci_demo --steps 50000
```
Saves checkpoints/logs in `runs/ci_demo/`.

#### Using Streaming Dataset Directly

```python
from training import CIStreamingDataset, make_dataloader, Curriculum
ds = CIStreamingDataset(n_rows=500, m_max=5, curriculum=Curriculum())
dl = make_dataloader(ds, batch_size=64)
batch = next(iter(dl))
# batch contains: x, y, z, z_mask, label
```

---

## How It Works

- On-the-fly data: Training is done with a streaming dataloader, which samples a new random SCM and CI queries each batch—no overfitting to static training sets.
- Curriculum: By default, CI classifier training progresses from easy (small S) to hard (large S).
- Fully pluggable SCM generator: To change the types of mechanisms/noises, edit/add to `causal_data_gen/mechanisms.py`.

---

## Components

- **causal_data_gen/**: Core data generation, SCM, random DAGs, mechanisms, Dataset saving/loading
- **ci_models/**: PyTorch CI classifier, implements a permutation-invariant deep net that predicts CI between pairs (i, j | S)
- **training/**: Streaming data utilities & curriculum training code for efficient, fair, and scalable experimentation
- **examples/**: Example scripts for dataset generation and running basic pipelines

---

## Extending Mechanisms/Noises
Add families in `causal_data_gen/mechanisms.py`. A mechanism is a factory with signature:
```python
def my_mechanism(rng, k):
    def f(P, eps): ...  # P: parent values (n, k), eps: noise (n,)
    return f, meta_dict
```
Add to `MECHANISM_FACTORIES` in that file.

---

## Notes & Tips
- **Observational data only** (i.i.d.). No interventions/environments yet.
- Variables are lightly clipped/standardized for stability.
- `scm.is_ci_true(i, j, S)` returns ground truth according to the latent DAG (d-sep).
- See example scripts in `/examples`.

---

## Citing & Attributions
This work combines original code with inspiration from the invariant causal learning and synthetic SCM literature. Please cite relevant papers if reusing core ideas!

