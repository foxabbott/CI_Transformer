# ci_transformer: Causal data generation + CI classifiers

Unified Python toolkit for:
- Generating diverse continuous SCMs and datasets
- Training and benchmarking conditional-independence (CI) classifiers
- Streaming, curriculum-based CI training workflows

---

## Install
Recommended (avoids PEP 668 issues):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```
Requires Python>=3.9 plus numpy, torch (installed via setup).

---

## Quickstart

### Generate an SCM and data
```python
from ci_transformer.causal_data_gen import CausalSCM, RandomDAGConfig

cfg = RandomDAGConfig(d=12, edge_prob=0.22, max_parents=4, ensure_connected=True)
scm = CausalSCM.random(cfg, seed=0)

X = scm.sample(n=5000, seed=1)
A = scm.adjacency()
print(scm.is_ci_true(i=3, j=7, S=[0, 2]))
```

### Save a dataset
```python
from ci_transformer.causal_data_gen import generate_dataset, RandomDAGConfig

ds, scm = generate_dataset(n=20000, cfg=RandomDAGConfig(d=10, edge_prob=0.25, max_parents=3), seed=123)
ds.save_npz("toy_scm_dataset.npz")
```

### Train a CI classifier (script)
```bash
python scripts/train_ci.py --out runs/ci_demo --steps 50000
```
Checkpoints/logs land in `runs/ci_demo/`.

### Use the streaming dataloader directly
```python
from ci_transformer.training import CIStreamingDataset, make_dataloader, Curriculum

ds = CIStreamingDataset(n_rows=500, m_max=5, curriculum=Curriculum())
dl = make_dataloader(ds, batch_size=64)
batch = next(iter(dl))  # contains x, y, z, z_mask, label
```

---

## Package layout
- `ci_transformer.causal_data_gen`: SCMs, DAG sampling, mechanisms, dataset I/O
- `ci_transformer.ci_models`: PyTorch CI classifier + configs
- `ci_transformer.training`: Streaming CI dataset + training loop utilities
- `scripts/train_ci.py`: CLI entry for CI training
- `tests/`: unit tests (e.g., CI model invariances)

---

## Testing
```bash
pytest
# or narrow:
pytest tests/ci_transformer/ci_models/test_invariants.py
```

---

## Notes
- Observational data only (no interventions/environments yet).
- Mechanism/noise families are pluggable (see `ci_transformer/causal_data_gen/mechanisms.py`).
- Imports should all be via `ci_transformer.*` after editable install.

