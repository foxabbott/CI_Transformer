# ci_training

Minimal training utilities to connect:
- `causal_synth` (random continuous SCMs + observational samples)
- `ci_set_transformer` (your invariant CI classifier model)

Implements:

**Step 1**: streaming CI dataset generator
  - Random SCM (random DAG + diverse mechanisms)
  - Sample observational matrix X with N rows
  - Sample random CI queries (i,j,S) where |S|=m
  - Ground-truth label from d-separation: `scm.is_ci_true(i,j,S)`
  - Returns tensors shaped exactly for `CISetTransformer`

**Step 2**: curriculum training
  - Trains first on easy tasks (m=0), then gradually increases to m up to 5.

## Install

Put `ci_training/` in your repo and install editable from its folder:

```bash
pip install -e .
```

You should already have editable installs for:
- causal_synth
- ci_set_transformer

## Quick run

```bash
python examples/train_ci.py --out runs/ci_demo --steps 50000
```

Checkpoints saved to `runs/ci_demo/`.

## Using the dataset directly

```python
from ci_training import CIStreamingDataset, make_dataloader, Curriculum

ds = CIStreamingDataset(n_rows=500, m_max=5, curriculum=Curriculum())
dl = make_dataloader(ds, batch_size=64)

batch = next(iter(dl))
# batch keys: x,y,z,z_mask,label
```
