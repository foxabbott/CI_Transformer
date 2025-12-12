from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, List, Dict, Any
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from causal_synth import CausalSCM, RandomDAGConfig

@dataclass
class Curriculum:
    """Simple curriculum over conditioning-set size m.

    Stages are (max_m, steps). During a stage, m is sampled uniformly from [0, max_m].
    After `steps`, we move to the next stage.
    """
    stages: Tuple[Tuple[int, int], ...] = ((0, 20_000), (1, 40_000), (2, 60_000), (5, 200_000))

    def max_m_at_step(self, step: int) -> int:
        s = 0
        for max_m, steps in self.stages:
            s += steps
            if step < s:
                return max_m
        return self.stages[-1][0]

    def sample_m(self, rng: np.random.Generator, step: int) -> int:
        max_m = self.max_m_at_step(step)
        return int(rng.integers(0, max_m + 1))

def _choose_pair(rng: np.random.Generator, d: int) -> Tuple[int, int]:
    i = int(rng.integers(0, d))
    j = int(rng.integers(0, d - 1))
    if j >= i:
        j += 1
    return i, j

def _choose_S(rng: np.random.Generator, d: int, i: int, j: int, m: int) -> List[int]:
    if m <= 0:
        return []
    candidates = [k for k in range(d) if k not in (i, j)]
    if len(candidates) == 0:
        return []
    m = min(m, len(candidates))
    S = rng.choice(candidates, size=m, replace=False).astype(int).tolist()
    return S

class CIStreamingDataset(IterableDataset):
    """Streaming generator of CI instances.

    Each yielded item is a dict:
      - x: (N,) float32
      - y: (N,) float32
      - z: (Mmax,N) float32 (padded with zeros)
      - z_mask: (Mmax,) bool
      - label: float32 scalar (1.0 means independent, 0.0 means dependent)
      - meta: optional small dict (i,j,S,d,m) useful for debugging

    Notes:
      - Resamples a new SCM every `scm_refresh` examples.
      - Resamples observational data matrix X every `data_refresh` examples (from same SCM),
        which gives more variation without changing the ground-truth graph.
    """
    def __init__(
        self,
        n_rows: int = 500,
        d_min: int = 5,
        d_max: int = 15,
        edge_prob: float = 0.25,
        max_parents: int = 4,
        ensure_connected: bool = True,
        m_max: int = 5,
        curriculum: Optional[Curriculum] = None,
        seed: int = 0,
        scm_refresh: int = 200,
        data_refresh: int = 20,
        standardize_vars: bool = True,
        include_meta: bool = False,
        target_p_indep: Optional[float] = 0.5,
        max_tries: int = 1000,
    ):
        super().__init__()
        assert 2 <= d_min <= d_max <= 15
        assert 0 <= m_max <= 15
        self.n_rows = n_rows
        self.d_min = d_min
        self.d_max = d_max
        self.edge_prob = edge_prob
        self.max_parents = max_parents
        self.ensure_connected = ensure_connected
        self.m_max = m_max
        self.curriculum = curriculum or Curriculum()
        self.seed = seed
        self.scm_refresh = scm_refresh
        self.data_refresh = data_refresh
        self.standardize_vars = standardize_vars
        self.include_meta = include_meta
        self.target_p_indep = target_p_indep
        self.max_tries = max_tries

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        # Make RNG distinct per worker
        rng = np.random.default_rng(self.seed + 10_000 * worker_id)
        global_step = 0

        scm = None
        X = None
        d = None

        while True:
            if scm is None or (global_step % self.scm_refresh == 0):
                d = int(rng.integers(self.d_min, self.d_max + 1))
                cfg = RandomDAGConfig(
                    d=d,
                    edge_prob=self.edge_prob,
                    max_parents=min(self.max_parents, d - 1),
                    ensure_connected=self.ensure_connected,
                )
                scm = CausalSCM.random(cfg, seed=int(rng.integers(0, 2**31 - 1)))
                X = None

            if X is None or (global_step % self.data_refresh == 0):
                X = scm.sample(
                    n=self.n_rows,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    standardize_vars=self.standardize_vars
                )  # (N,d) numpy float32

            # Decide desired label for this example (only if balancing is enabled)
            want_indep = None
            if self.target_p_indep is not None:
                want_indep = bool(rng.random() < float(self.target_p_indep))

            # Rejection sample queries until label matches desired label (or give up)
            for _try in range(self.max_tries):
                # Choose query
                i, j = _choose_pair(rng, d)
                m = self.curriculum.sample_m(rng, global_step)
                m = min(m, self.m_max)
                S = _choose_S(rng, d, i, j, m)

                # Ground-truth CI label via d-separation
                is_indep = scm.is_ci_true(i, j, S)
                if want_indep is None or (is_indep == want_indep):
                    break
            else:
                # If we couldn't find the desired label quickly, just accept the last sample
                pass

            # Extract columns
            x = torch.from_numpy(X[:, i].copy()).float()    # (N,)
            y = torch.from_numpy(X[:, j].copy()).float()    # (N,)
            # Pad Z to fixed m_max for easy batching
            z = torch.zeros((self.m_max, self.n_rows), dtype=torch.float32)
            z_mask = torch.zeros((self.m_max,), dtype=torch.bool)
            for t, k in enumerate(S[: self.m_max]):
                z[t] = torch.from_numpy(X[:, k].copy()).float()
                z_mask[t] = True

            item = {
                "x": x,
                "y": y,
                "z": z,
                "z_mask": z_mask,
                "label": torch.tensor(1.0 if is_indep else 0.0, dtype=torch.float32),
            }
            if self.include_meta:
                item["meta"] = {"i": i, "j": j, "S": S, "d": d, "m": len(S)}
            yield item
            global_step += 1

def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # batch is list of dicts
    x = torch.stack([b["x"] for b in batch], dim=0)          # (B,N)
    y = torch.stack([b["y"] for b in batch], dim=0)          # (B,N)
    z = torch.stack([b["z"] for b in batch], dim=0)          # (B,Mmax,N)
    z_mask = torch.stack([b["z_mask"] for b in batch], dim=0)# (B,Mmax)
    label = torch.stack([b["label"] for b in batch], dim=0)  # (B,)
    out = {"x": x, "y": y, "z": z, "z_mask": z_mask, "label": label}
    return out

def make_dataloader(
    dataset: CIStreamingDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )
