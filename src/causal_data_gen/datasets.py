from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .scm import CausalSCM, SCMMetadata
from .dag import RandomDAGConfig

Array = np.ndarray

@dataclass
class Dataset:
    X: Array
    meta: SCMMetadata
    seed: Optional[int] = None

    def save_npz(self, path: str):
        np.savez_compressed(
            path,
            X=self.X,
            adj=self.meta.adj,
            weights=self.meta.weights,
            order=np.array(self.meta.order, dtype=np.int32),
            parents=np.array([np.array(p, dtype=np.int32) for p in self.meta.parents], dtype=object),
            mechanisms=np.array(self.meta.mechanisms, dtype=object),
            noises=np.array(self.meta.noises, dtype=object),
            d=np.int32(self.meta.d),
            seed=-1 if self.seed is None else np.int32(self.seed),
        )

def generate_dataset(
    n: int = 10_000,
    cfg: Optional[RandomDAGConfig] = None,
    seed: Optional[int] = None,
    standardize_vars: bool = True,
) -> Tuple[Dataset, CausalSCM]:
    """Generate one observational dataset and return (dataset, scm)."""
    if cfg is None:
        cfg = RandomDAGConfig(d=10, edge_prob=0.25, max_parents=4)
    scm = CausalSCM.random(cfg, seed=seed)
    X = scm.sample(n=n, seed=seed, standardize_vars=standardize_vars)
    return Dataset(X=X, meta=scm.metadata(), seed=seed), scm
