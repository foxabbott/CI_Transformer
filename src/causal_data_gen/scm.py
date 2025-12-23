from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .dag import RandomDAGConfig, random_dag
from .mechanisms import MechanismFn, NoiseSpec, random_mechanism
from .utils import rng, topological_sort, standardize
from .ci import d_separated

Array = np.ndarray

@dataclass
class SCMMetadata:
    d: int
    adj: Array
    order: List[int]
    parents: List[List[int]]
    mechanisms: List[Dict[str, Any]]
    noises: List[Dict[str, Any]]

class CausalSCM:
    """A continuous SCM with diverse (possibly non-additive) mechanisms.

    Each variable i is generated as:
        x_i = f_i( x_pa(i), eps_i )
    where eps_i is drawn from a continuous distribution (not necessarily Gaussian).
    """

    def __init__(
        self,
        adj: Array,
        mechanisms: List[MechanismFn],
        noises: List[NoiseSpec],
        meta: SCMMetadata,
    ):
        self.adj = adj.astype(np.int8)
        self.mechanisms = mechanisms
        self.noises = noises
        self.meta = meta
        self._order = topological_sort(self.adj)
        self._parents = [np.where(self.adj[:, i] != 0)[0].astype(int).tolist() for i in range(self.adj.shape[0])]

    @classmethod
    def random(
        cls,
        cfg: RandomDAGConfig,
        seed: Optional[int] = None,
    ) -> "CausalSCM":
        r = rng(seed)
        adj, order = random_dag(cfg, seed=seed)
        d = cfg.d
        parents = [np.where(adj[:, i] != 0)[0].astype(int).tolist() for i in range(d)]
        mechs: List[MechanismFn] = []
        noises: List[NoiseSpec] = []
        mech_meta: List[Dict[str, Any]] = []
        noise_meta: List[Dict[str, Any]] = []

        for i in range(d):
            k = len(parents[i])
            mech_fn, noise_spec, meta = random_mechanism(r, k=max(k, 1))
            mechs.append(mech_fn)
            noises.append(noise_spec)
            mech_meta.append(meta)
            noise_meta.append({"name": noise_spec.name, "params": noise_spec.params})

        meta = SCMMetadata(
            d=d,
            adj=adj,
            order=order,
            parents=parents,
            mechanisms=mech_meta,
            noises=noise_meta,
        )
        return cls(adj=adj, mechanisms=mechs, noises=noises, meta=meta)

    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        standardize_vars: bool = True,
    ) -> Array:
        """Sample n observations (d-dimensional)."""
        r = rng(seed)
        d = self.adj.shape[0]
        X = np.zeros((n, d), dtype=np.float32)

        for i in self._order:
            pa = self._parents[i]
            eps = self.noises[i].sample(r, n).astype(np.float32)
            if len(pa) == 0:
                # still pass a dummy parent matrix with shape (n,1) for uniform mechanism signature
                P = np.zeros((n, 1), dtype=np.float32)
            else:
                P = X[:, pa]
            xi = self.mechanisms[i](P, eps).astype(np.float32)
            # gentle clipping to keep things bounded
            X[:, i] = np.clip(xi, -25.0, 25.0)

        if standardize_vars:
            X = standardize(X).astype(np.float32)
        return X

    def is_ci_true(self, i: int, j: int, S: List[int]) -> bool:
        """Ground-truth conditional independence via d-separation on the DAG (as represented by the adjacency matirx)."""
        return d_separated(self.adj, [i], [j], S)

    def adjacency(self) -> Array:
        return self.adj.copy()

    def metadata(self) -> SCMMetadata:
        return self.meta
