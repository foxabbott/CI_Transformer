from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .utils import rng

Array = np.ndarray

@dataclass
class RandomDAGConfig:
    d: int = 10
    edge_prob: float = 0.2
    max_parents: int = 4
    ensure_connected: bool = False
    weight_scale: float = 1.0

    def __post_init__(self):
        if self.d < 2 or self.d > 15:
            raise ValueError("d must be in [2, 15].")
        if not (0.0 <= self.edge_prob <= 1.0):
            raise ValueError("edge_prob must be in [0,1].")
        if self.max_parents < 0:
            raise ValueError("max_parents must be >= 0.")

def random_dag(cfg: RandomDAGConfig, seed: Optional[int] = None) -> Tuple[Array, Array, list[int]]:
    """Generate a random acyclic directed graph by sampling an ordering then adding forward edges.

    Returns:
        adj: (d,d) with adj[i,j]=1 if i->j
        weights: (d,d) with weights[i,j] nonzero if i->j
        order: a topological order used to ensure acyclicity
    """
    r = rng(seed)
    d = cfg.d
    order = r.permutation(d).tolist()

    adj = np.zeros((d, d), dtype=np.int8)
    weights = np.zeros((d, d), dtype=np.float32)

    pos = {node: k for k, node in enumerate(order)}

    for j in range(d):
        # choose potential parents among earlier nodes in order
        node_j = order[j]
        candidates = order[:j]
        if len(candidates) == 0:
            continue
        # sample edges
        mask = r.random(len(candidates)) < cfg.edge_prob
        parents = [candidates[k] for k in range(len(candidates)) if mask[k]]
        # enforce max_parents
        if cfg.max_parents and len(parents) > cfg.max_parents:
            parents = r.choice(parents, size=cfg.max_parents, replace=False).tolist()

        for p in parents:
            adj[p, node_j] = 1
            w = cfg.weight_scale * r.normal(0, 1)
            weights[p, node_j] = float(w)

    if cfg.ensure_connected:
        # add at least one incoming edge for each non-root by connecting to a previous node if needed
        for j in range(1, d):
            node_j = order[j]
            if adj[:, node_j].sum() == 0:
                p = order[r.integers(0, j)]
                adj[p, node_j] = 1
                weights[p, node_j] = float(cfg.weight_scale * r.normal(0, 1))

    # sanity check acyclicity (since edges only forward in order, it's acyclic)
    return adj, weights, order
